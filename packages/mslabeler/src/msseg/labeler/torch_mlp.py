"""A GPU dense classifier for the labeler: an sklearn-compatible MLP in PyTorch.

Why not ``sklearn.neural_network.MLPClassifier`` for everything: it trains on
the CPU one fold at a time, and the Optimize search is ~200 fits (5 folds x
40 trials) of that. Why not skorch: it would still fit folds one after another,
and its module/optimizer plumbing is far more than these few-hundred-row
tables need.

What this does instead
----------------------
* :class:`TorchMLPClassifier` is a drop-in for ``MLPClassifier`` in the
  labeler's pipeline (``fit(X, y, sample_weight)``, ``predict_proba``,
  ``classes_``, ``n_iter_``, pickles), on CUDA when available. It adds
  **dropout**, which sklearn's MLP does not have.
* :func:`train_stacked` trains F independent networks **as one batched
  computation** -- weights are ``(F, in, out)`` tensors driven by ``baddbmm``,
  Adam is elementwise so the folds never interact -- and :func:`cv_scores`
  uses it to fit every cross-validation fold of a candidate in one pass, with
  per-fold standardisation done on the padded batch. One trial is then one
  training loop instead of five, and each epoch is a handful of kernel
  launches regardless of the fold count.
* Early stopping tracks each fold's own validation loss (not accuracy, as
  sklearn does: the search objective is log-loss) and keeps each fold's best
  weights with a vectorised ``torch.where`` -- no per-epoch host sync beyond
  the single "everyone plateaued?" check.
* Class labels are encoded over the caller's full class list, so a fold whose
  training rows lack a class still produces a probability column for it (it
  is simply driven down), which is what a log-loss over all classes needs.

Determinism: every random draw (init, shuffles, dropout masks) comes from one
``torch.Generator`` seeded by ``random_state`` on the training device, so the
same rows, seed and device reproduce the same weights. Across devices (CPU vs
CUDA) results differ at float precision.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.base import BaseEstimator, ClassifierMixin
except ImportError:                       # sklearn is an optional extra
    class BaseEstimator:                  # type: ignore[no-redef]
        pass

    class ClassifierMixin:                # type: ignore[no-redef]
        pass


def have_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_device(device: str = "auto") -> str:
    """"cuda" when asked for (or auto) and available, else "cpu"."""
    import torch
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(device)


def device_label(device: str = "auto") -> str:
    """Human label for the readouts: "cuda (NVIDIA ... )" or "cpu"."""
    if not have_torch():
        return "unavailable"
    import torch
    dev = resolve_device(device)
    if dev.startswith("cuda"):
        try:
            return f"cuda ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "cuda"
    return "cpu"


# --------------------------------------------------------------------------- #
# Batched training
# --------------------------------------------------------------------------- #
def _pad(arrays: Sequence[np.ndarray], dtype, fill=0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Stack ragged (n_f, ...) arrays into (F, n_max, ...) + a (F, n_max) mask."""
    F = len(arrays)
    n_max = max(len(a) for a in arrays)
    tail = arrays[0].shape[1:]
    out = np.full((F, n_max) + tuple(tail), fill, dtype)
    mask = np.zeros((F, n_max), np.float32)
    for f, a in enumerate(arrays):
        out[f, :len(a)] = a
        mask[f, :len(a)] = 1.0
    return out, mask


def _forward(h, params, dropout: float, train: bool, gen=None):
    import torch
    n_layers = len(params)
    for l, (W, b) in enumerate(params):
        h = torch.baddbmm(b.unsqueeze(1), h, W)
        if l < n_layers - 1:
            h = torch.relu(h)
            if train and dropout > 0.0:
                keep = (torch.rand(h.shape, generator=gen, device=h.device) >= dropout)
                h = h * keep.to(h.dtype) / (1.0 - dropout)
    return h


def _init_params(F: int, sizes: Sequence[int], device, gen):
    """PyTorch Linear's default init (U(-1/sqrt(in), 1/sqrt(in))), stacked."""
    import torch
    params = []
    for din, dout in zip(sizes[:-1], sizes[1:]):
        bound = 1.0 / math.sqrt(din)
        W = (torch.rand((F, din, dout), generator=gen, device=device) * 2 - 1) * bound
        b = (torch.rand((F, dout), generator=gen, device=device) * 2 - 1) * bound
        params.append((W.requires_grad_(True), b.requires_grad_(True)))
    return params


def train_stacked(X_list: Sequence[np.ndarray], y_list: Sequence[np.ndarray],
                  w_list: Sequence[Optional[np.ndarray]], *, n_classes: int,
                  hidden: Sequence[int], alpha: float = 1e-4, lr: float = 1e-3,
                  batch_size: Any = "auto", max_iter: int = 1000, dropout: float = 0.0,
                  n_iter_no_change: int = 20, tol: float = 1e-4, seed: int = 0,
                  device: str = "auto", val_list: Optional[Sequence] = None,
                  monitor: Optional[Dict[str, Any]] = None
                  ) -> Dict[str, Any]:
    """Train ``F = len(X_list)`` independent MLPs at once.

    ``monitor`` = ``{"X": (F, n, d) tensor, "y": (F, n) tensor, "w": (F, n)
    tensor of per-fold-normalised weights, "every": int, "cb": callable}``
    (all on the training device): every ``every`` epochs the folds' mean
    weighted cross-entropy on that set is passed to ``cb(epoch, loss)`` --
    the search's pruning hook; ``cb`` may raise to abandon the run.

    ``y_list`` holds class INDICES in ``[0, n_classes)``; ``w_list`` per-row
    weights (None = 1). With ``val_list`` (per fold ``(Xv, yv)`` or None for
    no validation) training stops when every fold has gone
    ``n_iter_no_change`` epochs without its validation loss improving by
    ``tol``, and each fold keeps the weights of its best epoch; without it
    the same rule runs on the training loss and the last weights are kept.

    Returns ``{"params": [(W (F,in,out), b (F,out)) as numpy], "n_iter",
    "best_epoch": (F,), "loss_curve": [per-epoch summed train loss]}``."""
    import torch
    dev = torch.device(resolve_device(device))
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    F = len(X_list)
    d = int(X_list[0].shape[1])
    Xp, mask = _pad([np.asarray(x, np.float32) for x in X_list], np.float32)
    yp, _ = _pad([np.asarray(y, np.int64) for y in y_list], np.int64, 0)
    wl = [np.ones(len(x), np.float32) if w is None else np.asarray(w, np.float32)
          for x, w in zip(X_list, w_list)]
    wp, _ = _pad(wl, np.float32)
    wp = wp * mask
    wp = wp / np.maximum(wp.sum(1, keepdims=True), 1e-12)     # per-fold mean
    X = torch.from_numpy(Xp).to(dev)
    y = torch.from_numpy(yp).to(dev)
    w = torch.from_numpy(wp).to(dev)
    n_max = X.shape[1]

    has_val = val_list is not None and any(v is not None for v in val_list)
    if has_val:
        Xv_l = [np.zeros((1, d), np.float32) if v is None else np.asarray(v[0], np.float32)
                for v in val_list]
        yv_l = [np.zeros(1, np.int64) if v is None else np.asarray(v[1], np.int64)
                for v in val_list]
        Xvp, vmask = _pad(Xv_l, np.float32)
        yvp, _ = _pad(yv_l, np.int64, 0)
        for f, v in enumerate(val_list):
            if v is None:
                vmask[f] = 0.0
        Xv = torch.from_numpy(Xvp).to(dev)
        yv = torch.from_numpy(yvp).to(dev)
        wv = torch.from_numpy(vmask / np.maximum(vmask.sum(1, keepdims=True), 1e-12)).to(dev)
        monitored = torch.from_numpy(vmask.sum(1) > 0).to(dev)   # folds with a val set
    else:
        monitored = torch.ones(F, dtype=torch.bool, device=dev)

    sizes = [d] + [int(h) for h in hidden] + [int(n_classes)]
    params = _init_params(F, sizes, dev, gen)
    flat = [t for pair in params for t in pair]
    # fused=True folds the per-parameter Adam kernels into one launch; with
    # nets this small the training loop is launch-bound, not FLOP-bound.
    opt = torch.optim.Adam(flat, lr=float(lr), fused=(dev.type == "cuda"))
    bs = n_max if batch_size in (None, "auto") else max(1, min(int(batch_size), n_max))
    n_batches = int(math.ceil(n_max / bs))
    ar = torch.arange(F, device=dev)[:, None]

    best = torch.full((F,), float("inf"), device=dev)
    best_params = [(W.detach().clone(), b.detach().clone()) for W, b in params]
    best_epoch = torch.zeros(F, dtype=torch.long, device=dev)
    stale = torch.zeros(F, dtype=torch.long, device=dev)
    loss_curve: List[float] = []
    n_iter = 0

    def penalty():
        # sklearn's L2: (alpha / 2) * sum ||W||^2 / n, per fold.
        reg = 0.0
        for W, _b in params:
            reg = reg + (W * W).sum(dim=(1, 2))
        return 0.5 * float(alpha) * reg / n_max

    for epoch in range(int(max_iter)):
        n_iter = epoch + 1
        if n_batches == 1:
            logits = _forward(X, params, dropout, True, gen)
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, n_classes), y.reshape(-1), reduction="none").reshape(F, n_max)
            fold_loss = (ce * w).sum(1)
            loss = (fold_loss + penalty()).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss = fold_loss.detach()
        else:
            perm = torch.rand((F, n_max), generator=gen, device=dev).argsort(1)
            epoch_loss = torch.zeros(F, device=dev)
            for start in range(0, n_max, bs):
                idx = perm[:, start:start + bs]
                Xb, yb, wb = X[ar, idx], y[ar, idx], w[ar, idx]
                logits = _forward(Xb, params, dropout, True, gen)
                ce = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, n_classes), yb.reshape(-1),
                    reduction="none").reshape(F, -1)
                fold_loss = (ce * wb).sum(1)
                # wb sums to ~1/n_batches per batch: rescale so a step sees a
                # full-epoch-sized gradient, as a mean over the batch would.
                loss = (fold_loss * n_batches + penalty()).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                epoch_loss = epoch_loss + fold_loss.detach()
        loss_curve.append(float(epoch_loss.sum()))

        with torch.no_grad():
            if has_val:
                lv = _forward(Xv, params, 0.0, False)
                cev = torch.nn.functional.cross_entropy(
                    lv.reshape(-1, n_classes), yv.reshape(-1),
                    reduction="none").reshape(F, -1)
                metric = (cev * wv).sum(1)
            else:
                metric = epoch_loss
            improved = (metric < best - float(tol)) & monitored
            best = torch.where(improved, metric, best)
            best_epoch = torch.where(improved, torch.full_like(best_epoch, epoch), best_epoch)
            stale = torch.where(improved, torch.zeros_like(stale), stale + 1)
            if has_val:
                m3 = improved[:, None, None]
                m2 = improved[:, None]
                for l, (W, b) in enumerate(params):
                    bW, bb = best_params[l]
                    best_params[l] = (torch.where(m3, W.detach(), bW),
                                      torch.where(m2, b.detach(), bb))
            if monitor is not None and (epoch + 1) % int(monitor["every"]) == 0:
                lm = _forward(monitor["X"], params, 0.0, False)
                cem = torch.nn.functional.cross_entropy(
                    lm.reshape(-1, n_classes), monitor["y"].reshape(-1),
                    reduction="none").reshape(F, -1)
                monitor["cb"](epoch + 1, float((cem * monitor["w"]).sum(1).mean()))
            # The one host sync per epoch: has every monitored fold plateaued?
            if bool((stale >= int(n_iter_no_change))[monitored].all()):
                break

    if has_val:
        # Folds without a validation set keep their last weights.
        keep_last = ~monitored
        out_params = []
        for l, (W, b) in enumerate(params):
            bW, bb = best_params[l]
            out_params.append((torch.where(keep_last[:, None, None], W.detach(), bW),
                               torch.where(keep_last[:, None], b.detach(), bb)))
    else:
        out_params = [(W.detach(), b.detach()) for W, b in params]
    return {"params": [(W.cpu().numpy(), b.cpu().numpy()) for W, b in out_params],
            "n_iter": n_iter, "best_epoch": best_epoch.cpu().numpy(),
            "loss_curve": loss_curve}


def predict_stacked(params_np, X_list: Sequence[np.ndarray], device: str = "auto"
                    ) -> List[np.ndarray]:
    """Per-fold softmax probabilities ``(n_f, C)`` for stacked ``params``."""
    import torch
    dev = torch.device(resolve_device(device))
    Xp, mask = _pad([np.asarray(x, np.float32) for x in X_list], np.float32)
    with torch.no_grad():
        params = [(torch.from_numpy(W).to(dev), torch.from_numpy(b).to(dev)) for W, b in params_np]
        logits = _forward(torch.from_numpy(Xp).to(dev), params, 0.0, False)
        p = torch.softmax(logits, dim=2).cpu().numpy()
    return [p[f, :len(x)] for f, x in enumerate(X_list)]


# --------------------------------------------------------------------------- #
# Per-fold standardisation helpers (the pipeline's StandardScaler, per fold)
# --------------------------------------------------------------------------- #
def _fold_scaler(Xtr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd == 0] = 1.0                     # StandardScaler: constant column -> unscaled
    return mu, sd


def _val_split(X, y, w, fraction: float, seed: int):
    """A stratified hold-out inside a training fold (sklearn's early-stopping
    split); None when a class is too rare to stratify."""
    from sklearn.model_selection import train_test_split
    _c, counts = np.unique(y, return_counts=True)
    n_val = int(round(len(y) * fraction))
    if counts.min() < 2 or n_val < 1 or len(y) - n_val < 2:
        return None
    try:
        tr, va = train_test_split(np.arange(len(y)), test_size=fraction, stratify=y,
                                  random_state=seed)
    except ValueError:
        return None
    return tr, va


def cv_scores(X, y, folds: Sequence[Tuple[np.ndarray, np.ndarray]], classes,
              sample_weight=None, *, hidden, alpha=1e-4, lr=1e-3, batch_size="auto",
              max_iter=1000, dropout=0.0, early_stopping=False,
              validation_fraction=0.15, n_iter_no_change=20, tol=1e-4, seed=0,
              device="auto", report_cb=None, report_every: int = 25
              ) -> List[Tuple[float, float]]:
    """``[(log_loss, balanced_accuracy)]`` per fold, every fold trained in one
    stacked pass with its own standardisation (fit on its training rows).
    With ``report_cb`` the folds' mean held-out cross-entropy is reported as
    ``report_cb(epoch, loss)`` every ``report_every`` epochs while training
    (the pruning hook); the callback may raise to stop."""
    import torch
    from sklearn.metrics import log_loss, balanced_accuracy_score
    X = np.asarray(X, np.float32)
    y = np.asarray(y)
    classes = np.asarray(classes)
    pos = {int(c): i for i, c in enumerate(classes)}
    yi = np.asarray([pos[int(v)] for v in y], np.int64)
    w = None if sample_weight is None else np.asarray(sample_weight, np.float32)
    Xs, ys, ws, vs, tests = [], [], [], [], []
    for tr, te in folds:
        Xtr, ytr = X[tr], yi[tr]
        wtr = None if w is None else w[tr]
        mu, sd = _fold_scaler(Xtr)
        Xtr = (Xtr - mu) / sd
        val = None
        if early_stopping:
            split = _val_split(Xtr, ytr, wtr, validation_fraction, seed)
            if split is not None:
                a, b = split
                val = (Xtr[b], ytr[b])
                Xtr, ytr = Xtr[a], ytr[a]
                wtr = None if wtr is None else wtr[a]
        Xs.append(Xtr); ys.append(ytr); ws.append(wtr); vs.append(val)
        tests.append(((X[te] - mu) / sd, y[te]))
    monitor = None
    if report_cb is not None:
        dev = torch.device(resolve_device(device))
        Xe, emask = _pad([np.asarray(t[0], np.float32) for t in tests], np.float32)
        ye, _ = _pad([yi[te] for _tr, te in folds], np.int64, 0)
        we = emask / np.maximum(emask.sum(1, keepdims=True), 1e-12)
        monitor = {"X": torch.from_numpy(Xe).to(dev), "y": torch.from_numpy(ye).to(dev),
                   "w": torch.from_numpy(we.astype(np.float32)).to(dev),
                   "every": max(1, int(report_every)), "cb": report_cb}
    out = train_stacked(Xs, ys, ws, n_classes=len(classes), hidden=hidden, alpha=alpha,
                        lr=lr, batch_size=batch_size, max_iter=max_iter, dropout=dropout,
                        n_iter_no_change=n_iter_no_change, tol=tol, seed=seed,
                        device=device, val_list=vs if early_stopping else None,
                        monitor=monitor)
    probas = predict_stacked(out["params"], [t[0] for t in tests], device)
    scores = []
    for p, (_Xte, yte) in zip(probas, tests):
        p = np.clip(p, 1e-9, 1.0)
        p = p / p.sum(1, keepdims=True)
        scores.append((float(log_loss(yte, p, labels=classes)),
                       float(balanced_accuracy_score(yte, classes[p.argmax(1)]))))
    return scores


# --------------------------------------------------------------------------- #
# The sklearn-compatible estimator
# --------------------------------------------------------------------------- #
class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """``MLPClassifier``'s interface over :func:`train_stacked` with F = 1.

    Standardise inputs upstream (the labeler's pipeline does). Pickles hold
    the weights as numpy arrays, so a model trained on CUDA loads and predicts
    on a CPU-only machine (with torch installed)."""

    def __init__(self, hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3,
                 batch_size="auto", early_stopping=False, validation_fraction=0.15,
                 n_iter_no_change=20, max_iter=1000, tol=1e-4, dropout=0.0,
                 device="auto", random_state=0):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.max_iter = max_iter
        self.tol = tol
        self.dropout = dropout
        self.device = device
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        pos = {int(c): i for i, c in enumerate(self.classes_)}
        yi = np.asarray([pos[int(v)] for v in y], np.int64)
        w = None if sample_weight is None else np.asarray(sample_weight, np.float32)
        val = None
        if self.early_stopping:
            split = _val_split(X, yi, w, self.validation_fraction, self.random_state)
            if split is not None:
                a, b = split
                val = (X[b], yi[b])
                X, yi = X[a], yi[a]
                w = None if w is None else w[a]
        out = train_stacked([X], [yi], [w], n_classes=len(self.classes_),
                            hidden=tuple(self.hidden_layer_sizes), alpha=self.alpha,
                            lr=self.learning_rate_init, batch_size=self.batch_size,
                            max_iter=self.max_iter, dropout=self.dropout,
                            n_iter_no_change=self.n_iter_no_change, tol=self.tol,
                            seed=self.random_state, device=self.device,
                            val_list=[val] if val is not None else None)
        self.params_ = out["params"]
        self.n_iter_ = int(out["n_iter"])
        self.best_epoch_ = int(out["best_epoch"][0])
        self.loss_curve_ = list(out["loss_curve"])
        self.n_features_in_ = int(X.shape[1])
        self.device_ = resolve_device(self.device)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, np.float32)
        return predict_stacked(self.params_, [X], self.device)[0].astype(np.float64)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
