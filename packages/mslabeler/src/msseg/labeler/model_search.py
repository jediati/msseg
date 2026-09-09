"""Hyperparameter + feature-subset search for the labeler's dense FC classifier.

The labeler's "dense FC" kind was one constant architecture -- ``(64, 32)``
behind a StandardScaler -- fit once on every labeled region with no held-out
score of any kind. With a few hundred labeled regions and tens of scale-space
features that overfits, and nothing said how the model would do on a slice the
user had not labeled.

This module is the headless half of **Optimize network**: it describes a dense
network as a plain-data :class:`ModelSpec`, builds the sklearn pipeline for it
(:func:`build_estimator`), scores a spec by cross-validation that leaves whole
slices out (:func:`make_cv`, :func:`cv_evaluate`), and searches over depth,
per-layer width, L2 alpha, learning rate, batch size, early stopping and a
per-channel feature mask (:func:`run_search`, Optuna's TPE when it is
installed, otherwise a seeded random search over the same space).

Design notes
------------
* The pipeline still consumes the FULL profile schema: :class:`FeatureSubset`
  is a fitted transformer that drops columns *inside* the estimator, exactly
  as the dense-top-N kinds keep their forest mask inside. So the labeler's
  feature-name fingerprint, its profile-compatibility gate and its
  ``predict_proba`` consumer (the magic-fill ``proba`` metric, the
  P(class)/uncertainty colorings) are untouched, and a pickle predicts as
  trained.
* Feature masks are searched per **channel** (``base``, ``blur_s0.7``,
  ``hessian_largest_s3`` ...) rather than per column: the reductions of one
  channel stand or fall together, and ~15 booleans keep the space searchable
  where ~60 would not.
* The objective is mean held-out **log-loss** (with balanced accuracy recorded
  alongside): the class probabilities are what the magic-fill ``proba`` metric
  and the uncertainty coloring consume, so the search should calibrate them,
  not merely rank the argmax.
* Folds group by slice whenever three or more slices carry labels: regions on
  one slice share the scan's intensity drift, so a plain stratified split lets
  the network memorise the slice and reports an optimistic score. The reported
  generalisation is then "how does this do on a slice I have not labeled".
* Nothing here touches Tk. The labeler runs :func:`run_search` on a worker
  thread and drains ``progress_cb`` through a queue.
"""
from __future__ import annotations

import dataclasses
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# The architecture the un-tuned "dense FC" kind builds: the search's baseline
# and the spec Train falls back to when no search has run yet.
BASELINE_HIDDEN = (64, 32)
DEFAULT_MAX_ITER = 1000
# Early stopping inside MLPClassifier holds this fraction of the TRAINING fold
# out again; wider than sklearn's 0.1 so a few-hundred-row fold still leaves a
# usable validation set.
VALIDATION_FRACTION = 0.15
N_ITER_NO_CHANGE = 20

BATCH_CHOICES = ("auto", "16", "32", "64", "256", "512")
# A search TRIAL's epoch budget; the winner is refit afterwards with the
# full DEFAULT_MAX_ITER. Cross-validation is there to rank candidates, and a
# 300-epoch cap with a 10-epoch patience ranks them at a fraction of the
# cost of letting every candidate run to 1000.
SEARCH_MAX_ITER = 300
SEARCH_PATIENCE = 10
# Mini-batch training is a Python loop of ~1.5 ms per optimizer step on
# EITHER device (a dozen tiny kernels: launch-bound, not FLOP-bound), so a
# batch of 16 on 5000 rows is ~300 steps = 0.45 s per epoch stacked on torch
# -- minutes per trial -- where sklearn's MLP does that epoch in 6 ms per
# fold. Specs with a batch below this train on sklearn; torch keeps the
# full-batch and large-batch specs, where its stacked folds pay off.
TORCH_MIN_BATCH = 256
# How often (in epochs) a torch trial reports its held-out loss for pruning.
REPORT_EVERY = 25


# --------------------------------------------------------------------------- #
# The spec
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    """A dense FC network as plain data -- what the search returns, what the
    pickle and the session carry, what the Model tab renders."""
    hidden: Tuple[int, ...] = BASELINE_HIDDEN
    alpha: float = 1e-4
    learning_rate_init: float = 1e-3
    batch_size: Any = "auto"          # int or "auto"
    early_stopping: bool = False
    features: Optional[List[str]] = None   # None = every non-positional field
    max_iter: int = DEFAULT_MAX_ITER
    seed: int = 0
    # Provenance of the search that produced it (None for a hand-written spec).
    cv_metric: str = "log_loss"
    cv_score: Optional[float] = None       # mean held-out log-loss (lower = better)
    cv_bacc: Optional[float] = None        # mean held-out balanced accuracy
    baseline_score: Optional[float] = None
    baseline_bacc: Optional[float] = None
    cv_kind: str = ""                      # "slices" / "stratified"
    n_folds: int = 0
    n_groups: int = 0
    n_trials: int = 0                      # trials completed by the search
    searcher: str = ""                     # "optuna" / "random"
    n_features: int = 0                    # columns the search saw (0 = unknown)
    # Which MLP implements the net: "torch" (torch_mlp.TorchMLPClassifier, GPU
    # when available, folds trained as one stacked batch) or "sklearn"
    # (MLPClassifier, CPU); "auto" = torch when importable. A search resolves
    # it once and writes the resolved name, so a pickle is explicit.
    backend: str = "auto"
    dropout: float = 0.0                   # torch backend only (sklearn has none)
    # The architecture trial 1 of the search that produced this spec scored
    # (BASELINE_HIDDEN for Optimize; the fixed size in a size sweep).
    baseline_hidden: Tuple[int, ...] = BASELINE_HIDDEN

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["hidden"] = [int(h) for h in self.hidden]
        d["features"] = None if self.features is None else [str(n) for n in self.features]
        d["baseline_hidden"] = [int(h) for h in self.baseline_hidden]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelSpec":
        d = dict(d or {})
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        hidden = d.get("hidden") or BASELINE_HIDDEN
        d["hidden"] = tuple(int(h) for h in hidden)
        bs = d.get("batch_size", "auto")
        d["batch_size"] = "auto" if bs in (None, "auto") else int(bs)
        feats = d.get("features")
        d["features"] = None if feats is None else [str(n) for n in feats]
        bh = d.get("baseline_hidden") or BASELINE_HIDDEN
        d["baseline_hidden"] = tuple(int(h) for h in bh)
        return cls(**d)

    # -- rendering ----------------------------------------------------- #
    def layers_text(self) -> str:
        return "[" + ", ".join(str(h) for h in self.hidden) + "]"

    def features_text(self, n_total: Optional[int] = None) -> str:
        n_total = n_total or self.n_features or None
        if self.features is None:
            return "all features" if n_total is None else f"all {n_total} features"
        if n_total is None:
            return f"{len(self.features)} features"
        return f"{len(self.features)}/{n_total} features"

    def settings_text(self, n_total: Optional[int] = None) -> str:
        """The non-architecture settings on one line, for the sweep report:
        `alpha 1.3e-04 · lr 2e-03 · batch 32 · early stop · dropout 0.2 · 61/115 feats`."""
        bits = [f"alpha {self.alpha:.1e}", f"lr {self.learning_rate_init:.1e}",
                f"batch {self.batch_size}"]
        if self.early_stopping:
            bits.append("early stop")
        if self.dropout > 0.0 and effective_backend(self) == "torch":
            bits.append(f"dropout {self.dropout:.2f}")
        bits.append(self.features_text(n_total).replace(" features", " feats"))
        return " · ".join(bits)

    def brief(self, n_total: Optional[int] = None) -> str:
        """One short line for the model strip / hint."""
        bits = [self.layers_text(), self.features_text(n_total)]
        if self.cv_bacc is not None:
            bits.append(f"CV bacc {self.cv_bacc:.1%}")
        return " · ".join(bits)

    def describe(self, n_total: Optional[int] = None) -> str:
        """The Model tab paragraph: the whole pipeline plus how it scored."""
        bs = self.batch_size if self.batch_size != "auto" else "auto"
        es = (f"early stopping on a {VALIDATION_FRACTION:.0%} validation split"
              if self.early_stopping else "no early stopping")
        backend = effective_backend(self)
        if backend == "torch":
            cls = "TorchMLPClassifier"
            extra = f", dropout {self.dropout:.2g}"
        else:
            cls = "MLPClassifier"
            extra = ""
        text = (f"FeatureSubset({self.features_text(n_total)}) -> StandardScaler -> "
                f"{cls}(hidden layers {self.layers_text()}, alpha {self.alpha:.2g}, "
                f"learning rate {self.learning_rate_init:.2g}, batch {bs}{extra}, {es}, "
                f"max_iter {self.max_iter}, random_state {self.seed}), fit with "
                f"balanced sample weights; backend {backend_label(backend)}.")
        if self.cv_score is not None:
            how = ("leave-slices-out" if self.cv_kind == "slices" else "stratified")
            text += (f" Search: {self.searcher or 'search'}, {self.n_trials} trials, "
                     f"{self.n_folds}-fold {how} CV")
            if self.cv_kind == "slices":
                text += f" over {self.n_groups} slices"
            text += f": log-loss {self.cv_score:.3f}"
            if self.cv_bacc is not None:
                text += f", balanced acc {self.cv_bacc:.1%}"
            if self.baseline_score is not None:
                text += (f" (baseline {tuple(self.baseline_hidden)}, all features: "
                         f"{self.baseline_score:.3f}")
                if self.baseline_bacc is not None:
                    text += f", {self.baseline_bacc:.1%}"
                text += ")"
            text += "."
        return text


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
def _sklearn_bases():
    from sklearn.base import BaseEstimator, TransformerMixin
    return BaseEstimator, TransformerMixin


try:
    _BaseEstimator, _TransformerMixin = _sklearn_bases()
except ImportError:                      # sklearn is an optional extra
    class _BaseEstimator:                # type: ignore[no-redef]
        pass

    class _TransformerMixin:             # type: ignore[no-redef]
        pass


class FeatureSubset(_BaseEstimator, _TransformerMixin):
    """Keep the columns named in ``keep`` out of a matrix whose columns are
    ``names``; ``keep=None`` keeps every column. A module-level class so the
    pickled pipeline imports cleanly, and stateless beyond its constructor
    args so sklearn's clone/get_params contract holds."""

    def __init__(self, names=None, keep=None):
        self.names = names
        self.keep = keep

    def _indices(self):
        names = list(self.names or [])
        if self.keep is None:
            return list(range(len(names)))
        pos = {n: i for i, n in enumerate(names)}
        missing = [k for k in self.keep if k not in pos]
        if missing:
            raise ValueError(f"FeatureSubset: unknown feature(s) {missing[:4]}")
        return [pos[k] for k in self.keep]

    def fit(self, X, y=None, **_kw):
        X = np.asarray(X)
        n = len(self.names or [])
        if n and X.shape[1] != n:
            raise ValueError(f"FeatureSubset: expected {n} columns, got {X.shape[1]}")
        self.idx_ = np.asarray(self._indices(), int)
        self.n_features_in_ = int(X.shape[1])
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.idx_]

    def get_support(self):
        mask = np.zeros(self.n_features_in_, bool)
        mask[self.idx_] = True
        return mask

    def get_feature_names_out(self, input_features=None):
        names = list(self.names or (input_features or []))
        return np.asarray([names[i] for i in self.idx_], object)


BACKENDS = ("auto", "torch", "sklearn")


def resolve_backend(backend: str = "auto") -> str:
    """"torch" or "sklearn": the requested backend, falling back to sklearn
    when torch is not importable (an explicit "torch" included -- a pickle
    that names it still needs torch to unpickle, but a spec does not)."""
    from . import torch_mlp
    if backend == "sklearn":
        return "sklearn"
    return "torch" if torch_mlp.have_torch() else "sklearn"


def effective_backend(spec: "ModelSpec") -> str:
    """The backend that actually trains `spec`: its resolved backend, except
    that a torch spec with a mini-batch below TORCH_MIN_BATCH trains on
    sklearn (see TORCH_MIN_BATCH). `spec.backend` keeps the REQUEST."""
    be = resolve_backend(spec.backend)
    bs = spec.batch_size
    if be == "torch" and bs not in (None, "auto") and int(bs) < TORCH_MIN_BATCH:
        return "sklearn"
    return be


def backend_label(backend: str = "auto") -> str:
    """"torch on cuda (NVIDIA ...)" / "torch on cpu" / "sklearn (cpu)"."""
    from . import torch_mlp
    if resolve_backend(backend) == "torch":
        return f"torch on {torch_mlp.device_label('auto')}"
    return "sklearn (cpu)"


def build_estimator(spec: ModelSpec, names: Sequence[str], n_iter_no_change=None):
    """The sklearn pipeline for ``spec`` over a feature matrix whose columns are
    ``names``: FeatureSubset -> StandardScaler -> the MLP of the spec's
    EFFECTIVE backend (torch_mlp.TorchMLPClassifier, or sklearn's
    MLPClassifier). ``n_iter_no_change`` overrides the early-stop patience
    (the search uses SEARCH_PATIENCE; the default is the refit's)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    keep = None if spec.features is None else list(spec.features)
    common = dict(hidden_layer_sizes=tuple(spec.hidden), alpha=float(spec.alpha),
                  learning_rate_init=float(spec.learning_rate_init),
                  batch_size=spec.batch_size, early_stopping=bool(spec.early_stopping),
                  validation_fraction=VALIDATION_FRACTION,
                  n_iter_no_change=(N_ITER_NO_CHANGE if n_iter_no_change is None
                                    else int(n_iter_no_change)),
                  max_iter=int(spec.max_iter), random_state=int(spec.seed))
    if effective_backend(spec) == "torch":
        from .torch_mlp import TorchMLPClassifier
        mlp = TorchMLPClassifier(dropout=float(spec.dropout), device="auto", **common)
    else:
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(**common)
    return Pipeline([("select", FeatureSubset(list(names), keep)),
                     ("scale", StandardScaler()),
                     ("dense", mlp)])


def balanced_weights(y) -> np.ndarray:
    from sklearn.utils.class_weight import compute_sample_weight
    return np.asarray(compute_sample_weight("balanced", np.asarray(y)), float)


def fit_estimator(est, X, y, balanced: bool = True):
    """Fit with balanced sample weights (sklearn >= 1.7 accepts them on an
    MLP); older sklearn falls back to an unweighted fit rather than failing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if balanced:
            try:
                return est.fit(X, y, dense__sample_weight=balanced_weights(y))
            except TypeError:
                pass
        return est.fit(X, y)


# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #
def feature_groups(names: Sequence[str], schema=None) -> Dict[str, List[str]]:
    """``channel -> [column names]`` over ``names``, in first-seen order.
    ``schema`` is config_io.feature_schema()'s ``{name, channel, reduction}``
    list; a column with no channel (the headless fallback schema) is its own
    group, so the search degrades to per-column masks rather than failing."""
    chan = {}
    for e in schema or []:
        c = str(e.get("channel") or "")
        if c:
            chan[str(e.get("name"))] = c
    groups: Dict[str, List[str]] = {}
    for n in names:
        groups.setdefault(chan.get(n, n), []).append(n)
    return groups


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def make_cv(y, groups=None, n_splits: int = 5, seed: int = 0):
    """``(splitter, kind)``: StratifiedGroupKFold over slices when >= 3 slices
    carry labels, else StratifiedKFold; folds capped by the rarest class so
    every fold can hold every class. Raises ValueError when a class has fewer
    than 2 rows (no held-out score is possible)."""
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    y = np.asarray(y)
    _classes, counts = np.unique(y, return_counts=True)
    if len(_classes) < 2:
        raise ValueError("need labels from at least 2 classes")
    k = min(int(n_splits), int(counts.min()))
    if k < 2:
        raise ValueError("need at least 2 labeled regions in every class")
    n_groups = 0 if groups is None else len(set(np.asarray(groups).tolist()))
    if n_groups >= 3:
        k = min(k, n_groups)
        return StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed), "slices"
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed), "stratified"


def _full_proba(est, X, classes: np.ndarray) -> np.ndarray:
    """``predict_proba`` widened to every class in ``classes`` (a fold's train
    set may lack one), floored so log-loss stays finite."""
    p = np.asarray(est.predict_proba(X), float)
    own = np.asarray(est.classes_)
    out = np.full((len(X), len(classes)), 1e-9, float)
    pos = {int(c): i for i, c in enumerate(classes)}
    for j, c in enumerate(own):
        i = pos.get(int(c))
        if i is not None:
            out[:, i] = p[:, j]
    out /= out.sum(1, keepdims=True)
    return out


def cv_evaluate(spec: ModelSpec, X, y, cv, names: Sequence[str], groups=None,
                trial=None, max_iter=None, patience=None,
                report_every: int = REPORT_EVERY) -> Tuple[float, float]:
    """Mean held-out ``(log_loss, balanced_accuracy)`` of ``spec`` over ``cv``.

    ``max_iter`` / ``patience`` override the spec's epoch cap and early-stop
    patience for THIS evaluation (the search budget; the spec itself keeps
    the refit's). With an Optuna ``trial``, an sklearn spec reports the
    running mean after every fold (steps 0..k-1) and a torch spec -- whose
    folds train together -- reports the folds' mean held-out loss every
    ``report_every`` epochs (steps 25, 50, ...), so a hopeless spec is
    pruned early either way."""
    from sklearn.metrics import log_loss, balanced_accuracy_score
    if max_iter is not None:
        spec = dataclasses.replace(spec, max_iter=int(max_iter))
    n_iter_no_change = N_ITER_NO_CHANGE if patience is None else int(patience)
    X = np.asarray(X, float)
    y = np.asarray(y)
    classes = np.unique(y)
    losses, baccs = [], []
    split = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    if effective_backend(spec) == "torch":
        # Every fold in ONE stacked training pass (torch_mlp.cv_scores does
        # the per-fold standardisation); the column subset is shared, so it
        # is applied once here.
        from . import torch_mlp
        report_cb = None
        if trial is not None:
            def report_cb(epoch, loss):
                trial.report(float(loss), int(epoch))
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
        names = list(names)
        cols = (list(range(len(names))) if spec.features is None
                else [names.index(f) for f in spec.features])
        scores = torch_mlp.cv_scores(
            X[:, cols], y, list(split), classes, balanced_weights(y),
            hidden=tuple(spec.hidden), alpha=float(spec.alpha),
            lr=float(spec.learning_rate_init), batch_size=spec.batch_size,
            max_iter=int(spec.max_iter), dropout=float(spec.dropout),
            early_stopping=bool(spec.early_stopping),
            validation_fraction=VALIDATION_FRACTION, n_iter_no_change=n_iter_no_change,
            seed=int(spec.seed), report_cb=report_cb, report_every=report_every)
        losses = [s[0] for s in scores]
        baccs = [s[1] for s in scores]
        return float(np.mean(losses)), float(np.mean(baccs))
    for step, (tr, te) in enumerate(split):
        est = build_estimator(spec, names, n_iter_no_change=n_iter_no_change)
        fit_estimator(est, X[tr], y[tr])
        p = _full_proba(est, X[te], classes)
        losses.append(float(log_loss(y[te], p, labels=classes)))
        baccs.append(float(balanced_accuracy_score(y[te], classes[p.argmax(1)])))
        if trial is not None:
            trial.report(float(np.mean(losses)), step)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
    return float(np.mean(losses)), float(np.mean(baccs))


# --------------------------------------------------------------------------- #
# The search space
# --------------------------------------------------------------------------- #
@dataclass
class SearchSpace:
    max_layers: int = 4
    units: Tuple[int, int] = (8, 256)
    alpha: Tuple[float, float] = (1e-6, 1e-1)
    learning_rate: Tuple[float, float] = (1e-4, 1e-2)
    batch_choices: Tuple[str, ...] = BATCH_CHOICES
    early_stopping: bool = True          # search over it (False = always off)
    feature_search: bool = True          # search the per-channel mask
    dropout: Tuple[float, float] = (0.0, 0.5)   # torch backend only
    # A fixed architecture: the search then covers only the OTHER settings
    # (the size sweep runs one such search per rung of its ladder).
    fixed_hidden: Optional[Tuple[int, ...]] = None


def _batch_value(choice: str):
    return "auto" if choice == "auto" else int(choice)


def _torch_trains(backend: str, batch) -> bool:
    """Whether a spec drawn with this backend and batch trains on torch."""
    return backend == "torch" and (batch == "auto" or int(batch) >= TORCH_MIN_BATCH)


def _features_from_flags(groups: Dict[str, List[str]], flags: Dict[str, bool]):
    """The column list for a set of per-group include flags; None (= all) when
    every flag is off, so a degenerate draw still trains something."""
    keep = [n for g, cols in groups.items() if flags.get(g, True) for n in cols]
    if not keep or len(keep) == sum(len(c) for c in groups.values()):
        return None
    return keep


def suggest_spec(trial, groups: Dict[str, List[str]], space: SearchSpace,
                 seed: int, max_iter: int, backend: str = "sklearn") -> ModelSpec:
    """Draw one spec from an Optuna trial (define-by-run: the layer count
    decides how many width parameters exist; dropout exists only for torch)."""
    if space.fixed_hidden:
        hidden = tuple(int(h) for h in space.fixed_hidden)
    else:
        n_layers = trial.suggest_int("n_layers", 1, space.max_layers)
        hidden = tuple(trial.suggest_int(f"units_{i}", space.units[0], space.units[1],
                                         log=True) for i in range(n_layers))
    alpha = trial.suggest_float("alpha", space.alpha[0], space.alpha[1], log=True)
    lr = trial.suggest_float("lr", space.learning_rate[0], space.learning_rate[1], log=True)
    batch = _batch_value(trial.suggest_categorical("batch", list(space.batch_choices)))
    es = trial.suggest_categorical("early_stopping", [False, True]) if space.early_stopping else False
    dropout = 0.0
    # Define-by-run: a mini-batch below TORCH_MIN_BATCH trains on sklearn
    # (effective_backend), which has no dropout, so the parameter exists only
    # for the draws torch will train.
    if _torch_trains(backend, batch) and space.dropout[1] > space.dropout[0]:
        dropout = trial.suggest_float("dropout", space.dropout[0], space.dropout[1])
    feats = None
    if space.feature_search and len(groups) > 1:
        flags = {g: bool(trial.suggest_categorical(f"use_{g}", [True, False])) for g in groups}
        feats = _features_from_flags(groups, flags)
    return ModelSpec(hidden=hidden, alpha=alpha, learning_rate_init=lr, batch_size=batch,
                     early_stopping=bool(es), features=feats, max_iter=max_iter, seed=seed,
                     backend=backend, dropout=float(dropout))


def random_spec(rng: np.random.RandomState, groups: Dict[str, List[str]],
                space: SearchSpace, seed: int, max_iter: int,
                backend: str = "sklearn") -> ModelSpec:
    """The same space, drawn uniformly (log-uniform where Optuna would) -- the
    searcher when optuna is not installed."""
    if space.fixed_hidden:
        hidden = tuple(int(h) for h in space.fixed_hidden)
    else:
        n_layers = int(rng.randint(1, space.max_layers + 1))
        lo, hi = math.log(space.units[0]), math.log(space.units[1])
        hidden = tuple(int(round(math.exp(rng.uniform(lo, hi)))) for _ in range(n_layers))
    alpha = float(math.exp(rng.uniform(math.log(space.alpha[0]), math.log(space.alpha[1]))))
    lr = float(math.exp(rng.uniform(math.log(space.learning_rate[0]),
                                    math.log(space.learning_rate[1]))))
    batch = _batch_value(space.batch_choices[int(rng.randint(len(space.batch_choices)))])
    es = bool(rng.randint(2)) if space.early_stopping else False
    dropout = 0.0
    if _torch_trains(backend, batch) and space.dropout[1] > space.dropout[0]:
        dropout = float(rng.uniform(space.dropout[0], space.dropout[1]))
    feats = None
    if space.feature_search and len(groups) > 1:
        flags = {g: bool(rng.randint(2)) for g in groups}
        feats = _features_from_flags(groups, flags)
    return ModelSpec(hidden=hidden, alpha=alpha, learning_rate_init=lr, batch_size=batch,
                     early_stopping=es, features=feats, max_iter=max_iter, seed=seed,
                     backend=backend, dropout=dropout)


def baseline_params(space: SearchSpace, groups: Dict[str, List[str]],
                    backend: str = "sklearn") -> Dict[str, Any]:
    """The un-tuned dense FC as Optuna params, enqueued as trial 0 so the search
    starts from -- and can never lose to -- what Train would have built."""
    p: Dict[str, Any] = {"alpha": 1e-4, "lr": 1e-3, "batch": "auto"}
    if not space.fixed_hidden:              # a fixed size has no size params
        p["n_layers"] = len(BASELINE_HIDDEN)
        for i, h in enumerate(BASELINE_HIDDEN):
            p[f"units_{i}"] = int(h)
    if space.early_stopping:
        p["early_stopping"] = False
    if backend == "torch" and space.dropout[1] > space.dropout[0]:
        p["dropout"] = 0.0
    if space.feature_search and len(groups) > 1:
        for g in groups:
            p[f"use_{g}"] = True
    return p


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #
@dataclass
class SearchResult:
    spec: ModelSpec
    estimator: Any                       # refit on every row
    baseline: ModelSpec
    n_trials: int
    elapsed_s: float
    stopped: bool = False                # cancelled or timed out before n_trials
    importances: List[Tuple[str, float]] = field(default_factory=list)


def have_optuna() -> bool:
    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        return False


def _progress(cb, done, total, best):
    if cb is not None:
        cb(done, total, best)


def run_search(X, y, groups, names: Sequence[str], schema=None, *,
               n_trials: int = 40, timeout_s: Optional[float] = None, seed: int = 0,
               max_iter: int = DEFAULT_MAX_ITER, space: Optional[SearchSpace] = None,
               n_splits: int = 5, progress_cb: Optional[Callable] = None,
               stop_event=None, searcher: Optional[str] = None,
               importances: bool = True, backend: str = "auto",
               patience: Optional[int] = None,
               refit_max_iter: Optional[int] = None) -> SearchResult:
    """Search the space, refit the winner on every row, return it.

    ``max_iter`` and ``patience`` are the per-TRIAL budget (epoch cap and
    early-stop patience of every cross-validated fit; the labeler passes
    SEARCH_MAX_ITER / SEARCH_PATIENCE). ``refit_max_iter`` is the cap the
    winner is refit with and carries in its spec (None = ``max_iter``).
    ``groups`` is the slice index of each row (None = no grouping). Progress:
    ``progress_cb(trials_done, n_trials, best_spec_or_None)`` from the calling
    thread after every trial; ``stop_event.is_set()`` ends the search after the
    current trial, keeping the best so far. ``searcher`` forces "optuna" or
    "random"; the default is optuna when importable. ``backend`` is resolved
    once ("torch" when importable unless "sklearn" is asked for) and written
    into every spec."""
    t0 = time.perf_counter()
    X = np.asarray(X, float)
    y = np.asarray(y)
    names = list(names)
    space = space or SearchSpace()
    grp = feature_groups(names, schema)
    cv, cv_kind = make_cv(y, groups, n_splits=n_splits, seed=seed)
    n_groups = 0 if groups is None else len(set(np.asarray(groups).tolist()))
    use = searcher or ("optuna" if have_optuna() else "random")
    be = resolve_backend(backend)

    base_hidden = (tuple(int(h) for h in space.fixed_hidden) if space.fixed_hidden
                   else BASELINE_HIDDEN)

    def stamp(spec: ModelSpec, loss: float, bacc: float) -> ModelSpec:
        return dataclasses.replace(spec, cv_score=loss, cv_bacc=bacc, cv_kind=cv_kind,
                                   n_folds=int(cv.get_n_splits()), n_groups=n_groups,
                                   searcher=use, baseline_hidden=base_hidden)

    def stop_now():
        if stop_event is not None and stop_event.is_set():
            return True
        return timeout_s is not None and (time.perf_counter() - t0) >= timeout_s

    # Trial 0 is always the un-tuned baseline: the search then reports its gain
    # over what Train would have built, and cannot return something worse.
    base = ModelSpec(hidden=base_hidden, max_iter=max_iter, seed=seed, backend=be,
                     baseline_hidden=base_hidden)
    b_loss, b_bacc = cv_evaluate(base, X, y, cv, names, groups, patience=patience)
    baseline = stamp(base, b_loss, b_bacc)
    best: ModelSpec = baseline
    done = 1
    _progress(progress_cb, done, n_trials, best)
    stopped = False

    if n_trials > 1 and not stop_now():
        if use == "optuna":
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=seed),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
            study.enqueue_trial(baseline_params(space, grp, be))
            state = {"done": done, "best": best}

            def objective(trial):
                spec = suggest_spec(trial, grp, space, seed, max_iter, be)
                loss, bacc = cv_evaluate(spec, X, y, cv, names, groups, trial=trial,
                                         patience=patience)
                trial.set_user_attr("bacc", bacc)
                trial.set_user_attr("spec", spec.to_dict())
                return loss

            def on_trial(st, tr):
                state["done"] += 1
                if tr.state == optuna.trial.TrialState.COMPLETE and \
                        tr.value is not None and tr.value < state["best"].cv_score:
                    state["best"] = stamp(ModelSpec.from_dict(tr.user_attrs["spec"]),
                                          float(tr.value), float(tr.user_attrs["bacc"]))
                _progress(progress_cb, state["done"], n_trials, state["best"])
                if stop_now():
                    st.stop()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                study.optimize(objective, n_trials=n_trials - 1, callbacks=[on_trial],
                               catch=(ValueError, FloatingPointError))
            done, best = state["done"], state["best"]
        else:
            use = "random"
            rng = np.random.RandomState(seed)
            for _ in range(n_trials - 1):
                spec = random_spec(rng, grp, space, seed, max_iter, be)
                try:
                    loss, bacc = cv_evaluate(spec, X, y, cv, names, groups,
                                             patience=patience)
                except (ValueError, FloatingPointError):
                    loss, bacc = math.inf, 0.0
                done += 1
                if loss < best.cv_score:
                    best = stamp(spec, loss, bacc)
                _progress(progress_cb, done, n_trials, best)
                if stop_now():
                    break
    stopped = done < n_trials

    best = dataclasses.replace(best, baseline_score=b_loss, baseline_bacc=b_bacc,
                               n_trials=done, searcher=use, n_features=len(names))
    if refit_max_iter is not None:
        best = dataclasses.replace(best, max_iter=int(refit_max_iter))
    est = fit_estimator(build_estimator(best, names), X, y)
    imps = feature_importances(est, X, y, names, seed) if importances else []
    return SearchResult(spec=best, estimator=est, baseline=baseline, n_trials=done,
                        elapsed_s=time.perf_counter() - t0, stopped=stopped,
                        importances=imps)


def feature_importances(est, X, y, names: Sequence[str], seed: int = 0,
                        n_repeats: int = 5) -> List[Tuple[str, float]]:
    """Permutation importance (drop in log-loss when a column is shuffled) of
    the fitted pipeline on ``X``: the dense kinds' per-feature story, sorted
    descending. Resubstitution, so a ranking rather than a held-out number."""
    try:
        from sklearn.inspection import permutation_importance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = permutation_importance(est, np.asarray(X, float), np.asarray(y),
                                       scoring="neg_log_loss", n_repeats=n_repeats,
                                       random_state=seed)
    except Exception:
        return []
    out = [(str(n), float(v)) for n, v in zip(names, r.importances_mean)]
    out.sort(key=lambda t: -t[1])
    return out


# --------------------------------------------------------------------------- #
# Size sweep: how small can the network be?
# --------------------------------------------------------------------------- #
# The default ladder, largest first: the labeler's baseline down to a single
# tiny layer. Each rung is searched with its architecture FIXED and the other
# settings (alpha, learning rate, batch, early stopping, dropout, feature
# subset) free, so a small net is judged at its own best settings rather than
# at the big net's.
DEFAULT_SIZE_LADDER = ((64, 32), (32, 16), (16, 8), (8, 4), (4,))
# A rung whose best held-out log-loss exceeds the ladder's best by more than
# this fraction is where the network "breaks down".
SWEEP_TOLERANCE = 0.05


def parse_sizes(text: str) -> List[Tuple[int, ...]]:
    """``"64-32, 32x16, 16 8; 4"`` -> ``[(64, 32), (32, 16), (16, 8), (4,)]``.
    Rungs are separated by commas / semicolons / newlines, layers within a rung
    by ``-``, ``x`` or spaces. Raises ValueError on anything else."""
    out: List[Tuple[int, ...]] = []
    for chunk in str(text).replace(";", ",").replace("\n", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p for p in chunk.replace("x", "-").replace("X", "-").replace(" ", "-").split("-") if p]
        try:
            layers = tuple(int(p) for p in parts)
        except ValueError:
            raise ValueError(f"size sweep: cannot read '{chunk}' (use e.g. 64-32)")
        if not layers or any(h < 1 for h in layers):
            raise ValueError(f"size sweep: layers must be >= 1 in '{chunk}'")
        out.append(layers)
    if not out:
        raise ValueError("size sweep: no sizes given")
    return out


def size_text(hidden: Sequence[int]) -> str:
    return "-".join(str(int(h)) for h in hidden)


def n_params(hidden: Sequence[int], n_in: int, n_classes: int) -> int:
    """Weights + biases of a dense net ``n_in -> hidden... -> n_classes``."""
    sizes = [int(n_in)] + [int(h) for h in hidden] + [int(n_classes)]
    return sum((a + 1) * b for a, b in zip(sizes[:-1], sizes[1:]))


@dataclass
class SweepRow:
    hidden: Tuple[int, ...]
    n_params: int
    cv_score: float
    cv_bacc: float
    spec: ModelSpec
    estimator: Any                      # refit on every row (run_search does)
    n_trials: int
    elapsed_s: float

    @property
    def size(self) -> str:
        return size_text(self.hidden)


@dataclass
class SweepResult:
    rows: List[SweepRow]
    names: List[str]
    n_classes: int
    elapsed_s: float
    stopped: bool = False

    def best_index(self) -> int:
        return int(min(range(len(self.rows)), key=lambda i: self.rows[i].cv_score))

    def relative_loss(self, i: int) -> float:
        best = self.rows[self.best_index()].cv_score
        return (self.rows[i].cv_score - best) / max(best, 1e-12)

    def smallest_within(self, tolerance: float = SWEEP_TOLERANCE) -> int:
        """Index of the fewest-parameter rung whose loss is within
        ``tolerance`` (relative) of the best rung's."""
        ok = [i for i in range(len(self.rows)) if self.relative_loss(i) <= tolerance]
        return int(min(ok, key=lambda i: self.rows[i].n_params))

    def smaller_than_best(self) -> List[int]:
        """Rung indices with fewer parameters than the best rung, largest first."""
        best_n = self.rows[self.best_index()].n_params
        below = [i for i in range(len(self.rows)) if self.rows[i].n_params < best_n]
        return sorted(below, key=lambda i: -self.rows[i].n_params)

    def breakdown_index(self, tolerance: float = SWEEP_TOLERANCE) -> Optional[int]:
        """Going SMALLER than the best rung, the first rung outside the
        tolerance -- where shrinking the net starts to cost -- or None when
        every smaller rung holds (or none is smaller). A larger rung that
        scores worse is not a breakdown, only a rung that did not pay off."""
        for i in self.smaller_than_best():
            if self.relative_loss(i) > tolerance:
                return i
        return None

    def holds_below_breakdown(self, tolerance: float = SWEEP_TOLERANCE) -> List[int]:
        """Rungs smaller than the breakdown rung that are nevertheless within
        tolerance: the ladder is not monotonic there, which with few trials per
        rung is search noise rather than a real recovery."""
        k = self.breakdown_index(tolerance)
        if k is None:
            return []
        kn = self.rows[k].n_params
        return [i for i in self.smaller_than_best()
                if self.rows[i].n_params < kn and self.relative_loss(i) <= tolerance]

    def summary(self, tolerance: float = SWEEP_TOLERANCE) -> str:
        b = self.rows[self.best_index()]
        s = self.rows[self.smallest_within(tolerance)]
        text = (f"best: {b.size} (log-loss {b.cv_score:.3f}, bal. acc {b.cv_bacc:.1%}); "
                f"smallest within {tolerance:.0%}: {s.size} ({s.n_params:,} params, "
                f"log-loss {s.cv_score:.3f}, bal. acc {s.cv_bacc:.1%})")
        k = self.breakdown_index(tolerance)
        if k is None:
            text += ("; no smaller rung breaks down" if self.smaller_than_best()
                     else "; no rung is smaller than the best")
        else:
            r = self.rows[k]
            text += (f"; breaks down at {r.size} (+{self.relative_loss(k):.0%} log-loss, "
                     f"bal. acc {r.cv_bacc:.1%})")
            holds = self.holds_below_breakdown(tolerance)
            if holds:
                text += (" -- not monotonic: " + ", ".join(self.rows[i].size for i in holds)
                         + " still hold(s); more trials per size would settle it")
        if self.stopped:
            text += " -- stopped early"
        return text

    def report_lines(self, tolerance: float = SWEEP_TOLERANCE) -> List[str]:
        """A fixed-width table, one rung per line, for the log / a text file."""
        head = f"{'size':<10}{'params':>9}  {'log-loss':>8}  {'bal.acc':>7}  {'vs best':>8}  {'trials':>6}  settings"
        lines = [head]
        for i, r in enumerate(self.rows):
            rel = self.relative_loss(i)
            mark = "" if rel <= tolerance else " !"
            lines.append(f"{r.size:<10}{r.n_params:>9,}  {r.cv_score:>8.3f}  {r.cv_bacc:>7.1%}  "
                         f"{'+' + format(rel, '.0%') if rel > 0 else 'best':>8}  {r.n_trials:>6}  "
                         f"{r.spec.settings_text(len(self.names))}{mark}")
        lines.append(self.summary(tolerance))
        return lines

    def to_dict(self) -> Dict[str, Any]:
        return {"names": list(self.names), "n_classes": int(self.n_classes),
                "elapsed_s": float(self.elapsed_s), "stopped": bool(self.stopped),
                "rows": [{"hidden": list(r.hidden), "n_params": int(r.n_params),
                          "cv_score": float(r.cv_score), "cv_bacc": float(r.cv_bacc),
                          "n_trials": int(r.n_trials), "elapsed_s": float(r.elapsed_s),
                          "spec": r.spec.to_dict()} for r in self.rows],
                "report": self.report_lines()}


def run_size_sweep(X, y, groups, names: Sequence[str], schema=None, *,
                   sizes: Sequence[Sequence[int]] = DEFAULT_SIZE_LADDER,
                   trials_per_size: int = 20, timeout_s: Optional[float] = None,
                   seed: int = 0, max_iter: int = DEFAULT_MAX_ITER,
                   patience: Optional[int] = None, refit_max_iter: Optional[int] = None,
                   space: Optional[SearchSpace] = None, n_splits: int = 5,
                   progress_cb: Optional[Callable] = None, stop_event=None,
                   searcher: Optional[str] = None, backend: str = "auto") -> SweepResult:
    """One fixed-architecture :func:`run_search` per rung of ``sizes``.

    Progress: ``progress_cb(rung_index, n_rungs, hidden, trials_done,
    trials_total, finished_row_or_None)`` -- per trial while a rung runs, and
    once more with its :class:`SweepRow` when it completes. ``timeout_s`` is
    the budget for the WHOLE sweep (each rung gets what is left);
    ``stop_event`` ends the sweep after the current trial, keeping the rungs
    already scored. Every other argument is :func:`run_search`'s."""
    t0 = time.perf_counter()
    X = np.asarray(X, float)
    y = np.asarray(y)
    names = list(names)
    space = space or SearchSpace()
    n_classes = len(np.unique(y))
    rows: List[SweepRow] = []
    stopped = False
    sizes = [tuple(int(h) for h in s) for s in sizes]
    for k, hidden in enumerate(sizes):
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break
        remaining = None
        if timeout_s is not None:
            remaining = timeout_s - (time.perf_counter() - t0)
            if remaining <= 0:
                stopped = True
                break
        rung_space = dataclasses.replace(space, fixed_hidden=hidden)

        def cb(done, total, best, k=k, hidden=hidden):
            if progress_cb is not None:
                progress_cb(k, len(sizes), hidden, done, total, None)

        t_r = time.perf_counter()
        res = run_search(X, y, groups, names, schema, n_trials=trials_per_size,
                         timeout_s=remaining, seed=seed, max_iter=max_iter, space=rung_space,
                         n_splits=n_splits, progress_cb=cb, stop_event=stop_event,
                         searcher=searcher, importances=False, backend=backend,
                         patience=patience, refit_max_iter=refit_max_iter)
        n_in = len(names) if res.spec.features is None else len(res.spec.features)
        row = SweepRow(hidden=hidden, n_params=n_params(hidden, n_in, n_classes),
                       cv_score=float(res.spec.cv_score), cv_bacc=float(res.spec.cv_bacc),
                       spec=res.spec, estimator=res.estimator, n_trials=res.n_trials,
                       elapsed_s=time.perf_counter() - t_r)
        rows.append(row)
        if progress_cb is not None:
            progress_cb(k, len(sizes), hidden, res.n_trials, trials_per_size, row)
        if res.stopped:
            stopped = True
            break
    if not rows:
        raise ValueError("size sweep: no rung completed")
    return SweepResult(rows=rows, names=names, n_classes=n_classes,
                       elapsed_s=time.perf_counter() - t0, stopped=stopped)
