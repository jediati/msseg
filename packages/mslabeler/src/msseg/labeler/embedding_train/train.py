"""The objective and the loop.

For an anchor region ``i`` the positive is a region reached by a random walk
of 1..`walk` hops over the arc graph of its item; the negatives are the rest
of the batch (symmetric InfoNCE on the projection head, temperature `temp`).
Two more terms: a linear reconstruction of the clean standardised row from
the latent (weight `rec`), so a channel the walk does not need is not thrown
away, and a VICReg-style variance + covariance penalty on the latent
(weight `vc`), so every dimension keeps unit spread and they decorrelate.

Augmentation is input corruption, applied independently to both views:
whole column GROUPS dropped to zero with probability `group_drop` (a group is
a channel of the harvest's schema, so "one sigma of one plane" goes at once),
and single columns replaced by the value of a random other row in the batch
with probability `corrupt` (SCARF). Positional columns never enter.

``train`` returns the ``EncoderBundle`` and the per-epoch history. The PCA
baseline takes the same inputs and produces the same kind of bundle, so the
two are interchangeable downstream.
"""
from __future__ import annotations

import math
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import context
from ..embedding import EncoderBundle, Layer, input_convention, prepare_inputs
from ..fields import DEFAULT, FieldConventions
from .model import build_torch_model, pca_layer
from .shards import Harvest

ARCHS = ("mlp", "pca")
DEFAULT_LOG_COLUMNS = ("area", "bbox_w", "bbox_h")
HOLDOUT_FRACTION = 0.05


@dataclass
class TrainSettings:
    dim: int = 8
    arch: str = "mlp"
    hidden: Tuple[int, ...] = (64, 32)
    proj: int = 32
    epochs: int = 50
    batch: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-4
    walk: int = 2
    temp: float = 0.1
    rec: float = 0.1
    vc: float = 0.05
    group_drop: float = 0.15
    corrupt: float = 0.1
    seed: int = 0
    device: str = "auto"
    log_columns: Tuple[str, ...] = DEFAULT_LOG_COLUMNS
    extra_drop: Tuple[str, ...] = ()        # columns to leave out on top of the positional ones

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        d["log_columns"] = list(self.log_columns)
        d["extra_drop"] = list(self.extra_drop)
        return d


@dataclass
class Design:
    """The trainer's view of a harvest: the input matrix, its convention,
    the column groups and the graph."""
    names: List[str]
    Z: np.ndarray                       # float32 [n, n_in], standardised
    mean: np.ndarray
    std: np.ndarray
    log_mask: np.ndarray
    group_of: np.ndarray                # int [n_in]
    n_groups: int
    graph: context.Graph
    anchors: np.ndarray                 # rows with at least one neighbour
    scope: Optional[str]
    info: Dict[str, Any] = field(default_factory=dict)


def feature_columns(harvest: Harvest, conv: FieldConventions = DEFAULT,
                    extra_drop: Sequence[str] = ()) -> List[str]:
    """The harvest's columns minus the positional set (the harvest records
    its own when it has one) and `extra_drop`."""
    positional = set(harvest.doc.get("positional") or conv.positional)
    positional |= {str(c) for c in extra_drop}
    return [n for n in harvest.names if n not in positional]


def column_groups(names: Sequence[str], schema) -> Tuple[np.ndarray, int]:
    """``(group index per column, n_groups)`` from a ``{name, channel,
    reduction}`` schema; a column the schema does not name is its own group."""
    chan = {}
    for e in schema or []:
        c = str(e.get("channel") or "")
        if c:
            chan[str(e.get("name"))] = c
    keys: Dict[str, int] = {}
    out = np.zeros(len(names), np.int64)
    for i, n in enumerate(names):
        g = chan.get(str(n), f"__col__{n}")
        out[i] = keys.setdefault(g, len(keys))
    return out, len(keys)


def build_design(harvest: Harvest, settings: TrainSettings, conv: FieldConventions = DEFAULT
                 ) -> Design:
    names = feature_columns(harvest, conv, settings.extra_drop)
    if not names:
        raise ValueError("the harvest has no feature columns after dropping the positional ones")
    idx = [harvest.names.index(n) for n in names]
    X = harvest.rows[:, idx]
    mean, std, log_mask = input_convention(X, names, settings.log_columns)
    Z = prepare_inputs(X, mean, std, log_mask).astype(np.float32)
    group_of, n_groups = column_groups(names, harvest.doc.get("schema"))
    g = context.directed_graph(harvest.ia, harvest.ib, harvest.n_rows, np)
    deg = np.diff(g.indptr)
    anchors = np.nonzero(deg > 0)[0]
    level = harvest.doc.get("level")
    scope = harvest.doc.get("scope") or (f"L{int(level)}" if level is not None else None)
    info = {"n_rows": int(harvest.n_rows), "n_arcs": int(g.n_edges), "n_anchors": int(len(anchors)),
            "n_shards": len(harvest.shards), "n_in": len(names), "n_groups": int(n_groups)}
    return Design(names, Z, mean, std, log_mask, group_of, n_groups, g, anchors, scope, info)


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def random_walk(g: context.Graph, start: np.ndarray, walk: int, rng: np.random.Generator
                ) -> np.ndarray:
    """One endpoint per start row: a uniform random walk of 1..`walk` hops
    (length drawn per row). A row with no neighbours stays where it is."""
    cur = np.asarray(start, np.int64).copy()
    hops = rng.integers(1, max(1, int(walk)) + 1, size=len(cur))
    indptr, dst = g.indptr, g.dst
    for h in range(int(hops.max()) if len(hops) else 0):
        live = hops > h
        c = cur[live]
        deg = indptr[c + 1] - indptr[c]
        ok = deg > 0
        pick = indptr[c] + np.floor(rng.random(len(c)) * np.maximum(deg, 1)).astype(np.int64)
        nxt = np.where(ok, dst[np.minimum(pick, len(dst) - 1)] if len(dst) else c, c)
        cur[live] = nxt
    return cur


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
def _device(name: str):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _corrupt(x, group_of_t, n_groups, settings: TrainSettings, gen):
    """Both corruptions on a batch tensor, returning a new tensor."""
    import torch
    n, f = x.shape
    out = x
    if settings.group_drop > 0:
        drop_g = torch.rand(n, n_groups, generator=gen, device=x.device) < settings.group_drop
        drop = drop_g[:, group_of_t]                                   # (n, f)
        out = torch.where(drop, torch.zeros_like(out), out)
    if settings.corrupt > 0:
        perm = torch.randperm(n, generator=gen, device=x.device)
        swap = torch.rand(n, f, generator=gen, device=x.device) < settings.corrupt
        out = torch.where(swap, x[perm], out)
    return out


def _vc_penalty(z):
    import torch
    n, d = z.shape
    zc = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(zc.var(dim=0, unbiased=False) + 1e-4)
    var_loss = torch.relu(1.0 - std).mean()
    if n < 2:
        return var_loss, torch.zeros((), device=z.device)
    cov = (zc.T @ zc) / (n - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var_loss, (off ** 2).sum() / d


def _nce(ha, hp, temp: float):
    import torch
    import torch.nn.functional as F
    ha = F.normalize(ha, dim=1)
    hp = F.normalize(hp, dim=1)
    logits = ha @ hp.T / float(temp)
    target = torch.arange(len(ha), device=ha.device)
    loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
    acc = (logits.argmax(dim=1) == target).float().mean()
    return loss, acc


def train(harvest: Harvest, settings: TrainSettings, log: Callable[[str], None] = print,
          conv: FieldConventions = DEFAULT) -> Tuple[EncoderBundle, List[Dict[str, Any]]]:
    if settings.arch not in ARCHS:
        raise ValueError(f"unknown arch {settings.arch!r} (expected one of {ARCHS})")
    t0 = time.perf_counter()
    d = build_design(harvest, settings, conv)
    log(f"design: {d.info['n_rows']} rows x {d.info['n_in']} columns in {d.info['n_groups']} groups, "
        f"{d.info['n_arcs']} arcs, {d.info['n_anchors']} anchors, scope {d.scope or 'none'}")
    provenance = {"harvest": {k: harvest.doc.get(k) for k in
                              ("profile_hash", "level", "roi", "halo", "factors", "slides",
                               "persistence_abs", "blank", "tissue")},
                  "design": dict(d.info), "settings": settings.to_dict(),
                  "python": platform.python_version()}
    if settings.arch == "pca":
        layer, ratio = pca_layer(d.Z, settings.dim, seed=settings.seed)
        layers = [layer]
        history = [{"epoch": 0, "explained_variance": [float(r) for r in ratio]}]
        log(f"pca-{len(ratio)}: explained variance {ratio.sum():.3f}")
        bundle = _finish(d, layers, provenance, {"arch": "pca"}, history, t0)
        return bundle, history
    return _train_mlp(d, settings, provenance, log, t0)


def _train_mlp(d: Design, settings: TrainSettings, provenance, log, t0):
    import torch
    if len(d.anchors) < 2:
        raise ValueError("the harvest has no region arcs to walk on; use --arch pca or "
                         "harvest with a pipeline that reports region_arcs()")
    dev = _device(settings.device)
    torch.manual_seed(int(settings.seed))
    rng = np.random.default_rng(int(settings.seed))
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(settings.seed))

    model = build_torch_model(d.Z.shape[1], settings.hidden, settings.dim, settings.proj).to(dev)
    Zt = torch.from_numpy(d.Z).to(dev)
    group_of_t = torch.from_numpy(d.group_of).to(dev)

    # A held-out anchor set scores the objective without corruption at the
    # end, so the reported accuracy is not the training batches' own.
    anchors = rng.permutation(d.anchors)
    n_hold = int(min(max(len(anchors) * HOLDOUT_FRACTION, 0), 20_000))
    n_hold = n_hold if len(anchors) - n_hold >= 2 else 0
    hold, fit = anchors[:n_hold], anchors[n_hold:]
    batch = int(max(2, min(settings.batch, len(fit))))
    steps_per_epoch = max(1, len(fit) // batch)
    total_steps = steps_per_epoch * int(settings.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=float(settings.lr),
                            weight_decay=float(settings.weight_decay))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1.0 + math.cos(math.pi * min(s, total_steps) / max(total_steps, 1))))
    log(f"train: {settings.arch} {d.Z.shape[1]} -> {' -> '.join(map(str, settings.hidden))} -> "
        f"{settings.dim} on {dev}, {len(fit)} anchors ({n_hold} held out), batch {batch}, "
        f"{settings.epochs} epochs x {steps_per_epoch} steps")

    history: List[Dict[str, Any]] = []
    step = 0
    for epoch in range(1, int(settings.epochs) + 1):
        te = time.perf_counter()
        model.train()
        order = rng.permutation(fit)
        sums = np.zeros(5, np.float64)
        for k in range(steps_per_epoch):
            a = order[k * batch:(k + 1) * batch]
            p = random_walk(d.graph, a, settings.walk, rng)
            at = torch.from_numpy(a).to(dev)
            pt = torch.from_numpy(p).to(dev)
            xa_clean = Zt[at]
            xa = _corrupt(xa_clean, group_of_t, d.n_groups, settings, gen)
            xp = _corrupt(Zt[pt], group_of_t, d.n_groups, settings, gen)
            za, ha, ra = model(xa)
            _zp, hp, _rp = model(xp)
            nce, acc = _nce(ha, hp, settings.temp)
            rec = torch.mean((ra - xa_clean) ** 2)
            var_l, cov_l = _vc_penalty(za)
            loss = nce + settings.rec * rec + settings.vc * (var_l + cov_l)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            sums += np.array([loss.item(), nce.item(), rec.item(), (var_l + cov_l).item(),
                              acc.item()])
        m = sums / steps_per_epoch
        entry = {"epoch": epoch, "loss": float(m[0]), "nce": float(m[1]), "rec": float(m[2]),
                 "vc": float(m[3]), "acc": float(m[4]), "lr": float(opt.param_groups[0]["lr"]),
                 "seconds": time.perf_counter() - te}
        history.append(entry)
        log(f"epoch {epoch:3d}  loss {m[0]:.4f}  nce {m[1]:.4f}  rec {m[2]:.4f}  "
            f"vc {m[3]:.4f}  acc {m[4]:.3f}  ({entry['seconds']:.1f}s)")

    # Held-out score: clean inputs, walk positives, batch negatives.
    heldout = None
    if n_hold >= 2:
        model.eval()
        with torch.no_grad():
            accs = []
            for k in range(0, n_hold, batch):
                a = hold[k:k + batch]
                if len(a) < 2:
                    continue
                p = random_walk(d.graph, a, settings.walk, rng)
                _za, ha, _ = model(Zt[torch.from_numpy(a).to(dev)])
                _zp, hp, _ = model(Zt[torch.from_numpy(p).to(dev)])
                _l, acc = _nce(ha, hp, settings.temp)
                accs.append((float(acc), len(a)))
            if accs:
                heldout = float(sum(a * n for a, n in accs) / sum(n for _a, n in accs))
        log(f"held-out walk accuracy: {heldout:.3f} over {n_hold} anchors "
            f"(chance {1.0 / batch:.4f})")

    layers = model.encoder_layers()
    extra = {"arch": "mlp", "torch": torch.__version__, "device": str(dev),
             "heldout_walk_acc": heldout, "heldout_batch": batch}
    bundle = _finish(d, layers, provenance, extra, history, t0)
    return bundle, history


def _finish(d: Design, layers: List[Layer], provenance, extra, history, t0) -> EncoderBundle:
    """Measure the latent's whitening over the harvest and assemble the bundle."""
    probe = EncoderBundle(names=list(d.names), mean=d.mean, std=d.std, log_mask=d.log_mask,
                          layers=layers, latent_mean=np.zeros(layers[-1].W.shape[1]),
                          latent_std=np.ones(layers[-1].W.shape[1]), scope=d.scope)
    z = np.concatenate([probe.encode_prepared(d.Z[k:k + 65536], whiten=False)
                        for k in range(0, len(d.Z), 65536)]).astype(np.float64)
    lm = z.mean(axis=0)
    ls = z.std(axis=0)
    ls = np.where(ls > 1e-8, ls, 1.0)
    meta = dict(provenance)
    meta.update(extra)
    meta["latent_std_raw"] = [float(v) for v in z.std(axis=0)]
    meta["final"] = dict(history[-1]) if history else {}
    meta["train_seconds"] = time.perf_counter() - t0
    return EncoderBundle(names=list(d.names), mean=d.mean, std=d.std, log_mask=d.log_mask,
                         layers=layers, latent_mean=lm, latent_std=ls, scope=d.scope, meta=meta)
