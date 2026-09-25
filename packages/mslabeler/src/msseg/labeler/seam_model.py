"""The seam model: boundary vs interior from a seam descriptor.

A **seam descriptor** is the per-seam feature vector the model reads:

  pair       ``|d|`` and the product of the two flanks' region descriptors --
             the z-scored statistics rows (``embed = "features"``), or the
             base region net's last hidden layer when one is trained
             (``"net"``, via ``edge_model.embed``; ``"auto"`` picks it when
             available). Symmetric in the flanks, like the edge model.
  barrier    ``edge_model.barrier``: the arc's saddle depth and the flanks'
             extremum gap (two columns, zero without MSC saddles).
  edges      the '-> edges' model's p(diff) for the pair, plus a present flag.
  geometry   crack length, its log, the bbox aspect, the tortuosity
             (length / chord) and whether the seam is a loop.

The model is a balanced logistic regression (or a small MLP) behind a
``StandardScaler``, fit on every seam a scope or a trace has labelled
(``seams.SEAM_INTERIOR`` = 0, ``SEAM_BOUNDARY`` = 1); it scores every seam of
every item with a **boundaryness** in [0, 1] that the trace tool's ``model``
toll and the overlay's boundaryness colouring read. ``evaluate_seams`` is the
leave-items-out report (``model_search.make_cv`` over the catalogue's groups).

Headless (sklearn under the labeler's ``[classify]`` extra); pytest-covered.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import edge_model, seam_path
from .fields import DEFAULT
from .seams import SEAM_BOUNDARY

FEATURE_KINDS = ("pair", "barrier", "edges", "geometry")
DEFAULT_FEATURES = FEATURE_KINDS
MODELS = ("logistic", "mlp")
EMBEDS = ("auto", "features", "net")
GEOMETRY_NAMES = ("length", "log_length", "bbox_aspect", "tortuosity", "is_loop")


@dataclass
class SeamSpec:
    """What the seam model is, as plain data (rides the pickle)."""
    features: Tuple[str, ...] = DEFAULT_FEATURES
    model: str = "logistic"
    C: float = 1.0
    embed: str = "auto"
    layer: int = -1
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"features": list(self.features), "model": self.model, "C": float(self.C),
                "embed": self.embed, "layer": int(self.layer), "seed": int(self.seed)}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "SeamSpec":
        d = d or {}
        feats = tuple(f for f in (d.get("features") or DEFAULT_FEATURES) if f in FEATURE_KINDS)
        return cls(features=feats or DEFAULT_FEATURES,
                   model=str(d.get("model") or "logistic"),
                   C=float(d.get("C", 1.0)), embed=str(d.get("embed") or "auto"),
                   layer=int(d.get("layer", -1)), seed=int(d.get("seed", 0)))

    def describe(self) -> str:
        return f"{self.model} on {'+'.join(self.features)} (embed {self.embed})"


# --------------------------------------------------------------------------- #
# The seam descriptor
# --------------------------------------------------------------------------- #
def region_feature_names(table, conv=DEFAULT) -> List[str]:
    """The table's non-positional columns: what the ``features`` embedding
    reads (the same rule the region classifier uses)."""
    return [n for n in table.names if n not in conv.positional]


def _design(table, names, np_) -> np.ndarray:
    cols = []
    for n in names:
        c = table.column(n)
        if c is None:
            raise ValueError(f"the statistics table has no column {n!r}")
        cols.append(np_.nan_to_num(np_.asarray(c, np_.float64), nan=0.0))
    if not cols:
        return np_.zeros((table.n_rows, 0), np_.float64)
    return np_.stack(cols, axis=1)


def zscore_rows(table, names, np_) -> np.ndarray:
    """The region descriptors over `names`, each column z-scored over the
    slice (a constant column becomes zero), NaN as zero. float32 ``[n, d]``."""
    X = _design(table, names, np_)
    if X.shape[1] == 0:
        return X.astype(np_.float32)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np_.where(sd > 1e-12, sd, 1.0)
    return ((X - mu) / sd).astype(np_.float32)


def has_embedding(pipeline) -> bool:
    try:
        edge_model.hidden_widths(pipeline)
        return True
    except (TypeError, AttributeError):
        return False


def embed_used(spec: SeamSpec, base_pipeline) -> str:
    """Which region embedding the pair block reads under `spec` with this base."""
    if spec.embed == "net":
        if base_pipeline is None or not has_embedding(base_pipeline):
            raise ValueError("embed 'net' needs a trained dense region model")
        return "net"
    if spec.embed == "auto" and base_pipeline is not None and has_embedding(base_pipeline):
        return "net"
    return "features"


def seam_features(graph, table, names, np_, conv=DEFAULT, base_pipeline=None, arcs=None,
                  pdiff=None, spec: Optional[SeamSpec] = None):
    """``(F float32[S, d], feature_names)``: the seam descriptor of every seam
    of `graph`. `names` are the region feature columns the pair block reads
    (the base net's input names when it embeds, else the table's own).
    Seams whose flank has no table row get zero pair terms."""
    spec = spec or SeamSpec()
    S = graph.n_seams
    ia, ib, keep = seam_path.flank_rows(graph, table, np_, conv)
    blocks: List[np.ndarray] = []
    fnames: List[str] = []
    if "pair" in spec.features:
        how = embed_used(spec, base_pipeline)
        if how == "net":
            X = _design(table, names, np_)
            Z = edge_model.embed(base_pipeline, X)[spec.layer]
            prefix = [f"h{k}" for k in range(Z.shape[1])]
        else:
            Z = zscore_rows(table, names, np_)
            prefix = list(names)
        d = Z.shape[1]
        absd = np_.zeros((S, d), np_.float32)
        prod = np_.zeros((S, d), np_.float32)
        if keep.any():
            absd[keep] = np_.abs(Z[ia] - Z[ib])
            prod[keep] = Z[ia] * Z[ib]
        blocks += [absd, prod]
        fnames += [f"absdiff_{n}" for n in prefix] + [f"prod_{n}" for n in prefix]
    if "barrier" in spec.features:
        sad = None
        if arcs is not None and arcs.get("saddle") is not None:
            sad = seam_path.arc_values(graph, arcs, arcs["saddle"], np_)
        ext = table.column(conv.extremum_value_field)
        ea = eb = None
        if ext is not None:
            ext = np_.asarray(ext, np_.float64)
            ea = np_.zeros(S, np_.float64)
            eb = np_.zeros(S, np_.float64)
            ea[keep], eb[keep] = ext[ia], ext[ib]
        bar = edge_model.barrier(sad, ea, eb, n=S)
        blocks.append(np_.nan_to_num(bar, nan=0.0).astype(np_.float32))
        fnames += ["barrier_depth", "barrier_gap"]
    if "edges" in spec.features:
        pd = (seam_path.arc_values(graph, arcs, pdiff, np_)
              if pdiff is not None and arcs is not None else np_.full(S, np_.nan))
        present = np_.isfinite(pd)
        col = np_.where(present, pd, 0.5)
        blocks.append(np_.stack([col, present.astype(np_.float64)], axis=1).astype(np_.float32))
        fnames += ["pdiff", "pdiff_present"]
    if "geometry" in spec.features:
        L = graph.lengths(np_).astype(np_.float64)
        bb = graph.bboxes(np_).astype(np_.float64)
        w = bb[:, 2] - bb[:, 0]
        h = bb[:, 3] - bb[:, 1]
        aspect = (np_.minimum(w, h) + 1.0) / (np_.maximum(w, h) + 1.0)
        if S:
            first = graph.points[graph.offsets[:-1]].astype(np_.float64)
            last = graph.points[graph.offsets[1:] - 1].astype(np_.float64)
            chord = np_.hypot(first[:, 0] - last[:, 0], first[:, 1] - last[:, 1])
        else:
            chord = np_.zeros(0)
        tort = L / (chord + 1.0)
        loop = (graph.j0 < 0).astype(np_.float64)
        blocks.append(np_.stack([L, np_.log1p(L), aspect, tort, loop], axis=1).astype(np_.float32))
        fnames += list(GEOMETRY_NAMES)
    if not blocks:
        raise ValueError("the seam spec selects no features")
    F = np_.concatenate(blocks, axis=1).astype(np_.float32)
    return np_.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0), fnames


def gather_seams(items) -> Dict[str, np.ndarray]:
    """Stack ``(key, F, cls, group)`` per item into one design matrix:
    ``{"F", "y", "group", "item", "seam"}`` (`y` is the seam class, 0 =
    unknown)."""
    Fs, ys, gs, ks, ss = [], [], [], [], []
    for key, F, cls, group in items:
        F = np.asarray(F, np.float32)
        Fs.append(F)
        ys.append(np.asarray(cls, np.int64).reshape(-1))
        gs.extend([group] * len(F))
        ks.extend([key] * len(F))
        ss.append(np.arange(len(F), dtype=np.int64))
    if not Fs:
        return {"F": np.zeros((0, 0), np.float32), "y": np.zeros(0, np.int64),
                "group": np.zeros(0, object), "item": np.zeros(0, object),
                "seam": np.zeros(0, np.int64)}
    return {"F": np.concatenate(Fs, axis=0), "y": np.concatenate(ys),
            "group": np.asarray(gs, object), "item": np.asarray(ks, object),
            "seam": np.concatenate(ss)}


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
@dataclass
class SeamModel:
    model: Any
    spec: SeamSpec
    n_in: int
    feature_names: List[str]
    names_hash: str = ""            # over the seam descriptor names
    net_hash: str = ""              # the base net's, when the pair block embeds with it
    embed_used: str = "features"
    n_seams: int = 0
    n_boundary: int = 0
    fit_s: float = 0.0
    report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "spec": self.spec.to_dict(), "n_in": int(self.n_in),
                "feature_names": list(self.feature_names), "names_hash": self.names_hash,
                "net_hash": self.net_hash, "embed_used": self.embed_used,
                "n_seams": int(self.n_seams), "n_boundary": int(self.n_boundary),
                "fit_s": float(self.fit_s), "report": self.report}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SeamModel":
        return cls(model=d["model"], spec=SeamSpec.from_dict(d.get("spec")),
                   n_in=int(d["n_in"]), feature_names=list(d.get("feature_names") or []),
                   names_hash=str(d.get("names_hash", "")), net_hash=str(d.get("net_hash", "")),
                   embed_used=str(d.get("embed_used") or "features"),
                   n_seams=int(d.get("n_seams", 0)), n_boundary=int(d.get("n_boundary", 0)),
                   fit_s=float(d.get("fit_s", 0.0)), report=d.get("report"))

    def brief(self) -> str:
        text = (f"{self.spec.model} on {self.n_seams:,} seams "
                f"({self.n_boundary / max(1, self.n_seams):.0%} boundary, {self.embed_used})")
        m = (self.report or {}).get("mean")
        if m:
            text += f" · held-out AUC {m['auc']:.2f}, bal.acc {m['bacc']:.0%}"
        return text

    def describe(self) -> str:
        return f"{self.spec.describe()}; {self.n_in} inputs; {self.brief()}"


def _make_estimator(spec: SeamSpec):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if spec.model == "mlp":
        from sklearn.neural_network import MLPClassifier
        return make_pipeline(StandardScaler(),
                             MLPClassifier((32, 16), max_iter=1000, random_state=int(spec.seed)))
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=3000,
                                            C=float(spec.C)))


def _fit(est, F, y, spec: SeamSpec):
    if spec.model == "mlp":
        from sklearn.utils.class_weight import compute_sample_weight
        try:
            return est.fit(F, y, mlpclassifier__sample_weight=compute_sample_weight("balanced", y))
        except TypeError:
            return est.fit(F, y)
    return est.fit(F, y)


def _proba_boundary(est, F) -> np.ndarray:
    p = np.asarray(est.predict_proba(F), np.float64)
    classes = [int(c) for c in est.classes_]
    if 1 in classes:
        out = p[:, classes.index(1)]
    else:
        out = np.zeros(len(F))
    return np.clip(out, 1e-4, 1 - 1e-4).astype(np.float32)


def labeled_binary(y) -> Tuple[np.ndarray, np.ndarray]:
    """``(mask, yb)``: the labelled seams and their boundary (1) / not (0)
    target -- class 1 is the "not a boundary" role, every higher class a
    kind of boundary."""
    y = np.asarray(y, np.int64)
    mask = y > 0
    return mask, (y >= SEAM_BOUNDARY).astype(int)


def fit_seam_model(F, y, spec: Optional[SeamSpec], feature_names: Sequence[str],
                   embed: str = "features", net_hash: str = "") -> SeamModel:
    """Fit on the labelled seams (``y > 0``). Raises ValueError when one class
    is missing -- draw a scope (interior) and a trace (boundary) first."""
    spec = spec or SeamSpec()
    F = np.asarray(F, np.float32)
    mask, yb = labeled_binary(y)
    Fl, yl = F[mask], yb[mask]
    if len(yl) == 0 or yl.min() == yl.max():
        raise ValueError("need both boundary and interior seams - draw a scope "
                         "(interior by default) and a trace (boundary)")
    t0 = time.perf_counter()
    est = _make_estimator(spec)
    _fit(est, Fl, yl, spec)
    return SeamModel(model=est, spec=spec, n_in=int(F.shape[1]),
                     feature_names=list(feature_names),
                     names_hash=edge_model.names_hash(list(feature_names)),
                     net_hash=str(net_hash), embed_used=str(embed),
                     n_seams=int(len(yl)), n_boundary=int(yl.sum()),
                     fit_s=time.perf_counter() - t0)


def predict_boundaryness(model: SeamModel, F) -> np.ndarray:
    """float32 ``[S]`` p(boundary) per seam."""
    F = np.asarray(F, np.float32)
    if F.shape[1] != model.n_in:
        raise ValueError(f"seam model expects {model.n_in} inputs, got {F.shape[1]}")
    if len(F) == 0:
        return np.zeros(0, np.float32)
    return _proba_boundary(model.model, F)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_seams(F, y, groups, spec: Optional[SeamSpec] = None, seed: int = 0,
                   progress_cb=None, stop_event=None, n_splits: int = 5) -> Dict[str, Any]:
    """Leave-items-out CV of the seam model: per fold a fresh estimator on
    the training items' labelled seams, scored on the held-out ones
    (``edge_model._score``: balanced accuracy, AUC, log-loss, boundary
    recall / precision), averaged."""
    from . import model_search
    spec = spec or SeamSpec()
    t0 = time.perf_counter()
    F = np.asarray(F, np.float32)
    mask, yb = labeled_binary(y)
    Fl, yl = F[mask], yb[mask]
    gl = np.asarray(groups, object)[mask] if groups is not None else None
    cv, kind = model_search.make_cv(yl, gl, n_splits=n_splits, seed=seed)
    splits = list(cv.split(Fl, yl, gl if kind == "slices" else None))
    folds = []
    stopped = False
    for f, (tr, te) in enumerate(splits, start=1):
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break
        est = _make_estimator(spec)
        _fit(est, Fl[tr], yl[tr], spec)
        folds.append(edge_model._score(yl[te], _proba_boundary(est, Fl[te])))
        if progress_cb is not None:
            progress_cb(f, len(splits))
    mean = {}
    if folds:
        for k in folds[0]:
            vals = np.asarray([fd[k] for fd in folds], np.float64)
            mean[k] = float(np.nanmean(vals)) if k != "n" else int(vals.sum())
    return {"n": int(len(yl)), "n_boundary": int(yl.sum()), "folds": folds, "mean": mean,
            "cv_kind": kind, "n_folds": len(splits), "elapsed_s": time.perf_counter() - t0,
            "stopped": stopped, "spec": spec.to_dict()}


def summary(report: Dict[str, Any]) -> str:
    m = report.get("mean") or {}
    if not m:
        return "no fold could be scored"
    text = (f"held-out log-loss {m['logloss']:.3f}, bal.acc {m['bacc']:.1%}, AUC {m['auc']:.3f}, "
            f"boundary recall {m['diff_recall']:.0%} / precision {m['diff_precision']:.0%} · "
            f"{report.get('n_folds', 0)}-fold {report.get('cv_kind', '')} CV on "
            f"{report.get('n', 0)} seams ({report.get('n_boundary', 0)} boundary)")
    if report.get("stopped"):
        text += " (stopped early)"
    return text


__all__ = ["FEATURE_KINDS", "DEFAULT_FEATURES", "MODELS", "EMBEDS", "SeamSpec",
           "region_feature_names", "zscore_rows", "has_embedding", "embed_used",
           "seam_features", "gather_seams", "SeamModel", "fit_seam_model",
           "predict_boundaryness", "labeled_binary", "evaluate_seams", "summary"]
