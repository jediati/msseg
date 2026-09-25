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
``StandardScaler``, fit on every seam a scope or a trace has labelled, one
class per class of the polyline task's vocabulary (class 1 is the "not a
boundary" role, the rest are kinds of boundary). It gives every seam of every
item its class probabilities and a **boundaryness** in [0, 1] -- 1 - p(class
1) -- that the trace tool's ``model`` toll and the boundaryness colouring read. ``evaluate_seams`` is the
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
from .seams import SEAM_BOUNDARY, SEAM_INTERIOR

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
    # The task's seam classes the model was fit on (class 1 = "not a
    # boundary"), and the vocabulary size its probability columns span.
    classes: List[int] = field(default_factory=lambda: [1, 2])
    n_classes: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "spec": self.spec.to_dict(), "n_in": int(self.n_in),
                "feature_names": list(self.feature_names), "names_hash": self.names_hash,
                "net_hash": self.net_hash, "embed_used": self.embed_used,
                "n_seams": int(self.n_seams), "n_boundary": int(self.n_boundary),
                "fit_s": float(self.fit_s), "report": self.report,
                "classes": [int(c) for c in self.classes], "n_classes": int(self.n_classes)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SeamModel":
        return cls(model=d["model"], spec=SeamSpec.from_dict(d.get("spec")),
                   n_in=int(d["n_in"]), feature_names=list(d.get("feature_names") or []),
                   names_hash=str(d.get("names_hash", "")), net_hash=str(d.get("net_hash", "")),
                   embed_used=str(d.get("embed_used") or "features"),
                   n_seams=int(d.get("n_seams", 0)), n_boundary=int(d.get("n_boundary", 0)),
                   fit_s=float(d.get("fit_s", 0.0)), report=d.get("report"),
                   classes=[int(c) for c in (d.get("classes") or [1, 2])],
                   n_classes=int(d.get("n_classes") or 3))

    def brief(self) -> str:
        text = (f"{self.spec.model} · {len(self.classes)} classes on {self.n_seams:,} seams "
                f"({self.n_boundary / max(1, self.n_seams):.0%} boundary, {self.embed_used})")
        m = (self.report or {}).get("mean")
        if m:
            text += f" · held-out bal.acc {m['bacc']:.0%}, boundary AUC {m['auc']:.2f}"
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


def _proba_columns(est, F, n_classes) -> np.ndarray:
    """``(S, n_classes)`` class probabilities, column k = class k (columns of
    classes the estimator never saw stay 0)."""
    p = np.asarray(est.predict_proba(F), np.float64)
    out = np.zeros((len(F), int(n_classes)), np.float64)
    for j, c in enumerate(est.classes_):
        if 0 <= int(c) < out.shape[1]:
            out[:, int(c)] = p[:, j]
    return out


def boundaryness_of(proba) -> np.ndarray:
    """p(any boundary) = 1 - p(class 1, "not a boundary")."""
    proba = np.asarray(proba, np.float64)
    p1 = proba[:, SEAM_INTERIOR] if proba.shape[1] > SEAM_INTERIOR else np.zeros(len(proba))
    return np.clip(1.0 - p1, 1e-4, 1 - 1e-4).astype(np.float32)


def labeled_binary(y) -> Tuple[np.ndarray, np.ndarray]:
    """``(mask, yb)``: the labelled seams and their boundary (1) / not (0)
    target -- class 1 is the "not a boundary" role, every higher class a
    kind of boundary."""
    y = np.asarray(y, np.int64)
    mask = y > 0
    return mask, (y >= SEAM_BOUNDARY).astype(int)


def fit_seam_model(F, y, spec: Optional[SeamSpec], feature_names: Sequence[str],
                   embed: str = "features", net_hash: str = "",
                   n_classes: Optional[int] = None) -> SeamModel:
    """Fit on the labelled seams (``y > 0``), one class per task seam class.
    Raises ValueError with fewer than two classes labelled -- e.g. scope some
    seams into class 1 (not a boundary) and trace some into class 2."""
    spec = spec or SeamSpec()
    F = np.asarray(F, np.float32)
    y = np.asarray(y, np.int64)
    mask = y > 0
    Fl, yl = F[mask], y[mask]
    classes = sorted(int(c) for c in np.unique(yl))
    if len(classes) < 2:
        raise ValueError("need seams of at least two classes - e.g. a scope in class 1 "
                         "(not a boundary) and a trace in class 2")
    t0 = time.perf_counter()
    est = _make_estimator(spec)
    _fit(est, Fl, yl, spec)
    return SeamModel(model=est, spec=spec, n_in=int(F.shape[1]),
                     feature_names=list(feature_names),
                     names_hash=edge_model.names_hash(list(feature_names)),
                     net_hash=str(net_hash), embed_used=str(embed),
                     n_seams=int(len(yl)), n_boundary=int((yl >= SEAM_BOUNDARY).sum()),
                     fit_s=time.perf_counter() - t0, classes=classes,
                     n_classes=int(n_classes) if n_classes else max(classes) + 1)


def _check_inputs(model: SeamModel, F):
    F = np.asarray(F, np.float32)
    if F.shape[1] != model.n_in:
        raise ValueError(f"seam model expects {model.n_in} inputs, got {F.shape[1]}")
    return F


def predict_proba(model: SeamModel, F) -> np.ndarray:
    """``(S, model.n_classes)`` p(class k) per seam, column k = class k."""
    F = _check_inputs(model, F)
    if len(F) == 0:
        return np.zeros((0, model.n_classes), np.float64)
    return _proba_columns(model.model, F, model.n_classes)


def predict_boundaryness(model: SeamModel, F) -> np.ndarray:
    """float32 ``[S]`` p(any boundary) per seam."""
    F = _check_inputs(model, F)
    if len(F) == 0:
        return np.zeros(0, np.float32)
    return boundaryness_of(predict_proba(model, F))


def predicted_class(proba) -> np.ndarray:
    """int ``[S]``: the most probable class per seam (0 where nothing is)."""
    proba = np.asarray(proba, np.float64)
    if proba.size == 0:
        return np.zeros(len(proba), np.int64)
    out = np.argmax(proba, axis=1).astype(np.int64)
    out[proba.max(axis=1) <= 0] = 0
    return out


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_seams(F, y, groups, spec: Optional[SeamSpec] = None, seed: int = 0,
                   progress_cb=None, stop_event=None, n_splits: int = 5) -> Dict[str, Any]:
    """Leave-items-out CV of the seam model: per fold a fresh estimator on
    the training items' labelled seams, scored on the held-out ones --
    balanced accuracy over the classes, per-class recall / precision, the
    boundary AUC (1 - p(class 1) against "any boundary class"), log-loss --
    averaged over the folds."""
    from . import model_search
    spec = spec or SeamSpec()
    t0 = time.perf_counter()
    F = np.asarray(F, np.float32)
    y = np.asarray(y, np.int64)
    mask = y > 0
    Fl, yl = F[mask], y[mask]
    gl = np.asarray(groups, object)[mask] if groups is not None else None
    classes = sorted(int(c) for c in np.unique(yl))
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
        folds.append(_score_fold(yl[te], _proba_columns(est, Fl[te], max(classes) + 1),
                                 classes))
        if progress_cb is not None:
            progress_cb(f, len(splits))
    mean = {}
    if folds:
        for k in ("bacc", "auc", "logloss"):
            mean[k] = float(np.nanmean([fd[k] for fd in folds]))
        mean["n"] = int(sum(fd["n"] for fd in folds))
        mean["per_class"] = {
            c: {m: float(np.nanmean([fd["per_class"][c][m] for fd in folds]))
                for m in ("recall", "precision")}
            for c in classes}
    return {"n": int(len(yl)), "n_boundary": int((yl >= SEAM_BOUNDARY).sum()),
            "classes": classes, "folds": folds, "mean": mean, "cv_kind": kind,
            "n_folds": len(splits), "elapsed_s": time.perf_counter() - t0,
            "stopped": stopped, "spec": spec.to_dict()}


def _score_fold(y, proba, classes) -> Dict[str, Any]:
    """One held-out fold's scores (NaN where a fold cannot tell)."""
    from sklearn.metrics import (balanced_accuracy_score, log_loss,
                                 precision_recall_fscore_support, roc_auc_score)
    y = np.asarray(y, np.int64)
    yhat = predicted_class(proba)
    bacc = float(balanced_accuracy_score(y, yhat)) if len(y) else float("nan")
    yb = y >= SEAM_BOUNDARY
    auc = (float(roc_auc_score(yb, boundaryness_of(proba)))
           if yb.any() and (~yb).any() else float("nan"))
    p = np.clip(proba[:, classes], 1e-9, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    try:
        ll = float(log_loss(y, p, labels=classes))
    except ValueError:
        ll = float("nan")
    prec, rec, _f, _s = precision_recall_fscore_support(y, yhat, labels=classes,
                                                        zero_division=0)
    per = {c: {"recall": float(rec[i]), "precision": float(prec[i])}
           for i, c in enumerate(classes)}
    return {"n": int(len(y)), "bacc": bacc, "auc": auc, "logloss": ll, "per_class": per}


def summary(report: Dict[str, Any]) -> str:
    m = report.get("mean") or {}
    if not m:
        return "no fold could be scored"
    text = (f"held-out bal.acc {m['bacc']:.1%}, boundary AUC {m['auc']:.3f}, "
            f"log-loss {m['logloss']:.3f} · "
            f"{report.get('n_folds', 0)}-fold {report.get('cv_kind', '')} CV on "
            f"{report.get('n', 0)} seams ({report.get('n_boundary', 0)} boundary)")
    if report.get("stopped"):
        text += " (stopped early)"
    return text


__all__ = ["FEATURE_KINDS", "DEFAULT_FEATURES", "MODELS", "EMBEDS", "SeamSpec",
           "region_feature_names", "zscore_rows", "has_embedding", "embed_used",
           "seam_features", "gather_seams", "SeamModel", "fit_seam_model",
           "predict_boundaryness", "labeled_binary", "evaluate_seams", "summary"]
