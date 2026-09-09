"""Edge models: a pair classifier over the living-region graph, on top of the
region net, and neighbour voting that refines the region net's predictions.

The region classifier scores each living MSC region from its statistics row
alone. Regions are not independent -- they tile the slice and touch through
saddles -- and the edge-pairs experiment (docs/mscoupon_edge_pairs.md) showed
that a logistic model on the pair of a region net's hidden activations tells
same-class from different-class edges better than the net's own argmax, and
that letting the graph vote fixes the isolated flips.

This module is the headless half of the ``-> edges`` model kinds:

* :func:`embed` -- the hidden ReLU layers of a fitted region pipeline
  (sklearn ``MLPClassifier`` or ``torch_mlp.TorchMLPClassifier``), numpy only.
* :func:`gather_edges` -- per-slice arcs (label ids) to global row-index pairs.
* :class:`EdgeModel` / :func:`fit_edge_model` / :func:`predict_pdiff` -- the
  pair model: symmetric features ``|e_a - e_b|``, ``e_a * e_b`` and a saddle
  barrier, a balanced logistic regression (or a small MLP) on top.
* :func:`vote` -- rounds of ``score_i(c) = log P_i(c) + lam * sum_j [log(1 -
  p_ij) if class_j == c else log p_ij]`` over the region graph.
* :func:`evaluate_edges` -- leave-slices-out report: the pair model against
  the region net's implied answers, and region accuracy before / after voting.

Nothing here touches Tk or the engine; the labeler gathers rows/arcs on its
thread and hands numpy arrays in.
"""
from __future__ import annotations

import dataclasses
import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import magic_fill

LAYER_CHOICES = {"last": -1, "previous": -2}
EDGE_MODELS = ("logistic", "mlp")
FEATURE_KINDS = ("absdiff", "prod", "barrier")
# The report rows evaluate_edges scores (first two come from the region net).
EVAL_ROWS = ("argmax differs", "1 - sum P_a P_b", "learned")
_EPS = 1e-4


@dataclass
class EdgeSpec:
    """What the pair model is, as plain data (rides the pickle and the session)."""
    layer: int = -1                       # -1 = last hidden layer, -2 = previous
    features: Tuple[str, ...] = FEATURE_KINDS
    model: str = "logistic"               # "logistic" | "mlp"
    C: float = 1.0                        # logistic inverse regularisation
    lam: float = 1.0                      # voting weight of the edge terms
    rounds: int = 3                       # voting rounds (0 = no refinement)
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["features"] = list(self.features)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EdgeSpec":
        d = dict(d or {})
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        feats = tuple(str(f) for f in (d.get("features") or FEATURE_KINDS)
                      if str(f) in FEATURE_KINDS) or FEATURE_KINDS
        d["features"] = feats
        d["layer"] = int(d.get("layer", -1))
        d["model"] = str(d.get("model") or "logistic")
        if d["model"] not in EDGE_MODELS:
            d["model"] = "logistic"
        d["C"] = float(d.get("C", 1.0))
        d["lam"] = float(d.get("lam", 1.0))
        d["rounds"] = int(d.get("rounds", 3))
        d["seed"] = int(d.get("seed", 0))
        return cls(**d)

    def layer_text(self, widths: Optional[Sequence[int]] = None) -> str:
        name = "last" if self.layer == -1 else ("previous" if self.layer == -2 else str(self.layer))
        if widths:
            try:
                return f"{name} ({int(widths[self.layer])}-d)"
            except (IndexError, TypeError, ValueError):
                pass
        return name

    def describe(self, widths: Optional[Sequence[int]] = None, n_in: Optional[int] = None) -> str:
        feats = ", ".join({"absdiff": "|d|", "prod": "product", "barrier": "barrier"}[f]
                          for f in self.features)
        head = "logistic" if self.model == "logistic" else "MLP 32-16"
        text = f"edges: {head} on the {self.layer_text(widths)} layer ({feats}"
        if n_in is not None:
            text += f"; {n_in} inputs"
        text += ")"
        if self.model == "logistic":
            text += f", C {self.C:g}"
        text += f", vote lam {self.lam:g} x {self.rounds} round(s)"
        return text


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def _hidden_layers(pipeline) -> Tuple[Any, List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """(net, hidden (W, b) pairs, output (W, b)) of a fitted pipeline whose last
    step is an MLP; TypeError for anything else (a forest has no embedding)."""
    steps = getattr(pipeline, "steps", None)
    if not steps:
        raise TypeError("region model is not a pipeline with a dense net")
    net = steps[-1][1]
    if hasattr(net, "coefs_") and hasattr(net, "intercepts_"):
        if str(getattr(net, "activation", "relu")) != "relu":
            raise TypeError("only relu hidden layers are supported")
        pairs = [(np.asarray(W, np.float32), np.asarray(b, np.float32))
                 for W, b in zip(net.coefs_, net.intercepts_)]
    elif hasattr(net, "params_"):
        pairs = [(np.asarray(W, np.float32)[0], np.asarray(b, np.float32)[0])
                 for W, b in net.params_]
    else:
        raise TypeError("region model has no hidden layers")
    if len(pairs) < 2:
        raise TypeError("region model has no hidden layers")
    return net, pairs[:-1], pairs[-1]


def embed(pipeline, X) -> List[np.ndarray]:
    """The hidden ReLU activations of every hidden layer, ``[h1, h2, ...]``,
    each ``float32[n, width]``, for rows ``X`` in the pipeline's input space
    (every step but the last is applied first, so FeatureSubset / scaler
    pipelines and the top-N ones all work)."""
    _net, hidden, _out = _hidden_layers(pipeline)
    Z = np.asarray(X, np.float64)
    for _name, step in pipeline.steps[:-1]:
        Z = step.transform(Z)
    h = np.asarray(Z, np.float32)
    out = []
    for W, b in hidden:
        h = np.maximum(h @ W + b, 0.0).astype(np.float32)
        out.append(h)
    return out


def hidden_widths(pipeline) -> List[int]:
    _net, hidden, _out = _hidden_layers(pipeline)
    return [int(W.shape[1]) for W, _b in hidden]


def net_hash(pipeline) -> str:
    """A fingerprint of the region net's OUTPUT layer: an edge model trained on
    one net's embedding must not be applied to another's."""
    _net, _hidden, (W, b) = _hidden_layers(pipeline)
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(W, np.float32).tobytes())
    h.update(np.ascontiguousarray(b, np.float32).tobytes())
    return h.hexdigest()


def names_hash(names: Sequence[str]) -> str:
    return hashlib.sha1("\x1f".join(str(n) for n in names).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Pair features
# --------------------------------------------------------------------------- #
def barrier(saddle, ext_a, ext_b, n=None) -> np.ndarray:
    """``float32[n, 2]``: ``saddle - max(ext_a, ext_b)`` (the persistence-style
    depth of the saddle joining two regions) and ``|ext_a - ext_b|``. Always
    two columns -- zeros where saddles (pixel adjacency) or extremum values
    are unavailable -- so the pair model's input width never depends on the
    slice."""
    if n is None:
        n = len(ext_a) if ext_a is not None else (len(saddle) if saddle is not None else 0)
    out = np.zeros((int(n), 2), np.float32)
    if ext_a is None or ext_b is None or n == 0:
        return out
    ea = np.asarray(ext_a, np.float64)
    eb = np.asarray(ext_b, np.float64)
    out[:, 1] = np.abs(ea - eb)
    if saddle is not None:
        s = np.asarray(saddle, np.float64)
        ok = np.isfinite(s)
        out[ok, 0] = s[ok] - np.maximum(ea[ok], eb[ok])
    return out


def pair_features(emb: np.ndarray, a, b, bar=None,
                  features: Sequence[str] = FEATURE_KINDS) -> np.ndarray:
    """``float32[n, d]`` symmetric in (a, b): ``|e_a - e_b|``, ``e_a * e_b`` and
    the barrier columns, as `features` selects."""
    a = np.asarray(a, np.intp)
    b = np.asarray(b, np.intp)
    ea = np.asarray(emb, np.float32)[a]
    eb = np.asarray(emb, np.float32)[b]
    parts = []
    if "absdiff" in features:
        parts.append(np.abs(ea - eb))
    if "prod" in features:
        parts.append(ea * eb)
    if "barrier" in features:
        if bar is None or len(np.asarray(bar)) != len(a):
            bar = np.zeros((len(a), 2), np.float32)
        parts.append(np.asarray(bar, np.float32).reshape(len(a), -1))
    if not parts:
        parts.append(np.abs(ea - eb))
    return np.concatenate(parts, axis=1).astype(np.float32)


def n_inputs(width: int, features: Sequence[str] = FEATURE_KINDS) -> int:
    d = 0
    if "absdiff" in features:
        d += width
    if "prod" in features:
        d += width
    if "barrier" in features:
        d += 2
    return d or width


# --------------------------------------------------------------------------- #
# Edges
# --------------------------------------------------------------------------- #
def gather_edges(slices: Sequence[Tuple[np.ndarray, Optional[Dict[str, Any]],
                                        np.ndarray, int]]) -> Dict[str, np.ndarray]:
    """Per-slice arcs to global row-index pairs.

    ``slices`` holds ``(fids, arcs, cls, slice_idx)`` per slice, in the order
    the rows are stacked: ``fids`` the table's label ids (one per row), ``arcs``
    the record's ``{"a", "b", "saddle", "source"}`` (label ids; None = no
    edges), ``cls`` the per-row class (0 = unlabeled). Arcs naming an id absent
    from the table are dropped (``magic_fill.index_arcs``), as are self-loops.

    Returns ``{"a", "b"}`` (global rows), ``"saddle"`` (NaN where none),
    ``"slice"``, ``"both"`` (both ends labeled), ``"diff"`` (both labeled,
    different classes), ``"n_rows"``."""
    A, B, S, K = [], [], [], []
    base = 0
    cls_all = []
    for fids, arcs, cls, k in slices:
        fids = np.asarray(fids, np.intp)
        cls = np.asarray(cls, int)
        cls_all.append(cls)
        if arcs is not None and len(arcs.get("a", ())):
            ia, ib, keep = magic_fill.index_arcs(arcs, fids, np)
            sad = arcs.get("saddle")
            sad = (np.full(len(ia), np.nan) if sad is None
                   else np.asarray(sad, np.float64)[keep])
            ok = ia != ib
            A.append(ia[ok] + base)
            B.append(ib[ok] + base)
            S.append(sad[ok])
            K.append(np.full(int(ok.sum()), int(k), int))
        base += len(fids)
    cls_all = np.concatenate(cls_all) if cls_all else np.zeros(0, int)
    a = np.concatenate(A).astype(np.intp) if A else np.zeros(0, np.intp)
    b = np.concatenate(B).astype(np.intp) if B else np.zeros(0, np.intp)
    s = np.concatenate(S) if S else np.zeros(0, np.float64)
    k = np.concatenate(K) if K else np.zeros(0, int)
    both = (cls_all[a] > 0) & (cls_all[b] > 0) if len(a) else np.zeros(0, bool)
    diff = both & (cls_all[a] != cls_all[b]) if len(a) else np.zeros(0, bool)
    return {"a": a, "b": b, "saddle": s, "slice": k, "both": both, "diff": diff,
            "n_rows": int(base)}


# --------------------------------------------------------------------------- #
# The pair model
# --------------------------------------------------------------------------- #
@dataclass
class EdgeModel:
    model: Any                            # fitted sklearn pipeline -> P(different)
    spec: EdgeSpec
    n_in: int
    names_hash: str                       # region feature names it was built over
    net_hash: str                         # the region net's output layer
    width: int                            # embedding width used
    n_edges: int = 0                      # labeled pairs it was fit on
    n_diff: int = 0                       # ... of which different-class
    used_saddle: bool = False             # any finite saddle in training
    fit_s: float = 0.0
    report: Optional[Dict[str, Any]] = None   # last evaluate_edges() result

    def to_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "spec": self.spec.to_dict(), "n_in": int(self.n_in),
                "names_hash": self.names_hash, "net_hash": self.net_hash,
                "width": int(self.width), "n_edges": int(self.n_edges),
                "n_diff": int(self.n_diff), "used_saddle": bool(self.used_saddle),
                "fit_s": float(self.fit_s), "report": self.report}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EdgeModel":
        return cls(model=d["model"], spec=EdgeSpec.from_dict(d.get("spec")),
                   n_in=int(d["n_in"]), names_hash=str(d.get("names_hash", "")),
                   net_hash=str(d.get("net_hash", "")), width=int(d.get("width", 0)),
                   n_edges=int(d.get("n_edges", 0)), n_diff=int(d.get("n_diff", 0)),
                   used_saddle=bool(d.get("used_saddle", False)),
                   fit_s=float(d.get("fit_s", 0.0)), report=d.get("report"))

    def brief(self) -> str:
        r = (self.report or {}).get("edge", {}).get("learned")
        text = f"edges {self.n_edges:,} pairs, {self.n_diff / max(1, self.n_edges):.0%} boundaries"
        if r:
            text += f" · learned {r['diff_recall']:.0%}/{r['diff_precision']:.0%}"
            base = (self.report or {}).get("edge", {}).get("argmax differs")
            if base:
                text += f" vs argmax {base['diff_recall']:.0%}/{base['diff_precision']:.0%}"
        return text

    def describe(self, widths: Optional[Sequence[int]] = None) -> str:
        text = self.spec.describe(widths, self.n_in)
        text += (f"; fit on {self.n_edges:,} labeled pairs "
                 f"({self.n_diff / max(1, self.n_edges):.0%} boundaries, "
                 f"{'MSC saddles' if self.used_saddle else 'no saddles'})")
        r = (self.report or {}).get("edge", {}).get("learned")
        if r:
            base = (self.report or {}).get("edge", {}).get("argmax differs", {})
            text += (f". Held-out boundary recall/precision {r['diff_recall']:.0%}/"
                     f"{r['diff_precision']:.0%} (argmax {base.get('diff_recall', 0):.0%}/"
                     f"{base.get('diff_precision', 0):.0%})")
            reg = (self.report or {}).get("region", {})
            if reg.get("before") and reg.get("after"):
                text += (f"; held-out region errors {reg['before']['errors']} -> "
                         f"{reg['after']['errors']} with voting")
        return text


def _make_pair_estimator(spec: EdgeSpec):
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


def _fit_pair(est, F, y, spec: EdgeSpec):
    if spec.model == "mlp":
        from sklearn.utils.class_weight import compute_sample_weight
        try:
            return est.fit(F, y, mlpclassifier__sample_weight=compute_sample_weight("balanced", y))
        except TypeError:
            return est.fit(F, y)
    return est.fit(F, y)


def _layer(emb: List[np.ndarray], layer: int) -> np.ndarray:
    try:
        return emb[layer]
    except IndexError:
        return emb[-1]


def fit_edge_model(base_pipeline, X, cls, edges: Dict[str, np.ndarray], ext=None,
                   spec: Optional[EdgeSpec] = None, names: Sequence[str] = ()) -> EdgeModel:
    """Fit the pair model on every edge whose two ends are labeled.

    ``X`` / ``cls`` are ALL rows (unlabeled rows are class 0; they matter
    because edges reach them at prediction time), ``edges`` from
    :func:`gather_edges`, ``ext`` the per-row extremum value (``ext_filtered``)
    or None. Raises ValueError without both same- and different-class pairs."""
    spec = spec or EdgeSpec()
    t0 = time.perf_counter()
    emb = _layer(embed(base_pipeline, X), spec.layer)
    sel = np.nonzero(edges["both"])[0]
    a, b = edges["a"][sel], edges["b"][sel]
    y = edges["diff"][sel].astype(int)
    if len(sel) == 0 or len(np.unique(y)) < 2:
        raise ValueError("need labeled edges of both kinds (same-class and "
                         "different-class) - label two touching regions of "
                         "different classes")
    sad = edges["saddle"][sel]
    bar = barrier(sad, None if ext is None else np.asarray(ext)[a],
                  None if ext is None else np.asarray(ext)[b], n=len(a))
    F = pair_features(emb, a, b, bar, spec.features)
    est = _fit_pair(_make_pair_estimator(spec), F, y, spec)
    return EdgeModel(model=est, spec=spec, n_in=int(F.shape[1]),
                     names_hash=names_hash(names), net_hash=net_hash(base_pipeline),
                     width=int(emb.shape[1]), n_edges=int(len(y)), n_diff=int(y.sum()),
                     used_saddle=bool(np.isfinite(sad).any()),
                     fit_s=time.perf_counter() - t0)


def predict_pdiff(edge: EdgeModel, base_pipeline, X, a, b, saddle=None, ext=None,
                  emb: Optional[np.ndarray] = None) -> np.ndarray:
    """``float32[n]`` P(the edge crosses classes) for row pairs (a, b), clipped
    to ``[1e-4, 1 - 1e-4]``. ``emb`` may pass a precomputed embedding of ``X``
    (the layer the model uses). Raises ValueError when the base net is not
    the one the edge model was fit on."""
    if edge.net_hash and edge.net_hash != net_hash(base_pipeline):
        raise ValueError("edge model was fit on a different region net - retrain")
    a = np.asarray(a, np.intp)
    b = np.asarray(b, np.intp)
    if len(a) == 0:
        return np.zeros(0, np.float32)
    if emb is None:
        emb = _layer(embed(base_pipeline, X), edge.spec.layer)
    bar = barrier(saddle, None if ext is None else np.asarray(ext)[a],
                  None if ext is None else np.asarray(ext)[b], n=len(a))
    F = pair_features(emb, a, b, bar, edge.spec.features)
    if F.shape[1] != edge.n_in:
        raise ValueError(f"edge model expects {edge.n_in} inputs, got {F.shape[1]}")
    p = np.asarray(edge.model.predict_proba(F), np.float64)
    classes = list(np.asarray(edge.model.classes_).tolist())
    col = classes.index(1) if 1 in classes else -1
    return np.clip(p[:, col], _EPS, 1.0 - _EPS).astype(np.float32)


# --------------------------------------------------------------------------- #
# Voting
# --------------------------------------------------------------------------- #
def vote(P, classes, a, b, pdiff, lam: float = 1.0, rounds: int = 3) -> np.ndarray:
    """Refine argmax classes with the region graph.

    ``P`` is ``(n, C)`` in ``classes`` column order (rows may sum to 0: those
    regions are unscored and keep whatever the argmax gives), ``a`` / ``b``
    index rows, ``pdiff`` is P(different) per edge. Each round scores
    ``log P_i(c) + lam * sum over edges (i, j) of [log(1 - p_ij) if class_j ==
    c else log p_ij]`` with the neighbours' CURRENT classes, and stops early
    when nothing changes. ``lam <= 0`` or ``rounds <= 0`` returns the argmax."""
    P = np.asarray(P, np.float64)
    classes = np.asarray(classes)
    a = np.asarray(a, np.intp)
    b = np.asarray(b, np.intp)
    cur = classes[P.argmax(1)]
    if rounds <= 0 or lam <= 0.0 or len(a) == 0 or P.shape[1] < 2:
        return cur
    pd = np.clip(np.asarray(pdiff, np.float64), _EPS, 1.0 - _EPS)
    lpd, lps = np.log(pd), np.log1p(-pd)
    logP = np.log(np.clip(P, 1e-6, 1.0))
    for _ in range(int(rounds)):
        score = logP.copy()
        for ci, c in enumerate(classes):
            contrib_a = np.where(cur[b] == c, lps, lpd)      # what j says about i
            contrib_b = np.where(cur[a] == c, lps, lpd)      # what i says about j
            np.add.at(score[:, ci], a, lam * contrib_a)
            np.add.at(score[:, ci], b, lam * contrib_b)
        new = classes[score.argmax(1)]
        if np.array_equal(new, cur):
            break
        cur = new
    return cur


# --------------------------------------------------------------------------- #
# Evaluation (leave-slices-out)
# --------------------------------------------------------------------------- #
def _score(y, p) -> Dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, log_loss
    y = np.asarray(y, int)
    p = np.asarray(p, np.float64)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    hard = (p >= 0.5).astype(int)
    tp = int(((hard == 1) & (y == 1)).sum())
    fn = int(((hard == 0) & (y == 1)).sum())
    fp = int(((hard == 1) & (y == 0)).sum())
    return {"n": int(len(y)),
            "bacc": float(balanced_accuracy_score(y, hard)) if len(np.unique(y)) > 1 else float("nan"),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
            "logloss": float(log_loss(y, pc, labels=[0, 1])),
            "diff_recall": tp / max(1, tp + fn),
            "diff_precision": tp / max(1, tp + fp)}


def _region_score(truth, pred) -> Dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score
    truth = np.asarray(truth, int)
    pred = np.asarray(pred, int)
    return {"n": int(len(truth)), "acc": float((pred == truth).mean()) if len(truth) else float("nan"),
            "bacc": float(balanced_accuracy_score(truth, pred)) if len(np.unique(truth)) > 1 else float("nan"),
            "errors": int((pred != truth).sum())}


def evaluate_edges(make_estimator: Callable[[], Any], X, cls, grp, edges: Dict[str, np.ndarray],
                   ext=None, spec: Optional[EdgeSpec] = None, seed: int = 0,
                   fit: Optional[Callable] = None, progress_cb: Optional[Callable] = None,
                   stop_event=None, n_splits: int = 5) -> Dict[str, Any]:
    """Leave-slices-out report for the pair model and for voting.

    Per fold the region net is refit (``make_estimator()`` -> a fresh
    pipeline, fitted with ``fit(est, X, y)``, default
    ``model_search.fit_estimator``) on the training slices' labeled rows; the
    edge model is fit on the training slices' labeled edges; both are scored
    on the test slices. ``progress_cb(fold, n_folds)`` after every fold;
    ``stop_event`` ends the evaluation after the current fold."""
    from . import model_search
    spec = spec or EdgeSpec()
    fit = fit or model_search.fit_estimator
    X = np.asarray(X, np.float64)
    cls = np.asarray(cls, int)
    grp = np.asarray(grp, int)
    lab = np.nonzero(cls > 0)[0]
    cv, kind = model_search.make_cv(cls[lab], grp[lab], n_splits=n_splits, seed=seed)
    n_folds = int(cv.get_n_splits())
    pooled: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {name: [] for name in EVAL_ROWS}
    before, after, truth = [], [], []
    folds, skipped = [], []
    stopped = False
    t0 = time.perf_counter()
    for f, (tr, te) in enumerate(cv.split(X[lab], cls[lab], grp[lab])):
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break
        tr_rows, te_rows = lab[tr], lab[te]
        test_slices = np.unique(grp[te_rows])
        in_test = np.isin(edges["slice"], test_slices)
        e_tr = np.nonzero(edges["both"] & ~in_test)[0]
        e_te = np.nonzero(edges["both"] & in_test)[0]
        y_tr = edges["diff"][e_tr].astype(int)
        y_te = edges["diff"][e_te].astype(int)
        if len(np.unique(y_tr)) < 2 or len(e_te) == 0:
            skipped.append(f"fold {f}: test slices {test_slices.tolist()} - "
                           + ("no labeled edges to score" if len(e_te) == 0
                              else "training edges of one kind only"))
            if progress_cb is not None:
                progress_cb(f + 1, n_folds)
            continue
        est = fit(make_estimator(), X[tr_rows], cls[tr_rows])
        P = np.asarray(est.predict_proba(X), np.float64)
        classes = np.asarray(est.classes_)
        pred = classes[P.argmax(1)]
        sub_edges = {k: (v[e_tr] if isinstance(v, np.ndarray) and len(v) == len(edges["a"]) else v)
                     for k, v in edges.items()}
        try:
            em = fit_edge_model(est, X, cls, sub_edges, ext, spec)
        except ValueError as exc:
            skipped.append(f"fold {f}: {exc}")
            continue
        emb = _layer(embed(est, X), spec.layer)
        a_te, b_te = edges["a"][e_te], edges["b"][e_te]
        pooled["argmax differs"].append((y_te, (pred[a_te] != pred[b_te]).astype(np.float64)))
        pooled["1 - sum P_a P_b"].append((y_te, 1.0 - (P[a_te] * P[b_te]).sum(1)))
        p_te = predict_pdiff(em, est, X, a_te, b_te, edges["saddle"][e_te], ext, emb=emb)
        pooled["learned"].append((y_te, p_te))
        # Voting over EVERY edge of the test slices (labeled or not).
        e_all = np.nonzero(in_test)[0]
        a_all, b_all = edges["a"][e_all], edges["b"][e_all]
        p_all = predict_pdiff(em, est, X, a_all, b_all, edges["saddle"][e_all], ext, emb=emb)
        refined = vote(P, classes, a_all, b_all, p_all, spec.lam, spec.rounds)
        before.append(pred[te_rows]); after.append(refined[te_rows]); truth.append(cls[te_rows])
        rb = _region_score(cls[te_rows], pred[te_rows])
        ra = _region_score(cls[te_rows], refined[te_rows])
        folds.append(f"fold {f}: test slices {test_slices.tolist()}, {len(te_rows)} labeled "
                     f"regions, {len(e_te)} labeled edges ({int(y_te.sum())} different); "
                     f"region errors {rb['errors']} -> {ra['errors']}")
        if progress_cb is not None:
            progress_cb(f + 1, n_folds)
    report: Dict[str, Any] = {"cv_kind": kind, "n_folds": n_folds, "folds": folds,
                              "skipped": skipped, "stopped": stopped,
                              "elapsed_s": time.perf_counter() - t0, "edge": {},
                              "region": {}, "spec": spec.to_dict()}
    for name, pairs in pooled.items():
        if pairs:
            y = np.concatenate([p[0] for p in pairs])
            p = np.concatenate([p[1] for p in pairs])
            report["edge"][name] = _score(y, p)
    if truth:
        t = np.concatenate(truth)
        report["region"] = {"before": _region_score(t, np.concatenate(before)),
                            "after": _region_score(t, np.concatenate(after))}
    return report


def report_lines(report: Dict[str, Any]) -> List[str]:
    """The evaluation as a fixed-width text table (log / status)."""
    lines = [f"{'edge model':22s} {'n':>6} {'bal.acc':>8} {'AUC':>6} {'logloss':>8} "
             f"{'diff recall':>11} {'diff prec':>9}"]
    for name in EVAL_ROWS:
        s = report.get("edge", {}).get(name)
        if s:
            lines.append(f"{name:22s} {s['n']:>6} {s['bacc']:>8.3f} {s['auc']:>6.3f} "
                         f"{s['logloss']:>8.3f} {s['diff_recall']:>11.1%} {s['diff_precision']:>9.1%}")
    reg = report.get("region", {})
    for key, label in (("before", "region net alone"), ("after", "+ neighbour voting")):
        r = reg.get(key)
        if r:
            lines.append(f"{label:22s} acc {r['acc']:.3f}  bal.acc {r['bacc']:.3f}  "
                         f"held-out errors {r['errors']} / {r['n']}")
    for s in report.get("skipped", []):
        lines.append("skipped " + s)
    return lines
