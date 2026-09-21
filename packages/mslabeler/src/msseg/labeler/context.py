"""Neighbourhood context for the region classifier: extra columns per region
computed from the living-region adjacency graph.

The classifier sees one statistics row per region. A region's own pixels do
not always say enough -- a shallow dip inside metal and a real void can share
a ``mean_base`` -- but its neighbourhood does, and this module turns that
neighbourhood into ORDINARY columns appended to the table, so Train, Optimize
(the per-group feature mask), the compatibility gate and the pickle path all
keep working unchanged. A ``ContextSpec`` says which columns:

* ring reductions over the arc neighbours of each region (``ring_mean``,
  ``ring_min``, ``ring_max``, ``ring_std``) and the contrast against the
  ring (``ring_contrast`` = own value minus the ring mean);
* a second hop (``hop2_mean``: the weighted mean applied twice);
* per-item descriptors (``slice_mean``, ``slice_contrast``);

each over a chosen set of SOURCE columns (every non-positional column, only
the ``ext_*`` columns -- the seeding extremum is the proxy for what a region
looks like away from its boundary -- or only the ``mean_*`` columns), and each
under a chosen neighbour WEIGHTING: ``uniform`` (every arc neighbour counts
once; needs nothing beyond the arcs), ``area`` (by the neighbour's pixel
count) or ``contact`` (by the shared boundary length, derived from the label
raster on demand and cached on the arcs dict under the OPTIONAL key
``length``). Nothing here reads a saddle value: the saddle lives on the
topology field and only reports what the reduction to a scalar already made
large, so edges carry adjacency and geometry only.

Column names are ``<kind>__<source>`` for uniform weights and
``<kind>[<weight>]__<source>`` otherwise (``ring_mean[contact]__ext_base``).
The kind prefixes never start with the conventions' ``mean_`` / ``std_`` /
``hist`` spellings, so ``FieldConventions.channel_names`` and the histogram
regex do not see phantom channels; ``schema_entries`` files each kind (and
weighting) as its own channel so the Optimize mask toggles a whole rung.

Headless numpy: ``np`` is passed in, nothing imports sklearn or Tk. The
augmented table is a NEW ``FeatureTable``; ``rec["stats"]`` is never touched
(the magic fill's ``cosine`` metric reads the raw table).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import magic_fill
from .fields import DEFAULT, FieldConventions
from .labeling import MAX_CLASSES
from .table import FeatureTable

RING_KINDS = ("ring_mean", "ring_min", "ring_max", "ring_std", "ring_contrast")
HOP_KINDS = ("hop2_mean",)
SLICE_KINDS = ("slice_mean", "slice_contrast")
KINDS = RING_KINDS + HOP_KINDS + SLICE_KINDS
GRAPH_KINDS = RING_KINDS + HOP_KINDS
# Labels as context: the ring's annotated classes as fractions (see LabelSpec).
LABEL_KIND = "nbr_class"
WEIGHTS = ("uniform", "area", "contact")
SLICE_WEIGHTS = ("uniform", "area")
SOURCES = ("all", "ext", "mean")
SEP = "__"
# Columns are reduced in chunks so the per-edge gather (2 * n_arcs rows) never
# holds the whole source block at once.
COLUMN_CHUNK = 16
# The kind prefixes, for `is_context_column` (a name like `ring_mean__x`).
_ALL_KINDS = KINDS + (LABEL_KIND,)
_KIND_PREFIXES = tuple(f"{k}{SEP}" for k in _ALL_KINDS) + tuple(f"{k}[" for k in _ALL_KINDS)


# --------------------------------------------------------------------------- #
# The spec
# --------------------------------------------------------------------------- #
LATENT_WEIGHTS = WEIGHTS + ("latent",)
LATENT_LAYERS = {"last": -1, "previous": -2}


@dataclass
class LatentSpec:
    """The latent-ring head: a second net fit on the row PLUS columns built
    from the base net's hidden-layer embedding of the region's ring -- the
    weighted mean of the neighbours' embeddings and the H0 shape of the ring
    in latent space (see ``ring_h0``). Unlike the raw rungs these columns are
    a function of the fitted model, not of the profile, so they never enter
    the feature fingerprint; the head rides the pickle as ``stack["latent"]``."""
    layer: int = -1                       # -1 = last hidden layer, -2 = previous
    weight: str = "uniform"               # uniform | area | contact | latent (softmax)
    h0: bool = True                       # add the H0 shape columns
    z: float = 1.0                        # components: merges above mean + z * std
    max_degree: int = 24                  # ring cap for the batched Prim

    def to_dict(self) -> Dict[str, Any]:
        return {"layer": int(self.layer), "weight": str(self.weight), "h0": bool(self.h0),
                "z": float(self.z), "max_degree": int(self.max_degree)}

    @classmethod
    def from_dict(cls, d: Any) -> "LatentSpec":
        d = dict(d or {}) if isinstance(d, dict) else {}
        weight = str(d.get("weight") or "uniform")
        if weight not in LATENT_WEIGHTS:
            weight = "uniform"
        try:
            layer = int(d.get("layer", -1))
        except (TypeError, ValueError):
            layer = -1
        try:
            z = float(d.get("z", 1.0))
        except (TypeError, ValueError):
            z = 1.0
        try:
            md = max(1, int(d.get("max_degree", 24)))
        except (TypeError, ValueError):
            md = 24
        return cls(layer=layer, weight=weight, h0=bool(d.get("h0", True)), z=z, max_degree=md)

    def brief(self) -> str:
        text = "latent(mean"
        if self.weight != "uniform":
            text += f"[{self.weight}]"
        if self.h0:
            text += ",h0"
        return text + ")"


@dataclass
class LabelSpec:
    """Labels as context: for every region, the weighted fraction of its ring
    that is annotated with each class (``nbr_class__c<k>``) and with any
    class (``nbr_class__any``). The region's own label never enters (the
    ring excludes it); at training every labeled neighbour is hidden with
    probability `dropout` (a seeded draw per slice) so the net learns to
    work with a partly labeled ring, which is what it meets at prediction
    time -- where it sees every annotation on the slice. That makes the
    interactive loop transductive: annotations change predictions, so the
    cache empties when the store changes. A held-out score with the slice's
    own labels visible is optimistic; the ablation harness reports both."""
    dropout: float = 0.3
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"dropout": float(self.dropout), "seed": int(self.seed)}

    @classmethod
    def from_dict(cls, d: Any) -> "LabelSpec":
        d = dict(d or {}) if isinstance(d, dict) else {}
        try:
            p = float(d.get("dropout", 0.3))
        except (TypeError, ValueError):
            p = 0.3
        p = min(max(p, 0.0), 0.95)
        try:
            seed = int(d.get("seed", 0))
        except (TypeError, ValueError):
            seed = 0
        return cls(dropout=p, seed=seed)

    def brief(self) -> str:
        return f"labels(p={self.dropout:g})"


@dataclass
class ContextSpec:
    """Which context columns to build, as plain data (rides the session view,
    the pickle's ``stack["context"]`` and the model record)."""
    kinds: Tuple[str, ...] = ()
    weights: Tuple[str, ...] = ("uniform",)
    source: str = "all"
    latent: Optional[LatentSpec] = None
    labels: Optional[LabelSpec] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"kinds": list(self.kinds), "weights": list(self.weights),
             "source": str(self.source)}
        if self.latent is not None:
            d["latent"] = self.latent.to_dict()
        if self.labels is not None:
            d["labels"] = self.labels.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Any) -> "ContextSpec":
        """Tolerant: unknown kinds / weights are dropped, duplicates collapse,
        an empty weight list means uniform, an unknown source means all."""
        d = dict(d or {}) if isinstance(d, dict) else {}
        kinds = _dedupe(str(k) for k in (d.get("kinds") or ()) if str(k) in KINDS)
        weights = _dedupe(str(w) for w in (d.get("weights") or ()) if str(w) in WEIGHTS)
        source = str(d.get("source") or "all")
        if source not in SOURCES:
            source = "all"
        latent = d.get("latent")
        latent = LatentSpec.from_dict(latent) if isinstance(latent, dict) else None
        labels = d.get("labels")
        labels = LabelSpec.from_dict(labels) if isinstance(labels, dict) else None
        return cls(kinds=tuple(kinds), weights=tuple(weights) or ("uniform",),
                   source=source, latent=latent, labels=labels)

    def empty(self) -> bool:
        return not self.kinds and self.latent is None and self.labels is None

    def needs_graph(self) -> bool:
        return any(k in GRAPH_KINDS for k in self.kinds) or self.labels is not None

    def key(self) -> str:
        """A short stable digest of the spec, for caches keyed on it."""
        text = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    def brief(self) -> str:
        """`ctx: ring(mean,std,contrast) +hop2 [contact] on ext +latent(mean,h0)`;
        empty -> ''."""
        if self.empty():
            return ""
        text = "ctx:"
        if self.kinds:
            groups: Dict[str, List[str]] = {}
            for k in self.kinds:
                head, _, tail = k.partition("_")
                groups.setdefault(head, []).append(tail)
            parts = [f"{head}({','.join(t)})" for head, t in groups.items()]
            text += " " + " +".join(parts)
            if tuple(self.weights) != ("uniform",):
                text += " [" + ",".join(self.weights) + "]"
            if self.source != "all":
                text += f" on {self.source}"
        if self.latent is not None:
            text += " +" + self.latent.brief()
        if self.labels is not None:
            text += " +" + self.labels.brief()
        return text

    def describe(self, base_names: Optional[Sequence[str]] = None,
                 conv: FieldConventions = DEFAULT) -> str:
        if self.empty():
            return "no context features"
        text = self.brief()
        if base_names is not None and (self.kinds or self.labels is not None):
            n = len(column_names(self, base_names, conv))
            text += f"; {n} column(s)"
        return text


def _dedupe(items) -> List[str]:
    out: List[str] = []
    for it in items:
        if it not in out:
            out.append(it)
    return out


# --------------------------------------------------------------------------- #
# Column naming
# --------------------------------------------------------------------------- #
def column_name(kind: str, weight: str, source: str) -> str:
    return f"{kind}{SEP}{source}" if weight == "uniform" else f"{kind}[{weight}]{SEP}{source}"


def is_context_column(name: str) -> bool:
    return SEP in name and name.startswith(_KIND_PREFIXES)


def split_column(name: str) -> Optional[Tuple[str, str, str]]:
    """``(kind, weight, source)`` of a context column name, or None."""
    if not is_context_column(name):
        return None
    head, _, source = name.partition(SEP)
    if head.endswith("]") and "[" in head:
        kind, _, weight = head[:-1].partition("[")
    else:
        kind, weight = head, "uniform"
    return kind, weight, source


def weights_for(kind: str, spec: ContextSpec) -> Tuple[str, ...]:
    """The weightings a kind is built under: the spec's for graph kinds;
    only uniform / area make sense for the per-item descriptors."""
    if kind in SLICE_KINDS:
        return tuple(w for w in spec.weights if w in SLICE_WEIGHTS) or ("uniform",)
    return tuple(spec.weights) or ("uniform",)


def source_columns(spec: ContextSpec, names: Sequence[str],
                   conv: FieldConventions = DEFAULT) -> List[str]:
    """The base columns a spec reduces over: `names` minus the positional /
    id columns and minus any context column already present (so augmenting
    twice cannot nest), then narrowed by the spec's source."""
    base = [n for n in names
            if n not in conv.positional and n != conv.id_field and not is_context_column(n)]
    if spec.source == "ext":
        return [n for n in base if n.startswith("ext_")]
    if spec.source == "mean":
        return [n for n in base if n.startswith(conv.mean_prefix)]
    return base


def label_sources(n_classes: int = MAX_CLASSES) -> List[str]:
    """The label block's pseudo-source columns: one per class id 1..K-1, then
    ``any`` (the labeled fraction of the ring)."""
    return [f"c{k}" for k in range(1, int(n_classes))] + ["any"]


def column_names(spec: ContextSpec, base_names: Sequence[str],
                 conv: FieldConventions = DEFAULT) -> List[str]:
    """Every context column the spec adds over `base_names`, in the order
    `augment` appends them: kind, then weighting, then source column; the
    label block (``nbr_class``) last."""
    if spec is None or (not spec.kinds and spec.labels is None):
        return []
    cols = source_columns(spec, base_names, conv)
    out = [column_name(kind, w, c)
           for kind in spec.kinds for w in weights_for(kind, spec) for c in cols]
    if spec.labels is not None:
        out += [column_name(LABEL_KIND, w, c)
                for w in weights_for(LABEL_KIND, spec) for c in label_sources()]
    return out


def schema_entries(spec: ContextSpec, base_names: Sequence[str],
                   conv: FieldConventions = DEFAULT) -> List[Dict[str, str]]:
    """``{name, channel, reduction}`` per context column, the shape of
    ``config_io.feature_schema()``: the channel is the kind (and weighting),
    so ``model_search.feature_groups`` masks a whole rung at once."""
    out = []
    for n in column_names(spec, base_names, conv):
        kind, w, src = split_column(n)
        out.append({"name": n, "channel": kind if w == "uniform" else f"{kind}[{w}]",
                    "reduction": src})
    return out


# --------------------------------------------------------------------------- #
# Contact lengths (shared boundary, from the label raster)
# --------------------------------------------------------------------------- #
def contact_lengths(labels, a, b, np):
    """``int64[n_arcs]``: the number of 4-neighbour pixel pairs on the boundary
    between the two regions of each arc -- 0 where an arc joins regions that
    never touch pixel-wise (an MSC arc through a trimmed pixel) or names an
    id absent from the raster. Symmetric in (a, b). One O(pixels) pass."""
    a = np.asarray(a, np.int64).ravel()
    b = np.asarray(b, np.int64).ravel()
    out = np.zeros(len(a), np.int64)
    lab = np.asarray(labels)
    if lab.size == 0 or len(a) == 0:
        return out
    K = int(lab.max()) + 1
    if K <= 0:
        return out
    p = np.concatenate([lab[:, :-1].ravel(), lab[:-1, :].ravel()]).astype(np.int64)
    q = np.concatenate([lab[:, 1:].ravel(), lab[1:, :].ravel()]).astype(np.int64)
    keep = (p != q) & (p >= 0) & (q >= 0)
    p, q = p[keep], q[keep]
    keys, counts = np.unique(np.minimum(p, q) * K + np.maximum(p, q), return_counts=True)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    valid = (lo >= 0) & (hi < K) & (lo != hi)
    if not valid.any() or len(keys) == 0:
        return out
    want = lo[valid] * K + hi[valid]
    pos = np.searchsorted(keys, want)
    pos_c = np.minimum(pos, len(keys) - 1)
    hit = keys[pos_c] == want
    got = np.zeros(len(want), np.int64)
    got[hit] = counts[pos_c[hit]]
    out[valid] = got
    return out


def ensure_contact(arcs, labels, np):
    """The arcs' contact lengths, computing and caching them on the dict
    under ``length`` (an optional key: nothing else requires it). Pass the
    dict the region provider hands out (``regions.arcs(key, np)``)."""
    length = arcs.get("length") if arcs is not None else None
    if length is None:
        length = contact_lengths(labels, arcs["a"], arcs["b"], np)
        arcs["length"] = length
    return np.asarray(length)


# --------------------------------------------------------------------------- #
# The directed neighbour graph and its reductions
# --------------------------------------------------------------------------- #
class Graph:
    """The undirected region graph as a sorted directed edge list: for every
    kept arc (a, b) the edges a->b and b->a, sorted by source, plus the row
    pointers. `length` (if given) is carried per directed edge."""
    __slots__ = ("n", "src", "dst", "indptr", "length")

    def __init__(self, n, src, dst, indptr, length=None):
        self.n, self.src, self.dst, self.indptr, self.length = n, src, dst, indptr, length

    @property
    def n_edges(self) -> int:
        return int(len(self.src))


def directed_graph(ia, ib, n, np, length=None) -> Graph:
    """Build the graph over ROW indices: self loops dropped, a pair listed
    twice (or in both orders) kept once, with the first listing's length."""
    ia = np.asarray(ia, np.intp).ravel()
    ib = np.asarray(ib, np.intp).ravel()
    n = int(n)
    if length is not None:
        length = np.asarray(length, np.float64).ravel()
    ok = (ia != ib) & (ia >= 0) & (ib >= 0) & (ia < n) & (ib < n)
    ia, ib = ia[ok], ib[ok]
    if length is not None:
        length = length[ok]
    lo = np.minimum(ia, ib)
    hi = np.maximum(ia, ib)
    _, first = np.unique(lo.astype(np.int64) * max(n, 1) + hi, return_index=True)
    lo, hi = lo[first], hi[first]
    if length is not None:
        length = length[first]
    src = np.concatenate([lo, hi])
    dst = np.concatenate([hi, lo])
    ln = None if length is None else np.concatenate([length, length])
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    if ln is not None:
        ln = ln[order]
    indptr = np.zeros(n + 1, np.intp)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return Graph(n, src, dst, indptr, ln)


def neighbour_weights(kind: str, g: Graph, np, area=None):
    """``float64[n_edges]`` the weight of each directed edge i->j when row i
    aggregates neighbour j: 1 (uniform), ``area[j]`` (area) or the shared
    boundary length (contact). Raises ValueError when the data the kind
    needs is missing."""
    m = g.n_edges
    if kind == "uniform":
        return np.ones(m, np.float64)
    if kind == "area":
        if area is None:
            raise ValueError("area weighting needs the table's area column")
        w = np.asarray(area, np.float64)[g.dst]
        return np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if kind == "contact":
        if g.length is None:
            raise ValueError("contact weighting needs the arcs' contact lengths "
                             "(ensure_contact over the label raster)")
        return np.where(np.isfinite(g.length) & (g.length > 0), g.length, 0.0)
    raise ValueError(f"unknown neighbour weighting {kind!r}")


def ring_reduce(X, g: Graph, w, reductions: Sequence[str], np) -> Dict[str, Any]:
    """Reduce every row's neighbours: ``{"mean" | "min" | "max" | "std":
    (n, f)}`` for the reductions asked for. A row with no neighbours (or none
    with positive weight) keeps its OWN value for mean / min / max and 0 for
    std, so a later contrast reads 0 rather than NaN. Sort-once segment
    reductions (``ufunc.reduceat``), columns in chunks."""
    X = np.asarray(X, np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n, f = X.shape
    want = set(reductions)
    need_mean = bool(want & {"mean", "std"})
    counts = np.diff(g.indptr)
    has = counts > 0
    starts = g.indptr[:-1][has]
    W = np.bincount(g.src, weights=w, minlength=n) if g.n_edges else np.zeros(n)
    pos = has & (W > 0)
    out: Dict[str, Any] = {}
    mean = X.copy() if need_mean else None
    std = np.zeros((n, f)) if "std" in want else None
    mn = X.copy() if "min" in want else None
    mx = X.copy() if "max" in want else None
    if g.n_edges and (has.any()):
        inv = np.zeros(n)
        inv[pos] = 1.0 / W[pos]
        for c0 in range(0, f, COLUMN_CHUNK):
            c1 = min(f, c0 + COLUMN_CHUNK)
            G = X[g.dst, c0:c1]
            if need_mean:
                S = np.add.reduceat(G * w[:, None], starts, axis=0)
                m = np.empty((n, c1 - c0))
                m[has] = S
                m *= inv[:, None]
                mean[pos, c0:c1] = m[pos]
                if std is not None:
                    S2 = np.add.reduceat(G * G * w[:, None], starts, axis=0)
                    e2 = np.empty((n, c1 - c0))
                    e2[has] = S2
                    e2 *= inv[:, None]
                    var = e2[pos] - mean[pos, c0:c1] ** 2
                    std[pos, c0:c1] = np.sqrt(np.maximum(var, 0.0))
            if mn is not None:
                mn[has, c0:c1] = np.minimum.reduceat(G, starts, axis=0)
            if mx is not None:
                mx[has, c0:c1] = np.maximum.reduceat(G, starts, axis=0)
    if "mean" in want:
        out["mean"] = mean
    if "std" in want:
        out["std"] = std
    if "min" in want:
        out["min"] = mn
    if "max" in want:
        out["max"] = mx
    return out


def hop2_mean(X, g: Graph, w, np):
    """The weighted ring mean applied twice: what a region's neighbours'
    neighbourhoods look like (a graph convolution without the nonlinearity)."""
    m1 = ring_reduce(X, g, w, ("mean",), np)["mean"]
    return ring_reduce(m1, g, w, ("mean",), np)["mean"]


def visible_classes(row_classes, spec: LabelSpec, np, rng=None):
    """The per-row classes a label block may see: `row_classes` (0 =
    unlabeled) with, when `rng` is given (training), every labeled row hidden
    with probability `spec.dropout`. Without an rng (prediction) every
    annotation shows."""
    cls = np.asarray(row_classes, int).copy()
    if rng is not None and spec.dropout > 0 and (cls > 0).any():
        hide = rng.random(len(cls)) < float(spec.dropout)
        cls[hide] = 0
    return cls


def label_columns(visible, g: Graph, w, np, n_classes: int = MAX_CLASSES):
    """``(n, K)`` the ring's weighted fraction annotated with each class
    1..K-1, then with any class. A region's own label is not in its ring;
    a region without neighbours (or without positive weight) reads 0."""
    visible = np.asarray(visible, int)
    n = len(visible)
    ind = np.zeros((n, int(n_classes)), np.float64)
    for k in range(1, int(n_classes)):
        ind[:, k - 1] = visible == k
    ind[:, -1] = visible > 0
    r = ring_reduce(ind, g, w, ("mean",), np)["mean"]
    W = np.bincount(g.src, weights=w, minlength=n) if g.n_edges else np.zeros(n)
    r[W <= 0] = 0.0                      # never the row's own indicator
    return r


def slice_mean(X, np, area=None):
    """``(f,)`` the per-item mean of every column, area-weighted when `area`
    is given (and carries any positive weight)."""
    X = np.asarray(X, np.float64)
    if X.shape[0] == 0:
        return np.zeros(X.shape[1])
    if area is not None:
        a = np.asarray(area, np.float64)
        a = np.where(np.isfinite(a) & (a > 0), a, 0.0)
        if a.sum() > 0:
            return (X * a[:, None]).sum(0) / a.sum()
    return X.mean(0)


# --------------------------------------------------------------------------- #
# augment: the table plus its context columns
# --------------------------------------------------------------------------- #
def augment(table, arcs, spec: ContextSpec, np, conv: FieldConventions = DEFAULT,
            labels=None, extra=None):
    """`table` with the spec's context columns appended: a NEW FeatureTable,
    or `table` itself when the spec adds nothing (so identity is testable
    and an empty spec costs nothing).

    `arcs` is the item's region graph over label ids (``{"a", "b", ...}``;
    an optional ``length`` is used by contact weighting); None derives
    4-neighbour adjacency from `labels`, which is also what contact lengths
    are measured on. The label block reads ``extra["row_classes"]`` (the
    per-row class, 0 = unlabeled) and, at training, ``extra["rng"]`` for the
    dropout draw. Raises ValueError when a graph kind is asked for and no
    graph is available, when a weighting lacks its data, or when the label
    block has no row classes."""
    if spec is None or (not spec.kinds and spec.labels is None):
        return table
    names = list(table.names)
    new_names = column_names(spec, names, conv)
    if not new_names or all(n in table._col for n in new_names):
        return table
    X = np.asarray(table.values, np.float64)
    n = X.shape[0]
    cols = source_columns(spec, names, conv)
    idx = [table._col[c] for c in cols]
    Xs = X[:, idx].copy()
    Xs[~np.isfinite(Xs)] = 0.0
    area = None if conv.area_field is None else table.column(conv.area_field)
    extra = extra or {}

    graph_kinds = [k for k in spec.kinds if k in GRAPH_KINDS]
    if spec.labels is not None:
        graph_kinds.append(LABEL_KIND)
    g = None
    if graph_kinds:
        if arcs is None:
            if labels is None:
                raise ValueError("context: ring features need the region graph "
                                 "(arcs) or the label raster to derive it from")
            arcs = magic_fill.arcs_from_labels(labels, np)
        ids = table.column(conv.id_field)
        if ids is None:
            raise ValueError(f"context: the table has no {conv.id_field!r} column")
        ia, ib, keep = magic_fill.index_arcs(arcs, ids, np)
        length = None
        if any("contact" in weights_for(k, spec) for k in graph_kinds):
            length = arcs.get("length")
            if length is None:
                if labels is None:
                    raise ValueError("context: contact weighting needs the label "
                                     "raster to measure shared boundaries")
                length = ensure_contact(arcs, labels, np)
            length = np.asarray(length, np.float64)[keep]
        g = directed_graph(ia, ib, n, np, length)

    # One reduction pass per weighting, then the blocks in spec order.
    ring_cache: Dict[str, Dict[str, Any]] = {}
    hop_cache: Dict[str, Any] = {}
    weight_cache: Dict[str, Any] = {}

    def weights(kind):
        if kind not in weight_cache:
            weight_cache[kind] = neighbour_weights(kind, g, np, area)
        return weight_cache[kind]

    def ring(kind_w):
        if kind_w not in ring_cache:
            need = set()
            for k in spec.kinds:
                if k in RING_KINDS and kind_w in weights_for(k, spec):
                    need.add({"ring_mean": "mean", "ring_contrast": "mean",
                              "ring_min": "min", "ring_max": "max",
                              "ring_std": "std"}[k])
            ring_cache[kind_w] = ring_reduce(Xs, g, weights(kind_w), sorted(need), np)
        return ring_cache[kind_w]

    blocks = []
    for kind in spec.kinds:
        for wk in weights_for(kind, spec):
            if kind == "slice_mean" or kind == "slice_contrast":
                mu = slice_mean(Xs, np, area if wk == "area" else None)
                blocks.append(np.broadcast_to(mu, Xs.shape) if kind == "slice_mean"
                              else Xs - mu)
            elif kind == "hop2_mean":
                if wk not in hop_cache:
                    hop_cache[wk] = hop2_mean(Xs, g, weights(wk), np)
                blocks.append(hop_cache[wk])
            elif kind == "ring_contrast":
                blocks.append(Xs - ring(wk)["mean"])
            else:
                blocks.append(ring(wk)[kind[len("ring_"):]])
    if spec.labels is not None:
        rc = extra.get("row_classes")
        if rc is None:
            raise ValueError("context: the labels block needs the rows' classes "
                             "(extra['row_classes'])")
        if len(rc) != n:
            raise ValueError("context: row_classes does not match the table")
        vis = visible_classes(rc, spec.labels, np, extra.get("rng"))
        for wk in weights_for(LABEL_KIND, spec):
            blocks.append(label_columns(vis, g, weights(wk), np))
    new = np.concatenate([np.asarray(b, np.float64) for b in blocks], axis=1)
    return FeatureTable(names + new_names, np.concatenate([X, new], axis=1))


# --------------------------------------------------------------------------- #
# The latent ring: columns from the base net's embedding of the neighbourhood
# --------------------------------------------------------------------------- #
H0_NAMES = ("h0_largest", "h0_ratio", "h0_attach", "h0_components")


def latent_names(spec: LatentSpec, width: int) -> List[str]:
    """The head's extra columns, in order: ``lat_mean__h<j>`` (or
    ``lat_mean[<weight>]__h<j>``) per embedding dimension, then the H0 four."""
    kind = "lat_mean" if spec.weight == "uniform" else f"lat_mean[{spec.weight}]"
    out = [f"{kind}{SEP}h{j}" for j in range(int(width))]
    if spec.h0:
        out += list(H0_NAMES)
    return out


def _edge_distances(emb, g: Graph, np):
    """Euclidean latent distance per directed edge."""
    if g.n_edges == 0:
        return np.zeros(0)
    d = emb[g.src] - emb[g.dst]
    return np.sqrt((d * d).sum(1))


def _per_slice(values, slice_of_edge, np, fn):
    """``fn`` (a numpy reduction) of `values` within each slice id, mapped
    back per edge; an empty input reads 0."""
    if len(values) == 0:
        return np.zeros(0)
    if slice_of_edge is None:
        return np.full(len(values), float(fn(values)))
    out = np.zeros(len(values))
    for s in np.unique(slice_of_edge):
        m = slice_of_edge == s
        out[m] = fn(values[m])
    return out


def latent_weights(spec: LatentSpec, emb, g: Graph, np, area=None, row_slice=None):
    """The directed-edge weights the latent mean uses: the raw weightings, or
    ``latent`` = ``exp(-d_ij / tau)`` with tau the median latent arc distance
    of the edge's slice (a per-slice scale, so the softmax needs no units)."""
    if spec.weight != "latent":
        return neighbour_weights(spec.weight, g, np, area)
    d = _edge_distances(emb, g, np)
    sl = None if row_slice is None else np.asarray(row_slice)[g.src]
    tau = _per_slice(d, sl, np, np.median)
    tau = np.where(tau > 0, tau, 1.0)
    return np.exp(-d / tau)


def ring_h0(emb, g: Graph, np, z: float = 1.0, max_degree: int = 24, row_slice=None):
    """``(n, 4)``: the H0 (single-linkage) shape of each region's ring in latent
    space -- ``largest`` merge, ``ratio`` largest / second largest, ``attach``
    (the region's own death: its distance to its nearest neighbour) and
    ``components`` (1 + merges above mean + z * std of the slice's arc
    distances). Rows are bucketed by degree and every bucket runs one
    vectorised Prim over its ``(m, d+1, d+1)`` distance matrices, so the cost
    is sum(m_d * d^2) rather than one Python call per region. A ring wider
    than `max_degree` keeps its nearest neighbours. A region without
    neighbours reads zeros (components 1, ratio 1)."""
    emb = np.asarray(emb, np.float64)
    n = g.n
    out = np.zeros((n, 4))
    out[:, 1] = 1.0
    out[:, 3] = 1.0
    if g.n_edges == 0 or n == 0:
        return out
    deg = np.diff(g.indptr)
    d_edge = _edge_distances(emb, g, np)
    sl = None if row_slice is None else np.asarray(row_slice)[g.src]
    mu = _per_slice(d_edge, sl, np, np.mean)
    sd = _per_slice(d_edge, sl, np, np.std)
    thresh_edge = mu + float(z) * sd                      # per directed edge (its slice's)
    thresh_row = np.zeros(n)
    thresh_row[g.src] = thresh_edge                       # any edge of the row: same slice
    eps = 1e-12
    for d in np.unique(deg[deg > 0]):
        rows = np.nonzero(deg == d)[0]
        m = len(rows)
        dd = int(min(d, max_degree))
        idx = g.indptr[rows][:, None] + np.arange(int(d))[None, :]      # (m, d) directed edges
        if d > dd:                                                      # keep the nearest
            order = np.argsort(d_edge[idx], axis=1)[:, :dd]
            idx = np.take_along_axis(idx, order, axis=1)
        pts = np.concatenate([emb[rows][:, None, :], emb[g.dst[idx]]], axis=1)  # (m, k, w)
        k = dd + 1
        diff = pts[:, :, None, :] - pts[:, None, :, :]
        D = np.sqrt((diff * diff).sum(-1))                              # (m, k, k)
        ar = np.arange(m)
        in_tree = np.zeros((m, k), bool)
        in_tree[:, 0] = True
        best = D[:, 0, :].copy()
        best[:, 0] = np.inf
        merges = np.zeros((m, k - 1))
        for step in range(k - 1):
            j = best.argmin(1)
            merges[:, step] = best[ar, j]
            in_tree[ar, j] = True
            best = np.minimum(best, D[ar, j, :])
            best[in_tree] = np.inf
        s = -np.sort(-merges, axis=1)
        largest = s[:, 0]
        second = s[:, 1] if k - 1 >= 2 else np.zeros(m)
        out[rows, 0] = largest
        out[rows, 1] = np.where(second > eps, largest / np.maximum(second, eps), 1.0)
        out[rows, 2] = D[:, 0, 1:].min(1)
        out[rows, 3] = 1.0 + (merges > thresh_row[rows][:, None]).sum(1)
    return out


def latent_columns(emb, g: Graph, spec: LatentSpec, np, area=None, row_slice=None):
    """``(n, k)`` float64, the columns `latent_names(spec, width)` names: the
    weighted ring mean of the embedding (a region without neighbours keeps
    its own embedding) and, when asked, the H0 four."""
    emb = np.asarray(emb, np.float64)
    if emb.ndim == 1:
        emb = emb[:, None]
    w = latent_weights(spec, emb, g, np, area, row_slice)
    blocks = [ring_reduce(emb, g, w, ("mean",), np)["mean"]]
    if spec.h0:
        blocks.append(ring_h0(emb, g, np, z=spec.z, max_degree=spec.max_degree,
                              row_slice=row_slice))
    return np.concatenate(blocks, axis=1)


class LatentContextModel:
    """The head fit on ``X ++ latent_columns``, with what it needs to be
    applied again: the spec, the width of the embedding it was built over,
    the base net's `net_hash` (a head belongs to one base) and the
    `names_hash` of the base's feature names."""
    __slots__ = ("head", "spec", "names", "width", "n_in", "net_hash", "names_hash",
                 "fit_s", "n_rows")

    def __init__(self, head, spec: LatentSpec, names: Sequence[str], width: int,
                 net_hash: str, names_hash: str, fit_s: float = 0.0, n_rows: int = 0):
        self.head = head
        self.spec = spec
        self.names = list(names)              # base names + latent names, the head's input
        self.width = int(width)
        self.n_in = len(self.names)
        self.net_hash = str(net_hash)
        self.names_hash = str(names_hash)
        self.fit_s = float(fit_s)
        self.n_rows = int(n_rows)

    def to_dict(self) -> Dict[str, Any]:
        return {"head": self.head, "spec": self.spec.to_dict(), "names": list(self.names),
                "width": int(self.width), "net_hash": self.net_hash,
                "names_hash": self.names_hash, "fit_s": float(self.fit_s),
                "n_rows": int(self.n_rows)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LatentContextModel":
        return cls(head=d["head"], spec=LatentSpec.from_dict(d.get("spec")),
                   names=list(d.get("names") or []), width=int(d.get("width", 0)),
                   net_hash=str(d.get("net_hash", "")), names_hash=str(d.get("names_hash", "")),
                   fit_s=float(d.get("fit_s", 0.0)), n_rows=int(d.get("n_rows", 0)))

    def describe(self) -> str:
        k = len(latent_names(self.spec, self.width))
        return (f"latent head on {self.spec.brief()} ({self.width}-d embedding, "
                f"+{k} columns, {self.n_in} inputs; fit on {self.n_rows:,} labeled rows, "
                f"{self.fit_s * 1e3:.0f} ms)")


def embedding_of(base_pipeline, X, spec: LatentSpec, np):
    """The hidden layer `spec.layer` of the base net for rows `X`
    (``edge_model.embed``); TypeError when the base has no embedding."""
    from . import edge_model
    layers = edge_model.embed(base_pipeline, X)
    try:
        return np.asarray(layers[spec.layer], np.float64)
    except IndexError:
        return np.asarray(layers[-1], np.float64)


def design_with_latent(base_pipeline, X, g: Graph, spec: LatentSpec, np, area=None,
                       row_slice=None):
    """``(X ++ latent columns, width)`` for rows `X` over graph `g`."""
    emb = embedding_of(base_pipeline, X, spec, np)
    L = latent_columns(emb, g, spec, np, area=area, row_slice=row_slice)
    return np.concatenate([np.asarray(X, np.float64), L], axis=1), int(emb.shape[1])


def fit_latent_head(base_pipeline, X, cls, g: Graph, spec: LatentSpec, names: Sequence[str],
                    make_head, np, area=None, row_slice=None) -> LatentContextModel:
    """Fit the head on the labeled rows (``cls > 0``) of ``X ++ latent``.
    `make_head(all_names)` returns ``(estimator, fit)``: the head pipeline and
    the function that fits it (the caller fits it the way it fits the base).
    Raises ValueError without two labeled classes."""
    import time
    from . import edge_model
    t0 = time.perf_counter()
    XL, width = design_with_latent(base_pipeline, X, g, spec, np, area, row_slice)
    all_names = list(names) + latent_names(spec, width)
    cls = np.asarray(cls, int)
    lab = cls > 0
    if len(np.unique(cls[lab])) < 2:
        raise ValueError("need labels from at least 2 classes to fit the latent head")
    est, fit = make_head(all_names)
    head = fit(est, XL[lab], cls[lab])
    return LatentContextModel(head=head, spec=spec, names=all_names, width=width,
                              net_hash=edge_model.net_hash(base_pipeline),
                              names_hash=edge_model.names_hash(names),
                              fit_s=time.perf_counter() - t0, n_rows=int(lab.sum()))


def predict_latent(model: LatentContextModel, base_pipeline, X, g: Graph, np, area=None,
                   row_slice=None):
    """``(proba, classes)`` of the head for rows `X` over graph `g`. Raises
    ValueError when the base net is not the one the head was fit on."""
    from . import edge_model
    if model.net_hash and model.net_hash != edge_model.net_hash(base_pipeline):
        raise ValueError("latent head was fit on a different region net - retrain")
    XL, width = design_with_latent(base_pipeline, X, g, model.spec, np, area, row_slice)
    if XL.shape[1] != model.n_in:
        raise ValueError(f"latent head expects {model.n_in} inputs, got {XL.shape[1]}")
    proba = np.asarray(model.head.predict_proba(XL), np.float32)
    return proba, np.asarray(model.head.classes_, int)
