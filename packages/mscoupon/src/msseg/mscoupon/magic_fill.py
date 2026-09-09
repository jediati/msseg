"""Magic fill: region growing over the living-region adjacency graph.

A press on a region (the SEED) grows a set of adjacent living regions while a
dissimilarity stays under a threshold; dragging moves the threshold. All the
work that depends on the seed happens ONCE at the press, as a *join ladder*:
for every region, the threshold at which it joins the seed's flood. That is
the bottleneck (minimax) path cost from the seed over the arc graph, so a
region joins when every hop on its best path is below the threshold -- the
flood can never tunnel through a dissimilar region to reach a similar one.

The ladder is ORDERED by the flood's own discovery order (a priority flood:
lowest bottleneck first, ties by the most seed-like frontier region), and the
drag selects a prefix of that order rather than a threshold-closed set. The
difference matters exactly when the seed is an outlier -- a bright region on
a dark slice: its first neighbour's dissimilarity is then the bottleneck for
most of the slice, so hundreds of regions share one join value and a
threshold jumps from one region to half the slice. The flood order still adds
them one at a time, connected, most similar first; the HUD threshold reads
the join value of the last region admitted. A drag tick is one prefix slice
and a LUT, nothing else.

Arcs come from the pipeline (``rec["arcs"]``: MSCEER's living-region arcs
with saddle values) or from the 4-neighbour pixel fallback here. Metrics read
the per-region ``FeatureTable`` (``mean_<channel>``, ``std_<channel>``) or the
arc saddles:

  mean           |dmean| over the selected channels, each z-scored by that
                 column's spread over the slice (so channels with different
                 units add up sensibly).
  bhattacharyya  per-channel Gaussian overlap from mean and std, summed.
  cosine         1 - cos between the regions' whole statistics rows: every
                 column except ids/positions (POSITIONAL_FIELDS), each
                 z-scored over the slice. Ignores the channel list.
  proba          total variation (half the L1) between the classifier's
                 class-probability vectors, handed in as extra["proba"]
                 (indexed by label id); a region the model never scored is
                 at distance 1 from everything, so it joins last.
  barrier        |saddle - seed extremum value|: purely topological, the
                 persistence-style flood anchored at a point. Needs saddles,
                 so it is unavailable on the pixel fallback.

The flood also takes a geometric HOP GAIN g >= 1: the accumulated cost of a
path is the bottleneck inflated by g per hop, C(v_k) = max_i w_i * g^(k-i+1),
so with g > 1 a region far from the seed costs more than an equally similar
neighbour and the ladder is no longer flat across a homogeneous plateau. g = 1
is the pure bottleneck; the HUD threshold then reads in the inflated units.

  anchor mode    every candidate is compared with the SEED (no drift).
  chain mode     each arc compares its two endpoints (follows gradients).
  barrier        is always per arc.

Pure numpy (``np`` passed in, like ``labeling.py``); scipy is used for the
minimum spanning tree when present, with a heapq fallback. No Tk, no engine.
"""
from __future__ import annotations

import heapq
import math
import re

METRICS = ("mean", "bhattacharyya", "histogram", "cosine", "proba", "barrier", "learned")
MODES = ("anchor", "chain")

# Metrics that need the saddle value per arc (unavailable on pixel adjacency).
EDGE_ONLY_METRICS = ("barrier",)
# Metrics defined per ARC rather than per region (no anchor/chain choice):
# the saddle barrier, and the edge model's learned P(different) per arc.
ARC_METRICS = ("barrier", "learned")
# Metrics that read the channel list (the others use the whole row / extra).
CHANNEL_METRICS = ("mean", "bhattacharyya", "histogram")
# The per-region histogram columns (`hist00_base`, ...): a distribution, not a
# scalar, so `cosine` leaves them out and `histogram` reads them as one.
HIST_RE = re.compile(r"^hist\d+_(.+)$")
# Metrics that need a per-region array the table does not carry: metric -> the
# key build_ladder expects in `extra`.
EXTRA_METRICS = {"proba": "proba", "learned": "pdiff"}
# Statistics columns that say WHERE a region is, not what it looks like --
# never part of the cosine row (the labeler's classifier excludes the same).
POSITIONAL_FIELDS = frozenset({"feature_id", "min_x", "max_x", "min_y", "max_y",
                               "ext_x", "ext_y"})
# Distance between probability vectors: "tv" (total variation) or "hellinger".
PROBA_DIST = "tv"


# --------------------------------------------------------------------------- #
# Adjacency
# --------------------------------------------------------------------------- #
def arcs_from_labels(labels, np):
    """Fallback adjacency from the label raster: every unordered pair of
    distinct non-negative ids that touch 4-neighbourly. Returns the same dict
    shape the engine stores for MSCEER arcs, with ``saddle`` None and
    ``source`` "pixels". One O(pixels) numpy pass (~0.3 s at 3232^2)."""
    lab = np.asarray(labels)
    if lab.size == 0:
        z = np.zeros(0, np.int32)
        return {"a": z, "b": z.copy(), "saddle": None, "source": "pixels"}
    K = int(lab.max()) + 1
    p = np.concatenate([lab[:, :-1].ravel(), lab[:-1, :].ravel()])
    q = np.concatenate([lab[:, 1:].ravel(), lab[1:, :].ravel()])
    keep = (p != q) & (p >= 0) & (q >= 0)
    p, q = p[keep], q[keep]
    lo = np.minimum(p, q).astype(np.int64)
    hi = np.maximum(p, q).astype(np.int64)
    key = np.unique(lo * K + hi)
    return {"a": (key // K).astype(np.int32), "b": (key % K).astype(np.int32),
            "saddle": None, "source": "pixels"}


def channel_names(table):
    """The measurement channels the table carries a ``mean_`` column for, in
    table order (``base`` first when present)."""
    names = [n[5:] for n in table.names if n.startswith("mean_")]
    if "base" in names:
        names.remove("base")
        names.insert(0, "base")
    return names


def index_arcs(arcs, ids, np):
    """Translate arc endpoints (label ids) into ROW indices of a table whose
    ``feature_id`` column is `ids`. Returns ``(ia, ib, keep)`` where `keep` is
    the mask over the original arcs (an arc naming an id absent from the table
    is dropped) -- apply it to the saddle array too."""
    ids = np.asarray(ids, dtype=np.intp)
    a = np.asarray(arcs["a"], dtype=np.intp)
    b = np.asarray(arcs["b"], dtype=np.intp)
    K = int(max(ids.max() if len(ids) else -1,
                a.max() if len(a) else -1,
                b.max() if len(b) else -1)) + 1
    row_of = np.full(max(K, 1), -1, dtype=np.intp)
    row_of[ids] = np.arange(len(ids), dtype=np.intp)
    ia = row_of[a] if len(a) else a
    ib = row_of[b] if len(b) else b
    keep = (ia >= 0) & (ib >= 0)
    return ia[keep], ib[keep], keep


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _column(table, name, np):
    col = table.column(name)
    if col is None:
        raise ValueError(f"statistic {name!r} is not in the feature table")
    return np.asarray(col, dtype=np.float64)


def _spread(col, np):
    s = float(col.std()) if len(col) else 0.0
    return s if s > 1e-12 else 1.0


def _std_floor(sd, m, np):
    # Area-1 regions carry std 0; a Gaussian needs some width, so floor at a
    # tiny fraction of the column's own spread.
    floor = 1e-6 * _spread(m, np)
    return np.maximum(sd, floor)


def row_vectors(table, metric, channels, np, extra=None):
    """One vector per table row for `metric`, plus the distance kind that
    compares them: ``(X float64[n_rows, d], kind)``. Every region metric goes
    through here so node (seed vs all) and arc (a vs b) dissimilarities are
    the same function applied to different index pairs."""
    if metric == "mean":
        cols = [_column(table, f"mean_{c}", np) for c in channels]
        if not cols:
            raise ValueError("no channels selected")
        X = np.stack([(c - c.mean()) / _spread(c, np) for c in cols], axis=1)
        return X, "euclidean"
    if metric == "bhattacharyya":
        means = [_column(table, f"mean_{c}", np) for c in channels]
        if not means:
            raise ValueError("no channels selected")
        stds = [_std_floor(_column(table, f"std_{c}", np), m, np)
                for c, m in zip(channels, means)]
        X = np.concatenate([np.stack(means, axis=1), np.stack(stds, axis=1)], axis=1)
        return X, "bhattacharyya"
    if metric == "histogram":
        X = histogram_vectors(table, channels, np)
        return X, "hellinger"
    if metric == "cosine":
        names = [n for n in table.names
                 if n not in POSITIONAL_FIELDS and not HIST_RE.match(n)]
        if not names:
            raise ValueError("no statistics columns for cosine")
        cols = []
        for n in names:
            c = _column(table, n, np).copy()
            c[~np.isfinite(c)] = 0.0
            cols.append((c - c.mean()) / _spread(c, np))
        return np.stack(cols, axis=1), "cosine"
    if metric == "proba":
        P = None if extra is None else extra.get("proba")
        if P is None:
            raise ValueError("'proba' needs class probabilities (extra['proba']) "
                             "- Classify first")
        P = np.asarray(P, dtype=np.float64)
        if P.ndim != 2:
            raise ValueError("proba must be an (n_ids, n_classes) array")
        ids = np.asarray(_column(table, "feature_id", np), dtype=np.intp)
        X = np.zeros((len(ids), P.shape[1]), dtype=np.float64)
        ok = (ids >= 0) & (ids < len(P))
        X[ok] = P[ids[ok]]                      # ids outside P stay unscored
        return X, str((extra or {}).get("dist") or PROBA_DIST)
    if metric in EDGE_ONLY_METRICS:
        raise ValueError(f"{metric!r} is an arc metric, not a region metric")
    raise ValueError(f"unknown metric {metric!r}")


def histogram_columns(table):
    """channel -> [column names] of the table's histogram bins, in bin order."""
    out = {}
    for n in table.names:
        m = HIST_RE.match(n)
        if m:
            out.setdefault(m.group(1), []).append(n)
    return out


def histogram_vectors(table, channels, np):
    """One row per region: the concatenated bin fractions of every selected
    channel that carries a histogram, scaled by 1/k so the whole row sums to
    one and the Hellinger distance stays in [0, 1]. Channels without bins are
    skipped; none at all is an error (the spec has no histogram)."""
    by_channel = histogram_columns(table)
    picked = [c for c in channels if c in by_channel] or list(by_channel)
    if not picked:
        raise ValueError("no histogram columns: enable statistics.histogram first")
    blocks = []
    for c in picked:
        cols = [np.nan_to_num(_column(table, n, np), nan=0.0) for n in by_channel[c]]
        blocks.append(np.stack(cols, axis=1))
    X = np.concatenate(blocks, axis=1) / float(len(picked))
    return X


def pairwise(X, I, J, kind, np):
    """dist(X[I], X[J]) elementwise for index arrays I and J (either may be a
    scalar, broadcasting against the other): float64."""
    A = X[np.asarray(I, dtype=np.intp)]
    B = X[np.asarray(J, dtype=np.intp)]
    if kind == "euclidean":
        return np.sqrt(((A - B) ** 2).sum(axis=-1))
    if kind == "bhattacharyya":
        k = X.shape[1] // 2
        m1, s1 = A[..., :k], A[..., k:]
        m2, s2 = B[..., :k], B[..., k:]
        v1, v2 = s1 ** 2, s2 ** 2
        bc = (0.25 * np.log(0.25 * (v1 / v2 + v2 / v1 + 2.0))
              + 0.25 * (m1 - m2) ** 2 / (v1 + v2))
        return bc.sum(axis=-1)
    if kind == "cosine":
        na = np.sqrt((A * A).sum(axis=-1))
        nb = np.sqrt((B * B).sum(axis=-1))
        dot = (A * B).sum(axis=-1)
        denom = na * nb
        ok = (na > 1e-12) & (nb > 1e-12)
        safe = np.where(ok, denom, 1.0)              # no divide-by-zero warning
        out = np.where(ok, 1.0 - dot / safe, 1.0)    # a zero vector: distance 1
        return np.clip(np.asarray(out, dtype=np.float64), 0.0, 2.0)
    if kind in ("tv", "hellinger"):
        if kind == "tv":
            d = 0.5 * np.abs(A - B).sum(axis=-1)
        else:
            d = np.sqrt(0.5 * ((np.sqrt(np.maximum(A, 0.0))
                                - np.sqrt(np.maximum(B, 0.0))) ** 2).sum(axis=-1))
        unscored = (A.sum(axis=-1) <= 0.0) | (B.sum(axis=-1) <= 0.0)
        d = np.asarray(d, dtype=np.float64)
        if np.ndim(d):
            d[unscored] = 1.0
        elif unscored:
            d = np.float64(1.0)
        return d
    raise ValueError(f"unknown distance {kind!r}")


def node_dissimilarity(table, seed_row, metric, channels, np, extra=None):
    """d(seed, r) for every row r (anchor mode). float64[n_rows], 0 at the seed."""
    X, kind = row_vectors(table, metric, channels, np, extra)
    return pairwise(X, seed_row, np.arange(len(X)), kind, np)


def edge_dissimilarity(table, ia, ib, metric, channels, np, extra=None):
    """d(a, b) per arc (chain mode). float64[n_arcs]."""
    X, kind = row_vectors(table, metric, channels, np, extra)
    return pairwise(X, ia, ib, kind, np)


def barrier_weights(saddle, seed_ext_value, np):
    """|saddle - seed extremum value| per arc: how high the flood must rise
    from the seed's well to spill over into the neighbour."""
    return np.abs(np.asarray(saddle, dtype=np.float64) - float(seed_ext_value))


def edge_weights(table, ia, ib, seed_row, metric, mode, channels, np,
                 saddle=None, seed_ext_value=None, extra=None, pdiff=None):
    """One weight per arc, whatever the metric/mode: the bottleneck search
    below only ever sees arcs. `pdiff` is the edge model's P(different)
    per (kept) arc for the `learned` metric."""
    if metric == "learned":
        if pdiff is None:
            raise ValueError("'learned' needs per-arc p(diff) (extra['pdiff']) "
                             "- Classify with an edge model first")
        return np.asarray(pdiff, dtype=np.float64)
    if metric in EDGE_ONLY_METRICS:
        if saddle is None:
            raise ValueError(f"{metric!r} needs saddle values (pixel adjacency has none)")
        if seed_ext_value is None:
            raise ValueError(f"{metric!r} needs the seed's extremum value (ext_filtered)")
        return barrier_weights(saddle, seed_ext_value, np)
    if mode == "anchor":
        d = node_dissimilarity(table, seed_row, metric, channels, np, extra)
        return np.maximum(d[ia], d[ib])
    if mode == "chain":
        return edge_dissimilarity(table, ia, ib, metric, channels, np, extra)
    raise ValueError(f"unknown mode {mode!r}")


# --------------------------------------------------------------------------- #
# Bottleneck (minimax) path cost from the seed
# --------------------------------------------------------------------------- #
def bottleneck_join(n_nodes, ia, ib, w, seed, np, use_scipy=None):
    """float64[n_nodes]: for every node the smallest threshold t such that a
    path from `seed` exists whose every arc weight is <= t (inf = unreachable,
    0 at the seed). Arcs are undirected.

    The minimax path lives on the minimum spanning tree, so with scipy this is
    one MST plus a tree walk; without it, a heap-based minimax Dijkstra.
    This is the hop_gain = 1 reference (and test) API: build_ladder uses
    growth_order, which also yields the flood order and takes the gain (a
    path-length term the MST cannot express)."""
    n = int(n_nodes)
    join = np.full(max(n, 1), np.inf, dtype=np.float64)
    if n == 0:
        return join
    join[seed] = 0.0
    ia = np.asarray(ia, dtype=np.intp)
    ib = np.asarray(ib, dtype=np.intp)
    w = np.asarray(w, dtype=np.float64)
    if len(ia) == 0:
        return join
    if use_scipy is None or use_scipy:
        try:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
        except ImportError:
            if use_scipy:
                raise
        else:
            return _bottleneck_scipy(n, ia, ib, w, seed, join, np, coo_matrix,
                                     minimum_spanning_tree, breadth_first_order)
    return _bottleneck_heap(n, ia, ib, w, seed, join, np)


def _bottleneck_scipy(n, ia, ib, w, seed, join, np, coo_matrix, mst_fn, bfs_fn):
    # scipy drops explicit zeros from a sparse graph, so shift every weight to
    # be >= 1; the shift is monotone, which leaves the MST (and so the minimax
    # order) unchanged, and it is undone on the way out.
    # coo_matrix SUMS duplicate entries; a pair listed twice (or as both
    # (a,b) and (b,a)) must instead keep its LOWEST weight -- the bottleneck of
    # parallel arcs is the easiest of them. Pipeline arcs are unique pairs,
    # so this only ever pays off for the pixel fallback and hand-built graphs.
    lo = np.minimum(ia, ib)
    hi = np.maximum(ia, ib)
    order = np.lexsort((w, hi, lo))
    lo, hi, w = lo[order], hi[order], w[order]
    first = np.r_[True, (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])]
    ia, ib, w = lo[first], hi[first], w[first]
    wmin = float(w.min())
    shift = 1.0 - wmin
    g = coo_matrix((w + shift, (ia, ib)), shape=(n, n)).tocsr()
    mst = mst_fn(g)
    sym = (mst + mst.T).tocsr()
    order, pred = bfs_fn(sym, seed, directed=False, return_predecessors=True)
    if len(order) <= 1:
        return join
    kids = order[1:]
    par = pred[kids]
    wpar = np.asarray(sym[par, kids]).ravel() - shift
    # Parents precede children in BFS order, so one pass in that order sees
    # every parent's final value before its children read it.
    j = join
    for v, p, wv in zip(kids.tolist(), par.tolist(), wpar.tolist()):
        jp = j[p]
        j[v] = jp if jp > wv else wv
    return join


def _csr(n, ia, ib, w, np):
    """The undirected graph (both directions) as CSR Python lists, for the
    heap searches: (neighbours, weights, row pointers)."""
    src = np.concatenate([ia, ib])
    dst = np.concatenate([ib, ia])
    ww = np.concatenate([w, w])
    order = np.argsort(src, kind="stable")
    src, dst, ww = src[order], dst[order], ww[order]
    counts = np.bincount(src, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.intp)
    np.cumsum(counts, out=indptr[1:])
    return dst.tolist(), ww.tolist(), indptr.tolist()


def growth_order(n_nodes, ia, ib, w, seed, np, node_key=None, hop_gain=1.0):
    """Priority flood from the seed: ``(order, join)``.

    `join` is the bottleneck cost of bottleneck_join() inflated by `hop_gain`
    per hop -- along seed=v0..vk, ``C(vk) = max_i w_i * g**(k-i+1)`` -- so
    g = 1 is the bottleneck exactly and g > 1 makes distance from the seed
    cost something. The label-setting search only needs the extension
    ``f(c, w) = max(c, w) * g`` to be >= c and monotone in c, which holds for
    g >= 1 and w >= 0 (g < 1 raises). `order` lists the reachable nodes in
    the order the flood admits them -- non-decreasing join, ties broken by a
    secondary key: ``node_key[v]`` when given (anchor mode: the region's own
    dissimilarity to the seed), else the weight of the arc the node is entered
    through (chain / barrier). Every prefix of `order` is connected to the
    seed (each popped node was pushed by a popped neighbour), which is what
    makes a rank on it a usable drag axis when many regions tie (see the
    module docstring). Unreached nodes follow in index order with join inf."""
    n = int(n_nodes)
    g = float(hop_gain)
    if not g >= 1.0:
        raise ValueError(f"hop_gain must be >= 1, got {hop_gain!r}")
    join = np.full(max(n, 1), np.inf, dtype=np.float64)
    if n == 0:
        return np.zeros(0, dtype=np.intp), join
    ia = np.asarray(ia, dtype=np.intp)
    ib = np.asarray(ib, dtype=np.intp)
    w = np.asarray(w, dtype=np.float64)
    seed = int(seed)
    nbr, ww, ptr = _csr(n, ia, ib, w, np) if len(ia) else ([], [], [0] * (n + 1))
    nk = None if node_key is None else np.asarray(node_key, dtype=np.float64).tolist()
    b1 = [math.inf] * n            # best (primary, secondary) seen per node
    b2 = [math.inf] * n
    done = [False] * n
    b1[seed] = 0.0
    b2[seed] = 0.0
    heap = [(0.0, 0.0, seed)]
    popped = []
    while heap:
        c1, c2, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        popped.append(u)
        for k in range(ptr[u], ptr[u + 1]):
            v = nbr[k]
            if done[v]:
                continue
            wk = ww[k]
            p1 = (c1 if c1 > wk else wk) * g
            p2 = nk[v] if nk is not None else wk
            if p1 < b1[v] or (p1 == b1[v] and p2 < b2[v]):
                b1[v] = p1
                b2[v] = p2
                heapq.heappush(heap, (p1, p2, v))
    reached = np.asarray(popped, dtype=np.intp)
    join[reached] = np.asarray(b1, dtype=np.float64)[reached]
    rest = np.flatnonzero(~np.asarray(done, dtype=bool))
    return np.concatenate([reached, rest]), join


def _bottleneck_heap(n, ia, ib, w, seed, join, np):
    dst_l, ww_l, ptr = _csr(n, ia, ib, w, np)
    best = [math.inf] * n
    best[seed] = 0.0
    done = [False] * n
    heap = [(0.0, int(seed))]
    while heap:
        cost, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        for k in range(ptr[u], ptr[u + 1]):
            v = dst_l[k]
            c = cost if cost > ww_l[k] else ww_l[k]
            if c < best[v]:
                best[v] = c
                heapq.heappush(heap, (c, v))
    join[:] = best
    return join


# --------------------------------------------------------------------------- #
# The ladder: everything a drag tick needs, computed once per press
# --------------------------------------------------------------------------- #
class Ladder:
    """Per-region join thresholds for one seed, in flood order for rank lookups.

    ``ids[i]`` / ``join[i]`` follow the table's row order; ``order`` is the
    flood's discovery order (growth_order: non-decreasing join, ties most
    seed-like first, every prefix connected; unreachable rows last) and
    ``sorted_join`` is ``join[order]``. ``cum_area[k-1]`` is the pixel count
    of the first k regions on the ladder (None when the table has no
    ``area``). ``n_reach`` counts reachable regions, the seed included -- the
    drag never goes past it."""

    __slots__ = ("ids", "join", "order", "sorted_join", "cum_area", "n_reach",
                 "seed_id", "seed_row", "metric", "mode", "channels", "ia", "ib",
                 "hop_gain")

    def __init__(self, ids, join, order, sorted_join, cum_area, n_reach,
                 seed_id, seed_row, metric, mode, channels, ia=None, ib=None,
                 hop_gain=1.0):
        self.ids = ids
        self.hop_gain = float(hop_gain)
        self.ia = ia                   # the arc graph in ROW space (ring queries)
        self.ib = ib
        self.join = join
        self.order = order
        self.sorted_join = sorted_join
        self.cum_area = cum_area
        self.n_reach = int(n_reach)
        self.seed_id = int(seed_id)
        self.seed_row = int(seed_row)
        self.metric = metric
        self.mode = mode
        self.channels = tuple(channels)


def build_ladder(table, arcs, seed_id, metric, mode, channels, np,
                 hop_gain=1.0, extra=None):
    """Join ladder for `seed_id` (a label id present in `table`). `extra`
    carries what the table does not (``{"proba": (K, C)}`` for the proba
    metric); `hop_gain` is the flood's per-hop multiplier (>= 1)."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    need = EXTRA_METRICS.get(metric)
    if need is not None and (extra is None or extra.get(need) is None):
        raise ValueError(f"{metric!r} needs {need} (extra[{need!r}]) - Classify first")
    fid = table.column("feature_id")
    if fid is None:
        raise ValueError("feature table has no feature_id column")
    ids = np.asarray(fid, dtype=np.intp)
    hit = np.flatnonzero(ids == int(seed_id))
    if len(hit) == 0:
        raise ValueError(f"region {seed_id} is not in the feature table")
    seed_row = int(hit[0])
    channels = [str(c) for c in channels]

    ia, ib, keep = index_arcs(arcs, ids, np)
    saddle = arcs.get("saddle")
    if saddle is not None:
        saddle = np.asarray(saddle, dtype=np.float64)[keep]
    pdiff = None
    if metric == "learned":
        # One P(different) per arc of `arcs`, in its order (the labeler
        # computes it against the record's own arcs at Classify time).
        pd = None if extra is None else extra.get("pdiff")
        if pd is None:
            raise ValueError("'learned' needs per-arc p(diff) (extra['pdiff']) "
                             "- Classify with an edge model first")
        pd = np.asarray(pd, dtype=np.float64)
        if len(pd) != len(arcs["a"]):
            raise ValueError("extra['pdiff'] must hold one value per arc")
        pdiff = pd[keep]
    seed_ext = None
    if metric in EDGE_ONLY_METRICS:
        ext = table.column("ext_filtered")
        if ext is None:
            raise ValueError(f"{metric!r} needs the ext_filtered statistic")
        seed_ext = float(ext[seed_row])
    node_key = None
    if metric not in ARC_METRICS and mode == "anchor":
        # Anchor mode: the region's own dissimilarity is both the arc weight
        # ingredient and the tie-breaker (most seed-like first).
        node_key = node_dissimilarity(table, seed_row, metric, channels, np, extra)
        w = np.maximum(node_key[ia], node_key[ib])
    else:
        w = edge_weights(table, ia, ib, seed_row, metric, mode, channels, np,
                         saddle=saddle, seed_ext_value=seed_ext, extra=extra,
                         pdiff=pdiff)
    order, join = growth_order(len(ids), ia, ib, w, seed_row, np,
                               node_key=node_key, hop_gain=hop_gain)
    sorted_join = join[order]
    n_reach = int(np.count_nonzero(np.isfinite(join)))
    area = table.column("area")
    cum_area = (np.cumsum(np.asarray(area, dtype=np.int64)[order])
                if area is not None else None)
    return Ladder(ids, join, order, sorted_join, cum_area, n_reach,
                  seed_id, seed_row, metric, mode, channels, ia=ia, ib=ib,
                  hop_gain=hop_gain)


def regions_for_rank(ladder, k):
    """Label ids of the first k regions in flood order (the seed always): the
    set the drag shows at rank k. Connected by construction."""
    n = ladder.n_reach
    k = min(max(int(k), 1), max(n, 1))
    return ladder.ids[ladder.order[:k]]


def ring_for_rank(ladder, k, np):
    """Label ids of the regions ADJACENT to the first k in flood order and not
    among them -- the blobber's ring: what a fill of rank k is bounded by.
    Empty once the fill has taken every reachable region."""
    n_rows = len(ladder.ids)
    if n_rows == 0 or ladder.ia is None or len(ladder.ia) == 0:
        return ladder.ids[:0]
    n = ladder.n_reach
    k = min(max(int(k), 1), max(n, 1))
    inside = np.zeros(n_rows, dtype=bool)
    inside[ladder.order[:k]] = True
    ia, ib = ladder.ia, ladder.ib
    ring = np.zeros(n_rows, dtype=bool)
    ring[ia[inside[ib] & ~inside[ia]]] = True
    ring[ib[inside[ia] & ~inside[ib]]] = True
    return ladder.ids[ring]


def regions_at(ladder, t, np):
    """Label ids of every region whose join threshold is <= t (the seed
    always): the threshold-closed set. The drag does NOT use this (a tie
    would admit a whole group at once, see the module docstring); it is the
    reference the flood order refines."""
    return ladder.ids[ladder.join <= float(t)]


def rank_at(ladder, t, np):
    """The rank a previously released threshold t maps to on this ladder:
    every region with join <= t, except that a tie AT t admits only its first
    member (the join value alone cannot say how far into a tied group the
    user had dragged, and admitting the whole group is the outlier-seed jump
    this ordering exists to avoid). Clamped to [1, n_reach]."""
    n = ladder.n_reach
    if n <= 1:
        return 1
    s = ladder.sorted_join[:n]
    k = min(int(np.searchsorted(s, float(t), side="right")),
            int(np.searchsorted(s, float(t), side="left")) + 1)
    return min(max(k, 1), n)


def threshold_for_rank(ladder, k):
    """The join value of the k-th region on the ladder (k >= 1); the threshold
    that admits exactly the first k regions (plus any tied with the k-th)."""
    n = ladder.n_reach
    k = min(max(int(k), 1), max(n, 1))
    return float(ladder.sorted_join[k - 1]) if n else 0.0


def initial_rank(ladder, np, max_frac=0.05, min_k=1):
    """A data-driven first threshold: the largest RELATIVE gap in the join
    ladder within its first `max_frac` (at least two rungs), so the initial
    fill stops where the similarity structure has its first natural break.
    Clamped to [min_k, that window]; 1 (the seed alone) when the ladder is
    too short to have a break."""
    n = ladder.n_reach
    if n <= 1:
        return 1
    M = min(max(int(max_frac * n), 2), n)
    s = ladder.sorted_join[:M]
    if M < 3:
        return min(max(min_k, 1), M)
    # Gaps between rungs 1..M-1 (skip the seed->first gap, which says nothing
    # about the structure among the candidates).
    nxt, cur = s[2:], s[1:-1]
    rel = (nxt - cur) / (nxt - s[0] + 1e-12)
    i = int(np.argmax(rel)) + 1            # rung index whose next gap is largest
    return min(max(i + 1, min_k, 1), M)


def drag_to_rank(k0, dy_px, n_reach, px_per_step=4.0, accel_px=12.0):
    """Rank after a vertical drag of `dy_px` screen pixels (positive = UP =
    more regions). Linear near the start (one region per few pixels, so the
    first rungs are individually reachable) plus a quadratic term so a long
    drag still sweeps a thousand-region ladder."""
    dy = float(dy_px)
    mag = abs(dy)
    steps = mag / float(px_per_step) + (mag / float(accel_px)) ** 2
    k = int(k0) + int(math.copysign(int(round(steps)), dy)) if dy else int(k0)
    return min(max(k, 1), max(int(n_reach), 1))
