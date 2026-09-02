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
  barrier        |saddle - seed extremum value|: purely topological, the
                 persistence-style flood anchored at a point. Needs saddles,
                 so it is unavailable on the pixel fallback.

  anchor mode    every candidate is compared with the SEED (no drift).
  chain mode     each arc compares its two endpoints (follows gradients).
  barrier        is always per arc.

Pure numpy (``np`` passed in, like ``labeling.py``); scipy is used for the
minimum spanning tree when present, with a heapq fallback. No Tk, no engine.
"""
from __future__ import annotations

import heapq
import math

METRICS = ("mean", "bhattacharyya", "barrier")
MODES = ("anchor", "chain")

# Metrics that need the saddle value per arc (unavailable on pixel adjacency).
EDGE_ONLY_METRICS = ("barrier",)


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


def node_dissimilarity(table, seed_row, metric, channels, np):
    """d(seed, r) for every row r (anchor mode). float64[n_rows], 0 at the seed."""
    if metric == "mean":
        acc = None
        for c in channels:
            m = _column(table, f"mean_{c}", np)
            z = (m - m[seed_row]) / _spread(m, np)
            acc = z * z if acc is None else acc + z * z
        if acc is None:
            raise ValueError("no channels selected")
        return np.sqrt(acc)
    if metric == "bhattacharyya":
        acc = None
        for c in channels:
            m = _column(table, f"mean_{c}", np)
            sd = _std_floor(_column(table, f"std_{c}", np), m, np)
            v1, v2 = sd[seed_row] ** 2, sd ** 2
            bc = (0.25 * np.log(0.25 * (v1 / v2 + v2 / v1 + 2.0))
                  + 0.25 * (m - m[seed_row]) ** 2 / (v1 + v2))
            acc = bc if acc is None else acc + bc
        if acc is None:
            raise ValueError("no channels selected")
        return acc
    if metric in EDGE_ONLY_METRICS:
        raise ValueError(f"{metric!r} is an arc metric, not a region metric")
    raise ValueError(f"unknown metric {metric!r}")


def edge_dissimilarity(table, ia, ib, metric, channels, np):
    """d(a, b) per arc (chain mode). float64[n_arcs]."""
    if metric == "mean":
        acc = None
        for c in channels:
            m = _column(table, f"mean_{c}", np)
            z = (m[ia] - m[ib]) / _spread(m, np)
            acc = z * z if acc is None else acc + z * z
        if acc is None:
            raise ValueError("no channels selected")
        return np.sqrt(acc)
    if metric == "bhattacharyya":
        acc = None
        for c in channels:
            m = _column(table, f"mean_{c}", np)
            sd = _std_floor(_column(table, f"std_{c}", np), m, np)
            v1, v2 = sd[ia] ** 2, sd[ib] ** 2
            bc = (0.25 * np.log(0.25 * (v1 / v2 + v2 / v1 + 2.0))
                  + 0.25 * (m[ia] - m[ib]) ** 2 / (v1 + v2))
            acc = bc if acc is None else acc + bc
        if acc is None:
            raise ValueError("no channels selected")
        return acc
    if metric in EDGE_ONLY_METRICS:
        raise ValueError(f"{metric!r} needs saddle values; use barrier_weights")
    raise ValueError(f"unknown metric {metric!r}")


def barrier_weights(saddle, seed_ext_value, np):
    """|saddle - seed extremum value| per arc: how high the flood must rise
    from the seed's well to spill over into the neighbour."""
    return np.abs(np.asarray(saddle, dtype=np.float64) - float(seed_ext_value))


def edge_weights(table, ia, ib, seed_row, metric, mode, channels, np,
                 saddle=None, seed_ext_value=None):
    """One weight per arc, whatever the metric/mode: the bottleneck search
    below only ever sees arcs."""
    if metric in EDGE_ONLY_METRICS:
        if saddle is None:
            raise ValueError(f"{metric!r} needs saddle values (pixel adjacency has none)")
        if seed_ext_value is None:
            raise ValueError(f"{metric!r} needs the seed's extremum value (ext_filtered)")
        return barrier_weights(saddle, seed_ext_value, np)
    if mode == "anchor":
        d = node_dissimilarity(table, seed_row, metric, channels, np)
        return np.maximum(d[ia], d[ib])
    if mode == "chain":
        return edge_dissimilarity(table, ia, ib, metric, channels, np)
    raise ValueError(f"unknown mode {mode!r}")


# --------------------------------------------------------------------------- #
# Bottleneck (minimax) path cost from the seed
# --------------------------------------------------------------------------- #
def bottleneck_join(n_nodes, ia, ib, w, seed, np, use_scipy=None):
    """float64[n_nodes]: for every node the smallest threshold t such that a
    path from `seed` exists whose every arc weight is <= t (inf = unreachable,
    0 at the seed). Arcs are undirected.

    The minimax path lives on the minimum spanning tree, so with scipy this is
    one MST plus a tree walk; without it, a heap-based minimax Dijkstra."""
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


def growth_order(n_nodes, ia, ib, w, seed, np, node_key=None):
    """Priority flood from the seed: ``(order, join)``.

    `join` is the bottleneck cost of bottleneck_join(); `order` lists the
    reachable nodes in the order the flood admits them -- non-decreasing join,
    ties broken by a secondary key: ``node_key[v]`` when given (anchor mode:
    the region's own dissimilarity to the seed), else the weight of the arc
    the node is entered through (chain / barrier). Every prefix of `order` is
    connected to the seed, which is what makes a rank on it a usable drag
    axis when many regions tie (see the module docstring). Unreached nodes
    follow in index order with join inf."""
    n = int(n_nodes)
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
            p1 = c1 if c1 > wk else wk
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
                 "seed_id", "seed_row", "metric", "mode", "channels")

    def __init__(self, ids, join, order, sorted_join, cum_area, n_reach,
                 seed_id, seed_row, metric, mode, channels):
        self.ids = ids
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


def build_ladder(table, arcs, seed_id, metric, mode, channels, np):
    """Join ladder for `seed_id` (a label id present in `table`)."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
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
    seed_ext = None
    if metric in EDGE_ONLY_METRICS:
        ext = table.column("ext_filtered")
        if ext is None:
            raise ValueError(f"{metric!r} needs the ext_filtered statistic")
        seed_ext = float(ext[seed_row])
    node_key = None
    if metric not in EDGE_ONLY_METRICS and mode == "anchor":
        # Anchor mode: the region's own dissimilarity is both the arc weight
        # ingredient and the tie-breaker (most seed-like first).
        node_key = node_dissimilarity(table, seed_row, metric, channels, np)
        w = np.maximum(node_key[ia], node_key[ib])
    else:
        w = edge_weights(table, ia, ib, seed_row, metric, mode, channels, np,
                         saddle=saddle, seed_ext_value=seed_ext)
    order, join = growth_order(len(ids), ia, ib, w, seed_row, np, node_key=node_key)
    sorted_join = join[order]
    n_reach = int(np.count_nonzero(np.isfinite(join)))
    area = table.column("area")
    cum_area = (np.cumsum(np.asarray(area, dtype=np.int64)[order])
                if area is not None else None)
    return Ladder(ids, join, order, sorted_join, cum_area, n_reach,
                  seed_id, seed_row, metric, mode, channels)


def regions_for_rank(ladder, k):
    """Label ids of the first k regions in flood order (the seed always): the
    set the drag shows at rank k. Connected by construction."""
    n = ladder.n_reach
    k = min(max(int(k), 1), max(n, 1))
    return ladder.ids[ladder.order[:k]]


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
