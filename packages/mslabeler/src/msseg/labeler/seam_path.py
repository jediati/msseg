"""Tolls and the livewire: shortest paths along the seams of a seam graph.

A **toll** is the cost per crack of walking a seam; a trace is the cheapest
path between two anchors, so a low toll on the seams that look like real
boundaries makes the path hug them. Tolls come from an **affinity** in [0, 1]
per seam (1 = certainly a boundary) as ``toll = eps + (1 - affinity)``, and
the affinity from one of:

  geometric      0 everywhere: the path is the shortest in cracks.
  feature        the z-scored mean dissimilarity of the two flanks over the
                 selected channels (magic_fill's ``mean`` metric), scaled by
                 its 95th percentile over the slice.
  bhattacharyya  the per-channel Gaussian overlap of the flanks, likewise.
  barrier        the arc's saddle depth |saddle - max(ext_a, ext_b)| (needs
                 MSC region arcs with saddles), likewise.
  edges          the edge model's p(diff) for the flank pair (needs a
                 '-> edges' model classified at this commit).
  model          the seam model's boundaryness (needs a trained seam model).

The **livewire** anchors at any point on any seam (a virtual node joined to
the seam's two junctions at the partial-length cost), runs ONE single-source
Dijkstra over the junction graph per anchor (heapq, the ``magic_fill``
pattern), and answers ``path_to`` for any other seam point by walking
predecessors -- so a mouse move is a lookup, not a search. A click extends
the path with a new anchor; the committed polyline is the concatenated legs
with collinear runs compressed. ``restrict`` (a seam mask) confines the
search to a scope.

Pure numpy + heapq, ``np`` passed in. No Tk.
"""
from __future__ import annotations

import heapq
import math

from . import magic_fill
from .fields import DEFAULT

TOLLS = ("geometric", "feature", "bhattacharyya", "barrier", "edges", "model")
TOLL_EPS = 0.05
# The tolls that need the statistics table / the record's arcs / a model.
TABLE_TOLLS = ("feature", "bhattacharyya", "barrier")
ARC_TOLLS = ("barrier", "edges")
MODEL_TOLLS = ("model",)
# Unbounded dissimilarities are scaled by this percentile over the slice so
# one outlier pair cannot squash every other seam onto toll 1.
_SCALE_PERCENTILE = 95.0


# --------------------------------------------------------------------------- #
# Affinity and tolls
# --------------------------------------------------------------------------- #
def arc_values(graph, arcs, values, np):
    """Per-seam value looked up from a per-arc array through the flank pair
    ``(a, b)``; NaN for a seam whose pair has no arc (MSC mode drops some
    saddles) and for a NaN arc value."""
    S = graph.n_seams
    out = np.full(S, np.nan, np.float64)
    if arcs is None or values is None or S == 0:
        return out
    a = np.asarray(arcs["a"], np.int64)
    b = np.asarray(arcs["b"], np.int64)
    if len(a) == 0:
        return out
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    K = int(max(hi.max(), graph.b.max())) + 1
    keys = lo * K + hi
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    vals = np.asarray(values, np.float64)[order]
    q = graph.pair_keys(np, K)
    pos = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
    hit = keys[pos] == q
    out[hit] = vals[pos[hit]]
    return out


def _scaled(aff, np):
    """Unbounded dissimilarity -> [0, 1] by its 95th percentile; NaN -> 0."""
    aff = np.asarray(aff, np.float64)
    ok = np.isfinite(aff)
    if not ok.any():
        return np.zeros(len(aff), np.float64)
    scale = float(np.percentile(aff[ok], _SCALE_PERCENTILE))
    out = np.zeros(len(aff), np.float64)
    if scale > 0:
        out[ok] = np.clip(aff[ok] / scale, 0.0, 1.0)
    return out


def flank_rows(graph, table, np, conv=DEFAULT):
    """Table row indices of each seam's flanks: ``(ia, ib, keep)`` over the
    seams (a seam whose flank has no row is dropped from ia/ib)."""
    ids = table.column(conv.id_field)
    if ids is None:
        raise ValueError(f"the statistics table has no {conv.id_field!r} column")
    return magic_fill.index_arcs({"a": graph.a, "b": graph.b}, ids, np)


def seam_affinity(graph, toll, np, table=None, arcs=None, channels=("base",),
                  pdiff=None, boundaryness=None, conv=DEFAULT):
    """float64 ``[S]`` affinity in [0, 1] per seam for `toll`. Raises
    ValueError with a user-facing message when what the toll needs is
    missing (the caller shows it and refuses the press)."""
    S = graph.n_seams
    if toll not in TOLLS:
        raise ValueError(f"unknown toll {toll!r}")
    if toll == "geometric" or S == 0:
        return np.zeros(S, np.float64)
    if toll in ("feature", "bhattacharyya"):
        if table is None:
            raise ValueError(f"{toll} needs the statistics table - Run first")
        metric = "mean" if toll == "feature" else "bhattacharyya"
        chans = list(channels) or ["base"]
        X, kind = magic_fill.row_vectors(table, metric, chans, np, conv=conv)
        ia, ib, keep = flank_rows(graph, table, np, conv)
        aff = np.full(S, np.nan, np.float64)
        aff[keep] = magic_fill.pairwise(X, ia, ib, kind, np)
        return _scaled(aff, np)
    if toll == "barrier":
        if arcs is None or arcs.get("saddle") is None:
            raise ValueError("barrier needs saddle values (MSC region arcs)")
        if table is None:
            raise ValueError("barrier needs the statistics table - Run first")
        sad = arc_values(graph, arcs, arcs["saddle"], np)
        ext = table.column(conv.extremum_value_field)
        if ext is None:
            raise ValueError(f"barrier needs the {conv.extremum_value_field!r} column")
        ia, ib, keep = flank_rows(graph, table, np, conv)
        ext = np.asarray(ext, np.float64)
        ea = np.full(S, np.nan)
        eb = np.full(S, np.nan)
        ea[keep], eb[keep] = ext[ia], ext[ib]
        return _scaled(np.abs(sad - np.maximum(ea, eb)), np)
    if toll == "edges":
        if pdiff is None or arcs is None:
            raise ValueError("edges needs an edge model's p(diff) at this commit - "
                             "pick a '-> edges' kind, Train (R), then Classify")
        aff = arc_values(graph, arcs, pdiff, np)
        return np.clip(np.nan_to_num(aff, nan=0.0), 0.0, 1.0)
    if toll == "model":
        if boundaryness is None:
            raise ValueError("model needs a trained seam model - Train seams first")
        aff = np.asarray(boundaryness, np.float64)
        if len(aff) != S:
            raise ValueError("boundaryness does not match this seam graph")
        return np.clip(np.nan_to_num(aff, nan=0.0), 0.0, 1.0)
    raise ValueError(f"unknown toll {toll!r}")


def seam_tolls(affinity, np, eps=TOLL_EPS):
    """Cost per crack: ``eps + (1 - affinity)``, in (eps, 1 + eps]."""
    return float(eps) + (1.0 - np.clip(np.asarray(affinity, np.float64), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Polylines
# --------------------------------------------------------------------------- #
def compress_collinear(points):
    """The polyline with interior points on straight runs (and exact repeats)
    dropped: a list of ``(x, y)`` ints. Endpoints are always kept."""
    pts = []
    for x, y in points:
        p = (int(x), int(y))
        if not pts or pts[-1] != p:
            pts.append(p)
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        d1 = (pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        d2 = (pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        # Same direction iff the cross product is zero and the dot is positive.
        if d1[0] * d2[1] - d1[1] * d2[0] != 0 or d1[0] * d2[0] + d1[1] * d2[1] <= 0:
            out.append(pts[i])
    out.append(pts[-1])
    return out


# --------------------------------------------------------------------------- #
# The livewire
# --------------------------------------------------------------------------- #
class Livewire:
    """Shortest paths along seams from a movable anchor.

    ``anchor(seam, idx)`` runs the search; ``path_to(seam, idx)`` returns the
    raster corner polyline (a list of ``(x, y)`` ints) from the current
    anchor to that point, or None when it is unreachable; ``extend`` freezes
    that path as a leg and re-anchors there; ``commit_points`` is every leg
    joined and compressed. A point on a seam is ``(seam, idx)`` with ``idx``
    indexing ``graph.seam_points(seam)``.
    """

    def __init__(self, graph, tolls, np, restrict=None):
        self.graph = graph
        self.np = np
        S = graph.n_seams
        self.tolls = np.asarray(tolls, np.float64).reshape(S)
        self.lengths = graph.lengths(np)
        cost = self.lengths * self.tolls
        if restrict is not None:
            self.allowed = np.asarray(restrict, bool).reshape(S)
            cost = np.where(self.allowed, cost, np.inf)
        else:
            self.allowed = np.ones(S, bool)
        self.cost = cost.tolist()
        self.toll_list = self.tolls.tolist()
        self.len_list = self.lengths.tolist()
        self.j0 = graph.j0.tolist()
        self.j1 = graph.j1.tolist()
        self.nbr, self.seam, self.ptr = graph.junction_csr(np)
        self.anchors = []          # [(seam, idx)]
        self.legs = []             # [[(x, y), ...]] one per extend
        self.leg_costs = []
        self.leg_seams = []        # [set(seam ids)] per leg
        self._dist = None
        self._pred = None

    # -- state ------------------------------------------------------------- #
    @property
    def active(self):
        return bool(self.anchors)

    @property
    def anchor_point(self):
        return self.anchors[-1] if self.anchors else None

    @property
    def total_cost(self):
        return float(sum(self.leg_costs))

    @property
    def seams_on_path(self):
        out = set()
        for s in self.leg_seams:
            out |= s
        return out

    def allowed_at(self, seam):
        return bool(self.allowed[int(seam)])

    # -- search ------------------------------------------------------------- #
    def anchor(self, seam, idx):
        """Start (or restart) the trace at a seam point."""
        self.anchors = [(int(seam), int(idx))]
        self.legs, self.leg_costs, self.leg_seams = [], [], []
        self._dijkstra(int(seam), int(idx))

    def _dijkstra(self, s, idx):
        J = self.graph.n_junctions
        dist = [math.inf] * J
        pred = [None] * J             # (prev junction | -1, seam, direction)
        heap = []
        if self.j0[s] >= 0 and self.allowed[s]:
            t = self.toll_list[s]
            n = self.len_list[s]
            for e, c, direction in ((self.j0[s], idx * t, -1), (self.j1[s], (n - idx) * t, +1)):
                if c < dist[e]:
                    dist[e] = c
                    pred[e] = (-1, s, direction)
                    heapq.heappush(heap, (c, e))
        nbr, seam, ptr, cost, j0 = self.nbr, self.seam, self.ptr, self.cost, self.j0
        while heap:
            c, u = heapq.heappop(heap)
            if c > dist[u]:
                continue
            for k in range(ptr[u], ptr[u + 1]):
                sm = seam[k]
                w = cost[sm]
                if w == math.inf:
                    continue
                v = nbr[k]
                nc = c + w
                if nc < dist[v]:
                    dist[v] = nc
                    pred[v] = (u, sm, +1 if j0[sm] == u else -1)
                    heapq.heappush(heap, (nc, v))
        self._dist = dist
        self._pred = pred

    # -- reconstruction ------------------------------------------------------ #
    def _seam_run(self, s, i0, i1):
        """Corner points of seam `s` from index i0 to i1 inclusive (either
        direction), as a list of (x, y)."""
        pts = self.graph.seam_points(s)
        if i1 >= i0:
            seg = pts[i0:i1 + 1]
        else:
            seg = pts[i1:i0 + 1][::-1]
        return [(int(x), int(y)) for x, y in seg]

    def _to_junction(self, e):
        """(points, seams) from the anchor to junction `e` along the search
        tree; None when unreachable."""
        if self._pred is None or e < 0 or self._dist[e] == math.inf:
            return None
        hops = []
        u = e
        while True:
            prev, sm, direction = self._pred[u]
            hops.append((sm, direction))
            if prev < 0:
                break
            u = prev
        hops.reverse()
        s0, i0 = self.anchors[-1]
        sm, direction = hops[0]
        n = self.len_list[sm]
        pts = self._seam_run(sm, i0, n if direction > 0 else 0)
        # A seam counts only when a crack of it is walked: an anchor sitting
        # on the junction itself walks none of its own seam.
        seams = {sm} if len(pts) > 1 else set()
        for sm, direction in hops[1:]:
            n = self.len_list[sm]
            run = self._seam_run(sm, 0, n) if direction > 0 else self._seam_run(sm, n, 0)
            pts.extend(run[1:])
            seams.add(sm)
        return pts, seams

    def _route(self, s, i1):
        """(points, cost, seams) of the cheapest path from the anchor to
        ``(s, i1)``, or None."""
        if not self.anchors:
            return None
        s0, i0 = self.anchors[-1]
        s, i1 = int(s), int(i1)
        best = None
        n = self.len_list[s]
        t = self.toll_list[s]
        if s == s0 and self.allowed[s]:
            if self.j0[s] < 0:                        # a loop: either way round
                fwd = (i1 - i0) % n
                bwd = (i0 - i1) % n
                pts = self.graph.seam_points(s)
                if fwd <= bwd:
                    idx = [(i0 + k) % n for k in range(fwd + 1)]
                    c = fwd * t
                else:
                    idx = [(i0 - k) % n for k in range(bwd + 1)]
                    c = bwd * t
                best = ([(int(pts[k, 0]), int(pts[k, 1])) for k in idx], c,
                        {s} if len(idx) > 1 else set())
            else:
                best = (self._seam_run(s, i0, i1), abs(i1 - i0) * t,
                        {s} if i1 != i0 else set())
        if self.j0[s] >= 0 and self.allowed[s] and self._dist is not None:
            for e, ie in ((self.j0[s], 0), (self.j1[s], n)):
                d = self._dist[e]
                if d == math.inf:
                    continue
                c = d + abs(i1 - ie) * t
                if best is not None and c >= best[1]:
                    continue
                tj = self._to_junction(e)
                if tj is None:
                    continue
                pts, seams = tj
                tail = self._seam_run(s, ie, i1)
                best = (pts + tail[1:], c, seams | ({s} if len(tail) > 1 else set()))
        return best

    def path_to(self, seam, idx):
        r = self._route(seam, idx)
        return None if r is None else r[0]

    def cost_to(self, seam, idx):
        r = self._route(seam, idx)
        return None if r is None else r[1]

    # -- legs ------------------------------------------------------------- #
    def extend(self, seam, idx):
        """Freeze the path to ``(seam, idx)`` as a leg and re-anchor there.
        False (and no change) when the point is unreachable."""
        r = self._route(seam, idx)
        if r is None:
            return False
        pts, c, seams = r
        self.legs.append(pts)
        self.leg_costs.append(float(c))
        self.leg_seams.append(seams)
        self.anchors.append((int(seam), int(idx)))
        self._dijkstra(int(seam), int(idx))
        return True

    def drop_last(self):
        """BackSpace: drop the last leg (re-anchoring at the previous point);
        with only the first anchor left, drop that too. False when empty."""
        if not self.anchors:
            return False
        if self.legs:
            self.legs.pop()
            self.leg_costs.pop()
            self.leg_seams.pop()
            self.anchors.pop()
            s, i = self.anchors[-1]
            self._dijkstra(s, i)
        else:
            self.anchors = []
            self._dist = self._pred = None
        return True

    def points(self):
        """Every leg joined (raster corners, uncompressed)."""
        out = []
        for leg in self.legs:
            out.extend(leg if not out else leg[1:])
        return out

    def commit_points(self):
        """The trace to store: the legs joined and compressed; [] with no leg."""
        return compress_collinear(self.points())


__all__ = ["TOLLS", "TOLL_EPS", "TABLE_TOLLS", "ARC_TOLLS", "MODEL_TOLLS", "arc_values",
           "flank_rows", "seam_affinity", "seam_tolls", "compress_collinear", "Livewire"]
