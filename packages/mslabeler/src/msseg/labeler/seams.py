"""The seam graph: the polylines along which regions meet.

Vocabulary (docs/seam_labeling.md). A **region** is a living basin, one id in
the int32 label raster (-1 = background). A **corner** is a pixel-corner
lattice point at integer image coordinates -- pixel ``(ix, iy)`` covers
``[ix, ix+1) x [iy, iy+1)``, so a raster of ``h x w`` pixels has
``(h+1) x (w+1)`` corners. A **crack** is the unit lattice step between two
corners that separates two differently labelled pixels; its id is
label-INDEPENDENT (``2*c`` for the step from corner ``c = iy*(w+1) + ix`` to
``(ix+1, iy)``, ``2*c + 1`` for the step to ``(ix, iy+1)``), which is what
lets a stored trace be matched against a *different* decomposition of the
same raster. A **seam** is a maximal chain of cracks between the same two
regions (its **flanks**, ``a < b``) from junction to junction; a **loop** is
a seam that closes on itself with no junction (an island). A **junction** is
a corner where three or more regions meet, or where a seam ends: a degree-1
corner (against background or the raster edge) or a degree-2 corner whose
two cracks separate different flank pairs.

``seams_from_labels`` is the pure numpy/Python reference for the C++
``msseg::extract_seam_graph`` (``mscoupon_py.seam_graph``); both produce the
same CANONICAL arrays -- an open seam runs from the smaller junction id to the
larger (a seam from a junction back to itself runs toward its smaller second
corner), a loop starts at its smallest ``(y, x)`` corner and leaves it in +x,
and seams are sorted by ``(a, b, j0, j1, first corner, second corner,
length)`` -- so the reference is the fallback when the extension lacks the
symbol, and the parity test is what certifies the C++.

Pure numpy, ``np`` passed in like ``labeling.py``. No Tk, no engine.
"""
from __future__ import annotations

from .labeling import IDENTITY, Placement, scalar_lut

SEAM_UNKNOWN, SEAM_INTERIOR, SEAM_BOUNDARY = 0, 1, 2
SEAM_CLASSES = ("unknown", "interior", "boundary")
# RGBA per seam class; class 0 (unknown) is transparent.
SEAM_COLORS = ((0, 0, 0, 0), (70, 190, 255, 255), (255, 60, 200, 255))
# A seam takes a trace's class when at least this fraction of its cracks lie
# under the trace (a stored trace re-resolves against a NEW decomposition by
# crack coverage, so the threshold is what tolerates a partial overlap).
SEAM_COVER_TAU = 0.5


# --------------------------------------------------------------------------- #
# Cracks
# --------------------------------------------------------------------------- #
def crack_id(x0, y0, x1, y1, w):
    """The label-independent id of the crack between two ADJACENT corners."""
    cw = int(w) + 1
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    if y0 == y1 and abs(x1 - x0) == 1:
        return 2 * (y0 * cw + min(x0, x1))
    if x0 == x1 and abs(y1 - y0) == 1:
        return 2 * (min(y0, y1) * cw + x0) + 1
    raise ValueError(f"corners ({x0}, {y0}) and ({x1}, {y1}) are not adjacent")


def cracks_of_polyline(points, w, h, np):
    """int64 crack ids under a lattice polyline of integer corners ``(n, 2)``.

    Consecutive points may be any distance apart along ONE axis (a stored
    trace compresses collinear runs); a step that moves on both axes is walked
    x first, then y (a staircase), so a corrupt or resampled point never
    raises. Steps outside the lattice are dropped."""
    pts = np.asarray(points, np.int64).reshape(-1, 2)
    if len(pts) < 2:
        return np.zeros(0, np.int64)
    w, h = int(w), int(h)
    cw = w + 1
    out = []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        if x1 != x0 and 0 <= y0 <= h:          # horizontal run at y0
            xs = np.arange(min(x0, x1), max(x0, x1), dtype=np.int64)
            xs = xs[(xs >= 0) & (xs < w)]
            out.append(2 * (y0 * cw + xs))
        if y1 != y0 and 0 <= x1 <= w:          # vertical run at x1
            ys = np.arange(min(y0, y1), max(y0, y1), dtype=np.int64)
            ys = ys[(ys >= 0) & (ys < h)]
            out.append(2 * (ys * cw + x1) + 1)
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def _pair_cracks(p0, p1, cw, np):
    """Crack ids of unit steps ``p0 -> p1`` (both ``(n, 2)`` int arrays)."""
    horizontal = p0[:, 1] == p1[:, 1]
    xm = np.minimum(p0[:, 0], p1[:, 0]).astype(np.int64)
    ym = np.minimum(p0[:, 1], p1[:, 1]).astype(np.int64)
    ids = np.where(horizontal, 2 * (p0[:, 1].astype(np.int64) * cw + xm),
                   2 * (ym * cw + p0[:, 0].astype(np.int64)) + 1)
    return ids


# --------------------------------------------------------------------------- #
# The reference extraction
# --------------------------------------------------------------------------- #
def seams_from_labels(labels, np):
    """The seam graph of an int32 ``(h, w)`` label raster as the seven arrays
    ``(a, b, j0, j1, offsets, points, junction_xy)`` -- the same canonical
    form as ``mscoupon_py.seam_graph``. Vectorized crack and junction
    detection, a Python walk per crack for the chaining (~1-2 s at 3232^2;
    the C++ is the fast path)."""
    lab = np.asarray(labels)
    if lab.ndim != 2:
        raise ValueError("labels must be a 2D raster")
    h, w = lab.shape
    empty = (np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int32),
             np.zeros(0, np.int32), np.zeros(1, np.int64), np.zeros((0, 2), np.int32),
             np.zeros((0, 2), np.int32))
    if h == 0 or w == 0:
        return empty
    lab = lab.astype(np.int64, copy=False)
    cw = w + 1
    P = np.full((h + 2, w + 2), -1, np.int64)
    P[1:-1, 1:-1] = lab
    TL, TR, BL, BR = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]   # (h+1, w+1)

    def crack(p, q):
        return (p != q) & (p >= 0) & (q >= 0)

    right, down, left, up = crack(TR, BR), crack(BL, BR), crack(TL, BL), crack(TL, TR)
    deg = (right.astype(np.int8) + down.astype(np.int8)
           + left.astype(np.int8) + up.astype(np.int8))
    K = int(lab.max()) + 1 if lab.size else 1

    def key(p, q):
        return np.minimum(p, q) * K + np.maximum(p, q)

    keys = np.stack([key(TR, BR), key(BL, BR), key(TL, BL), key(TL, TR)])
    masks = np.stack([right, down, left, up])
    big = np.int64(K) * K + 1
    key_max = np.where(masks, keys, -1).max(axis=0)
    key_min = np.where(masks, keys, big).min(axis=0)
    junction = (deg >= 3) | (deg == 1) | ((deg == 2) & (key_max != key_min))
    jc = np.flatnonzero(junction.ravel())                # raster order == id order
    jid = {int(c): i for i, c in enumerate(jc.tolist())}

    R = right.ravel().tolist()
    D = down.ravel().tolist()
    J = junction.ravel().tolist()
    n_corners = cw * (h + 1)
    visited = bytearray(2 * n_corners)

    def has(c, d):
        if d == 0:
            return R[c]
        if d == 1:
            return D[c]
        if d == 2:
            return c >= 1 and R[c - 1]
        return c >= cw and D[c - cw]

    def cid(c, d):
        if d == 0:
            return 2 * c
        if d == 1:
            return 2 * c + 1
        if d == 2:
            return 2 * (c - 1)
        return 2 * (c - cw) + 1

    def step(c, d):
        return (c + 1, c + cw, c - 1, c - cw)[d]

    def flanks(c, d):
        iy, ix = divmod(c, cw)
        if d == 2:
            ix -= 1
            d = 0
        elif d == 3:
            iy -= 1
            d = 1
        if d == 0:
            p, q = int(P[iy, ix + 1]), int(P[iy + 1, ix + 1])
        else:
            p, q = int(P[iy + 1, ix]), int(P[iy + 1, ix + 1])
        return (p, q) if p < q else (q, p)

    def walk(c, d, loop):
        start = c
        pts = [c]
        while True:
            visited[cid(c, d)] = 1
            n = step(c, d)
            pts.append(n)
            if J[n]:
                return n, pts
            if loop and n == start:
                return n, pts
            back = (d + 2) & 3
            for dd in range(4):
                if dd != back and has(n, dd):
                    d = dd
                    break
            else:
                return n, pts
            c = n

    seams = []      # (a, b, j0, j1, corner list)
    for c in jc.tolist():
        for d in range(4):
            if not has(c, d) or visited[cid(c, d)]:
                continue
            a, b = flanks(c, d)
            end, pts = walk(c, d, False)
            j0, j1 = jid[c], jid[end]
            flip = j1 < j0 or (j0 == j1 and pts[-2] < pts[1])
            if flip:
                pts.reverse()
                j0, j1 = j1, j0
            seams.append((a, b, j0, j1, pts))
    for c in range(n_corners):
        for d in (0, 1):
            if not has(c, d) or visited[cid(c, d)]:
                continue
            a, b = flanks(c, d)
            _end, pts = walk(c, d, True)
            body = pts[:-1]
            k = body.index(min(body))
            body = body[k:] + body[:k]
            if body[1] != body[0] + 1:
                body = [body[0]] + body[1:][::-1]
            seams.append((a, b, -1, -1, body + [body[0]]))

    seams.sort(key=lambda s: (s[0], s[1], s[2], s[3], s[4][0], s[4][1], len(s[4])))
    S = len(seams)
    a = np.fromiter((s[0] for s in seams), np.int32, S)
    b = np.fromiter((s[1] for s in seams), np.int32, S)
    j0 = np.fromiter((s[2] for s in seams), np.int32, S)
    j1 = np.fromiter((s[3] for s in seams), np.int32, S)
    lens = np.fromiter((len(s[4]) for s in seams), np.int64, S)
    offsets = np.zeros(S + 1, np.int64)
    np.cumsum(lens, out=offsets[1:])
    flat = np.fromiter((c for s in seams for c in s[4]), np.int64, int(offsets[-1]))
    points = np.stack([flat % cw, flat // cw], axis=1).astype(np.int32)
    junction_xy = np.stack([jc % cw, jc // cw], axis=1).astype(np.int32)
    return a, b, j0, j1, offsets, points, junction_xy


def seam_graph_arrays(labels, np, ext=None):
    """``ext.seam_graph(labels)`` when the extension module has it, else the
    reference. `ext` is the compiled module or None."""
    fn = getattr(ext, "seam_graph", None) if ext is not None else None
    if fn is not None:
        lab = np.ascontiguousarray(labels, dtype=np.int32)
        return tuple(fn(lab))
    return seams_from_labels(labels, np)


# --------------------------------------------------------------------------- #
# The graph object
# --------------------------------------------------------------------------- #
class SeamGraph:
    """One item's seam graph at one commit, with the item's ``Placement`` so
    corner points convert to and from image coordinates (a coupon slice is the
    identity; a whole-slide ROI is offset and scaled)."""

    __slots__ = ("a", "b", "j0", "j1", "offsets", "points", "junction_xy", "shape",
                 "placement", "rev", "_cache")

    def __init__(self, a, b, j0, j1, offsets, points, junction_xy, shape,
                 placement=None, rev=0):
        self.a, self.b, self.j0, self.j1 = a, b, j0, j1
        self.offsets, self.points, self.junction_xy = offsets, points, junction_xy
        self.shape = (int(shape[0]), int(shape[1]))
        self.placement = placement if placement is not None else IDENTITY
        self.rev = int(rev)
        self._cache = {}

    @classmethod
    def from_arrays(cls, arrays, shape, np, placement=None, rev=0):
        a, b, j0, j1, offsets, points, junction_xy = arrays
        return cls(np.asarray(a, np.int32), np.asarray(b, np.int32),
                   np.asarray(j0, np.int32), np.asarray(j1, np.int32),
                   np.asarray(offsets, np.int64),
                   np.asarray(points, np.int32).reshape(-1, 2),
                   np.asarray(junction_xy, np.int32).reshape(-1, 2),
                   shape, placement, rev)

    @classmethod
    def from_labels(cls, labels, np, ext=None, placement=None, rev=0):
        lab = np.asarray(labels)
        return cls.from_arrays(seam_graph_arrays(lab, np, ext), lab.shape, np, placement, rev)

    # -- sizes ------------------------------------------------------------- #
    @property
    def n_seams(self):
        return int(len(self.a))

    @property
    def n_junctions(self):
        return int(len(self.junction_xy))

    @property
    def width(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]

    def lengths(self, np):
        """Cracks per seam, int64 ``[S]``."""
        if "lengths" not in self._cache:
            self._cache["lengths"] = (np.diff(self.offsets) - 1).astype(np.int64)
        return self._cache["lengths"]

    def is_loop(self, np):
        return self.j0 < 0

    def seam_points(self, i):
        """The corner points ``(n, 2)`` of seam `i` (raster corner coords)."""
        return self.points[int(self.offsets[i]):int(self.offsets[i + 1])]

    # -- cracks ----------------------------------------------------------- #
    def all_cracks(self, np):
        """``(crack_ids int64[C], seam_of int32[C])`` for every crack of every
        seam, in seam order (unsorted)."""
        if "cracks" not in self._cache:
            n = len(self.points)
            if n < 2:
                self._cache["cracks"] = (np.zeros(0, np.int64), np.zeros(0, np.int32))
            else:
                keep = np.ones(n - 1, bool)
                keep[self.offsets[1:-1] - 1] = False       # steps that cross seams
                p0 = self.points[:-1][keep]
                p1 = self.points[1:][keep]
                ids = _pair_cracks(p0, p1, self.width + 1, np)
                seam_of = np.repeat(np.arange(self.n_seams, dtype=np.int32), self.lengths(np))
                self._cache["cracks"] = (ids, seam_of)
        return self._cache["cracks"]

    def crack_index(self, np):
        """``(sorted crack ids, seam_of)`` for ``searchsorted`` lookups."""
        if "index" not in self._cache:
            ids, seam_of = self.all_cracks(np)
            order = np.argsort(ids, kind="stable")
            self._cache["index"] = (ids[order], seam_of[order])
        return self._cache["index"]

    def seam_of_cracks(self, crack_ids, np):
        """Seam index per crack id (-1 for a crack on no seam)."""
        ids, seam_of = self.crack_index(np)
        q = np.asarray(crack_ids, np.int64)
        if len(ids) == 0 or len(q) == 0:
            return np.full(len(q), -1, np.int32)
        pos = np.searchsorted(ids, q)
        pos = np.minimum(pos, len(ids) - 1)
        hit = ids[pos] == q
        out = np.where(hit, seam_of[pos], -1).astype(np.int32)
        return out

    def seam_cracks(self, i, np):
        """Crack ids of seam `i`, in point order."""
        pts = self.seam_points(i)
        return _pair_cracks(pts[:-1], pts[1:], self.width + 1, np)

    # -- per-seam derived --------------------------------------------------- #
    def pair_keys(self, np, K=None):
        """``a * K + b`` per seam, for joining with an arc table."""
        if K is None:
            K = int(max(self.b.max(), self.a.max())) + 1 if self.n_seams else 1
        return self.a.astype(np.int64) * int(K) + self.b.astype(np.int64)

    def bboxes(self, np):
        """``(S, 4)`` int32 ``x0, y0, x1, y1`` over each seam's corners."""
        if "bboxes" not in self._cache:
            if self.n_seams == 0:
                self._cache["bboxes"] = np.zeros((0, 4), np.int32)
            else:
                starts = self.offsets[:-1]
                lo = np.minimum.reduceat(self.points, starts, axis=0)
                hi = np.maximum.reduceat(self.points, starts, axis=0)
                self._cache["bboxes"] = np.concatenate([lo, hi], axis=1).astype(np.int32)
        return self._cache["bboxes"]

    def junction_csr(self, np):
        """The junction graph as CSR lists ``(nbr, seam, indptr)`` -- for each
        junction, its neighbouring junctions and the seam that joins them,
        both directions; loops (no junction) are absent."""
        if "csr" not in self._cache:
            open_ = np.flatnonzero(self.j0 >= 0)
            src = np.concatenate([self.j0[open_], self.j1[open_]]).astype(np.intp)
            dst = np.concatenate([self.j1[open_], self.j0[open_]]).astype(np.intp)
            sid = np.concatenate([open_, open_]).astype(np.intp)
            order = np.argsort(src, kind="stable")
            src, dst, sid = src[order], dst[order], sid[order]
            n = self.n_junctions
            counts = np.bincount(src, minlength=n) if len(src) else np.zeros(n, np.intp)
            indptr = np.zeros(n + 1, dtype=np.intp)
            np.cumsum(counts, out=indptr[1:])
            self._cache["csr"] = (dst.tolist(), sid.tolist(), indptr.tolist())
        return self._cache["csr"]

    # -- coordinates -------------------------------------------------------- #
    def to_image(self, pts):
        """Raster corner points -> image coordinates (floats)."""
        pl = self.placement
        return [pl.to_image(x, y) for x, y in pts]

    def to_raster(self, pts, np):
        """Image points -> rounded raster corner points, int64 ``(n, 2)``."""
        pl = self.placement
        arr = np.asarray([pl.to_raster(x, y) for x, y in pts], np.float64).reshape(-1, 2)
        return np.rint(arr).astype(np.int64)


# --------------------------------------------------------------------------- #
# Snapping, rasters, LUTs
# --------------------------------------------------------------------------- #
def nearest_seam_point(graph, labels, x, y, np, radius=8):
    """Snap a raster point to the nearest crack within `radius`: returns
    ``(seam, idx)`` -- the seam and the index (into ``graph.seam_points``) of
    the crack's endpoint nearer to ``(x, y)`` -- or None when no crack lies in
    the window. The window is re-derived from `labels`, so it is exact on any
    raster the graph was built from."""
    lab = np.asarray(labels)
    h, w = lab.shape
    x, y = float(x), float(y)
    r = int(radius)
    cx, cy = int(np.floor(x)), int(np.floor(y))
    # Corner window [x0, x1] x [y0, y1], pixels [x0-1, x1] x [y0-1, y1].
    x0, x1 = max(cx - r, 0), min(cx + r + 1, w)
    y0, y1 = max(cy - r, 0), min(cy + r + 1, h)
    if x1 < x0 or y1 < y0:
        return None
    px0, py0 = max(x0 - 1, 0), max(y0 - 1, 0)
    sub = lab[py0:y1 + 1, px0:x1 + 1]
    P = np.full((sub.shape[0] + 2, sub.shape[1] + 2), -1, np.int64)
    P[1:-1, 1:-1] = sub
    # Corner (i, j) of P-space lattice = raster corner (px0 + j, py0 + i).
    TL, TR, BL, BR = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]

    def crack(p, q):
        return (p != q) & (p >= 0) & (q >= 0)

    right = crack(TR, BR)
    down = crack(BL, BR)
    best = None
    for mask, mx, my in ((right, 0.5, 0.0), (down, 0.0, 0.5)):
        iy, ix = np.nonzero(mask)
        if len(ix) == 0:
            continue
        gx = ix + px0
        gy = iy + py0
        d2 = (gx + mx - x) ** 2 + (gy + my - y) ** 2
        k = int(np.argmin(d2))
        if best is None or d2[k] < best[0]:
            best = (float(d2[k]), int(gx[k]), int(gy[k]), mx == 0.5)
    if best is None or best[0] > (r + 1.0) ** 2:
        return None
    _d2, gx, gy, horizontal = best
    cw = graph.width + 1
    cid = 2 * (gy * cw + gx) + (0 if horizontal else 1)
    s = int(graph.seam_of_cracks(np.asarray([cid], np.int64), np)[0])
    if s < 0:
        return None
    ids = graph.seam_cracks(s, np)
    ks = np.flatnonzero(ids == cid)
    if len(ks) == 0:
        return None
    k = int(ks[0])
    pts = graph.seam_points(s)
    da = (pts[k, 0] - x) ** 2 + (pts[k, 1] - y) ** 2
    db = (pts[k + 1, 0] - x) ** 2 + (pts[k + 1, 1] - y) ** 2
    return s, (k if da <= db else k + 1)


def seam_pixel_raster(graph, np):
    """int32 ``(h, w)``: the seam index on BOTH flank pixels of every crack,
    -1 elsewhere -- a 2-pixel line that survives nearest-neighbour zoom-out.
    At a junction the last seam in graph order wins, which is arbitrary but
    stable."""
    h, w = graph.shape
    out = np.full((h, w), -1, np.int32)
    n = len(graph.points)
    if n < 2:
        return out
    keep = np.ones(n - 1, bool)
    keep[graph.offsets[1:-1] - 1] = False
    p0 = graph.points[:-1][keep]
    p1 = graph.points[1:][keep]
    seam = np.repeat(np.arange(graph.n_seams, dtype=np.int32), graph.lengths(np))
    horizontal = p0[:, 1] == p1[:, 1]
    xm = np.minimum(p0[:, 0], p1[:, 0])
    ym = np.minimum(p0[:, 1], p1[:, 1])
    # Horizontal crack at (xm, y): pixels (xm, y-1) and (xm, y).
    hy, hx, hs = p0[horizontal, 1], xm[horizontal], seam[horizontal]
    out[hy - 1, hx] = hs
    out[hy, hx] = hs
    # Vertical crack at (x, ym): pixels (x-1, ym) and (x, ym).
    vx, vy, vs = p0[~horizontal, 0], ym[~horizontal], seam[~horizontal]
    out[vy, vx - 1] = vs
    out[vy, vx] = vs
    return out


def seam_class_lut(seam_class, np, colors=None, alpha=255):
    """``(S, 4)`` uint8 RGBA LUT for the seam overlay: seam -> its class
    colour; class 0 (unknown) is transparent."""
    table = np.asarray(SEAM_COLORS if colors is None else colors, np.uint8)
    cls = np.asarray(seam_class, np.intp)
    cls = np.clip(cls, 0, len(table) - 1)
    lut = table[cls].copy()
    lut[:, 3] = np.where(cls > 0, int(max(0, min(255, alpha))), 0).astype(np.uint8)
    return lut


def seam_scalar_lut(values, np, alpha=255, mask=None):
    """``(S, 4)`` uint8 LUT mapping a per-seam scalar in [0, 1] (boundaryness)
    onto the labeler's colour ramp."""
    return scalar_lut(values, np, alpha=alpha, mask=mask)


__all__ = [
    "SEAM_UNKNOWN", "SEAM_INTERIOR", "SEAM_BOUNDARY", "SEAM_CLASSES", "SEAM_COLORS",
    "SEAM_COVER_TAU", "Placement", "crack_id", "cracks_of_polyline", "seams_from_labels",
    "seam_graph_arrays", "SeamGraph", "nearest_seam_point", "seam_pixel_raster",
    "seam_class_lut", "seam_scalar_lut",
]
