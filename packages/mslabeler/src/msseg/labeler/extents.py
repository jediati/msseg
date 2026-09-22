"""Extents: a region set's outline, so a fill crosses levels.

A magic fill or a blob commits as ``taps`` -- one seed per region at its
seeding extremum -- which is the representation that re-resolves under a
persistence change on the SAME field. On another level those seeds land in
different, smaller regions and the extent dissolves. The set's **outline**
does not: closed loops in slide coordinates, stored in ``meta["outline"]``,
are what a taps gesture resolves by off its drawing level
(docs/design_multi_model_tasks.md §7.1, §8.2).

The outline is the seam graph of the binary inside / outside mask
(``seams.SeamGraph`` over a 0/1 raster padded with 0, so every boundary
closes): loops come out as loops, and the open seams that meet at a
checkerboard corner -- a degree-4 junction -- are chained through the
junction graph into closed walks (every junction of a binary mask has even
degree, so a walk that takes any unused seam always returns). The fill is
**even-odd**: loops XOR, so a hole stays a hole and a ring stays a ring.

Headless: numpy, PIL and ``seams``.
"""
from __future__ import annotations

from .labeling import IDENTITY
from .seams import SeamGraph


def default_ext():
    """The compiled extension's ``seam_graph`` when the coupon package has
    it (the fast path), else None (the numpy reference)."""
    try:
        from msseg import mscoupon as _m
        return getattr(_m, "_ext", None)
    except ImportError:
        return None


def outline_loops(inside, np, ext=None):
    """The closed loops bounding the True pixels of a 2D bool mask, each a
    list of ``(x, y)`` raster CORNER coordinates (pixel ``(i, j)`` spans
    corners ``(i, j)`` to ``(i+1, j+1)``), collinear runs compressed."""
    m = np.asarray(inside, bool)
    if m.ndim != 2 or not m.any():
        return []
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), np.int32)
    pad[1:-1, 1:-1] = m
    g = SeamGraph.from_labels(pad, np, ext=ext)
    loops = []
    used = set()
    for i in range(g.n_seams):
        if int(g.j0[i]) < 0:                          # a loop already
            loops.append([tuple(p) for p in g.seam_points(i).tolist()])
            used.add(i)
    nbr, seam, indptr = g.junction_csr(np)
    jxy = g.junction_xy.tolist()

    def oriented(s, from_j):
        pts = [tuple(p) for p in g.seam_points(s).tolist()]
        if pts and tuple(jxy[from_j]) == pts[0]:
            return pts, int(g.j1[s]) if int(g.j0[s]) == from_j else int(g.j0[s])
        far = int(g.j1[s]) if int(g.j0[s]) == from_j else int(g.j0[s])
        return pts[::-1], far

    for i in range(g.n_seams):
        if i in used:
            continue
        start = int(g.j0[i])
        cur_j, cur_s = start, i
        walk = []
        while True:
            used.add(cur_s)
            pts, nxt = oriented(cur_s, cur_j)
            walk.extend(pts if not walk else pts[1:])
            cur_j = nxt
            if cur_j == start:
                break
            found = None
            for k in range(indptr[cur_j], indptr[cur_j + 1]):
                if seam[k] not in used:
                    found = seam[k]
                    break
            if found is None:                          # cannot happen on a 0/1 mask
                break
            cur_s = found
        loops.append(walk)
    out = []
    for loop in loops:
        pts = _compress_loop([(int(x) - 1, int(y) - 1) for x, y in loop])   # undo the pad
        if len(pts) >= 3:
            out.append(pts)
    return out


def _compress_loop(pts):
    """A closed loop's corners only: exact repeats and points on straight
    runs dropped CYCLICALLY (a loop's first point is wherever the walk
    started, usually mid-edge, so ``compress_collinear`` alone would keep it)."""
    clean = []
    for p in pts:
        if not clean or clean[-1] != p:
            clean.append(p)
    if len(clean) > 1 and clean[0] == clean[-1]:
        clean.pop()
    n = len(clean)
    if n < 3:
        return clean
    keep = []
    for i in range(n):
        (x0, y0), (x1, y1), (x2, y2) = clean[i - 1], clean[i], clean[(i + 1) % n]
        if (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1) != 0:
            keep.append((int(x1), int(y1)))
    return keep


def outline_of_ids(labels, ids, np, place=None, ext=None):
    """The outline of the regions `ids` of an int32 label raster, as loops of
    ``[x, y]`` IMAGE coordinates through `place` (default identity), JSON
    ready -- what a magic fill / blob / accepted prediction records."""
    lab = np.asarray(labels)
    want = sorted({int(i) for i in ids if int(i) >= 0})
    if lab.size == 0 or not want:
        return []
    top = int(lab.max())
    lut = np.zeros(max(top, max(want)) + 2, bool)
    lut[want] = True
    inside = lut[np.maximum(lab, 0)] & (lab >= 0)
    ys, xs = np.nonzero(inside)
    if len(ys) == 0:
        return []
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    place = place or IDENTITY
    loops = outline_loops(inside[y0:y1, x0:x1], np, ext=ext)
    out = []
    for loop in loops:
        pts = []
        for x, y in loop:
            ix, iy = place.to_image(x + x0, y + y0)
            pts.append([float(ix), float(iy)])
        out.append(pts)
    return out


def extent_mask(loops, w, h, np):
    """The even-odd fill of `loops` (lists of ``(x, y)`` raster CORNER
    coordinates) over an ``h x w`` raster, as ``(mask, ya, xa)`` over the
    loops' bbox crop -- or None when nothing lands on the raster.

    A pixel is inside when its CENTRE is: the loops are drawn on a grid twice
    as fine, where a corner ``(x, y)`` is ``(2x, 2y)`` and pixel ``(i, j)``'s
    centre is ``(2i+1, 2j+1)``, and the centres are sampled -- so a loop
    around one pixel fills one pixel (PIL's inclusive fill on the corner grid
    itself would take a row and a column too many)."""
    from PIL import Image, ImageDraw
    pts_all = [(float(x), float(y)) for loop in (loops or ()) for x, y in loop]
    if not pts_all:
        return None
    xa = max(0, int(np.floor(min(x for x, _ in pts_all))) - 1)
    ya = max(0, int(np.floor(min(y for _, y in pts_all))) - 1)
    xb = min(int(w), int(np.ceil(max(x for x, _ in pts_all))) + 1)
    yb = min(int(h), int(np.ceil(max(y for _, y in pts_all))) + 1)
    if xb <= xa or yb <= ya:
        return None
    cw, ch = 2 * (xb - xa), 2 * (yb - ya)
    mask = np.zeros((yb - ya, xb - xa), bool)
    for loop in loops:
        if len(loop) < 3:
            continue
        img = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(img).polygon([(2.0 * (float(x) - xa), 2.0 * (float(y) - ya))
                                     for x, y in loop], fill=1, outline=None)
        big = np.asarray(img, dtype=bool)
        mask ^= big[1::2, 1::2]
    return mask, ya, xa
