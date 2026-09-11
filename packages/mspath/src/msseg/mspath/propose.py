"""Where should the next ROI go?

The overview is one prime and covers the whole slide, so a classifier trained
on a handful of annotations can score every coarse region on it. What it is
*unsure* about is where a full-resolution ROI is worth the 2-9 seconds -- which
is the loop HistomicsML runs (low-magnification superpixels, a confidence
heatmap, and the annotator sent where the model is weakest) and what ilastik's
uncertainty layer is for.

Two honest limits, both of which shape what is here:

* **The coarse and fine decompositions are not nested.** A level-4 region is
  not the union of the level-0 regions under it -- the discrete gradient is
  recomputed per level and persistence is not a resolution hierarchy. So this
  proposes *where to look*, and never claims a coarse label transfers down.
  Nothing here writes a label.
* **Uncertainty clusters.** The least confident regions of a slide are usually
  neighbours on one boundary, and twenty ROIs on one boundary teach a model
  almost nothing that one does. So the ranking is followed by a spacing rule,
  which is the whole difference between a useful proposal set and a pile.

Everything is a pure function over a statistics table and a probability
matrix; the app supplies both and turns the results into ROI records.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

METHODS = ("entropy", "margin", "confidence")


def region_scores(proba, np, method: str = "entropy"):
    """(n,) uncertainty per region from an (n, k) probability matrix, 0 = sure.

    `entropy` weighs the whole distribution and is the default: with three or
    more classes a region the model splits evenly between two is genuinely
    less settled than one it splits between all. `margin` (1 - the gap between
    the top two) is the classic active-learning score and asks only about the
    decision boundary. `confidence` (1 - the top probability) is the cheapest
    and the bluntest.
    """
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] == 0:
        return np.zeros(0 if p.ndim != 2 else p.shape[0], np.float64)
    p = np.clip(p, 0.0, 1.0)
    if method == "confidence":
        return 1.0 - p.max(axis=1)
    if method == "margin":
        if p.shape[1] < 2:
            return 1.0 - p.max(axis=1)
        part = np.sort(p, axis=1)
        return 1.0 - (part[:, -1] - part[:, -2])
    if method != "entropy":
        raise ValueError(f"unknown method {method!r} (expected one of {METHODS})")
    k = p.shape[1]
    if k < 2:
        return np.zeros(p.shape[0], np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    return h / float(np.log(k))            # normalised, so scores compare across models


def boundary_scores(arcs, pdiff, n_rows, np):
    """(n,) mean p(different class) over each region's arcs, 0 where unknown.

    The edge model asks a different question from the region model: not "what
    is this?" but "do these two differ?". A region whose neighbours it is
    unsure about is on a boundary the model cannot place, which is exactly the
    tissue worth looking at closely -- and it is information the per-region
    probabilities do not contain.
    """
    out = np.zeros(int(n_rows), np.float64)
    if arcs is None or pdiff is None:
        return out
    a = np.asarray(arcs.get("a"), np.intp)
    b = np.asarray(arcs.get("b"), np.intp)
    d = np.asarray(pdiff, np.float64)
    if not len(a) or len(a) != len(b) or len(d) != len(a):
        return out
    total = np.zeros(int(n_rows), np.float64)
    count = np.zeros(int(n_rows), np.float64)
    for side in (a, b):
        ok = (side >= 0) & (side < n_rows)
        np.add.at(total, side[ok], d[ok])
        np.add.at(count, side[ok], 1.0)
    nz = count > 0
    out[nz] = total[nz] / count[nz]
    return out


def combine(region, boundary, np, weight: float = 0.5):
    """Blend the two signals. `weight` 0 is the region model alone, 1 the edge
    model alone; both are already in [0, 1], so no rescaling is needed."""
    w = float(min(max(weight, 0.0), 1.0))
    r = np.asarray(region, np.float64)
    if boundary is None:
        return r
    b = np.asarray(boundary, np.float64)
    if b.shape != r.shape:
        return r
    return (1.0 - w) * r + w * b


def candidates(table, scores, np, conv, min_area: float = 0.0):
    """[(score, x, y, region_id)] sorted by score descending.

    **`scores` is indexed by REGION ID, not by table row.** That is the shape
    the classifier's outputs have -- `region_class` and `region_proba` are
    sized by the label raster's id space, because that is what a LUT indexes --
    while the statistics table has one row per living region, whose ids are
    whatever the decomposition assigned. Zipping the two positionally lines up
    the wrong regions and produces a perfectly plausible, entirely wrong
    ranking, so the id column does the gather.

    Positions come from the table's extremum columns, which are already in
    slide coordinates, so a proposal is a place on the slide and not on the
    raster that happened to find it. Regions below `min_area` are dropped: the
    smallest regions of a decomposition carry the noisiest scores on it, and a
    proposal set made of them is a tour of speckles.
    """
    if table is None or conv.extremum_xy is None:
        return []
    xs = table.column(conv.extremum_xy[0])
    ys = table.column(conv.extremum_xy[1])
    ids = table.column(conv.id_field)
    if xs is None or ys is None or ids is None:
        return []
    s = np.asarray(scores, np.float64)
    if s.size == 0 or len(ids) == 0:
        return []
    idx = np.asarray(ids, np.intp)
    keep = (idx >= 0) & (idx < len(s))
    if min_area > 0 and conv.area_field:
        area = table.column(conv.area_field)
        if area is not None:
            keep &= np.asarray(area, np.float64) >= float(min_area)
    rows = np.flatnonzero(keep)
    if not len(rows):
        return []
    vals = s[idx[rows]]
    order = rows[np.argsort(-vals, kind="stable")]
    return [(float(s[idx[i]]), float(xs[i]), float(ys[i]), int(ids[i]))
            for i in order]


def space_out(cands, spacing: float, count: int):
    """Greedily take the highest-scoring candidates no closer than `spacing`.

    A plain top-k lands every proposal on the one boundary the model is worst
    at. Rejecting a candidate that is within `spacing` of one already taken is
    the cheapest rule that fixes it, and it is the right one here: the
    candidates are already sorted, so this is one pass and the result is still
    score-ordered.
    """
    out = []
    for score, x, y, rid in cands:
        if len(out) >= int(count):
            break
        if any((x - px) ** 2 + (y - py) ** 2 < spacing * spacing
               for _s, px, py, _r in out):
            continue
        out.append((score, x, y, rid))
    return out


def rois_around(picks, *, level: int, size: int, level_scale: float,
                slide_shape, max_px: Optional[int] = None) -> List[Dict[str, Any]]:
    """ROI records centred on each pick.

    `size` is in pixels AT `level` -- that is the number that decides what the
    prime costs -- and the record is in slide coordinates, which is what an
    item key carries. A box is clamped to the slide by MOVING it rather than
    by shrinking it, so every proposal costs the same and a region near an
    edge is not quietly given a smaller look than one in the middle.
    """
    side = int(size) * float(level_scale)
    if max_px is not None and int(size) * int(size) > int(max_px):
        side = (float(max_px) ** 0.5) * float(level_scale)
    sh, sw = int(slide_shape[0]), int(slide_shape[1])
    w = h = max(1, int(round(side)))
    w, h = min(w, sw), min(h, sh)
    out = []
    for score, x, y, rid in picks:
        x0 = int(round(x - w / 2.0))
        y0 = int(round(y - h / 2.0))
        x0 = max(0, min(x0, sw - w))
        y0 = max(0, min(y0, sh - h))
        out.append({"level": int(level), "x": x0, "y": y0, "w": w, "h": h,
                    "score": float(score), "region": int(rid)})
    return out


def propose(table, proba, np, conv, *, arcs=None, pdiff=None, method: str = "entropy",
            boundary_weight: float = 0.0, count: int = 8, level: int = 0,
            size: int = 2048, level_scale: float = 1.0, slide_shape=(1, 1),
            spacing: Optional[float] = None, min_area: float = 0.0,
            max_px: Optional[int] = None) -> List[Dict[str, Any]]:
    """The whole loop: score, rank, space out, and turn into ROI records.

    `proba` and `pdiff` are in the classifier's own space -- rows indexed by
    REGION ID, arcs given as region ids -- and `table` supplies the id column
    that maps them onto its rows. `spacing` defaults to one ROI width, so two
    proposals never overlap: the point of a second box is to see somewhere
    else.
    """
    scores = region_scores(proba, np, method)
    if boundary_weight > 0:
        scores = combine(scores, boundary_scores(arcs, pdiff, len(scores), np), np,
                         boundary_weight)
    cands = candidates(table, scores, np, conv, min_area=min_area)
    width = float(size) * float(level_scale)
    picks = space_out(cands, width if spacing is None else float(spacing), count)
    return rois_around(picks, level=level, size=size, level_scale=level_scale,
                       slide_shape=slide_shape, max_px=max_px)


def summarise(rois: Sequence[Dict[str, Any]], method: str) -> str:
    """One line for the status bar."""
    if not rois:
        return "No proposals: classify the overview first, or lower the area gate."
    lo = min(r["score"] for r in rois)
    hi = max(r["score"] for r in rois)
    return (f"{len(rois)} ROI(s) proposed by {method} uncertainty "
            f"({lo:.3f}-{hi:.3f}), spaced apart.")
