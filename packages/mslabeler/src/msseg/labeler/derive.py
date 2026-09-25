"""Label derivation: from the store's gestures to labels for the three
targets -- regions, arcs, seams -- in one place.

The labeler trains three kinds of estimator: a REGION classifier (rows =
living regions), an ARC / edge model (pairs of adjacent regions: same class
or different) and a SEAM model (crack chains between two flanks: boundary or
interior). The store holds two kinds of gesture: region gestures (squiggle,
box, polygon, taps) and seam gestures (scope, trace). This module is the
conversion matrix between them (docs/design_multi_model_tasks.md §7.2), so
no estimator reads gestures directly and one gesture serves every target:

    drawn \\ wanted        region class   arc same/diff          seam boundary/interior
    region SAMPLE         itself         both ends labelled     both flanks labelled ->
                                         -> same/diff           boundary iff they differ
    region EXTENT         itself         + extent | unlabelled  + extent | unlabelled
                                           -> diff                -> boundary
    scope                 --             arcs of its seams      itself (interior)
                                         -> same
    trace (boundary)      --             arcs of its seams      itself
                                         -> diff

A **sample** ("these regions are class k") says nothing about its
neighbours. An **extent** ("this set IS the object": a blob's core, a magic
fill released whole, a lasso, an enclosure) says its outside neighbours are
not it -- but only the UNLABELLED ones: a neighbour a sample already calls
the same class stays interior / same, so "fill, then fill again to extend
the gland" never derives a boundary inside the gland, and the seam and arc
targets say the same thing about the same pair. The instance boundary
between two touching same-class objects is an explicit trace's job
(``EXTENT_EDGE_VS_SAME_CLASS`` keeps the other reading one constant away).

Precedence: samples, then extents, then the explicit seam gestures (scopes
by uid, then traces by uid -- ``seam_labeling.ordered``), each overwriting.
Derived labels are never stored: derivation runs against the consumer's own
decomposition, which is what lets a gesture serve two tasks, two
persistences or two levels.

Headless. Callers pass already-rasterized ``touched_sets`` (they own the
memo); nothing here rasterizes a region gesture.
"""
from __future__ import annotations

from collections import namedtuple

from .extents import extent_mask
from .labeling import resolve_sets, resolve_slice, touched_sets
from .seam_labeling import seam_sets as _explicit_seam_sets
from .seams import SEAM_BOUNDARY, SEAM_COVER_TAU, SEAM_INTERIOR, SEAM_UNKNOWN

# What a seam between an extent and a same-class LABELLED neighbour gets from
# the extent: None = nothing (it stays interior, like the arc's "same").
# SEAM_BOUNDARY would read every extent edge as an instance boundary.
EXTENT_EDGE_VS_SAME_CLASS = None

# Gestures written before the flag existed: a magic fill / blob core with an
# outline was always an extent in intent; the blob's ring never (its outer
# edge is where the ring stopped); an accepted prediction never.
EXTENT_COMPAT_TOOLS = ("magic", "blobber")

SeamLabels = namedtuple("SeamLabels", "cls sets derived")


# --------------------------------------------------------------------------- #
# Which gestures are extents
# --------------------------------------------------------------------------- #
def is_extent(it):
    meta = getattr(it, "meta", None) or {}
    if "extent" in meta:
        return meta.get("extent") is True
    return (meta.get("tool") in EXTENT_COMPAT_TOOLS and bool(meta.get("outline"))
            and meta.get("part") != "ring")


def extent_sets(sets):
    """The ``(gesture, ids)`` pairs of ``touched_sets`` that are extents."""
    return [(it, ids) for it, ids in sets if is_extent(it)]


# --------------------------------------------------------------------------- #
# The three targets
# --------------------------------------------------------------------------- #
def region_labels(gestures, labels, np, layer=None):
    """Region class per region id (0 = unlabelled): region gestures in uid
    order, a later one painting over an earlier one."""
    return resolve_slice(gestures, labels, np, layer)


def _class_table(region_class, K, np):
    rc = np.zeros(max(int(K), 1), np.uint8)
    if region_class is not None and len(region_class):
        m = min(len(rc), len(region_class))
        rc[:m] = np.asarray(region_class)[:m]
    return rc


def _membership(ids, K, np):
    inE = np.zeros(max(int(K), 1), bool)
    idx = [int(i) for i in ids if 0 <= int(i) < len(inE)]
    if idx:
        inE[idx] = True
    return inE


def seam_labels(graph, np, region_class=None, extents=(), seam_gestures=(),
                tau=SEAM_COVER_TAU, boundary_class=SEAM_BOUNDARY):
    """``SeamLabels(cls uint8[S], sets, derived bool[S])`` for an item's seam
    graph: `cls` is the seam class per seam (0 unknown / 1 interior / 2
    boundary); `sets` the explicit seam gestures' masks (for the hover
    lookup and the readout, as ``seam_labeling.seam_sets`` gives them);
    `derived` marks the seams whose label came from region gestures alone.
    `boundary_class` is the class a DERIVED boundary gets (a polyline task
    with several boundary kinds picks which one its region input means)."""
    S = graph.n_seams
    cls = np.full(S, SEAM_UNKNOWN, np.uint8)
    derived = np.zeros(S, bool)
    if S:
        a = graph.a.astype(np.intp)
        b = graph.b.astype(np.intp)
        K = int(max(a.max(), b.max())) + 1
        rc = _class_table(region_class, K, np)
        ca, cb = rc[a], rc[b]
        both = (ca > 0) & (cb > 0)
        cls[both & (ca != cb)] = SEAM_BOUNDARY
        cls[both & (ca == cb)] = SEAM_INTERIOR
        for _it, ids in extents:
            if not ids:
                continue
            inE = _membership(ids, K, np)
            ea, eb = inE[a], inE[b]
            edge = ea != eb
            other = np.where(ea, cb, ca)                 # the flank outside the extent
            cls[edge & (other == 0)] = SEAM_BOUNDARY
            if EXTENT_EDGE_VS_SAME_CLASS is not None:
                cls[edge & (other > 0) & (ca == cb)] = EXTENT_EDGE_VS_SAME_CLASS
        derived = cls > 0
        if boundary_class != SEAM_BOUNDARY:
            cls[cls == SEAM_BOUNDARY] = boundary_class
    sets = _explicit_seam_sets(list(seam_gestures), graph, np, tau) if seam_gestures else []
    explicit = np.zeros(S, bool)
    for it, mask in sets:
        if mask.any():
            cls[mask] = int(it.class_id)
            explicit |= mask
    derived &= ~explicit
    return SeamLabels(cls, sets, derived)


def arc_labels(arcs, np, region_class=None, extents=(), seam_gestures=(), graph=None,
               tau=SEAM_COVER_TAU):
    """``(both bool[n], diff bool[n])`` over the arcs in their own (label-id)
    order: `both` says the pair is labelled, `diff` that its two regions are
    of different classes -- what ``edge_model.fit_edge_model`` trains on."""
    a = np.asarray(arcs["a"], np.intp)
    b = np.asarray(arcs["b"], np.intp)
    n = len(a)
    if n == 0:
        return np.zeros(0, bool), np.zeros(0, bool)
    K = int(max(a.max(), b.max())) + 1
    rc = _class_table(region_class, K, np)
    ca, cb = rc[a], rc[b]
    both = (ca > 0) & (cb > 0)
    diff = both & (ca != cb)
    for _it, ids in extents:
        if not ids:
            continue
        inE = _membership(ids, K, np)
        ea, eb = inE[a], inE[b]
        hit = (ea != eb) & (np.where(ea, cb, ca) == 0)
        both |= hit
        diff |= hit
    if graph is not None and seam_gestures:
        for it, mask in _explicit_seam_sets(list(seam_gestures), graph, np, tau):
            if not mask.any():
                continue
            hit = _arcs_of_seams(graph, a, b, mask, np)
            both |= hit
            if int(it.class_id) == SEAM_BOUNDARY:
                diff |= hit
            else:
                diff &= ~hit
    return both, diff


def _arcs_of_seams(graph, a, b, mask, np):
    """bool over the arcs ``(a, b)``: those whose unordered flank pair is a
    seam under `mask` (the reverse of ``seam_path.arc_values``)."""
    ga = graph.a[mask].astype(np.int64)
    gb = graph.b[mask].astype(np.int64)
    if len(ga) == 0:
        return np.zeros(len(a), bool)
    K = int(max(a.max(), b.max(), ga.max(), gb.max())) + 1
    keys = np.minimum(a, b).astype(np.int64) * K + np.maximum(a, b).astype(np.int64)
    skeys = np.minimum(ga, gb) * K + np.maximum(ga, gb)
    return np.isin(keys, skeys)


# --------------------------------------------------------------------------- #
# Closed traces -> enclosures
# --------------------------------------------------------------------------- #
def is_closed(points):
    """A polyline that returns to its start (a trace closed on its first
    anchor)."""
    pts = list(points)
    return len(pts) >= 4 and tuple(pts[0]) == tuple(pts[-1])


def enclosed_ids(loop_points, labels, np):
    """The region ids whose pixel centres lie inside a closed loop of raster
    CORNER points -- ``(ids, mask, ya, xa)`` -- or None when the loop
    encloses nothing (a retraced leg, a loop off the raster)."""
    pts = [(float(x), float(y)) for x, y in loop_points]
    if len(pts) < 4:
        return None
    h, w = np.asarray(labels).shape
    em = extent_mask([pts], w, h, np)
    if em is None:
        return None
    mask, ya, xa = em
    if not mask.any():
        return None
    sub = np.asarray(labels)[ya:ya + mask.shape[0], xa:xa + mask.shape[1]]
    ids = sorted(int(v) for v in np.unique(sub[mask]) if v >= 0)
    if not ids:
        return None
    return ids, mask, ya, xa


# --------------------------------------------------------------------------- #
# Everything for one item (tests, headless scripts)
# --------------------------------------------------------------------------- #
def labels_for_item(region_gestures, seam_gestures, labels, graph, arcs, np, layer=None,
                    tau=SEAM_COVER_TAU):
    """``(region_class, SeamLabels | None, both | None, diff | None)`` for an
    item from its gestures, rasterizing the region gestures once."""
    sets = touched_sets(list(region_gestures), labels, np, layer)
    region_class = resolve_sets(sets, labels, np)
    ext = extent_sets(sets)
    sl = (seam_labels(graph, np, region_class, ext, seam_gestures, tau)
          if graph is not None else None)
    both = diff = None
    if arcs is not None:
        both, diff = arc_labels(arcs, np, region_class, ext, seam_gestures, graph, tau)
    return region_class, sl, both, diff
