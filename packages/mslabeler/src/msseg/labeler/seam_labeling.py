"""Resolving seam gestures against a seam graph.

A seam gesture is geometry in image coordinates, like every other gesture
(``labeling.Interaction``; ``meta`` is never read here): a **scope** is a
box -- every seam whose corners all lie inside it is labelled, interior by
default -- and a **trace** is a polyline of corners along seams, whose cracks
are matched against the graph by their label-independent ids: a seam takes
the trace's class when at least ``SEAM_COVER_TAU`` of its cracks lie under
the trace. That is what lets a trace drawn at one persistence re-resolve at
another: the corner lattice does not move, only the seams do, so a seam that
merged into a longer one keeps its label while it is still half covered, and
one that vanished simply has no successor.

Precedence: scopes first (in uid order), then traces (in uid order), each
overwriting -- so a scope widened AFTER a trace does not erase the trace.
This is the one deliberate departure from ``resolve_slice``'s pure uid order.

Pure numpy, ``np`` passed in.
"""
from __future__ import annotations

from .labeling import SEAM_TOOLS
from .seams import SEAM_COVER_TAU, SEAM_UNKNOWN, cracks_of_polyline


def scope_box(interaction, graph, np):
    """The scope's rect in raster corner coords ``(xa, ya, xb, yb)`` (floats,
    inclusive), or None for a degenerate gesture."""
    pts = interaction.points
    if len(pts) < 2:
        return None
    pl = graph.placement
    (x0, y0), (x1, y1) = pl.to_raster(*pts[0]), pl.to_raster(*pts[-1])
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def scope_mask(interaction, graph, np):
    """bool ``[S]``: the seams every corner of which lies inside the scope."""
    S = graph.n_seams
    box = scope_box(interaction, graph, np)
    if box is None or S == 0:
        return np.zeros(S, bool)
    xa, ya, xb, yb = box
    bb = graph.bboxes(np)
    return ((bb[:, 0] >= xa) & (bb[:, 1] >= ya) & (bb[:, 2] <= xb) & (bb[:, 3] <= yb))


def trace_cracks(interaction, graph, np):
    """The crack ids under a trace, on the graph's own lattice."""
    if len(interaction.points) < 2:
        return np.zeros(0, np.int64)
    pts = graph.to_raster(interaction.points, np)
    return cracks_of_polyline(pts, graph.width, graph.height, np)


def coverage(graph, crack_ids, np):
    """float64 ``[S]``: the fraction of each seam's cracks among `crack_ids`."""
    S = graph.n_seams
    if S == 0:
        return np.zeros(0, np.float64)
    ids = np.unique(np.asarray(crack_ids, np.int64))
    seam_of = graph.seam_of_cracks(ids, np)
    seam_of = seam_of[seam_of >= 0]
    counts = np.bincount(seam_of, minlength=S).astype(np.float64)
    return counts / np.maximum(graph.lengths(np), 1)


def trace_mask(interaction, graph, np, tau=SEAM_COVER_TAU):
    """bool ``[S]``: the seams a trace labels (coverage >= tau)."""
    if graph.n_seams == 0:
        return np.zeros(0, bool)
    cov = coverage(graph, trace_cracks(interaction, graph, np), np)
    return cov >= float(tau)


def gesture_mask(interaction, graph, np, tau=SEAM_COVER_TAU):
    if interaction.tool == "scope":
        return scope_mask(interaction, graph, np)
    if interaction.tool == "trace":
        return trace_mask(interaction, graph, np, tau)
    return np.zeros(graph.n_seams, bool)


def ordered(interactions):
    """The gestures in the order they apply: scopes by uid, then traces."""
    its = [it for it in interactions if it.tool in SEAM_TOOLS]
    return (sorted((it for it in its if it.tool == "scope"), key=lambda it: it.uid)
            + sorted((it for it in its if it.tool == "trace"), key=lambda it: it.uid))


def seam_sets(interactions, graph, np, tau=SEAM_COVER_TAU):
    """``[(interaction, bool[S])]`` in application order -- what each gesture
    labels, for the hover lookup and the readout."""
    return [(it, gesture_mask(it, graph, np, tau)) for it in ordered(interactions)]


def resolve_seams(interactions, graph, np, tau=SEAM_COVER_TAU, sets=None):
    """uint8 ``[S]`` seam class per seam: 0 unknown, else the class of the last
    gesture (scopes first, then traces) that covers it."""
    cls = np.full(graph.n_seams, SEAM_UNKNOWN, np.uint8)
    for it, mask in (sets if sets is not None else seam_sets(interactions, graph, np, tau)):
        if mask.any():
            cls[mask] = int(it.class_id)
    return cls


def seams_touching(sets, seam):
    """The uids of the gestures whose set contains `seam` (hover lookup)."""
    return [it.uid for it, mask in sets if 0 <= seam < len(mask) and mask[seam]]


__all__ = ["scope_box", "scope_mask", "trace_cracks", "coverage", "trace_mask",
           "gesture_mask", "ordered", "seam_sets", "resolve_seams", "seams_touching"]
