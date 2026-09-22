"""A trace drawn on one lattice, resolved on another: exact crack ids fail
(the finer seam meanders around the straight coarse trace) and the corridor
passes; on the trace's own lattice nothing changes."""
import numpy as np

from msseg.labeler import seam_labeling as sl
from msseg.labeler.labeling import Interaction, Placement
from msseg.labeler.seams import SEAM_BOUNDARY, SEAM_UNKNOWN, SeamGraph


def blocks_raster():
    lab = np.full((20, 20), -1, np.int32)
    lab[2:10, 2:10] = 0
    lab[2:10, 10:18] = 2
    lab[10:18, 2:10] = 5
    lab[10:18, 10:18] = 9
    return lab


def fine_meandering():
    """The blocks at 4x, with the 0|2 seam pushed one pixel right over most
    of its length: a straight coarse trace at x=40 then sits on fewer than
    half of its cracks."""
    fine = np.kron(blocks_raster(), np.ones((4, 4), np.int32))
    fine[8:30, 40] = 0              # 22 of the 32 rows of the seam move to x=41
    return fine


def seam_of(g, a, b):
    for i, (p, q) in enumerate(zip(g.a, g.b)):
        if (int(p), int(q)) == (a, b):
            return i
    raise KeyError((a, b))


def trace(points, meta=None, uid=1):
    return Interaction(uid, "wsi/a.svs", 0, 0, "trace", points, SEAM_BOUNDARY, meta=meta)


COARSE = Placement((0, 0), 4.0)     # the drawing lattice: 4 image px per raster px
# The 0|2 seam on the coarse graph runs from corner (10, 2) to (10, 10):
# image (40, 8) to (40, 40).
STRAIGHT = [(40.0, 8.0), (40.0, 40.0)]


def test_exact_ids_fail_off_lattice_and_the_corridor_passes():
    g_fine = SeamGraph.from_labels(fine_meandering(), np, placement=Placement((0, 0), 1.0))
    s02 = seam_of(g_fine, 0, 2)
    bare = trace(STRAIGHT)
    assert not sl.trace_mask(bare, g_fine, np)[s02], "exact ids: fewer than half the cracks"
    drawn_coarse = trace(STRAIGHT, meta={"level": 2, "scale": 4.0})
    mask = sl.trace_mask(drawn_coarse, g_fine, np)
    assert mask[s02], "the corridor (one coarse pixel wide) covers the meander"
    # Only that seam: its neighbours are not within a coarse pixel of the trace.
    assert mask.sum() == 1, [(int(g_fine.a[i]), int(g_fine.b[i])) for i in np.flatnonzero(mask)]
    cls = sl.resolve_seams([drawn_coarse], g_fine, np)
    assert cls[s02] == SEAM_BOUNDARY and (cls != SEAM_UNKNOWN).sum() == 1


def test_own_lattice_is_byte_identical():
    g = SeamGraph.from_labels(blocks_raster(), np, placement=COARSE)
    without = sl.trace_mask(trace(STRAIGHT), g, np)
    with_meta = sl.trace_mask(trace(STRAIGHT, meta={"level": 2, "scale": 4.0}), g, np)
    assert (without == with_meta).all() and without[seam_of(g, 0, 2)]
    # Corridor coverage on the own lattice would agree here too, but is not
    # what runs: a scale equal to the graph's selects the exact path.
    cov = sl.corridor_coverage(trace(STRAIGHT), g, 4.0, np)
    assert cov[seam_of(g, 0, 2)] == 1.0


def test_a_fine_trace_on_a_coarse_graph_covers_by_the_coarser_pixel():
    g = SeamGraph.from_labels(blocks_raster(), np, placement=COARSE)
    # Drawn at level 0 (scale 1) a pixel off the coarse seam's line.
    off_by_one = trace([(41.0, 8.0), (41.0, 40.0)], meta={"level": 0, "scale": 1.0})
    assert sl.trace_mask(off_by_one, g, np)[seam_of(g, 0, 2)], "r = max(1, 4) = one coarse px"
    far = trace([(52.0, 8.0), (52.0, 40.0)], meta={"level": 0, "scale": 1.0})
    assert not sl.trace_mask(far, g, np).any()


def test_corridor_is_empty_for_degenerate_input():
    g = SeamGraph.from_labels(blocks_raster(), np, placement=COARSE)
    assert not sl.corridor_coverage(trace([(1.0, 1.0)]), g, 4.0, np).any()
    empty = SeamGraph.from_labels(np.full((4, 4), -1, np.int32), np)
    assert sl.corridor_coverage(trace(STRAIGHT), empty, 1.0, np).shape == (0,)
