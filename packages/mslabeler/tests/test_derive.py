"""The conversion matrix (msseg.labeler.derive): region gestures -> arc and
seam labels, extents vs samples, explicit seam gestures last. Headless, over
the four-block raster of test_seam_labeling."""
import numpy as np

from msseg.labeler import derive
from msseg.labeler.labeling import Interaction
from msseg.labeler.magic_fill import arcs_from_labels
from msseg.labeler.seams import SEAM_BOUNDARY, SEAM_INTERIOR, SEAM_UNKNOWN, SeamGraph

KEY = "data/s0.tiff"


def blocks_raster():
    """Four 8x8 blocks with ids {0, 2, 5, 9}: seams 0|2 (x=10), 0|5 (y=10),
    2|9 (y=10), 5|9 (x=10)."""
    lab = np.full((20, 20), -1, np.int32)
    lab[2:10, 2:10] = 0
    lab[2:10, 10:18] = 2
    lab[10:18, 2:10] = 5
    lab[10:18, 10:18] = 9
    return lab


def g(uid, tool, pts, cls, meta=None):
    return Interaction(uid, KEY, 0, 0, tool, pts, cls, meta=meta)


def setup():
    lab = blocks_raster()
    graph = SeamGraph.from_labels(lab, np)
    arcs = arcs_from_labels(lab, np)
    sidx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(graph.a, graph.b))}
    aidx = {(int(min(a, b)), int(max(a, b))): i for i, (a, b) in enumerate(zip(arcs["a"], arcs["b"]))}
    return lab, graph, arcs, sidx, aidx


def test_samples_derive_seams_and_arcs_only_between_labelled_pairs():
    lab, graph, arcs, sidx, aidx = setup()
    gestures = [g(1, "squiggle", [(5, 5), (6, 5)], 1),      # block 0 -> class 1
                g(2, "squiggle", [(14, 5), (15, 5)], 1),    # block 2 -> class 1
                g(3, "squiggle", [(5, 14), (6, 14)], 2)]    # block 5 -> class 2
    rc, sl, both, diff = derive.labels_for_item(gestures, [], lab, graph, arcs, np)
    assert rc[0] == 1 and rc[2] == 1 and rc[5] == 2 and rc[9] == 0
    assert sl.cls[sidx[(0, 2)]] == SEAM_INTERIOR and sl.cls[sidx[(0, 5)]] == SEAM_BOUNDARY
    assert sl.cls[sidx[(2, 9)]] == SEAM_UNKNOWN and sl.cls[sidx[(5, 9)]] == SEAM_UNKNOWN
    assert sl.derived[sidx[(0, 2)]] and sl.derived[sidx[(0, 5)]] and not sl.derived[sidx[(2, 9)]]
    assert sl.sets == []
    assert both[aidx[(0, 2)]] and not diff[aidx[(0, 2)]]
    assert both[aidx[(0, 5)]] and diff[aidx[(0, 5)]]
    assert not both[aidx[(2, 9)]] and not both[aidx[(5, 9)]]


def test_an_extent_labels_its_unlabelled_neighbours_as_outside():
    lab, graph, arcs, sidx, aidx = setup()
    ext = g(1, "taps", [(5, 5), (14, 5)], 1, meta={"extent": True})    # blocks 0 and 2
    rc, sl, both, diff = derive.labels_for_item([ext], [], lab, graph, arcs, np)
    assert sl.cls[sidx[(0, 2)]] == SEAM_INTERIOR, "inside the extent"
    assert sl.cls[sidx[(0, 5)]] == SEAM_BOUNDARY and sl.cls[sidx[(2, 9)]] == SEAM_BOUNDARY
    assert sl.cls[sidx[(5, 9)]] == SEAM_UNKNOWN, "neither flank is the extent's"
    assert diff[aidx[(0, 5)]] and diff[aidx[(2, 9)]] and both[aidx[(0, 2)]] and not diff[aidx[(0, 2)]]
    assert not both[aidx[(5, 9)]]
    # A same-class SAMPLE outside the extent is not "outside": fill, then
    # patch the gland with a squiggle -- no boundary between them.
    patch = g(2, "squiggle", [(5, 14), (6, 14)], 1)                        # block 5 -> class 1
    rc, sl, both, diff = derive.labels_for_item([ext, patch], [], lab, graph, arcs, np)
    assert sl.cls[sidx[(0, 5)]] == SEAM_INTERIOR and not diff[aidx[(0, 5)]] and both[aidx[(0, 5)]]
    assert sl.cls[sidx[(2, 9)]] == SEAM_BOUNDARY, "the still-unlabelled neighbour"
    assert sl.cls[sidx[(5, 9)]] == SEAM_UNKNOWN, "a sample says nothing about its neighbours"
    # A different-class sample outside: boundary by the sample rule already.
    other = g(3, "squiggle", [(14, 14), (15, 14)], 2)                      # block 9 -> class 2
    rc, sl, both, diff = derive.labels_for_item([ext, other], [], lab, graph, arcs, np)
    assert sl.cls[sidx[(2, 9)]] == SEAM_BOUNDARY and diff[aidx[(2, 9)]]


def test_is_extent_flag_and_compat():
    assert derive.is_extent(g(1, "taps", [(1, 1)], 1, meta={"extent": True}))
    assert not derive.is_extent(g(1, "taps", [(1, 1)], 1, meta={"extent": False, "outline": [[[0, 0]]]}))
    assert not derive.is_extent(g(1, "squiggle", [(1, 1)], 1))
    # Stage-2 sessions: a magic / blob core with an outline was an extent in
    # intent; the ring never; an accepted prediction never.
    assert derive.is_extent(g(1, "taps", [(1, 1)], 1, meta={"tool": "magic", "outline": [[[0, 0]]]}))
    assert derive.is_extent(g(1, "taps", [(1, 1)], 1,
                              meta={"tool": "blobber", "part": "core", "outline": [[[0, 0]]]}))
    assert not derive.is_extent(g(1, "taps", [(1, 1)], 1,
                                  meta={"tool": "blobber", "part": "ring", "outline": [[[0, 0]]]}))
    assert not derive.is_extent(g(1, "taps", [(1, 1)], 1, meta={"tool": "accept", "outline": [[[0, 0]]]}))
    assert not derive.is_extent(g(1, "taps", [(1, 1)], 1, meta={"tool": "magic"})), "no outline yet"


def test_scope_and_trace_label_arcs_and_win_over_derived():
    lab, graph, arcs, sidx, aidx = setup()
    samples = [g(1, "squiggle", [(5, 5), (6, 5)], 1), g(2, "squiggle", [(14, 5), (15, 5)], 2)]
    # A scope over everything: every seam interior, every arc "same" -- even
    # the 0|2 pair the samples call different.
    scope = g(3, "scope", [(0, 0), (20, 20)], SEAM_INTERIOR)
    rc, sl, both, diff = derive.labels_for_item(samples, [scope], lab, graph, arcs, np)
    assert (sl.cls == SEAM_INTERIOR).all() and not sl.derived.any() and len(sl.sets) == 1
    assert both.all() and not diff.any()
    # An open trace along 0|2 (corners (10,2)..(10,10)): that seam boundary,
    # its arc different; nothing else touched.
    trace = g(4, "trace", [(10, 2), (10, 10)], SEAM_BOUNDARY)
    rc, sl, both, diff = derive.labels_for_item([], [trace], lab, graph, arcs, np)
    assert sl.cls[sidx[(0, 2)]] == SEAM_BOUNDARY and not sl.derived[sidx[(0, 2)]]
    assert (sl.cls != SEAM_UNKNOWN).sum() == 1
    assert both[aidx[(0, 2)]] and diff[aidx[(0, 2)]] and both.sum() == 1
    # Explicit beats derived: samples say 0|5 boundary, a scope says interior.
    samples2 = [g(1, "squiggle", [(5, 5), (6, 5)], 1), g(2, "squiggle", [(5, 14), (6, 14)], 2)]
    rc, sl, both, diff = derive.labels_for_item(samples2, [scope], lab, graph, arcs, np)
    assert sl.cls[sidx[(0, 5)]] == SEAM_INTERIOR and not sl.derived[sidx[(0, 5)]]
    assert not diff[aidx[(0, 5)]]
    # A trace on a seam whose flanks a sample calls the same: boundary (an
    # instance boundary), arc different.
    same = [g(1, "squiggle", [(5, 5), (6, 5)], 1), g(2, "squiggle", [(14, 5), (15, 5)], 1)]
    rc, sl, both, diff = derive.labels_for_item(same, [trace], lab, graph, arcs, np)
    assert sl.cls[sidx[(0, 2)]] == SEAM_BOUNDARY and diff[aidx[(0, 2)]]


def test_seam_labels_without_arcs_or_graph():
    lab, graph, arcs, sidx, aidx = setup()
    rc, sl, both, diff = derive.labels_for_item([], [], lab, None, None, np)
    assert sl is None and both is None and rc.max() == 0
    empty = SeamGraph.from_labels(np.full((4, 4), -1, np.int32), np)
    res = derive.seam_labels(empty, np, None, [], [])
    assert res.cls.shape == (0,) and res.derived.shape == (0,)
    b, d = derive.arc_labels({"a": np.zeros(0, np.int32), "b": np.zeros(0, np.int32)}, np)
    assert b.shape == (0,)


def test_enclosed_ids_and_is_closed():
    lab = blocks_raster()
    loop = [(2, 2), (10, 2), (10, 10), (2, 10), (2, 2)]        # round block 0, on cracks
    assert derive.is_closed(loop) and not derive.is_closed(loop[:-1])
    ids, mask, ya, xa = derive.enclosed_ids(loop, lab, np)
    assert ids == [0] and mask.sum() == 64
    wide = [(2, 2), (18, 2), (18, 10), (2, 10), (2, 2)]        # blocks 0 and 2
    assert derive.enclosed_ids(wide, lab, np)[0] == [0, 2]
    retraced = [(2, 2), (10, 2), (10, 10), (10, 2), (2, 2)]   # out and back: no area
    assert derive.enclosed_ids(retraced, lab, np) is None
    assert derive.enclosed_ids([(2, 2), (10, 2), (2, 2)], lab, np) is None
    assert derive.enclosed_ids([(30, 30), (40, 30), (40, 40), (30, 40), (30, 30)], lab, np) is None
