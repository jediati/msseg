"""Seam gestures in the store and their resolution against a seam graph
(msseg.labeler.seam_labeling). Headless, numpy only:

    pytest packages/mslabeler/tests/test_seam_labeling.py
"""
import numpy as np
import pytest

from msseg.labeler import seam_labeling as sl
from msseg.labeler.labeling import Interaction, LabelStore, Placement, SEAM_TOOLS
from msseg.labeler.seams import (SeamGraph, SEAM_BOUNDARY, SEAM_INTERIOR, SEAM_UNKNOWN)

KEY = "data/s0.tiff"


def blocks_raster():
    """The test_labeling raster: four 8x8 blocks with sparse ids {0, 2, 5, 9}
    in a -1 border. Seams: 0|2 at x=10 (y 2..10), 0|5 at y=10 (x 2..10),
    2|9 at y=10 (x 10..18), 5|9 at x=10 (y 10..18)."""
    lab = np.full((20, 20), -1, np.int32)
    lab[2:10, 2:10] = 0
    lab[2:10, 10:18] = 2
    lab[10:18, 2:10] = 5
    lab[10:18, 10:18] = 9
    return lab


def make(tool, points, class_id, uid, key=KEY):
    return Interaction(uid, key, 0, 0, tool, points, class_id)


def seam_index(g):
    return {(int(a), int(b)): i for i, (a, b) in enumerate(zip(g.a, g.b))}


# --------------------------------------------------------------------------- #
# the store
# --------------------------------------------------------------------------- #
def test_store_keeps_seam_gestures_apart():
    st = LabelStore()
    it = st.add("squiggle", [(1, 1)], 1, KEY)
    sc = st.add_seam("scope", [(0, 0), (5, 5)], SEAM_INTERIOR, KEY, meta={"tool": "scope"})
    tr = st.add_seam("trace", [(0, 0), (5, 0)], SEAM_BOUNDARY, KEY)
    assert [x.uid for x in st.interactions] == [1] and [x.uid for x in st.seams] == [2, 3]
    assert st.get(2) is sc and st.get(3) is tr and st.get(1) is it
    assert st.for_slice_seams(KEY) == [sc, tr] and st.for_slice(KEY) == [it]
    with pytest.raises(ValueError):
        st.add_seam("squiggle", [(0, 0)], 1, KEY)
    with pytest.raises(ValueError):
        st.add_seam("trace", [(0, 0)], 0, KEY)
    with pytest.raises(ValueError):
        st.add("trace", [(0, 0)], 1, KEY)
    assert st.to_json()["version"] == 3 and len(st.to_json()["seams"]) == 2
    rev = st.rev
    assert st.remove_many([1, 3]) == 2 and st.rev == rev + 1
    assert st.interactions == [] and [x.uid for x in st.seams] == [2]
    st.remove(2)
    assert st.seams == [] and st.to_json()["version"] == 2 and "seams" not in st.to_json()


def test_json_round_trip_and_unknown_seam_tools_are_dropped():
    st = LabelStore()
    st.add_seam("trace", [(1.5, 2.0), (7.0, 2.0)], SEAM_BOUNDARY, KEY, 0, 0,
                meta={"toll": "feature"})
    doc = st.to_json()
    doc["seams"].append({"uid": 9, "slice": KEY, "tool": "squiggle", "class": 1,
                         "points": [[0, 0]]})
    back = LabelStore.from_json(doc)
    assert [x.uid for x in back.seams] == [1] and back.seams[0].meta == {"toll": "feature"}
    assert back._next_uid == 2
    # A seam class below 1 is clamped up; the region class count is not consulted.
    doc["seams"][0]["class"] = 0
    assert LabelStore.from_json(doc).seams[0].class_id == 1
    # rebind places seam gestures like region gestures.
    back.rebind([{"name": "a", "folder": "data", "files": [r"C:\d\s0.tiff"]}])
    assert back.seams[0].bound and (back.seams[0].si, back.seams[0].li) == (0, 0)
    assert set(SEAM_TOOLS) == {"scope", "trace"}


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
def test_scope_requires_full_containment():
    lab = blocks_raster()
    g = SeamGraph.from_labels(lab, np)
    idx = seam_index(g)
    all_in = sl.scope_mask(make("scope", [(1, 1), (19, 19)], 1, 1), g, np)
    assert all_in.all()
    part = sl.scope_mask(make("scope", [(1, 1), (11, 11)], 1, 2), g, np)
    assert part[idx[(0, 2)]] and part[idx[(0, 5)]]
    assert not part[idx[(2, 9)]] and not part[idx[(5, 9)]]
    assert not sl.scope_mask(make("scope", [(1, 1)], 1, 3), g, np).any()


def test_trace_coverage_threshold():
    lab = blocks_raster()
    g = SeamGraph.from_labels(lab, np)
    idx = seam_index(g)
    full = sl.trace_mask(make("trace", [(10, 2), (10, 10)], 2, 1), g, np)
    assert full[idx[(0, 2)]] and full.sum() == 1
    cov = sl.coverage(g, sl.trace_cracks(make("trace", [(10, 2), (10, 6)], 2, 2), g, np), np)
    assert cov[idx[(0, 2)]] == pytest.approx(0.5)
    assert sl.trace_mask(make("trace", [(10, 2), (10, 6)], 2, 2), g, np)[idx[(0, 2)]]
    assert not sl.trace_mask(make("trace", [(10, 2), (10, 5)], 2, 3), g, np)[idx[(0, 2)]]
    # A trace over cracks that are on no seam (inside a region) labels nothing.
    assert not sl.trace_mask(make("trace", [(4, 4), (4, 8)], 2, 4), g, np).any()
    # Collinear runs are one segment; a corner turn is two.
    two = sl.trace_mask(make("trace", [(10, 2), (10, 10), (18, 10)], 2, 5), g, np)
    assert two[idx[(0, 2)]] and two[idx[(2, 9)]] and two.sum() == 2


def test_resolution_precedence_scopes_then_traces():
    lab = blocks_raster()
    g = SeamGraph.from_labels(lab, np)
    idx = seam_index(g)
    scope = make("scope", [(1, 1), (19, 19)], SEAM_INTERIOR, 1)
    trace = make("trace", [(10, 2), (10, 10)], SEAM_BOUNDARY, 2)
    cls = sl.resolve_seams([trace, scope], g, np)
    assert cls[idx[(0, 2)]] == SEAM_BOUNDARY
    assert (np.delete(cls, idx[(0, 2)]) == SEAM_INTERIOR).all()
    # A scope added AFTER the trace (higher uid) does not erase it.
    later = make("scope", [(1, 1), (19, 19)], SEAM_INTERIOR, 3)
    cls2 = sl.resolve_seams([trace, scope, later], g, np)
    assert (cls2 == cls).all()
    # Later traces win over earlier traces; nothing else is unknown.
    flip = make("trace", [(10, 2), (10, 10)], SEAM_INTERIOR, 4)
    assert sl.resolve_seams([scope, trace, flip], g, np)[idx[(0, 2)]] == SEAM_INTERIOR
    # Without a scope the untouched seams stay unknown.
    alone = sl.resolve_seams([trace], g, np)
    assert alone[idx[(0, 2)]] == SEAM_BOUNDARY and (np.delete(alone, idx[(0, 2)]) == SEAM_UNKNOWN).all()
    sets = sl.seam_sets([trace, scope], g, np)
    assert [it.uid for it, _m in sets] == [1, 2]
    assert sl.seams_touching(sets, idx[(0, 2)]) == [1, 2]
    assert sl.seams_touching(sets, idx[(5, 9)]) == [1]


def test_traces_re_resolve_after_a_merge():
    lab = blocks_raster()
    g = SeamGraph.from_labels(lab, np)
    idx = seam_index(g)
    short = make("trace", [(10, 2), (10, 10)], SEAM_BOUNDARY, 1)      # the 0|2 seam
    long = make("trace", [(10, 2), (10, 18)], SEAM_BOUNDARY, 2)       # 0|2 and 5|9
    cls = sl.resolve_seams([short, long], g, np)
    assert cls[idx[(0, 2)]] == SEAM_BOUNDARY and cls[idx[(5, 9)]] == SEAM_BOUNDARY
    # Merge the two top blocks (2 -> 0): the 0|2 seam is gone (the trace on it
    # labels nothing), 2|9 became 0|9, and 5|9 is untouched and still traced.
    merged = lab.copy()
    merged[merged == 2] = 0
    g2 = SeamGraph.from_labels(merged, np)
    idx2 = seam_index(g2)
    assert (0, 2) not in idx2 and (0, 9) in idx2 and (5, 9) in idx2
    assert not sl.trace_mask(short, g2, np).any()
    cls2 = sl.resolve_seams([short, long], g2, np)
    assert cls2[idx2[(5, 9)]] == SEAM_BOUNDARY
    assert cls2[idx2[(0, 9)]] == SEAM_UNKNOWN and cls2[idx2[(0, 5)]] == SEAM_UNKNOWN
    # Merge top (2 -> 0) AND bottom (9 -> 5): the centre junction vanishes and
    # y = 10 is ONE 0|5 seam of 16 cracks. A trace over half of it keeps the
    # label (coverage == tau), a shorter one does not.
    both = merged.copy()
    both[both == 9] = 5
    g3 = SeamGraph.from_labels(both, np)
    idx3 = seam_index(g3)
    assert list(idx3) == [(0, 5)] and g3.lengths(np).tolist() == [16] and g3.n_junctions == 2
    half = make("trace", [(2, 10), (10, 10)], SEAM_BOUNDARY, 3)
    less = make("trace", [(2, 10), (8, 10)], SEAM_BOUNDARY, 4)
    assert sl.resolve_seams([half], g3, np)[0] == SEAM_BOUNDARY
    assert sl.resolve_seams([less], g3, np)[0] == SEAM_UNKNOWN


def test_placement_maps_image_gestures_onto_the_raster():
    lab = blocks_raster()
    g = SeamGraph.from_labels(lab, np, placement=Placement((100, 200), 4.0))
    idx = seam_index(g)
    # Raster corners (10, 2)-(10, 10) are image points (140, 208)-(140, 240).
    tr = make("trace", [(140.0, 208.0), (140.0, 240.0)], SEAM_BOUNDARY, 1)
    m = sl.trace_mask(tr, g, np)
    assert m[idx[(0, 2)]] and m.sum() == 1
    sc = make("scope", [(104.0, 204.0), (144.0, 244.0)], SEAM_INTERIOR, 2)
    s = sl.scope_mask(sc, g, np)
    assert s[idx[(0, 2)]] and s[idx[(0, 5)]] and s.sum() == 2
    # A slightly off image point still rounds onto the lattice.
    tr2 = make("trace", [(141.0, 207.0), (139.0, 241.0)], SEAM_BOUNDARY, 3)
    assert sl.trace_mask(tr2, g, np)[idx[(0, 2)]]


def test_empty_graph():
    g = SeamGraph.from_labels(np.zeros((6, 6), np.int32), np)
    assert sl.resolve_seams([make("trace", [(0, 0), (3, 0)], 2, 1)], g, np).shape == (0,)
    assert sl.coverage(g, np.zeros(0, np.int64), np).shape == (0,)
