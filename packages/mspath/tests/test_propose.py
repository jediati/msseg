"""ROI proposal: scoring, spacing, and the geometry of the boxes.

All pure -- no slide, no Tk, no compiled pipeline. What is worth pinning is
that the ranking means what it says (a coin-flip region outranks a confident
one), that the spacing rule actually stops the whole set landing on one
boundary, and that a proposed box is a place on the SLIDE.
"""
import numpy as np
import pytest

from msseg.labeler import fields
from msseg.labeler.table import FeatureTable
from msseg.mspath import propose as P

CONV = fields.DEFAULT
NAMES = ["feature_id", "area", "ext_x", "ext_y", "mean_base"]


def table(rows):
    """rows: [(id, area, x, y)]

    Scores handed to `candidates` are indexed by region ID, so a test's score
    array is sized by the largest id, not by the number of rows -- which is the
    convention the classifier's outputs actually have."""
    vals = np.array([[r[0], r[1], r[2], r[3], 0.0] for r in rows],
                    np.float64).reshape(-1, len(NAMES))
    return FeatureTable(list(NAMES), vals)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def test_entropy_is_normalised_and_ordered():
    p = np.array([[1.0, 0.0], [0.5, 0.5], [0.9, 0.1]])
    s = P.region_scores(p, np, "entropy")
    assert s[0] == pytest.approx(0.0)
    assert s[1] == pytest.approx(1.0), "a coin flip is maximally uncertain"
    assert 0 < s[2] < 1 and s[2] < s[1]


def test_entropy_normalisation_makes_models_comparable():
    """Two classes and four classes both top out at 1.0, so a threshold or a
    blend weight means the same thing whatever the model predicts."""
    assert P.region_scores(np.full((1, 2), 0.5), np, "entropy")[0] == pytest.approx(1.0)
    assert P.region_scores(np.full((1, 4), 0.25), np, "entropy")[0] == pytest.approx(1.0)


def test_margin_asks_only_about_the_top_two():
    p = np.array([[0.5, 0.5, 0.0], [0.4, 0.35, 0.25]])
    s = P.region_scores(p, np, "margin")
    assert s[0] == pytest.approx(1.0)
    assert s[1] == pytest.approx(1.0 - 0.05)
    # entropy disagrees, and should: the second splits three ways
    e = P.region_scores(p, np, "entropy")
    assert e[1] > e[0]


def test_confidence_is_one_minus_the_top():
    s = P.region_scores(np.array([[0.7, 0.3]]), np, "confidence")
    assert s[0] == pytest.approx(0.3)


def test_degenerate_probabilities_do_not_raise():
    assert len(P.region_scores(np.zeros((0, 3)), np)) == 0
    assert P.region_scores(np.zeros((2, 0)), np).shape == (2,)
    assert P.region_scores(np.ones((3, 1)), np).tolist() == [0.0, 0.0, 0.0]
    # a zero probability must not become a NaN through log(0)
    assert np.isfinite(P.region_scores(np.array([[1.0, 0.0, 0.0]]), np)).all()


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError):
        P.region_scores(np.ones((2, 2)) / 2, np, "vibes")


# --------------------------------------------------------------------------- #
# the edge signal
# --------------------------------------------------------------------------- #
def test_boundary_scores_average_a_regions_arcs():
    arcs = {"a": [0, 0, 1], "b": [1, 2, 2]}
    pdiff = [1.0, 0.0, 0.5]
    s = P.boundary_scores(arcs, pdiff, 3, np)
    assert s[0] == pytest.approx(0.5)          # arcs 1.0 and 0.0
    assert s[1] == pytest.approx(0.75)         # arcs 1.0 and 0.5
    assert s[2] == pytest.approx(0.25)         # arcs 0.0 and 0.5


def test_a_region_with_no_arcs_scores_zero_not_nan():
    s = P.boundary_scores({"a": [0], "b": [1]}, [1.0], 4, np)
    assert s[2] == 0.0 and s[3] == 0.0 and np.isfinite(s).all()


@pytest.mark.parametrize("arcs,pdiff", [
    (None, [1.0]), ({"a": [0], "b": [1]}, None),
    ({"a": [], "b": []}, []), ({"a": [0, 1], "b": [1]}, [1.0, 1.0]),   # mismatched
])
def test_missing_or_inconsistent_edges_score_zero(arcs, pdiff):
    assert not P.boundary_scores(arcs, pdiff, 3, np).any()


def test_out_of_range_arc_endpoints_are_ignored():
    s = P.boundary_scores({"a": [0, 99], "b": [1, -3]}, [1.0, 1.0], 2, np)
    assert np.isfinite(s).all() and s[0] == pytest.approx(1.0)


def test_combine_weights_the_two_signals():
    r = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert P.combine(r, b, np, 0.0).tolist() == [1.0, 0.0]
    assert P.combine(r, b, np, 1.0).tolist() == [0.0, 1.0]
    assert P.combine(r, b, np, 0.5).tolist() == [0.5, 0.5]
    assert P.combine(r, None, np, 0.5).tolist() == [1.0, 0.0]
    assert P.combine(r, np.array([0.0]), np, 0.5).tolist() == [1.0, 0.0]   # wrong shape


# --------------------------------------------------------------------------- #
# ranking and spacing
# --------------------------------------------------------------------------- #
def by_id(pairs, n):
    """A score array indexed by region id."""
    out = np.zeros(n, np.float64)
    for rid, v in pairs:
        out[rid] = v
    return out


def test_candidates_are_sorted_and_carry_slide_positions():
    t = table([(5, 100, 1000.0, 2000.0), (6, 100, 50.0, 60.0)])
    out = P.candidates(t, by_id([(5, 0.1), (6, 0.9)], 8), np, CONV)
    assert [c[3] for c in out] == [6, 5]
    assert out[0][1:3] == (50.0, 60.0)


def test_scores_are_read_by_region_id_not_by_row():
    """The bug this guards: the classifier's outputs are sized by the label
    raster's id space, the table has one row per living region, and zipping
    them positionally ranks the wrong regions convincingly."""
    t = table([(7, 100, 10.0, 10.0), (2, 100, 90.0, 90.0)])
    scores = by_id([(7, 0.9), (2, 0.1)], 8)
    assert [c[3] for c in P.candidates(t, scores, np, CONV)] == [7, 2]
    # positionally zipped, row 0 would have taken score[0] = 0.0 and lost
    assert P.candidates(t, scores, np, CONV)[0][0] == pytest.approx(0.9)


def test_an_id_beyond_the_score_array_is_dropped_not_wrapped():
    t = table([(1, 100, 0.0, 0.0), (99, 100, 10.0, 10.0)])
    out = P.candidates(t, by_id([(1, 0.5)], 4), np, CONV)
    assert [c[3] for c in out] == [1]


def test_the_area_gate_drops_speckles():
    t = table([(1, 5, 0.0, 0.0), (2, 500, 10.0, 10.0)])
    scores = by_id([(1, 0.9), (2, 0.1)], 4)
    assert [c[3] for c in P.candidates(t, scores, np, CONV, min_area=100)] == [2]


def test_candidates_without_positions_are_no_candidates():
    t = FeatureTable(["feature_id", "area"], np.zeros((2, 2)))
    assert P.candidates(t, np.ones(4), np, CONV) == []


def test_spacing_stops_the_whole_set_landing_on_one_boundary():
    """Five nearly-colocated uncertain regions and one far away: a plain top-k
    returns the cluster, which teaches a model almost nothing."""
    cands = [(0.99 - i * 0.01, 100.0 + i, 100.0 + i, i) for i in range(5)]
    cands.append((0.5, 5000.0, 5000.0, 99))
    picks = P.space_out(cands, spacing=500.0, count=3)
    assert [p[3] for p in picks] == [0, 99], picks


def test_spacing_zero_is_a_plain_top_k():
    cands = [(0.9, 0.0, 0.0, 1), (0.8, 0.0, 0.0, 2)]
    assert [p[3] for p in P.space_out(cands, 0.0, 5)] == [1, 2]


def test_picks_stay_score_ordered():
    cands = [(0.9, 0.0, 0.0, 1), (0.7, 900.0, 0.0, 2), (0.8, 1800.0, 0.0, 3)]
    assert [p[0] for p in P.space_out(cands, 100.0, 3)] == [0.9, 0.7, 0.8]


# --------------------------------------------------------------------------- #
# the boxes
# --------------------------------------------------------------------------- #
def test_a_box_is_centred_on_its_region_in_slide_coordinates():
    rois = P.rois_around([(0.9, 5000.0, 6000.0, 1)], level=0, size=1000,
                         level_scale=1.0, slide_shape=(20000, 20000))
    r = rois[0]
    assert (r["w"], r["h"]) == (1000, 1000)
    assert r["x"] + r["w"] // 2 == 5000 and r["y"] + r["h"] // 2 == 6000
    assert r["region"] == 1 and r["score"] == pytest.approx(0.9)


def test_size_is_in_level_pixels_so_the_cost_is_the_same_at_any_level():
    """The number that decides what a prime costs is the level-pixel side; a
    coarse level's box is proportionally wider on the slide."""
    a = P.rois_around([(1.0, 8000.0, 8000.0, 1)], level=0, size=512,
                      level_scale=1.0, slide_shape=(50000, 50000))[0]
    b = P.rois_around([(1.0, 8000.0, 8000.0, 1)], level=4, size=512,
                      level_scale=16.0, slide_shape=(50000, 50000))[0]
    assert a["w"] == 512 and b["w"] == 512 * 16


def test_a_box_at_the_edge_is_moved_not_shrunk():
    """Every proposal must cost the same, and a region near an edge must not be
    given a smaller look than one in the middle."""
    rois = P.rois_around([(1.0, 5.0, 5.0, 1), (1.0, 9995.0, 9995.0, 2)],
                         level=0, size=1000, level_scale=1.0, slide_shape=(10000, 10000))
    assert all(r["w"] == 1000 and r["h"] == 1000 for r in rois)
    assert rois[0]["x"] == 0 and rois[0]["y"] == 0
    assert rois[1]["x"] == 9000 and rois[1]["y"] == 9000


def test_a_box_larger_than_the_slide_is_clamped_to_it():
    r = P.rois_around([(1.0, 50.0, 50.0, 1)], level=0, size=5000,
                      level_scale=1.0, slide_shape=(100, 200))[0]
    assert (r["w"], r["h"], r["x"], r["y"]) == (200, 100, 0, 0)


def test_the_budget_caps_the_box():
    r = P.rois_around([(1.0, 50000.0, 50000.0, 1)], level=0, size=8192,
                      level_scale=1.0, slide_shape=(100000, 100000),
                      max_px=4096 * 4096)[0]
    assert r["w"] == 4096 and r["h"] == 4096


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_propose_ranks_spaces_and_places():
    t = table([(1, 100, 1000.0, 1000.0),      # confident
               (2, 100, 1020.0, 1000.0),      # uncertain, next to 3
               (3, 100, 1040.0, 1000.0),      # the most uncertain
               (4, 100, 9000.0, 9000.0)])     # uncertain, far away
    # indexed by region id, so row 0 is the id-0 region that does not exist
    proba = np.array([[1.0, 0.0],                     # id 0: unused
                      [0.99, 0.01], [0.55, 0.45], [0.5, 0.5], [0.6, 0.4]])
    rois = P.propose(t, proba, np, CONV, count=2, level=0, size=500,
                     level_scale=1.0, slide_shape=(20000, 20000))
    assert [r["region"] for r in rois] == [3, 4], rois
    assert all(r["w"] == 500 for r in rois)
    assert "2 ROI(s) proposed" in P.summarise(rois, "entropy")


def test_the_boundary_signal_can_change_the_answer():
    """The two signals ask different questions, so they can disagree -- which
    is the only reason to have both. (Three regions, not two: in a two-region
    graph every arc touches both, so a boundary score cannot distinguish them.)
    """
    t = table([(1, 100, 0.0, 0.0),            # unsure, no boundary
               (2, 100, 9000.0, 9000.0),      # confident, on a boundary
               (3, 100, 9400.0, 9400.0)])     # confident, its neighbour
    proba = np.array([[1.0, 0.0],                     # id 0: unused
                      [0.5, 0.5], [0.99, 0.01], [0.99, 0.01]])
    arcs = {"a": [2], "b": [3]}                       # region ids, not rows
    pdiff = [1.0]
    by_region = P.propose(t, proba, np, CONV, count=1, slide_shape=(20000, 20000))
    by_edges = P.propose(t, proba, np, CONV, count=1, arcs=arcs, pdiff=pdiff,
                         boundary_weight=1.0, slide_shape=(20000, 20000))
    assert by_region[0]["region"] == 1, "the region model wants its coin flip"
    assert by_edges[0]["region"] == 2, "the edge model wants the boundary"


def test_nothing_to_propose_says_so():
    assert P.propose(table([]), np.zeros((0, 2)), np, CONV) == []
    assert "No proposals" in P.summarise([], "entropy")
