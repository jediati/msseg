"""Tolls and the livewire (msseg.labeler.seam_path). Headless, numpy only:

    pytest packages/mslabeler/tests/test_seam_path.py
"""
import numpy as np
import pytest

from msseg.labeler import seam_path as sp
from msseg.labeler.seams import SeamGraph
from msseg.labeler.table import FeatureTable


def grid_raster(n=3, size=4):
    """An n x n grid of size x size blocks with ids 0..n*n-1 (row-major)."""
    lab = np.zeros((n * size, n * size), np.int32)
    for r in range(n):
        for c in range(n):
            lab[r * size:(r + 1) * size, c * size:(c + 1) * size] = r * n + c
    return lab


def seam_index(g):
    return {(int(a), int(b)): i for i, (a, b) in enumerate(zip(g.a, g.b))}


def point_on(g, seam, xy):
    pts = g.seam_points(seam).tolist()
    return seam, pts.index(list(xy))


def cracks(points):
    return sum(abs(x1 - x0) + abs(y1 - y0) for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]))


# --------------------------------------------------------------------------- #
# tolls
# --------------------------------------------------------------------------- #
def test_geometric_and_scaled_feature_affinity():
    lab = grid_raster(2, 3)
    g = SeamGraph.from_labels(lab, np)
    assert (sp.seam_affinity(g, "geometric", np) == 0).all()
    table = FeatureTable(["feature_id", "mean_base", "std_base", "ext_filtered"],
                         np.array([[0, 0.0, 1.0, 0.0], [1, 0.0, 1.0, 0.0],
                                   [2, 10.0, 1.0, 5.0], [3, 10.0, 1.0, 5.0]], np.float64))
    aff = sp.seam_affinity(g, "feature", np, table=table, channels=["base"])
    idx = seam_index(g)
    assert aff[idx[(0, 1)]] == 0 and aff[idx[(2, 3)]] == 0
    assert aff[idx[(0, 2)]] == 1 and aff[idx[(1, 3)]] == 1
    tolls = sp.seam_tolls(aff, np)
    assert tolls[idx[(0, 2)]] == pytest.approx(sp.TOLL_EPS)
    assert tolls[idx[(0, 1)]] == pytest.approx(1 + sp.TOLL_EPS)
    with pytest.raises(ValueError):
        sp.seam_affinity(g, "feature", np)
    with pytest.raises(ValueError):
        sp.seam_affinity(g, "nope", np)


def test_arc_tolls_join_through_the_flank_pair():
    lab = grid_raster(2, 3)
    g = SeamGraph.from_labels(lab, np)
    idx = seam_index(g)
    arcs = {"a": np.array([0, 0, 2], np.int32), "b": np.array([1, 2, 3], np.int32),
            "saddle": np.array([1.0, 9.0, 2.0], np.float32), "source": "msc"}
    vals = sp.arc_values(g, arcs, arcs["saddle"], np)
    assert vals[idx[(0, 2)]] == 9 and np.isnan(vals[idx[(1, 3)]])   # no arc for (1, 3)
    table = FeatureTable(["feature_id", "ext_filtered"],
                         np.array([[0, 0.0], [1, 0.0], [2, 1.0], [3, 1.0]], np.float64))
    aff = sp.seam_affinity(g, "barrier", np, table=table, arcs=arcs)
    assert aff[idx[(0, 2)]] == 1 and aff[idx[(1, 3)]] == 0
    pdiff = np.array([0.1, 0.9, np.nan], np.float32)
    aff = sp.seam_affinity(g, "edges", np, arcs=arcs, pdiff=pdiff)
    assert aff[idx[(0, 1)]] == pytest.approx(0.1) and aff[idx[(0, 2)]] == pytest.approx(0.9)
    assert aff[idx[(2, 3)]] == 0 and aff[idx[(1, 3)]] == 0
    with pytest.raises(ValueError):
        sp.seam_affinity(g, "edges", np, arcs=arcs)
    with pytest.raises(ValueError):
        sp.seam_affinity(g, "barrier", np, table=table, arcs={"a": [], "b": [], "saddle": None})
    aff = sp.seam_affinity(g, "model", np, boundaryness=np.full(g.n_seams, 0.7))
    assert np.allclose(aff, 0.7)
    with pytest.raises(ValueError):
        sp.seam_affinity(g, "model", np)


# --------------------------------------------------------------------------- #
# the livewire
# --------------------------------------------------------------------------- #
def test_geometric_path_is_the_straight_seam_chain():
    g = SeamGraph.from_labels(grid_raster(3, 4), np)
    lw = sp.Livewire(g, sp.seam_tolls(np.zeros(g.n_seams), np), np)
    idx = seam_index(g)
    # Anchor mid-seam on the 0|1 seam (x = 4, y 0..4) at (4, 2); target on
    # the 7|8 seam (x = 8, y 8..12) at (8, 10).
    lw.anchor(*point_on(g, idx[(0, 1)], (4, 2)))
    assert lw.active
    pts = lw.path_to(*point_on(g, idx[(7, 8)], (8, 10)))
    assert pts[0] == (4, 2) and pts[-1] == (8, 10)
    assert cracks(pts) == 12                      # |dx| + |dy| on the lattice
    assert cracks(pts) * (1 + sp.TOLL_EPS) == pytest.approx(lw.cost_to(*point_on(g, idx[(7, 8)], (8, 10))))
    # Same-seam target: the direct run.
    assert lw.path_to(*point_on(g, idx[(0, 1)], (4, 4))) == [(4, 2), (4, 3), (4, 4)]
    assert lw.path_to(*point_on(g, idx[(0, 1)], (4, 0))) == [(4, 2), (4, 1), (4, 0)]


def test_tolls_reroute_and_scope_restricts():
    g = SeamGraph.from_labels(grid_raster(3, 4), np)
    idx = seam_index(g)
    aff = np.zeros(g.n_seams)
    # Make the 1|4 seam (y = 4, x 4..8) and 4|5 (x = 8, y 4..8) attractive:
    # the path from (4, 4) to (8, 8) is then those two, not 3|4 + 4|7.
    aff[idx[(1, 4)]] = 1.0
    aff[idx[(4, 5)]] = 1.0
    lw = sp.Livewire(g, sp.seam_tolls(aff, np), np)
    lw.anchor(*point_on(g, idx[(1, 4)], (4, 4)))
    pts = lw.path_to(*point_on(g, idx[(4, 5)], (8, 8)))
    assert pts == [(4, 4), (5, 4), (6, 4), (7, 4), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8)]
    # A detour can beat the direct run on an expensive seam: anchor at (4, 4)
    # on 1|4 with 1|4 itself made expensive and the ring around region 4
    # cheap -- the target (7, 4) on the same seam is reached the long way.
    aff2 = np.ones(g.n_seams)
    aff2[idx[(1, 4)]] = 0.0
    lw2 = sp.Livewire(g, sp.seam_tolls(aff2, np), np)
    lw2.anchor(*point_on(g, idx[(1, 4)], (4, 4)))
    pts2 = lw2.path_to(*point_on(g, idx[(1, 4)], (7, 4)))
    assert pts2[0] == (4, 4) and pts2[-1] == (7, 4) and cracks(pts2) > 3
    # Restricting to the seams around region 4 blocks the far corner.
    allow = np.zeros(g.n_seams, bool)
    for pair in ((1, 4), (3, 4), (4, 5), (4, 7)):
        allow[idx[pair]] = True
    lw3 = sp.Livewire(g, sp.seam_tolls(aff, np), np, restrict=allow)
    lw3.anchor(*point_on(g, idx[(1, 4)], (4, 4)))
    assert lw3.path_to(*point_on(g, idx[(4, 5)], (8, 8))) is not None
    assert lw3.path_to(*point_on(g, idx[(7, 8)], (8, 10))) is None
    assert lw3.cost_to(*point_on(g, idx[(7, 8)], (8, 10))) is None
    assert not lw3.allowed_at(idx[(7, 8)])


def test_extend_drop_and_commit():
    g = SeamGraph.from_labels(grid_raster(3, 4), np)
    idx = seam_index(g)
    lw = sp.Livewire(g, sp.seam_tolls(np.zeros(g.n_seams), np), np)
    lw.anchor(*point_on(g, idx[(0, 1)], (4, 0)))
    assert lw.commit_points() == []
    assert lw.extend(*point_on(g, idx[(1, 4)], (8, 4)))
    assert lw.extend(*point_on(g, idx[(4, 5)], (8, 8)))
    assert len(lw.legs) == 2 and lw.anchor_point == point_on(g, idx[(4, 5)], (8, 8))
    assert lw.total_cost == pytest.approx(12 * (1 + sp.TOLL_EPS))
    assert lw.seams_on_path == {idx[(0, 1)], idx[(1, 4)], idx[(4, 5)]}
    assert lw.commit_points() == [(4, 0), (4, 4), (8, 4), (8, 8)]
    assert cracks(lw.points()) == 12
    assert lw.drop_last() and len(lw.legs) == 1
    assert lw.anchor_point == point_on(g, idx[(1, 4)], (8, 4))
    assert lw.commit_points() == [(4, 0), (4, 4), (8, 4)]
    assert lw.drop_last() and lw.drop_last() and not lw.active
    assert not lw.drop_last()


def test_loop_anchor_goes_either_way_round():
    lab = np.zeros((8, 8), np.int32)
    lab[2:5, 2:6] = 1                      # an island: one loop of 14 cracks
    g = SeamGraph.from_labels(lab, np)
    assert g.n_seams == 1 and g.j0[0] == -1
    lw = sp.Livewire(g, sp.seam_tolls(np.zeros(1), np), np)
    lw.anchor(0, 0)
    pts = lw.path_to(0, 12)                # 2 cracks backwards, not 12 forwards
    assert cracks(pts) == 2 and pts[0] == (2, 2) and pts[-1] == tuple(g.seam_points(0)[12])
    assert cracks(lw.path_to(0, 5)) == 5
    assert lw.extend(0, 5) and lw.commit_points()[0] == (2, 2)


def test_compress_collinear():
    assert sp.compress_collinear([(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2)]) == \
        [(0, 0), (2, 0), (2, 2), (3, 2)]
    assert sp.compress_collinear([(0, 0), (0, 0), (1, 0)]) == [(0, 0), (1, 0)]
    assert sp.compress_collinear([(5, 5)]) == [(5, 5)]
    # A reversal is a direction change, so the turn-back point is kept.
    assert sp.compress_collinear([(0, 0), (1, 0), (2, 0), (1, 0)]) == [(0, 0), (2, 0), (1, 0)]


def test_seams_on_path_counts_only_walked_seams():
    g = SeamGraph.from_labels(grid_raster(3, 4), np)
    idx = seam_index(g)
    lw = sp.Livewire(g, sp.seam_tolls(np.zeros(g.n_seams), np), np)
    # Anchor mid-seam on 0|1, extend to the junction (4, 4) named through the
    # 3|4 seam's end, then on to (8, 4) along 1|4: the 3|4 seam was never
    # walked and must not be counted.
    lw.anchor(*point_on(g, idx[(0, 1)], (4, 2)))
    assert lw.extend(*point_on(g, idx[(3, 4)], (4, 4)))
    assert lw.leg_seams[-1] == {idx[(0, 1)]}
    assert lw.extend(*point_on(g, idx[(1, 4)], (8, 4)))
    assert lw.leg_seams[-1] == {idx[(1, 4)]}
    assert lw.seams_on_path == {idx[(0, 1)], idx[(1, 4)]}
    # A zero-length same-seam target walks nothing.
    lw.anchor(*point_on(g, idx[(0, 1)], (4, 2)))
    assert lw._route(*point_on(g, idx[(0, 1)], (4, 2)))[2] == set()
