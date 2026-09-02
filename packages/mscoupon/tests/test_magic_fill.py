"""Pure-function tests for the labeler's magic fill (msseg.mscoupon.magic_fill).

Runs headless with numpy only (scipy exercised when present, the heapq path
always):

    pytest packages/mscoupon/tests/test_magic_fill.py
"""
import math

import numpy as np
import pytest

from msseg.mscoupon import magic_fill as mf
from msseg.mscoupon.common import FeatureTable

from test_labeling import blocks_raster

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

SCIPY_MODES = [False] + ([True] if HAVE_SCIPY else [])


def blocks_table(with_std=True, with_ext=False):
    """The per-region table matching blocks_raster(): ids {0,2,5,9}."""
    names = ["feature_id", "area", "mean_base"]
    cols = [[0.0, 64.0, 1.5], [2.0, 80.0, 2.5], [5.0, 96.0, 3.5], [9.0, 112.0, 4.5]]
    if with_std:
        names.append("std_base")
        for r, sd in zip(cols, (0.1, 0.1, 0.5, 0.0)):
            r.append(sd)
    if with_ext:
        names.append("ext_filtered")
        for r, e in zip(cols, (1.0, 2.0, 3.0, 4.0)):
            r.append(e)
    return FeatureTable(names, np.array(cols, np.float64))


# --------------------------------------------------------------------------- #
# adjacency
# --------------------------------------------------------------------------- #
def test_arcs_from_labels_4_neighbour_pairs_only():
    arcs = mf.arcs_from_labels(blocks_raster(), np)
    pairs = set(zip(arcs["a"].tolist(), arcs["b"].tolist()))
    assert pairs == {(0, 2), (0, 5), (2, 9), (5, 9)}     # no diagonals
    assert arcs["saddle"] is None and arcs["source"] == "pixels"
    assert arcs["a"].dtype == np.int32 and (arcs["a"] < arcs["b"]).all()


def test_arcs_from_labels_ignores_background_and_empty():
    lab = np.full((4, 4), -1, np.int32)
    arcs = mf.arcs_from_labels(lab, np)
    assert len(arcs["a"]) == 0
    arcs = mf.arcs_from_labels(np.zeros((0, 0), np.int32), np)
    assert len(arcs["a"]) == 0


def test_index_arcs_drops_ids_missing_from_table():
    arcs = {"a": np.array([0, 2, 7], np.int32), "b": np.array([2, 9, 9], np.int32)}
    ids = np.array([0, 2, 5, 9])
    ia, ib, keep = mf.index_arcs(arcs, ids, np)
    assert keep.tolist() == [True, True, False]
    assert ia.tolist() == [0, 1] and ib.tolist() == [1, 3]


def test_channel_names_base_first():
    t = FeatureTable(["feature_id", "mean_blur_s1.5", "std_blur_s1.5", "mean_base"],
                     np.zeros((1, 4)))
    assert mf.channel_names(t) == ["base", "blur_s1.5"]


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_mean_metric_zero_at_seed_and_zscored():
    t = blocks_table()
    d = mf.node_dissimilarity(t, 0, "mean", ["base"], np)
    assert d[0] == 0.0
    sd = np.std([1.5, 2.5, 3.5, 4.5])
    assert np.allclose(d, [0, 1, 2, 3] / sd)


def test_edge_metric_matches_pairwise_node_metric():
    t = blocks_table()
    ia, ib = np.array([0, 1]), np.array([1, 3])
    d = mf.edge_dissimilarity(t, ia, ib, "mean", ["base"], np)
    sd = np.std([1.5, 2.5, 3.5, 4.5])
    assert np.allclose(d, [1 / sd, 2 / sd])


def test_bhattacharyya_identity_symmetry_monotone():
    t = blocks_table()
    d = mf.node_dissimilarity(t, 0, "bhattacharyya", ["base"], np)
    assert d[0] == 0.0
    assert (d[1:] > 0).all()
    d2 = mf.node_dissimilarity(t, 1, "bhattacharyya", ["base"], np)
    assert math.isclose(d[1], d2[0])            # symmetric
    # region 9 has std 0: floored, finite.
    assert np.isfinite(d).all()


def test_bhattacharyya_needs_std_column():
    t = blocks_table(with_std=False)
    with pytest.raises(ValueError, match="std_base"):
        mf.node_dissimilarity(t, 0, "bhattacharyya", ["base"], np)


def test_anchor_edge_weight_is_max_of_endpoints():
    t = blocks_table()
    ia, ib = np.array([0, 1, 2]), np.array([1, 3, 3])
    d = mf.node_dissimilarity(t, 0, "mean", ["base"], np)
    w = mf.edge_weights(t, ia, ib, 0, "mean", "anchor", ["base"], np)
    assert np.allclose(w, np.maximum(d[ia], d[ib]))


def test_barrier_weights_need_saddles():
    t = blocks_table(with_ext=True)
    ia, ib = np.array([0]), np.array([1])
    with pytest.raises(ValueError, match="saddle"):
        mf.edge_weights(t, ia, ib, 0, "barrier", "anchor", ["base"], np)
    w = mf.edge_weights(t, ia, ib, 0, "barrier", "anchor", ["base"], np,
                        saddle=np.array([2.5]), seed_ext_value=1.0)
    assert w.tolist() == [1.5]


def test_unknown_metric_or_mode_raise():
    t = blocks_table()
    with pytest.raises(ValueError):
        mf.node_dissimilarity(t, 0, "cosine", ["base"], np)
    with pytest.raises(ValueError):
        mf.edge_weights(t, np.array([0]), np.array([1]), 0, "mean", "sideways",
                        ["base"], np)


# --------------------------------------------------------------------------- #
# bottleneck join
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("use_scipy", SCIPY_MODES)
def test_bottleneck_join_hand_graph(use_scipy):
    # 0-1 (1), 1-2 (5), 0-3 (3), 3-2 (2); node 4 isolated. Best route to 2 is
    # 0-3-2 with bottleneck 3, not 0-1-2 with bottleneck 5.
    ia = np.array([0, 1, 0, 3]); ib = np.array([1, 2, 3, 2])
    w = np.array([1.0, 5.0, 3.0, 2.0])
    join = mf.bottleneck_join(5, ia, ib, w, 0, np, use_scipy=use_scipy)
    assert join[:4].tolist() == [0.0, 1.0, 3.0, 3.0]
    assert math.isinf(join[4])


@pytest.mark.parametrize("use_scipy", SCIPY_MODES)
def test_bottleneck_join_zero_weights_survive(use_scipy):
    ia = np.array([0, 1]); ib = np.array([1, 2])
    join = mf.bottleneck_join(3, ia, ib, np.zeros(2), 0, np, use_scipy=use_scipy)
    assert join.tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("use_scipy", SCIPY_MODES)
def test_bottleneck_join_seed_only_when_no_arcs(use_scipy):
    join = mf.bottleneck_join(3, np.zeros(0, np.intp), np.zeros(0, np.intp),
                              np.zeros(0), 1, np, use_scipy=use_scipy)
    assert join[1] == 0.0 and math.isinf(join[0]) and math.isinf(join[2])


@pytest.mark.parametrize("use_scipy", SCIPY_MODES)
def test_bottleneck_join_random_graph_agrees_with_brute_force(use_scipy):
    rng = np.random.default_rng(7)
    n, m = 12, 30
    ia = rng.integers(0, n, m); ib = rng.integers(0, n, m)
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]
    w = rng.random(len(ia))
    join = mf.bottleneck_join(n, ia, ib, w, 0, np, use_scipy=use_scipy)
    # Brute force: threshold sweep + connectivity.
    for v in range(n):
        reach = np.inf
        for t in sorted(set(w.tolist())):
            comp = {0}
            changed = True
            while changed:
                changed = False
                for a, b, ww in zip(ia, ib, w):
                    if ww <= t and ((a in comp) != (b in comp)):
                        comp.add(a); comp.add(b); changed = True
            if v in comp:
                reach = t
                break
        if v == 0:
            reach = 0.0
        assert math.isclose(join[v], reach) or (math.isinf(join[v]) and math.isinf(reach))


# --------------------------------------------------------------------------- #
# ladder
# --------------------------------------------------------------------------- #
def test_build_ladder_on_blocks():
    t = blocks_table()
    arcs = mf.arcs_from_labels(blocks_raster(), np)
    lad = mf.build_ladder(t, arcs, 0, "mean", "anchor", ["base"], np)
    sd = np.std([1.5, 2.5, 3.5, 4.5])
    assert lad.seed_row == 0 and lad.seed_id == 0 and lad.n_reach == 4
    assert np.allclose(lad.join, [0, 1, 2, 3] / sd)
    assert lad.ids[lad.order].tolist() == [0, 2, 5, 9]
    assert lad.cum_area.tolist() == [64, 144, 240, 352]
    assert sorted(mf.regions_at(lad, lad.join[2] + 1e-9, np).tolist()) == [0, 2, 5]
    assert mf.regions_for_rank(lad, 3).tolist() == [0, 2, 5]
    assert mf.regions_for_rank(lad, 0).tolist() == [0]
    assert mf.regions_for_rank(lad, 99).tolist() == [0, 2, 5, 9]
    assert mf.rank_at(lad, -1.0, np) == 1
    assert mf.rank_at(lad, 10.0, np) == 4
    assert mf.rank_at(lad, lad.join[1], np) == 2
    assert mf.rank_at(lad, lad.join[1] + 1e-9, np) == 2
    assert mf.threshold_for_rank(lad, 3) == lad.sorted_join[2]
    assert mf.threshold_for_rank(lad, 99) == lad.sorted_join[3]
    assert mf.threshold_for_rank(lad, 0) == 0.0


def test_build_ladder_chain_and_barrier_modes():
    t = blocks_table(with_ext=True)
    arcs = mf.arcs_from_labels(blocks_raster(), np)
    chain = mf.build_ladder(t, arcs, 0, "mean", "chain", ["base"], np)
    sd = np.std([1.5, 2.5, 3.5, 4.5])
    # 0-2 costs 1, 2-9 costs 2, 0-5 costs 2: minimax to 9 is 2 either way.
    assert np.allclose(chain.join, [0, 1, 2, 2] / sd)
    with pytest.raises(ValueError, match="saddle"):
        mf.build_ladder(t, arcs, 0, "barrier", "anchor", ["base"], np)
    arcs_s = dict(arcs, saddle=np.array([1.2, 3.0, 5.0, 3.5], np.float32))
    bar = mf.build_ladder(t, arcs_s, 0, "barrier", "anchor", ["base"], np)
    # seed ext 1.0: 0-2 |1.2-1|=.2, 0-5 2.0, 2-9 4.0, 5-9 2.5 -> 9 via 5 at 2.5
    assert np.allclose(bar.join, [0.0, 0.2, 2.0, 2.5])


def test_build_ladder_rejects_bad_inputs():
    t = blocks_table()
    arcs = mf.arcs_from_labels(blocks_raster(), np)
    with pytest.raises(ValueError, match="not in the feature table"):
        mf.build_ladder(t, arcs, 7, "mean", "anchor", ["base"], np)
    with pytest.raises(ValueError):
        mf.build_ladder(t, arcs, 0, "nope", "anchor", ["base"], np)
    with pytest.raises(ValueError):
        mf.build_ladder(t, arcs, 0, "mean", "nope", ["base"], np)
    with pytest.raises(ValueError, match="mean_blur"):
        mf.build_ladder(t, arcs, 0, "mean", "anchor", ["blur"], np)


def _ladder_from_join(join, area=None):
    join = np.asarray(join, np.float64)
    order = np.argsort(join, kind="stable")
    cum = None if area is None else np.cumsum(np.asarray(area)[order])
    return mf.Ladder(np.arange(len(join)), join, order, join[order], cum,
                     int(np.isfinite(join).sum()), 0, 0, "mean", "anchor", ["base"])


def test_initial_rank_natural_break_and_clamps():
    # 100 reachable rungs; the first 5% (5 rungs) hold a clear break after
    # rung 3 (0, .1, .12, .11 | 5, 5.2, ...).
    s = [0.0, 0.10, 0.12, 0.11] + [5.0 + 0.2 * i for i in range(96)]
    lad = _ladder_from_join(s)
    assert mf.initial_rank(lad, np) == 4
    # Flat ladder: no break wins outright, still clamped inside the window.
    lad = _ladder_from_join(np.linspace(0, 1, 100))
    k = mf.initial_rank(lad, np)
    assert 1 <= k <= 5
    # Tiny ladders: seed only.
    assert mf.initial_rank(_ladder_from_join([0.0]), np) == 1
    assert mf.initial_rank(_ladder_from_join([0.0, 1.0]), np) == 1
    assert mf.initial_rank(_ladder_from_join([0.0, 1.0, np.inf]), np) == 1


def test_drag_to_rank_monotone_and_clamped():
    ks = [mf.drag_to_rank(10, dy, 1000) for dy in range(-400, 401, 8)]
    assert ks == sorted(ks)
    assert mf.drag_to_rank(10, 0, 1000) == 10
    assert mf.drag_to_rank(10, 4, 1000) == 11
    assert mf.drag_to_rank(10, -4, 1000) == 9
    assert mf.drag_to_rank(10, -10_000, 1000) == 1
    assert mf.drag_to_rank(10, 10_000, 1000) == 1000
    assert mf.drag_to_rank(10, 400, 1000) > 500      # a long drag sweeps far

# --------------------------------------------------------------------------- #
# flood order: ties from an outlier seed grow one region at a time
# --------------------------------------------------------------------------- #
def _connected(ids, arcs):
    ids = set(int(i) for i in ids)
    if len(ids) <= 1:
        return True
    adj = {}
    for a, b in zip(arcs["a"].tolist(), arcs["b"].tolist()):
        if a in ids and b in ids:
            adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    seen, todo = set(), [next(iter(ids))]
    while todo:
        u = todo.pop()
        if u in seen:
            continue
        seen.add(u)
        todo.extend(adj.get(u, []))
    return seen == ids


def _chain_table(means):
    return FeatureTable(["feature_id", "area", "mean_base"],
                        np.array([[i, 10.0, m] for i, m in enumerate(means)]))


def test_outlier_seed_ties_grow_one_region_per_rank():
    # Bright seed 0 on a dark chain: the gateway's dissimilarity (region 3's
    # own, the largest on the path) is the bottleneck for 3, 4 and 5 alike --
    # a threshold-closed set jumps from 3 regions to 6; the flood order does not.
    means = [10.0, 2.0, 1.9, 1.5, 1.8, 2.2]
    t = _chain_table(means)
    arcs = {"a": np.array([0, 1, 2, 3, 4], np.int32),
            "b": np.array([1, 2, 3, 4, 5], np.int32), "saddle": None}
    lad = mf.build_ladder(t, arcs, 0, "mean", "anchor", ["base"], np)
    assert np.isclose(lad.join[3], lad.join[4]) and np.isclose(lad.join[4], lad.join[5])
    assert len(mf.regions_at(lad, mf.threshold_for_rank(lad, 4), np)) == 6   # the tie
    for k in range(1, 7):
        ids = mf.regions_for_rank(lad, k)
        assert len(ids) == k
        assert _connected(ids, arcs)
    assert lad.ids[lad.order].tolist() == [0, 1, 2, 3, 4, 5]
    assert np.all(np.diff(lad.sorted_join) >= 0)
    # rank_at stops at the first tied region rather than admitting the group.
    assert mf.rank_at(lad, mf.threshold_for_rank(lad, 4), np) == 4
    assert mf.rank_at(lad, mf.threshold_for_rank(lad, 3), np) == 3


def test_flood_order_prefers_most_seed_like_among_ties():
    # Seed 0 (bright) -> gateway 1; behind it 2 (dark) and 3 (closer to the
    # seed) tie on join, so 3 must come first; 4 hangs off 2 and follows it.
    means = [10.0, 2.0, 1.0, 1.9, 1.95]
    t = _chain_table(means)
    arcs = {"a": np.array([0, 1, 1, 2], np.int32),
            "b": np.array([1, 2, 3, 4], np.int32), "saddle": None}
    lad = mf.build_ladder(t, arcs, 0, "mean", "anchor", ["base"], np)
    order = lad.ids[lad.order].tolist()
    assert order[:3] == [0, 1, 3], order
    assert order.index(4) > order.index(2)
    for k in range(1, 6):
        assert _connected(mf.regions_for_rank(lad, k), arcs)


def test_growth_order_join_matches_bottleneck_join_and_prefixes_connect():
    rng = np.random.default_rng(11)
    n, m = 40, 120
    ia = rng.integers(0, n, m); ib = rng.integers(0, n, m)
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]
    w = np.round(rng.random(len(ia)), 1)          # coarse -> plenty of ties
    order, join = mf.growth_order(n, ia, ib, w, 0, np)
    ref = mf.bottleneck_join(n, ia, ib, w, 0, np, use_scipy=False)
    assert np.allclose(join[np.isfinite(ref)], ref[np.isfinite(ref)])
    assert np.isinf(join[np.isinf(ref)]).all()
    reach = int(np.isfinite(join).sum())
    assert len(order) == n and len(set(order.tolist())) == n
    assert np.all(np.diff(join[order[:reach]]) >= 0)
    arcs = {"a": ia.astype(np.int32), "b": ib.astype(np.int32)}
    for k in range(1, reach + 1):
        assert _connected(order[:k], arcs)
    # Unreached nodes come last, and a secondary node key changes ties only.
    assert np.isinf(join[order[reach:]]).all()
    order2, join2 = mf.growth_order(n, ia, ib, w, 0, np, node_key=rng.random(n))
    assert np.allclose(join2[np.isfinite(join)], join[np.isfinite(join)])


def test_growth_order_degenerate_inputs():
    order, join = mf.growth_order(0, [], [], [], 0, np)
    assert len(order) == 0
    order, join = mf.growth_order(3, [], [], [], 1, np)
    assert order.tolist() == [1, 0, 2] and join[1] == 0.0 and np.isinf(join[0])
