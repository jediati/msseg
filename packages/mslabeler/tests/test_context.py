"""context.py: neighbourhood context columns -- contact lengths, the ring /
hop / slice reductions under each weighting, augment(), naming and the
compatibility gate. Pure numpy; no Tk, no sklearn, no extension."""
import numpy as np
import pytest

from msseg.labeler import bundle
from msseg.labeler import context as cx
from msseg.labeler import magic_fill
from msseg.labeler.fields import DEFAULT
from msseg.labeler.table import FeatureTable

from test_labeling import blocks_raster
from test_training import table as blocks_table


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def grid_arcs(side=3, extra_isolated=None):
    """A side x side grid of regions with dense ids 0..side*side-1 and
    4-neighbour arcs, listed a<b. `extra_isolated` adds one id with no arcs."""
    a, b = [], []
    for i in range(side * side):
        r, q = divmod(i, side)
        if q + 1 < side:
            a.append(i); b.append(i + 1)
        if r + 1 < side:
            a.append(i); b.append(i + side)
    arcs = {"a": np.asarray(a, np.int32), "b": np.asarray(b, np.int32),
            "saddle": None, "source": "pixels"}
    n = side * side + (1 if extra_isolated is not None else 0)
    return arcs, n


def grid_table(side=3, extra_isolated=None):
    arcs, n = grid_arcs(side, extra_isolated)
    ids = np.arange(n, dtype=float)
    # x = id, y = id squared, area = id + 1
    vals = np.stack([ids, ids + 1.0, ids, ids ** 2], axis=1)
    return FeatureTable(["feature_id", "area", "mean_x", "mean_y"], vals), arcs


def graph_of(table, arcs, length=None):
    ia, ib, keep = magic_fill.index_arcs(arcs, table.column("feature_id"), np)
    ln = None if length is None else np.asarray(length)[keep]
    return cx.directed_graph(ia, ib, table.n_rows, np, ln)


# --------------------------------------------------------------------------- #
# Contact lengths
# --------------------------------------------------------------------------- #
def test_contact_lengths_on_blocks():
    lab = blocks_raster()
    arcs = magic_fill.arcs_from_labels(lab, np)
    pairs = set(zip(arcs["a"].tolist(), arcs["b"].tolist()))
    assert pairs == {(0, 2), (0, 5), (2, 9), (5, 9)}
    ln = cx.contact_lengths(lab, arcs["a"], arcs["b"], np)
    assert ln.tolist() == [8, 8, 8, 8]
    # symmetric in (a, b); the whole 4-neighbour boundary is accounted for
    assert cx.contact_lengths(lab, arcs["b"], arcs["a"], np).tolist() == [8, 8, 8, 8]
    assert ln.sum() == 32
    # a pair that never touches, an unknown id and a self pair read 0
    got = cx.contact_lengths(lab, [0, 0, 5, 7], [9, 2, 5, 0], np)
    assert got.tolist() == [0, 8, 0, 0]


def test_ensure_contact_caches_on_the_arcs_dict():
    lab = blocks_raster()
    arcs = magic_fill.arcs_from_labels(lab, np)
    ln = cx.ensure_contact(arcs, lab, np)
    assert "length" in arcs and arcs["length"] is ln or np.array_equal(arcs["length"], ln)
    arcs["length"][:] = 3
    assert cx.ensure_contact(arcs, lab, np).tolist() == [3, 3, 3, 3]     # cached, not recomputed


def test_contact_lengths_empty():
    assert cx.contact_lengths(np.zeros((0, 0), np.int32), [0], [1], np).tolist() == [0]
    assert cx.contact_lengths(blocks_raster(), [], [], np).shape == (0,)


# --------------------------------------------------------------------------- #
# Ring reductions
# --------------------------------------------------------------------------- #
def test_uniform_ring_mean_on_a_grid():
    t, arcs = grid_table(3, extra_isolated=9)
    g = graph_of(t, arcs)
    X = t.column("mean_x")
    r = cx.ring_reduce(X, g, cx.neighbour_weights("uniform", g, np), ("mean", "min", "max", "std"), np)
    m = r["mean"][:, 0]
    assert m[0] == pytest.approx(2.0)          # corner: {1, 3}
    assert m[1] == pytest.approx(2.0)          # edge: {0, 2, 4}
    assert m[4] == pytest.approx(4.0)          # centre: {1, 3, 5, 7}
    assert m[9] == 9.0                         # isolated: own value
    assert r["min"][0, 0] == 1.0 and r["max"][0, 0] == 3.0
    assert r["std"][0, 0] == pytest.approx(1.0)
    assert r["std"][9, 0] == 0.0
    assert r["min"][9, 0] == 9.0 and r["max"][9, 0] == 9.0


def test_area_and_contact_weights():
    t, arcs = grid_table(3)
    # area = id + 1: corner 0 sees 1 (area 2) and 3 (area 4)
    g = graph_of(t, arcs)
    w = cx.neighbour_weights("area", g, np, t.column("area"))
    m = cx.ring_reduce(t.column("mean_x"), g, w, ("mean",), np)["mean"][:, 0]
    assert m[0] == pytest.approx((1 * 2 + 3 * 4) / 6)
    # contact: give arc (0,1) length 1 and arc (0,3) length 3
    length = np.ones(len(arcs["a"]))
    for i, (a, b) in enumerate(zip(arcs["a"].tolist(), arcs["b"].tolist())):
        if (a, b) == (0, 3):
            length[i] = 3.0
    g = graph_of(t, arcs, length)
    w = cx.neighbour_weights("contact", g, np)
    m = cx.ring_reduce(t.column("mean_x"), g, w, ("mean",), np)["mean"][:, 0]
    assert m[0] == pytest.approx((1 * 1 + 3 * 3) / 4)
    # zero contact everywhere -> a row keeps its own value
    g0 = graph_of(t, arcs, np.zeros(len(arcs["a"])))
    m0 = cx.ring_reduce(t.column("mean_x"), g0, cx.neighbour_weights("contact", g0, np),
                        ("mean",), np)["mean"][:, 0]
    assert m0.tolist() == t.column("mean_x").tolist()


def test_weights_refuse_missing_data():
    t, arcs = grid_table(3)
    g = graph_of(t, arcs)
    with pytest.raises(ValueError):
        cx.neighbour_weights("area", g, np)
    with pytest.raises(ValueError):
        cx.neighbour_weights("contact", g, np)
    with pytest.raises(ValueError):
        cx.neighbour_weights("bogus", g, np)


def test_duplicate_and_reversed_arcs_change_nothing():
    t, arcs = grid_table(3)
    g1 = graph_of(t, arcs)
    twice = {"a": np.concatenate([arcs["a"], arcs["b"], arcs["a"]]),
             "b": np.concatenate([arcs["b"], arcs["a"], arcs["b"]])}
    g2 = graph_of(t, twice)
    assert g2.n_edges == g1.n_edges
    for g in (g1, g2):
        r = cx.ring_reduce(t.values[:, 2:], g, cx.neighbour_weights("uniform", g, np),
                           ("mean", "std"), np)
        if g is g1:
            ref = r
        else:
            assert np.allclose(r["mean"], ref["mean"]) and np.allclose(r["std"], ref["std"])


def test_hop2_is_hop1_applied_twice():
    t, arcs = grid_table(4)
    g = graph_of(t, arcs)
    w = cx.neighbour_weights("uniform", g, np)
    X = t.values[:, 2:]
    m1 = cx.ring_reduce(X, g, w, ("mean",), np)["mean"]
    m2 = cx.ring_reduce(m1, g, w, ("mean",), np)["mean"]
    assert np.allclose(cx.hop2_mean(X, g, w, np), m2)
    assert not np.allclose(m2, m1)


def test_chunked_columns_match_unchunked(monkeypatch):
    t, arcs = grid_table(5)
    g = graph_of(t, arcs)
    w = cx.neighbour_weights("uniform", g, np)
    X = np.random.default_rng(0).standard_normal((t.n_rows, 37))
    full = cx.ring_reduce(X, g, w, ("mean", "min", "max", "std"), np)
    monkeypatch.setattr(cx, "COLUMN_CHUNK", 5)
    small = cx.ring_reduce(X, g, w, ("mean", "min", "max", "std"), np)
    for k in full:
        assert np.allclose(full[k], small[k])


def test_slice_mean():
    X = np.array([[1.0, 10.0], [3.0, 30.0]])
    assert cx.slice_mean(X, np).tolist() == [2.0, 20.0]
    assert cx.slice_mean(X, np, area=[1.0, 3.0]).tolist() == [2.5, 25.0]
    assert cx.slice_mean(X, np, area=[0.0, 0.0]).tolist() == [2.0, 20.0]   # no weight -> plain
    assert cx.slice_mean(np.zeros((0, 2)), np).tolist() == [0.0, 0.0]


# --------------------------------------------------------------------------- #
# Naming and the spec
# --------------------------------------------------------------------------- #
def test_column_names_order_and_schema():
    spec = cx.ContextSpec(kinds=("ring_mean", "slice_contrast"), weights=("uniform", "contact"),
                          source="ext")
    names = ["feature_id", "area", "ext_x", "ext_y", "mean_base", "ext_base", "ext_filtered"]
    got = cx.column_names(spec, names)
    assert got == ["ring_mean__ext_base", "ring_mean__ext_filtered",
                   "ring_mean[contact]__ext_base", "ring_mean[contact]__ext_filtered",
                   "slice_contrast__ext_base", "slice_contrast__ext_filtered"]  # slice: uniform only
    sch = cx.schema_entries(spec, names)
    assert [e["channel"] for e in sch] == ["ring_mean"] * 2 + ["ring_mean[contact]"] * 2 + ["slice_contrast"] * 2
    assert [e["reduction"] for e in sch][:2] == ["ext_base", "ext_filtered"]
    for n in got:
        assert cx.is_context_column(n)
        kind, w, src = cx.split_column(n)
        assert cx.column_name(kind, w, src) == n
    assert not cx.is_context_column("mean_base") and cx.split_column("ext_base") is None
    assert cx.column_names(cx.ContextSpec(), names) == []
    assert cx.column_names(cx.ContextSpec(kinds=("ring_mean",), source="mean"), names) == ["ring_mean__mean_base"]


def test_context_names_never_look_like_channels_or_bins():
    spec = cx.ContextSpec(kinds=cx.KINDS, weights=cx.WEIGHTS)
    names = ["feature_id", "area", "mean_base", "std_base", "hist00_base", "ext_filtered"]
    t = FeatureTable(names, np.zeros((0, len(names))))
    aug = FeatureTable(names + cx.column_names(spec, names), None)
    assert DEFAULT.channel_names(aug) == DEFAULT.channel_names(t) == ["base"]
    assert DEFAULT.histogram_columns(aug) == {"base": ["hist00_base"]}
    for n in cx.column_names(spec, names):
        assert n not in DEFAULT.positional


def test_spec_round_trip_and_tolerance():
    spec = cx.ContextSpec(kinds=("ring_std", "hop2_mean"), weights=("contact",), source="mean")
    back = cx.ContextSpec.from_dict(spec.to_dict())
    assert back == spec and back.key() == spec.key()
    tol = cx.ContextSpec.from_dict({"kinds": ["ring_mean", "nope", "ring_mean"],
                                    "weights": ["latent"], "source": "?", "junk": 1})
    assert tol.kinds == ("ring_mean",) and tol.weights == ("uniform",) and tol.source == "all"
    assert cx.ContextSpec.from_dict(None) == cx.ContextSpec()
    assert cx.ContextSpec.from_dict(None).empty() and not spec.empty()
    assert cx.ContextSpec().to_dict() == {"kinds": [], "weights": ["uniform"], "source": "all"}
    assert cx.ContextSpec().brief() == "" and cx.ContextSpec().describe() == "no context features"
    assert spec.brief() == "ctx: ring(std) +hop2(mean) [contact] on mean"
    assert cx.ContextSpec(kinds=("ring_mean", "ring_contrast")).brief() == "ctx: ring(mean,contrast)"
    assert spec.describe(["feature_id", "mean_base", "mean_blur"]).endswith("; 4 column(s)")
    assert spec.key() != cx.ContextSpec(kinds=("ring_std",)).key()


# --------------------------------------------------------------------------- #
# augment
# --------------------------------------------------------------------------- #
def test_augment_empty_spec_is_identity():
    t = blocks_table()
    assert cx.augment(t, None, cx.ContextSpec(), np) is t
    assert cx.augment(t, None, None, np) is t
    # a spec whose source selects nothing adds nothing
    t2 = FeatureTable(["feature_id", "area"], np.array([[0.0, 1.0]]))
    assert cx.augment(t2, None, cx.ContextSpec(kinds=("ring_mean",), source="ext"), np) is t2


def test_augment_blocks_from_arcs_and_from_labels():
    t = blocks_table()
    lab = blocks_raster()
    spec = cx.ContextSpec(kinds=("ring_mean", "ring_contrast"), source="ext")
    arcs = magic_fill.arcs_from_labels(lab, np)
    a1 = cx.augment(t, arcs, spec, np)
    a2 = cx.augment(t, None, spec, np, labels=lab)
    assert a1 is not t and a1.names == list(t.names) + ["ring_mean__ext_filtered",
                                                       "ring_contrast__ext_filtered"]
    assert np.array_equal(a1.values, a2.values)
    assert np.array_equal(a1.values[:, :len(t.names)], t.values)      # base block untouched
    assert t.names == list(blocks_table().names)                        # the input is not mutated
    # ext_filtered = fid / 10; region 0 touches 2 and 5 -> mean 0.35, contrast -0.35
    assert a1.column("ring_mean__ext_filtered")[0] == pytest.approx(0.35)
    assert a1.column("ring_contrast__ext_filtered")[0] == pytest.approx(-0.35)
    # region 9 touches 2 and 5 too
    assert a1.column("ring_contrast__ext_filtered")[3] == pytest.approx(0.9 - 0.35)
    assert a1.values.dtype == np.float64


def test_augment_every_kind_and_weight_on_blocks():
    t = blocks_table()
    lab = blocks_raster()
    arcs = magic_fill.arcs_from_labels(lab, np)
    spec = cx.ContextSpec(kinds=cx.KINDS, weights=cx.WEIGHTS)
    aug = cx.augment(t, arcs, spec, np, labels=lab)
    assert aug.names[len(t.names):] == cx.column_names(spec, t.names)
    assert np.isfinite(aug.values).all()
    assert "length" in arcs                          # contact lengths were derived and cached
    # all four blocks have equal area and equal contact, so the three
    # weightings agree on the ring mean here
    for w in ("area", "contact"):
        assert np.allclose(aug.column(f"ring_mean[{w}]__mean_base"), aug.column("ring_mean__mean_base"))
    # slice_mean is one number per column; slice_contrast sums to zero
    assert np.allclose(aug.column("slice_mean__mean_base"), t.column("mean_base").mean())
    assert aug.column("slice_contrast__mean_base").sum() == pytest.approx(0.0)
    # augmenting the augmented table again adds nothing
    assert cx.augment(aug, arcs, spec, np) is aug


def test_augment_refuses_without_a_graph_or_contact_data():
    t = blocks_table()
    with pytest.raises(ValueError):
        cx.augment(t, None, cx.ContextSpec(kinds=("ring_mean",)), np)
    arcs = magic_fill.arcs_from_labels(blocks_raster(), np)
    with pytest.raises(ValueError):
        cx.augment(t, arcs, cx.ContextSpec(kinds=("ring_mean",), weights=("contact",)), np)
    # slice kinds need no graph at all
    aug = cx.augment(t, None, cx.ContextSpec(kinds=("slice_mean",)), np)
    assert aug.names[-1] == "slice_mean__ext_filtered"


def test_augment_ignores_arcs_naming_unknown_ids_and_nonfinite_values():
    t = blocks_table()
    vals = t.values.copy()
    vals[1, t.names.index("mean_base")] = np.nan
    t = FeatureTable(list(t.names), vals)
    arcs = {"a": np.array([0, 0, 77], np.int32), "b": np.array([2, 5, 0], np.int32),
            "saddle": None, "source": "pixels"}
    aug = cx.augment(t, arcs, cx.ContextSpec(kinds=("ring_mean",), source="mean"), np)
    # region 0's ring is {2, 5}; region 2's mean_base (NaN) counts as 0
    assert aug.column("ring_mean__mean_base")[0] == pytest.approx((0.0 + 5.0) / 2)
    assert np.isfinite(aug.column("ring_mean__mean_base")).all()


# --------------------------------------------------------------------------- #
# The compatibility gate sees context columns as ordinary columns
# --------------------------------------------------------------------------- #
def test_compat_gate_with_context_columns():
    base = ["area", "mean_base", "ext_filtered"]
    spec = cx.ContextSpec(kinds=("ring_mean",))
    names = base + cx.column_names(spec, base)
    msg = bundle.compat_message(names, base, "p", "load")
    assert msg is not None and "ring_mean__area" in msg
    assert bundle.compat_message(names, base + cx.column_names(spec, base), "p", "load") is None
    other = base + cx.column_names(cx.ContextSpec(kinds=("ring_std",)), base)
    assert bundle.compat_message(names, other, "p", "load") is not None


# --------------------------------------------------------------------------- #
# The latent ring: H0 shape, latent columns, the head
# --------------------------------------------------------------------------- #
def star_graph(n_leaves):
    """Node 0 joined to every other node (a ring of n_leaves around it)."""
    a = np.zeros(n_leaves, np.intp)
    b = np.arange(1, n_leaves + 1, dtype=np.intp)
    return cx.directed_graph(a, b, n_leaves + 1, np)


def test_ring_h0_outlier_and_uniform_rings():
    # Node 0's ring is a tight cluster of four at distance ~10 from it: the
    # single-linkage attach of 0 is the largest merge, the ratio is big, and
    # the components count sees one split at z = 1.
    emb = np.array([[10.0, 0.0], [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]])
    g = star_graph(4)
    h = cx.ring_h0(emb, g, np, z=1.0)
    largest, ratio, attach, comps = h[0]
    assert attach == pytest.approx(9.9, abs=0.2)
    assert largest == pytest.approx(attach)
    assert ratio > 50
    # In a pure star every arc is an outlier-to-cluster edge, so the slice's
    # typical arc distance IS 9.9 and no merge stands out. With the cluster
    # wired up too (the usual case: most arcs join like regions) the outlier
    # merge is far above mean + z * std and the ring splits in two.
    a = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
    b = np.array([1, 2, 3, 4, 2, 3, 4, 3, 4, 4])
    g2 = cx.directed_graph(a, b, 5, np)
    assert cx.ring_h0(emb, g2, np, z=1.0)[0, 3] == 2
    assert comps == 1
    # A uniform ring: all merges alike, ratio ~1, one component.
    emb2 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    h2 = cx.ring_h0(emb2, g, np, z=1.0)
    assert h2[0, 0] == pytest.approx(1.0) and h2[0, 1] == pytest.approx(1.0)
    assert h2[0, 2] == pytest.approx(1.0) and h2[0, 3] == 1
    # A leaf sees only the centre: one merge, ratio 1, attach = its distance.
    assert h2[1, 0] == pytest.approx(1.0) and h2[1, 1] == 1.0 and h2[1, 2] == pytest.approx(1.0)
    # No neighbours at all: zeros with components 1 and ratio 1.
    g0 = cx.directed_graph([], [], 3, np)
    assert cx.ring_h0(np.zeros((3, 2)), g0, np).tolist() == [[0, 1, 0, 1]] * 3


def _prim_reference(D):
    n = len(D); in_tree = [0]; merges = []
    while len(in_tree) < n:
        best = (np.inf, None)
        for i in in_tree:
            for j in range(n):
                if j not in in_tree and D[i, j] < best[0]:
                    best = (D[i, j], j)
        merges.append(best[0]); in_tree.append(best[1])
    return sorted(merges, reverse=True)


def test_ring_h0_matches_a_reference_mst_and_caps_degree():
    rng = np.random.default_rng(3)
    emb = rng.standard_normal((9, 3))
    g = star_graph(8)
    h = cx.ring_h0(emb, g, np, z=0.5)
    pts = emb[[0] + list(range(1, 9))]
    D = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    ref = _prim_reference(D)
    assert h[0, 0] == pytest.approx(ref[0]) and h[0, 1] == pytest.approx(ref[0] / ref[1])
    assert h[0, 2] == pytest.approx(D[0, 1:].min())
    # Capped at 3 neighbours: the ring keeps the three nearest.
    hc = cx.ring_h0(emb, g, np, max_degree=3)
    near = np.argsort(D[0, 1:])[:3] + 1
    pts3 = emb[[0] + near.tolist()]
    D3 = np.sqrt(((pts3[:, None] - pts3[None]) ** 2).sum(-1))
    ref3 = _prim_reference(D3)
    assert hc[0, 0] == pytest.approx(ref3[0]) and hc[0, 2] == pytest.approx(D3[0, 1:].min())


def test_latent_columns_names_weights_and_slices():
    t, arcs = grid_table(3)
    g = graph_of(t, arcs)
    emb = np.stack([t.column("mean_x"), -t.column("mean_x")], axis=1)
    spec = cx.LatentSpec(weight="uniform", h0=True)
    names = cx.latent_names(spec, 2)
    assert names == ["lat_mean__h0", "lat_mean__h1", "h0_largest", "h0_ratio", "h0_attach", "h0_components"]
    L = cx.latent_columns(emb, g, spec, np)
    assert L.shape == (9, 6)
    assert L[0, 0] == pytest.approx(2.0) and L[0, 1] == pytest.approx(-2.0)   # corner ring {1,3}
    assert cx.latent_names(cx.LatentSpec(weight="contact", h0=False), 3) == \
        ["lat_mean[contact]__h0", "lat_mean[contact]__h1", "lat_mean[contact]__h2"]
    # latent weights: a softmax over -d / tau -- the closer neighbour counts more.
    Ls = cx.latent_columns(emb, g, cx.LatentSpec(weight="latent", h0=False), np)
    # corner 0: neighbours 1 (d = sqrt 2) and 3 (d = 3 sqrt 2) -> mean pulled toward 1
    assert Ls[0, 0] < 2.0
    # per-slice scale: two identical slices give identical columns
    row_slice = np.array([0] * 9)
    assert np.allclose(cx.latent_columns(emb, g, cx.LatentSpec(weight="latent", h0=True), np,
                                         row_slice=row_slice), 
                       cx.latent_columns(emb, g, cx.LatentSpec(weight="latent", h0=True), np))
    # area / contact weights want their data
    with pytest.raises(ValueError):
        cx.latent_columns(emb, g, cx.LatentSpec(weight="contact"), np)
    La = cx.latent_columns(emb, g, cx.LatentSpec(weight="area", h0=False), np, area=t.column("area"))
    assert La[0, 0] == pytest.approx((1 * 2 + 3 * 4) / 6)


def test_context_spec_carries_a_latent_block():
    spec = cx.ContextSpec(latent=cx.LatentSpec(weight="latent", z=0.5))
    assert not spec.empty() and spec.kinds == ()
    back = cx.ContextSpec.from_dict(spec.to_dict())
    assert back == spec and back.latent.weight == "latent" and back.latent.z == 0.5
    assert spec.brief() == "ctx: +latent(mean[latent],h0)"
    assert cx.ContextSpec(kinds=("ring_mean",), latent=cx.LatentSpec()).brief() == \
        "ctx: ring(mean) +latent(mean,h0)"
    assert cx.column_names(spec, ["area", "mean_base"]) == []      # not in the fingerprint
    assert cx.augment(blocks_table(), None, spec, np) is blocks_table() or \
        cx.augment(blocks_table(), None, spec, np).names == list(blocks_table().names)
    tol = cx.LatentSpec.from_dict({"weight": "nope", "layer": "x", "z": None, "max_degree": 0})
    assert tol == cx.LatentSpec(max_degree=1)
    assert cx.ContextSpec.from_dict({"latent": None}).latent is None


def test_latent_head_fit_predict_and_round_trip(tmp_path):
    pytest.importorskip("sklearn")
    import pickle
    from msseg.labeler import model_search as ms
    from msseg.labeler import edge_model as em
    from test_edge_model import lattice, region_net, NAMES
    X, cls, grp, ext, per_slice = lattice(n_slices=4, side=5)
    base = region_net(X, cls)
    edges = em.gather_edges(per_slice)
    g = cx.directed_graph(edges["a"], edges["b"], edges["n_rows"], np)
    spec = cx.LatentSpec(weight="uniform", h0=True)

    def make_head(all_names):
        est = ms.build_estimator(ms.ModelSpec(hidden=(8,), backend="sklearn", max_iter=200), all_names)
        return est, ms.fit_estimator

    model = cx.fit_latent_head(base, X, cls, g, spec, NAMES, make_head, np, row_slice=grp)
    assert model.width == 8 and model.n_in == 6 + 8 + 4
    assert model.names[:6] == NAMES and model.names[-4:] == list(cx.H0_NAMES)
    assert model.names_hash == em.names_hash(NAMES) and model.net_hash == em.net_hash(base)
    P, classes = cx.predict_latent(model, base, X, g, np, row_slice=grp)
    assert P.shape == (len(X), 2) and set(classes.tolist()) == {1, 2}
    lab = cls > 0
    assert (classes[P[lab].argmax(1)] == cls[lab]).mean() > 0.8
    # round trip through a pickled dict
    d = pickle.loads(pickle.dumps(model.to_dict()))
    back = cx.LatentContextModel.from_dict(d)
    P2, _c = cx.predict_latent(back, base, X, g, np, row_slice=grp)
    assert np.allclose(P, P2) and "latent head" in back.describe()
    # another base net is refused
    other = region_net(X, cls, hidden=(8, 4))
    with pytest.raises(ValueError):
        cx.predict_latent(model, other, X, g, np)
    # one labeled class only
    one = cls.copy(); one[one == 2] = 1
    with pytest.raises(ValueError):
        cx.fit_latent_head(base, X, one, g, spec, NAMES, make_head, np)


# --------------------------------------------------------------------------- #
# Labels as context
# --------------------------------------------------------------------------- #
def test_label_columns_fractions_self_mask_and_isolated_rows():
    from msseg.labeler.labeling import MAX_CLASSES
    t, arcs = grid_table(3, extra_isolated=9)
    g = graph_of(t, arcs)
    w = cx.neighbour_weights("uniform", g, np)
    # classes: 0 -> 1, 1 -> 2, 3 -> 2, the rest unlabeled; node 9 isolated + labeled
    rc = np.zeros(10, int); rc[0] = 1; rc[1] = 2; rc[3] = 2; rc[9] = 1
    L = cx.label_columns(rc, g, w, np)
    assert L.shape == (10, MAX_CLASSES)
    # corner 0 sees {1, 3}: both class 2 -> c1 0, c2 1, any 1 -- never its own class 1
    assert L[0, 0] == 0.0 and L[0, 1] == 1.0 and L[0, -1] == 1.0
    # node 4 (centre) sees {1, 3, 5, 7}: two of four labeled, both class 2
    assert L[4, 1] == pytest.approx(0.5) and L[4, -1] == pytest.approx(0.5) and L[4, 0] == 0.0
    # node 1 sees {0, 2, 4}: one class-1 neighbour
    assert L[1, 0] == pytest.approx(1 / 3) and L[1, 1] == 0.0
    # the isolated labeled node reads 0 everywhere, not its own indicator
    assert L[9].tolist() == [0.0] * MAX_CLASSES
    assert (L >= 0).all() and (L <= 1).all()


def test_visible_classes_dropout_is_seeded_and_off_at_prediction():
    rc = np.array([1, 2, 1, 2, 1, 2, 0, 0, 1, 2] * 10)
    spec = cx.LabelSpec(dropout=0.5, seed=3)
    assert np.array_equal(cx.visible_classes(rc, spec, np), rc)          # no rng: everything shows
    a = cx.visible_classes(rc, spec, np, np.random.default_rng(3))
    b = cx.visible_classes(rc, spec, np, np.random.default_rng(3))
    assert np.array_equal(a, b) and (a == 0).sum() > (rc == 0).sum()
    assert ((a == 0) | (a == rc)).all(), "dropout only hides, never relabels"
    hidden = ((rc > 0) & (a == 0)).mean()
    assert 0.3 < hidden / (rc > 0).mean() < 0.7
    assert np.array_equal(cx.visible_classes(rc, cx.LabelSpec(dropout=0.0), np,
                                             np.random.default_rng(0)), rc)


def test_augment_with_labels_block_names_gate_and_dropout():
    from msseg.labeler.labeling import MAX_CLASSES
    from msseg.labeler.training import TrainingSetBuilder
    from test_training import store_two_classes
    t = blocks_table()
    lab = blocks_raster()
    store = store_two_classes()
    rc = TrainingSetBuilder().row_classes(store.for_slice("k0"), lab, t.column("feature_id"), np)
    assert rc.tolist() == [2, 2, 1, 1]                 # box -> 1, squiggle over {0, 2} -> 2
    spec = cx.ContextSpec(labels=cx.LabelSpec(dropout=0.0))
    names = cx.column_names(spec, t.names)
    assert names == [f"nbr_class__c{k}" for k in range(1, MAX_CLASSES)] + ["nbr_class__any"]
    assert [e["channel"] for e in cx.schema_entries(spec, t.names)] == ["nbr_class"] * MAX_CLASSES
    assert all(cx.is_context_column(n) for n in names) and cx.split_column(names[-1]) == \
        ("nbr_class", "uniform", "any")
    assert spec.brief() == "ctx: +labels(p=0)" and not spec.empty() and spec.needs_graph()
    with pytest.raises(ValueError):
        cx.augment(t, None, spec, np, labels=lab)                         # needs row classes
    aug = cx.augment(t, None, spec, np, labels=lab, extra={"row_classes": rc})
    assert aug.names[len(t.names):] == names
    # region 0 (class 2) touches 2 (class 2) and 5 (class 1): half and half, all labeled
    assert aug.column("nbr_class__c1")[0] == pytest.approx(0.5)
    assert aug.column("nbr_class__c2")[0] == pytest.approx(0.5)
    assert aug.column("nbr_class__any")[0] == pytest.approx(1.0)
    # with a contact weighting the block gets its own column set too
    two = cx.ContextSpec(labels=cx.LabelSpec(), weights=("uniform", "contact"))
    assert cx.column_names(two, t.names)[MAX_CLASSES] == "nbr_class[contact]__c1"
    # the gate treats the block as ordinary columns
    base = ["area", "mean_base", "std_base", "ext_filtered"]
    assert bundle.compat_message(base + names, base, "p", "t") is not None
    assert bundle.compat_message(base + names, base + cx.column_names(spec, base), "p", "t") is None
    # dropout at training: a seeded rng hides labels; a p = 1-ish draw empties the block
    full = cx.ContextSpec(labels=cx.LabelSpec(dropout=0.95))
    hidden = cx.augment(t, None, full, np, labels=lab,
                        extra={"row_classes": rc, "rng": np.random.default_rng(1)})
    assert hidden.column("nbr_class__any").sum() < aug.column("nbr_class__any").sum()
    # the spec round-trips with its label block
    back = cx.ContextSpec.from_dict(full.to_dict())
    assert back == full and cx.LabelSpec.from_dict({"dropout": 7, "seed": "x"}) == cx.LabelSpec(0.95, 0)
