"""represent: the one place region embeddings are chosen, resolved, cached
and compared -- parity with the per-tool code it replaces, the layer
convention, options, the cache, similarity and the inspection helpers."""
import numpy as np
import pytest

from msseg.labeler import edge_model as em
from msseg.labeler import magic_fill as mf
from msseg.labeler import represent as rp
from msseg.labeler.table import FeatureTable

NAMES = [f"mean_c{i}" for i in range(6)]


def stats_table(n=40, seed=0):
    rng = np.random.default_rng(seed)
    names = (["feature_id", "area", "min_x", "ext_x"] + NAMES
             + ["std_c0", "std_c1", "hist00_c0", "hist01_c0",
                "ring_mean__mean_c0", "ring_mean[area]__mean_c1",
                "fm0123abcd__mean_00", "fm0123abcd__mean_01"])
    vals = rng.standard_normal((n, len(names)))
    vals[:, 0] = np.arange(n) * 2                          # sparse ids
    vals[:, 1] = rng.integers(5, 50, n)
    vals[:, 10:12] = np.abs(vals[:, 10:12])                 # the stds
    h = rng.random((n, 2))
    vals[:, 12:14] = h / h.sum(1, keepdims=True)
    return FeatureTable(names, vals)


def region_net(hidden=(16, 8), backend="sklearn", seed=0):
    pytest.importorskip("sklearn")
    from msseg.labeler import model_search as ms
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((200, 6))
    cls = np.where(X[:, 0] > 0, 1, 2)
    est = ms.build_estimator(ms.ModelSpec(hidden=hidden, backend=backend, max_iter=200), NAMES)
    return ms.fit_estimator(est, X, cls), X


class Stack:
    def __init__(self, clf, names, context=None, latent=None):
        self.clf, self.names, self.context, self.latent = clf, names, context, latent


# --------------------------------------------------------------------------- #
# the reference and the layer convention
# --------------------------------------------------------------------------- #
def test_ref_round_trip_and_defaults():
    r = rp.EmbeddingRef()
    assert r.to_dict() == {} and rp.EmbeddingRef.from_dict({}) == r
    r2 = rp.EmbeddingRef(source="net", layer=2, provider="t_abc123")
    assert rp.EmbeddingRef.from_dict(r2.to_dict()) == r2
    r3 = rp.EmbeddingRef(block="channels", channels=("base", "blur"))
    assert r3.to_dict() == {"block": "channels", "channels": ["base", "blur"]}
    assert rp.EmbeddingRef.from_dict(r3.to_dict()) == r3
    assert rp.EmbeddingRef.from_dict({"layer": 1, "junk": 3}).layer == 1
    with pytest.raises(ValueError):
        rp.EmbeddingRef(source="nope")
    assert not rp.EmbeddingRef().cross_item
    assert rp.EmbeddingRef(block="histogram").cross_item
    assert rp.EmbeddingRef(source="net").cross_item
    assert rp.EmbeddingRef(block="family", family="fm0123abcd", norm="none").cross_item


def test_layer_conventions():
    assert rp.legacy_layer(-1, 3) is None
    assert rp.legacy_layer(-2, 3) == 2
    assert rp.legacy_layer(0, 3) == 1 and rp.legacy_layer(2, 3) == 3
    with pytest.raises(ValueError):
        rp.legacy_layer(-3, 1)
    assert rp.to_legacy_layer(None) == -1 and rp.to_legacy_layer(2) == 1
    assert rp.resolve_layer(None, 3) == 3
    assert rp.resolve_layer(0, 3) == 0 and rp.resolve_layer(2, 3) == 2
    with pytest.raises(ValueError, match="h4 was picked"):
        rp.resolve_layer(4, 3)
    with pytest.raises(ValueError, match="no hidden layer"):
        rp.resolve_layer(None, 0)


# --------------------------------------------------------------------------- #
# model introspection
# --------------------------------------------------------------------------- #
def test_layers_and_embed_layers_match_edge_model():
    est, X = region_net()
    assert rp.layers_of(est) == [(0, 6, "input"), (1, 16, "h1"), (2, 8, "h2")]
    assert rp.depth_of(est) == 2
    layers = em.embed_layers(est, X)
    assert [l.shape[1] for l in layers] == [6, 16, 8]
    for a, b in zip(layers[1:], em.embed(est, X)):
        assert np.array_equal(a, b)
    # The input layer is the scaler's output (FeatureSubset + StandardScaler).
    h0 = em.embed_layers(est, X, upto=0)
    assert len(h0) == 1 and np.allclose(h0[0].mean(0), 0, atol=1e-5)
    assert len(em.embed_layers(est, X, upto=1)) == 2


def test_layers_of_a_forest_and_nothing():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    y = (X[:, 0] > 0).astype(int) + 1
    forest = make_pipeline(StandardScaler(), RandomForestClassifier(5)).fit(X, y)
    assert rp.layers_of(forest) == [(0, 4, "input")]
    assert rp.depth_of(forest) == 0
    assert rp.layers_of(RandomForestClassifier(5).fit(X, y)) == []
    assert em.embed_layers(forest, X, upto=0)[0].shape == (50, 4)


def test_torch_layers():
    pytest.importorskip("torch")
    est, X = region_net(hidden=(8, 4), backend="torch")
    assert [k for k, _w, _n in rp.layers_of(est)] == [0, 1, 2]
    for a, b in zip(em.embed_layers(est, X)[1:], em.embed(est, X)):
        assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# families
# --------------------------------------------------------------------------- #
def test_families_and_row():
    t = stats_table()
    fams = rp.families(t.names)
    assert fams["hist"] == ["hist00_c0", "hist01_c0"]
    assert fams["ring_mean"] == ["ring_mean__mean_c0"]
    assert fams["ring_mean[area]"] == ["ring_mean[area]__mean_c1"]
    assert fams["fm0123abcd"] == ["fm0123abcd__mean_00", "fm0123abcd__mean_01"]
    assert "feature_id" not in sum(fams.values(), []) and "min_x" not in fams["stats"]
    assert "area" in fams["stats"]
    row = rp.row_names(t.names)
    assert "hist00_c0" not in row and "ext_x" not in row and "ring_mean__mean_c0" in row


# --------------------------------------------------------------------------- #
# resolve: parity with the magic fill's vectors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("block,metric,chans", [
    ("row", "cosine", []), ("channels", "mean", ["c0", "c2"]),
    ("gauss", "bhattacharyya", ["c0", "c1"]), ("histogram", "histogram", ["c0"])])
def test_columns_parity(block, metric, chans):
    t = stats_table()
    X, kind = mf.row_vectors(t, metric, chans, np)
    e = rp.resolve(rp.EmbeddingRef(block=block, channels=tuple(chans)), table=t, np=np)
    assert np.array_equal(e.Z, X.astype(np.float32)) and e.distance == kind
    assert e.dim == len(e.names)
    assert np.array_equal(e.ids, np.arange(len(X)) * 2)


def test_family_and_names_blocks():
    t = stats_table()
    e = rp.resolve(rp.EmbeddingRef(block="family", family="fm0123abcd", norm="none"),
                   table=t, np=np)
    assert np.array_equal(e.Z, t.values[:, -2:].astype(np.float32))
    z = rp.resolve(rp.EmbeddingRef(block="names", names=("mean_c0", "mean_c1")), table=t, np=np)
    assert np.allclose(z.Z.mean(0), 0, atol=1e-5) and np.allclose(z.Z.std(0), 1, atol=1e-4)
    with pytest.raises(ValueError, match="lacks"):
        rp.resolve(rp.EmbeddingRef(block="names", names=("nope",)), table=t, np=np)
    with pytest.raises(ValueError):
        rp.resolve(rp.EmbeddingRef(block="family", family="emb99999999"), table=t, np=np)


def test_net_and_proba_resolve():
    est, _X = region_net()
    t = stats_table()
    st = Stack(est, NAMES)
    e = rp.resolve(rp.EmbeddingRef(source="net"), table=t, np=np, stack=st)
    X = t.values[:, 4:10]
    assert np.allclose(e.Z, em.embed(est, X)[-1]) and e.names[0] == "h2_00"
    e1 = rp.resolve(rp.EmbeddingRef(source="net", layer=1), table=t, np=np, stack=st)
    assert e1.dim == 16
    e0 = rp.resolve(rp.EmbeddingRef(source="net", layer=0), table=t, np=np, stack=st)
    assert e0.dim == 6 and e0.names[0] == "input_00"
    with pytest.raises(ValueError, match="h3"):
        rp.resolve(rp.EmbeddingRef(source="net", layer=3), table=t, np=np, stack=st)
    with pytest.raises(ValueError, match="lacks"):
        rp.resolve(rp.EmbeddingRef(source="net"), table=t, np=np, stack=Stack(est, ["nope"] * 6))
    with pytest.raises(ValueError, match="no trained"):
        rp.resolve(rp.EmbeddingRef(source="net"), table=t, np=np, stack=None)
    with pytest.raises(ValueError, match="Classify"):
        rp.resolve(rp.EmbeddingRef(source="proba"), table=t, np=np)
    P = np.zeros((100, 4), np.float32)
    P[::2, 1] = 1.0
    pr = rp.resolve(rp.EmbeddingRef(source="proba"), table=t, np=np, pred_entry=(0, None, P))
    assert pr.Z.shape == (40, 4) and pr.distance == "tv" and (pr.Z[:, 1] == 1).all()


def test_head_needs_its_input():
    est, _X = region_net()
    t = stats_table()

    class Latent:
        head = est
    st = Stack(est, NAMES, latent=Latent())
    with pytest.raises(ValueError, match="graph"):
        rp.resolve(rp.EmbeddingRef(source="head"), table=t, np=np, stack=st)
    e = rp.resolve(rp.EmbeddingRef(source="head", layer=1), table=t, np=np, stack=st,
                   head_input=lambda m: m)
    assert e.dim == 16


# --------------------------------------------------------------------------- #
# options
# --------------------------------------------------------------------------- #
def test_options_populate_from_table_and_model():
    t = stats_table()
    est, _X = region_net()
    opts = rp.options(t.names, stack=Stack(est, NAMES), pred_ok=True,
                      providers=[("t_000001", "glands", Stack(est, NAMES), None),
                                 ("t_000002", "bubbles", Stack(est, ["zz"] * 6), None),
                                 ("t_000003", "blocked", Stack(est, NAMES), "measured at L4")])
    labels = [o.label for o in opts]
    assert labels[0].startswith("row") and any(l.startswith("histograms") for l in labels)
    assert any("foundation fm0123abcd" in l for l in labels)
    assert any("context ring_mean[area]" in l for l in labels)
    net = [o for o in opts if o.ref.source == "net" and o.ref.provider is None]
    assert [o.ref.layer for o in net] == [0, 1, None]           # the last is 'last'
    assert net[-1].label == "h2 · 8-d (region net, last)"
    assert net[0].label == "input · 6-d (region net, scaled input)"
    assert all(o.usable for o in net)
    assert any(o.ref.source == "proba" for o in opts)
    other = [o for o in opts if o.ref.provider == "t_000001"]
    assert other and all(o.usable and o.group == "other tasks" for o in other)
    assert all(not o.usable for o in opts if o.ref.provider == "t_000002")
    assert all(o.why_not == "measured at L4" for o in opts if o.ref.provider == "t_000003")
    # Every usable statistics/model option resolves.
    for o in opts:
        if o.usable and o.ref.provider is None and o.ref.source in ("columns", "net"):
            rp.resolve(o.ref, table=t, np=np, stack=Stack(est, NAMES))
    assert rp.option_for(net[1].ref, opts) is net[1]
    assert rp.options(t.names, stack=None)[0].ref == rp.EmbeddingRef()


# --------------------------------------------------------------------------- #
# the cache
# --------------------------------------------------------------------------- #
def test_cache_hits_and_misses():
    est, _X = region_net()
    t = stats_table()
    st = Stack(est, NAMES)
    c = rp.EmbeddingCache()
    ref = rp.EmbeddingRef(source="net")
    a = c.resolve("item", ref, commit=7, table=t, np=np, stack=st)
    b = c.resolve("item", ref, commit=7, table=t, np=np, stack=st)
    assert a is b and c.hits == 1 and c.misses == 1
    c.resolve("item", ref, commit=8, table=t, np=np, stack=st)       # another record
    c.resolve("item", rp.EmbeddingRef(source="net", layer=1), commit=7, table=t, np=np, stack=st)
    assert c.misses == 3
    est2, _ = region_net(seed=3)                                        # another net
    c.resolve("item", ref, commit=7, table=t, np=np, stack=Stack(est2, NAMES))
    assert c.misses == 4
    assert c.resolve("item", ref, commit=7, table=t, np=np, stack=st) is a


# --------------------------------------------------------------------------- #
# comparing
# --------------------------------------------------------------------------- #
def test_distance_parity_and_l2n():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4))
    I, J = np.arange(10), np.arange(10)[::-1]
    for kind in ("euclidean", "cosine"):
        assert np.array_equal(rp.distance(X, I, J, kind, np), mf.pairwise(X, I, J, kind, np))
    d = rp.distance(X, I, J, "l2n", np)
    assert (d >= 0).all() and (d <= 2 + 1e-9).all()
    assert np.allclose(rp.distance(X * 7.0, I, J, "l2n", np), d)


def test_similarity_and_top_k():
    Z = np.array([[1, 0], [0.9, 0.1], [0, 1], [-1, 0]], np.float32)
    s = rp.similarity(Z, Z[0], "cosine", np)
    assert s[0] == pytest.approx(1.0) and s.argmin() == 3
    assert rp.similarity(Z, Z[[0, 2]], "cosine", np, "max")[2] == pytest.approx(1.0)
    assert rp.similarity(Z, Z[[0, 2]], "cosine", np, "mean")[2] == pytest.approx(0.5)
    assert np.allclose(rp.similarity(Z, Z[0], "dot", np), Z @ Z[0])
    e = rp.similarity(Z, Z[0], "euclidean", np)
    assert e[0] == pytest.approx(0.0, abs=1e-6) and e.argmin() == 3
    u = rp.to_unit(rp.similarity(Z, Z[0], "dot", np), "dot", np)
    assert u.min() == 0 and u.max() == 1
    rows, sims = rp.top_k(Z, Z[0], "cosine", np, k=2, exclude=[0])
    assert list(rows) == [1, 2] and sims[0] > sims[1]
    with pytest.raises(ValueError):
        rp.similarity(Z, np.ones(3), "cosine", np)
    pb = rp.pair_block(Z, [0, 1], [2, 3], np)
    assert pb.shape == (2, 4) and np.allclose(pb[:, :2], np.abs(Z[[0, 1]] - Z[[2, 3]]))


def test_project_2d_is_deterministic():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((300, 5)) * np.array([5, 2, 1, 0.5, 0.1])
    P = rp.project_2d(Z, np)
    assert P.shape == (300, 2) and np.allclose(P, rp.project_2d(Z, np))
    assert P[:, 0].std() > P[:, 1].std()
    assert np.allclose(np.abs(P), np.abs(rp.project_2d(-Z, np)), atol=1e-5)
    assert rp.project_2d(np.zeros((0, 3)), np).shape == (0, 2)


def test_separability_ranks_the_informative_block():
    rng = np.random.default_rng(0)
    n = 300
    cls = rng.integers(1, 3, n)
    groups = rng.integers(0, 4, n)
    good = np.c_[np.where(cls == 1, 1.0, -1.0), rng.standard_normal(n) * 0.1] \
        + rng.standard_normal((n, 2)) * 0.2
    bad = rng.standard_normal((n, 2))
    sg = rp.separability(good, cls, groups, np)
    sb = rp.separability(bad, cls, groups, np)
    assert sg["knn"] > 0.95 and sg["centroid"] > 0.95 and sg["groups"] == 4
    assert sb["knn"] < 0.7 and sg["silhouette"] > sb["silhouette"]
    assert rp.separability(good, np.ones(n, int), groups, np) is None
    one = rp.separability(good, cls, np.zeros(n), np)                 # one item: halves
    assert one is not None and one["knn"] > 0.9


# --------------------------------------------------------------------------- #
# consumers: a missing layer is refused, never substituted
# --------------------------------------------------------------------------- #
def test_shallower_net_refuses_previous_layer():
    from msseg.labeler import context
    est, X = region_net(hidden=(8,))
    with pytest.raises(ValueError, match="hidden"):
        context.embedding_of(est, X, context.LatentSpec(layer=-2), np)
    assert context.embedding_of(est, X, context.LatentSpec(layer=-1), np).shape == (len(X), 8)
    with pytest.raises(ValueError):
        em._layer(em.embed(est, X), 3)
    assert em._layer(em.embed(est, X), -1).shape == (len(X), 8)
    assert em.spec_layer(em.EdgeSpec(), 2) is None
    assert em.spec_layer(em.EdgeSpec(layer=-2), 2) == 1


def test_by_id_scatters_rows_onto_region_ids():
    t = stats_table(n=5)
    e = rp.resolve(rp.EmbeddingRef(), table=t, np=np)
    vals, mask = e.by_id(np.arange(5, dtype=np.float32), np, K=12)
    assert vals.shape == (12,) and list(np.flatnonzero(mask)) == [0, 2, 4, 6, 8]
    assert vals[8] == 4.0 and vals[1] == 0.0
    assert e.row_of(6) == 3 and e.row_of(7) is None
