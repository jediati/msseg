"""The region encoder: the bundle, the harvest format, and the trainer on a
synthetic harvest whose regions come in three spatial kinds.

The synthetic harvest is what the objective is for: rows carry a
kind-specific signal plus nuisance columns, and the region graph is a grid
whose arcs mostly join same-kind regions; the walk objective has to find the
kinds without ever seeing them.
"""
import json
import os

import numpy as np
import pytest

from msseg.labeler import embedding as emb
from msseg.labeler.embedding_train import shards as S
from msseg.labeler.embedding_train import train as T
from msseg.labeler.table import FeatureTable

NAMES = ["feature_id", "area", "min_x", "max_x", "min_y", "max_y", "ext_x", "ext_y",
         "mean_base", "std_base", "mean_blur_s1", "mean_blur_s2", "nuis_a", "nuis_b"]
SCHEMA = ([{"name": n, "channel": "base", "reduction": n.split("_")[0]} for n in ("mean_base", "std_base")]
          + [{"name": n, "channel": "blur_s1", "reduction": "mean"} for n in ("mean_blur_s1",)]
          + [{"name": n, "channel": "blur_s2", "reduction": "mean"} for n in ("mean_blur_s2",)])


def synthetic_item(rng, side=24, kinds=3, signal=1.0, nuisance=8.0):
    """A side x side grid of regions; the kind is a smooth spatial field
    (blobs), so neighbours mostly share it. Returns (rows, ia, ib, kind)."""
    yy, xx = np.mgrid[0:side, 0:side]
    field = np.sin(xx / 4.0) + np.cos(yy / 5.0) + 0.3 * rng.standard_normal((side, side))
    kind = np.digitize(field, np.quantile(field, np.linspace(0, 1, kinds + 1)[1:-1]))
    kind = kind.ravel()
    n = side * side
    rows = np.zeros((n, len(NAMES)), np.float64)
    rows[:, 0] = np.arange(n)
    rows[:, 1] = rng.integers(20, 400, n)
    rows[:, 2] = xx.ravel() * 10; rows[:, 3] = rows[:, 2] + 9
    rows[:, 4] = yy.ravel() * 10; rows[:, 5] = rows[:, 4] + 9
    rows[:, 6] = rows[:, 2] + 4; rows[:, 7] = rows[:, 4] + 4
    means = np.array([[0.0, 1.0, -1.0], [1.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [0.5, 0.5, -1.0]])
    for j, col in enumerate((8, 9, 10, 11)):
        rows[:, col] = signal * means[j, kind] + 0.5 * rng.standard_normal(n)
    rows[:, 12] = nuisance * rng.standard_normal(n)
    rows[:, 13] = nuisance * rng.standard_normal(n)
    idx = np.arange(n).reshape(side, side)
    ia = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    ib = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    return rows, ia, ib, kind


def write_harvest(tmp_path, n_items=6, seed=0):
    rng = np.random.default_rng(seed)
    w = S.HarvestWriter(str(tmp_path), NAMES,
                        {"schema": SCHEMA, "positional": ["feature_id", "min_x", "max_x", "min_y",
                                                          "max_y", "ext_x", "ext_y"],
                         "level": 4})
    kinds = []
    for t in range(n_items):
        rows, ia, ib, kind = synthetic_item(rng)
        w.write(0, t, 1.0, rows, ia, ib, None, {"key": f"item{t}", "level": 4})
        kinds.append(kind)
    return w, np.concatenate(kinds)


def nearest_centroid_accuracy(z, kind, rng):
    """Fit centroids on half the rows, score the other half."""
    n = len(z)
    perm = rng.permutation(n)
    fit, test = perm[: n // 2], perm[n // 2:]
    cents = np.stack([z[fit][kind[fit] == k].mean(axis=0) for k in np.unique(kind)])
    pred = np.argmin(((z[test][:, None, :] - cents[None]) ** 2).sum(-1), axis=1)
    return float((pred == kind[test]).mean())


# --------------------------------------------------------------------------- #
# the bundle
# --------------------------------------------------------------------------- #
def make_bundle():
    rng = np.random.default_rng(1)
    names = ["area", "mean_base", "std_base"]
    X = np.c_[rng.integers(1, 500, 50), rng.standard_normal(50), rng.random(50)]
    mean, std, log_mask = emb.input_convention(X, names, ["area"])
    layers = [emb.Layer(rng.standard_normal((3, 4)).astype(np.float32), np.zeros(4, np.float32), "silu"),
              emb.Layer(rng.standard_normal((4, 2)).astype(np.float32), np.ones(2, np.float32), None)]
    b = emb.EncoderBundle(names, mean, std, log_mask, layers, np.array([0.1, -0.2]),
                          np.array([2.0, 0.5]), scope="L4", meta={"arch": "mlp"})
    return b, X, names


def test_bundle_round_trip_and_columns(tmp_path):
    b, X, names = make_bundle()
    z = b.embed(X, names)
    assert z.shape == (50, 2) and z.dtype == np.float32
    path = b.save(str(tmp_path / "enc.msenc"))
    c = emb.EncoderBundle.load(path)
    assert c.hash == b.hash and c.scope == "L4" and c.meta["arch"] == "mlp"
    np.testing.assert_allclose(c.embed(X, names), z, rtol=1e-6, atol=1e-6)
    # any column order, extra columns ignored, a missing one refused
    Xr = np.c_[X[:, 2], X[:, 0], np.zeros(50), X[:, 1]]
    np.testing.assert_allclose(c.embed(Xr, ["std_base", "area", "other", "mean_base"]), z, rtol=1e-6)
    with pytest.raises(ValueError):
        c.embed(X[:, :2], names[:2])
    # the column names carry the hash and the conventions see no phantom channel
    cols = c.column_names()
    assert cols == [f"emb{c.hash8}__z00", f"emb{c.hash8}__z01"]
    assert all(emb.is_embedding_column(n) for n in cols)
    from msseg.labeler.fields import DEFAULT
    table = FeatureTable(names, X.astype(np.float64))
    out = c.columns(table)
    assert list(out.names) == names + cols and out.values.shape == (50, 5)
    assert table.values.shape == (50, 3)                 # the input is untouched
    assert DEFAULT.channel_names(out) == ["base"]        # no `emb...` channel appears
    assert {e["channel"] for e in c.schema_entries()} == {f"emb{c.hash8}"}


def test_bundle_hash_mismatch_is_refused(tmp_path):
    import zipfile
    b, _X, _names = make_bundle()
    path = b.save(str(tmp_path / "enc.msenc"))
    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read("meta.json"))
        weights = zf.read("weights.npz")
    meta["hash"] = "0" * 40
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("weights.npz", weights)
        zf.writestr("meta.json", json.dumps(meta))
    with pytest.raises(ValueError):
        emb.EncoderBundle.load(path)


def test_input_convention_handles_constant_and_nonfinite_columns():
    X = np.array([[1.0, 5.0, np.nan], [1.0, 7.0, 2.0], [1.0, 9.0, np.inf]])
    mean, std, log_mask = emb.input_convention(X, ["c", "v", "n"], [])
    assert std[0] == 1.0 and mean[0] == 1.0
    Z = emb.prepare_inputs(X, mean, std, log_mask)
    assert np.isfinite(Z).all() and Z[:, 0].tolist() == [0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# the harvest format
# --------------------------------------------------------------------------- #
def test_harvest_round_trip_offsets_arcs_and_resumes(tmp_path):
    w, kinds = write_harvest(tmp_path, n_items=3)
    assert w.has(0, 1, 1.0) and not w.has(0, 9, 1.0)
    r = S.HarvestReader(str(tmp_path))
    h = r.load()
    assert h.n_rows == 3 * 24 * 24 and len(h.shards) == 3
    assert h.ia.max() < h.n_rows and h.ib.max() < h.n_rows
    # arcs never cross a shard
    assert (h.row_shard[h.ia] == h.row_shard[h.ib]).all()
    assert h.column("area") is not None and h.doc["level"] == 4
    # resuming with the same columns keeps the shards; other columns are refused
    w2 = S.HarvestWriter(str(tmp_path), NAMES)
    assert w2.resumed and len(w2.doc["shards"]) == 3
    with pytest.raises(ValueError):
        S.HarvestWriter(str(tmp_path), NAMES[:-1])
    # a bounded load takes whole shards
    part = r.load(max_rows=600, seed=1)
    assert part.n_rows in (576, 1152)
    with pytest.raises(ValueError):
        w.write(0, 5, 1.0, np.zeros((4, len(NAMES))), [0, 3], [1, 9], None, {})


def test_random_walk_stays_on_the_graph():
    from msseg.labeler import context
    rng = np.random.default_rng(0)
    rows, ia, ib, _k = synthetic_item(rng, side=6)
    g = context.directed_graph(ia, ib, len(rows), np)
    start = np.arange(len(rows))
    end = T.random_walk(g, start, 1, rng)
    # every 1-hop endpoint is an actual neighbour
    nbrs = {i: set(g.dst[g.indptr[i]:g.indptr[i + 1]].tolist()) for i in range(len(rows))}
    assert all(e in nbrs[s] for s, e in zip(start, end))
    # an isolated row stays put
    g2 = context.directed_graph([], [], 3, np)
    assert T.random_walk(g2, np.array([0, 1, 2]), 3, rng).tolist() == [0, 1, 2]


def test_column_groups_follow_the_schema():
    names = ["mean_base", "std_base", "mean_blur_s1", "mean_blur_s2", "nuis_a"]
    group_of, n = T.column_groups(names, SCHEMA)
    assert n == 4 and group_of[0] == group_of[1] and group_of[2] != group_of[3]
    assert group_of[4] not in group_of[:4]


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def test_pca_trains_without_torch_and_drops_positional_columns(tmp_path):
    write_harvest(tmp_path, n_items=2)
    h = S.HarvestReader(str(tmp_path)).load()
    bundle, hist = T.train(h, T.TrainSettings(arch="pca", dim=3), log=lambda m: None)
    assert bundle.dim == 3 and bundle.scope == "L4"
    assert "feature_id" not in bundle.names and "ext_x" not in bundle.names
    assert bundle.log_mask[bundle.names.index("area")]
    z = bundle.embed(h.rows, h.names)
    assert z.shape == (h.n_rows, 3)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-3)
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=1e-3)
    assert hist[0]["explained_variance"]


def test_walk_objective_separates_the_kinds(tmp_path):
    torch = pytest.importorskip("torch")
    w, kinds = write_harvest(tmp_path, n_items=8, seed=3)
    h = S.HarvestReader(str(tmp_path)).load()
    rng = np.random.default_rng(0)
    pca, _ = T.train(h, T.TrainSettings(arch="pca", dim=2), log=lambda m: None)
    acc_pca = nearest_centroid_accuracy(pca.embed(h.rows, h.names), kinds, rng)
    settings = T.TrainSettings(arch="mlp", dim=2, hidden=(32, 16), epochs=25, batch=512,
                               walk=1, device="cpu", seed=0)
    bundle, hist = T.train(h, settings, log=lambda m: None)
    z = bundle.embed(h.rows, h.names)
    acc = nearest_centroid_accuracy(z, kinds, rng)
    # the walk objective must separate the three kinds in two dimensions
    # without ever seeing them (PCA-2 is printed as the reference; on this
    # synthetic row the standardised signal columns dominate it too, which
    # is exactly why the design's probe compares both on real data)
    print(f"nearest-centroid accuracy: walk {acc:.3f}, pca {acc_pca:.3f}")
    assert acc > 0.75, (acc, acc_pca)
    assert hist[-1]["acc"] > hist[0]["acc"]
    assert bundle.meta["heldout_walk_acc"] is not None
    # the bundle survives a round trip and its numpy inference matches torch's
    path = bundle.save(str(tmp_path / "enc.msenc"))
    again = emb.EncoderBundle.load(path)
    np.testing.assert_allclose(again.embed(h.rows, h.names), z, rtol=1e-5, atol=1e-5)
    assert again.meta["settings"]["walk"] == 1 and again.meta["design"]["n_rows"] == h.n_rows


def test_train_cli_writes_a_bundle(tmp_path):
    from msseg.labeler.embedding_train import cli
    write_harvest(tmp_path / "h", n_items=2)
    out = tmp_path / "enc.msenc"
    rc = cli.main([str(tmp_path / "h"), "--out", str(out), "--arch", "pca", "--dim", "4",
                   "--quiet", "--history", str(tmp_path / "hist.json")])
    assert rc == 0 and out.is_file() and (tmp_path / "hist.json").is_file()
    assert emb.EncoderBundle.load(str(out)).dim == 4
