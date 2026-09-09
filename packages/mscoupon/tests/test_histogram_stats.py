"""Per-region histograms on the Python side: the config round trip, the
workflow summary, the 3D assembly's bin-wise merge, the magic-fill metric,
the classifier's feature grouping, and (with the extension) the projected
columns of a primed slice.
"""
import json

import numpy as np
import pytest

from msseg.mscoupon import assembly, config_io, magic_fill, model_search, session
from msseg.mscoupon.common import FeatureTable


def test_histogram_block_round_trips_and_is_off_by_default():
    doc = config_io.statistics_to_json([{"kind": "base"}], ["mean"], True, 0)
    assert "histogram" not in doc
    assert config_io.statistics_from_json(doc)["histogram"] == {"bins": 0, "channels": [], "ranges": {}}
    hist = {"bins": 16, "channels": ["base", "blur_s1.5"], "ranges": {"*": [0, 1], "blur_s1.5": [-1, 2]}}
    doc = config_io.statistics_to_json([{"kind": "base"}], ["mean"], True, 0, True, hist)
    assert doc["histogram"] == {"bins": 16, "channels": ["base", "blur_s1.5"],
                                "ranges": {"*": [0.0, 1.0], "blur_s1.5": [-1.0, 2.0]}}
    assert config_io.statistics_from_json(doc)["histogram"] == doc["histogram"]
    assert config_io.histogram_to_json({"bins": 0}) is None
    assert config_io.histogram_to_json({"bins": 9999})["bins"] == config_io.HIST_MAX_BINS
    p = session.default_profile("p")
    p["statistics"] = doc
    assert session.profile_from_json(p) == p
    assert session.profile_summary(p).splitlines()[1] == "stats: base→1ch×1+16h×2"


def test_assembly_histograms_merge_bin_wise():
    lbl = np.array([[1, 1, 2], [1, 2, 2]], dtype=np.int64)
    base = np.array([[0.1, 0.6, 5.0], [np.nan, -3.0, 0.3]], dtype=np.float32)
    st = assembly.node_stats(lbl, 2, base, base, True, 0.0, 1.0, [("base", base)],
                             {"bins": 4, "ranges": {"base": (0.0, 1.0)}})
    h = st[assembly.chan_col("base", "hist")]
    assert h.shape == (2, 4)
    assert h[0].tolist() == [1.0, 0.0, 1.0, 0.0]        # 0.1, 0.6; NaN skipped
    assert h[1].tolist() == [1.0, 1.0, 0.0, 1.0]        # -3 clamps low, 0.3, 5 clamps high
    # Two slices whose components link into one 3D feature: bins add.
    labels = [np.array([[0, 0]], dtype=np.int64), np.array([[0, 0]], dtype=np.int64)]
    b0 = np.array([[0.1, 0.6]], dtype=np.float32)
    b1 = np.array([[0.9, 0.95]], dtype=np.float32)
    out = assembly.assemble_cc(labels, [None, None], [b0, b1], [b0, b1], connectivity=6,
                               histogram={"bins": 4, "ranges": {"base": (0.0, 1.0)}})
    row = out["global_table"][0]
    assert row["voxel_count"] == 4
    assert [row[f"hist{b:02d}_base"] for b in range(4)] == [0.25, 0.0, 0.25, 0.5]


def _hist_table():
    names = ["feature_id", "area", "mean_base", "ext_x", "ext_y",
             "hist00_base", "hist01_base", "hist02_base"]
    values = np.array([[0, 4, 0.5, 1, 1, 1.0, 0.0, 0.0],
                       [1, 4, 0.5, 2, 2, 0.0, 1.0, 0.0],
                       [2, 4, 0.5, 3, 3, 0.5, 0.5, 0.0]], dtype=np.float64)
    return FeatureTable(names, values)


def test_histogram_metric_and_cosine_exclusion():
    table = _hist_table()
    X, kind = magic_fill.row_vectors(table, "histogram", ["base"], np)
    assert kind == "hellinger" and X.shape == (3, 3)
    d = magic_fill.pairwise(X, 0, np.array([0, 1, 2]), kind, np)
    assert d[0] == 0.0 and d[1] == pytest.approx(1.0) and 0.0 < d[2] < 1.0
    # cosine ignores the bins (only mean_base survives, so every row is alike).
    Xc, _ = magic_fill.row_vectors(table, "cosine", [], np)
    assert Xc.shape == (3, 2)        # area, mean_base
    assert magic_fill.histogram_columns(table) == {"base": ["hist00_base", "hist01_base", "hist02_base"]}
    bare = FeatureTable(["feature_id", "mean_base"], np.zeros((2, 2)))
    with pytest.raises(ValueError):
        magic_fill.row_vectors(bare, "histogram", ["base"], np)
    assert "histogram" in magic_fill.METRICS and "histogram" in magic_fill.CHANNEL_METRICS


def test_feature_groups_keep_a_channels_bins_together():
    schema = [{"name": "mean_base", "channel": "base", "reduction": "mean"},
              {"name": "hist00_base", "channel": "base", "reduction": "hist00"},
              {"name": "hist01_base", "channel": "base", "reduction": "hist01"},
              {"name": "area", "channel": "", "reduction": ""}]
    groups = model_search.feature_groups(["mean_base", "hist00_base", "hist01_base", "area"], schema)
    assert groups == {"base": ["mean_base", "hist00_base", "hist01_base"], "area": ["area"]}


def test_extension_projects_histogram_columns_last():
    engine = pytest.importorskip("msseg.mscoupon")
    params = json.dumps({"statistics": {"channels": ["base"], "reductions": ["mean"],
                                        "histogram": {"bins": 8, "ranges": {"*": [0, 1]}}}})
    try:
        fields = list(engine.feature_fields(params))
    except RuntimeError:
        pytest.skip("extension predates histograms")
    assert fields[-8:] == [f"hist{b:02d}_base" for b in range(8)]
    assert fields.index("ext_filtered") == len(fields) - 9
    chans = engine.stat_channels(params)
    assert chans[0]["hist"] and tuple(chans[0]["hist_range"]) == (0.0, 1.0)
    rng = np.random.default_rng(1)
    base = rng.random((24, 30), dtype=np.float32)
    pipe = engine.prime_slice(base, base, params)
    names, values = pipe.feature_table()
    names = list(names)
    cols = [names.index(f"hist{b:02d}_base") for b in range(8)]
    assert np.allclose(values[:, cols].sum(axis=1), 1.0)
    with pytest.raises(RuntimeError):
        engine.feature_fields(json.dumps({"statistics": {"histogram": {"bins": 8}}}))   # no range


def test_optimize_search_ingests_colour_and_histogram_columns():
    """The classifier side, end to end and headless: a colour+histogram profile
    primes a slice, its feature table trains through the Optimize search with
    the feature-subset mask, and a channel's bins ride with the channel."""
    engine = pytest.importorskip("msseg.mscoupon")
    from scipy import ndimage
    rng = np.random.default_rng(5)
    base = ndimage.gaussian_filter(rng.random((96, 96)), 1.5).astype(np.float32)
    planes = np.stack([base, np.sqrt(base), 1.0 - base]).astype(np.float32)
    params = json.dumps({
        "input": {"color": {"channels": 3}},
        "msc": {"persistence_percent": 2.0},
        "statistics": {"channels": ["base", "color", {"kind": "dizenzo", "sigmas": [1.0]}],
                       "reductions": ["mean", "std"],
                       "histogram": {"bins": 6, "channels": ["base", "color_c2"],
                                     "ranges": {"*": [0, 1]}}}})
    try:
        pipe = engine.prime_slice(base, base, params, planes)
    except RuntimeError:
        pytest.skip("extension predates colour statistics")
    names, values = pipe.feature_table()
    names = list(names)
    keep = [n for n in names if n not in magic_fill.POSITIONAL_FIELDS]
    X = values[:, [names.index(n) for n in keep]]
    assert X.shape[0] >= 12, "enough regions to train on"
    y = (X[:, keep.index("mean_color_c2")] > np.median(X[:, keep.index("mean_color_c2")])).astype(int)
    schema = engine.feature_schema(params)
    groups = model_search.feature_groups(keep, schema)
    assert set(groups["color_c2"]) >= {"mean_color_c2", "std_color_c2", "ext_color_c2"} | {
        f"hist{b:02d}_color_c2" for b in range(6)}, groups["color_c2"]
    assert "dizenzo_largest_s1" in groups
    result = model_search.run_search(X, y, None, keep, schema, n_trials=2, seed=1,
                                     searcher="random", backend="sklearn", max_iter=50,
                                     importances=False)
    est = result.estimator
    proba = est.predict_proba(X)
    assert proba.shape == (X.shape[0], 2) and np.allclose(proba.sum(axis=1), 1.0)
    # The FeatureSubset step sees the full schema and masks by channel group.
    subset = est.named_steps["select"]
    assert list(subset.names) == keep
