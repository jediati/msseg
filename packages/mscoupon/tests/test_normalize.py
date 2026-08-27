"""Tests for msseg.mscoupon.normalize.

These cover the Python surface: the zero-masking policy (the ``omit_zeros``
parameter this subpackage was built around), the ``TwoPoint`` mapping, and the
config plumbing that carries a normalize stage out to the C++ CLI.

The numerical agreement between the Python wrappers and the compiled measures is
covered separately by ``gmm_parity.py`` (Python vs sklearn) and by the C++ suite
in ``mscoupon_tests.cpp``. Everything here runs without the extension built --
the wrappers fall back to numpy -- so this file is safe in a bare checkout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# src-layout: make msseg.mscoupon importable straight from the checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msseg.mscoupon import config_io                                    # noqa: E402
from msseg.mscoupon.normalize import (PERCENTILE_NAMES, TwoPoint,       # noqa: E402
                                      get_valid_pixels, measure_histogram,
                                      measure_two_point, measure_two_regions,
                                      fit_two_gaussians, two_point_from_result)

MU_LOW, MU_HIGH = 10.0, 25.0


def bimodal(n=60_000, seed=0):
    """A clean two-population sample: the shape every measure is meant to find."""
    rng = np.random.default_rng(seed)
    which = rng.choice(2, n, p=[0.4, 0.6])
    mus = np.array([MU_LOW, MU_HIGH])
    sigmas = np.array([1.0, 2.0])
    return rng.normal(mus[which], sigmas[which]).astype(np.float32)


# --------------------------------------------------------------------------- #
# The zero policy
# --------------------------------------------------------------------------- #
def test_get_valid_pixels_omit_zeros_default():
    x = np.array([0.0, 1.0, 2.0, 0.0, 3.0], dtype=np.float32)
    assert get_valid_pixels(x).tolist() == [1.0, 2.0, 3.0]
    assert get_valid_pixels(x, omit_zeros=False).tolist() == [0.0, 1.0, 2.0, 0.0, 3.0]


def test_get_valid_pixels_drops_nonfinite_but_only_for_floats():
    x = np.array([1.0, np.nan, np.inf, -np.inf, 2.0], dtype=np.float32)
    assert get_valid_pixels(x).tolist() == [1.0, 2.0]

    # Integer arrays have no NaN/Inf to test for; only the zero rule applies.
    i = np.array([0, 1, 2, 0], dtype=np.int16)
    assert get_valid_pixels(i).tolist() == [1.0, 2.0]
    assert get_valid_pixels(i, omit_zeros=False).tolist() == [0.0, 1.0, 2.0, 0.0]


def test_get_valid_pixels_returns_float64_1d():
    out = get_valid_pixels(np.arange(1, 13, dtype=np.int16).reshape(3, 4))
    assert out.ndim == 1 and out.dtype == np.float64


# --------------------------------------------------------------------------- #
# TwoPoint
# --------------------------------------------------------------------------- #
def test_two_point_maps_normalized_to_raw():
    tp = TwoPoint(2.0, 6.0)
    assert tp.valid and tp.scale == 4.0
    # The headline contract: 0.7 means 0.3*low + 0.7*high.
    assert tp.to_raw(0.7) == pytest.approx(0.3 * 2.0 + 0.7 * 6.0)
    assert tp.to_raw(0.0) == pytest.approx(2.0)
    assert tp.to_raw(1.0) == pytest.approx(6.0)
    assert tp.to_norm(tp.to_raw(0.42)) == pytest.approx(0.42)
    # Values outside the landmarks stay meaningful; nothing clamps by default.
    assert tp.to_raw(-0.5) < 2.0 and tp.to_raw(1.5) > 6.0


def test_two_point_rejects_degenerate_pairs():
    assert not TwoPoint(5.0, 5.0).valid
    assert not TwoPoint(6.0, 2.0).valid


def test_two_point_apply_is_affine_and_can_be_in_place():
    raw = np.linspace(0.0, 10.0, 64, dtype=np.float32).reshape(8, 8)
    tp = TwoPoint(2.0, 6.0)

    out = tp.apply(raw)
    assert out == pytest.approx((raw - 2.0) / 4.0, abs=1e-5)
    assert raw[0, 1] != out[0, 1], "apply() must not modify its input by default"

    buf = raw.copy()
    tp.apply(buf, out=buf)
    assert buf == pytest.approx(out, abs=1e-5)

    # A degenerate pair is a no-op rather than a divide by zero.
    same = raw.copy()
    assert TwoPoint(1.0, 1.0).apply(same) == pytest.approx(raw)


def test_two_point_apply_clamps_only_on_request():
    raw = np.array([[0.0, 4.0, 12.0]], dtype=np.float32)
    tp = TwoPoint(2.0, 6.0)
    assert tp.apply(raw).min() < 0.0
    clamped = tp.apply(raw, clamp=True)
    assert clamped.min() >= 0.0 and clamped.max() <= 1.0


# --------------------------------------------------------------------------- #
# The statistics property that lets normalization be a filter
# --------------------------------------------------------------------------- #
def test_statistics_transform_affine_mean_scale_only_std():
    """mean maps affinely, std maps by scale alone -- the offset cancels.

    This is why no statistic needs a per-field "is it a location or a spread"
    classification: normalizing the channel gets it right automatically.
    """
    raw = bimodal(20_000, seed=3).astype(np.float64)
    tp = TwoPoint(MU_LOW, MU_HIGH)
    norm = tp.apply(raw).astype(np.float64)

    assert np.mean(norm) == pytest.approx(tp.to_norm(np.mean(raw)), rel=1e-5)
    assert np.std(norm) == pytest.approx(np.std(raw) / tp.scale, rel=1e-5)


# --------------------------------------------------------------------------- #
# Measures
# --------------------------------------------------------------------------- #
def test_gmm_recovers_the_two_populations():
    r = fit_two_gaussians(bimodal(), preset="two_gaussian")
    assert r["mu_1"] == pytest.approx(MU_LOW, abs=0.2)
    assert r["mu_2"] == pytest.approx(MU_HIGH, abs=0.2)
    assert r["mu_1"] < r["mu_2"], "components are sorted by increasing mean"
    assert r["weight_1"] + r["weight_2"] == pytest.approx(1.0, abs=1e-6)


def test_gmm_omit_zeros_changes_the_valid_count():
    x = bimodal(20_000, seed=1)
    padded = np.zeros(x.size * 2, dtype=np.float32)
    padded[1::2] = x

    dropped = fit_two_gaussians(padded, preset="two_gaussian")
    kept = fit_two_gaussians(padded, preset="two_gaussian", omit_zeros=False)
    assert dropped["n_valid_pixels"] == x.size
    assert kept["n_valid_pixels"] == padded.size
    # With the no-data zeros excluded the real populations are still found.
    assert dropped["mu_1"] == pytest.approx(MU_LOW, abs=0.3)
    assert dropped["mu_2"] == pytest.approx(MU_HIGH, abs=0.3)


def test_histogram_peaks_are_ordered_by_intensity():
    r = measure_histogram(bimodal(200_000, seed=2), bins=512)
    assert r["peak_low"] < r["peak_high"]
    assert r["peak_low"] == pytest.approx(MU_LOW, abs=0.6)
    assert r["peak_high"] == pytest.approx(MU_HIGH, abs=0.6)
    for name in PERCENTILE_NAMES:
        assert name in r


def test_regions_keep_zeros_by_default():
    """The region measure's zero policy is deliberately the opposite of the others.

    Its rectangles are hand-picked physical regions, so a zero there is a
    measurement, not the no-data background.
    """
    image = np.zeros((10, 10), dtype=np.float32)
    image[:5, :] = 4.0

    stats = measure_two_regions(image, (0, 10), (0, 10), (0, 5), (0, 10))
    assert stats["air"]["n_pixels"] == 100
    assert stats["air"]["mean"] == pytest.approx(2.0)

    dropped = measure_two_regions(image, (0, 10), (0, 10), (0, 5), (0, 10),
                                  omit_zeros=True)
    assert dropped["air"]["n_pixels"] == 50
    assert dropped["air"]["mean"] == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# measure_two_point
# --------------------------------------------------------------------------- #
def test_measure_two_point_defaults_per_method():
    px = bimodal(50_000, seed=4)
    gmm = measure_two_point(px, method="gmm", preset="two_gaussian")
    assert gmm.low == pytest.approx(MU_LOW, abs=0.3)
    assert gmm.high == pytest.approx(MU_HIGH, abs=0.3)

    hist = measure_two_point(px, method="histogram", bins=512)
    assert hist.low == pytest.approx(MU_LOW, abs=0.6)
    assert hist.high == pytest.approx(MU_HIGH, abs=0.6)


def test_measure_two_point_manual_and_fallback():
    assert measure_two_point(np.zeros((4, 4), np.float32), method="manual",
                             low=3.0, high=11.0) == TwoPoint(3.0, 11.0)

    with pytest.raises(ValueError):
        measure_two_point(np.zeros((4, 4), np.float32), method="manual", low=1.0)

    # A constant slice has no two populations; the manual pair is the fallback
    # so one unusable slice cannot abort a whole stack.
    flat = np.full((32, 32), 4.0, dtype=np.float32)
    assert measure_two_point(flat, method="histogram", low=0.0, high=8.0) == TwoPoint(0.0, 8.0)

    with pytest.raises(Exception):
        measure_two_point(flat, method="histogram")


def test_two_point_from_result_reports_unknown_landmarks():
    r = fit_two_gaussians(bimodal(20_000, seed=5), preset="two_gaussian")
    assert two_point_from_result(r, "mu_1", "mu_2").valid
    with pytest.raises(KeyError):
        two_point_from_result(r, "mu_1", "nope")


# --------------------------------------------------------------------------- #
# Config plumbing
# --------------------------------------------------------------------------- #
def test_normalize_is_a_known_filter_operation():
    assert "normalize" in config_io.FILTER_OPERATIONS
    params = dict((name, kind) for name, kind, _ in config_io.FILTER_SCHEMA["normalize"])
    assert params["method"].startswith("choice:")
    assert params["low"] == "optfloat" and params["high"] == "optfloat"


def test_build_config_emits_base_filters_only_when_used():
    common = dict(files=["/data/a.tif"], output_folder="/out")

    plain = config_io.build_config(filters=[], **common)
    assert "base_filters" not in plain, "an unnormalized workflow exports as before"

    normalized = config_io.build_config(
        filters=[],
        base_filters=[{"operation": "normalize",
                       "params": {"method": "gmm", "low_from": "", "high": ""}}],
        **common)
    # Blank optional params are dropped, so the CLI falls back to the method's
    # own defaults instead of reading "" or 0.0 as a real value.
    assert normalized["base_filters"] == [
        {"operation": "normalize", "params": {"method": "gmm"}}]


def test_build_config_drops_none_stages_from_the_base_chain():
    cfg = config_io.build_config(
        files=["/data/a.tif"], output_folder="/out", filters=[],
        base_filters=[{"operation": "none", "params": {}}])
    assert "base_filters" not in cfg


# --------------------------------------------------------------------------- #
# Reading a config back (the viewer's "Load config.json" / "Restore last")
#
# The writers above fold several controls into more than one JSON key and omit
# keys sitting at their defaults, so the read side is not a mirror image -- each
# of those asymmetries gets its own test. Everything here is pure Python: pass
# `fields=` explicitly rather than calling query_fields(), which needs the
# extension.
# --------------------------------------------------------------------------- #
def _representative_config():
    """A config exercising every block the viewer writes."""
    return config_io.build_config(
        files=["/data/a.tif", "/data/b.tif"], output_folder="/out",
        filters=[{"operation": "blur", "params": {"sigma": 2.0}}],
        base_filters=[{"operation": "normalize",
                       "params": {"method": "gmm", "low_from": "", "high_from": "",
                                  "low": "", "high": "", "omit_value": 43.0,
                                  "downsample_factor": 1, "clamp": False}}],
        persistence_percent=10.0, manifold="ascending", accurate=False,
        extremum_sample_radius=0, min_area=25,
        feature_filters=[{"field": "area", "op": "ge", "value": 50.0}],
        pixel_filters=[{"channel": "filtered", "mode": "omit", "op": "lt",
                        "value": 0.1}],
        connectivity=6, cores_per_slice=4, concurrent_slices=1)


def _rebuild(state):
    """build_config from a config_to_state() dict -- what the viewer does after
    a load, once the state has gone through the widgets."""
    return config_io.build_config(
        files=state["files"], output_folder=state["output_folder"],
        filters=state["filters"], base_filters=state["base_filters"],
        persistence_percent=state["persistence_percent"],
        manifold=state["manifold"], accurate=state["accurate"],
        extremum_sample_radius=state["extremum_sample_radius"],
        min_area=state["min_area"], feature_filters=state["feature_filters"],
        pixel_filters=state["pixel_filters"], connectivity=state["connectivity"],
        cores_per_slice=state["cores_per_slice"],
        concurrent_slices=state["concurrent_slices"], folder=state["folder"])


def test_config_to_state_round_trips_a_built_config():
    cfg = _representative_config()
    notes = []
    assert _rebuild(config_io.config_to_state(cfg, fields=["area"], notes=notes)) == cfg
    assert notes == [], notes


def test_filter_cards_round_trip_is_a_fixpoint():
    """The FIRST pass is deliberately not the identity: a stage carrying only
    {"method": "gmm"} expands to a card holding every param the schema declares,
    which re-exports as the full set. Idempotence from the second pass on is the
    invariant the viewer lives on -- reloading a config it just wrote must not
    keep changing it."""
    to, fro = config_io.filters_to_json, config_io.filters_from_json
    start = [{"operation": "normalize", "params": {"method": "gmm"}}]
    once = to(fro(start))
    assert once != start, "the first pass fills in the schema defaults"
    assert to(fro(once)) == once
    assert to(fro(to(fro(once)))) == once


def test_nullfloat_null_becomes_blank_but_absent_becomes_the_default():
    """omit_value's three states must stay three states: an explicit null is
    "keep every pixel", an absent key is the default sentinel 0, and a number is
    itself. Collapsing null into the default would silently reintroduce the
    sentinel on the next save."""
    fro, to = config_io.filters_from_json, config_io.filters_to_json

    def card(params):
        return fro([{"operation": "normalize", "params": params}])[0]["params"]

    def exported(params):
        return to([{"operation": "normalize", "params": card(params)}])[0]["params"]

    assert card({"omit_value": None})["omit_value"] == ""
    assert card({})["omit_value"] == 0.0
    assert card({"omit_value": 43.0})["omit_value"] == 43.0

    assert exported({"omit_value": None})["omit_value"] is None
    assert exported({})["omit_value"] == 0.0
    assert exported({"omit_value": 43.0})["omit_value"] == 43.0


def test_optfloat_blank_survives_the_round_trip():
    fro, to = config_io.filters_from_json, config_io.filters_to_json
    card = fro([{"operation": "normalize",
                 "params": {"method": "manual", "low": 0.2}}])[0]
    assert card["params"]["low"] == 0.2
    assert card["params"]["high"] == "", "an unset bound stays blank, not 0.0"
    out = to([card])[0]["params"]
    assert out["low"] == 0.2 and "high" not in out


def test_config_to_state_reads_the_legacy_singular_filter():
    state = config_io.config_to_state(
        {"filter": {"operation": "blur", "params": {"sigma": 1.5}}})
    assert [c["operation"] for c in state["filters"]] == ["blur"]
    assert state["filters"][0]["params"]["sigma"] == 1.5


def test_config_to_state_inverts_the_accurate_pair():
    # One checkbox, two JSON keys: either one set means "accurate".
    for asc, dsc, want in ((True, False, True), (False, True, True),
                           (True, True, True), (False, False, False)):
        state = config_io.config_to_state(
            {"msc": {"accurate_ascending": asc, "accurate_descending": dsc}})
        assert state["accurate"] is want, (asc, dsc)


def test_config_to_state_prefers_requested_parallelism():
    both = config_io.config_to_state(
        {"msc": {"requested_parallelism": 8}, "execution": {"threads_per_slice": 8}})
    assert both["cores_per_slice"] == 8
    assert config_io.config_to_state(
        {"execution": {"threads_per_slice": 3}})["cores_per_slice"] == 3
    assert config_io.config_to_state({})["cores_per_slice"] is None


def test_config_to_state_reads_the_extremum_radius_from_either_block():
    from_msc = config_io.config_to_state({"msc": {"extremum_sample_radius": 2}})
    assert from_msc["extremum_sample_radius"] == 2
    from_stats = config_io.config_to_state({"statistics": {"extremum_sample_radius": 3}})
    assert from_stats["extremum_sample_radius"] == 3
    both = config_io.config_to_state({"msc": {"extremum_sample_radius": 2},
                                      "statistics": {"extremum_sample_radius": 3}})
    assert both["extremum_sample_radius"] == 3, "an explicit statistics block wins"
    assert config_io.config_to_state({})["extremum_sample_radius"] == 0


def test_config_to_state_reports_a_percentless_config():
    notes = []
    state = config_io.config_to_state({"msc": {"persistence_absolute": 0.0007}},
                                      notes=notes)
    assert state["persistence_percent"] is None, "a percent is never invented"
    assert state["persistence_absolute"] == 0.0007
    assert notes and "persistence_absolute" in notes[0]
    legacy = config_io.config_to_state({"msc": {"persistence": 0.5}})
    assert legacy["persistence_absolute"] == 0.5, "the legacy alias is an absolute"


def test_config_to_state_survives_junk():
    """A hand-edited config must never raise: in a Tk callback an exception is a
    traceback on stderr and a button that did nothing."""
    state = config_io.config_to_state({"msc": "nonsense", "filters": 7, "input": [],
                                       "feature_filters": {"field": "area"},
                                       "assembly": None, "segments": 3})
    assert state["filters"] == [] and state["feature_filters"] == []
    assert state["manifold"] is None and state["connectivity"] is None
    assert state["min_area"] is None and state["folder"] == "" and state["files"] == []
    assert state["accurate"] is False
    assert config_io.config_to_state(None)["filters"] == []
    assert config_io.config_to_state("nope")["cores_per_slice"] is None


def test_filters_from_json_skips_unknown_operations_and_params():
    notes = []
    cards = config_io.filters_from_json(
        [{"operation": "wibble", "params": {}},
         {"operation": "blur", "params": {"sigma": 2.0, "nope": 1}},
         {"operation": "blur", "params": {"sigma": "abc"}},
         {"operation": "hessian_eigenvalues", "params": {"component": "sideways"}},
         "not an object"], notes)
    assert [c["operation"] for c in cards] == ["blur", "blur", "hessian_eigenvalues"]
    assert cards[0]["params"] == {"sigma": 2.0}
    assert cards[1]["params"]["sigma"] == 1.0, "a non-number falls back to the default"
    assert cards[2]["params"]["component"] == "largest"
    joined = " | ".join(notes)
    for expected in ("wibble", "nope", "abc", "sideways", "not an object"):
        assert expected in joined, (expected, joined)


def test_queries_from_json_drops_fields_outside_the_schema():
    notes = []
    rows = config_io.queries_from_json(
        [{"field": "area", "op": "ge", "value": 50},
         {"field": "relevance_base", "op": "gt", "value": 0.2},
         {"field": "area", "op": "wat", "value": 1},
         {"field": "", "op": "gt", "value": 1}],
        fields=["area"], notes=notes)
    assert rows == [{"field": "area", "op": "ge", "value": 50.0, "value2": 0.0}]
    assert any("relevance_base" in n for n in notes)
    assert any("wat" in n for n in notes)
    # With no field list there is no schema to check against (a headless caller
    # has none), so a named field is kept.
    assert len(config_io.queries_from_json([{"field": "relevance_base"}])) == 1


def test_pixel_filters_from_json_drops_bad_channels_and_coerces_the_rest():
    notes = []
    rows = config_io.pixel_filters_from_json(
        [{"channel": "green", "mode": "keep", "op": "gt", "value": 1},
         {"channel": "base", "mode": "sideways", "op": "wat", "value": "0.5"}], notes)
    assert rows == [{"channel": "base", "mode": "keep", "op": "gt", "value": 0.5}]
    assert any("green" in n for n in notes)


def test_config_to_state_resolves_relative_input_files():
    import os
    root = os.path.join(os.sep, "data")
    state = config_io.config_to_state(
        {"input": {"folder": root,
                   "files": ["a.tif", os.path.join(os.sep, "elsewhere", "b.tif")]}})
    assert state["files"] == [os.path.join(root, "a.tif"),
                              os.path.join(os.sep, "elsewhere", "b.tif")]


# --------------------------------------------------------------------------- #
# Session file (the auto-saved "last used configuration")
# --------------------------------------------------------------------------- #
def test_split_session_accepts_a_bare_appconfig():
    """One code path for a hand-written config, an exported config and a saved
    session -- the bare config simply carries no GUI half."""
    cfg = {"input": {"folder": "/d"}, "msc": {}}
    assert config_io.split_session(cfg) == (cfg, {})
    assert config_io.split_session("nonsense") == ({}, {})


def test_build_session_adds_only_the_gui_key():
    cfg = config_io.build_config(files=["/d/a.tif"], output_folder="/out", filters=[])
    doc = config_io.build_session(cfg, {"folder": "/d", "subsequences": []})
    assert set(doc) - set(cfg) == {config_io.SESSION_KEY}
    back, gui = config_io.split_session(doc)
    assert back == cfg, "the wrapper must leave the AppConfig half runnable"
    assert gui["folder"] == "/d"


def test_serialize_session_is_stable_under_key_order():
    # The auto-save decides "did anything change?" by comparing this text, so a
    # reordered dict must not read as a change and trigger a write every tick.
    a = config_io.serialize_session({"b": 1, "a": {"y": 2, "x": 3}})
    b = config_io.serialize_session({"a": {"x": 3, "y": 2}, "b": 1})
    assert a == b


def test_session_path_follows_the_platform_config_dir(monkeypatch, tmp_path):
    import os
    if os.name == "nt":
        monkeypatch.setenv("APPDATA", str(tmp_path))
    else:
        monkeypatch.delenv("APPDATA", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config_io.app_data_dir() == os.path.join(str(tmp_path), "mscoupon")
    assert config_io.session_path().endswith("last_session.json")

    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert config_io.app_data_dir().endswith(os.path.join(".config", "mscoupon"))


def test_write_session_text_is_atomic_and_best_effort(tmp_path):
    target = tmp_path / "sub" / "last_session.json"
    assert config_io.write_session_text('{"a": 1}', str(target)) is True
    assert config_io.read_json_file(str(target)) == {"a": 1}
    assert not (tmp_path / "sub" / "last_session.json.tmp").exists()

    # An unwritable destination reports False rather than raising -- auto-save
    # must never interrupt the session it is trying to preserve.
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    assert config_io.write_session_text("{}", str(blocker / "nope.json")) is False

    # Missing, malformed and non-object files all read back as None.
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "arr.json").write_text("[1, 2]", encoding="utf-8")
    assert config_io.read_json_file(str(tmp_path / "bad.json")) is None
    assert config_io.read_json_file(str(tmp_path / "arr.json")) is None
    assert config_io.read_json_file(str(tmp_path / "missing.json")) is None


# --------------------------------------------------------------------------- #
# Statistics schema + the 3D seeding extremum
# --------------------------------------------------------------------------- #
def test_query_fields_drops_feature_id_and_filtered_aggregates():
    names = config_io.query_fields()
    assert "feature_id" not in names, "an id is not a selectable statistic"
    assert "mean_base" in names and "ext_base" in names and "relevance_base" in names
    # The filtered aggregates had no reader, so they are off unless asked for.
    assert "mean_filtered" not in names
    assert "std_filtered" not in names


def test_query_fields_follows_the_statistics_spec():
    pytest.importorskip("msseg.mscoupon.mscoupon_py")
    import json
    with_filtered = config_io.query_fields(
        json.dumps({"statistics": {"channels": ["base", "filtered"]}}))
    assert "mean_filtered" in with_filtered, "opting in restores them"

    lean = config_io.query_fields(
        json.dumps({"statistics": {"reductions": ["mean"], "extremum": False}}))
    assert "mean_base" in lean
    assert "std_base" not in lean, "a disabled reduction disappears"
    assert "ext_base" not in lean, "so does the extremum"
    no_relevance = config_io.query_fields(
        json.dumps({"statistics": {"relevance": False}}))
    assert "relevance_base" not in no_relevance


def _one_component_stack(ext_values):
    """A Z-slice stack, one 2x2 component per slice, all overlapping. `ext_values`
    sets each slice's filtered minimum so a chosen slice is the deepest."""
    import numpy as np
    labels, base, filt = [], [], []
    for v in ext_values:
        labels.append(np.zeros((2, 2), dtype=np.int64))       # one MSC feature, id 0
        base.append(np.full((2, 2), 0.5, dtype=np.float32))
        f = np.full((2, 2), 0.9, dtype=np.float32)
        f[1, 1] = v                                           # the seeding pixel
        filt.append(f)
    for i, b in enumerate(base):
        b[1, 1] = 0.1 * (i + 1)                               # distinct ext_base per slice
    return labels, base, filt


def test_3d_extremum_is_the_deepest_slice_as_a_tuple():
    """The 3D feature inherits the extremum of its deepest constituent slice, and
    position and value come from the SAME slice."""
    from msseg.mscoupon import assembly
    # Deepest is the MIDDLE slice, so a first- or last-wins bug stays visible.
    labels, base, filt = _one_component_stack([0.8, 0.2, 0.6])
    out = assembly.assemble_cc(labels, [None] * 3, base, filt, connectivity=6,
                               ascending=True)
    assert out["n_global"] == 1
    row = out["global_table"][0]
    assert row["ext_filtered"] == pytest.approx(0.2)
    assert row["ext_z"] == 1, "the middle slice wins"
    assert row["ext_base"] == pytest.approx(0.2), "ext_base is that same slice's"
    assert (row["ext_x"], row["ext_y"]) == (1.0, 1.0)


def test_3d_extremum_flips_for_descending():
    from msseg.mscoupon import assembly
    labels, base, filt = _one_component_stack([0.8, 0.2, 0.6])
    out = assembly.assemble_cc(labels, [None] * 3, base, filt, connectivity=6,
                               ascending=False)
    row = out["global_table"][0]
    # Descending seeds from the maximum; 0.9 is the plateau value every slice
    # shares, so the winner is whichever slice attains it first -- the point is
    # simply that it is no longer the 0.2 minimum.
    assert row["ext_filtered"] == pytest.approx(0.9)


def test_3d_relevance_uses_percentile_slice_ranges():
    import numpy as np
    from msseg.mscoupon import assembly

    labels = [np.zeros((2, 2), dtype=np.int64) for _ in range(2)]
    base = [np.arange(4, dtype=np.float32).reshape(2, 2),
            np.arange(10, 14, dtype=np.float32).reshape(2, 2)]
    out = assembly.assemble_cc(
        labels, [None, None], base, base, connectivity=6,
        relevance_low_percentile=25.0, relevance_high_percentile=75.0)
    row = out["global_table"][0]
    # Per-slice ranges are [0.75,2.25] and [10.75,12.25]; global carry takes
    # the lowest floor and highest ceiling, while the feature spans [0,13].
    assert row["relevance_base"] == pytest.approx(13.0 / 11.5)
    assert assembly._relevance_base(0.0, 0.0, 0.0, 0.0) == 0.0
    assert np.isinf(assembly._relevance_base(0.0, 1.0, 0.0, 0.0))


def test_per_slice_reductions_run_across_slices():
    """area/bbox reductions describe how the footprint varies slice to slice --
    the voxel-pooled field statistics cannot say that."""
    import numpy as np
    from msseg.mscoupon import assembly
    # Areas 4, 2, 4 across three overlapping slices.
    labels = [np.zeros((2, 2), dtype=np.int64) for _ in range(3)]
    base = [np.full((2, 2), 0.5, dtype=np.float32) for _ in range(3)]
    filt = [np.full((2, 2), 0.5, dtype=np.float32) for _ in range(3)]
    # Trim the middle slice down to 2 pixels via a pixel rule on the base channel.
    base[1][1, :] = 0.0
    out = assembly.assemble_cc(labels, [None] * 3, base, filt,
                               pixel_rules=[{"channel": "base", "mode": "keep",
                                             "op": "gt", "value": 0.25}],
                               connectivity=6, ascending=True)
    row = out["global_table"][0]
    assert row["voxel_count"] == 10, "sum over slices is unchanged"
    assert row["area_min"] == pytest.approx(2.0)
    assert row["area_max"] == pytest.approx(4.0)
    assert row["area_mean"] == pytest.approx(10.0 / 3.0)


# --------------------------------------------------------------------------- #
# The no-data sentinel is a value, not always zero
# --------------------------------------------------------------------------- #
def test_get_valid_pixels_omits_a_configured_value():
    x = np.array([0.0, 43.0, 1.0, 2.0, 43.0, 3.0], dtype=np.float32)
    assert get_valid_pixels(x).tolist() == [43.0, 1.0, 2.0, 43.0, 3.0]      # default 0
    assert get_valid_pixels(x, omit_value=43).tolist() == [0.0, 1.0, 2.0, 3.0]
    assert get_valid_pixels(x, omit_value=None).tolist() == x.tolist()
    # The legacy boolean still works, and an explicit value wins over it.
    assert get_valid_pixels(x, omit_zeros=False).tolist() == x.tolist()
    assert get_valid_pixels(x, omit_value=43, omit_zeros=True).tolist() == [0.0, 1.0, 2.0, 3.0]


def test_sentinel_survives_float32_rounding():
    """The comparison happens in the image's dtype, so a sentinel that is not
    exactly representable still matches what was stored."""
    v = np.float32(0.1)
    x = np.array([v, 1.0, v, 2.0], dtype=np.float32)
    assert get_valid_pixels(x, omit_value=0.1).tolist() == [1.0, 2.0]


def test_gmm_recovers_populations_only_with_the_right_sentinel():
    """A 43-padded stack: the wrong sentinel leaves the plateau in as a
    population, which is the whole reason the value is configurable."""
    rng = np.random.default_rng(7)
    px = np.concatenate([rng.normal(10.0, 1.0, 5000),
                         rng.normal(30.0, 1.0, 5000),
                         np.full(20000, 43.0)]).astype(np.float32)

    naive = fit_two_gaussians(px, preset="two_gaussian", compute_hard_stats=False)
    assert naive["n_valid_pixels"] == px.size, "sentinel 0 matches nothing here"
    assert abs(naive["mu_2"] - 43.0) < 3.0, "the padding is fitted as a population"

    fixed = fit_two_gaussians(px, preset="two_gaussian", omit_value=43.0,
                              compute_hard_stats=False)
    assert fixed["n_valid_pixels"] == 10000
    assert abs(fixed["mu_1"] - 10.0) < 1.0
    assert abs(fixed["mu_2"] - 30.0) < 1.0


def test_normalize_filter_config_carries_the_sentinel():
    """A blank sentinel must export as null ("keep every pixel"), not vanish and
    fall back to the default of 0."""
    common = dict(files=["/data/a.tif"], output_folder="/out", filters=[])

    cfg = config_io.build_config(
        base_filters=[{"operation": "normalize",
                       "params": {"method": "gmm", "omit_value": 43.0}}], **common)
    assert cfg["base_filters"][0]["params"]["omit_value"] == 43.0

    blank = config_io.build_config(
        base_filters=[{"operation": "normalize",
                       "params": {"method": "gmm", "omit_value": ""}}], **common)
    assert "omit_value" in blank["base_filters"][0]["params"]
    assert blank["base_filters"][0]["params"]["omit_value"] is None


# --------------------------------------------------------------------------- #
# Derived measurement channels (the scale-space stack)
# --------------------------------------------------------------------------- #
def _stat_params(channels, reductions=("mean", "min", "max")):
    import json
    return json.dumps({"statistics": {"channels": list(channels),
                                      "reductions": list(reductions),
                                      "extremum": True}})


def test_derived_channels_expand_to_fields():
    """A sigma list is a cross-product: one entry per (kind, sigma), and two per
    sigma for hessian. Every channel contributes its enabled reductions."""
    params = _stat_params(["base",
                           {"kind": "blur", "sigmas": [0.7, 1.5, 3.0]},
                           {"kind": "hessian", "sigmas": [1.5]}])
    names = [c["name"] for c in config_io.stat_channels(params)]
    assert names == ["base", "blur_s0.7", "blur_s1.5", "blur_s3",
                     "hessian_largest_s1.5", "hessian_smallest_s1.5"], names

    fields = set(config_io.query_fields(params))
    for f in ("mean_blur_s0.7", "min_blur_s1.5", "max_blur_s3",
              "mean_hessian_largest_s1.5", "ext_hessian_smallest_s1.5"):
        assert f in fields, f
    assert "std_blur_s1.5" not in fields, "a disabled reduction stays off"
    assert "mean_blur_s2" not in fields, "an unrequested sigma has no field"


def test_channel_and_reduction_pickers_cover_every_field():
    """The GUI's two-level picker must reach every selectable field, and must not
    read the geometry columns as reductions of a channel."""
    params = _stat_params(["base", {"kind": "edges", "sigmas": [1.0, 2.0]}])
    order, by_channel = config_io.channels_and_reductions(params)
    fields = set(config_io.query_fields(params))

    covered = set()
    for channel in order:
        for reduction in by_channel[channel]:
            name = config_io.compose_field(channel, reduction)
            assert name in fields, (channel, reduction, name)
            covered.add(name)
    assert covered == fields, fields - covered

    assert config_io.split_field("mean_edges_s2", params) == ("edges_s2", "mean")
    # min_x is geometry, not the `min` reduction of a channel called `x`.
    assert config_io.split_field("min_x", params) == (config_io.GEOMETRY_CHANNEL, "min_x")


def test_statistics_block_round_trips_through_the_gui_shape():
    """The whole block used to be discarded on load except extremum_sample_radius,
    so a config naming a derived channel could not survive the viewer."""
    channels = [{"kind": "base"},
                {"kind": "blur", "sigmas": [0.7, 1.5]},
                {"kind": "hessian", "sigmas": [2.0], "sort_by_absolute_value": False}]
    cfg = config_io.build_config(
        files=["/data/a.tif"], output_folder="/out", filters=[],
        stat_channels=channels, stat_reductions=["mean", "max"])
    assert cfg["statistics"]["channels"][0] == "base"
    assert cfg["statistics"]["channels"][1] == {"kind": "blur", "sigmas": [0.7, 1.5]}
    assert cfg["statistics"]["channels"][2]["sort_by_absolute_value"] is False

    state = config_io.config_to_state(cfg)
    assert state["stat_reductions"] == ["mean", "max"]
    kinds = {c["kind"]: c for c in state["stat_channels"]}
    assert kinds["blur"]["sigmas"] == [0.7, 1.5]
    assert kinds["hessian"]["sort_by_absolute_value"] is False


def test_default_statistics_block_is_not_emitted():
    """A workflow that never touched the statistics panel exports exactly what it
    did before this feature existed."""
    cfg = config_io.build_config(
        files=["/data/a.tif"], output_folder="/out", filters=[],
        stat_channels=[{"kind": "base"}],
        stat_reductions=["mean", "min", "max", "std"])
    assert "statistics" not in cfg


def test_loading_a_derived_channel_config_keeps_its_queries():
    """A query naming a derived channel must be validated against the DOCUMENT's
    own statistics block, not the caller's current field list -- otherwise every
    such row is silently dropped on load."""
    cfg = config_io.build_config(
        files=["/data/a.tif"], output_folder="/out", filters=[],
        stat_channels=[{"kind": "base"}, {"kind": "blur", "sigmas": [1.5]}],
        stat_reductions=["mean", "min", "max"],
        feature_filters=[{"field": "mean_blur_s1.5", "op": "gt", "value": 0.5}])
    notes = []
    # `fields` deliberately does NOT mention the derived channel, mimicking a GUI
    # sitting on the default spec at the moment the config is opened.
    state = config_io.config_to_state(cfg, fields=["area", "mean_base"], notes=notes)
    assert [q["field"] for q in state["feature_filters"]] == ["mean_blur_s1.5"], notes


def test_3d_assembly_reports_derived_channels():
    """The GUI's numpy 3D assembly must produce the same field names the CLI's
    matcher does, for every measurement channel."""
    import numpy as np
    from msseg.mscoupon import assembly

    labels = [np.zeros((4, 4), dtype=np.int64) for _ in range(2)]
    base = [np.full((4, 4), 2.0, dtype=np.float32) for _ in range(2)]
    filt = [np.full((4, 4), 0.5, dtype=np.float32) for _ in range(2)]
    blur = [np.full((4, 4), 7.0, dtype=np.float32) for _ in range(2)]
    channels_list = [[("base", base[z]), ("blur_s1.5", blur[z])] for z in range(2)]

    out = assembly.assemble_cc(labels, [None] * 2, base, filt,
                               connectivity=6, ascending=True,
                               channels_list=channels_list,
                               reductions=["mean", "min", "max"])
    row = out["global_table"][0]
    assert row["mean_base"] == pytest.approx(2.0)
    assert row["mean_blur_s1.5"] == pytest.approx(7.0)
    assert row["min_blur_s1.5"] == pytest.approx(7.0)
    assert row["max_blur_s1.5"] == pytest.approx(7.0)
    # Every channel is sampled at the SAME seeding pixel.
    assert row["ext_blur_s1.5"] == pytest.approx(7.0)
    assert row["ext_base"] == pytest.approx(2.0)
    # A disabled reduction produces no field, on derived channels too.
    assert "std_blur_s1.5" not in row
