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
# Statistics schema + the 3D seeding extremum
# --------------------------------------------------------------------------- #
def test_query_fields_drops_feature_id_and_filtered_aggregates():
    names = config_io.query_fields()
    assert "feature_id" not in names, "an id is not a selectable statistic"
    assert "mean_base" in names and "ext_base" in names
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
