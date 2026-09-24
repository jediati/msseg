"""``ComputeEngine`` re-measures a primed slice when only the statistics moved:
the record's columns follow the spec, its labels do not change, and a slice
the engine was never told about (no stamp) is left alone."""
import json

import numpy as np
import pytest


def _params(channels):
    return json.dumps({"msc": {"manifold": "ascending", "persistence_percent": 5.0,
                               "accurate_ascending": False, "accurate_descending": False},
                       "statistics": {"channels": channels, "reductions": ["mean", "max"],
                                      "extremum": True}})


def _primed(ext, params):
    from msseg.mscoupon.engine import ComputeEngine
    from msseg.mscoupon.fingerprints import measure_fingerprint_of
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:64, 0:80]
    filt = (np.sin(xx / 5.0) * np.cos(yy / 4.0) + 0.05 * rng.random((64, 80))).astype(np.float32)
    base = (0.5 * filt + 0.2).astype(np.float32)
    pipe = ext.prime_slice(base, filt, params)
    eng = ComputeEngine(lambda *a: {})
    eng.primed = [{"files": ["a.tif"], "base": [base], "filtered": [filt], "pipes": [pipe],
                   "normalizers": [None], "color": [None],
                   "measured": [measure_fingerprint_of(params)]}]
    return eng, base, filt


def _slice_params(params):
    return {"pct": 5.0, "queries": [], "min_area": None, "json": params}


def test_a_statistics_change_remeasures_the_slice_it_reads():
    ext = pytest.importorskip("msseg.mscoupon.mscoupon_py")
    if not hasattr(ext.Msc2DPipeline, "remeasure"):
        pytest.skip("extension predates Msc2DPipeline.remeasure")
    a = _params(["base"])
    b = _params(["base", {"kind": "blur", "sigmas": [1.5]}])
    eng, base, filt = _primed(ext, a)
    rec_a = eng.ensure_slice(0, 0, _slice_params(a), ext, np)
    assert "mean_blur_s1.5" not in rec_a["stats"].names
    assert not eng.measure_stale(0, 0, a) and eng.measure_stale(0, 0, b)

    eng.commit_selection()
    rec_b = eng.ensure_slice(0, 0, _slice_params(b), ext, np)
    assert "mean_blur_s1.5" in rec_b["stats"].names
    assert np.array_equal(rec_a["labels"], rec_b["labels"])
    assert not eng.measure_stale(0, 0, b)
    fresh = ext.prime_slice(base, filt, b)
    fresh.select_persistence(fresh.value_range() * 0.05)
    names, values = fresh.feature_table()
    assert list(names) == rec_b["stats"].names
    assert np.array_equal(values, rec_b["stats"].values, equal_nan=True)


def test_an_unstamped_record_is_taken_as_current():
    ext = pytest.importorskip("msseg.mscoupon.mscoupon_py")
    a = _params(["base"])
    eng, _b, _f = _primed(ext, a)
    del eng.primed[0]["measured"]
    b = _params(["base", "filtered"])
    rec = eng.ensure_slice(0, 0, _slice_params(b), ext, np)
    assert "mean_filtered" not in rec["stats"].names, "no stamp, no re-measure"
    assert eng.primed[0]["measured"] is not None
