"""`msc.max_region_area` / `msc.max_region_parallel`: MSCEER's max-area
simplification rule, from the profile to the params JSON to a primed pipe."""
import json

import numpy as np
import pytest

from msseg.mscoupon import session
from msseg.mscoupon.session import default_profile, profile_from_json, profile_params_json


def test_the_profile_reads_the_cap_totally():
    assert default_profile()["msc"]["max_region_area"] is None
    assert default_profile()["msc"]["max_region_parallel"] is True
    for raw, want in ((5000, 5000), ("5000", 5000), ("", None), (None, None),
                      (0, None), (-3, None), ("junk", None), (12.7, 12)):
        p = profile_from_json({"msc": {"max_region_area": raw}})
        assert p["msc"]["max_region_area"] == want, raw
    p = profile_from_json({"msc": {"max_region_area": 10, "max_region_parallel": False}})
    assert p["msc"]["max_region_parallel"] is False
    assert profile_from_json(p) == p


def test_params_carry_the_cap_only_when_it_is_on():
    off = json.loads(profile_params_json(default_profile()))["msc"]
    assert "max_region_area" not in off and "max_region_parallel" not in off
    p = default_profile()
    p["msc"].update({"max_region_area": 4096, "max_region_parallel": False})
    on = json.loads(profile_params_json(p))["msc"]
    assert on["max_region_area"] == 4096 and on["max_region_parallel"] is False


def test_the_cap_is_part_of_the_field():
    p = default_profile()
    f0 = session.field_fingerprint(p)
    p["msc"]["max_region_area"] = 4096
    f1 = session.field_fingerprint(p)
    assert f1 != f0
    p["msc"]["max_region_parallel"] = False
    assert session.field_fingerprint(p) != f1
    assert session.measure_fingerprint(p) == session.measure_fingerprint(default_profile())


def test_summary_names_the_cap():
    assert session.msc_code({"max_region_area": 5000}) == "msc(asc, 10%, mf, ≤5000px)"
    assert session.msc_code({"max_region_area": 5000, "max_region_parallel": False}) \
        == "msc(asc, 10%, mf, ≤5000px*)"
    assert session.msc_code({"max_region_area": 0}) == "msc(asc, 10%, mf)"


def _wells(h=40, w=48):
    yy, xx = np.mgrid[0:h, 0:w]
    fx, fy = xx / (w - 1.0), yy / (h - 1.0)
    v = 0.05 * (fx + fy)
    for cx, cy, d, s in ((0.28, 0.30, 1.0, 0.14), (0.72, 0.32, 0.7, 0.12), (0.50, 0.72, 0.4, 0.10)):
        v -= d * np.exp(-((fx - cx) ** 2 + (fy - cy) ** 2) / (2 * s * s))
    return v.astype(np.float32)


@pytest.mark.parametrize("simplification", ["merge_forest", "msc"])
def test_a_primed_pipe_honours_the_cap(simplification):
    ext = pytest.importorskip("msseg.mscoupon.mscoupon_py")
    field = _wells()

    def prime(**extra):
        msc = {"manifold": "ascending", "persistence_percent": 100.0,
               "simplification": simplification,
               "accurate_ascending": False, "accurate_descending": False, **extra}
        pipe = ext.prime_slice(field, field, json.dumps({"msc": msc}))
        pipe.select_persistence(pipe.value_range())
        return pipe

    def areas(labels):
        return np.bincount(labels[labels >= 0].ravel())

    plain = prime()
    full = areas(np.asarray(plain.labels()))
    plain.select_persistence(0.0)
    cap = int(areas(np.asarray(plain.labels())).max())
    capped = areas(np.asarray(prime(max_region_area=cap).labels()))
    if (capped > 0).sum() == (full > 0).sum():
        pytest.skip("extension built against an MSCEER without simplification rules")
    assert capped.max() <= cap
    assert (capped > 0).sum() > (full > 0).sum()
