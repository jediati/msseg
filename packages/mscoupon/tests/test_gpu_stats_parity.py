"""The device statistics path against the host loop, on the specs that used
to be host-only: colour planes, per-plane derived channels, histograms.

Exact where the CPU is exact (area, bbox, min, max, extremum sample, every
histogram count); the sums differ only in floating-point association."""
import json

import numpy as np
import pytest


def _prime(engine, base, filt, planes, spec, gpu):
    params = {"msc": {"manifold": "ascending", "persistence_percent": 5.0,
                      "accurate_ascending": False, "accurate_descending": False,
                      "use_gpu_gradient": gpu, "use_gpu_stats": gpu},
              "input": {"color": {"channels": 3}},
              "statistics": spec}
    return engine.prime_slice(base, filt, json.dumps(params), planes)


@pytest.mark.parametrize("spec", [
    {"channels": ["base", "color"], "reductions": ["mean", "min", "max", "std"], "extremum": True},
    {"channels": ["base", {"kind": "blur", "sigmas": [1.0, 2.5], "source": "color"},
                  {"kind": "gradmag", "sigmas": [1.5], "source": "color"},
                  {"kind": "hessian", "sigmas": [2.0], "source": "color"},
                  {"kind": "laplacian", "sigmas": [1.0]}],
     "reductions": ["mean", "min", "max", "std"], "extremum": True},
    {"channels": ["base", {"kind": "blur", "sigmas": [1.5], "source": "color"}],
     "reductions": ["mean", "max"], "extremum": True,
     "histogram": {"bins": 8, "channels": ["base", "blur_c1_s1.5"], "ranges": {"*": [0.0, 1.0]}}},
    # A named measurement source. The device path DECLINES this one -- a source's
    # planes are built on the host and would collide with the colour slots -- so
    # what this case pins is that the fallback is correct rather than silent: the
    # host numbers must still be the host numbers, and `use_gpu_stats` must not
    # have quietly produced them from the wrong bank.
    {"sources": {"he": [{"operation": "stain_deconvolution", "params": {"preset": "he"}}]},
     "channels": ["base", "he", {"kind": "blur", "sigmas": [1.5], "source": "he"}],
     "reductions": ["mean", "min", "max", "std"], "extremum": True},
])
def test_device_path_matches_host_loop(spec):
    engine = pytest.importorskip("msseg.mscoupon")
    rng = np.random.default_rng(7)
    h, w = 96, 128
    planes = rng.random((3, h, w), dtype=np.float32)
    base = planes.mean(axis=0).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    filt = (np.sin(xx / 7.0) * np.cos(yy / 5.0) + 0.05 * rng.random((h, w))).astype(np.float32)
    cpu = _prime(engine, base, filt, planes, spec, gpu=False)
    gpu = _prime(engine, base, filt, planes, spec, gpu=True)
    if "gpu_stats" not in gpu.build_timings():
        # Either no device path in this build, or the spec declined it. A
        # declined spec must still agree with itself: the host loop ran on both
        # sides, so the rows have to match exactly.
        if spec.get("sources"):
            assert np.array_equal(cpu.labels(), gpu.labels())
            names_c, rows_c = cpu.feature_table()
            names_g, rows_g = gpu.feature_table()
            assert list(names_c) == list(names_g)
            assert np.array_equal(rows_c, rows_g), "a declined spec falls back, it does not drift"
        pytest.skip("no device statistics path in this build / on this machine")
    assert "gpu_stats" not in cpu.build_timings()
    assert np.array_equal(cpu.labels(), gpu.labels()), "the GPU gradient is bit-identical"
    names_c, tab_c = cpu.feature_table()
    names_g, tab_g = gpu.feature_table()
    assert list(names_c) == list(names_g)
    assert tab_c.shape == tab_g.shape and tab_c.shape[0] > 3
    for j, name in enumerate(names_c):
        exact = name.startswith(("min_", "max_", "ext_", "hist", "area", "bbox", "feature_id",
                                 "min_x", "max_x", "min_y", "max_y"))
        if exact:
            assert np.array_equal(tab_c[:, j], tab_g[:, j]), name
        else:
            assert np.allclose(tab_c[:, j], tab_g[:, j], rtol=1e-5, atol=1e-6), name
