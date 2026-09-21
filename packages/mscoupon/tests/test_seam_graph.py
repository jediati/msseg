"""C++ `mscoupon_py.seam_graph` against the numpy reference
`msseg.labeler.seams.seams_from_labels`, array for array. Skips without the
compiled extension (or an extension that predates the symbol).

    pytest packages/mscoupon/tests/test_seam_graph.py
"""
import numpy as np
import pytest

from msseg.labeler.seams import seams_from_labels

mscoupon = pytest.importorskip("msseg.mscoupon")
_ext = getattr(mscoupon, "_ext", None)
if _ext is None or not hasattr(_ext, "seam_graph"):
    pytest.skip("mscoupon_py.seam_graph is not built", allow_module_level=True)


def random_raster(rng):
    h, w = rng.integers(2, 48, size=2)
    k = int(rng.integers(1, 12))
    seeds = rng.integers(0, [w, h], size=(k, 2))
    ys, xs = np.mgrid[0:h, 0:w]
    d = (xs[..., None] - seeds[:, 0]) ** 2 + (ys[..., None] - seeds[:, 1]) ** 2
    lab = np.argmin(d, axis=-1).astype(np.int32) * int(rng.integers(1, 5))
    lab[rng.random((h, w)) < 0.1] = -1                      # holes and gaps
    if rng.random() < 0.4:
        lab[:, -1] = -1
    if rng.random() < 0.4:
        lab[h // 2, w // 3] = 777                           # a one-pixel region
    if rng.random() < 0.3:                                  # an island
        y0, x0 = int(rng.integers(0, h)), int(rng.integers(0, w))
        lab[y0:y0 + 3, x0:x0 + 2] = 555
    return lab


def assert_same(lab):
    ref = seams_from_labels(lab, np)
    got = _ext.seam_graph(np.ascontiguousarray(lab, np.int32))
    assert len(got) == 7
    for r, g in zip(ref, got):
        assert np.asarray(g).shape == np.asarray(r).shape
        assert np.array_equal(np.asarray(g), np.asarray(r))


def test_parity_on_random_rasters():
    rng = np.random.default_rng(2026)
    for _ in range(40):
        assert_same(random_raster(rng))


def test_parity_on_hand_cases():
    lab = np.zeros((12, 12), np.int32)
    lab[:6, 6:] = 1
    lab[6:, :6] = 2
    lab[6:, 6:] = 3
    assert_same(lab)
    lab = np.zeros((10, 10), np.int32)
    lab[3:7, 2:5] = 1
    assert_same(lab)
    assert_same(np.array([[0, 1], [-1, 2]], np.int32))
    assert_same(np.zeros((0, 0), np.int32))
    assert_same(np.zeros((4, 4), np.int32))


def test_dtype_coercion_and_shape_errors():
    lab = np.array([[0, 1], [2, 3]], np.int64)
    a, b, j0, j1, offsets, points, jxy = _ext.seam_graph(lab)
    assert a.dtype == np.int32 and offsets.dtype == np.int64 and points.shape[1] == 2
    with pytest.raises(RuntimeError):
        _ext.seam_graph(np.zeros(4, np.int32))
