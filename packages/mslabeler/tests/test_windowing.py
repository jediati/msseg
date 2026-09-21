"""The lazy per-channel brightness window (msseg.labeler.windowing)."""
import numpy as np

from msseg.labeler import windowing
from msseg.labeler.sources import ArrayImageSource


def test_ramp_gives_the_percentiles_as_fractions():
    ramp = np.linspace(0.0, 1.0, 9999, dtype=np.float32).reshape(101, 99)
    src = ArrayImageSource(ramp)
    lo, hi = windowing.source_window(src, np)
    assert abs(lo - 0.01) < 2e-3 and abs(hi - 0.99) < 2e-3, (lo, hi)
    # not the data's own units: fractions of the source's range
    src2 = ArrayImageSource(ramp * 100.0 - 50.0)
    lo2, hi2 = windowing.source_window(src2, np)
    assert abs(lo2 - lo) < 1e-5 and abs(hi2 - hi) < 1e-5


def test_degenerate_inputs_open_the_full_range():
    assert windowing.percentile_window(np.zeros(0), (0.0, 1.0)) == (0.0, 1.0)
    assert windowing.percentile_window(np.ones(50), (1.0, 1.0)) == (0.0, 1.0)
    assert windowing.percentile_window(np.ones(50), (0.0, 2.0)) == (0.0, 1.0), \
        "a constant image has equal percentiles: full range, not an empty window"
    assert windowing.percentile_window(np.ones(5), (float("nan"), 1.0)) == (0.0, 1.0)
    assert windowing.percentile_window(np.ones(5), None) == (0.0, 1.0)
    nan = np.full((8, 8), np.nan, np.float32)
    assert windowing.window_sample(ArrayImageSource(nan), np).size == 0


def test_nans_are_left_out_of_the_sample():
    a = np.linspace(0.0, 1.0, 400, dtype=np.float32).reshape(20, 20)
    a[0, :5] = np.nan
    s = windowing.window_sample(ArrayImageSource(a), np)
    assert s.size == 395 and np.isfinite(s).all()


def test_colour_planes_are_pooled():
    planes = np.stack([np.zeros((10, 10)), np.ones((10, 10)), np.full((10, 10), 0.5)])
    src = ArrayImageSource(planes.astype(np.float32))
    s = windowing.window_sample(src, np)
    assert s.size == 300 and s.min() == 0.0 and s.max() == 1.0


def test_big_arrays_are_strided_to_the_budget():
    big = np.random.default_rng(0).random((4000, 4000), dtype=np.float32)
    s = windowing.window_sample(ArrayImageSource(big), np, budget=1_000_000)
    assert 250_000 <= s.size <= 1_000_000, s.size


def test_uint8_range_maps_by_the_dtype_range():
    # A source whose value_range is the dtype's (a uint8 pyramid): the
    # fractions are p/255, whatever the data happens to cover.
    class Src:
        levels = 1
        def level_shape(self, level):
            return (16, 16)
        def read_region(self, level, x, y, w, h):
            return np.full((h, w), 51, np.uint8)
        def value_range(self):
            return (0.0, 255.0)
    lo, hi = windowing.percentile_window(windowing.window_sample(Src(), np), (0.0, 255.0))
    assert (lo, hi) == (0.0, 1.0), "constant: full range"
    sample = np.array([51, 204], np.float32)
    lo, hi = windowing.percentile_window(sample, (0.0, 255.0), lo=0, hi=100)
    assert abs(lo - 51 / 255) < 1e-6 and abs(hi - 204 / 255) < 1e-6


def test_pyramid_like_source_reads_its_coarsest_level():
    calls = []

    class Src:
        levels = 4
        def level_shape(self, level):
            return (64 >> level, 64 >> level)
        def read_region(self, level, x, y, w, h):
            calls.append((level, x, y, w, h))
            return np.linspace(0.0, 1.0, w * h, dtype=np.float32).reshape(h, w)
        def value_range(self):
            return (0.0, 1.0)
    s = windowing.window_sample(Src(), np)
    assert calls == [(3, 0, 0, 8, 8)] and s.size == 64


def test_placed_raster_is_sampled_not_its_fill():
    """A placed item raster on a huge slide: the sample is the raster, whose
    read_region would otherwise fill the slide with the raster's minimum."""
    class Placed:
        levels = 6
        def __init__(self):
            self.raster = np.linspace(0.2, 0.8, 900, dtype=np.float32).reshape(30, 30)
        def level_shape(self, level):
            return (90000 >> level, 47040 >> level)
        def read_region(self, *a):
            raise AssertionError("the raster fast path must be used")
        def value_range(self):
            return (0.2, 0.8)
    src = Placed()
    s = windowing.window_sample(src, np)
    assert s.size == 900
    lo, hi = windowing.source_window(src, np)
    assert 0.0 < lo < 0.05 and 0.95 < hi < 1.0
