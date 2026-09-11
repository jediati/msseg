"""``PyramidImageSource``: tile assembly, the LRU, and the level arithmetic.

The reader backends need packages a CI box may not have (OpenSlide's native
library, ``large_image``, ``imagecodecs``), and what is worth pinning is not
their decoding but everything this module does around it -- snapping a rect to
the tile grid, stitching the tiles back into exactly the requested rect,
zero-filling past an edge, evicting by bytes, and reporting a level's true
downsample. A fake backend over an in-memory pyramid exercises all of that,
and counts its own reads so the cache can be proved to work.
"""
import numpy as np
import pytest

from msseg.labeler import pyramid as P


class FakeBackend:
    """Levels are exact halvings of a base array, so a level pixel's value
    identifies its position and a wrong offset shows up as a wrong value."""
    name = "fake"

    def __init__(self, h=300, w=400, channels=3, levels=4, dtype=np.uint8):
        self.dtype = np.dtype(dtype)
        self.channels = channels
        rng = np.random.default_rng(0)
        shape = (h, w, channels) if channels > 1 else (h, w)
        top = (rng.random(shape) * 200 + 5).astype(self.dtype)
        self._levels = [top]
        for _ in range(levels - 1):
            prev = self._levels[-1]
            self._levels.append(np.ascontiguousarray(prev[::2, ::2]))
        self.level_count = len(self._levels)
        self.height, self.width = h, w
        self.reads = 0
        self.read_px = 0

    def close(self):
        pass

    def downsample(self, level):
        return float(self.height) / float(self._levels[level].shape[0])

    def level_dims(self, level):
        s = self._levels[level].shape
        return int(s[1]), int(s[0])

    def read(self, level, x, y, w, h):
        self.reads += 1
        self.read_px += w * h
        return self._levels[level][y:y + h, x:x + w]


def source(tile=64, cache_mb=64.0, **kw):
    be = FakeBackend(**kw)
    src = P.PyramidImageSource.__new__(P.PyramidImageSource)
    src.path = "<fake>"
    src.be = be
    src.tile = tile
    src._budget = int(cache_mb * (1 << 20))
    src._cache = P.OrderedDict()
    src._bytes = 0
    src.hits = src.misses = 0
    src._range = None
    return src


def test_protocol_surface_and_level_arithmetic():
    src = source()
    assert (src.levels, src.channels, src.native) == (4, 3, True)
    assert src.level_shape(0) == (300, 400) and src.level_shape(2) == (75, 100)
    assert src.level_scale(0) == 1.0 and src.level_scale(1) == 2.0
    # a level out of range clamps rather than raising
    assert src.level_shape(99) == src.level_shape(3)
    # uint8 windows in its dtype range without touching the file
    assert src.value_range() == (0.0, 255.0) and src.be.reads == 0
    # coarsest level still finer than the requested scale
    assert (src.best_level(1.0), src.best_level(4.0), src.best_level(1e6)) == (0, 2, 3)


def test_float_value_range_is_measured_on_the_coarsest_level():
    src = source(dtype=np.float32)
    lo, hi = src.value_range()
    top = src.be._levels[-1]
    assert (lo, hi) == (float(top.min()), float(top.max()))


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("rect", [(0, 0, 64, 64), (30, 20, 50, 40), (100, 90, 200, 150),
                                  (63, 63, 3, 3), (0, 0, 400, 300)])
def test_read_region_matches_a_plain_slice(channels, rect):
    """Whatever the tile grid, the assembled rect must equal the level's own
    slice -- this is the test that catches an off-by-one in the stitch."""
    src = source(channels=channels)
    x, y, w, h = rect
    got = src.read_region(0, x, y, w, h)
    assert got.shape == ((h, w, channels) if channels > 1 else (h, w))
    assert np.array_equal(got, src.be._levels[0][y:y + h, x:x + w])


def test_read_region_at_a_level_is_that_levels_slice():
    src = source()
    got = src.read_region(2, 10, 5, 40, 30)
    assert np.array_equal(got, src.be._levels[2][5:35, 10:50])


def test_off_edge_reads_are_zero_filled_not_clipped():
    """A halo'd ROI reads straight off an edge; it must still get the shape it
    asked for, with the outside zeroed."""
    src = source()
    got = src.read_region(0, 380, 290, 40, 30)
    assert got.shape == (30, 40, 3)
    assert np.array_equal(got[:10, :20], src.be._levels[0][290:300, 380:400])
    assert not got[10:].any() and not got[:, 20:].any()
    # entirely outside: all zeros, and nothing was read
    before = src.be.reads
    assert not src.read_region(0, 1000, 1000, 16, 16).any()
    assert src.be.reads == before


def test_negative_origin_places_the_data_at_the_right_offset():
    src = source()
    got = src.read_region(0, -10, -5, 40, 30)
    assert not got[:5].any() and not got[:, :10].any()
    assert np.array_equal(got[5:, 10:], src.be._levels[0][0:25, 0:30])


def test_tiles_are_cached_and_reused_across_overlapping_reads():
    src = source(tile=64)
    a = src.read_region(0, 100, 100, 128, 128)
    cold_reads = src.be.reads
    assert cold_reads > 1 and src.misses == cold_reads and src.hits == 0
    b = src.read_region(0, 100, 100, 128, 128)          # identical: all hits
    assert np.array_equal(a, b)
    assert src.be.reads == cold_reads and src.hits == cold_reads
    src.read_region(0, 110, 110, 128, 128)              # shifted: mostly hits
    assert src.be.reads < 2 * cold_reads
    assert src.cache_bytes > 0
    src.clear_cache()
    assert src.cache_bytes == 0


def test_the_cache_evicts_by_bytes_and_always_keeps_one_tile():
    # a budget below a single tile: reads still succeed, one entry survives
    src = source(tile=64, cache_mb=64 * 64 * 3 / (1 << 20) * 1.5)
    got = src.read_region(0, 0, 0, 256, 256)
    assert np.array_equal(got, src.be._levels[0][0:256, 0:256])
    assert len(src._cache) == 1 and src.cache_bytes == src._cache[next(iter(src._cache))].nbytes


def test_a_grayscale_backend_is_broadcast_when_channels_says_colour():
    """The protocol's channel count is what the canvas trusts, so a backend
    that hands back 2-D data for a colour source must be widened, not left to
    break the composite."""
    src = source(channels=1)
    src.be.channels = 3                       # claim colour, still serve 2-D
    got = src.read_region(0, 0, 0, 32, 32)
    assert got.shape == (32, 32, 3)
    assert np.array_equal(got[..., 0], got[..., 2])


def test_backends_available_is_a_subset_of_the_known_names():
    names = P.backends_available()
    known = ("openslide", "large_image", "tifffile", "array")
    assert set(names) <= set(known)
    assert names == [n for n in known if n in names]


def test_open_backend_reports_every_failure():
    with pytest.raises(RuntimeError) as e:
        P.open_backend("no-such-file.svs")
    msg = str(e.value)
    assert "no pyramid backend could open" in msg
    for name in P.backends_available():
        assert name in msg, "an installed backend's failure must be named"


# --------------------------------------------------------------------------- #
# the in-memory backend: a plain TIFF is a one-level pyramid held whole
# --------------------------------------------------------------------------- #
def test_halve_is_a_box_average_in_the_arrays_dtype():
    a = np.array([[0, 2, 10, 10], [4, 6, 10, 10]], np.uint8)
    got = P._halve(a)
    assert got.dtype == np.uint8 and got.tolist() == [[3, 10]]
    f = P._halve(np.ones((3, 5), np.float32) * 2.5)           # odd edges drop
    assert f.shape == (1, 2) and f.dtype == np.float32 and (f == 2.5).all()
    rgb = P._halve(np.full((4, 4, 3), 7, np.uint8))
    assert rgb.shape == (2, 2, 3) and (rgb == 7).all()


def test_a_plain_strip_tiff_opens_through_the_array_backend(tmp_path):
    """OpenSlide refuses an untiled file and the tifffile store needs zarr;
    the first real folder had exactly such a file beside the slide, and the
    whole run died on it."""
    tifffile = pytest.importorskip("tifffile")
    rng = np.random.default_rng(1)
    img = (rng.random((600, 900, 3)) * 255).astype(np.uint8)
    path = tmp_path / "crop.tiff"
    tifffile.imwrite(str(path), img, photometric="rgb")     # strips, no pyramid
    src = P.PyramidImageSource(str(path))
    assert src.be.name == "array"
    assert (src.channels, src.level_shape(0), src.level_scale(0)) == (3, (600, 900), 1.0)
    assert src.levels > 1 and max(src.level_shape(src.levels - 1)) <= 256
    assert np.array_equal(src.read_region(0, 100, 50, 64, 32), img[50:82, 100:164])
    assert src.value_range() == (0.0, 255.0)
    # level 1 is the 2x2 average of level 0, not a decimation
    l1 = src.read_region(1, 0, 0, 4, 4)
    want = P._halve(img)[:4, :4]
    assert np.array_equal(l1, want)


def test_a_grayscale_plain_tiff_is_one_channel(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    img = np.arange(300 * 400, dtype=np.uint16).reshape(300, 400)
    path = tmp_path / "gray.tiff"
    tifffile.imwrite(str(path), img)
    src = P.PyramidImageSource(str(path))
    assert src.be.name == "array" and src.channels == 1
    assert src.read_region(0, 0, 0, 8, 8).shape == (8, 8)
    assert src.value_range() == (0.0, 65535.0)


def test_the_in_memory_budget_is_enforced(tmp_path, monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "big.tiff"
    tifffile.imwrite(str(path), np.zeros((64, 64, 3), np.uint8), photometric="rgb")
    monkeypatch.setattr(P, "IN_MEMORY_BUDGET", 1000)       # bytes
    with pytest.raises(RuntimeError) as e:
        P.open_backend(str(path), backend="array")
    assert "budget" in str(e.value)
