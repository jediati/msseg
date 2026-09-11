"""Tiled/pyramidal slide reading: the base image of a gigapixel labeler.

A whole-slide file is a pyramid of tiled levels -- 90 000 x 47 040 RGB at
level 0, halving to a thumbnail, stored as JPEG-compressed 256 x 256 tiles --
so the base image is never resident and every read is a window at a level.
``PyramidImageSource`` is the ``ImageSource`` over such a file.

Three things this module exists to get right:

* **Whatever reader is installed.** OpenSlide, ``large_image`` and ``tifffile``
  + ``zarr`` all read tiled pyramids and none of them is a safe hard
  dependency: tifffile needs ``imagecodecs`` for a JPEG-compressed file,
  ``large_image`` is a heavy install, and OpenSlide needs its native library.
  So the backends are tried in order and the first that opens the file wins;
  ``backends_available()`` reports what a machine actually has.
* **Native values, in colour.** The reader hands back the file's own dtype and
  every sample, so the canvas windows in the data's real range and a pathology
  slide is not silently reduced to grayscale on the way in.
* **Tiles, cached.** A pan re-reads mostly the same pixels; a decode per frame
  is what makes a pyramid feel slower than an array. Reads are assembled from
  a tile grid held in a byte-budgeted LRU, so a drag costs the newly exposed
  tiles only.

Level ``k`` is the file's own level ``k``, and ``level_scale(k)`` is its true
downsample read from the file rather than assumed to be ``2**k`` -- levels
floor their dimensions, so the deep levels of a 90 000-row slide come out at
515.6x, not 512x.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
class _Backend:
    """The reader seam: ``width``/``height``/``level_count``/``channels``/
    ``dtype``, plus ``downsample(k)``, ``level_dims(k)`` and
    ``read(level, x, y, w, h)``.

    ``read`` takes LEVEL coordinates and returns ``(h, w)`` or ``(h, w, C)`` in
    the file's own dtype; the caller has already clipped the rect to the level.
    """
    name = "?"


class _OpenSlideBackend(_Backend):
    """OpenSlide: its own decoders, so a JPEG-tiled TIFF needs nothing extra.
    ``read_region`` takes the location in LEVEL-0 pixels but the size in level
    pixels -- the one asymmetry worth naming."""
    name = "openslide"

    def __init__(self, path):
        import openslide
        self._sl = openslide.OpenSlide(str(path))
        self.width, self.height = self._sl.dimensions
        self.level_count = int(self._sl.level_count)
        self._ds = [float(d) for d in self._sl.level_downsamples]
        self._dims = [(int(w), int(h)) for w, h in self._sl.level_dimensions]
        self.dtype = np.dtype(np.uint8)
        self.channels = 3                      # served RGBA; the alpha is dropped

    def close(self):
        self._sl.close()

    def downsample(self, level: int) -> float:
        return self._ds[level]

    def level_dims(self, level: int) -> Tuple[int, int]:
        return self._dims[level]

    def read(self, level, x, y, w, h):
        s = self._ds[level]
        im = self._sl.read_region((int(round(x * s)), int(round(y * s))), int(level),
                                  (int(w), int(h)))
        return np.asarray(im, dtype=np.uint8)[..., :3]


class _LargeImageBackend(_Backend):
    """large_image. Its own level numbering runs the other way, so this keeps
    the module's convention (0 = full resolution) and asks for regions in base
    pixels, which is numbering-independent."""
    name = "large_image"

    def __init__(self, path):
        import large_image
        self._src = large_image.open(str(path))
        md = self._src.getMetadata()
        self.width, self.height = int(md["sizeX"]), int(md["sizeY"])
        self.level_count = max(1, int(md.get("levels") or 1))
        self.dtype = np.dtype(np.uint8)
        self.channels = 3

    def close(self):
        pass

    def downsample(self, level: int) -> float:
        return float(2 ** int(level))

    def level_dims(self, level: int) -> Tuple[int, int]:
        s = self.downsample(level)
        return max(1, int(self.width / s)), max(1, int(self.height / s))

    def read(self, level, x, y, w, h):
        import large_image
        s = self.downsample(level)
        arr, _ = self._src.getRegion(
            region={"left": int(x * s), "top": int(y * s),
                    "right": int((x + w) * s), "bottom": int((y + h) * s),
                    "units": "base_pixels"},
            output={"maxWidth": int(w), "maxHeight": int(h)},
            format=large_image.constants.TILE_FORMAT_NUMPY)
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[..., :3]
        return arr


class _TiffFileBackend(_Backend):
    """tifffile through its zarr store. Needs ``zarr``, and ``imagecodecs`` for
    anything but an uncompressed file -- which is why it is tried last."""
    name = "tifffile"

    def __init__(self, path):
        import tifffile
        import zarr
        self._tf = tifffile.TiffFile(str(path))
        series = self._tf.series[0]
        self._store = series.aszarr()
        node = zarr.open(self._store, mode="r")
        n = len(series.levels)
        self._arrays = [node[str(i)] for i in range(n)] if n > 1 else [node]
        a0 = self._arrays[0]
        self.height, self.width = int(a0.shape[0]), int(a0.shape[1])
        self.level_count = len(self._arrays)
        self.dtype = np.dtype(a0.dtype)
        self.channels = int(a0.shape[2]) if a0.ndim == 3 else 1

    def close(self):
        self._store.close()
        self._tf.close()

    def downsample(self, level: int) -> float:
        return float(self.height) / float(self._arrays[level].shape[0])

    def level_dims(self, level: int) -> Tuple[int, int]:
        s = self._arrays[level].shape
        return int(s[1]), int(s[0])

    def read(self, level, x, y, w, h):
        arr = np.asarray(self._arrays[level][y:y + h, x:x + w])
        if arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[..., :3]
        return arr


# The largest image the in-memory backend will decode whole. Above this a
# file needs a real tiled reader; below it, building the ladder ourselves is
# cheaper than any of them.
IN_MEMORY_BUDGET = 512 * (1 << 20)


class _ArrayBackend(_Backend):
    """The last resort: decode the whole image with tifffile and build the
    level ladder in memory by 2x2 averaging.

    A plain strip TIFF -- an exported crop, a coupon slice, anything that is
    not tiled and pyramidal -- is refused by OpenSlide and needs ``zarr``
    under tifffile's store, yet a 38 MB image is trivially held whole. So it
    is, up to ``IN_MEMORY_BUDGET``: the same ``ImageSource`` contract, served
    from arrays, and a file the tiled readers cannot open still opens.
    """
    name = "array"

    def __init__(self, path):
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            page = tf.pages[0]
            if page.nbytes > IN_MEMORY_BUDGET:
                raise ValueError(f"{page.nbytes / 1e6:.0f} MB exceeds the in-memory "
                                 f"budget of {IN_MEMORY_BUDGET / 1e6:.0f} MB")
            a = page.asarray()
        if a.ndim == 3 and a.shape[2] > 3:
            a = a[..., :3]
        if a.ndim == 3 and a.shape[2] == 2:            # gray + alpha
            a = a[..., :1]
        if a.ndim == 3 and a.shape[2] == 1:
            a = a[..., 0]
        self.dtype = np.dtype(a.dtype)
        self.channels = int(a.shape[2]) if a.ndim == 3 else 1
        self.height, self.width = int(a.shape[0]), int(a.shape[1])
        self._levels = [np.ascontiguousarray(a)]
        while max(self._levels[-1].shape[:2]) > 256 and len(self._levels) < 16:
            self._levels.append(_halve(self._levels[-1]))
        self.level_count = len(self._levels)

    def close(self):
        self._levels = []

    def downsample(self, level: int) -> float:
        return float(self.height) / float(self._levels[level].shape[0])

    def level_dims(self, level: int) -> Tuple[int, int]:
        sh = self._levels[level].shape
        return int(sh[1]), int(sh[0])

    def read(self, level, x, y, w, h):
        return self._levels[level][y:y + h, x:x + w]


def _halve(a):
    """2x2 box average in the array's own dtype (odd edges drop a row/col)."""
    h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    if h < 2 or w < 2:
        return a[:max(1, h // 2), :max(1, w // 2)]
    b = a[:h, :w]
    acc = (b[0::2, 0::2].astype(np.float32) + b[1::2, 0::2] + b[0::2, 1::2] + b[1::2, 1::2]) / 4.0
    if np.issubdtype(a.dtype, np.integer):
        return np.rint(acc).astype(a.dtype)
    return acc.astype(a.dtype)


_BACKENDS = (_OpenSlideBackend, _LargeImageBackend, _TiffFileBackend, _ArrayBackend)

_REQUIRES = {"openslide": ("openslide",), "large_image": ("large_image",),
             "tifffile": ("tifffile", "zarr"), "array": ("tifffile",)}


def backends_available():
    """The backend names importable here, in the order they are tried."""
    import importlib
    out = []
    for cls in _BACKENDS:
        try:
            for mod in _REQUIRES[cls.name]:
                importlib.import_module(mod)
            out.append(cls.name)
        except Exception:
            pass
    return out


def open_backend(path, backend: Optional[str] = None):
    """The first backend that opens `path` (or the named one). The error lists
    every failure, so a missing package and an unreadable file are told
    apart."""
    tried = []
    for cls in _BACKENDS:
        if backend is not None and cls.name != backend:
            continue
        try:
            return cls(path)
        except Exception as exc:                     # a failed import or a failed open
            tried.append(f"{cls.name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("no pyramid backend could open %r\n  %s"
                       % (str(path), "\n  ".join(tried) or "no backend named " + str(backend)))


# --------------------------------------------------------------------------- #
# the source
# --------------------------------------------------------------------------- #
_DTYPE_RANGE = {np.dtype(np.uint8): (0.0, 255.0), np.dtype(np.uint16): (0.0, 65535.0),
                np.dtype(np.int16): (-32768.0, 32767.0)}


class PyramidImageSource:
    """``ImageSource`` over a tiled pyramid: native values, colour preserved,
    reads assembled from a byte-budgeted LRU of decoded tiles.

    `tile` is the cache's grid, not the file's. A multiple of the file's own
    tile size keeps one viewport read to a handful of entries while letting a
    pan reuse everything it has already touched.
    """
    native = True

    def __init__(self, path, backend: Optional[str] = None, tile: int = 512,
                 cache_mb: float = 256.0):
        self.path = str(path)
        self.be = open_backend(path, backend)
        self.tile = int(tile)
        self._budget = int(cache_mb * (1 << 20))
        self._cache = OrderedDict()                   # (level, ty, tx) -> ndarray
        self._bytes = 0
        self.hits = self.misses = 0
        self._range = None

    # -- protocol ---------------------------------------------------------- #
    @property
    def levels(self) -> int:
        return self.be.level_count

    @property
    def channels(self) -> int:
        return int(self.be.channels)

    def level_shape(self, level: int) -> Tuple[int, int]:
        w, h = self.be.level_dims(self._clamp(level))
        return h, w

    def level_scale(self, level: int) -> float:
        return float(self.be.downsample(self._clamp(level)))

    def best_level(self, scale: float) -> int:
        """The coarsest level whose pixels are still finer than `scale`
        full-res px per screen px (never coarser than the screen)."""
        level = 0
        while level + 1 < self.levels and self.level_scale(level + 1) <= max(scale, 1.0):
            level += 1
        return level

    def value_range(self) -> Tuple[float, float]:
        """The dtype's range for integer data; measured on the coarsest level
        for float data, where there is no such thing."""
        if self._range is None:
            rng = _DTYPE_RANGE.get(np.dtype(self.be.dtype))
            if rng is None:
                top = self.levels - 1
                h, w = self.level_shape(top)
                a = self.read_region(top, 0, 0, w, h)
                rng = (float(np.nanmin(a)), float(np.nanmax(a))) if a.size else (0.0, 1.0)
            self._range = rng
        return self._range

    def read_region(self, level: int, x: int, y: int, w: int, h: int):
        """(h, w) or (h, w, C) at `level`, in the file's dtype. A rect running
        off the level is zero-filled outside it rather than clipped, so the
        caller always gets the shape it asked for -- which is what lets a
        halo'd ROI read straight off an edge."""
        level = self._clamp(level)
        lh, lw = self.level_shape(level)
        w = max(1, int(w)); h = max(1, int(h)); x = int(x); y = int(y)
        shape = (h, w, self.channels) if self.channels > 1 else (h, w)
        out = np.zeros(shape, dtype=self.be.dtype)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(lw, x + w), min(lh, y + h)
        if x1 <= x0 or y1 <= y0:
            return out
        t = self.tile
        for ty in range(y0 // t, (y1 - 1) // t + 1):
            for tx in range(x0 // t, (x1 - 1) // t + 1):
                tile = self._tile(level, ty, tx, lw, lh)
                tx0, ty0 = tx * t, ty * t
                sx0, sy0 = max(x0, tx0), max(y0, ty0)
                sx1, sy1 = min(x1, tx0 + tile.shape[1]), min(y1, ty0 + tile.shape[0])
                if sx1 <= sx0 or sy1 <= sy0:
                    continue
                out[sy0 - y:sy1 - y, sx0 - x:sx1 - x] = \
                    tile[sy0 - ty0:sy1 - ty0, sx0 - tx0:sx1 - tx0]
        return out

    # -- tiles ------------------------------------------------------------- #
    def _clamp(self, level: int) -> int:
        return max(0, min(int(level), self.levels - 1))

    def _tile(self, level, ty, tx, lw, lh):
        key = (level, ty, tx)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            self.hits += 1
            return hit
        self.misses += 1
        t = self.tile
        x, y = tx * t, ty * t
        arr = np.asarray(self.be.read(level, x, y, min(t, lw - x), min(t, lh - y)))
        if self.channels > 1 and arr.ndim == 2:
            arr = np.repeat(arr[..., None], self.channels, axis=2)
        self._cache[key] = arr
        self._bytes += arr.nbytes
        # One tile always stays: a read that is larger than the whole budget
        # must still make progress rather than evict what it is assembling.
        while self._bytes > self._budget and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._bytes -= old.nbytes
        return arr

    def clear_cache(self):
        self._cache.clear()
        self._bytes = 0

    @property
    def cache_bytes(self) -> int:
        return self._bytes

    def close(self):
        self.clear_cache()
        try:
            self.be.close()
        except Exception:
            pass

    def __repr__(self):
        h, w = self.level_shape(0)
        return (f"PyramidImageSource({self.path!r}, {self.be.name}, "
                f"{w}x{h}x{self.channels}, {self.levels} levels)")
