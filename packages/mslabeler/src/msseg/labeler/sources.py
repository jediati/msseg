"""Concrete ``ImageSource`` / ``LabelLayer`` implementations.

``ArrayImageSource`` wraps an in-memory raster (the coupon viewer's case): one
level, native float values, so the canvas windows in the data's own range.
``PyramidImageSource`` -- the whole-slide case, over a tiled pyramid -- lives
in ``pyramid.py`` and is re-exported here, so both base-image implementations
are importable from one place. ``ArrayLabelLayer`` serves a full-resolution
int32 region raster by level-space crop, and hands the whole raster to callers
that can use it (``full()``), which is how the canvas keeps its exact viewport
gather for in-memory labels.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .pyramid import PyramidImageSource, backends_available

# True when ANY pyramidal backend (OpenSlide, large_image, tifffile + zarr)
# is importable -- which reader it is, is pyramid.py's business.
HAVE_PYRAMID = bool(backends_available())

__all__ = ["ArrayImageSource", "PyramidImageSource", "ArrayLabelLayer",
           "level_index_vectors", "HAVE_PYRAMID"]


def level_index_vectors(np_, level_scale, x, y, w, h, full_h, full_w):
    """Row/column index vectors into a full-resolution raster for a level-space
    rect: level pixel (i, j) samples the full-resolution pixel under its
    centre, clipped to the raster. (For level scale 1 this is the identity.)"""
    s = float(level_scale)
    ys = np_.clip(((y + np_.arange(h) + 0.5) * s).astype(np_.intp), 0, full_h - 1)
    xs = np_.clip(((x + np_.arange(w) + 0.5) * s).astype(np_.intp), 0, full_w - 1)
    if s == 1.0:                       # exact rows/cols, not centre samples
        ys = np_.clip(y + np_.arange(h), 0, full_h - 1)
        xs = np_.clip(x + np_.arange(w), 0, full_w - 1)
    return ys, xs


class ArrayImageSource:
    """One-level ImageSource over an in-memory (H, W) or planar (C, H, W)
    array. Colour planes are shown as RGB from the first three (a lone extra
    plane is repeated), kept as (H, W, 3) float32 -- windowed with one shared
    [lo, hi] so the channels keep their relative brightness."""
    native = True                      # values are the data's own; window in them

    def __init__(self, array, path: Optional[str] = None):
        arr = np.asarray(array, dtype=np.float32)
        if arr.ndim == 3:
            planes = arr[:3] if arr.shape[0] >= 3 else np.repeat(arr[:1], 3, axis=0)
            arr = np.ascontiguousarray(np.transpose(planes, (1, 2, 0)))
        self.array = arr
        self.path = path
        self._range = (float(arr.min()), float(arr.max())) if arr.size else (0.0, 1.0)

    @property
    def levels(self) -> int:
        return 1

    @property
    def channels(self) -> int:
        return 3 if self.array.ndim == 3 else 1

    def level_shape(self, level: int) -> Tuple[int, int]:
        return tuple(int(v) for v in self.array.shape[:2])

    def level_scale(self, level: int) -> float:
        return 1.0

    def best_level(self, scale: float) -> int:
        return 0

    def value_range(self) -> Tuple[float, float]:
        return self._range

    def read_region(self, level: int, x: int, y: int, w: int, h: int):
        return self.array[y:y + h, x:x + w]


class ArrayLabelLayer:
    """LabelLayer over a full-resolution int32 region raster (-1 = background)."""

    def __init__(self, labels, rev: int = 0):
        self.labels = np.asarray(labels)
        self._rev = int(rev)

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(int(v) for v in self.labels.shape[:2])

    @property
    def n_ids(self) -> int:
        return int(self.labels.max()) + 1 if self.labels.size else 1

    @property
    def rev(self) -> int:
        return self._rev

    def crop(self, level: int, x: int, y: int, w: int, h: int):
        full_h, full_w = self.shape
        ys, xs = level_index_vectors(np, float(2 ** int(level)), x, y, w, h, full_h, full_w)
        return self.labels[ys][:, xs]

    def id_at(self, x: int, y: int) -> int:
        full_h, full_w = self.shape
        if 0 <= x < full_w and 0 <= y < full_h:
            return int(self.labels[y, x])
        return -1

    def full(self):
        return self.labels
