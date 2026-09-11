"""``LabelLayer`` for an item that covers only part of a slide.

The canvas draws in **slide coordinates** -- level-0 pixels of the whole
slide -- whatever item is on screen. That one decision is what lets an ROI's
regions composite in place over the slide's own pixels, lets several items be
shown at once, and keeps hover, ``center_on`` and the annotation geometry in a
single space. So a region raster computed on a 2048-square ROI at level 0, or
on a whole level-4 overview, is served here as if it were a full-slide raster
that happens to be ``-1`` (background) everywhere outside the item.

``full()`` returns None for anything but a layer that already IS the slide at
level 0, which sends the canvas down its ``crop()`` path: the exact viewport
gather it uses for coupon slices would have to materialise a 4-gigapixel
raster first.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class RoiLabelLayer:
    """Region ids of one item, addressed in slide coordinates.

    `origin` is the item's top-left in slide pixels and `scale` the slide
    pixels per label pixel (16.0 for a level-4 overview, 1.0 for a level-0
    ROI), so a slide point maps to a label pixel by ``(p - origin) / scale``.
    `n_ids` is passed in rather than measured: ``labels.max()`` over a
    16-megapixel raster on every frame is exactly the kind of image-sized work
    this layer exists to avoid.
    """

    def __init__(self, labels, origin=(0, 0), scale=1.0, slide_shape=None,
                 rev: int = 0, n_ids: Optional[int] = None):
        self.labels = np.asarray(labels)
        self.ox, self.oy = int(origin[0]), int(origin[1])
        self.scale = float(scale) or 1.0
        lh, lw = (int(v) for v in self.labels.shape[:2])
        self._shape = (tuple(int(v) for v in slide_shape) if slide_shape is not None
                       else (int(round(self.oy + lh * self.scale)),
                             int(round(self.ox + lw * self.scale))))
        self._rev = int(rev)
        self._n_ids = (int(n_ids) if n_ids is not None
                       else (int(self.labels.max()) + 1 if self.labels.size else 1))

    # -- protocol ---------------------------------------------------------- #
    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def native_level(self) -> int:
        """The canvas ladder level whose pixels are nearest this raster's own:
        where a gesture should be resolved. floor(log2(scale)), so a 1/16
        overview is level 4 and a 1/128.1 one is level 7."""
        return max(0, int(np.floor(np.log2(self.scale)))) if self.scale > 1 else 0

    @property
    def n_ids(self) -> int:
        return self._n_ids

    @property
    def rev(self) -> int:
        return self._rev

    def crop(self, level: int, x: int, y: int, w: int, h: int):
        """Ids over a level-space rect, ``-1`` outside the item.

        The canvas's overlay ladder is powers of two of the slide, so a level
        pixel is ``2**level`` slide pixels; each output pixel samples the label
        pixel under its centre.
        """
        w = max(1, int(w)); h = max(1, int(h))
        s = float(2 ** int(level))
        # level rect -> slide centres -> label pixels
        sx = (int(x) + np.arange(w) + 0.5) * s
        sy = (int(y) + np.arange(h) + 0.5) * s
        cx = np.floor((sx - self.ox) / self.scale).astype(np.intp)
        cy = np.floor((sy - self.oy) / self.scale).astype(np.intp)
        lh, lw = self.labels.shape[:2]
        okx = (cx >= 0) & (cx < lw)
        oky = (cy >= 0) & (cy < lh)
        out = np.full((h, w), -1, np.int32)
        if not okx.any() or not oky.any():
            return out
        block = self.labels[np.clip(cy, 0, lh - 1)][:, np.clip(cx, 0, lw - 1)]
        keep = oky[:, None] & okx[None, :]
        np.copyto(out, block.astype(np.int32, copy=False), where=keep)
        return out

    def id_at(self, x: int, y: int) -> int:
        """The id at a SLIDE point, or -1 outside the item."""
        cx = int((int(x) - self.ox) // self.scale)
        cy = int((int(y) - self.oy) // self.scale)
        lh, lw = self.labels.shape[:2]
        if 0 <= cx < lw and 0 <= cy < lh:
            return int(self.labels[cy, cx])
        return -1

    # -- exact placement, for gesture resolution --------------------------- #
    def placement(self):
        """``(origin_x, origin_y, scale)``: slide point -> raster index is
        ``(p - origin) / scale``. What `labeling.touched_ids_over` resolves
        against, so a gesture reads the raster's own grid rather than the
        canvas ladder's nearest power of two."""
        return (float(self.ox), float(self.oy), float(self.scale))

    @property
    def raster_shape(self):
        return tuple(int(v) for v in self.labels.shape[:2])

    def crop_raster(self, rx: int, ry: int, rw: int, rh: int):
        """A rect of the raster in ITS OWN indices, ``-1`` outside it. A plain
        slice with zero-fill at the edges -- no resampling."""
        lh, lw = self.labels.shape[:2]
        rw, rh = max(0, int(rw)), max(0, int(rh))
        out = np.full((rh, rw), -1, np.int32)
        x0, y0 = max(0, int(rx)), max(0, int(ry))
        x1, y1 = min(lw, int(rx) + rw), min(lh, int(ry) + rh)
        if x1 > x0 and y1 > y0:
            out[y0 - int(ry):y1 - int(ry), x0 - int(rx):x1 - int(rx)] = \
                self.labels[y0:y1, x0:x1]
        return out

    def full(self):
        """The raster only when it already is the whole slide at level 0 --
        otherwise None, so the canvas crops instead of gathering over a raster
        it would have to invent."""
        if (self.ox, self.oy) == (0, 0) and self.scale == 1.0 \
                and tuple(self.labels.shape[:2]) == self._shape:
            return self.labels
        return None

    # -- helpers ----------------------------------------------------------- #
    def slide_rect(self) -> Tuple[int, int, int, int]:
        """The item's footprint in slide pixels: (x, y, w, h)."""
        lh, lw = self.labels.shape[:2]
        return (self.ox, self.oy, int(round(lw * self.scale)), int(round(lh * self.scale)))

    def __repr__(self):
        x, y, w, h = self.slide_rect()
        return (f"RoiLabelLayer({self.labels.shape[1]}x{self.labels.shape[0]} @1/{self.scale:g} "
                f"-> slide ({x},{y}) {w}x{h}, {self._n_ids} ids, rev {self._rev})")


class PlacedImageSource:
    """``ImageSource`` over one item's scalar raster -- its base channel or its
    topology field -- placed on the slide.

    The canvas draws in slide coordinates, and a 2940 x 5625 overview raster
    handed over as a plain array would land at the slide's origin at 1:1. So
    this serves it the way the pyramid is served: a ladder of power-of-two
    levels of the WHOLE slide, each read answered by sampling the raster under
    the requested level pixels and filling the rest with the raster's minimum.
    The ladder matters for more than placement: a view that fits the slide asks
    for the slide's width in level pixels, and at level 0 that is 47 040 by
    90 000 -- four gigapixels for one frame.
    """
    native = True

    def __init__(self, raster, origin=(0, 0), scale=1.0, slide_shape=None, path=None):
        self.raster = np.asarray(raster, dtype=np.float32)
        self.ox, self.oy = int(origin[0]), int(origin[1])
        self.scale = float(scale) or 1.0
        lh, lw = (int(v) for v in self.raster.shape[:2])
        self._shape = (tuple(int(v) for v in slide_shape) if slide_shape is not None
                       else (int(round(self.oy + lh * self.scale)),
                             int(round(self.ox + lw * self.scale))))
        self.path = path
        finite = self.raster[np.isfinite(self.raster)]
        self._range = ((float(finite.min()), float(finite.max())) if finite.size
                       else (0.0, 1.0))
        # as many levels as it takes for the top one to be a thumbnail
        top = max(self._shape)
        self._levels = 1
        while top / (2 ** self._levels) > 256 and self._levels < 16:
            self._levels += 1
        self._levels += 1

    @property
    def levels(self) -> int:
        return self._levels

    @property
    def channels(self) -> int:
        return 1

    def level_shape(self, level: int):
        s = self.level_scale(level)
        return max(1, int(self._shape[0] / s)), max(1, int(self._shape[1] / s))

    def level_scale(self, level: int) -> float:
        return float(2 ** max(0, min(int(level), self._levels - 1)))

    def best_level(self, scale: float) -> int:
        level = 0
        while level + 1 < self._levels and self.level_scale(level + 1) <= max(scale, 1.0):
            level += 1
        return level

    def value_range(self):
        return self._range

    def read_region(self, level: int, x: int, y: int, w: int, h: int):
        w = max(1, int(w)); h = max(1, int(h))
        s = self.level_scale(level)
        sx = (int(x) + np.arange(w) + 0.5) * s
        sy = (int(y) + np.arange(h) + 0.5) * s
        cx = np.floor((sx - self.ox) / self.scale).astype(np.intp)
        cy = np.floor((sy - self.oy) / self.scale).astype(np.intp)
        lh, lw = self.raster.shape[:2]
        okx = (cx >= 0) & (cx < lw)
        oky = (cy >= 0) & (cy < lh)
        out = np.full((h, w), self._range[0], np.float32)
        if okx.any() and oky.any():
            block = self.raster[np.clip(cy, 0, lh - 1)][:, np.clip(cx, 0, lw - 1)]
            np.copyto(out, block, where=oky[:, None] & okx[None, :])
        # A NaN (a filter's undefined pixel, or a raster that is nothing but)
        # would reach the canvas's window as NaN and paint garbage; the
        # range's floor is what "no value" looks like everywhere else here.
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=self._range[0], posinf=self._range[1],
                                neginf=self._range[0])
        return out

    def value_at(self, x: int, y: int):
        """The raster value under a SLIDE point, or None outside the item."""
        cx = int((int(x) - self.ox) // self.scale)
        cy = int((int(y) - self.oy) // self.scale)
        lh, lw = self.raster.shape[:2]
        if 0 <= cx < lw and 0 <= cy < lh:
            return float(self.raster[cy, cx])
        return None
