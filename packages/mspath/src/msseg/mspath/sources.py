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
