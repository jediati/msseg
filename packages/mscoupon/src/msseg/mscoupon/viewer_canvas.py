"""Zoomable/pannable slice canvas for the mscoupon viewer.

Shows one slice at a time: a grayscale base image with brightness/contrast, plus
alpha-composited overlay channels (filter field, segmentation, mask). The base is
served pyramidally by ``large_image`` when available (``getRegion`` reads only the
viewport at an appropriate level); otherwise it falls back to the in-memory base
array. Overlays are full-resolution numpy arrays cropped+resized to the viewport
(nearest-neighbour for label/mask layers) so they stay aligned to the base.

Coordinates ``view_x``/``view_y`` and ``scale`` are in full-resolution base pixels
(scale = base px per screen px), so overlays and future annotations map trivially.
Requires numpy + Pillow (both mscoupon deps); ``large_image`` is optional.
"""
from __future__ import annotations

import io
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

try:  # optional pyramidal backend for the base image
    import large_image
    _HAVE_LARGE_IMAGE = True
except Exception:  # pragma: no cover
    _HAVE_LARGE_IMAGE = False


class SliceCanvas(tk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, background="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self._source = None          # large_image source (base), or None
        self._source_path = None     # path currently open in _source (cache key)
        self._base = None            # in-memory base array (H,W) float, fallback
        self._base_min = 0.0
        self._base_max = 1.0
        self.image_width = 1
        self.image_height = 1

        self._overlays = []          # list of (rgba uint8 HxWx4, visible bool)
        self._vmin = 0.0             # window fractions of [base_min, base_max]
        self._vmax = 1.0
        self._alpha = 0.5

        self.scale = 1.0
        self.view_x = 0.0
        self.view_y = 0.0
        self._photo = None
        self._job = None
        self._drag = None

        self.canvas.bind("<Configure>", lambda e: self._schedule())
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom(e.x, e.y, 1 / 1.25))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(e.x, e.y, 1.25))
        self.canvas.bind("<ButtonPress-1>", self._drag_start)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))

    # -- content ------------------------------------------------------- #
    def set_base(self, array=None, path=None):
        """Set the base image from an in-memory array and/or a file path
        (the path is used pyramidally by large_image when available). The
        large_image source is cached by path so repeated renders of the same
        slice (e.g. dragging the persistence slider) don't re-open the file."""
        if path != self._source_path:
            self._source = None
            self._source_path = path
            if path and _HAVE_LARGE_IMAGE:
                try:
                    self._source = large_image.open(str(path))
                    md = self._source.getMetadata()
                    self.image_width = int(md["sizeX"])
                    self.image_height = int(md["sizeY"])
                except Exception:
                    self._source = None
        if array is not None:
            self._base = np.asarray(array, dtype=np.float32)
            self.image_height, self.image_width = self._base.shape[:2]
            self._base_min = float(self._base.min())
            self._base_max = float(self._base.max())

    def set_overlays(self, overlays):
        """overlays: list of dicts {"rgba": HxWx4 uint8, "visible": bool}."""
        self._overlays = [(o["rgba"], o.get("visible", True)) for o in overlays]

    def set_window(self, vmin, vmax):
        self._vmin, self._vmax = float(vmin), float(vmax)

    def set_alpha(self, alpha):
        self._alpha = float(alpha)

    def fit(self):
        w = max(self.canvas.winfo_width(), 1)
        h = max(self.canvas.winfo_height(), 1)
        self.scale = max(self.image_width / w, self.image_height / h, 1e-6)
        self.view_x = (self.image_width - w * self.scale) / 2
        self.view_y = (self.image_height - h * self.scale) / 2
        self._schedule()

    # -- interaction --------------------------------------------------- #
    def _on_wheel(self, e):
        self._zoom(e.x, e.y, 1 / 1.25 if e.delta > 0 else 1.25)

    def _zoom(self, sx, sy, factor):
        ix = self.view_x + sx * self.scale
        iy = self.view_y + sy * self.scale
        self.scale = min(max(self.scale * factor, 0.05),
                         max(self.image_width, self.image_height))
        self.view_x = ix - sx * self.scale
        self.view_y = iy - sy * self.scale
        self._schedule()

    def _drag_start(self, e):
        self._drag = (e.x, e.y, self.view_x, self.view_y)

    def _drag_move(self, e):
        if not self._drag:
            return
        sx, sy, vx, vy = self._drag
        self.view_x = vx - (e.x - sx) * self.scale
        self.view_y = vy - (e.y - sy) * self.scale
        self._schedule()

    # -- rendering ----------------------------------------------------- #
    def _schedule(self):
        if self._job is not None:
            self.after_cancel(self._job)
        self._job = self.after(15, self.render)

    def _base_region(self, left, top, right, bottom, out_w, out_h):
        """Return an (out_h, out_w) uint8 grayscale array for the base region."""
        lo = self._base_min + self._vmin * (self._base_max - self._base_min)
        hi = self._base_min + self._vmax * (self._base_max - self._base_min)
        span = (hi - lo) or 1.0
        if self._source is not None:
            png, _ = self._source.getRegion(
                region={"left": left, "top": top, "right": right, "bottom": bottom,
                        "units": "base_pixels"},
                output={"maxWidth": out_w, "maxHeight": out_h}, encoding="PNG")
            im = Image.open(io.BytesIO(png)).convert("L").resize((out_w, out_h))
            g = np.asarray(im, dtype=np.float32) / 255.0
            g = np.clip((g - self._vmin) / ((self._vmax - self._vmin) or 1.0), 0, 1)
            return (g * 255).astype(np.uint8)
        crop = self._base[top:bottom, left:right]
        im = Image.fromarray(np.clip((crop - lo) / span, 0, 1).astype(np.float32))
        im = im.resize((out_w, out_h), Image.BILINEAR)
        return (np.asarray(im, dtype=np.float32) * 255).astype(np.uint8)

    def render(self):
        self._job = None
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw <= 1 or ch <= 1 or (self._base is None and self._source is None):
            return
        left = max(0, int(self.view_x)); top = max(0, int(self.view_y))
        right = min(self.image_width, int(self.view_x + cw * self.scale))
        bottom = min(self.image_height, int(self.view_y + ch * self.scale))
        if right <= left or bottom <= top:
            return
        out_w = max(1, int((right - left) / self.scale))
        out_h = max(1, int((bottom - top) / self.scale))

        gray = self._base_region(left, top, right, bottom, out_w, out_h)
        rgb = np.dstack([gray, gray, gray]).astype(np.float32)

        for rgba, visible in self._overlays:
            if not visible or rgba is None:
                continue
            crop = rgba[top:bottom, left:right]
            im = Image.fromarray(crop, mode="RGBA").resize((out_w, out_h), Image.NEAREST)
            ov = np.asarray(im, dtype=np.float32)
            a = (ov[:, :, 3:4] / 255.0) * self._alpha
            rgb = rgb * (1 - a) + ov[:, :, :3] * a

        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb.astype(np.uint8), mode="RGB"))
        self.canvas.delete("view")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo, tags="view")
