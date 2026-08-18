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
import time
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

        self._overlays = []          # tagged: ("rgba",rgba,vis) | ("label",labels,lut,vis)
        self._vmin = 0.0             # window fractions of [base_min, base_max]
        self._vmax = 1.0
        self._alpha = 0.5

        self.scale = 1.0
        self.view_x = 0.0
        self.view_y = 0.0
        self._photo = None
        self._job = None
        self._drag = None
        self.on_hover = None         # optional callback(ix, iy) | callback(None)
        self._hud_mode = None        # None | "busy" (animated) | "stale" (static)
        self._hud_text = ""
        self._hud_job = None
        self._hud_phase = 0

        self.canvas.bind("<Configure>", lambda e: self._schedule())
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom(e.x, e.y, 1 / 1.25))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(e.x, e.y, 1.25))
        self.canvas.bind("<ButtonPress-1>", self._drag_start)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)

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
        """overlays: list of dicts, either a pre-colored RGBA layer
        ``{"rgba": HxWx4 uint8, "visible": bool}`` (e.g. the filtered field) or a
        label+LUT layer ``{"labels": HxW int, "lut": (K,4) uint8, "visible": bool}``
        recolored at render time over the viewport (segmentation / mask). Stored as
        tagged tuples: ``("rgba", rgba, vis)`` or ``("label", labels, lut, vis)``."""
        tagged = []
        for o in overlays:
            vis = o.get("visible", True)
            if "labels" in o:
                tagged.append(("label", o["labels"], o["lut"], vis))
            else:
                tagged.append(("rgba", o["rgba"], vis))
        self._overlays = tagged

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

    def screen_to_image(self, sx, sy):
        """Map a canvas pixel (sx, sy) to full-resolution base pixel (ix, iy).

        Exact inverse of the render mapping ``image = view + screen * scale``."""
        return int(self.view_x + sx * self.scale), int(self.view_y + sy * self.scale)

    def _on_motion(self, e):
        if self.on_hover is None:
            return
        ix, iy = self.screen_to_image(e.x, e.y)
        if 0 <= ix < self.image_width and 0 <= iy < self.image_height:
            self.on_hover(ix, iy)
        else:
            self.on_hover(None)

    def _on_leave(self, _e):
        if self.on_hover is not None:
            self.on_hover(None)

    # -- canvas HUD: "recomputing" spinner / "out of date" badge ------- #
    _BUSY_FRAMES = "|/-\\"

    def set_hud(self, mode, text=""):
        """Show a status badge over the canvas. mode: None (clear), "busy"
        (animated spinner -- assembly in flight), or "stale" (static warning --
        the displayed result is out of date)."""
        if mode == self._hud_mode and text == self._hud_text:
            return
        self._hud_mode = mode
        self._hud_text = text
        if self._hud_job is not None:
            self.after_cancel(self._hud_job)
            self._hud_job = None
        self.canvas.delete("hud")
        if mode is not None:
            self._hud_tick()

    def _hud_tick(self):
        self._hud_job = None
        self.canvas.delete("hud")
        if self._hud_mode is None:
            return
        if self._hud_mode == "busy":
            frame = self._BUSY_FRAMES[self._hud_phase % len(self._BUSY_FRAMES)]
            self._hud_phase += 1
            label = f"{frame}  {self._hud_text or 'Recomputing'}…"
            fill, outline = "#1e5a8a", "#8ac0ff"
        else:  # stale
            label = f"⚠  {self._hud_text or 'Out of date'}"
            fill, outline = "#8a5a1e", "#ffcf8a"
        t = self.canvas.create_text(22, 20, anchor="nw", text=label, fill="#ffffff",
                                    font=("TkDefaultFont", 11, "bold"), tags=("hud",))
        box = self.canvas.bbox(t)
        if box:
            pad = 6
            r = self.canvas.create_rectangle(box[0] - pad, box[1] - pad, box[2] + pad,
                                             box[3] + pad, fill=fill, outline=outline,
                                             tags=("hud",))
            self.canvas.tag_lower(r, t)
        if self._hud_mode == "busy":
            self._hud_job = self.after(120, self._hud_tick)   # animate only when busy

    # -- rendering ----------------------------------------------------- #
    def _schedule(self):
        if self._job is not None:
            self.after_cancel(self._job)
        self._job = self.after(15, self.render)

    def _base_region(self, left, top, right, bottom, out_w, out_h):
        """Return an (out_h, out_w) uint8 grayscale array for the base region."""
        # Window over the base channel's own [min, max]. The in-memory base holds
        # the native (float32) values, so we window there directly -- this is the
        # correct, detail-preserving path. large_image's getRegion, by contrast,
        # returns display-scaled 8-bit data whose values don't match the native
        # [base_min, base_max] range, so it's only a fallback (windowed in [0,1]
        # fraction space) for when no in-memory base is available.
        lo = self._base_min + self._vmin * (self._base_max - self._base_min)
        hi = self._base_min + self._vmax * (self._base_max - self._base_min)
        span = (hi - lo) or 1.0
        if self._base is not None:
            crop = self._base[top:bottom, left:right]
            im = Image.fromarray(np.clip((crop - lo) / span, 0, 1).astype(np.float32))
            im = im.resize((out_w, out_h), Image.BILINEAR)
            return (np.asarray(im, dtype=np.float32) * 255).astype(np.uint8)
        # Fallback: no in-memory base -> use the pyramidal source. Its 8-bit output
        # is already display-scaled, so window in normalized [0,1] fraction space.
        png, _ = self._source.getRegion(
            region={"left": left, "top": top, "right": right, "bottom": bottom,
                    "units": "base_pixels"},
            output={"maxWidth": out_w, "maxHeight": out_h}, encoding="PNG")
        im = Image.open(io.BytesIO(png)).convert("L").resize((out_w, out_h))
        g = np.asarray(im, dtype=np.float32) / 255.0
        g = np.clip((g - self._vmin) / ((self._vmax - self._vmin) or 1.0), 0, 1)
        return (g * 255).astype(np.uint8)

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

        t0 = time.perf_counter()
        gray = self._base_region(left, top, right, bottom, out_w, out_h)
        rgb = np.dstack([gray, gray, gray]).astype(np.float32)
        t_base = time.perf_counter()

        n_ov = 0
        # Nearest-neighbour source indices for the visible region -> viewport grid
        # (shared by label overlays; O(out_w+out_h) to build).
        ys = xs = None
        for entry in self._overlays:
            kind, visible = entry[0], entry[-1]
            if not visible:
                continue
            if kind == "rgba":
                rgba = entry[1]
                if rgba is None:
                    continue
                crop = rgba[top:bottom, left:right]
                im = Image.fromarray(crop, mode="RGBA").resize((out_w, out_h), Image.NEAREST)
                ov = np.asarray(im, dtype=np.float32)
            else:  # "label": recolor the label image through the LUT at render time
                labels, lut = entry[1], entry[2]
                if labels is None or lut is None:
                    continue
                if ys is None:
                    ys = np.clip((top + (np.arange(out_h) + 0.5) * (bottom - top) / out_h)
                                 .astype(np.intp), top, bottom - 1)
                    xs = np.clip((left + (np.arange(out_w) + 0.5) * (right - left) / out_w)
                                 .astype(np.intp), left, right - 1)
                sub = labels[ys][:, xs]                       # (out_h, out_w) feature ids
                ov = lut[np.where(sub >= 0, sub, 0)].astype(np.float32)
                ov[sub < 0] = 0                               # background -> transparent
            a = (ov[:, :, 3:4] / 255.0) * self._alpha
            rgb = rgb * (1 - a) + ov[:, :, :3] * a
            n_ov += 1
        t_ov = time.perf_counter()

        # Blit at the region's true screen position (image = view + screen*scale),
        # so panning/zooming stay consistent even when the image doesn't fully
        # cover the viewport (letterboxed / panned past an edge).
        screen_x = int(round((left - self.view_x) / self.scale))
        screen_y = int(round((top - self.view_y) / self.scale))
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb.astype(np.uint8), mode="RGB"))
        self.canvas.delete("view")
        self.canvas.create_image(screen_x, screen_y, anchor="nw", image=self._photo, tags="view")
        if self._hud_mode is not None:
            self.canvas.tag_raise("hud")   # keep the HUD above the freshly-blitted image
        t_blit = time.perf_counter()

        # Only log slow frames (the first fit / a heavy composite) so live
        # pan/zoom doesn't flood the terminal.
        total_ms = 1e3 * (t_blit - t0)
        if total_ms >= 50.0:
            src = "large_image" if self._source is not None else "in-memory"
            print(f"[mscoupon]   canvas.render {out_w}x{out_h} via {src}: "
                  f"base={1e3 * (t_base - t0):.0f}ms "
                  f"overlays({n_ov})={1e3 * (t_ov - t_base):.0f}ms "
                  f"blit={1e3 * (t_blit - t_ov):.0f}ms total={total_ms:.0f}ms", flush=True)
