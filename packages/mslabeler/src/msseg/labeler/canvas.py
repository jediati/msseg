"""Zoomable/pannable slice canvas.

Shows one item at a time: a base image with brightness/contrast, plus
alpha-composited overlay layers (a filter field, a segmentation, a mask, the
labeler's class layers). The base is any ``ImageSource``: an in-memory array
(``ArrayImageSource``, exact float windowing) or a tiled pyramid served by
level (``PyramidImageSource`` over ``large_image``), read only over the
viewport at the level nearest the current zoom. Region overlays are
``LabelLayer``s recoloured through a LUT at render time; a layer that can hand
over its whole raster (``full()``) is gathered exactly onto the viewport grid,
one that cannot is asked for a crop at the best level and resized nearest.

Coordinates ``view_x``/``view_y`` and ``scale`` are in full-resolution base
pixels (scale = base px per screen px), so overlays and annotations map
trivially. Requires numpy + Pillow; ``large_image`` is optional.
"""
from __future__ import annotations

import time
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

from .perf import FrameProfiler
from .sources import (ArrayImageSource, ArrayLabelLayer, PyramidImageSource,
                      HAVE_LARGE_IMAGE as _HAVE_LARGE_IMAGE, level_index_vectors)

class SliceCanvas(tk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, background="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Two candidate base sources: an in-memory array (preferred: exact,
        # windowed in the data's own range) and a pyramid opened from a path
        # (the fallback when there is no array, cached by path so repeated
        # renders of one slice never re-open the file).
        self._array_src = None       # ArrayImageSource | None
        self._pyramid_src = None     # PyramidImageSource | None
        self._source_path = None     # path the pyramid source was opened for
        self._base_min = 0.0         # the active source's value range
        self._base_max = 1.0
        self.image_width = 1
        self.image_height = 1

        self._overlays = []          # tagged: ("rgba",rgba,vis) | ("label",layer,lut,vis)
        self._vmin = 0.0             # window fractions of [base_min, base_max]
        self._vmax = 1.0
        self._alpha = 0.5

        self.scale = 1.0
        self.view_x = 0.0
        self.view_y = 0.0
        self._photo = None
        self._job = None
        self._pending_since = None   # when the oldest unserved repaint was asked for
        self._drag = None
        self.on_hover = None         # optional callback(ix, iy) | callback(None)
        # Optional drawing-tool controller with on_press/on_move/on_release(e)
        # -> bool. Returning True claims the event (suppresses the pan); the
        # controller draws its own feedback as canvas items tagged "draw",
        # which render() keeps above each fresh blit.
        self.tool = None
        # Optional callback() fired after the view transform changes (zoom,
        # pan, fit) -- screen-space annotations must be redrawn at the new
        # scale/offset.
        self.on_view_changed = None
        # Optional callback(event) fired on a right-CLICK -- a Button-3 press
        # and release with no drag in between. Right-drag still pans (see the
        # bindings below), so a context menu cannot cost navigation; a viewer
        # that leaves this None behaves exactly as before.
        self.on_context = None
        self._ctx_press = None
        self._hud_mode = None        # None | "busy" (animated) | "stale" | "info" (static)
        # One transient label+LUT layer above every overlay: the "will be
        # painted" preview of a gesture in flight. Set/cleared by the drawing
        # tool, untouched by set_overlays(), so a repaint landing mid-drag
        # keeps the preview -- and composited at the LUT's own alpha, not the
        # overlay slider's, because the preview IS the point while it shows.
        self._transient = None
        self._hud_text = ""
        self._hud_job = None
        self._hud_phase = 0
        # Where a drag's milliseconds actually go (see perf.py). Off unless
        # MSSEG_CANVAS_PROFILE=1; Ctrl+P toggles it live.
        self.perf = FrameProfiler("canvas")
        self._painting = False       # guard: the forced paint re-enters the loop
        self._perf_level = 0         # pyramid level the last frame read from
        self._lut_cache = {}         # (id(lut), alpha) -> derived blend LUTs

        self.canvas.bind("<Configure>", lambda e: self._schedule())
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom(e.x, e.y, 1 / 1.25))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(e.x, e.y, 1.25))
        self.canvas.bind("<ButtonPress-1>", self._drag_start)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._drag_end)
        # Middle-drag and right-drag always pan, so a drawing tool on button 1
        # never locks navigation out.
        self.canvas.bind("<ButtonPress-2>", self._pan_start)
        self.canvas.bind("<B2-Motion>", self._pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._pan_end)
        self.canvas.bind("<ButtonPress-3>", self._context_press)
        self.canvas.bind("<B3-Motion>", self._pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._context_release)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)
        # Ctrl+P anywhere in the window: turn the frame profiler on/off without
        # restarting the app (bound on the toplevel because the canvas rarely
        # holds focus -- the pointer is what is over it, not the keyboard).
        try:
            self.winfo_toplevel().bind("<Control-p>", self._toggle_profile, add="+")
            self.winfo_toplevel().bind("<Control-P>", self._toggle_profile, add="+")
        except Exception:       # pragma: no cover - headless/odd master
            pass

    # -- content ------------------------------------------------------- #
    @property
    def source(self):
        """The ImageSource being drawn: the in-memory array when there is one,
        else the pyramid, else None."""
        return self._array_src if self._array_src is not None else self._pyramid_src

    @property
    def has_base(self):
        return self.source is not None

    # Legacy views kept for callers (and the selftests) that read them: the
    # in-memory array, and the raw large_image source behind the pyramid.
    @property
    def _base(self):
        return None if self._array_src is None else self._array_src.array

    @property
    def _source(self):
        return None if self._pyramid_src is None else self._pyramid_src._src

    def set_source(self, source, path=None):
        """Make `source` (an ImageSource) the base. A ``native`` source (values
        in the data's own range) takes the in-memory slot and wins over the
        pyramid; a non-native one (display-scaled) takes the pyramid slot."""
        if getattr(source, "native", True):
            self._array_src = source
        else:
            self._pyramid_src = source
            self._source_path = path if path is not None else getattr(source, "path", None)
        self._sync_dims()

    def _sync_dims(self):
        src = self.source
        if src is None:
            return
        self.image_height, self.image_width = src.level_shape(0)
        self._base_min, self._base_max = src.value_range()

    def set_base(self, array=None, path=None, reset_array=False):
        """Set the base image from an in-memory array and/or a file path
        (the path is opened pyramidally by large_image when available). The
        pyramid is cached by path so repeated renders of the same slice
        (e.g. dragging the persistence slider) don't re-open the file.
        reset_array=True drops a previously-set in-memory array, so a
        path-only base (the preview fallback) actually renders via the
        pyramidal source instead of the stale array."""
        if reset_array and array is None:
            self._array_src = None
        if path != self._source_path:
            self._pyramid_src = None
            self._source_path = path
            if path and _HAVE_LARGE_IMAGE:
                try:
                    self._pyramid_src = PyramidImageSource(str(path))
                except Exception:
                    self._pyramid_src = None
        if array is not None:
            self._array_src = ArrayImageSource(array, path)
        self._sync_dims()

    @property
    def base_is_rgb(self):
        return self._array_src is not None and self._array_src.channels == 3

    @staticmethod
    def _as_layer(o):
        """A LabelLayer from an overlay dict: ``{"layer": LabelLayer}`` as is,
        ``{"labels": array}`` wrapped."""
        if o.get("layer") is not None:
            return o["layer"]
        labels = o.get("labels")
        return None if labels is None else ArrayLabelLayer(labels)

    def set_overlays(self, overlays):
        """overlays: list of dicts, either a pre-colored RGBA layer
        ``{"rgba": HxWx4 uint8, "visible": bool}`` (e.g. the filtered field) or a
        region layer ``{"labels": HxW int | "layer": LabelLayer, "lut": (K,4) uint8,
        "visible": bool}`` recolored at render time over the viewport
        (segmentation / mask). Stored as tagged tuples: ``("rgba", rgba, vis)`` or
        ``("label", layer, lut, vis)``."""
        tagged = []
        for o in overlays:
            vis = o.get("visible", True)
            if "labels" in o or "layer" in o:
                tagged.append(("label", self._as_layer(o), o["lut"], vis))
            else:
                tagged.append(("rgba", o["rgba"], vis))
        self._overlays = tagged

    def set_transient(self, overlay):
        """Set (or clear, with None) the transient preview layer: a
        ``{"labels": HxW int | "layer": LabelLayer, "lut": (K,4) uint8}`` dict as
        in set_overlays."""
        if overlay is None:
            self._transient = None
        else:
            self._transient = ("label", self._as_layer(overlay), overlay["lut"], True)

    def set_window(self, vmin, vmax):
        self._vmin, self._vmax = float(vmin), float(vmax)

    def set_alpha(self, alpha):
        self._alpha = float(alpha)

    def set_view(self, view_x=None, view_y=None, scale=None):
        """Move the view (full-resolution pixel offsets, base px per screen px)
        and repaint; None keeps a component. The public way to scroll without
        re-centring (see ``center_on`` for that)."""
        if scale is not None:
            self.scale = min(max(float(scale), 0.05), max(self.image_width, self.image_height))
        if view_x is not None:
            self.view_x = float(view_x)
        if view_y is not None:
            self.view_y = float(view_y)
        self._schedule()
        self._view_changed()

    @property
    def hud(self):
        """``(mode, text)`` of the status badge (see ``set_hud``)."""
        return self._hud_mode, self._hud_text

    def fit(self):
        w = max(self.canvas.winfo_width(), 1)
        h = max(self.canvas.winfo_height(), 1)
        self.scale = max(self.image_width / w, self.image_height / h, 1e-6)
        self.view_x = (self.image_width - w * self.scale) / 2
        self.view_y = (self.image_height - h * self.scale) / 2
        self._schedule()
        self._view_changed()

    # -- interaction --------------------------------------------------- #
    def _toggle_profile(self, _e=None):
        self.perf.set_enabled(not self.perf.enabled)
        return "break"

    def _on_wheel(self, e):
        self.perf.event("wheel")
        self._zoom(e.x, e.y, 1 / 1.25 if e.delta > 0 else 1.25)

    def _zoom(self, sx, sy, factor):
        ix = self.view_x + sx * self.scale
        iy = self.view_y + sy * self.scale
        self.scale = min(max(self.scale * factor, 0.05),
                         max(self.image_width, self.image_height))
        self.view_x = ix - sx * self.scale
        self.view_y = iy - sy * self.scale
        self._schedule()
        self._view_changed()

    def _view_changed(self):
        # Timed into the profiler's between-frames bucket: this callback runs
        # once per motion event (the labeler re-projects every screen-space
        # annotation item here), so with the repaint debounced it can cost
        # several times a frame without appearing in render()'s own timings.
        if self.on_view_changed is not None:
            with self.perf.span("view_cb"):
                self.on_view_changed()

    def center_on(self, ix, iy, zoom_to=1.0):
        """Scroll so image point (ix, iy) sits at the canvas centre, zooming
        in to at most `zoom_to` image pixels per screen pixel first (None
        keeps the current zoom) -- the way a region list navigates."""
        if zoom_to is not None:
            self.scale = max(min(self.scale, float(zoom_to)), 0.05)
        w = max(self.canvas.winfo_width(), 1)
        h = max(self.canvas.winfo_height(), 1)
        self.view_x = float(ix) - w / 2.0 * self.scale
        self.view_y = float(iy) - h / 2.0 * self.scale
        self._schedule()
        self._view_changed()

    def _drag_start(self, e):
        if self.tool is not None:
            t0 = time.perf_counter()
            claimed = self.tool.on_press(e)
            if claimed:
                self._drag = None
                self.perf.gesture("tool")
                self.perf.add("tool.press", 1e3 * (time.perf_counter() - t0))
                return
        self._pan_start(e)

    def _drag_move(self, e):
        self.perf.event("drag")
        if self.tool is not None:
            with self.perf.span("tool.move"):
                claimed = self.tool.on_move(e)
            if claimed:
                return
        self._pan_move(e)

    def _drag_end(self, e):
        if self.tool is not None:
            with self.perf.span("tool.release"):
                self.tool.on_release(e)
        self._drag = None
        self.perf.end_gesture()

    # A right-press starts a pan optimistically; only a release close enough
    # to the press counts as a click. Same click-vs-drag threshold idiom the
    # labeler uses for dragging interaction rows between class panels.
    _CLICK_SLOP_PX = 5

    def _context_press(self, e):
        self._ctx_press = (e.x, e.y)
        self._pan_start(e)

    def _context_release(self, e):
        press, self._ctx_press = self._ctx_press, None
        self._drag = None
        self.perf.end_gesture()
        if press is None or self.on_context is None:
            return
        if abs(e.x - press[0]) + abs(e.y - press[1]) <= self._CLICK_SLOP_PX:
            self.on_context(e)

    def _pan_start(self, e):
        self._drag = (e.x, e.y, self.view_x, self.view_y)
        self.perf.gesture("pan")

    def _pan_end(self, _e=None):
        self._drag = None
        self.perf.end_gesture()

    def _pan_drag(self, e):
        """Middle/right-button pan tick (button 1 arrives via _drag_move, which
        counts the event itself before offering it to the drawing tool)."""
        self.perf.event("pan")
        self._pan_move(e)

    def _pan_move(self, e):
        if not self._drag:
            return
        sx, sy, vx, vy = self._drag
        self.view_x = vx - (e.x - sx) * self.scale
        self.view_y = vy - (e.y - sy) * self.scale
        self._schedule()
        self._view_changed()

    def screen_to_image(self, sx, sy):
        """Map a canvas pixel (sx, sy) to full-resolution base pixel (ix, iy).

        Exact inverse of the render mapping ``image = view + screen * scale``."""
        return int(self.view_x + sx * self.scale), int(self.view_y + sy * self.scale)

    def _on_motion(self, e):
        if self.on_hover is None:
            return
        ix, iy = self.screen_to_image(e.x, e.y)
        # Charged to the next frame: hover does not fire while a button is
        # down (Tk sends B<n>-Motion then), but it competes with the repaint
        # for the same event loop the moment the button comes up.
        with self.perf.span("hover"):
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
        (animated spinner -- assembly in flight), "stale" (static warning --
        the displayed result is out of date), or "info" (static, neutral --
        a tool's live readout, e.g. the magic-fill threshold)."""
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
        elif self._hud_mode == "info":
            label = self._hud_text
            fill, outline = "#2a2a2a", "#cccccc"
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
    # Derived blend LUTs kept for this many (LUT, alpha) pairs: a region LUT is
    # ~4 MB per derived table at 363k regions, and the live set is the two or
    # three overlays plus a gesture's transient.
    _LUT_CACHE_MAX = 4

    _DEBOUNCE_MS = 15
    # ...but a repaint may never be postponed by more than this past the first
    # request that asked for it. Coalescing a burst of motion events into one
    # frame is right; letting the burst defer the frame indefinitely is not --
    # events arrive every ~9 ms during a drag, so a plainly re-armed 15 ms
    # timer never expires and the canvas repaints only when the pointer pauses.
    # (Measured on a 76 ms frame: 179 events, 174 repaints cancelled, 5 frames
    # in 1.85 s = 2.7 fps, with 80% of the drag spent rendering nothing.) Past
    # the deadline the already-armed job is left alone rather than re-armed, so
    # the worst case is deadline + debounce and there is no cancel storm.
    _MAX_DEFER_MS = 30

    def _schedule(self):
        now = time.perf_counter()
        if self._pending_since is None:
            self._pending_since = now
        elif self._job is not None and \
                1e3 * (now - self._pending_since) >= self._MAX_DEFER_MS:
            self.perf.count("deadline")
            return                       # let the armed repaint through
        if self._job is not None:
            self.after_cancel(self._job)
            self.perf.coalesced()
        self.perf.request()
        self._job = self.after(self._DEBOUNCE_MS, self.render)

    def invalidate(self):
        """Repaint soon (debounced): the public way to say a layer changed."""
        self._schedule()

    def _base_region(self, left, top, right, bottom, out_w, out_h):
        """Return an (out_h, out_w) uint8 grayscale array for the base region,
        or (out_h, out_w, 3) when the base is colour."""
        src = self.source
        lo = self._base_min + self._vmin * (self._base_max - self._base_min)
        hi = self._base_min + self._vmax * (self._base_max - self._base_min)
        span = (hi - lo) or 1.0
        if getattr(src, "native", True):
            # Native values: window in the data's own range, then resample.
            # An in-memory array has one level, so this is the exact crop; a
            # native pyramid would be read at the level nearest the zoom.
            level = self._perf_level = src.best_level(self.scale)
            sc = src.level_scale(level)
            crop = src.read_region(level, int(left / sc), int(top / sc),
                                   max(1, int(round((right - left) / sc))),
                                   max(1, int(round((bottom - top) / sc))))
            self.perf.mark("base.read")
            self.perf.count("base.src_px", int(crop.shape[0]) * int(crop.shape[1]))
            norm = np.clip((crop - lo) / span, 0, 1)
            self.perf.mark("base.window")
            if norm.ndim == 3:
                im = Image.fromarray((norm * 255).astype(np.uint8), "RGB")
                im = im.resize((out_w, out_h), Image.BILINEAR)
                out = np.asarray(im, dtype=np.uint8)
                self.perf.mark("base.resize")
                return out
            im = Image.fromarray(norm.astype(np.float32))
            im = im.resize((out_w, out_h), Image.BILINEAR)
            out = (np.asarray(im, dtype=np.float32) * 255).astype(np.uint8)
            self.perf.mark("base.resize")
            return out
        # Display-scaled source (the pyramid): its 8-bit output does not match
        # the native range, so window in normalized [0,1] fraction space.
        level = self._perf_level = src.best_level(self.scale)
        sc = src.level_scale(level)
        g = src.read_region(level, int(left / sc), int(top / sc),
                            max(1, int(round((right - left) / sc))),
                            max(1, int(round((bottom - top) / sc))))
        # A pyramid read is a file read + a PNG decode, and it happens on every
        # frame: expect this to dominate when there is no in-memory array.
        self.perf.mark("base.read")
        self.perf.count("base.src_px", int(np.size(g)))
        im = Image.fromarray(np.asarray(g, dtype=np.uint8)).resize((out_w, out_h))
        self.perf.mark("base.resize")
        g = np.asarray(im, dtype=np.float32) / 255.0
        g = np.clip((g - self._vmin) / ((self._vmax - self._vmin) or 1.0), 0, 1)
        out = (g * 255).astype(np.uint8)
        self.perf.mark("base.window")
        return out

    def _label_region(self, layer, left, top, right, bottom, out_w, out_h, cache):
        """(out_h, out_w) region ids of `layer` over the viewport. A layer that
        hands over its raster is gathered exactly onto the viewport grid
        (index vectors shared across layers through `cache`); otherwise its
        crop at the best level is resized nearest-neighbour."""
        full = layer.full()
        if full is not None:
            if cache.get("ys") is None:
                cache["ys"] = np.clip((top + (np.arange(out_h) + 0.5) * (bottom - top) / out_h)
                                      .astype(np.intp), top, bottom - 1)
                cache["xs"] = np.clip((left + (np.arange(out_w) + 0.5) * (right - left) / out_w)
                                      .astype(np.intp), left, right - 1)
            # NB the row gather materialises (out_h x full_width) before the
            # column gather narrows it -- independent of the zoom, so a heavy
            # "ov.gather" that does not shrink when zoomed in is this.
            out = full[cache["ys"]][:, cache["xs"]]
            self.perf.mark("ov.gather")
            return out
        level = 0
        sc = 1.0
        while 2.0 ** (level + 1) <= max(self.scale, 1.0):
            level += 1
            sc = 2.0 ** level
        sub = layer.crop(level, int(left / sc), int(top / sc),
                         max(1, int(round((right - left) / sc))),
                         max(1, int(round((bottom - top) / sc))))
        self.perf.mark("ov.crop")
        im = Image.fromarray(np.asarray(sub, dtype=np.int32)).resize((out_w, out_h), Image.NEAREST)
        out = np.asarray(im, dtype=np.int32)
        self.perf.mark("ov.resize")
        return out

    def _blend_luts(self, lut, alpha):
        """``(premultiplied colour, 1 - alpha)`` float32 LUTs for `lut` at
        overlay alpha `alpha`, with a transparent row appended so a background
        id (-1) selects it by numpy's negative-index wrap.

        The per-pixel composite used to be
        ``a = ov[:,:,3:4]/255 * alpha; rgb = rgb*(1-a) + ov[:,:,:3]*a`` over a
        float32 copy of the gathered RGBA block: five full-size temporaries and
        two STRIDED reads of a 4-channel array, ~26 ms per overlay on a
        900x640 viewport. Every one of those scalar steps depends only on the
        region id, so folding them into the LUT -- once per LUT row instead of
        once per pixel -- leaves two contiguous gathers and an in-place
        multiply-add. It is exact, not an approximation: the same float32
        operations on the same values, so the composite is bit-identical.

        Both come back 3 wide, so the gathers land on CONTIGUOUS (h, w, 3)
        arrays: numpy's multiply against a broadcast (h, w, 1) weight is 15x
        slower than against a real (h, w, 3) one (5.4 ms vs 0.35 ms at this
        size), far more than the wider gather costs.

        Cached per (LUT identity, alpha), most-recently-used kept: a region LUT
        has a row per region -- 363k of them on a real slice -- so rebuilding
        it every frame would cost as much as it saves, and a magic-fill drag
        hands over a fresh preview LUT on every tick."""
        key = (id(lut), float(alpha))
        hit = self._lut_cache.pop(key, None)
        if hit is not None and hit[0] is lut:
            self._lut_cache[key] = hit        # touch: keep it over the transient
            return hit[1], hit[2]
        while len(self._lut_cache) >= self._LUT_CACHE_MAX:
            self._lut_cache.pop(next(iter(self._lut_cache)))
        a = lut[:, 3].astype(np.float32) / 255.0
        if alpha != 1.0:                      # a transient layer keeps its own
            a = a * alpha
        n = lut.shape[0]
        premul = np.zeros((n + 1, 3), np.float32)
        premul[:n] = lut[:, :3].astype(np.float32) * a[:, None]
        one_minus = np.ones((n + 1, 3), np.float32)
        one_minus[:n] = (1 - a)[:, None]
        self._lut_cache[key] = (lut, premul, one_minus)
        return premul, one_minus

    def render(self):
        self._job = self._pending_since = None
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw <= 1 or ch <= 1 or self.source is None:
            return
        left = max(0, int(self.view_x)); top = max(0, int(self.view_y))
        right = min(self.image_width, int(self.view_x + cw * self.scale))
        bottom = min(self.image_height, int(self.view_y + ch * self.scale))
        if right <= left or bottom <= top:
            return
        out_w = max(1, int((right - left) / self.scale))
        out_h = max(1, int((bottom - top) / self.scale))

        perf = self.perf
        src_kind = "in-memory" if self._array_src is not None else "pyramid"
        with perf.frame(out=f"{out_w}x{out_h}", crop=f"{right - left}x{bottom - top}",
                        scale=f"{self.scale:.4g}", src=src_kind,
                        items=len(self.canvas.find_all()) if perf.enabled else 0) as fr:
            self._render_body(left, top, right, bottom, out_w, out_h)
            if fr is not None:          # known only once the base has been read
                fr.info["lvl"] = self._perf_level

    def _render_body(self, left, top, right, bottom, out_w, out_h):
        perf = self.perf
        t0 = time.perf_counter()
        gray = self._base_region(left, top, right, bottom, out_w, out_h)
        # A colour base already comes back as (out_h, out_w, 3).
        rgb = (gray.astype(np.float32) if gray.ndim == 3
               else np.dstack([gray, gray, gray]).astype(np.float32))
        perf.mark("base.rgb")
        t_base = time.perf_counter()

        n_ov = 0
        # Nearest-neighbour source indices for the visible region -> viewport grid
        # (shared by label overlays; O(out_w+out_h) to build).
        index_cache = {}
        entries = list(self._overlays)
        if self._transient is not None:
            entries.append(self._transient)
        for entry in entries:
            kind, visible = entry[0], entry[-1]
            if not visible:
                continue
            transient = entry is self._transient
            if kind == "rgba":
                rgba = entry[1]
                if rgba is None:
                    continue
                crop = rgba[top:bottom, left:right]
                im = Image.fromarray(crop, mode="RGBA").resize((out_w, out_h), Image.NEAREST)
                ov = np.asarray(im, dtype=np.float32)
                a = ov[:, :, 3:4] / 255.0
                if not transient:
                    a = a * self._alpha
                weight, colour = 1 - a, ov[:, :, :3] * a
                perf.mark("ov.rgba")
            else:  # "label": recolor the region layer through the LUT at render time
                layer, lut = entry[1], entry[2]
                if layer is None or lut is None:
                    continue
                sub = self._label_region(layer, left, top, right, bottom, out_w, out_h,
                                         index_cache)   # (out_h, out_w) region ids
                if sub.dtype.kind == "i":
                    # Any background id, not just -1, must land on the LUT's
                    # transparent row (which -1 reaches by negative wrap).
                    np.maximum(sub, -1, out=sub)
                premul, one_minus = self._blend_luts(lut, 1.0 if transient else self._alpha)
                weight, colour = one_minus[sub], premul[sub]
                perf.mark("ov.lut")
            # rgb = rgb * (1 - a) + colour, in place: the arrays are 900x640x3
            # floats and each temporary is another 7 MB through the cache.
            np.multiply(rgb, weight, out=rgb)
            np.add(rgb, colour, out=rgb)
            perf.mark("ov.blend")
            n_ov += 1
        perf.count("overlays", n_ov)
        t_ov = time.perf_counter()

        # Blit at the region's true screen position (image = view + screen*scale),
        # so panning/zooming stay consistent even when the image doesn't fully
        # cover the viewport (letterboxed / panned past an edge).
        screen_x = int(round((left - self.view_x) / self.scale))
        screen_y = int(round((top - self.view_y) / self.scale))
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb.astype(np.uint8), mode="RGB"))
        perf.mark("photo")
        self.canvas.delete("view")
        self.canvas.create_image(screen_x, screen_y, anchor="nw", image=self._photo, tags="view")
        if self._hud_mode is not None:
            self.canvas.tag_raise("hud")   # keep the HUD above the freshly-blitted image
        self.canvas.tag_raise("draw")      # tool rubber-band items (no-op when unused)
        perf.mark("canvas")
        t_blit = time.perf_counter()

        # create_image only QUEUES a redraw -- Tk copies the photo to the window
        # from the event loop's idle phase, after render() has returned, and
        # redraws every canvas item overlapping it while it is there. That cost
        # is the gap between "canvas.render 100 ms" and 1-2 fps on screen, so
        # while profiling we force the paint here and time it. (Only then: the
        # forced flush changes when work happens, and the guard keeps a
        # <Configure> fired from inside it from re-entering.)
        if perf.enabled and not self._painting:
            self._painting = True
            try:
                self.canvas.update_idletasks()
            finally:
                self._painting = False
            perf.mark("paint")

        # Only log slow frames (the first fit / a heavy composite) so live
        # pan/zoom doesn't flood the terminal. NB this covers the composite
        # ONLY -- not the Tk paint above, nor the per-event work between
        # frames; turn the profiler on (Ctrl+P) for the whole picture.
        total_ms = 1e3 * (t_blit - t0)
        if total_ms >= 50.0 and not perf.enabled:
            src = "in-memory" if self._array_src is not None else "pyramid"
            print(f"[mscoupon]   canvas.render {out_w}x{out_h} via {src}: "
                  f"base={1e3 * (t_base - t0):.0f}ms "
                  f"overlays({n_ov})={1e3 * (t_ov - t_base):.0f}ms "
                  f"blit={1e3 * (t_blit - t_ov):.0f}ms "
                  f"composite={total_ms:.0f}ms (excludes the Tk paint)", flush=True)
