"""The drawing tools: the gesture controller (squiggle / box / lasso / taps,
SHIFT-accept) and the magic fill / blobber controller, plus the region ->
representative-point bridge. They drive an ``AnnotationShell`` through its
documented surface (viewer, regions, catalogue, store, the Tk variables,
_commit_* and the preview trio) and never touch an engine.
"""
from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from . import edge_model, magic_fill, model_search, fields
from . import bundle as model_bundle
from .labeling import (LabelStore, MAX_CLASSES, TOOLS, resolve_slice, resolve_sets,
                          touched_sets, class_lut, scalar_lut, line_pixels, polygon_mask,
                          preview_lut)
from .training import TrainingSetBuilder, TrainingProblem
from .widgets import ScrollFrame, attach_tooltip
from .defaults import *  # noqa: F401,F403




import time

from . import seam_labeling, seam_path
from .labeling import SEAM_TOOLS
from .seams import SEAM_CLASSES, SEAM_COLORS, nearest_seam_point


class DrawController:
    """The canvas drawing tool (SliceCanvas.tool): claims button-1 while a
    class is armed, collects the gesture in IMAGE coordinates (floats -- so a
    box drawn zoomed-out stays accurate), and draws its own rubber-band as
    canvas items tagged "draw" in screen coordinates, recomputed from the
    stored image points each move so zooming mid-drag stays consistent.

    While the slice has a region raster it also previews, on the canvas's
    transient layer, the regions the gesture WILL paint on release -- the same
    rasterization the commit performs (incrementally for a squiggle, per move
    for a box/lasso), in a brightened class color. The "magic" tool is
    delegated whole (press, drag, release) to a MagicFillController."""

    MIN_SCREEN_PX = 3      # squiggle/lasso point spacing (screen px)
    _SHIFT = 0x0001        # Tk event.state modifier bit
    # Box/lasso previews sample their slab with a stride once it exceeds this
    # many pixels: a full-image box on 3232^2 then costs ~10 ms per move
    # instead of ~100. The commit itself is still exact (touched_ids).
    PREVIEW_BUDGET_PX = 1_000_000

    def __init__(self, app):
        self.app = app
        self._pts = None           # image-coord points of the gesture in flight
        self._last_screen = None
        self._accept = False       # SHIFT-box: accept predictions under the box
        self._pv = None            # "will be painted" preview state, or None
        self.magic = MagicFillController(app)
        # The seam tools: a livewire trace along the seams and a scope box.
        self.trace = TraceController(app)

    def _image_pt(self, e):
        v = self.app.viewer
        return (v.view_x + e.x * v.scale, v.view_y + e.y * v.scale)

    def on_press(self, e):
        if self.app._current() is None:
            return False           # nothing on screen to label
        # Drawing takes the keyboard back: with a combobox or entry still
        # focused (the Magic row, say) every hotkey -- Ctrl-Z above all --
        # would be swallowed as "typing" after the gesture.
        self.app._unfocus_entries()
        # SHIFT = the accept tool: a box, regardless of the selected tool or
        # armed class, that turns the predictions under it into real labels.
        self._accept = bool(e.state & self._SHIFT)
        tool = self.app.tool_var.get()
        if self.trace.active and (self._accept or tool not in SEAM_TOOLS):
            self.trace.cancel("trace abandoned (tool changed)")
        if not self._accept and tool in SEAM_TOOLS:
            # Before the armed-class test: seam gestures carry their own class.
            return self.trace.on_press(e, scope=(tool == "scope"))
        if not self._accept and tool in ("magic", "blobber"):
            return self.magic.on_press(e, ring=(tool == "blobber"))
        if not self._accept and self.app.active_class_var.get() <= 0:
            return False           # no class armed -> pan as in the viewer
        self._pts = [self._image_pt(e)]
        self._last_screen = (e.x, e.y)
        self._preview_begin()
        self._preview_update()
        return True

    def _tool(self):
        return "box" if self._accept else self.app.tool_var.get()

    def on_move(self, e):
        if self.trace.active:
            return self.trace.on_move(e)
        if self.magic.active:
            return self.magic.on_move(e)
        if self._pts is None:
            return False
        changed = True
        if self._tool() == "box":
            # A box is its two corners: the anchor plus the live corner.
            if len(self._pts) > 1:
                self._pts[-1] = self._image_pt(e)
            else:
                self._pts.append(self._image_pt(e))
        else:
            lx, ly = self._last_screen
            if abs(e.x - lx) + abs(e.y - ly) >= self.MIN_SCREEN_PX:
                self._pts.append(self._image_pt(e))
                self._last_screen = (e.x, e.y)
            else:
                changed = False
        if changed:
            self._preview_update()
        self._draw_feedback()
        return True

    def on_release(self, e):
        if self.trace.active:
            return self.trace.on_release(e)
        if self.magic.active:
            return self.magic.on_release(e)
        if self._pts is None:
            return False
        pts, self._pts = self._pts, None
        accept, self._accept = self._accept, False
        self._preview_end()
        self.app.viewer.canvas.delete("draw")
        if accept:
            if len(pts) >= 2:
                self.app._accept_predictions(pts)
            return True
        tool = self.app.tool_var.get()
        # A single-click tap IS a squiggle (the polyline's start point is always
        # part of the gesture); box/lasso need an actual drag to mean anything.
        need = 1 if tool == "squiggle" else 2
        if len(pts) >= need:
            self.app._commit_interaction(tool, pts)
        return True

    def cancel(self):
        """Escape: abandon whatever is in flight (any tool) without committing.
        Returns True when there was something to abandon."""
        if self.trace.active:
            return self.trace.cancel()
        if self.magic.active:
            return self.magic.cancel()
        if self._pts is None:
            return False
        self._pts = None
        self._accept = False
        self._preview_end()
        self.app.viewer.canvas.delete("draw")
        self.app.status_var.set("gesture cancelled")
        return True

    def redraw(self):
        """Re-project whatever the tools draw in screen space after a
        zoom/pan (the trace in flight)."""
        self.trace.redraw()

    # -- "will be painted" preview --------------------------------------- #
    def _preview_begin(self):
        """Arm the preview for the gesture just started. Nothing to preview
        (no region raster yet, or an accept box with no live predictions)
        leaves the tools exactly as they were: rubber band only."""
        self._pv = None
        app = self.app
        cur = app._current()
        rec = app.regions.record(app.catalogue.key_of(*cur)) if cur is not None else None
        if rec is None or rec.get("labels") is None:
            return
        labels = rec["labels"]
        pred = None
        if self._accept:
            pr = app._pred.get(app.catalogue.key_of(*cur))
            if pr is None or pr[0] != rec.get("commit"):
                return             # nothing would be accepted: no preview
            pred = pr[1]
        place = app._region_placement()
        # The stroke width the commit will use (resolve_opts): one screen
        # pixel in raster pixels, when the app records a scale of intent.
        cur = app._current()
        draw = (app._draw_meta(*cur) if cur is not None else None) or {}
        px = draw.get("px")
        width = float(px) / (place.scale or 1.0) if isinstance(px, (int, float)) and px > 0 else None
        self._pv = {"labels": labels,
                    "K": int(labels.max()) + 1 if labels.size else 1,
                    "ids": set(), "shown": None, "pred": pred,
                    "place": place, "width": width,
                    "rgba": app.store.rgba(int(app.active_class_var.get()))}
        app._begin_preview()

    def _preview_update(self):
        pv = self._pv
        if pv is None or not self._pts:
            return
        import numpy as np
        labels = pv["labels"]
        h, w = labels.shape
        # The gesture is in image coordinates; the raster is the item's. For a
        # coupon slice these are the same numbers and this is a no-op.
        place = pv["place"]
        pts = self._pts if place.identity else place.points_to_raster(self._pts)
        tool = self._tool()
        if tool == "squiggle":
            # Incremental and exact: only the newest segment is rasterized --
            # at the width the commit will use, so the preview promises what
            # the commit paints.
            (x0, y0), (x1, y1) = (pts[-2], pts[-1]) if len(pts) > 1 else (pts[0], pts[0])
            width = pv.get("width")
            if width is not None and width > 1.5:
                from .labeling import stroke_mask
                sm = stroke_mask([(x0, y0), (x1, y1)], width, w, h, np)
                if sm is not None:
                    mask, ya, xa = sm
                    sub = labels[ya:ya + mask.shape[0], xa:xa + mask.shape[1]]
                    pv["ids"].update(int(v) for v in np.unique(sub[mask]) if v >= 0)
            else:
                ys, xs = line_pixels(x0, y0, x1, y1, w, h, np)
                if len(ys):
                    pv["ids"].update(int(v) for v in np.unique(labels[ys, xs]) if v >= 0)
        elif tool == "box":
            pv["ids"] = self._slab_ids(labels, pts[0], pts[-1], np)   # already raster
        elif len(pts) >= 3:                     # polygon
            ids = set()
            pm = polygon_mask(pts, w, h, np)
            if pm is not None:
                mask, ya, xa = pm
                sub = labels[ya:ya + mask.shape[0], xa:xa + mask.shape[1]]
                st = self._stride(mask.size)
                vals = np.unique(sub[::st, ::st][mask[::st, ::st]])
                ids = set(int(v) for v in vals if v >= 0)
            pv["ids"] = ids
        if pv["ids"] == pv["shown"]:
            return
        pv["shown"] = set(pv["ids"])
        if pv["pred"] is not None:
            # Accept box: each region in the color of the class it would get.
            n = self.app.store.n_classes
            pred = pv["pred"]
            colors = {i: self.app.store.rgba(int(pred[i])) for i in pv["ids"]
                      if i < len(pred) and 1 <= int(pred[i]) < n}
            self.app._preview_regions(labels, pv["K"], list(colors), colors)
        else:
            self.app._preview_regions(labels, pv["K"], pv["ids"], pv["rgba"])

    def _preview_end(self):
        if self._pv is None:
            return
        self._pv = None
        self.app._end_preview()

    @classmethod
    def _stride(cls, n_px):
        if n_px <= cls.PREVIEW_BUDGET_PX:
            return 1
        return max(1, int(math.ceil(math.sqrt(n_px / cls.PREVIEW_BUDGET_PX))))

    def _slab_ids(self, labels, p0, p1, np):
        h, w = labels.shape
        xa, xb = sorted((int(round(p0[0])), int(round(p1[0]))))
        ya, yb = sorted((int(round(p0[1])), int(round(p1[1]))))
        xa, xb = max(xa, 0), min(xb, w - 1)
        ya, yb = max(ya, 0), min(yb, h - 1)
        if xa > xb or ya > yb:
            return set()
        st = self._stride((xb - xa + 1) * (yb - ya + 1))
        vals = np.unique(labels[ya:yb + 1:st, xa:xb + 1:st])
        return set(int(v) for v in vals if v >= 0)

    def _draw_feedback(self):
        v = self.app.viewer
        c = v.canvas
        c.delete("draw")
        if self._accept:
            color = "#ffffff"          # accept box: neutral, dashed
        else:
            cls = self.app.active_class_var.get()
            color = (self.app._class_color_hex(cls)
                     if 0 < cls < MAX_CLASSES else "#ffffff")
        scr = [((x - v.view_x) / v.scale, (y - v.view_y) / v.scale)
               for x, y in self._pts]
        if len(scr) < 2:
            return
        if self._tool() == "box":
            (x0, y0), (x1, y1) = scr[0], scr[-1]
            kw = {"outline": color, "width": 2, "tags": "draw"}
            if self._accept:
                kw["dash"] = (4, 3)
            c.create_rectangle(x0, y0, x1, y1, **kw)
        else:
            flat = [coord for pt in scr for coord in pt]
            c.create_line(*flat, fill=color, width=2, tags="draw")
            if self._tool() == "polygon":
                # Preview the auto-close edge.
                c.create_line(scr[-1][0], scr[-1][1], scr[0][0], scr[0][1],
                              fill=color, width=1, dash=(3, 2), tags="draw")


class MagicFillController:
    """The magic-fill tool: press on a region with a class armed, and a flood
    grows from it over the living-region adjacency graph while a
    dissimilarity stays under a threshold; drag UP for a higher threshold
    (more regions), DOWN for a lower one; release paints; Escape abandons.

    Everything seed-dependent is computed once at the press as a join ladder
    (magic_fill.build_ladder); a drag tick is a rank on that ladder -- a prefix
    of the flood's discovery order, so every pixel of drag adds or removes ONE
    connected region even where many tie (an outlier seed) -- and the
    threshold is always in the data's own units (shown on the canvas HUD). The first threshold is the one last
    released with the same metric/mode/channels this session, else a natural
    break in the ladder. The result commits as ONE "taps" interaction with a
    point per grown region at its seeding extremum, so it re-resolves after a
    persistence change through the same geometric path as any gesture.

    The "blobber" is the same fill with a RING: the regions immediately
    adjacent to the core (magic_fill.ring_for_rank) preview and commit in a
    second class -- the one after the active class by default (wrapping past
    the last), or a fixed id from the options row. Click on a void with class
    2 armed and the void is class 2 with its bounding regions in class 3;
    the drag grows the core and the ring follows. Committed as two taps
    interactions, ring first so the core wins if a re-decomposition ever
    merges a ring point's region into a core point's."""

    def __init__(self, app):
        self.app = app
        self._s = None             # session dict while a fill is in flight
        self._last = {}            # (metric, mode, channels) -> last released t

    def ring_class(self, cls):
        """The blobber's ring class for active class `cls`: the options row's
        explicit id when it is a real class other than `cls`, else the next
        class after `cls` (wrapping past the last). None when no other class
        exists (a two-class store has only class 1)."""
        n = self.app.store.n_classes
        want = self.app.blob_ring_var.get()
        if want != "next":
            try:
                k = int(want)
            except ValueError:
                k = -1
            if 1 <= k < n and k != cls:
                return k
        for k in list(range(cls + 1, n)) + list(range(1, cls)):
            return k
        return None

    @property
    def active(self):
        return self._s is not None

    def options(self):
        app = self.app
        metric = app.magic_metric_var.get()
        mode = app.magic_mode_var.get()
        chans = [c.strip() for c in app.magic_channels_var.get().split(",")
                 if c.strip()]
        if metric not in magic_fill.METRICS:
            metric = "mean"
        if mode not in magic_fill.MODES:
            mode = "anchor"
        return metric, mode, chans or ["base"]

    def hop_gain(self):
        return _bounded_float(self.app.magic_gain_var.get(), _DEFAULT_HOP_GAIN,
                              *_HOP_GAIN_RANGE)

    def drag_px(self):
        return _bounded_float(self.app.magic_drag_var.get(), _DEFAULT_DRAG_PX,
                              *_DRAG_PX_RANGE)

    def on_press(self, e, ring=False):
        app = self.app
        cur = app._current()
        cls = int(app.active_class_var.get())
        if cur is None or not (1 <= cls < app.store.n_classes):
            return False
        ring_cls = None
        if ring:
            ring_cls = self.ring_class(cls)
            if ring_cls is None:
                app._notify("Blobber needs a second class for the ring (Classes >= 3).")
                return False
        rec = app.regions.record(app.catalogue.key_of(*cur))
        if rec is None or rec.get("labels") is None or rec.get("stats") is None:
            app._notify("Magic fill needs computed regions - Run first.")
            return False
        import numpy as np
        v = app.viewer
        labels = rec["labels"]
        h, w = labels.shape
        x, y = v.view_x + e.x * v.scale, v.view_y + e.y * v.scale
        # The press is an image point; the seed is a raster index.
        place = app._region_placement()
        rx, ry = place.to_raster(x, y)
        ix, iy = int(round(rx)), int(round(ry))
        if not (0 <= ix < w and 0 <= iy < h):
            return False
        seed = int(labels[iy, ix])
        if seed < 0:
            return False                     # background: pan as usual
        # The record's MSC arcs, or (extension without region_arcs()) pixel
        # adjacency the provider derives once per record.
        arcs = app.regions.arcs(app.catalogue.key_of(*cur), np)
        metric, mode, want = self.options()
        table = rec["stats"]
        avail = magic_fill.channel_names(table, app.FIELDS)
        chans = [c for c in want if c in avail]
        if not chans:
            chans = ["base"] if "base" in avail else avail[:1]
        if metric in magic_fill.EDGE_ONLY_METRICS and arcs.get("saddle") is None:
            app._notify(f"{metric} needs saddle values (MSC region arcs) - using mean")
            metric = "mean"
        extra = None
        if metric in magic_fill.EXTRA_METRICS:
            # Class probabilities come from the classifier, not the table, and
            # only count at the commit they were made for. No silent fallback:
            # painting with `mean` under a row that says `proba` would mislead
            # while tuning, so the press is refused like one on background.
            pr = app._pred.get(app.catalogue.key_of(*cur))
            if pr is None or pr[0] != rec.get("commit"):
                app._notify(f"{metric} needs predictions at this commit - Classify first")
                return False
            if metric == "learned":
                # The edge model's p(diff) per arc, computed against THIS
                # record's arcs at Classify time.
                aux = app._pred_aux(pr)
                if aux is None or aux.get("pdiff") is None:
                    app._notify("learned needs an edge model at this commit - pick "
                                "a '-> edges' kind, Train (R), then Classify")
                    return False
                extra = {"pdiff": aux["pdiff"]}
            else:
                extra = {"proba": pr[2]}
        gain = self.hop_gain()
        if _bounded_float(app.magic_gain_var.get(), None, *_HOP_GAIN_RANGE) is None:
            app.status_var.set(f"hop\u00d7 is not a number - using {gain:g}")
        try:
            ladder = magic_fill.build_ladder(table, arcs, seed, metric, mode,
                                             chans, np, hop_gain=gain, extra=extra,
                                             conv=app.FIELDS)
        except ValueError as exc:
            app.status_var.set(f"magic fill: {exc}")
            return False
        key = (metric, mode,
               tuple(chans) if metric in magic_fill.CHANNEL_METRICS else (),
               round(gain, 6))
        last_t = self._last.get(key)
        k0 = (magic_fill.rank_at(ladder, last_t, np) if last_t is not None
              else magic_fill.initial_rank(ladder, np))
        self._s = {"si": cur[0], "li": cur[1], "commit": rec.get("commit"),
                   "labels": labels,
                   "K": int(labels.max()) + 1 if labels.size else 1,
                   "ladder": ladder, "seed": seed, "seed_pt": (x, y),
                   "press_y": e.y, "k0": k0, "k": None, "t": None, "ids": None,
                   "rgba": app.store.rgba(cls), "cls": cls, "key": key,
                   "source": arcs.get("source"),
                   "tool": "blobber" if ring else "magic",
                   "drag_px": self.drag_px(),
                   "ring_cls": ring_cls,
                   "ring_rgba": app.store.rgba(ring_cls) if ring_cls else None,
                   "ring": None,
                   "hud": v.hud}
        app._begin_preview()
        self._preview(k0)
        return True

    def on_move(self, e):
        s = self._s
        if s is None:
            return False
        rec = self.app.regions.record(self.app.catalogue.key_of(s["si"], s["li"]))
        if rec is None or rec.get("commit") != s["commit"]:
            self.cancel("magic fill cancelled: regions changed under it")
            return True
        drag = s["drag_px"]
        k = magic_fill.drag_to_rank(s["k0"], s["press_y"] - e.y,
                                    s["ladder"].n_reach,
                                    px_per_step=drag, accel_px=3.0 * drag)
        if k != s["k"]:
            self._preview(k)
        return True

    def _preview(self, k):
        s = self._s
        ladder = s["ladder"]
        import numpy as np
        t = magic_fill.threshold_for_rank(ladder, k)
        ids = magic_fill.regions_for_rank(ladder, k)   # a prefix of the flood
        s["k"], s["t"], s["ids"] = k, t, ids
        n = len(ids)
        px = ""
        if ladder.cum_area is not None and n:
            px = f"  {int(ladder.cum_area[min(n, len(ladder.cum_area)) - 1])} px"
        gtxt = f" g{ladder.hop_gain:g}" if ladder.hop_gain != 1.0 else ""
        if s["ring_cls"] is None:
            self.app._preview_regions(s["labels"], s["K"], ids, s["rgba"],
                                      emphasize=s["seed"])
            self.app.viewer.set_hud(
                "info", f"magic {ladder.metric}/{ladder.mode}{gtxt}  t={t:.3g}  "
                        f"{n}/{ladder.n_reach} regions{px}")
            return
        ring = magic_fill.ring_for_rank(ladder, k, np)
        s["ring"] = ring
        colors = {int(i): s["rgba"] for i in ids}
        colors.update({int(i): s["ring_rgba"] for i in ring})
        self.app._preview_regions(s["labels"], s["K"], list(colors), colors,
                                  emphasize=s["seed"])
        self.app.viewer.set_hud(
            "info", f"blob {ladder.metric}/{ladder.mode}{gtxt}  t={t:.3g}  "
                    f"{n} core (class {s['cls']}) + {len(ring)} ring "
                    f"(class {s['ring_cls']}){px}")

    def on_release(self, e):
        s = self._s
        if s is None:
            return False
        self._s = None
        self._finish(s)
        ids = s.get("ids")
        if ids is None or len(ids) == 0:
            return True
        self._last[s["key"]] = float(s["t"])
        ladder = s["ladder"]
        meta = {"tool": s["tool"],
                "seed": [float(s["seed_pt"][0]), float(s["seed_pt"][1])],
                "seed_id": int(s["seed"]), "threshold": float(s["t"]),
                "metric": ladder.metric, "mode": ladder.mode,
                "channels": list(ladder.channels), "arcs": s["source"],
                "hop_gain": float(ladder.hop_gain)}
        parts = []
        ring = s.get("ring")
        if s["ring_cls"] is not None and ring is not None and len(ring):
            # Ring first: a later uid paints over an earlier one, so the core
            # wins should a re-decomposition merge a ring point into it.
            parts.append(([int(i) for i in ring], s["ring_cls"],
                          dict(meta, part="ring", n_regions=int(len(ring)))))
        core_meta = dict(meta, n_regions=int(len(ids)))
        if s["ring_cls"] is not None:
            core_meta["part"] = "core"
        parts.append(([int(i) for i in ids], s["cls"], core_meta))
        self.app._commit_blob(s["si"], s["li"], s["labels"], parts)
        return True

    def cancel(self, why="magic fill cancelled"):
        s = self._s
        if s is None:
            return False
        self._s = None
        self._finish(s)
        self.app.status_var.set(why)
        return True

    def _finish(self, s):
        v = self.app.viewer
        self.app._end_preview()
        if v is not None and v.hud[0] == "info":
            v.set_hud(*s["hud"])       # give the canvas HUD back to the engine


class TraceController:
    """The seam tools (docs/seam_labeling.md).

    **Trace** (tool "trace", key T): a press snaps to the nearest crack of
    any seam and anchors a ``seam_path.Livewire`` there -- one Dijkstra over
    the junction graph with the toll picked in the Seams row; every mouse
    move then shows the cheapest path from the anchor to the seam point
    under the pointer (a predecessor walk, not a search); a click freezes
    that path as a leg and re-anchors; Enter or a double-click commits the
    legs as ONE "trace" gesture in the picked seam class (boundary by
    default); BackSpace drops the last leg; Escape abandons. An anchor
    inside a scope confines the search to that scope's seams.

    **Scope** (tool "scope", key S): a box drag, previewing the seams that lie
    fully inside it on the transient layer; the release commits a "scope"
    gesture in the picked class (interior by default).

    Points are stored in IMAGE coordinates like every gesture; the snapping
    and the search run on the item's raster through ``_region_placement``.
    """

    SNAP_RADIUS = 8            # raster px around the pointer searched for a crack
    DOUBLE_CLICK_S = 0.35      # a second click on the same point commits

    def __init__(self, app):
        self.app = app
        self._s = None             # trace in flight
        self._box = None           # scope box in flight
        self._last_click = None    # (monotonic s, (seam, idx))

    @property
    def active(self):
        return self._s is not None or self._box is not None

    def _image_pt(self, e):
        v = self.app.viewer
        return (v.view_x + e.x * v.scale, v.view_y + e.y * v.scale)

    def _context(self):
        """(cur, key, rec, graph, np) for the item on screen, or None."""
        app = self.app
        cur = app._current()
        if cur is None:
            return None
        key = app.catalogue.key_of(*cur)
        rec = app.regions.record(key)
        if rec is None or rec.get("labels") is None:
            return None
        import numpy as np
        graph = app.regions.seams(key, np)
        if graph is None or graph.n_seams == 0:
            return None
        return cur, key, rec, graph, np

    # -- dispatch ------------------------------------------------------ #
    def on_press(self, e, scope=False):
        self.app._unfocus_entries()
        if scope:
            if self._s is not None:
                self.cancel("trace abandoned")
            return self._scope_press(e)
        return self._trace_press(e)

    def on_move(self, e):
        if self._box is not None:
            return self._scope_move(e)
        return self._s is not None

    def on_release(self, e):
        if self._box is not None:
            return self._scope_release(e)
        return self._s is not None

    def cancel(self, why=None):
        """Escape: abandon the gesture in flight. True when there was one."""
        if self._box is not None:
            b, self._box = self._box, None
            self._finish(b["hud"])
            self.app.status_var.set(why or "scope cancelled")
            return True
        if self._s is not None:
            s, self._s = self._s, None
            self._finish(s["hud"])
            self.app.status_var.set(why or "trace cancelled")
            return True
        return False

    def redraw(self):
        if self._s is not None:
            self._draw_trace()
        elif self._box is not None:
            self._draw_box()

    def _finish(self, hud):
        app = self.app
        v = app.viewer
        app._end_preview()
        if v is not None:
            v.canvas.delete("trace")
            if v.hud[0] == "info":
                v.set_hud(*hud)         # give the canvas HUD back to the engine

    # -- scope ---------------------------------------------------------- #
    def _scope_press(self, e):
        ctx = self._context()
        if ctx is None:
            self.app._notify("Scope needs computed regions - Run first.")
            return False
        cur, key, rec, graph, np = ctx
        v = self.app.viewer
        self._box = {"si": cur[0], "li": cur[1], "commit": rec.get("commit"), "rec": rec,
                     "graph": graph, "pts": [self._image_pt(e)], "mask": None,
                     "cls": int(self.app.seam_class_var.get()), "hud": v.hud}
        self.app._begin_preview()
        return True

    def _scope_move(self, e):
        b = self._box
        pt = self._image_pt(e)
        if len(b["pts"]) > 1:
            b["pts"][-1] = pt
        else:
            b["pts"].append(pt)
        self._scope_preview()
        self._draw_box()
        return True

    def _scope_preview(self):
        b = self._box
        import numpy as np
        from .labeling import Interaction
        it = Interaction(0, "", b["si"], b["li"], "scope", b["pts"], b["cls"])
        mask = seam_labeling.scope_mask(it, b["graph"], np)
        if b["mask"] is not None and np.array_equal(mask, b["mask"]):
            return
        b["mask"] = mask
        raster = self.app._seam_raster_for(b["si"], b["li"], b["rec"], b["graph"], np)
        lut = np.zeros((b["graph"].n_seams, 4), np.uint8)
        r, g, bb, _a = SEAM_COLORS[b["cls"]] if 0 <= b["cls"] < len(SEAM_COLORS) else SEAM_COLORS[-1]
        lut[mask] = (min(255, r + 60), min(255, g + 60), min(255, bb + 60), 255)
        self.app.viewer.set_transient(self.app._region_overlay(raster, lut, np))
        name = SEAM_CLASSES[b["cls"]] if 0 <= b["cls"] < len(SEAM_CLASSES) else str(b["cls"])
        self.app.viewer.set_hud("info", f"scope: {int(mask.sum())} seams -> {name}")

    def _scope_release(self, e):
        b, self._box = self._box, None
        self._finish(b["hud"])
        pts = b["pts"]
        if len(pts) < 2 or pts[0] == pts[-1]:
            return True
        n = int(b["mask"].sum()) if b["mask"] is not None else 0
        self.app._commit_seam("scope", [pts[0], pts[-1]], b["cls"],
                              meta={"tool": "scope", "seams": n})
        return True

    def _draw_box(self):
        b = self._box
        v = self.app.viewer
        c = v.canvas
        c.delete("trace")
        if b is None or len(b["pts"]) < 2:
            return
        color = self.app._seam_color_hex(b["cls"])
        (x0, y0), (x1, y1) = [((x - v.view_x) / v.scale, (y - v.view_y) / v.scale)
                              for x, y in (b["pts"][0], b["pts"][-1])]
        c.create_rectangle(x0, y0, x1, y1, outline=color, width=2, dash=(6, 3),
                           tags=("draw", "trace"))

    # -- trace ---------------------------------------------------------- #
    def _snap(self, s_or_ctx, x, y):
        """Image point -> (seam, idx) of the nearest crack, or None."""
        place, graph, labels, np = s_or_ctx
        rx, ry = place.to_raster(x, y)
        return nearest_seam_point(graph, labels, rx, ry, np, radius=self.SNAP_RADIUS)

    def _trace_press(self, e):
        app = self.app
        ctx = self._context()
        if ctx is None:
            app._notify("Trace needs computed regions - Run first.")
            return False
        cur, key, rec, graph, np = ctx
        s = self._s
        if s is not None and (s["commit"] != rec.get("commit") or (s["si"], s["li"]) != cur):
            self.cancel("trace cancelled: regions changed under it")
            s = None
        place = app._region_placement()
        x, y = self._image_pt(e)
        hit = self._snap((place, graph, rec["labels"], np), x, y)
        if hit is None:
            app._notify(f"No seam within {self.SNAP_RADIUS} px - press nearer a region boundary")
            return False
        now = time.monotonic()
        if s is None:
            try:
                tolls = app._seam_tolls_for(key, rec, graph, np)
            except ValueError as exc:
                app._notify(f"trace: {exc}")
                return False
            restrict = app._scope_restrict(cur[0], cur[1], rec, graph, hit[0], np)
            lw = seam_path.Livewire(graph, tolls, np, restrict=restrict)
            lw.anchor(*hit)
            v = app.viewer
            self._s = {"si": cur[0], "li": cur[1], "commit": rec.get("commit"),
                       "graph": graph, "labels": rec["labels"], "np": np, "place": place,
                       "lw": lw, "cls": int(app.seam_class_var.get()),
                       "toll": app.seam_toll_var.get(), "restricted": restrict is not None,
                       "hover": None, "hud": v.hud}
            app._begin_preview()          # no region hover outlines while tracing
            self._last_click = (now, hit)
            self._hud()
            self._draw_trace()
            return True
        last = self._last_click
        self._last_click = (now, hit)
        if last is not None and now - last[0] < self.DOUBLE_CLICK_S and last[1] == hit:
            self.commit()
            return True
        lw = s["lw"]
        if hit == lw.anchor_point:
            return True
        if not lw.extend(*hit):
            app._notify("Unreachable along the seams"
                        + (" inside the scope" if s["restricted"] else ""))
            return True
        s["hover"] = None
        self._hud()
        self._draw_trace()
        return True

    def on_hover(self, ix, iy):
        """The pointer moved over the image (image coords): show the path from
        the anchor to the seam point under it."""
        s = self._s
        if s is None or ix is None or iy is None:
            return
        hit = self._snap((s["place"], s["graph"], s["labels"], s["np"]), ix, iy)
        path = None if hit is None else s["lw"].path_to(*hit)
        s["hover"] = (hit, path)
        self._hud(hit)
        self._draw_trace()

    def _hud(self, hover=None):
        s = self._s
        if s is None:
            return
        lw = s["lw"]
        text = (f"trace {s['toll']}{' (scoped)' if s['restricted'] else ''}  "
                f"anchors {len(lw.anchors)}  cost {lw.total_cost:.3g}  "
                f"seams {len(lw.seams_on_path)}")
        if hover is not None:
            c = lw.cost_to(*hover)
            text += f"  +{c:.3g}" if c is not None else "  (unreachable)"
        text += "  |  click: anchor - Enter: commit - BackSpace: undo leg - Esc: abandon"
        self.app.viewer.set_hud("info", text)

    def _screen(self, s, pts):
        v = self.app.viewer
        place = s["place"]
        out = []
        for x, y in pts:
            ix, iy = place.to_image(x, y)
            out.append(((ix - v.view_x) / v.scale, (iy - v.view_y) / v.scale))
        return out

    def _draw_trace(self):
        s = self._s
        v = self.app.viewer
        if v is None:
            return
        c = v.canvas
        c.delete("trace")
        if s is None:
            return
        color = self.app._seam_color_hex(s["cls"])
        lw = s["lw"]
        tags = ("draw", "trace")
        pts = lw.points()
        if len(pts) >= 2:
            flat = [coord for pt in self._screen(s, pts) for coord in pt]
            c.create_line(*flat, fill=color, width=3, tags=tags)
        hover = s.get("hover")
        if hover is not None and hover[1] is not None and len(hover[1]) >= 2:
            flat = [coord for pt in self._screen(s, hover[1]) for coord in pt]
            c.create_line(*flat, fill=color, width=2, dash=(5, 3), tags=tags)
        graph = s["graph"]
        for seam, idx in lw.anchors:
            x, y = graph.seam_points(seam)[idx]
            (sx, sy), = self._screen(s, [(x, y)])
            c.create_oval(sx - 4, sy - 4, sx + 4, sy + 4, outline=color, width=2, tags=tags)

    def commit(self):
        """Enter / double-click: store the legs as one trace gesture."""
        s = self._s
        if s is None:
            return False
        lw = s["lw"]
        pts = lw.commit_points()
        self._s = None
        self._finish(s["hud"])
        if len(pts) < 2:
            self.app.status_var.set("trace needs a second anchor - nothing committed")
            return True
        meta = {"tool": "trace", "toll": s["toll"], "anchors": int(len(lw.anchors)),
                "seams": int(len(lw.seams_on_path)), "cost": round(lw.total_cost, 4),
                "scoped": bool(s["restricted"])}
        self.app._commit_seam("trace", s["graph"].to_image(pts), s["cls"], meta=meta)
        return True

    def drop_last(self):
        """BackSpace: drop the last leg; with only the anchor left, abandon."""
        s = self._s
        if s is None:
            return False
        s["lw"].drop_last()
        if not s["lw"].active:
            return self.cancel("trace abandoned")
        s["hover"] = None
        self._hud()
        self._draw_trace()
        return True


def _extremum_points(labels, ids, table, np, conv=fields.DEFAULT, place=None):
    """One IMAGE point per region id: its seeding extremum (`conv.extremum_xy`
    from the feature table) when that pixel really carries the id, else the
    region's first pixel in raster order. Both lookups are vectorised -- the
    fallback is one pass over the raster, not one per region.

    In and out are image coordinates -- the table's positional columns are in
    them, and the result is stored as a gesture's geometry -- while the raster
    lookups in between are in the item's own indices; `place` is the map
    (identity when the item is the image).
    """
    from .labeling import IDENTITY
    place = place or IDENTITY
    h, w = labels.shape
    pts = {}
    if table is not None and ids and conv.extremum_xy is not None:
        fid, ex, ey = (table.column(conv.id_field), table.column(conv.extremum_xy[0]),
                       table.column(conv.extremum_xy[1]))
        if fid is not None and ex is not None and ey is not None and len(fid):
            fid = np.asarray(fid, np.intp)
            K = int(max(int(fid.max()), max(ids))) + 1
            row_of = np.full(K, -1, np.intp)
            row_of[fid] = np.arange(len(fid))
            for i in ids:
                r = int(row_of[i]) if 0 <= i < K else -1
                if r < 0:
                    continue
                x, y = float(ex[r]), float(ey[r])
                rx, ry = place.to_raster(x, y)
                cx, cy = int(round(rx)), int(round(ry))
                if 0 <= cx < w and 0 <= cy < h and int(labels[cy, cx]) == i:
                    pts[i] = (float(round(x)), float(round(y)))
    missing = [i for i in ids if i not in pts]
    if missing:
        flat = labels.ravel()
        idx = np.flatnonzero(np.isin(flat, np.asarray(missing, labels.dtype)))
        if len(idx):
            vals = flat[idx]
            order = np.argsort(vals, kind="stable")
            vals, idx = vals[order], idx[order]
            first = np.r_[True, vals[1:] != vals[:-1]]
            for val, pos in zip(vals[first].tolist(), idx[first].tolist()):
                pts[int(val)] = place.to_image(pos % w, pos // w)
    return [pts[i] for i in ids if i in pts]
