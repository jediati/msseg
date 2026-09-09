"""mscoupon interactive labeler: annotate MSC regions with classes.

A fork of the viewer (``mscoupon-gui``) for fast class annotation: arm a class
(left-click its color swatch, or the numeric hotkey; right-click the swatch
recolors it), then draw over the slice --

    squiggle   every living MSC region under the polyline gets the class
    box        every region intersecting the dragged rectangle
    lasso      every region under the filled (auto-closed) drag polygon

Class 0 is "no label"; while it is armed the left mouse button pans as in the
viewer (middle-drag always pans). Each gesture is an *interaction*, listed in
its class's subpanel on the right; interactions resolve in creation order (a
later gesture paints over an earlier one), can be dragged between class
subpanels (or moved via right-click), and are stored as raw geometry -- so
changing persistence and hitting Rerun re-resolves them against the new
regions for free.

The class layer renders through the canvas's label-overlay path: one RGBA LUT
per slice (region id -> class color), rebuilt only when an interaction or the
segmentation commit changes, gathered over the visible crop at render time.
The ``regions`` checkbox says whether that layer is drawn; the dropdown beside
it says how it is *colored* -- the region-id palette, ``P(class k)``, or
prediction ``uncertainty`` (1 - the top-two probability margin). The scalar
modes replace the id LUT rather than tinting it, and appear only once Classify
has filled the probability cache. ``outlines`` paints every on-slice gesture
at once instead of only what the pointer is over, and a right-CLICK on the
image plane (a right-DRAG still pans) offers the same move/delete menu the
class-list rows do.

Train freezes a prediction per region; the confusion matrix under the class
stack then counts frozen-prediction against live-label, so labeling more moves
only the true axis -- "old prediction, new value". Clicking a cell highlights
its regions on the current slice. Early counts are resubstitution (the truth
IS the training set); the reading that matters is what moves after a Train.

Loading a classifier saved under different ``statistics`` no longer just
refuses: the v2 pickle carries the statistics it was trained under, and the
labeler offers to build a profile from them (keeping the active profile's
filters/MSC/selection) and switch to it -- which drops the primed data, since
the per-slice feature table is baked at prime time.

Everything else -- sequences, filter chains, statistics, priming, persistence,
config export, session autosave -- is inherited from the viewer. The exported
folder additionally receives ``annotations.json`` (the raw gestures), and the
session autosaves under its own file (``mscoupon-labeler``), never the
viewer's.

Run:  mscoupon-labeler [folder]     |     mscoupon-labeler --selftest
"""
from __future__ import annotations

import os
import sys
import json
import math
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from . import config_io
from . import session
from .app import MscouponApp
from .common import log
from .widgets import ScrollFrame, attach_tooltip
from .labeling import (LabelStore, MAX_CLASSES, TOOLS,
                       resolve_slice, resolve_sets, touched_sets, class_lut,
                       scalar_lut, line_pixels, polygon_mask, preview_lut)
from . import magic_fill
from . import model_search
from . import edge_model

# How faint the inherited region overlay is drawn under the class layer
# (0..255); the class colors themselves stay fully opaque in the LUT and are
# scaled by the shared alpha slider like every overlay.
_REGION_ALPHA = 90
# Classifier predictions render between the region layer and the user's own
# labels, slightly translucent so drawn labels stay distinguishable on top.
_PRED_ALPHA = 170
# A scalar regions layer (class probability / uncertainty) IS the point of the
# view, not orientation under it, so it is drawn much less faintly than the
# region-id layer it replaces.
_SCALAR_ALPHA = 150

# Regions coloring modes; the per-class ones are generated ("P(class 2)").
_MODE_ID = "label id"
_MODE_UNCERTAINTY = "uncertainty"
# ...and, with an edge model, which regions neighbour voting flipped and
# how boundary-like each region is (its max p(diff) over its arcs).
_MODE_FLIPPED = "flipped by neighbours"
_MODE_PDIFF = "boundary p(diff)"

# Classifier kinds are shared by the dropdown, factory, and pickle loader so
# adding one cannot leave a trained model that the UI does not recognize.
_TUNED_KIND = "dense (tuned)"
# "custom FC": a dense net whose hidden sizes are typed on the Model tab.
# The "-> edges" kinds stack an edge model (edge_model.py: a pair classifier
# on the base net's hidden layer over the living-region graph, plus
# neighbour voting) on top of a base net; Train fits the base, then the
# edges. `_clf` is always the BASE pipeline, `_edge_model` the top.
_CUSTOM_KIND = "custom FC"
_TUNED_EDGE_KIND = "dense (tuned) -> edges"
_CUSTOM_EDGE_KIND = "custom FC -> edges"
_EDGE_KINDS = (_TUNED_EDGE_KIND, _CUSTOM_EDGE_KIND)
_MODEL_KINDS = ("random forest", "dense FC", "dense-top-16", "dense-top-32", _TUNED_KIND,
                _CUSTOM_KIND) + _EDGE_KINDS
_DEFAULT_CUSTOM_HIDDEN = "16-8"
_EDGE_LAM_RANGE = (0.0, 5.0)
_EDGE_ROUNDS_RANGE = (0, 10)
_EDGE_C_RANGE = (1e-3, 1e3)


def _is_edge_kind(kind):
    return kind in _EDGE_KINDS


def _base_kind(kind):
    """The base net a kind trains: the kind itself, or the one under `-> edges`."""
    if kind == _TUNED_EDGE_KIND:
        return _TUNED_KIND
    if kind == _CUSTOM_EDGE_KIND:
        return _CUSTOM_KIND
    return kind


def _edge_kind_of(base):
    return _TUNED_EDGE_KIND if base == _TUNED_KIND else (_CUSTOM_EDGE_KIND if base == _CUSTOM_KIND else None)
_DENSE_TOP_N = {"dense-top-16": 16, "dense-top-32": 32}
# The architectures behind the kinds. Module constants rather than literals
# inside _make_model so the Model tab's readout is formatted from the same
# values the estimator is built with and cannot drift from them.
_FOREST_TREES = 200
_OOB_MIN_SAMPLES = 20          # forest OOB score needs a few samples per tree
# ...and shared with model_search, whose search starts from exactly this net.
_MLP_HIDDEN = model_search.BASELINE_HIDDEN
_MLP_MAX_ITER = model_search.DEFAULT_MAX_ITER
# "Optimize network" defaults (the Model tab edits them; they ride the session
# view) and how often the Tk thread drains the worker's progress queue.
# An overnight-sized default: a trial is a 5-fold CV of one candidate, ~5-10 s
# with the search budget (SEARCH_MAX_ITER / SEARCH_PATIENCE), so 300 trials is
# under an hour and the 8 h limit is the real stop for a larger label set.
_SEARCH_TRIALS = 300
_SEARCH_TIMEOUT_MIN = 480                # the entry is in MINUTES
_SEARCH_TIMEOUT_S = _SEARCH_TIMEOUT_MIN * 60
_SEARCH_TRIALS_RANGE = (1, 100000)
_SEARCH_TIMEOUT_RANGE = (0, 7 * 86400)  # seconds; 0 = no time limit
_SEARCH_PUMP_MS = 150
# Trials per rung of the size sweep: enough for alpha / lr / batch / early
# stopping (and the feature mask) to settle at a fixed architecture.
_SWEEP_TRIALS = 20
_SWEEP_TRIALS_RANGE = (1, 10000)


def _hms(seconds):
    """`47s`, `12m 05s`, `3h 07m` for the search progress line."""
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def _model_description(kind, spec=None, n_features=None):
    """One-paragraph, read-only description of what `_make_model(kind)` builds.
    For the tuned kind that is the search winner `spec` (over `n_features`
    columns), or what Train falls back to while no search has run."""
    if kind == _TUNED_KIND:
        if spec is None:
            return ("No search yet: Train builds the dense FC baseline "
                    f"(FeatureSubset(all) -> StandardScaler -> MLPClassifier(hidden "
                    f"layers {_MLP_HIDDEN}, max_iter {_MLP_MAX_ITER})) with balanced "
                    "sample weights. Optimize network (O) searches depth, layer widths, "
                    "L2 alpha, learning rate, batch size, early stopping and a "
                    "per-channel feature subset by leave-slices-out cross-validation "
                    "(Optuna when installed, else random), then installs the winner. "
                    f"Backend: {model_search.backend_label('auto')}.")
        return spec.describe(n_features)
    if kind == "dense FC":
        return (f"StandardScaler -> MLPClassifier(hidden layers {_MLP_HIDDEN}, "
                f"max_iter {_MLP_MAX_ITER}, random_state 0). The scaler lives "
                "inside the pipeline, so the pickle predicts exactly as trained.")
    if kind in _DENSE_TOP_N:
        return (f"SelectFromModel(RandomForest, {_FOREST_TREES} trees, balanced) "
                f"keeps the top {_DENSE_TOP_N[kind]} features by importance "
                "(fewer if the profile has fewer) -> StandardScaler -> "
                f"MLPClassifier(hidden layers {_MLP_HIDDEN}, max_iter "
                f"{_MLP_MAX_ITER}, random_state 0).")
    return (f"RandomForestClassifier: {_FOREST_TREES} trees, class_weight "
            f"balanced, OOB score once >= {_OOB_MIN_SAMPLES} labeled regions, "
            "n_jobs -1, random_state 0.")


# The center notebook's tabs, in order. Named (not indexed) in the session
# view state so a reordered tab list still restores.
_CENTER_TABS = ("Processing", "View", "Model", "Analysis")
# Toolbar hint labels (workflow / model): link-blue, and how often the
# workflow text is re-snapshotted from the panel.
_HINT_COLOR = "#1a4fa0"
_HINT_POLL_MS = 700

# Statistics fields that are POSITIONS, not appearance: where a region sits in
# the slice says nothing about what material it is, and coordinate features
# were exactly what dragged k-means across the label boundary. Never fed to
# the classifier.
_NON_FEATURE_FIELDS = set(magic_fill.POSITIONAL_FIELDS)   # one list, shared with cosine

_TOOL_LABELS = (("squiggle", "squiggle"), ("box", "box"), ("polygon", "lasso"),
                ("magic", "magic"), ("blobber", "blobber"))
# What the tool selector offers (a superset of the STORED tools: "magic" and
# "blobber" are ways of producing "taps" interactions, not gesture types).
# Blobber ring-class choices: the class after the active one, or a fixed id.
_RING_CHOICES = ("next",) + tuple(str(k) for k in range(1, MAX_CLASSES))
# Magic-fill flood options: the per-hop cost multiplier (1 = pure bottleneck;
# clamped to [1, 2] because the cost is w * g^hops and hop depths reach the
# hundreds on a 3232^2 slice) and the drag sensitivity in screen px / region.
_DEFAULT_HOP_GAIN = 1.1
_HOP_GAIN_RANGE = (1.0, 2.0)
_DEFAULT_DRAG_PX = 4.0
_DRAG_PX_RANGE = (1.0, 64.0)


def _bounded_float(text, default, lo, hi):
    """float(text) clamped to [lo, hi], or `default` when it is not a number
    -- an option entry mid-edit must never raise out of a press."""
    try:
        v = float(str(text).strip())
    except (TypeError, ValueError):
        return default
    if v != v:                       # NaN
        return default
    return min(max(v, lo), hi)
_UI_TOOLS = tuple(v for v, _txt in _TOOL_LABELS)

# Focus-widget classes whose keystrokes must not arm classes (typing "1" into
# the persistence entry is not a request to arm class 1).
_TYPING_CLASSES = ("Entry", "TEntry", "Spinbox", "TSpinbox", "TCombobox",
                   "Listbox", "Text")


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

    # -- "will be painted" preview --------------------------------------- #
    def _preview_begin(self):
        """Arm the preview for the gesture just started. Nothing to preview
        (no region raster yet, or an accept box with no live predictions)
        leaves the tools exactly as they were: rubber band only."""
        self._pv = None
        app = self.app
        cur = app._current()
        rec = app.engine.record(*cur) if cur is not None else None
        if rec is None or rec.get("labels") is None:
            return
        labels = rec["labels"]
        pred = None
        if self._accept:
            pr = app._pred.get(cur)
            if pr is None or pr[0] != rec.get("commit"):
                return             # nothing would be accepted: no preview
            pred = pr[1]
        self._pv = {"labels": labels,
                    "K": int(labels.max()) + 1 if labels.size else 1,
                    "ids": set(), "shown": None, "pred": pred,
                    "rgba": app.store.rgba(int(app.active_class_var.get()))}
        app._begin_preview()

    def _preview_update(self):
        pv = self._pv
        if pv is None or not self._pts:
            return
        import numpy as np
        labels = pv["labels"]
        h, w = labels.shape
        pts = self._pts
        tool = self._tool()
        if tool == "squiggle":
            # Incremental and exact: only the newest segment is rasterized.
            (x0, y0), (x1, y1) = (pts[-2], pts[-1]) if len(pts) > 1 else (pts[0], pts[0])
            ys, xs = line_pixels(x0, y0, x1, y1, w, h, np)
            if len(ys):
                pv["ids"].update(int(v) for v in np.unique(labels[ys, xs]) if v >= 0)
        elif tool == "box":
            pv["ids"] = self._slab_ids(labels, pts[0], pts[-1], np)
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
                app.status_var.set("Blobber needs a second class for the ring "
                                   "(Classes >= 3).")
                return False
        rec = app.engine.record(*cur)
        if rec is None or rec.get("labels") is None or rec.get("stats") is None:
            app.status_var.set("Magic fill needs computed regions - Rerun first.")
            return False
        import numpy as np
        v = app.viewer
        labels = rec["labels"]
        h, w = labels.shape
        x, y = v.view_x + e.x * v.scale, v.view_y + e.y * v.scale
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= ix < w and 0 <= iy < h):
            return False
        seed = int(labels[iy, ix])
        if seed < 0:
            return False                     # background: pan as usual
        arcs = rec.get("arcs")
        if arcs is None:
            # Extension without region_arcs(): pixel adjacency, once per
            # record (commit-keyed, so a Rerun recomputes it).
            t0 = time.perf_counter()
            arcs = magic_fill.arcs_from_labels(labels, np)
            rec["arcs"] = arcs
            log(f"magic fill: pixel adjacency for slice {cur[1]}: "
                f"{len(arcs['a'])} pairs ({1e3 * (time.perf_counter() - t0):.0f}ms)")
        metric, mode, want = self.options()
        table = rec["stats"]
        avail = magic_fill.channel_names(table)
        chans = [c for c in want if c in avail]
        if not chans:
            chans = ["base"] if "base" in avail else avail[:1]
        if metric in magic_fill.EDGE_ONLY_METRICS and arcs.get("saddle") is None:
            app.status_var.set(f"{metric} needs saddle values (MSC region arcs) "
                               "- using mean")
            metric = "mean"
        extra = None
        if metric in magic_fill.EXTRA_METRICS:
            # Class probabilities come from the classifier, not the table, and
            # only count at the commit they were made for. No silent fallback:
            # painting with `mean` under a row that says `proba` would mislead
            # while tuning, so the press is refused like one on background.
            pr = app._pred.get((cur[0], cur[1]))
            if pr is None or pr[0] != rec.get("commit"):
                app.status_var.set(f"{metric} needs predictions at this commit "
                                   "- Classify first")
                return False
            if metric == "learned":
                # The edge model's p(diff) per arc, computed against THIS
                # record's arcs at Classify time.
                aux = app._pred_aux(pr)
                if aux is None or aux.get("pdiff") is None:
                    app.status_var.set("learned needs an edge model at this commit - pick "
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
                                             chans, np, hop_gain=gain, extra=extra)
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
                   "hud": (v._hud_mode, v._hud_text)}
        app._begin_preview()
        self._preview(k0)
        return True

    def on_move(self, e):
        s = self._s
        if s is None:
            return False
        rec = self.app.engine.record(s["si"], s["li"])
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
        if v is not None and v._hud_mode == "info":
            v.set_hud(*s["hud"])       # give the canvas HUD back to the engine


def _extremum_points(labels, ids, table, np):
    """One image point per region id: its seeding extremum (ext_x/ext_y from
    the feature table) when that pixel really carries the id, else the
    region's first pixel in raster order. Both lookups are vectorised -- the
    fallback is one pass over the raster, not one per region."""
    h, w = labels.shape
    pts = {}
    if table is not None and ids:
        fid, ex, ey = (table.column("feature_id"), table.column("ext_x"),
                       table.column("ext_y"))
        if fid is not None and ex is not None and ey is not None and len(fid):
            fid = np.asarray(fid, np.intp)
            K = int(max(int(fid.max()), max(ids))) + 1
            row_of = np.full(K, -1, np.intp)
            row_of[fid] = np.arange(len(fid))
            for i in ids:
                r = int(row_of[i]) if 0 <= i < K else -1
                if r < 0:
                    continue
                x, y = int(round(float(ex[r]))), int(round(float(ey[r])))
                if 0 <= x < w and 0 <= y < h and int(labels[y, x]) == i:
                    pts[i] = (float(x), float(y))
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
                pts[int(val)] = (float(pos % w), float(pos // w))
    return [pts[i] for i in ids if i in pts]


class LabelerApp(MscouponApp):
    SESSION_APP = "mscoupon-labeler"

    def _default_profile(self, name="default"):
        return session.default_profile(name, relevance=False)

    def __init__(self, root, initial=None, autosave=True):
        # Labeler state first: the base __init__ calls the overridden build
        # methods, which read these.
        self.store = LabelStore()
        # (si, li) -> (commit, store.rev, lut|None, {uid: touched id set});
        # one rasterization pass serves both the class layer and the
        # which-interactions-touch-this-region hover lookup.
        self._class_luts = {}
        self._hover_key = None     # (si, li, region) whose geometry is on screen
        self._cm_cell = None       # selected confusion cell (true, pred)
        self._hover_uid = None     # row-hovered interaction whose geometry shows
        # Classifier state: model + its feature-column order, and per-slice
        # predicted region->class arrays keyed by the commit they were made at.
        self._clf = None
        self._clf_names = None
        self._clf_kind = "dense FC"
        self.model_kind_var = tk.StringVar(master=root, value="dense FC")
        # "dense (tuned)": the spec the CURRENT model was built from (rides its
        # pickle), the last search's winner (what Train rebuilds; survives a
        # kind switch), and the in-flight search's worker state.
        self._clf_spec = None
        self._search_spec = None
        self._search = None            # {"queue","stop","thread","names","n","t0"}
        self.search_trials_var = tk.StringVar(master=root, value=str(_SEARCH_TRIALS))
        self.search_timeout_var = tk.StringVar(master=root, value=str(_SEARCH_TIMEOUT_MIN))
        # Where a finished search saves its winner so a restart (or the
        # morning after an overnight run) finds it: the session reloads the
        # most recent recorded model. The selftest points this at a temp dir.
        self._models_dir = os.path.join(config_io.app_data_dir(self.SESSION_APP), "models")
        self.search_seed_var = tk.StringVar(master=root, value="0")
        self.search_features_var = tk.BooleanVar(master=root, value=True)
        self.search_backend_var = tk.StringVar(master=root, value="auto")
        self.search_progress_var = tk.StringVar(master=root, value="")
        # Size sweep (how small can the network be): the ladder, the trials
        # spent per rung, and the last result (rows shown in the Model tab).
        self.sweep_sizes_var = tk.StringVar(
            master=root, value=", ".join(model_search.size_text(h)
                                         for h in model_search.DEFAULT_SIZE_LADDER))
        self.sweep_trials_var = tk.StringVar(master=root, value=str(_SWEEP_TRIALS))
        self.sweep_summary_var = tk.StringVar(master=root, value="")
        self._sweep = None                   # last model_search.SweepResult
        # "-> edges" kinds: the pair model on top of the base net (rides the
        # classifier pickle), the custom base's hidden sizes, whether Train
        # keeps the base (to try edge variants on top), and the edge spec.
        self._edge_model = None
        self.custom_hidden_var = tk.StringVar(master=root, value=_DEFAULT_CUSTOM_HIDDEN)
        self.freeze_base_var = tk.BooleanVar(master=root, value=False)
        self.edge_layer_var = tk.StringVar(master=root, value="last")
        self.edge_feat_vars = {f: tk.BooleanVar(master=root, value=True)
                               for f in edge_model.FEATURE_KINDS}
        self.edge_model_var = tk.StringVar(master=root, value="logistic")
        self.edge_c_var = tk.StringVar(master=root, value="1")
        self.edge_lam_var = tk.StringVar(master=root, value="1")
        self.edge_rounds_var = tk.StringVar(master=root, value="3")
        self.edge_readout_var = tk.StringVar(master=root, value="")
        self.edge_progress_var = tk.StringVar(master=root, value="")
        # Provenance line above the classifier controls: which model is loaded,
        # how wide its feature vector is, and whether it still agrees with the
        # active profile (a mismatch blocks Classify, so say so up front).
        self.model_strip_var = tk.StringVar(master=root, value="no model")
        # Session-level model references: saved/loaded pickles + their feature
        # fingerprint, for the profile-compatibility check.
        self.models = []           # [{"path","fingerprint","kind","statistics"}]
        # (si, li) -> (commit, region_class uint8, region_proba float32)
        self._pred = {}
        self.show_pred_var = tk.BooleanVar(master=root, value=True)
        self.show_gt_var = tk.BooleanVar(master=root, value=True)
        # Master overlay switch (Tab toggles it): base image only when off.
        self.show_overlay_var = tk.BooleanVar(master=root, value=True)
        self.active_class_var = tk.IntVar(master=root, value=0)
        self._class_swatches = {}       # class_id -> arm/color swatch button
        self.active_class_var.trace_add("write", self._refresh_class_arm)
        self.tool_var = tk.StringVar(master=root, value="squiggle")
        # Magic-fill options (metric / compare mode / measurement channels);
        # part of the session's view state.
        self.magic_metric_var = tk.StringVar(master=root, value="mean")
        self.magic_mode_var = tk.StringVar(master=root, value="anchor")
        self.magic_channels_var = tk.StringVar(master=root, value="base")
        self.blob_ring_var = tk.StringVar(master=root, value="next")
        self.magic_gain_var = tk.StringVar(master=root, value=f"{_DEFAULT_HOP_GAIN:g}")
        self.magic_drag_var = tk.StringVar(master=root, value=f"{_DEFAULT_DRAG_PX:g}")
        # True while a gesture previews on the canvas: the pointer is busy
        # drawing, so the hover outlines stay off until it is released.
        self._hover_suppressed = False
        self.show_regions_var = tk.BooleanVar(master=root, value=True)
        self.region_mode_var = tk.StringVar(master=root, value=_MODE_ID)
        # Persistent annotation view: paint every on-slice gesture outline at
        # once, not just what the pointer is over.
        self.show_annot_var = tk.BooleanVar(master=root, value=False)
        self.n_classes_var = tk.IntVar(master=root, value=self.store.n_classes)
        self._class_title_labels = {}   # class_id -> title Label (counts text)
        self._drag_uid = None      # interaction row being dragged between classes
        self._drag_origin = None   # (x_root, y_root) at press: distinguishes click vs drag
        self._drop_panel = None    # class panel currently highlighted as target
        self._panel_relief = "flat"
        # Undo/redo over store mutations (add/delete/move/class-count), as
        # whole-store snapshots: the store is small and a snapshot restore
        # reuses the load path, so history can never drift from reality.
        self._undo_stack = []
        self._redo_stack = []
        # Set while the first render of a run happened on an unmapped View tab:
        # the base fits the image only on that first render, and an unmapped
        # canvas is 1x1, so the fit is redone when the tab shows.
        self._fit_pending = False
        self.workflow_hint_var = tk.StringVar(master=root, value="")
        self.model_hint_var = tk.StringVar(master=root, value="")
        self._hint_after = None
        super().__init__(root, initial=initial, autosave=autosave)
        root.title("mscoupon labeler")
        self._build_label_panel()
        if self.viewer is not None:
            self.viewer.tool = DrawController(self)
            # Zoom/pan invalidates screen-space annotation geometry.
            self.viewer.on_view_changed = self._redraw_hover_geometry
            # Right-CLICK (not right-drag, which still pans) on the image
            # plane offers the same menu as an interaction row.
            self.viewer.on_context = self._canvas_menu
        self._bind_hotkeys()
        self._hint_tick()          # first paint + the poll

    # ------------------------------------------------------------------ #
    # Center notebook: Processing | View | Model
    # ------------------------------------------------------------------ #
    # The labeler's window is three panes -- data navigation and processing
    # SELECTION on the left, annotation management and the classifier on the
    # right -- with a tabbed center: the compute profile is EDITED on the
    # Processing tab, the slice is viewed and drawn on the View tab (the
    # inherited `self.right`, so every viewer-area builder packs into it
    # unchanged), and the model is designed on the Model tab. The notebook
    # hides whatever tab is not active.
    def _build_center(self):
        self.center = ttk.Notebook(self.paned, width=900)
        self.paned.add(self.center, weight=1)
        self.processing_tab = ttk.Frame(self.center)
        self.right = ttk.Frame(self.center)          # the View tab
        self.model_tab = ttk.Frame(self.center)
        # Analysis: what the models DO with the annotations -- the region
        # list behind a confusion cell, the size sweep -- and, later, plots.
        self.analysis_tab = ttk.Frame(self.center)
        self._center_tabs = {"Processing": self.processing_tab,
                             "View": self.right,
                             "Model": self.model_tab,
                             "Analysis": self.analysis_tab}
        for name in _CENTER_TABS:
            self.center.add(self._center_tabs[name], text=name)

        # Processing: the profile management rows on top (anchored, not
        # filled -- the base rows pack fill="x" and would stretch across the
        # whole tab), then the four parameter sections in two columns inside
        # one scrolling frame, because filter cards grow the chains downward.
        self.profile_tools = ttk.Frame(self.processing_tab)
        self.profile_tools.pack(side="top", anchor="w", padx=6, pady=(4, 0))
        self.proc_scroll = ScrollFrame(self.processing_tab, width=900,
                                       canvas_width=880)
        self.proc_scroll.pack(side="top", fill="both", expand=True)
        body = self.proc_scroll.inner
        body.columnconfigure(0, weight=1, uniform="cols")
        body.columnconfigure(1, weight=1, uniform="cols")
        self.proc_col_a = ttk.Frame(body)
        self.proc_col_a.grid(row=0, column=0, sticky="nsew")
        self.proc_col_b = ttk.Frame(body)
        self.proc_col_b.grid(row=0, column=1, sticky="nsew")

        self._build_model_tab(self.model_tab)
        self._build_analysis_tab(self.analysis_tab)
        self.center.select(self.right)
        # Bound AFTER the initial select: <<NotebookTabChanged>> fires
        # synchronously on select(), and the viewer does not exist yet.
        self.center.bind("<<NotebookTabChanged>>", self._on_center_tab_changed)

    def _profile_tools_parent(self, section):
        return self.profile_tools

    # -- left pane: picker on top, resizable session lists, Run pinned ----- #
    # The labeler's left pane holds only three sections, so it needs no
    # scrolling; instead the session's three lists divide the height through
    # a vertical paned window, and Run is packed FIRST at the bottom so it
    # keeps its space whatever the window height.
    def _build_left_shell(self):
        self.left_pane = ttk.Frame(self.paned, width=376)
        self.left_bottom = ttk.Frame(self.left_pane)
        self.left_bottom.pack(side="bottom", fill="x")
        self.left = ttk.Frame(self.left_pane)
        self.left.pack(side="top", fill="both", expand=True)
        self.session_paned = None
        self._session_groups = {}

    def _left_section_parent(self, section):
        return self.left_bottom if section == "run" else self.left

    _SESSION_GROUP_WEIGHTS = {"folders": 1, "files": 3, "sequences": 3}

    def _session_group(self, name, section):
        if self.session_paned is None:
            self.session_paned = ttk.PanedWindow(section, orient="vertical")
            self.session_paned.pack(fill="both", expand=True, padx=2, pady=2)
        f = ttk.Frame(self.session_paned)
        self.session_paned.add(f, weight=self._SESSION_GROUP_WEIGHTS.get(name, 1))
        self._session_groups[name] = f
        return f

    def _build_left(self):
        super()._build_left()
        # The section and its lists grow with their panes, and each group's
        # buttons are packed at the bottom AHEAD of the list in pack order,
        # so a pane dragged short clips list rows rather than the buttons.
        self.session_frame.pack_configure(fill="both", expand=True)
        for lb in (self.folder_list, self.file_list):
            lb.master.pack_configure(fill="both", expand=True)
        self.subseq_list.master.pack_configure(fill="both", expand=True)
        self.folder_btn_row.pack_configure(side="bottom", before=self.folder_list.master)
        self.make_seq_btn.pack_configure(side="bottom", before=self.file_list.master)
        self.seq_btn_row.pack_configure(side="bottom", before=self.subseq_list.master)
        self._build_workflow_hint()

    def _processing_parent(self, section):
        if section in ("filters", "base"):
            return self.proc_col_a
        return self.proc_col_b

    # -- hints: the selected workflow and the active model ----------------- #
    # With the profile edited on one tab and the model on another, the panels
    # that USE them need a reminder: the workflow (as a compact chain,
    # session.profile_summary) sits in the Run box above the run settings,
    # the model above Train/Classify. Both are link-styled; a click opens
    # the tab that edits it.
    def _hint_label(self, parent, var, tab, tip):
        label = ttk.Label(parent, textvariable=var, foreground=_HINT_COLOR,
                          cursor="hand2", wraplength=330, justify="left")
        label.bind("<Button-1>", lambda e: self._show_center_tab(tab))
        attach_tooltip(label, tip)
        return label

    def _build_workflow_hint(self):
        """Into the Run section, above the cores / concurrent-slices row."""
        first = self.run_frame.winfo_children()[0]
        self.workflow_hint = self._hint_label(
            self.run_frame, self.workflow_hint_var, "Processing",
            "The active compute profile. Line 1: the topology field the MSC "
            "runs on and the MSC setting (manifold, persistence; mf = merge "
            "forest). Line 2: the base chain the statistics are measured on "
            "and channels × reductions. Click to open the Processing tab.\n"
            "Codes: " + ", ".join(f"{v}={k}" for k, v in session.OP_CODES.items()))
        self.workflow_hint.pack(fill="x", padx=6, pady=(4, 2), before=first)

    def _build_model_hint(self, ml):
        """Into the classifier section; packed first, so above Train/Classify."""
        self.model_hint = self._hint_label(
            ml, self.model_hint_var, "Model",
            "The classifier: the kind Train builds until a model exists, then "
            "the trained/loaded model and its feature count. Click to open "
            "the Model tab.")
        self.model_hint.pack(side="top", fill="x", padx=6, pady=(4, 0))

    def _show_center_tab(self, name):
        tab = getattr(self, "_center_tabs", {}).get(name)
        if tab is None:
            return
        try:
            self.center.select(tab)
        except tk.TclError:
            pass

    def _workflow_hint_text(self):
        try:
            profile = self._profile_from_ui()
        except Exception:
            return "workflow: ?"
        return session.profile_summary(profile)

    def _model_hint_text(self):
        if self._clf is None:
            return f"model: {self.model_kind_var.get()} · not trained"
        names = list(self._clf_names or [])
        bits = [self._clf_kind, f"{len(names)} feats"]
        if self._clf_spec is not None:
            bits[1] = self._clf_spec.brief(len(names))
        expected = self._expected_feature_names()
        if expected is not None and set(expected) != set(names):
            bits.append("⚠ profile mismatch")
        if self._edge_model is not None:
            r = (self._edge_model.report or {}).get("edge", {}).get("learned")
            bits.append("-> edges" + (f" {r['diff_recall']:.0%}/{r['diff_precision']:.0%}" if r else ""))
        return "model: " + " · ".join(bits)

    def _refresh_hints(self):
        """Repaint both hints; a var is only written when its text changed."""
        wv = getattr(self, "workflow_hint_var", None)
        if wv is None or not hasattr(self, "filter_cards"):
            return
        for var, text in ((wv, self._workflow_hint_text()),
                          (self.model_hint_var, self._model_hint_text())):
            if var.get() != text:
                var.set(text)

    def _hint_tick(self):
        """Poll rather than trace: the profile is edited through dozens of
        widgets (cards are rebuilt constantly), and the snapshot is cheap."""
        self._hint_after = None
        try:
            self._refresh_hints()
            self._hint_after = self.root.after(_HINT_POLL_MS, self._hint_tick)
        except tk.TclError:
            pass

    def _build_model_tab(self, parent):
        """Model design. For now: which estimator kind Train builds, and a
        read-only description of its architecture (formatted from the same
        constants _make_model uses)."""
        box = ttk.LabelFrame(parent, text="Region classifier")
        box.pack(side="top", fill="x", padx=6, pady=4)
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=(4, 2))
        ttk.Label(row, text="Kind:").pack(side="left")
        self.model_kind_combo = ttk.Combobox(row, textvariable=self.model_kind_var,
                                             state="readonly", values=_MODEL_KINDS,
                                             width=22)
        self.model_kind_combo.pack(side="left", padx=4)
        self.model_kind_combo.bind("<<ComboboxSelected>>", self._unfocus_entries)
        self.model_arch_var = tk.StringVar(master=self.root, value="")
        self.model_arch_label = ttk.Label(box, textvariable=self.model_arch_var,
                                          justify="left", wraplength=700,
                                          foreground="#333")
        self.model_arch_label.pack(anchor="w", padx=6, pady=(0, 6))
        # A trace rather than the combobox event: Load classifier… sets the
        # kind programmatically and the readout must follow that too.
        self.model_kind_var.trace_add("write", lambda *_: self._on_kind_change())
        self._refresh_model_readout()
        self._build_edge_panel(parent)

        # -- Optimize network: the search over the dense FC space ------------ #
        srch = ttk.LabelFrame(parent, text="Optimize network (dense FC search)")
        srch.pack(side="top", fill="x", padx=6, pady=4)
        row = ttk.Frame(srch); row.pack(fill="x", padx=4, pady=(4, 2))
        for label, var, width, tip in (
                ("Trials:", self.search_trials_var, 6,
                 "How many architectures to score. Trial 1 is always the "
                 "un-tuned dense FC baseline, so the winner never loses to it."),
                ("Time limit (min):", self.search_timeout_var, 6,
                 "Stop after this many minutes (the current trial finishes); "
                 "0 = no limit. The best so far is installed, saved under the "
                 "session's models folder and classified either way -- leave "
                 "it running overnight."),
                ("Seed:", self.search_seed_var, 4,
                 "Seeds the sampler, the folds and the networks: the same "
                 "labels and seed reproduce the same winner.")):
            ttk.Label(row, text=label).pack(side="left")
            ent = ttk.Entry(row, textvariable=var, width=width)
            ent.pack(side="left", padx=(2, 8))
            attach_tooltip(ent, tip)
        ttk.Label(row, text="Backend:").pack(side="left")
        self.search_backend_combo = ttk.Combobox(
            row, textvariable=self.search_backend_var, state="readonly",
            values=model_search.BACKENDS, width=7)
        self.search_backend_combo.pack(side="left", padx=(2, 8))
        self.search_backend_combo.bind("<<ComboboxSelected>>", self._unfocus_entries)
        attach_tooltip(self.search_backend_combo,
                       "torch: the GPU MLP (every fold of a trial trains as one "
                       "stacked batch; adds dropout to the search); sklearn: "
                       "MLPClassifier on the CPU; auto: torch when installed. "
                       f"Now: {model_search.backend_label('auto')}.")
        chk = ttk.Checkbutton(row, text="search feature subset",
                              variable=self.search_features_var)
        chk.pack(side="left", padx=(0, 8))
        attach_tooltip(chk, "Also search which measurement CHANNELS the network "
                            "sees (all reductions of a channel together); off = "
                            "every non-positional field.")
        row = ttk.Frame(srch); row.pack(fill="x", padx=4, pady=2)
        self.optimize_btn = ttk.Button(row, text="Optimize network (O)",
                                       command=self._optimize_network)
        self.optimize_btn.pack(side="left", padx=(0, 4))
        attach_tooltip(self.optimize_btn,
                       "Cross-validate candidate dense networks -- depth, widths, "
                       "L2 alpha, learning rate, batch size, early stopping, "
                       "feature subset -- leaving whole slices out, then install "
                       "the winner as the 'dense (tuned)' model, save it and "
                       "classify. Runs in the background; labeling stays live.\n"
                       f"Each trial trains at most {model_search.SEARCH_MAX_ITER} "
                       f"epochs (patience {model_search.SEARCH_PATIENCE}); mini-"
                       f"batches under {model_search.TORCH_MIN_BATCH} rows train on "
                       "sklearn's CPU MLP, full/large batches on torch; the winner "
                       f"is refit with the full {_MLP_MAX_ITER}-epoch budget.")
        self.cancel_search_btn = ttk.Button(row, text="Cancel", state="disabled",
                                            command=self._cancel_search)
        self.cancel_search_btn.pack(side="left")
        ttk.Label(srch, textvariable=self.search_progress_var, justify="left",
                  wraplength=700, foreground="#333").pack(anchor="w", padx=6,
                                                          pady=(0, 6))

    def _build_analysis_tab(self, parent):
        """The Analysis tab: the regions behind a confusion cell (double-click
        one to go there) above the size sweep. A home for plots later."""
        self._build_errors_panel(parent)
        self._build_sweep_panel(parent)

    def _build_errors_panel(self, parent):
        box = ttk.LabelFrame(parent, text="Predictions vs annotations: the regions behind a confusion cell")
        box.pack(side="top", fill="x", padx=6, pady=4)
        self.errors_header_var = tk.StringVar(
            master=self.root,
            value="Click a cell of the confusion matrix (right panel) to list its regions "
                  "here; double-click the cell to jump to this tab. Double-click a row "
                  "to go to that region.")
        ttk.Label(box, textvariable=self.errors_header_var, justify="left",
                  wraplength=760, foreground="#333").pack(anchor="w", padx=6, pady=(4, 2))
        holder = ttk.Frame(box); holder.pack(fill="x", padx=4, pady=(0, 4))
        cols = ("slice", "region", "true", "pred", "area", "p_true", "p_pred", "flipped")
        self.errors_tree = ttk.Treeview(holder, columns=cols, show="headings", height=8,
                                        selectmode="browse")
        for cid, text, width, anchor in (("slice", "slice", 220, "w"),
                                         ("region", "region id", 70, "e"),
                                         ("true", "annotated", 70, "e"),
                                         ("pred", "predicted", 70, "e"),
                                         ("area", "area (px)", 80, "e"),
                                         ("p_true", "P(annotated)", 90, "e"),
                                         ("p_pred", "P(predicted)", 90, "e"),
                                         ("flipped", "by neighbours", 90, "center")):
            self.errors_tree.heading(cid, text=text)
            self.errors_tree.column(cid, width=width, anchor=anchor, stretch=(cid == "slice"))
        sb = ttk.Scrollbar(holder, orient="vertical", command=self.errors_tree.yview)
        self.errors_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.errors_tree.pack(side="left", fill="x", expand=True)
        self.errors_tree.bind("<Double-1>", self._on_error_row_open)
        self.errors_tree.bind("<Return>", self._on_error_row_open)
        attach_tooltip(self.errors_tree,
                       "One row per region in the selected confusion cell, every slice, "
                       "largest first. P(...) are the current model's class probabilities; "
                       "'by neighbours' marks regions whose class the edge model's voting "
                       "changed. Double-click (or Enter) to open the slice centred on it.")
        self._error_rows = []

    def _confusion_cell_rows(self, i, j):
        """The regions counted in confusion cell (true i, predicted j) on
        every slice classified at the current commit, largest first."""
        import numpy as np
        rows = []
        for (si, li), entry in self._pred.items():
            rec = self.engine.record(si, li)
            if rec is None or rec.get("labels") is None or entry[0] != rec.get("commit"):
                continue
            truth = self._truth_from_cache(si, li, rec, np)
            if truth is None:
                continue
            pred, proba = entry[1], entry[2]
            n = min(len(truth), len(pred))
            hits = np.nonzero((truth[:n] == i) & (pred[:n] == j))[0]
            if not len(hits):
                continue
            aux = self._pred_aux(entry)
            table = rec.get("stats")
            area_of = {}
            if table is not None:
                fid, area = table.column("feature_id"), table.column("area")
                if fid is not None and area is not None:
                    area_of = {int(f): float(a) for f, a in zip(fid, area)}
            key = self._slice_key(si, li) or f"{si}:{li}"
            for r in hits.tolist():
                rows.append({"si": si, "li": li, "slice": key, "region": int(r),
                             "true": int(truth[r]), "pred": int(pred[r]),
                             "area": area_of.get(int(r), 0.0),
                             "p_true": float(proba[r, i]) if i < proba.shape[1] else 0.0,
                             "p_pred": float(proba[r, j]) if j < proba.shape[1] else 0.0,
                             "flipped": bool(aux is not None and aux.get("active")
                                             and int(aux["raw"][r]) != int(pred[r]))})
        rows.sort(key=lambda d: (d["si"], d["li"], -d["area"]))
        return rows

    def _fill_error_list(self, cell):
        """Repaint the Analysis tab's region list for confusion cell `cell`
        ((true, predicted) or None to clear)."""
        tree = getattr(self, "errors_tree", None)
        if tree is None:
            return
        try:
            tree.delete(*tree.get_children())
        except tk.TclError:
            return
        self._error_rows = []
        if cell is None:
            self.errors_header_var.set("No confusion cell selected - click one on the right "
                                       "panel (double-click jumps here).")
            return
        i, j = cell
        rows = self._confusion_cell_rows(i, j)
        self._error_rows = rows
        for k, d in enumerate(rows):
            tree.insert("", "end", iid=str(k), values=(
                d["slice"], d["region"], d["true"], d["pred"], f"{d['area']:.0f}",
                f"{d['p_true']:.3f}", f"{d['p_pred']:.3f}", "yes" if d["flipped"] else ""))
        what = "correct" if i == j else "misclassified"
        n_slices = len({(d["si"], d["li"]) for d in rows})
        self.errors_header_var.set(
            f"annotated {i} -> predicted {j}: {len(rows)} {what} region(s) on {n_slices} "
            f"slice(s), largest first. These are the CURRENT model's predictions against "
            f"the labels it was trained on (a fit check); Evaluate edges / Optimize report "
            f"held-out numbers. Double-click a row to go there.")

    def _on_confusion_open(self, i, j):
        """Double-click on a confusion cell: select it, list its regions on
        the Analysis tab and show that tab."""
        self._cm_cell = (i, j)
        self._refresh_confusion()
        self._refresh_render()
        self._fill_error_list(self._cm_cell)
        self._show_center_tab("Analysis")

    def _on_error_row_open(self, _e=None):
        tree = getattr(self, "errors_tree", None)
        if tree is None:
            return
        sel = tree.selection()
        if not sel:
            return
        try:
            d = self._error_rows[int(sel[0])]
        except (ValueError, IndexError):
            return
        self._goto_region(d["si"], d["li"], d["region"])

    def _goto_region(self, si, li, region):
        """Open the View tab on slice (si, li), centred on `region` (its
        seeding extremum), with the gestures touching it outlined and the
        selected confusion cell still highlighting it."""
        try:
            idx = self.flat_slices.index((si, li))
        except ValueError:
            self.status_var.set(f"slice {si}:{li} is not primed")
            return False
        self._goto_slice(idx)
        self._show_center_tab("View")
        rec = self.engine.record(si, li)
        pt = None
        if rec is not None and rec.get("labels") is not None:
            import numpy as np
            res = _extremum_points(rec["labels"], [int(region)], rec.get("stats"), np)
            pt = res.get(int(region)) if isinstance(res, dict) else (res[0] if res else None)
            if pt is not None and (pt[0] is None or pt[1] is None):
                pt = None
        self.status_var.set(f"region {int(region)} on {self._slice_key(si, li)}"
                            + ("" if pt is not None else " (no position known)"))
        if pt is None:
            return True

        def go():
            # Runs AFTER the View tab's own idle repaint (which may refit the
            # canvas when the first render happened hidden), so the centring
            # is what the user ends up seeing.
            v = self.viewer
            if v is None:
                return
            self._fit_pending = False
            v.center_on(pt[0], pt[1])
            self._hover_key = (si, li, int(region))
            self._hover_uid = None
            self._redraw_hover_geometry()
            sx = (pt[0] - v.view_x) / v.scale
            sy = (pt[1] - v.view_y) / v.scale
            v.canvas.create_oval(sx - 12, sy - 12, sx + 12, sy + 12, outline="#ffffff",
                                 width=2, dash=(4, 3), tags=("draw", "ihover"))
        try:
            self.root.after_idle(go)
        except tk.TclError:
            go()
        return True

    def _build_sweep_panel(self, parent):
        """The Analysis tab's lower half: the size sweep's controls and its
        report -- one row per rung, the best settings the search found for
        that architecture, and how far its held-out loss sits from the best
        rung's. A selected rung can be installed as the model."""
        sw = ttk.LabelFrame(parent, text="Size sweep: how small can the network be?")
        sw.pack(side="top", fill="both", expand=True, padx=6, pady=4)
        row = ttk.Frame(sw); row.pack(fill="x", padx=4, pady=(4, 2))
        ttk.Label(row, text="Sizes:").pack(side="left")
        ent = ttk.Entry(row, textvariable=self.sweep_sizes_var, width=36)
        ent.pack(side="left", padx=(2, 8))
        attach_tooltip(ent, "The ladder of architectures to score, largest first: "
                            "layers joined by '-', rungs by ',' (64-32, 32-16, 16-8, 8-4, 4). "
                            "Each rung is searched with its size FIXED and alpha, learning "
                            "rate, batch, early stopping, dropout and the feature subset "
                            "free, so a small net is judged at its own best settings.")
        ttk.Label(row, text="Trials/size:").pack(side="left")
        ent = ttk.Entry(row, textvariable=self.sweep_trials_var, width=5)
        ent.pack(side="left", padx=(2, 8))
        attach_tooltip(ent, "Search trials spent on each rung (trial 1 is the rung at "
                            "the baseline settings). The Optimize time limit caps the "
                            "whole sweep.")
        self.sweep_btn = ttk.Button(row, text="Sweep sizes", command=self._sweep_network)
        self.sweep_btn.pack(side="left", padx=(0, 4))
        attach_tooltip(self.sweep_btn, "Run one fixed-size search per rung (background; "
                                       "Cancel above stops it), install the best rung, "
                                       "save it and classify. The report fills in as "
                                       "rungs complete.")
        self.sweep_install_btn = ttk.Button(row, text="Install selected size",
                                            state="disabled",
                                            command=self._sweep_install_selected)
        self.sweep_install_btn.pack(side="left")
        attach_tooltip(self.sweep_install_btn,
                       "Make the selected rung the model (its refit estimator, saved "
                       "like an Optimize winner) and classify -- the way to take the "
                       "smallest network that still holds.")
        ttk.Label(sw, textvariable=self.sweep_summary_var, justify="left",
                  wraplength=700, foreground="#333").pack(anchor="w", padx=6, pady=(0, 2))
        holder = ttk.Frame(sw); holder.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        cols = ("size", "params", "logloss", "bacc", "delta", "trials", "settings")
        self.sweep_tree = ttk.Treeview(holder, columns=cols, show="headings",
                                       height=6, selectmode="browse")
        for cid, text, width, anchor in (("size", "size", 70, "w"),
                                         ("params", "params", 70, "e"),
                                         ("logloss", "CV log-loss", 80, "e"),
                                         ("bacc", "bal. acc", 70, "e"),
                                         ("delta", "vs best", 70, "e"),
                                         ("trials", "trials", 50, "e"),
                                         ("settings", "best settings", 380, "w")):
            self.sweep_tree.heading(cid, text=text)
            self.sweep_tree.column(cid, width=width, anchor=anchor,
                                   stretch=(cid == "settings"))
        self.sweep_tree.tag_configure("bad", foreground="#a00000")
        self.sweep_tree.tag_configure("pick", background="#e6f2e6")
        sb = ttk.Scrollbar(holder, orient="vertical", command=self.sweep_tree.yview)
        self.sweep_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.sweep_tree.pack(side="left", fill="both", expand=True)
        self.sweep_tree.bind("<<TreeviewSelect>>", lambda e: self._refresh_sweep_buttons())

    def _sweep_settings(self):
        """``(sizes, trials_per_size)`` from the panel; raises ValueError with
        a readable reason."""
        sizes = model_search.parse_sizes(self.sweep_sizes_var.get())
        try:
            trials = int(float(str(self.sweep_trials_var.get()).strip()))
        except (ValueError, TypeError):
            trials = _SWEEP_TRIALS
        lo, hi = _SWEEP_TRIALS_RANGE
        return sizes, max(lo, min(hi, trials))

    def _refresh_sweep_report(self, rows=None, stopped=False):
        """Repaint the table from `rows` (SweepRows so far); the summary and
        the "vs best" column are relative to the best rung present."""
        tree = getattr(self, "sweep_tree", None)
        if tree is None:
            return
        try:
            tree.delete(*tree.get_children())
        except tk.TclError:
            return
        rows = list(rows or [])
        if not rows:
            self.sweep_summary_var.set("")
            self._refresh_sweep_buttons()
            return
        n_total = len(self._sweep.names) if self._sweep is not None else None
        partial = model_search.SweepResult(rows=rows, names=list(self._sweep.names)
                                           if self._sweep is not None else [],
                                           n_classes=0, elapsed_s=0.0, stopped=stopped)
        pick = partial.smallest_within()
        for i, r in enumerate(rows):
            rel = partial.relative_loss(i)
            tags = []
            if rel > model_search.SWEEP_TOLERANCE:
                tags.append("bad")
            if i == pick:
                tags.append("pick")
            tree.insert("", "end", iid=str(i), tags=tuple(tags), values=(
                r.size, f"{r.n_params:,}", f"{r.cv_score:.3f}", f"{r.cv_bacc:.1%}",
                "best" if rel <= 0 else f"+{rel:.0%}", str(r.n_trials),
                r.spec.settings_text(n_total)))
        self.sweep_summary_var.set(partial.summary())
        self._refresh_sweep_buttons()

    def _refresh_sweep_buttons(self):
        btn = getattr(self, "sweep_install_btn", None)
        if btn is None:
            return
        ok = (self._sweep is not None and self._search is None
              and bool(self.sweep_tree.selection()))
        btn.config(state="normal" if ok else "disabled")

    def _refresh_model_readout(self):
        var = getattr(self, "model_arch_var", None)
        if var is not None:
            var.set(self._model_description_now(self.model_kind_var.get()))

    def _model_description_now(self, kind):
        """`_model_description` for the app's state: the tuned kind renders the
        last search's winner over the trained model's column count; the custom
        kind its typed sizes; an edge kind adds the stacked pair model."""
        base = _base_kind(kind)
        n = len(self._clf_names) if self._clf_names else None
        if base == _TUNED_KIND:
            spec = self._search_spec
            text = _model_description(_TUNED_KIND, spec, n if spec is not None else None)
        elif base == _CUSTOM_KIND:
            text = (f"custom FC: FeatureSubset(all) -> StandardScaler -> MLP(hidden layers "
                    f"{self._custom_hidden()}, max_iter {_MLP_MAX_ITER}), fit with balanced "
                    f"sample weights; backend {model_search.backend_label(self._search_settings()[4])}.")
        else:
            text = _model_description(kind)
        if _is_edge_kind(kind):
            widths = None
            try:
                widths = edge_model.hidden_widths(self._clf) if self._clf is not None else None
            except TypeError:
                widths = None
            spec = self._edge_spec_from_ui()
            edge = self._edge_model
            if edge is not None:
                text += " -> " + edge.describe(widths)
                if edge.spec != spec:
                    text += " [edge settings changed - Train (R) to apply]"
            else:
                text += " -> " + spec.describe(widths) + " [not trained yet - Train (R)]"
            if self.freeze_base_var.get():
                text += " [base frozen: Train refits only the edges]"
        return text

    def _custom_hidden(self):
        """The custom base's hidden sizes from the Model tab entry (first rung
        of a ladder-style string), (16, 8) when unreadable."""
        try:
            return model_search.parse_sizes(self.custom_hidden_var.get())[0]
        except (ValueError, IndexError):
            return tuple(model_search.parse_sizes(_DEFAULT_CUSTOM_HIDDEN)[0])

    def _edge_spec_from_ui(self):
        layer = edge_model.LAYER_CHOICES.get(self.edge_layer_var.get(), -1)
        feats = tuple(f for f in edge_model.FEATURE_KINDS if self.edge_feat_vars[f].get())
        model = self.edge_model_var.get()
        if model not in edge_model.EDGE_MODELS:
            model = "logistic"
        c = _bounded_float(self.edge_c_var.get(), 1.0, *_EDGE_C_RANGE)
        lam = _bounded_float(self.edge_lam_var.get(), 1.0, *_EDGE_LAM_RANGE)
        rounds = int(round(_bounded_float(self.edge_rounds_var.get(), 3.0, *_EDGE_ROUNDS_RANGE)))
        return edge_model.EdgeSpec(layer=int(layer), features=feats or edge_model.FEATURE_KINDS,
                                   model=model, C=float(c), lam=float(lam), rounds=rounds,
                                   seed=int(self._search_settings()[2]))

    def _apply_edge_spec(self, spec):
        """Push an EdgeSpec onto the Model tab controls."""
        for name, idx in edge_model.LAYER_CHOICES.items():
            if idx == spec.layer:
                self.edge_layer_var.set(name)
        for f, var in self.edge_feat_vars.items():
            var.set(f in spec.features)
        self.edge_model_var.set(spec.model)
        self.edge_c_var.set(f"{spec.C:g}")
        self.edge_lam_var.set(f"{spec.lam:g}")
        self.edge_rounds_var.set(str(int(spec.rounds)))

    def _on_kind_change(self):
        """The kind combobox / N: the readout follows, and switching between an
        edge kind and its base re-votes the cached predictions in place."""
        self._refresh_model_readout()
        if getattr(self, "_edge_model", None) is not None and getattr(self, "_pred", None):
            self._revote_all()
        self._refresh_edge_readout()

    def _build_edge_panel(self, parent):
        """The edge model behind the `-> edges` kinds: the custom base, the pair
        model's settings, the voting weight, an evaluation report."""
        box = ttk.LabelFrame(parent, text="Edge model (-> edges kinds): a pair model over the region graph")
        box.pack(side="top", fill="x", padx=6, pady=4)
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=(4, 2))
        ttk.Label(row, text="base hidden:").pack(side="left")
        en = ttk.Entry(row, textvariable=self.custom_hidden_var, width=8)
        en.pack(side="left", padx=(2, 6))
        en.bind("<Return>", lambda e: (self._unfocus_entries(), self._on_edge_settings_change()))
        en.bind("<FocusOut>", lambda e: self._on_edge_settings_change())
        attach_tooltip(en, "Hidden layer sizes of the custom FC base (e.g. 16-8), used by "
                           "the custom FC / custom FC -> edges kinds.")
        chk = ttk.Checkbutton(row, text="freeze base", variable=self.freeze_base_var,
                              command=self._on_edge_settings_change)
        chk.pack(side="left", padx=(0, 10))
        attach_tooltip(chk, "Train (R) on a -> edges kind keeps the current base net and "
                            "refits only the edge model: try edge variants on the same base.")
        ttk.Label(row, text="layer:").pack(side="left")
        cb = ttk.Combobox(row, textvariable=self.edge_layer_var, state="readonly",
                          values=list(edge_model.LAYER_CHOICES), width=8)
        cb.pack(side="left", padx=(2, 6))
        cb.bind("<<ComboboxSelected>>", lambda e: (self._unfocus_entries(), self._on_edge_settings_change()))
        attach_tooltip(cb, "Which hidden layer of the base net embeds a region: last (the "
                           "narrow one, best in the experiment) or the previous, wider one.")
        ttk.Label(row, text="model:").pack(side="left")
        cb = ttk.Combobox(row, textvariable=self.edge_model_var, state="readonly",
                          values=list(edge_model.EDGE_MODELS), width=8)
        cb.pack(side="left", padx=(2, 6))
        cb.bind("<<ComboboxSelected>>", lambda e: (self._unfocus_entries(), self._on_edge_settings_change()))
        attach_tooltip(cb, "The pair classifier: a balanced logistic regression, or an MLP 32-16.")
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=2)
        lbl = ttk.Label(row, text="pair inputs:")
        lbl.pack(side="left")
        attach_tooltip(lbl, "What the pair model sees for an edge between two regions, "
                            "built from the base net's hidden-layer embedding of each "
                            "(symmetric in the two regions).")
        for f, label, tip in (
                ("absdiff", "|d|", "Absolute difference of the two regions' embeddings, "
                                   "element by element: how far apart they sit."),
                ("prod", "product", "Element-wise product of the two embeddings: which "
                                     "features both regions share."),
                ("barrier", "barrier", "The saddle joining the two regions: its depth "
                                       "above their extrema (saddle - max(ext_a, ext_b)) "
                                       "and |ext_a - ext_b|. Zero on pixel adjacency.")):
            chk = ttk.Checkbutton(row, text=label, variable=self.edge_feat_vars[f],
                                  command=self._on_edge_settings_change)
            chk.pack(side="left", padx=(2, 4))
            attach_tooltip(chk, tip)
        for label, var, width, tip in (
                ("C:", self.edge_c_var, 5, "Logistic inverse regularisation (0.001..1000)."),
                ("lambda:", self.edge_lam_var, 5,
                 "Weight of the edge terms in neighbour voting (0 = the base net alone; "
                 "applies at once to cached predictions)."),
                ("rounds:", self.edge_rounds_var, 4,
                 "Voting rounds (0 = no refinement); applies at once.")):
            ttk.Label(row, text=label).pack(side="left", padx=(8, 0))
            en = ttk.Entry(row, textvariable=var, width=width)
            en.pack(side="left", padx=(2, 2))
            en.bind("<Return>", lambda e: (self._unfocus_entries(), self._on_edge_settings_change()))
            en.bind("<FocusOut>", lambda e: self._on_edge_settings_change())
            attach_tooltip(en, tip)
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=2)
        self.edge_eval_btn = ttk.Button(row, text="Evaluate edges", command=self._evaluate_edges)
        self.edge_eval_btn.pack(side="left", padx=(0, 6))
        attach_tooltip(self.edge_eval_btn,
                       "Leave-slices-out report for the pair model against the base net's own "
                       "answers (argmax differs / 1 - sum P_a P_b), and region errors before and "
                       "after voting. Refits the base per fold on a worker thread; Cancel above.")
        ttk.Label(row, textvariable=self.edge_progress_var, foreground="#333",
                  wraplength=560, justify="left").pack(side="left", fill="x", expand=True)
        holder = ttk.Frame(box); holder.pack(fill="x", padx=4, pady=(0, 4))
        cols = ("model", "n", "bacc", "auc", "logloss", "recall", "prec")
        self.edge_tree = ttk.Treeview(holder, columns=cols, show="headings", height=5)
        for cid, text, width, anchor in (("model", "edge model / region", 190, "w"),
                                         ("n", "n", 60, "e"), ("bacc", "bal. acc", 70, "e"),
                                         ("auc", "AUC", 60, "e"), ("logloss", "log-loss", 70, "e"),
                                         ("recall", "diff recall", 80, "e"),
                                         ("prec", "diff prec", 80, "e")):
            self.edge_tree.heading(cid, text=text)
            self.edge_tree.column(cid, width=width, anchor=anchor, stretch=(cid == "model"))
        self.edge_tree.pack(side="left", fill="x", expand=True)
        attach_tooltip(self.edge_tree,
                       "HELD-OUT numbers: every slice is scored by models that never saw "
                       "it (leave-slices-out folds). Edge rows: how well each answer tells "
                       "same-class from different-class edges (diff recall / precision = "
                       "boundary edges found / boundary calls that were right). Region rows: "
                       "accuracy on the labeled regions before and after neighbour voting. "
                       "The confusion matrix on the right is NOT held-out (the model saw "
                       "those labels), so its errors are fewer.")

    def _on_edge_settings_change(self, *_a):
        """Edge settings edited: lambda / rounds apply to the cached predictions
        at once; the rest wait for Train (the readout says so)."""
        self._refresh_model_readout()
        if self._edge_model is not None and self._pred:
            self._revote_all()
        self._refresh_edge_readout()

    def _fill_edge_report(self, report):
        tree = getattr(self, "edge_tree", None)
        if tree is None:
            return
        try:
            tree.delete(*tree.get_children())
        except tk.TclError:
            return
        for name in edge_model.EVAL_ROWS:
            s = (report or {}).get("edge", {}).get(name)
            if s:
                tree.insert("", "end", values=(name, s["n"], f"{s['bacc']:.3f}", f"{s['auc']:.3f}",
                                               f"{s['logloss']:.3f}", f"{s['diff_recall']:.1%}",
                                               f"{s['diff_precision']:.1%}"))
        reg = (report or {}).get("region", {})
        for key, label in (("before", "region net alone (held-out)"),
                           ("after", "+ neighbour voting (held-out)")):
            r = reg.get(key)
            if r:
                tree.insert("", "end", values=(label, r["n"], f"{r['bacc']:.3f}", "",
                                               f"acc {r['acc']:.3f}",
                                               f"{r['errors']} held-out errors", ""))

    def _center_tab_name(self):
        """Name of the selected center tab ("View" when unsure)."""
        try:
            selected = self.center.select()
        except tk.TclError:
            return "View"
        for name, frame in self._center_tabs.items():
            if str(frame) == str(selected):
                return name
        return "View"

    def _on_center_tab_changed(self, _e=None):
        """The View tab came back: repaint, and redo the one-time fit if the
        first render of this run happened while the canvas was unmapped."""
        if self._center_tab_name() != "View" or self.viewer is None:
            return

        def repaint():
            if self.viewer is None:
                return
            self._refresh_render()
            if self._fit_pending:
                self._fit_pending = False
                self.viewer.fit()

        try:
            self.root.after_idle(repaint)
        except tk.TclError:
            pass

    def _refresh_render(self):
        viewer = self.viewer
        first = (viewer is not None and viewer._base is None
                 and viewer._source is None)
        super()._refresh_render()
        if first and viewer is not None and viewer._base is not None:
            try:
                mapped = bool(viewer.canvas.winfo_viewable())
            except tk.TclError:
                mapped = True
            if not mapped:
                self._fit_pending = True

    # ------------------------------------------------------------------ #
    # Right-side overrides (viewer area is inherited unchanged)
    # ------------------------------------------------------------------ #
    def _build_right(self):
        """Labeler layout: image information and navigation hug the canvas."""
        self._build_viewer_area(self.right)
        ttk.Label(self.right, textvariable=self.hover_var, anchor="w",
                  font=("TkFixedFont", 8)).pack(fill="x", padx=4)

        row = ttk.Frame(self.right); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Slice:").pack(side="left")
        self._build_slice_nav(row)

        self._build_image_controls(self.right)
        self._build_live_panel(self.right)

    def _build_image_controls(self, parent):
        chan = ttk.Frame(parent); chan.pack(fill="x", padx=4, pady=2)
        ttk.Label(chan, text="Image:").pack(side="left", padx=(4, 0))
        self.background_var = tk.StringVar(value="base")
        self.background_combo = ttk.Combobox(chan, textvariable=self.background_var,
                                             values=["base", "filtered"], state="readonly",
                                             width=18)
        self.background_combo.pack(side="left", padx=4)
        self.background_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_render())
        ttk.Label(chan, text="Image Min/Max:").pack(side="left", padx=(12, 4))
        self._scale(chan, from_=0.0, to=1.0, variable=self.vmin_var, orient="horizontal",
                    command=lambda *_: self._refresh_render()
                    ).pack(side="left", fill="x", expand=True)
        self._scale(chan, from_=0.0, to=1.0, variable=self.vmax_var, orient="horizontal",
                    command=lambda *_: self._refresh_render()
                    ).pack(side="left", fill="x", expand=True)

        overlay = ttk.Frame(parent); overlay.pack(fill="x", padx=4, pady=2)
        self.overlay_master_check = ttk.Checkbutton(
            overlay, text="show overlay (Tab)", variable=self.show_overlay_var,
            command=self._on_overlay_toggle)
        self.overlay_master_check.pack(side="left", padx=(4, 8))
        ttk.Separator(overlay, orient="vertical").pack(side="left", fill="y",
                                                       padx=(0, 6))
        # One toggle instead of the viewer's five seg sources: the labeler works
        # on the full MSC labeling ("msc"), shown faintly under the class layer.
        # seg_source stays a valid viewer value so _needed_level() is always
        # "slice" and every inherited path keeps working; the mask is never on.
        self.show_regions_check = ttk.Checkbutton(
            overlay, text="color regions", variable=self.show_regions_var,
            command=self._on_regions_toggle)
        self.show_regions_check.pack(side="left", padx=(0, 4))
        # The checkbox says WHETHER the region layer is drawn; the dropdown
        # says how it is colored. The scalar modes need a probability cache,
        # so the list is repopulated by Classify and emptied by anything that
        # invalidates it.
        self.region_mode_combo = ttk.Combobox(
            overlay, textvariable=self.region_mode_var, values=[_MODE_ID],
            state="readonly", width=13)
        self.region_mode_combo.pack(side="left", padx=(0, 4))
        self.region_mode_combo.bind("<<ComboboxSelected>>",
                                    lambda e: self._refresh_render())
        ttk.Separator(overlay, orient="vertical").pack(side="left", fill="y",
                                                       padx=(6, 6))
        self.show_gt_check = ttk.Checkbutton(
            overlay, text="Show GT", variable=self.show_gt_var,
            command=self._refresh_render)
        self.show_gt_check.pack(side="left")
        self.show_pred_check = ttk.Checkbutton(
            overlay, text="Show Classification", variable=self.show_pred_var,
            command=self._refresh_render)
        self.show_pred_check.pack(side="left", padx=(6, 0))
        ttk.Separator(overlay, orient="vertical").pack(side="left", fill="y",
                                                       padx=(6, 6))
        self.show_annot_check = ttk.Checkbutton(
            overlay, text="show annotations", variable=self.show_annot_var,
            command=self._refresh_annotation_layer)
        self.show_annot_check.pack(side="left")
        self._overlay_dependents = (
            (self.show_regions_check, "normal"),
            (self.region_mode_combo, "readonly"),
            (self.show_gt_check, "normal"),
            (self.show_pred_check, "normal"),
            (self.show_annot_check, "normal"),
        )

        alpha = ttk.Frame(parent); alpha.pack(fill="x", padx=4, pady=2)
        ttk.Label(alpha, text="Overlay alpha:").pack(side="left")
        self._scale(alpha, from_=0.0, to=1.0, variable=self.alpha_var,
                    orient="horizontal",
                    command=lambda *_: self._refresh_render()
                    ).pack(side="left", fill="x", expand=True)
        self.mask_var = tk.BooleanVar(value=False)
        self.seg_source_var.set("msc" if self.show_regions_var.get() else "none")

    def _on_overlay_toggle(self):
        """Apply the master overlay state to every control in its row."""
        enabled = self.show_overlay_var.get()
        for widget, active_state in self._overlay_dependents:
            widget.configure(state=active_state if enabled else "disabled")
        self._refresh_annotation_layer()
        self._refresh_render()

    def _on_regions_toggle(self):
        self.seg_source_var.set("msc" if self.show_regions_var.get() else "none")
        self._on_seg_source_change()

    # -- regions coloring modes ------------------------------------------ #
    def _region_modes(self):
        """The coloring modes offered right now: the id LUT always, the scalar
        ones only once a probability cache exists."""
        modes = [_MODE_ID]
        if any(len(v) > 2 for v in self._pred.values()):
            modes += [f"P(class {k})" for k in range(1, self.store.n_classes)]
            modes.append(_MODE_UNCERTAINTY)
        if any((self._pred_aux(v) or {}).get("pdiff") is not None for v in self._pred.values()):
            modes += [_MODE_FLIPPED, _MODE_PDIFF]
        return modes

    @staticmethod
    def _pred_aux(entry):
        """The edge-model side of a prediction cache entry (None without one)."""
        return entry[3] if entry is not None and len(entry) > 3 else None

    def _refresh_region_modes(self):
        """Repopulate the dropdown, falling back to `label id` when the mode
        that was selected is no longer available."""
        combo = getattr(self, "region_mode_combo", None)
        if combo is None:
            return
        modes = self._region_modes()
        try:
            combo.config(values=modes)
        except tk.TclError:
            return
        if self.region_mode_var.get() not in modes:
            self.region_mode_var.set(_MODE_ID)

    def _region_scalar(self, entry, np):
        """(values, mask) for the selected scalar mode over `entry`'s regions,
        or None when the mode is `label id` / the entry has no probabilities."""
        mode = self.region_mode_var.get()
        if mode == _MODE_ID or len(entry) < 3:
            return None
        proba = entry[2]
        # A region the classifier never scored sums to 0; those stay invisible
        # rather than rendering as ramp-zero.
        mask = proba.sum(1) > 0
        if mode == _MODE_UNCERTAINTY:
            top = np.sort(proba, axis=1)
            return 1.0 - (top[:, -1] - top[:, -2]), mask
        if mode in (_MODE_FLIPPED, _MODE_PDIFF):
            aux = self._pred_aux(entry)
            if aux is None or aux.get("pdiff") is None:
                return None
            if mode == _MODE_FLIPPED:
                return (np.asarray(aux["raw"]) != np.asarray(entry[1])).astype(np.float32), mask
            out = np.zeros(len(entry[1]), np.float32)
            k = aux["keep"]
            pd = np.asarray(aux["pdiff"], np.float32)[k]
            np.maximum.at(out, np.asarray(aux["la"])[k], pd)
            np.maximum.at(out, np.asarray(aux["lb"])[k], pd)
            return out, mask
        try:
            k = int(mode[len("P(class "):-1])
        except ValueError:
            return None
        if not (0 <= k < proba.shape[1]):
            return None
        return proba[:, k], mask

    def _handle_event(self, ev):
        if ev[0] == "primed":
            # The commit bump already invalidates these; dropping them outright
            # also frees the old run's arrays.
            self._class_luts.clear()
            self._pred.clear()
            self._cm_cell = None
            self._refresh_region_modes()
        super()._handle_event(ev)
        if ev[0] == "primed":
            self._rebuild_class_panels()      # slice 0 is now on screen
        elif ev[0] == "assembly_done":
            self._update_class_titles()       # a new record can change counts

    def _rerun_selection(self):
        """A new region commit invalidates every frozen region prediction."""
        self._pred.clear()
        self._cm_cell = None
        self._refresh_region_modes()
        super()._rerun_selection()
        self._refresh_confusion()

    def _goto_slice(self, idx):
        cur = self._current()
        before = self._slice_key(*cur) if cur is not None else None
        super()._goto_slice(idx)
        cur = self._current()
        after = self._slice_key(*cur) if cur is not None else None
        if after != before:
            # The class panels list the ON-SLICE interactions only; swap them
            # with the slice.
            self._rebuild_class_panels()
            self._refresh_edge_readout()

    def _build_live_panel(self, parent):
        # persistence: region identity depends on it, so it stays adjustable;
        # commit is the Rerun button exactly as in the viewer. These controls
        # live directly under the image controls rather than in a subpanel.
        row = ttk.Frame(parent); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Persistence %:").pack(side="left")
        self.persist_entry = ttk.Entry(row, textvariable=self.persist_live_var, width=8)
        self.persist_entry.pack(side="left", padx=4)
        self.persist_entry.bind("<Return>", self._on_persistence_change)
        self.persist_entry.bind("<FocusOut>", self._on_persistence_change)
        self.persist_value_label = ttk.Label(row, text="")
        self.persist_value_label.pack(side="left", padx=4)

        # No per-slice selection / pixel trim / connectivity: the labeler works
        # on the full pre-filter MSC labeling (the guarded _rebuild_*_cards
        # no-op without their frames).
        self.rerun_btn = ttk.Button(row, text="update region simplification", state="disabled",
                                    command=self._rerun_selection)
        self.rerun_btn.pack(side="left", padx=4)

    def _image_window(self, _channel):
        """One fractional window follows whichever image channel is active."""
        return self.vmin_var.get(), self.vmax_var.get()

    # ------------------------------------------------------------------ #
    # Rendering: the class layer
    # ------------------------------------------------------------------ #
    def _seg_overlays(self, si, li, rec, data, np, min_colors):
        # A repaint means the slice/segmentation may have changed under any
        # hover geometry on screen; drop it (the next Motion redraws it).
        if self.viewer is not None and self._hover_key is not None:
            self.viewer.canvas.delete("ihover")
            self._hover_key = None
        if not self.show_overlay_var.get():
            return []                # master switch (Tab): base image only
        overlays = super()._seg_overlays(si, li, rec, data, np, min_colors)
        scalar = None
        if rec is not None and self.show_regions_var.get():
            entry = self._pred.get((si, li))
            if entry is not None and entry[0] == rec.get("commit"):
                scalar = self._region_scalar(entry, np)
        if scalar is not None:
            # A coloring mode REPLACES the id LUT rather than tinting it: the
            # ids and the scalar are two readings of the same regions, and
            # stacking them would just muddy both.
            values, mask = scalar
            overlays = [o for o in overlays if "lut" not in o]
            overlays.append({"labels": rec["labels"],
                             "lut": scalar_lut(values, np, _SCALAR_ALPHA, mask),
                             "visible": True})
        else:
            # The inherited region overlay is orientation, not the point: fade
            # it under the class layer (copy first -- _id_lut results may be
            # shared).
            for o in overlays:
                if "lut" in o:
                    o["lut"] = o["lut"].copy()
                    o["lut"][:, 3] = np.minimum(o["lut"][:, 3], _REGION_ALPHA)
        if rec is not None:
            # Classifier predictions under the user's own labels: the model's
            # view of every region, with the drawn ground truth on top.
            if self.show_pred_var.get():
                pr = self._pred.get((si, li))
                if pr is not None and pr[0] == rec.get("commit"):
                    plut = class_lut(pr[1], np,
                                     self._class_colors_rgba(np)).copy()
                    plut[:, 3] = (plut[:, 3].astype(np.uint16)
                                  * _PRED_ALPHA // 255).astype(np.uint8)
                    overlays.append({"labels": rec["labels"], "lut": plut,
                                     "visible": True})
            if self.show_gt_var.get():
                lut = self._class_lut_for(si, li, rec, np)
                if lut is not None:
                    overlays.append({"labels": rec["labels"], "lut": lut,
                                     "visible": True})
            # A selected confusion cell outranks everything: it is a question
            # about WHERE those regions are, so it goes on top, opaque.
            hits = self._confusion_hits()
            if hits:
                K = max(int(rec["labels"].max()) + 1 if rec["labels"].size
                        else 1, 1)
                hl = np.zeros((K, 4), np.uint8)
                ids = [r for r in hits if r < K]
                if ids:
                    hl[ids] = (255, 255, 255, 255)
                    overlays.append({"labels": rec["labels"], "lut": hl,
                                     "visible": True})
        return overlays

    def _labels_cache_for(self, si, li, rec, np):
        """(commit, rev, lut, {uid: touched ids}, per-class region counts,
        region id -> class) for
        one slice, memoized on (commit, store.rev): a gesture bumps rev
        (rebuild only this), a Rerun bumps the commit (a fresh labels raster
        arrives and the interactions re-resolve against it -- which is how
        annotations survive persistence changes). The touch map and the counts
        fall out of the same rasterization pass the LUT needs, so the hover
        lookup and the class-title totals cost nothing extra."""
        key = (si, li)
        commit = rec.get("commit")
        cached = self._class_luts.get(key)
        if cached is not None and cached[0] == commit and cached[1] == self.store.rev:
            return cached
        lut, touch, region_class = None, {}, None
        counts = np.zeros(MAX_CLASSES, np.int64)
        slice_key = self._slice_key(si, li)
        if slice_key is not None:
            its = self.store.for_slice(slice_key)
            if its:
                sets = touched_sets(its, rec["labels"], np)
                touch = {it.uid: ids for it, ids in sets}
                region_class = resolve_sets(sets, rec["labels"], np)
                lut = class_lut(region_class, np, self._class_colors_rgba(np))
                counts = np.bincount(region_class,
                                     minlength=MAX_CLASSES)[:MAX_CLASSES]
        # region_class rides along: the confusion matrix needs per-region truth
        # on every label edit, and this pass already produced it.
        entry = (commit, self.store.rev, lut, touch, counts, region_class)
        self._class_luts[key] = entry
        return entry

    def _class_colors_rgba(self, np):
        """(MAX_CLASSES, 4) RGBA table: the store's user-picked colors over
        the defaults. A color change bumps store.rev, so every LUT cache
        keyed on it rebuilds."""
        return np.asarray([self.store.rgba(k) for k in range(MAX_CLASSES)],
                          np.uint8)

    def _class_color_hex(self, k):
        return self.store.color(k)

    def _class_lut_for(self, si, li, rec, np):
        return self._labels_cache_for(si, li, rec, np)[2]

    def _touch_map_for(self, si, li, rec, np):
        return self._labels_cache_for(si, li, rec, np)[3]

    def _annotation_count(self, si, li):
        key = self._slice_key(si, li)
        return len(self.store.for_slice(key)) if key else 0

    def _slice_key(self, si, li):
        """Folder-qualified slice identity ("folder/basename") -- basenames
        collide across a session's folders, so the folder is part of the key."""
        try:
            s = self.subsequences[si]
            return f"{s.get('folder', '')}/{os.path.basename(s['files'][li])}"
        except (IndexError, KeyError, TypeError):
            return None

    # ------------------------------------------------------------------ #
    # Interactions
    # ------------------------------------------------------------------ #
    def _commit_interaction(self, tool, points):
        cur = self._current()
        if cur is None:
            return
        si, li = cur
        slice_key = self._slice_key(si, li)
        cls = int(self.active_class_var.get())
        if slice_key is None or not (1 <= cls < self.store.n_classes):
            return
        self._push_history()
        it = self.store.add(tool, points, cls, slice_key, si, li)
        self._rebuild_class_panels()
        self._refresh_render()
        self.status_var.set(f"#{it.uid} {tool} -> class {cls} ({slice_key})")

    # -- "will be painted" preview (shared by every drawing tool) ---------- #
    def _begin_preview(self):
        """A gesture is in flight: park the hover outlines until _end_preview
        (the pointer is drawing, not asking about regions)."""
        self._hover_suppressed = True
        if self.viewer is not None:
            self.viewer.canvas.delete("ihover")
        self._hover_key = None

    def _preview_regions(self, labels, K, ids, colors, emphasize=None):
        """Show `ids` on the canvas's transient layer in a brightened, opaque
        version of their class color (see labeling.preview_lut)."""
        v = self.viewer
        if v is None:
            return
        import numpy as np
        v.set_transient({"labels": labels,
                         "lut": preview_lut(K, ids, colors, np, emphasize=emphasize)})
        v._schedule()

    def _end_preview(self):
        self._hover_suppressed = False
        v = self.viewer
        if v is not None:
            v.set_transient(None)
            v._schedule()

    def _commit_magic(self, si, li, labels, ids, cls, meta):
        """Magic-fill release: one part, see _commit_blob."""
        self._commit_blob(si, li, labels, [(ids, cls, meta)])

    def _commit_blob(self, si, li, labels, parts):
        """Magic-fill / blobber release: each `(ids, cls, meta)` part becomes
        ONE "taps" interaction with a point per region at its seeding extremum
        (the pixel most likely to stay inside the region when the
        decomposition changes), so the fill re-resolves through the same
        geometric path as every other gesture. Parts are added in order (a
        later uid paints over an earlier one), the provenance rides along as
        display metadata, and the whole release is one undo step."""
        slice_key = self._slice_key(si, li)
        if slice_key is None:
            return
        import numpy as np
        rec = self.engine.record(si, li)
        table = rec.get("stats") if rec is not None else None
        added = []
        for ids, cls, meta in parts:
            ids = [int(i) for i in ids]
            if not ids or not (1 <= int(cls) < self.store.n_classes):
                continue
            pts = _extremum_points(labels, ids, table, np)
            if not pts:
                continue
            if not added:
                self._push_history()
            added.append(self.store.add("taps", pts, int(cls), slice_key, si, li,
                                        meta=meta))
        if not added:
            return
        self._rebuild_class_panels()
        self._refresh_render()
        what = " + ".join(f"#{it.uid} {(it.meta or {}).get('part', 'fill')} "
                          f"-> class {it.class_id} ({len(it.points)})"
                          for it in added)
        tool = (added[-1].meta or {}).get("tool", "magic")
        t = (added[-1].meta or {}).get("threshold", 0.0)
        self.status_var.set(f"{tool}: {what} at t={t:.3g}")

    def _accept_predictions(self, pts):
        """SHIFT-box release: turn the classifier's predictions under the box
        into real labels -- one "taps" interaction per predicted class, one
        point per accepted region (its first pixel inside the box), so the
        acceptance is geometric and re-resolves after a recompute like any
        other gesture. One undo step for the whole batch."""
        cur = self._current()
        if cur is None or len(pts) < 2:
            return
        si, li = cur
        rec = self.engine.record(si, li)
        pr = self._pred.get((si, li))
        if rec is None or rec.get("labels") is None or pr is None \
                or pr[0] != rec.get("commit"):
            self.status_var.set("Accept needs predictions - Classify first.")
            return
        import numpy as np
        labels = rec["labels"]
        region_class = pr[1]
        h, w = labels.shape
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        xa, xb = sorted((int(round(x0)), int(round(x1))))
        ya, yb = sorted((int(round(y0)), int(round(y1))))
        xa, xb = max(xa, 0), min(xb, w - 1)
        ya, yb = max(ya, 0), min(yb, h - 1)
        if xa > xb or ya > yb:
            return
        sub = labels[ya:yb + 1, xa:xb + 1]
        by_class = {}
        for r in np.unique(sub):
            r = int(r)
            if r < 0 or r >= len(region_class):
                continue
            k = int(region_class[r])
            if not (1 <= k < self.store.n_classes):
                continue
            yy, xx = np.argwhere(sub == r)[0]     # a pixel of r inside the box
            by_class.setdefault(k, []).append((float(xa + xx), float(ya + yy)))
        if not by_class:
            self.status_var.set("No predicted classes under the box.")
            return
        slice_key = self._slice_key(si, li)
        self._push_history()
        n = 0
        for k in sorted(by_class):
            self.store.add("taps", by_class[k], k, slice_key, si, li)
            n += len(by_class[k])
        self._rebuild_class_panels()
        self._refresh_render()
        self.status_var.set(f"Accepted {n} predicted region(s) into "
                            f"{len(by_class)} class(es)")

    def _delete_interaction(self, uid):
        if self.store.get(uid) is None:
            return
        self._push_history()
        self.store.remove(uid)
        self._rebuild_class_panels()
        self._refresh_render()

    def _move_interaction(self, uid, class_id):
        it = self.store.get(uid)
        if it is None or it.class_id == int(class_id):
            return
        self._push_history()
        self.store.set_class(uid, class_id)
        self._rebuild_class_panels()
        self._refresh_render()

    def _install_store(self, store, clear_history=True):
        """Adopt a loaded/restored store: rebind slice identities against the
        current subsequences and rebuild everything derived. Loading a file or
        a session starts a fresh history; undo/redo pass clear_history=False
        because they ARE the history."""
        self.store = store
        unbound = store.rebind(self.subsequences)
        self._class_luts.clear()
        if clear_history:
            self._undo_stack.clear()
            self._redo_stack.clear()
        self.n_classes_var.set(store.n_classes)
        if self.active_class_var.get() >= store.n_classes:
            self.active_class_var.set(0)
        self._rebuild_class_panels()
        self._refresh_render()
        if unbound:
            self.status_var.set(f"{unbound} interaction(s) reference slices not "
                                "in the current subsequences (kept, shown grey)")

    # -- undo / redo ----------------------------------------------------- #
    def _push_history(self):
        """Call BEFORE a store mutation. A new edit wipes the redo branch."""
        self._undo_stack.append(self.store.to_json())
        if len(self._undo_stack) > 200:
            del self._undo_stack[0]
        self._redo_stack.clear()

    def _undo(self):
        if not self._undo_stack:
            self.status_var.set("Nothing to undo")
            return
        self._redo_stack.append(self.store.to_json())
        self._install_store(LabelStore.from_json(self._undo_stack.pop()),
                            clear_history=False)
        self.status_var.set(f"Undo ({len(self._undo_stack)} left)")

    def _redo(self):
        if not self._redo_stack:
            self.status_var.set("Nothing to redo")
            return
        self._undo_stack.append(self.store.to_json())
        self._install_store(LabelStore.from_json(self._redo_stack.pop()),
                            clear_history=False)
        self.status_var.set(f"Redo ({len(self._redo_stack)} left)")

    # ------------------------------------------------------------------ #
    # Hotkeys
    # ------------------------------------------------------------------ #
    def _bind_hotkeys(self):
        for k in range(0, MAX_CLASSES):
            self.root.bind(str(k), self._on_class_key)
        self.root.bind("<Escape>", self._on_escape)
        self.root.bind("m", self._on_magic_key)
        self.root.bind("M", self._on_magic_key)
        self.root.bind("b", self._on_blob_key)
        self.root.bind("B", self._on_blob_key)
        self.root.bind("<Control-z>", self._on_undo_key)
        self.root.bind("<Control-y>", self._on_undo_key)
        self.root.bind("<Tab>", self._on_tab_toggle)
        # 'R': one keystroke = train + immediate reclassify. 'C': classify.
        self.root.bind("r", self._train_and_classify)
        self.root.bind("R", self._train_and_classify)
        self.root.bind("c", self._on_classify_key)
        self.root.bind("C", self._on_classify_key)
        # 'O': optimize the dense network (search + install + classify).
        self.root.bind("o", self._on_optimize_key)
        self.root.bind("O", self._on_optimize_key)
        # 'N': neighbours on/off -- flip between an edge kind and its base.
        self.root.bind("n", self._on_edge_key)
        self.root.bind("N", self._on_edge_key)

    def _on_edge_key(self, _e=None):
        """N: toggle neighbour voting by flipping between the current edge kind
        and its base (the cached predictions re-vote in place)."""
        if self._typing():
            return
        if self._edge_model is None:
            self.status_var.set("No edge model - pick a '-> edges' kind and Train (R)")
            return
        kind = self.model_kind_var.get()
        if _is_edge_kind(kind):
            self.model_kind_var.set(_base_kind(kind))
            self.status_var.set("Neighbour voting off - base net predictions (N to turn on)")
        else:
            target = _edge_kind_of(kind) or _edge_kind_of(_base_kind(self._clf_kind)) or _CUSTOM_EDGE_KIND
            self.model_kind_var.set(target)
            self.status_var.set("Neighbour voting on (N to turn off)")

    def _on_escape(self, _e=None):
        """Escape abandons a gesture in flight (any tool, preview and all);
        with nothing in flight it disarms the class, as before."""
        tool = self.viewer.tool if self.viewer is not None else None
        if tool is not None and hasattr(tool, "cancel") and tool.cancel():
            return
        self.active_class_var.set(0)

    def _unfocus_entries(self, _e=None):
        """Give the keyboard back to the window. The hotkeys are ignored while
        an entry/combobox has focus (typing "1" into a field must not arm a
        class), and Tk leaves focus on a combobox after a selection -- so the
        option widgets hand it back on selection/Return, and a canvas press
        does too."""
        try:
            self.root.focus_set()
        except tk.TclError:
            pass

    def _on_magic_key(self, _e=None):
        if self._typing():
            return
        self.tool_var.set("magic")

    def _on_blob_key(self, _e=None):
        if self._typing():
            return
        self.tool_var.set("blobber")

    def _typing(self):
        """True while a text-entry widget owns the keyboard focus."""
        try:
            w = self.root.focus_get()
            return w is not None and w.winfo_class() in _TYPING_CLASSES
        except tk.TclError:
            return False

    def _on_undo_key(self, e):
        if self._typing():
            return
        if e.keysym.lower() == "z":
            self._undo()
        else:
            self._redo()

    def _on_tab_toggle(self, _e=None):
        """Tab flips the master overlay switch. The toplevel binding fires
        before the "all"-tag focus traversal, so returning "break" consumes
        the key -- except while typing, where Tab keeps moving focus."""
        if self._typing():
            return None
        self.show_overlay_var.set(not self.show_overlay_var.get())
        self._on_overlay_toggle()
        return "break"

    def _on_class_key(self, e):
        if self._typing():
            return
        try:
            k = int(e.char)
        except (TypeError, ValueError):
            return
        if k == 0:
            self.active_class_var.set(0)
        elif k < self.store.n_classes:
            self.active_class_var.set(k)

    # ------------------------------------------------------------------ #
    # The label panel (third pane)
    # ------------------------------------------------------------------ #
    def _build_label_panel(self):
        # A plain frame, not a ScrollFrame: the class subpanels must DIVIDE the
        # available height between them (each scrolls its own interaction list),
        # which needs the holder to fill the pane rather than grow it.
        self.label_pane = ttk.Frame(self.paned, width=300)
        self.paned.add(self.label_pane, weight=0)
        panel = self.label_pane

        # Two titled halves: what the user DRAWS, and what the model does
        # with it. The classifier half is fixed-height and packs FIRST
        # (side="bottom"), so the annotation half takes every remaining pixel
        # for the class stack -- the one thing here that wants more room.
        ml = ttk.LabelFrame(panel, text="ML Region Classifier")
        ml.pack(side="bottom", fill="x", padx=4, pady=(2, 4))
        ann = ttk.LabelFrame(panel, text="Annotation")
        ann.pack(side="top", fill="both", expand=True, padx=4, pady=(4, 2))

        # -- Annotation ---------------------------------------------------- #
        row = ttk.Frame(ann); row.pack(side="top", fill="x", padx=4, pady=(4, 2))
        ttk.Label(row, text="Classes:").pack(side="left")
        self.n_classes_spin = ttk.Spinbox(row, from_=2, to=MAX_CLASSES,
                                          textvariable=self.n_classes_var,
                                          width=4, state="readonly",
                                          command=self._on_n_classes_change)
        self.n_classes_spin.pack(side="left", padx=4)
        ttk.Label(row, text="(class 0 = no label)").pack(side="left", padx=4)

        row = ttk.Frame(ann); row.pack(side="top", fill="x", padx=4, pady=2)
        ttk.Label(row, text="Tool:").pack(side="left")
        for value, txt in _TOOL_LABELS:
            rb = ttk.Radiobutton(row, text=txt, variable=self.tool_var,
                                 value=value)
            rb.pack(side="left", padx=2)
            if value == "magic":
                attach_tooltip(rb, "Magic fill (key M): press on a region, drag UP "
                                   "to grow over similar neighbours, DOWN to shrink, "
                                   "release to paint, Escape to abandon.")
            elif value == "blobber":
                attach_tooltip(rb, "Blobber (key B): a magic fill in the active "
                                   "class plus its immediately adjacent regions in "
                                   "the ring class (see 'ring' in the Magic row).")

        # Magic-fill options: how regions are compared while the fill grows.
        row = ttk.Frame(ann); row.pack(side="top", fill="x", padx=4, pady=2)
        self.magic_row = row
        ttk.Label(row, text="Magic:").pack(side="left")
        cb = ttk.Combobox(row, textvariable=self.magic_metric_var,
                          values=list(magic_fill.METRICS), state="readonly",
                          width=13)
        cb.pack(side="left", padx=2)
        cb.bind("<<ComboboxSelected>>", self._unfocus_entries)
        attach_tooltip(cb, "mean: |mean difference| over the channels (z-scored)\n"
                           "bhattacharyya: Gaussian overlap from mean and std\n"
                           "cosine: 1 - cosine over EVERY statistic column "
                           "(z-scored, positions excluded; ignores 'on')\n"
                           "proba: total variation between the classifier's class "
                           "probabilities (Classify first; ignores 'on')\n"
                           "barrier: saddle height above the seed (MSC arcs only)\n"
                           "learned: the edge model's p(different) per arc "
                           "(a '-> edges' kind, trained and classified)")
        cb = ttk.Combobox(row, textvariable=self.magic_mode_var,
                          values=list(magic_fill.MODES), state="readonly",
                          width=7)
        cb.pack(side="left", padx=2)
        cb.bind("<<ComboboxSelected>>", self._unfocus_entries)
        attach_tooltip(cb, "anchor: every region is compared with the SEED\n"
                           "chain: each region with the neighbour it grows from")
        ttk.Label(row, text="on").pack(side="left", padx=(4, 0))
        en = ttk.Entry(row, textvariable=self.magic_channels_var, width=12)
        en.pack(side="left", padx=2)
        en.bind("<Return>", self._unfocus_entries)
        attach_tooltip(en, "Comma-separated measurement channels as the statistics "
                           "spec names them (base, blur_s1.5, ...); unknown names "
                           "are ignored, none valid falls back to base. Used by "
                           "mean and bhattacharyya only. Enter to leave the field.")
        ttk.Label(row, text="ring").pack(side="left", padx=(6, 0))
        cb = ttk.Combobox(row, textvariable=self.blob_ring_var,
                          values=list(_RING_CHOICES), state="readonly", width=5)
        cb.pack(side="left", padx=2)
        cb.bind("<<ComboboxSelected>>", self._unfocus_entries)
        attach_tooltip(cb, "Blobber ring class: 'next' = the class after the active "
                           "one (wrapping), or a fixed class id.")

        # Flood feel: the per-hop gain and the drag sensitivity.
        row = ttk.Frame(ann); row.pack(side="top", fill="x", padx=4, pady=2)
        self.magic_row2 = row
        ttk.Label(row, text="hop\u00d7").pack(side="left")
        en = ttk.Entry(row, textvariable=self.magic_gain_var, width=5)
        en.pack(side="left", padx=2)
        en.bind("<Return>", self._unfocus_entries)
        attach_tooltip(en, "Per-hop cost multiplier, 1..2. 1 = pure bottleneck: a "
                           "region joins when every hop on its best path is under t. "
                           "Above 1 the cost also grows with graph distance from the "
                           "seed, so nearby regions win over equally similar far "
                           "ones (t then reads in inflated units).")
        ttk.Label(row, text="drag").pack(side="left", padx=(8, 0))
        en = ttk.Entry(row, textvariable=self.magic_drag_var, width=4)
        en.pack(side="left", padx=2)
        en.bind("<Return>", self._unfocus_entries)
        ttk.Label(row, text="px/region").pack(side="left")
        attach_tooltip(en, "Drag sensitivity: screen pixels of vertical drag per "
                           "region near the start of the drag (1..64); longer drags "
                           "accelerate.")

        # Packed before the class holder (side="bottom") so it lands directly
        # under the class panels, leaving the holder the cavity between.
        row = ttk.Frame(ann); row.pack(side="bottom", fill="x", padx=4, pady=(2, 4))
        ttk.Button(row, text="Save annotations…",
                   command=self._save_annotations).pack(side="left", fill="x",
                                                   expand=True, padx=(0, 2))
        ttk.Button(row, text="Load annotations…",
                   command=self._load_annotations).pack(side="left", fill="x",
                                                   expand=True, padx=(2, 0))

        self.classes_holder = ttk.Frame(ann)
        self.classes_holder.pack(side="top", fill="both", expand=True,
                                 padx=2, pady=4)
        self.classes_holder.columnconfigure(0, weight=1)
        self._class_panels = {}          # class_id -> LabelFrame (drop targets)

        # -- ML Region Classifier ------------------------------------------ #
        # Fixed height, so these read top-to-bottom in code order: pick a model
        # and train it, see what it did, then export the result or save it.
        # The model KIND is chosen on the Model tab; this panel trains,
        # applies and exports whatever kind is selected there. The hint above
        # the buttons says which kind / model that is and links to the tab.
        self._build_model_hint(ml)
        row = ttk.Frame(ml); row.pack(side="top", fill="x", padx=4, pady=(4, 2))
        ttk.Button(row, text="Train (R)",
                   command=self._train_classifier).pack(side="left", fill="x",
                                                        expand=True, padx=(0, 2))
        self.classify_btn = ttk.Button(row, text="Classify (C)",
                                       state="disabled", command=self._classify)
        self.classify_btn.pack(side="left", fill="x", expand=True, padx=2)

        # The edge model (the "-> edges" kinds): what it scored, whether
        # voting is on, how many regions it flipped on this slice.
        row = ttk.Frame(ml); row.pack(side="top", fill="x", padx=4)
        self.edge_readout = ttk.Label(row, textvariable=self.edge_readout_var,
                                      foreground="#555", anchor="w", justify="left",
                                      wraplength=290)
        self.edge_readout.pack(side="left", fill="x", expand=True)

        row = ttk.Frame(ml); row.pack(side="top", fill="x", padx=4)
        self.model_strip = ttk.Label(row, textvariable=self.model_strip_var,
                                     foreground="#555", anchor="w")
        self.model_strip.pack(side="left", fill="x", expand=True)

        self.confusion_holder = ttk.LabelFrame(ml, text="true \\ predicted")
        self.confusion_holder.pack(side="top", fill="x", padx=4, pady=(2, 4))
        attach_tooltip(self.confusion_holder,
                       "The CURRENT model's predictions against your annotations "
                       "(rows = annotated class, columns = predicted). The model was "
                       "trained on these same labels, so this is a fit check, not a "
                       "held-out score -- Evaluate edges and Optimize report held-out "
                       "numbers, which are higher. Click a cell to highlight its regions "
                       "on this slice and list them on the Analysis tab; double-click to "
                       "open that list.")
        self._cm_cells = {}              # scope -> {(true, pred): Label}
        self._cm_counts = {}             # scope -> {(true, pred): int}
        self._rebuild_confusion_grid()

        # Shortened from "Make image training set" / "Export as CSV" so both
        # fit on one 300 px row.
        row = ttk.Frame(ml); row.pack(side="top", fill="x", padx=4, pady=2)
        ttk.Button(row, text="Image training set…",
                   command=self._make_training_set).pack(side="left", fill="x",
                                                         expand=True, padx=(0, 2))
        ttk.Button(row, text="Export CSV…",
                   command=self._export_csv).pack(side="left", fill="x",
                                                  expand=True, padx=(2, 0))

        row = ttk.Frame(ml); row.pack(side="top", fill="x", padx=4, pady=(2, 4))
        ttk.Button(row, text="Save classifier…",
                   command=self._save_classifier).pack(side="left", fill="x",
                                                       expand=True, padx=(0, 2))
        ttk.Button(row, text="Load classifier…",
                   command=self._load_classifier).pack(side="left", fill="x",
                                                       expand=True, padx=(2, 0))

        self._rebuild_class_panels()
        self._refresh_model_strip()

    def _on_n_classes_change(self):
        try:
            n = int(self.n_classes_var.get())
        except (tk.TclError, ValueError):
            return
        if n == self.store.n_classes:
            return
        self._push_history()
        changed = self.store.set_n_classes(n)
        if self.active_class_var.get() >= n:
            self.active_class_var.set(0)
        if changed:
            self.status_var.set(f"{len(changed)} interaction(s) moved to class {n - 1}")
        self._rebuild_class_panels()
        self._refresh_render()

    def _rebuild_class_panels(self):
        """Full repaint from the store (interaction counts are small). Also
        repaints the sequence tree, whose "annot" column counts per-slice
        interactions.

        The subpanels split the holder's height equally (uniform grid rows);
        each one scrolls its own interaction list, so a class with many
        gestures never pushes the others off screen."""
        for w in list(self.classes_holder.winfo_children()):
            w.destroy()
        self._class_panels = {}
        visible = self._visible_interactions()
        self._class_title_labels = {}
        self._class_swatches = {}
        for k in range(1, self.store.n_classes):
            frame = ttk.LabelFrame(self.classes_holder)
            self.classes_holder.rowconfigure(k - 1, weight=1, uniform="cls")
            frame.grid(row=k - 1, column=0, sticky="nsew", padx=2, pady=1)
            self._class_panels[k] = frame
            if k == 1:
                self._panel_relief = str(frame.cget("relief"))
            color = self._class_color_hex(k)
            # Title bar: the counts label plus a color swatch that doubles as
            # the ARM control -- left-click draws with this class, right-click
            # recolors it. A full-width "draw (key k)" radiobutton per class
            # cost more vertical space than the interaction lists themselves.
            title = ttk.Frame(frame)
            swatch = tk.Button(title, width=2, bg=color, activebackground=color,
                               relief="raised", bd=1, cursor="hand2")
            swatch.bind("<ButtonRelease-1>",
                        lambda e, k=k: self.active_class_var.set(k))
            swatch.bind("<Button-3>", lambda e, k=k: self._pick_class_color(k))
            attach_tooltip(swatch, f"Left-click: draw with class {k} (key {k})\n"
                                   f"Right-click: change color")
            swatch.pack(side="left", padx=(0, 4))
            self._class_swatches[k] = swatch
            self._class_title_labels[k] = ttk.Label(title, text=f"Class {k}")
            self._class_title_labels[k].pack(side="left")
            frame.configure(labelwidget=title)
            lst = ScrollFrame(frame, width=240, canvas_width=224,
                              background="white")
            # Small minimum height: the grid's equal weights own the real size.
            lst.canvas.configure(height=48)
            lst.pack(side="top", fill="both", expand=True, padx=2, pady=(0, 2))
            for it in visible:
                if it.class_id == k:
                    self._build_interaction_row(lst.inner, it)
        # Rows past the last class keep their weight otherwise, so a later
        # regrow would hand a stale row a share of the height.
        for k in range(self.store.n_classes - 1, MAX_CLASSES):
            self.classes_holder.rowconfigure(k, weight=0, uniform="")
        self._update_class_titles()
        self._refresh_class_arm()
        # The matrix tracks exactly what the panels do -- class count, class
        # colors, and the label edit that triggered this -- so it rides the
        # same rebuild instead of being wired into all ten mutators. It reads
        # the caches _update_class_titles has just warmed, so it is nearly free
        # here; refreshing it earlier would pay for the rasterization twice.
        self._rebuild_confusion_grid()
        self._refresh_confusion()
        # Same reasoning: the persistent outlines are exactly the on-slice
        # interactions the panels list, so they repaint on the same signal.
        self._refresh_annotation_layer()
        self._refresh_subseq_list()      # keep the tree's annot counts live

    def _refresh_class_arm(self, *_trace):
        """Ring the armed class's swatch. Driven by a trace on
        active_class_var, which six paths write (swatch, digit hotkeys,
        Escape, store install, class-count clamp) -- and the panels are
        destroyed and rebuilt under it, hence the winfo_exists guard."""
        active = 0
        try:
            active = int(self.active_class_var.get())
        except (tk.TclError, ValueError):
            pass
        for k, w in getattr(self, "_class_swatches", {}).items():
            try:
                if not w.winfo_exists():
                    continue
                w.configure(relief="sunken" if k == active else "raised",
                            bd=3 if k == active else 1)
            except tk.TclError:
                continue

    # ------------------------------------------------------------------ #
    # Confusion matrix (frozen predictions vs. live labels)
    # ------------------------------------------------------------------ #
    def _rebuild_confusion_grid(self):
        """Lay out synchronized all-slice and current-slice matrix grids."""
        holder = getattr(self, "confusion_holder", None)
        if holder is None:
            return
        for w in list(holder.winfo_children()):
            w.destroy()
        self._cm_cells = {}
        n = self.store.n_classes
        for scope, title in (("all", "All slices"), ("current", "Current slice")):
            frame = ttk.LabelFrame(holder, text=title)
            frame.pack(fill="x", padx=2, pady=1)
            cells = {}
            self._cm_cells[scope] = cells
            tk.Label(frame, text="t\\p", width=3, background="white",
                     relief="flat").grid(row=0, column=0, sticky="nsew")
            for j in range(1, n):
                tk.Label(frame, text=str(j), width=4,
                         background=self._class_color_hex(j),
                         relief="flat").grid(row=0, column=j, sticky="nsew")
            for i in range(1, n):
                tk.Label(frame, text=str(i), width=3,
                         background=self._class_color_hex(i),
                         relief="flat").grid(row=i, column=0, sticky="nsew")
                for j in range(1, n):
                    cell = tk.Label(frame, text="0", width=4, background="white",
                                    relief="ridge", borderwidth=1, cursor="hand2")
                    cell.grid(row=i, column=j, sticky="nsew", padx=1, pady=1)
                    cell.bind("<Button-1>",
                              lambda e, i=i, j=j: self._on_confusion_click(i, j))
                    cell.bind("<Double-Button-1>",
                              lambda e, i=i, j=j: self._on_confusion_open(i, j))
                    cells[(i, j)] = cell
            for c in range(n):
                frame.columnconfigure(c, weight=1)
        if self._cm_cell not in self._cm_cells.get("all", {}):
            self._cm_cell = None         # the class count shrank under it

    def _confusion_counts(self, scope="all"):
        """{(true, pred): n} globally or on the current slice.

        Predictions stay frozen until the next Train/Classify, so labeling
        more moves only the true axis -- "old prediction, new value".

        Truth comes from _labels_cache_for, the rasterization the class LUT
        already paid for, so this adds no pass over the interactions."""
        import numpy as np
        counts = {}
        current = self._current() if scope == "current" else None
        items = ([(current, self._pred.get(current))] if current is not None
                 else []) if scope == "current" else self._pred.items()
        for (si, li), entry in items:
            if entry is None:
                continue
            rec = self.engine.record(si, li)
            if rec is None or rec.get("labels") is None:
                continue
            if entry[0] != rec.get("commit"):
                continue
            truth = self._truth_from_cache(si, li, rec, np)
            if truth is None:
                continue
            pred = entry[1]
            n = min(len(truth), len(pred))
            t, p = truth[:n], pred[:n]
            m = (t >= 1) & (p >= 1)
            if not m.any():
                continue
            pairs, freq = np.unique(np.stack([t[m], p[m]], 1), axis=0,
                                    return_counts=True)
            for (ti, pj), c in zip(pairs, freq):
                key = (int(ti), int(pj))
                counts[key] = counts.get(key, 0) + int(c)
        return counts

    def _truth_from_cache(self, si, li, rec, np):
        """Region id -> user class for one slice, straight out of the memoized
        rasterization the class LUT already pays for."""
        return self._labels_cache_for(si, li, rec, np)[5]

    def _refresh_confusion(self):
        self._cm_counts = {
            "all": self._confusion_counts("all"),
            "current": self._confusion_counts("current"),
        }
        for scope, cells in getattr(self, "_cm_cells", {}).items():
            counts = self._cm_counts.get(scope, {})
            for (i, j), cell in cells.items():
                try:
                    cell.configure(
                        text=str(counts.get((i, j), 0)),
                        background=("#cde8ff" if self._cm_cell == (i, j)
                                    else ("#f0f0f0" if i == j else "white")))
                except tk.TclError:
                    pass

    def _on_confusion_click(self, i, j):
        """Highlight the cell's regions on the CURRENT slice (clicking the
        selected cell again clears it)."""
        self._cm_cell = None if self._cm_cell == (i, j) else (i, j)
        self._refresh_confusion()
        self._refresh_render()
        self._fill_error_list(self._cm_cell)
        if self._cm_cell is None:
            self.status_var.set("Confusion highlight cleared.")
            return
        current = self._cm_counts.get("current", {}).get((i, j), 0)
        total = self._cm_counts.get("all", {}).get((i, j), 0)
        self.status_var.set(f"annotated {i} -> predicted {j}: "
                            f"{current} on this slice / {total} total - listed on the "
                            "Analysis tab (double-click the cell to open it)")

    def _confusion_hits(self):
        """Region ids on the current slice matching the selected cell."""
        if self._cm_cell is None:
            return set()
        cur = self._current()
        if cur is None:
            return set()
        si, li = cur
        rec = self.engine.record(si, li)
        entry = self._pred.get((si, li))
        if (rec is None or rec.get("labels") is None or entry is None
                or entry[0] != rec.get("commit")):
            return set()
        import numpy as np
        truth = self._truth_from_cache(si, li, rec, np)
        if truth is None:
            return set()
        i, j = self._cm_cell
        pred = entry[1]
        n = min(len(truth), len(pred))
        hit = (truth[:n] == i) & (pred[:n] == j)
        return set(int(r) for r in np.nonzero(hit)[0])

    def _visible_interactions(self):
        """The interactions listed in the class panels: the CURRENT slice's
        only (they swap with every slice change); everything when no slice is
        on screen (nothing primed yet, or a freshly loaded session)."""
        cur = self._current()
        if cur is None:
            return list(self.store.interactions)
        key = self._slice_key(*cur)
        return [it for it in self.store.interactions if it.slice_key == key]

    def _class_totals(self):
        """ALL-slice totals per class: (annotations, labeled regions). Region
        counts come from the per-slice resolution caches, so only slices that
        carry interactions AND a computed record contribute (the rest have
        nothing to count yet)."""
        import numpy as np
        annot = {}
        slices = {}                      # slice_key -> (si, li), bound only
        for it in self.store.interactions:
            annot[it.class_id] = annot.get(it.class_id, 0) + 1
            if it.bound:
                slices.setdefault(it.slice_key, (it.si, it.li))
        regions = {}
        for _key, (si, li) in slices.items():
            rec = self.engine.record(si, li)
            if rec is None or rec.get("labels") is None:
                continue
            counts = self._labels_cache_for(si, li, rec, np)[4]
            for k in range(1, self.store.n_classes):
                if k < len(counts) and counts[k]:
                    regions[k] = regions.get(k, 0) + int(counts[k])
        return annot, regions

    def _update_class_titles(self):
        annot, regions = self._class_totals()
        for k, lbl in getattr(self, "_class_title_labels", {}).items():
            try:
                # Terse: the full "Class k — annot: n — regions: m" overflowed
                # the 300 px pane. a = annotations (all slices), r = regions.
                lbl.configure(text=f"{k} · {annot.get(k, 0)}a · "
                                   f"{regions.get(k, 0)}r")
            except tk.TclError:
                pass

    def _pick_class_color(self, k):
        from tkinter import colorchooser
        hexv = colorchooser.askcolor(color=self._class_color_hex(k),
                                     title=f"Class {k} color",
                                     parent=self.root)[1]
        if not hexv:
            return
        self._push_history()             # a color change is undoable
        self.store.set_color(k, hexv)    # rev bump -> LUT caches rebuild
        self._rebuild_class_panels()
        self._refresh_render()

    def _build_interaction_row(self, parent, it):
        # Plain-tk widgets so the rows share the list's white background.
        row = tk.Frame(parent, background="white")
        row.pack(fill="x", padx=4, pady=0)
        # Delete on the LEFT: a narrow pane truncates the label, and a
        # right-packed ✕ is the first thing to disappear with it.
        tk.Button(row, text="✕", width=2, relief="flat", background="white",
                  activebackground="#ddd", padx=0, pady=0,
                  command=lambda uid=it.uid: self._delete_interaction(uid)
                  ).pack(side="left")
        # The rows normally all belong to the current slice, so the key is
        # noise -- except when it is NOT this slice's (nothing on screen, so
        # _visible_interactions lists everything) or failed to bind at all.
        cur = self._current()
        cur_key = self._slice_key(*cur) if cur is not None else None
        if not it.bound:
            where = f"  [{it.slice_key} (unbound)]"
        elif it.slice_key != cur_key:
            where = f"  [{it.slice_key}]"
        else:
            where = ""
        name = it.tool
        if it.meta and it.meta.get("tool"):
            name = str(it.meta["tool"])          # e.g. a magic fill's taps
            if it.meta.get("part"):
                name += f" {it.meta['part']}"    # blobber: core / ring
            if it.meta.get("n_regions") is not None:
                name += f" ({it.meta['n_regions']})"
        lbl = tk.Label(row, text=f"#{it.uid} {name}{where}",
                       foreground=("#000" if it.bound else "#888"),
                       background="white", anchor="w", cursor="hand2")
        lbl.pack(side="left", fill="x", expand=True)
        # Drag a row onto another class's subpanel to reassign it; a stationary
        # release is a CLICK, which recenters the view on the gesture.
        # Right-click offers move/delete as a menu (drag fallback, faster in
        # bulk); hovering shows the gesture's geometry on the canvas.
        lbl.bind("<ButtonPress-1>", lambda e, uid=it.uid: self._row_drag_start(e, uid))
        lbl.bind("<B1-Motion>", self._row_drag_motion)
        lbl.bind("<ButtonRelease-1>", self._row_drag_drop)
        lbl.bind("<Button-3>", lambda e, uid=it.uid: self._row_menu(e, uid))
        lbl.bind("<Enter>", lambda e, uid=it.uid: self._show_interaction_geometry(uid))
        lbl.bind("<Leave>", lambda e: self._hide_interaction_geometry())

    # -- row drag-and-drop between class subpanels ---------------------- #
    def _panel_under_pointer(self, e):
        w = self.root.winfo_containing(e.x_root, e.y_root)
        while w is not None:
            for k, frame in self._class_panels.items():
                if w is frame:
                    return k, frame
            w = getattr(w, "master", None)
        return None, None

    def _row_drag_start(self, e, uid):
        self._drag_uid = uid
        self._drag_origin = (e.x_root, e.y_root)

    def _row_drag_motion(self, e):
        if self._drag_uid is None:
            return
        _k, frame = self._panel_under_pointer(e)
        if frame is not self._drop_panel:
            if self._drop_panel is not None:
                try:
                    self._drop_panel.configure(relief=self._panel_relief)
                except tk.TclError:
                    pass
            self._drop_panel = frame
            if frame is not None:
                frame.configure(relief="ridge")

    def _row_drag_drop(self, e):
        uid, self._drag_uid = self._drag_uid, None
        origin, self._drag_origin = self._drag_origin, None
        if self._drop_panel is not None:
            try:
                self._drop_panel.configure(relief=self._panel_relief)
            except tk.TclError:
                pass
        self._drop_panel = None
        if uid is None:
            return
        # A release that never really moved is a click, not a drop.
        if origin is not None and (abs(e.x_root - origin[0]) +
                                   abs(e.y_root - origin[1])) < 5:
            self._on_row_click(uid)
            return
        k, _frame = self._panel_under_pointer(e)
        it = self.store.get(uid)
        if k is not None and it is not None and k != it.class_id:
            self._move_interaction(uid, k)

    def _interaction_menu(self, e, uid):
        """The move/delete menu for one interaction. Single source: the class
        list rows and the canvas right-click must offer the same thing."""
        it = self.store.get(uid)
        if it is None:
            return
        menu = tk.Menu(self.root, tearoff=0)
        for k in range(1, self.store.n_classes):
            if k != it.class_id:
                menu.add_command(label=f"Move to class {k}",
                                 command=lambda k=k: self._move_interaction(uid, k))
        menu.add_separator()
        menu.add_command(label="Delete", command=lambda: self._delete_interaction(uid))
        try:
            menu.tk_popup(e.x_root, e.y_root)
        finally:
            menu.grab_release()

    def _row_menu(self, e, uid):
        self._interaction_menu(e, uid)

    def _interaction_at(self, ix, iy):
        """The uid of the interaction "under" an image pixel: the last-drawn
        one whose gesture touched that pixel's region -- the same resolution
        order the class layer paints in, so the menu acts on the gesture the
        user can actually see there. Falls back to a row-hovered gesture."""
        cur = self._current()
        if cur is None or ix is None:
            return self._hover_uid
        si, li = cur
        rec = self.engine.record(si, li)
        if rec is None or rec.get("labels") is None:
            return self._hover_uid
        labels = rec["labels"]
        if not (0 <= iy < labels.shape[0] and 0 <= ix < labels.shape[1]):
            return self._hover_uid
        region = int(labels[iy, ix])
        if region < 0:
            return self._hover_uid
        import numpy as np
        touch = self._touch_map_for(si, li, rec, np)
        best = None
        for it in self.store.for_slice(self._slice_key(si, li)):
            ids = touch.get(it.uid)
            if ids and region in ids and (best is None or it.uid > best):
                best = it.uid
        return best if best is not None else self._hover_uid

    def _canvas_menu(self, e):
        """Right-CLICK on the image plane (a right-DRAG still pans): offer the
        row menu for whatever annotation is under the pointer, and nothing at
        all when there is none -- an empty popup would just be in the way."""
        if self.viewer is None:
            return
        ix, iy = self.viewer.screen_to_image(e.x, e.y)
        uid = self._interaction_at(ix, iy)
        if uid is not None:
            self._interaction_menu(e, uid)

    # -- interaction geometry on the canvas (hover + click-to-center) ---- #
    def _draw_interaction_geometry(self, it, tags=("draw", "ihover")):
        """Draw one gesture's geometry fully opaque over the slice: the
        polyline itself, or the outer boundary of a box / lasso. Does NOT
        clear first, so several touching gestures can stack.

        `tags` selects the layer: "ihover" is transient (every repaint of the
        overlays drops it) while "ipersist" survives, since the persistent
        annotation view is not tied to where the pointer happens to be."""
        v = self.viewer
        if it is None or v is None or not it.points or not it.bound:
            return
        if self._current() != (it.si, it.li):
            return
        c = v.canvas
        color = self._class_color_hex(it.class_id)
        scr = [((x - v.view_x) / v.scale, (y - v.view_y) / v.scale)
               for x, y in it.points]
        # "draw" (in every tag tuple) keeps the item above each fresh blit.
        if it.tool == "box" and len(scr) >= 2:
            (x0, y0), (x1, y1) = scr[0], scr[-1]
            c.create_rectangle(x0, y0, x1, y1, outline=color, width=3, tags=tags)
        elif it.tool == "taps":          # independent sample points
            for x, y in scr:
                c.create_oval(x - 4, y - 4, x + 4, y + 4, outline=color,
                              width=3, tags=tags)
            seed = (it.meta or {}).get("seed")
            if seed and len(seed) == 2:  # a magic fill: mark where it started
                sx = (float(seed[0]) - v.view_x) / v.scale
                sy = (float(seed[1]) - v.view_y) / v.scale
                c.create_oval(sx - 9, sy - 9, sx + 9, sy + 9, outline=color,
                              width=2, dash=(3, 2), tags=tags)
        elif len(scr) >= 2:
            pts = scr + [scr[0]] if it.tool == "polygon" else scr
            flat = [coord for pt in pts for coord in pt]
            c.create_line(*flat, fill=color, width=3, tags=tags)
        else:                            # a single-tap squiggle
            x, y = scr[0]
            c.create_oval(x - 4, y - 4, x + 4, y + 4, outline=color, width=3,
                          tags=tags)

    def _show_interaction_geometry(self, uid):
        if self.viewer is None:
            return
        self.viewer.canvas.delete("ihover")
        self._hover_key = None
        self._hover_uid = uid
        self._draw_interaction_geometry(self.store.get(uid))

    def _hide_interaction_geometry(self):
        if self.viewer is not None:
            self.viewer.canvas.delete("ihover")
        self._hover_key = None
        self._hover_uid = None

    def _refresh_annotation_layer(self):
        """Repaint the persistent outlines: every ON-SLICE gesture at once,
        instead of only what the pointer is over. Separate tag from the hover
        layer, which every overlay repaint drops."""
        v = self.viewer
        if v is None:
            return
        v.canvas.delete("ipersist")
        if not self.show_overlay_var.get() or not self.show_annot_var.get():
            return
        for it in self._visible_interactions():
            self._draw_interaction_geometry(it, tags=("draw", "ipersist"))

    def _redraw_hover_geometry(self):
        """Re-project whatever hover geometry is on screen after a zoom/pan
        (the items are drawn in screen coordinates)."""
        self._refresh_annotation_layer()
        v = self.viewer
        if v is None or (self._hover_uid is None and self._hover_key is None):
            return
        v.canvas.delete("ihover")
        if self._hover_uid is not None:
            self._draw_interaction_geometry(self.store.get(self._hover_uid))
            return
        si, li, region = self._hover_key
        rec = self.engine.record(si, li)
        if rec is None or rec.get("labels") is None:
            return
        import numpy as np
        touch = self._touch_map_for(si, li, rec, np)
        for it in self.store.for_slice(self._slice_key(si, li)):
            ids = touch.get(it.uid)
            if ids and region in ids:
                self._draw_interaction_geometry(it)

    def _on_hover(self, ix, iy=None):
        """Show image values/probabilities and highlight annotations for a region."""
        super()._on_hover(ix, iy)
        v = self.viewer
        if v is None:
            return
        cur = self._current()
        region = None
        rec = None
        if ix is not None and iy is not None and cur is not None:
            rec = self.engine.record(*cur)
            if rec is not None and rec.get("labels") is not None:
                labels = rec["labels"]
                if 0 <= iy < labels.shape[0] and 0 <= ix < labels.shape[1]:
                    r = int(labels[iy, ix])
                    if r >= 0:
                        region = r
                    ctx = self._hover_ctx
                    if ctx is not None:
                        base = ctx["base"]
                        filt = ctx["filt"]
                        probabilities = "-"
                        pred = self._pred.get(cur)
                        if (region is not None and pred is not None
                                and pred[0] == rec.get("commit")
                                and region < pred[2].shape[0]):
                            proba = pred[2][region]
                            if float(proba.sum()) > 0.0:
                                probabilities = " ".join(
                                    f"P(class {k})={float(proba[k]):.3f}"
                                    for k in range(1, self.store.n_classes))
                        self.hover_var.set(
                            f"x={ix} y={iy}  |  base={float(base[iy, ix]):.4g} "
                            f"filtered={float(filt[iy, ix]):.4g}  |  "
                            f"class probabilities: {probabilities}")
        if self._hover_suppressed:
            return                        # a gesture is previewing: no outlines
        key = None if region is None else (cur[0], cur[1], region)
        if key == self._hover_key:
            return
        self._hover_key = key
        self._hover_uid = None            # canvas hover replaces row hover
        v.canvas.delete("ihover")
        if key is None:
            return
        import numpy as np
        touch = self._touch_map_for(cur[0], cur[1], rec, np)
        for it in self.store.for_slice(self._slice_key(*cur)):
            ids = touch.get(it.uid)
            if ids and region in ids:
                self._draw_interaction_geometry(it)

    def _on_row_click(self, uid):
        """Center the view on the gesture's centroid at the current zoom,
        navigating to its slice first when it lives on another one."""
        it = self.store.get(uid)
        v = self.viewer
        if it is None or v is None or not it.points or not it.bound:
            return
        if self._current() != (it.si, it.li):
            try:
                idx = self.flat_slices.index((it.si, it.li))
            except ValueError:
                return
            self._goto_slice(idx)
        cx = sum(x for x, _y in it.points) / len(it.points)
        cy = sum(y for _x, y in it.points) / len(it.points)
        w = max(v.canvas.winfo_width(), 1)
        h = max(v.canvas.winfo_height(), 1)
        v.view_x = cx - (w / 2) * v.scale     # zoom (v.scale) stays as-is
        v.view_y = cy - (h / 2) * v.scale
        v.render()
        self._show_interaction_geometry(uid)  # redraw at the new view

    # ------------------------------------------------------------------ #
    # Persistence: annotations.json + session
    #
    # The document is the raw gesture geometry (tool, points, class, slice) --
    # annotations, not the MSC label raster that `rec["labels"]` means
    # everywhere else. It was called labels.json before that ambiguity bit;
    # the CONTENT is unchanged, so an old labels.json still loads here.
    # ------------------------------------------------------------------ #
    def _save_annotations(self):
        path = filedialog.asksaveasfilename(title="Save annotations.json",
                                            defaultextension=".json",
                                            initialfile="annotations.json",
                                            filetypes=[("JSON", "*.json")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.store.to_json(), f, indent=2)
        self.status_var.set(f"Wrote {path}")

    def _load_annotations(self):
        path = filedialog.askopenfilename(title="Load annotations.json",
                                          filetypes=[("JSON", "*.json")])
        if not path:
            return
        doc = config_io.read_json_file(path)
        if doc is None:
            self.status_var.set(f"Could not read {path}")
            return
        try:
            store = LabelStore.from_json(doc)
        except Exception as exc:
            self.status_var.set(f"Not an annotations.json: {exc}")
            return
        self._install_store(store)

    # ------------------------------------------------------------------ #
    # Classifier: train on the labeled regions, predict every region
    # ------------------------------------------------------------------ #
    def _all_stat_slices(self, action):
        """Materialize and return current records for every primed slice.

        Model operations are stack-wide and must never silently degrade to the
        subset visited by the lazy viewer. None means the operation must abort.
        """
        if not self.primed:
            self.status_var.set("Run first - no slices are primed.")
            return None
        if self.engine.pending_work():
            self.status_var.set("Busy computing - try again in a moment.")
            return None
        total = sum(len(p["pipes"]) for p in self.primed)
        out = []
        self._compute_badge(f"{action} 0/{total}")
        try:
            for si, p in enumerate(self.primed):
                for li in range(len(p["pipes"])):
                    self._compute_badge(f"{action} {len(out) + 1}/{total}")
                    rec = self._ensure_slice_record(si, li)
                    key = self._slice_key(si, li)
                    table = None if rec is None else rec.get("stats")
                    if (rec is None or rec.get("labels") is None or key is None
                            or getattr(table, "values", None) is None):
                        self.status_var.set(
                            f"{action} stopped: could not compute slice {si}:{li}.")
                        return None
                    out.append((si, li, key, rec, table))
        finally:
            self._clear_compute_badge()
        return out

    def _iter_stat_slices(self):
        """Cached current records only; callers needing completeness use
        _all_stat_slices()."""
        for si, p in enumerate(self.primed):
            for li in range(len(p["pipes"])):
                rec = self.engine.record(si, li)
                key = self._slice_key(si, li)
                if rec is None or rec.get("labels") is None or key is None:
                    continue
                table = rec.get("stats")
                if getattr(table, "values", None) is None:
                    continue
                yield si, li, key, rec, table

    @staticmethod
    def _feature_matrix(table, names, np):
        """(n_regions, len(names)) float64, or None if a column is missing."""
        cols = [table.column(n) for n in names]
        if any(c is None for c in cols):
            return None
        mat = np.stack(cols, axis=1).astype(np.float64)
        mat[~np.isfinite(mat)] = 0.0
        return mat

    def _train_classifier(self, preserve_view=False):
        """Fit the selected model kind on the labeled regions' statistics rows.

        Random forest is the default for exactly this data shape: a few
        hundred labels, tens of features of which only a handful matter (trees
        select thresholds per feature, so the noise dimensions that dragged
        k-means across the label boundary are simply never split on), no
        scaling sensitivity, and millisecond retrains -- with
        feature_importances_ naming the dimensions that matter (logged).
        "dense FC" is a small MLP behind an in-pipeline StandardScaler for
        when the boundary is not axis-aligned. The dense-top-N variants first
        fit a balanced forest and retain its N most important dimensions.
        "dense (tuned)" rebuilds the last Optimize winner (or the baseline
        while there is none) with balanced sample weights."""
        got = self._training_set()
        if got is None:
            return
        X, y, _groups, names = got
        kind = self.model_kind_var.get()
        base = _base_kind(kind)
        spec = self._search_spec if base == _TUNED_KIND else None
        build_spec = spec
        if base == _TUNED_KIND and spec is None:     # baseline on the picked backend
            build_spec = model_search.ModelSpec(max_iter=_MLP_MAX_ITER,
                                                backend=self._search_settings()[4])
        if base == _CUSTOM_KIND:
            spec = build_spec = model_search.ModelSpec(
                hidden=tuple(self._custom_hidden()), max_iter=_MLP_MAX_ITER,
                backend=self._search_settings()[4])
        # A frozen base: an edge kind keeps the current base net (same
        # features, has an embedding) and refits only the edges on top.
        frozen = (_is_edge_kind(kind) and bool(self.freeze_base_var.get())
                  and self._clf is not None and list(self._clf_names or []) == list(names)
                  and self._has_embedding(self._clf))
        t0 = time.perf_counter()
        if frozen:
            clf = self._clf
            spec = self._clf_spec
        else:
            clf = self._make_model(base, len(y), len(names), names=names, spec=build_spec)
            self._compute_badge("Training")
            try:
                if base in (_TUNED_KIND, _CUSTOM_KIND):
                    model_search.fit_estimator(clf, X, y)
                else:
                    clf.fit(X, y)
            finally:
                self._clear_compute_badge()
        dt_ms = 1e3 * (time.perf_counter() - t0)
        edge, edge_note = (None, "")
        if _is_edge_kind(kind):
            edge, edge_note = self._fit_edge_model_now(clf, names)
        self._install_model(clf, names, kind, spec, preserve_view, edge=edge)
        if base in (_TUNED_KIND, _CUSTOM_KIND):
            acc = f", train acc {clf.score(X, y):.1%}"
            hint = "" if spec is None else " - " + spec.brief(len(names))
            if frozen:
                hint += " (base frozen)"
        elif kind in _DENSE_TOP_N:
            selector = clf.named_steps["select"]
            selected = [(names[i], float(v)) for i, v in
                        enumerate(selector.estimator_.feature_importances_)
                        if selector.get_support()[i]]
            selected.sort(key=lambda t: -t[1])
            log(f"classifier trained: selected {len(selected)}/{len(names)} "
                "features: " + "  ".join(f"{n}={v:.3f}"
                                          for n, v in selected))
            acc = f", train acc {clf.score(X, y):.1%}"
            hint = " - selected: " + ", ".join(n for n, _v in selected[:3])
        elif hasattr(clf, "feature_importances_"):
            top = sorted(zip(names, clf.feature_importances_),
                         key=lambda t: -t[1])[:8]
            log("classifier trained: " + "  ".join(f"{n}={v:.3f}" for n, v in top))
            hint = " - top: " + ", ".join(n for n, _v in top[:3])
            acc = (f", OOB acc {clf.oob_score_:.1%}"
                   if getattr(clf, "oob_score", False) else "")
        else:                                # dense FC: no per-feature story
            acc = f", train acc {clf.score(X, y):.1%}"
            hint = ""
        self._refresh_model_strip()
        self.status_var.set(f"Trained {kind} on {len(y)} labeled regions in "
                            f"{dt_ms:.0f} ms{acc}{hint}{edge_note}")

    def _training_set(self):
        """``(X, y, groups, names)`` over every labeled region of every primed
        slice -- `groups` is each row's slice index, for leave-slices-out CV --
        or None with the reason in the status bar."""
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.status_var.set("scikit-learn is not installed - "
                                "pip install scikit-learn to enable training")
            return None
        import numpy as np
        slices = self._all_stat_slices("Preparing training")
        if slices is None:
            return None
        X, y, g, names = [], [], [], None
        for si, li, key, rec, table in slices:
            if names is None:
                names = [n for n in table.names if n not in _NON_FEATURE_FIELDS]
            mat = self._feature_matrix(table, names, np)
            fids = table.column("feature_id")
            if mat is None or fids is None:
                self.status_var.set(
                    f"Training stopped: incomplete statistics on slice {si}:{li}.")
                return None
            rc = resolve_slice(self.store.for_slice(key), rec["labels"], np)
            fid = fids.astype(int)
            ok = (fid >= 0) & (fid < len(rc))
            cls = np.zeros(len(fid), int)
            cls[ok] = rc[fid[ok]]
            m = cls > 0
            if m.any():
                X.append(mat[m])
                y.append(cls[m])
                g.append(np.full(int(m.sum()), si, int))
        if not X:
            self.status_var.set("No labeled regions on computed slices - "
                                "draw some (and Run/Rerun) first.")
            return None
        X = np.concatenate(X)
        y = np.concatenate(y)
        g = np.concatenate(g)
        if len(set(y.tolist())) < 2:
            self.status_var.set("Need labels from at least 2 classes to train.")
            return None
        return X, y, g, names

    def _install_model(self, clf, names, kind, spec=None, preserve_view=False, edge=None):
        """Make `clf` the current model (Train and Optimize share this tail).
        `edge` is the edge model fit on top of it (None drops any old one: an
        edge model belongs to exactly one base net)."""
        self._clf = clf
        self._clf_names = names
        self._clf_kind = kind
        self._clf_spec = spec
        self._edge_model = edge
        self._pred.clear()               # predictions belong to the old model
        if not preserve_view:
            self._cm_cell = None
            self._refresh_region_modes()
        self._refresh_confusion()
        self.classify_btn.config(state="normal")
        self._refresh_model_readout()
        self._refresh_edge_readout()

    @staticmethod
    def _has_embedding(clf):
        try:
            edge_model.hidden_widths(clf)
            return True
        except TypeError:
            return False

    def _record_arcs(self, rec, np):
        """The record's living-region arcs (MSC saddles), or pixel adjacency
        derived once and cached on the record, as the magic fill does."""
        arcs = rec.get("arcs")
        if arcs is None and rec.get("labels") is not None:
            arcs = magic_fill.arcs_from_labels(rec["labels"], np)
            rec["arcs"] = arcs
        return arcs

    def _edge_training_data(self, names):
        """Every region of every primed slice (class 0 = unlabeled) with its
        slice index and extremum value, plus the edges of the region graph as
        global row pairs -- what the edge model is fit and evaluated on. Tk
        thread only (records, the store); returns copies the worker may keep.
        None (with the reason in the status bar) when a slice lacks stats."""
        import numpy as np
        slices = self._all_stat_slices("Gathering edges")
        if slices is None:
            return None
        names = list(names)
        ext_col = names.index("ext_filtered") if "ext_filtered" in names else None
        X, cls, grp, ext, per = [], [], [], [], []
        for k, (si, li, key, rec, table) in enumerate(slices):
            mat = self._feature_matrix(table, names, np)
            fids = table.column("feature_id")
            if mat is None or fids is None:
                self.status_var.set(f"Edges stopped: incomplete statistics on slice {si}:{li}.")
                return None
            rc = resolve_slice(self.store.for_slice(key), rec["labels"], np)
            fid = fids.astype(int)
            ok = (fid >= 0) & (fid < len(rc))
            c = np.zeros(len(fid), int)
            c[ok] = rc[fid[ok]]
            X.append(mat); cls.append(c); grp.append(np.full(len(c), k, int))
            if ext_col is not None:
                ext.append(mat[:, ext_col])
            per.append((fid, self._record_arcs(rec, np), c, k))
        edges = edge_model.gather_edges(per)
        return (np.concatenate(X), np.concatenate(cls), np.concatenate(grp),
                np.concatenate(ext) if ext else None, edges, names)

    def _fit_edge_model_now(self, clf, names):
        """Fit the pair model on top of `clf`: (EdgeModel | None, status note)."""
        if not self._has_embedding(clf):
            return None, " - edges skipped: the base has no hidden layer to embed with"
        data = self._edge_training_data(names)
        if data is None:
            return None, " - edges skipped: no data"
        X_all, cls, _grp, ext, edges, _names = data
        spec = self._edge_spec_from_ui()
        self._compute_badge("Fitting edges")
        try:
            edge = edge_model.fit_edge_model(clf, X_all, cls, edges, ext, spec, names)
        except ValueError as exc:
            log(f"edge model not fit: {exc}")
            return None, f" - edges not fit: {exc}"
        finally:
            self._clear_compute_badge()
        log(f"edge model: {edge.describe()}; {int(edges['both'].sum())} labeled pairs of "
            f"{len(edges['a'])} edges, {edge.fit_s * 1e3:.0f} ms")
        return edge, (f" -> edges: {edge.n_edges:,} pairs, "
                      f"{edge.n_diff / max(1, edge.n_edges):.0%} boundaries, "
                      f"{'MSC saddles' if edge.used_saddle else 'pixel adjacency'}, "
                      f"{edge.fit_s * 1e3:.0f} ms")

    def _vote_entry(self, region_proba, aux, lam, rounds, np):
        """(final classes, flips) for one cached slice: neighbour voting in
        LABEL space over the classes that carry any probability mass; ids
        that are not living regions stay class 0."""
        raw = np.asarray(aux["raw"], np.uint8)
        cols = np.nonzero(region_proba.sum(0) > 0)[0]
        k = aux["keep"]
        if len(cols) < 2 or not k.any():
            return raw.copy(), 0
        P = region_proba[:, cols]
        voted = edge_model.vote(P, cols, np.asarray(aux["la"])[k], np.asarray(aux["lb"])[k],
                                np.asarray(aux["pdiff"])[k], lam, rounds)
        living = region_proba.sum(1) > 0
        final = raw.copy()
        final[living] = np.asarray(voted, np.uint8)[living]
        return final, int((final != raw).sum())

    def _revote_all(self):
        """Re-apply (or lift) neighbour voting on every cached slice from the
        cached per-arc p(diff): a kind flip or a lambda / rounds edit costs
        milliseconds, no forward pass. Entries classified before the edge
        model existed have no aux and force a Classify."""
        import numpy as np
        active = self._edge_model is not None and _is_edge_kind(self.model_kind_var.get())
        spec = self._edge_spec_from_ui()
        missing = False
        for key, entry in list(self._pred.items()):
            aux = self._pred_aux(entry)
            if aux is None:
                missing = missing or self._edge_model is not None
                continue
            if active:
                final, flips = self._vote_entry(entry[2], aux, spec.lam, spec.rounds, np)
            else:
                final, flips = np.asarray(aux["raw"], np.uint8).copy(), 0
            aux["lam"], aux["rounds"], aux["flips"], aux["active"] = spec.lam, spec.rounds, flips, active
            self._pred[key] = (entry[0], final, entry[2], aux)
        if missing and self._clf is not None:
            self._pred.clear()
            self._classify()
            return
        self._refresh_region_modes()
        self._refresh_confusion()
        self._refresh_render()
        self._refresh_edge_readout()

    def _refresh_edge_readout(self):
        var = getattr(self, "edge_readout_var", None)
        if var is None:
            return
        edge = self._edge_model
        kind = self.model_kind_var.get()
        if edge is None:
            var.set("edges: not trained - Train (R) on this kind" if _is_edge_kind(kind)
                    else "edges: none - pick a '-> edges' kind and Train (R)")
            return
        text = edge.brief()
        active = _is_edge_kind(kind)
        text += " · voting " + ("on" if active else "off") + " (N)"
        cur = self._current()
        entry = self._pred.get(cur) if cur is not None else None
        aux = self._pred_aux(entry)
        if active and aux is not None and aux.get("flips") is not None:
            text += f" · {aux['flips']} flipped on this slice"
        var.set(text)

    @staticmethod
    def _make_model(kind, n_samples, n_features=None, names=None, spec=None):
        if kind == _CUSTOM_KIND:
            if not names or spec is None:
                raise ValueError("custom FC needs the feature names and a spec")
            return model_search.build_estimator(spec, list(names))
        if kind == _TUNED_KIND:
            # The search winner (or the baseline before any search), built by
            # model_search so the estimator and its description share a spec.
            if not names:
                raise ValueError("dense (tuned) needs the feature names")
            spec = spec or model_search.ModelSpec(max_iter=_MLP_MAX_ITER)
            return model_search.build_estimator(spec, list(names))
        if kind == "dense FC":
            # The scaler lives INSIDE the pipeline: an MLP needs standardized
            # inputs, and keeping it in the estimator means the pickle /
            # predict paths stay identical to the forest's.
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.neural_network import MLPClassifier
            return make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=_MLP_HIDDEN, max_iter=_MLP_MAX_ITER,
                              random_state=0))
        if kind in _DENSE_TOP_N:
            if n_features is None or n_features < 1:
                raise ValueError("dense-top-N requires at least one feature")
            # Keep selection inside the estimator: the model continues to
            # consume the full profile schema, while its fitted forest mask is
            # applied identically after pickle/load and during prediction.
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_selection import SelectFromModel
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.neural_network import MLPClassifier
            top_n = min(_DENSE_TOP_N[kind], n_features)
            forest = RandomForestClassifier(
                n_estimators=_FOREST_TREES, class_weight="balanced", n_jobs=-1,
                random_state=0)
            return Pipeline([
                ("select", SelectFromModel(forest, threshold=-np.inf,
                                           max_features=top_n)),
                ("scale", StandardScaler()),
                ("dense", MLPClassifier(hidden_layer_sizes=_MLP_HIDDEN,
                                        max_iter=_MLP_MAX_ITER, random_state=0)),
            ])
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=_FOREST_TREES, class_weight="balanced",
                                      oob_score=n_samples >= _OOB_MIN_SAMPLES, n_jobs=-1,
                                      random_state=0)

    def _train_and_classify(self, _e=None):
        """'R': one keystroke = retrain + reclassify."""
        if self._typing():
            return
        before = self._clf
        # Retrain + classify is one visual operation: keep the selected region
        # coloring and confusion cell through the transient empty prediction
        # cache. _classify refreshes them against the new predictions.
        self._train_classifier(preserve_view=True)
        if self._clf is not None and self._clf is not before:
            self._classify()
            if not self._pred:            # classification failed after training
                self._cm_cell = None
                self._refresh_region_modes()

    def _on_optimize_key(self, _e=None):
        """'O': search the dense network, install + classify the winner."""
        if self._typing():
            return
        self._optimize_network()

    # -- Optimize network ------------------------------------------------ #
    def _search_settings(self):
        """``(n_trials, timeout_s, seed, feature_search, backend)`` from the
        Model tab, each clamped to its range / choices (a malformed entry
        reads as the default)."""
        def _int(var, default, lo, hi):
            try:
                v = int(float(str(var.get()).strip()))
            except (ValueError, TypeError):
                v = default
            return max(lo, min(hi, v))
        backend = self.search_backend_var.get()
        if backend not in model_search.BACKENDS:
            backend = "auto"
        # The time limit is entered in minutes and kept in seconds.
        try:
            minutes = float(str(self.search_timeout_var.get()).strip())
            timeout_s = int(round(minutes * 60.0))
        except (ValueError, TypeError):
            timeout_s = _SEARCH_TIMEOUT_S
        lo, hi = _SEARCH_TIMEOUT_RANGE
        timeout_s = max(lo, min(hi, timeout_s))
        return (_int(self.search_trials_var, _SEARCH_TRIALS, *_SEARCH_TRIALS_RANGE),
                timeout_s,
                _int(self.search_seed_var, 0, 0, 2**31 - 1),
                bool(self.search_features_var.get()), backend)

    def _sweep_settings_trials(self):
        try:
            return self._sweep_settings()[1]
        except ValueError:
            return _SWEEP_TRIALS

    def _apply_search_view(self, d):
        """Restore the search settings from a session view dict, field by
        field; a missing or malformed value leaves that setting alone."""
        if not isinstance(d, dict):
            return
        for key, var in (("trials", self.search_trials_var),
                         ("seed", self.search_seed_var)):
            v = d.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                var.set(str(int(v)))
        v = d.get("timeout_s")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            self.search_timeout_var.set(f"{float(v) / 60.0:g}")     # entry in minutes
        if isinstance(d.get("feature_search"), bool):
            self.search_features_var.set(d["feature_search"])
        if d.get("backend") in model_search.BACKENDS:
            self.search_backend_var.set(d["backend"])
        sizes = d.get("sweep_sizes")
        if isinstance(sizes, str) and sizes.strip():
            try:
                model_search.parse_sizes(sizes)
                self.sweep_sizes_var.set(sizes)
            except ValueError:
                pass
        v = d.get("sweep_trials")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            self.sweep_trials_var.set(str(int(v)))

    def _feature_schema_now(self):
        try:
            return config_io.feature_schema(self._params_json())
        except Exception:
            return None

    def _optimize_network(self, sync=False):
        """Search the dense FC space by leave-slices-out cross-validation and
        install the winner as the "dense (tuned)" model, then classify.

        The labeled rows are gathered on the Tk thread (they touch the engine
        and the label store); the search itself runs on a worker thread and
        reports through a queue that `_search_pump` drains, so labeling stays
        live and Cancel keeps the best so far. `sync` runs the worker inline
        (the selftest)."""
        if self._search is not None:
            self.status_var.set("A search is already running - Cancel it first.")
            return
        got = self._training_set()
        if got is None:
            return
        X, y, groups, names = got
        n_trials, timeout_s, seed, feat, backend = self._search_settings()
        try:
            model_search.make_cv(y, groups, seed=seed)
        except ValueError as exc:
            self.status_var.set(f"Cannot optimize: {exc}.")
            return
        schema = self._feature_schema_now()
        searcher = "optuna" if model_search.have_optuna() else "random"
        q = queue.Queue()
        stop = threading.Event()
        space = model_search.SearchSpace(feature_search=feat)

        def progress(done, total, best):
            q.put(("progress", done, total, best))

        def work():
            try:
                res = model_search.run_search(
                    X, y, groups, names, schema, n_trials=n_trials,
                    timeout_s=timeout_s or None, seed=seed,
                    max_iter=model_search.SEARCH_MAX_ITER,
                    patience=model_search.SEARCH_PATIENCE,
                    refit_max_iter=_MLP_MAX_ITER,
                    space=space, progress_cb=progress, stop_event=stop,
                    searcher=searcher, backend=backend)
                q.put(("done", res))
            except Exception as exc:            # reported, never raised off-thread
                q.put(("error", f"{type(exc).__name__}: {exc}"))

        n_slices = len(set(groups.tolist()))
        self._search = {"queue": q, "stop": stop, "thread": None, "names": names,
                        "n": int(len(y)), "t0": time.perf_counter()}
        self.model_kind_var.set(_TUNED_KIND)
        self._set_search_buttons(running=True)
        self.search_progress_var.set(
            f"{searcher}: scoring the baseline ({len(y)} labeled regions on "
            f"{n_slices} slice{'s' if n_slices != 1 else ''}, {len(names)} features)…")
        self._compute_badge("Optimizing")
        log(f"optimize: {searcher}, {model_search.backend_label(backend)}, {n_trials} "
            f"trials, {timeout_s or 'no'} s limit, seed {seed}, feature subset "
            f"{'on' if feat else 'off'}; {len(y)} rows, {n_slices} slices, "
            f"{len(names)} features; trial budget {model_search.SEARCH_MAX_ITER} epochs "
            f"/ patience {model_search.SEARCH_PATIENCE}, batches < "
            f"{model_search.TORCH_MIN_BATCH} on sklearn"
            + ("" if searcher == "optuna" else
               " (pip install optuna for the TPE searcher)"))
        if sync:
            work()
            self._search_pump()
            return
        t = threading.Thread(target=work, name="mscoupon-optimize", daemon=True)
        self._search["thread"] = t
        t.start()
        self.root.after(_SEARCH_PUMP_MS, self._search_pump)

    def _search_pump(self):
        st = self._search
        if st is None:
            return
        finished = False
        try:
            while not finished:
                ev = st["queue"].get_nowait()
                if ev[0] == "progress":
                    _, done, total, best = ev
                    text = f"trial {done}/{total}"
                    elapsed = time.perf_counter() - st["t0"]
                    if done > 0 and elapsed > 1.0:
                        text += (f" · {_hms(elapsed)} elapsed, ~{_hms(elapsed / done)}"
                                 f"/trial, ~{_hms(elapsed / done * max(0, total - done))} left")
                    if best is not None and best.cv_score is not None:
                        text += (f" · best: log-loss {best.cv_score:.3f}, balanced acc "
                                 f"{best.cv_bacc:.1%} · {best.layers_text()} · "
                                 f"{best.features_text(len(st['names']))}")
                    self.search_progress_var.set(text)
                    self._compute_badge(f"Optimizing {done}/{total}")
                elif ev[0] == "done":
                    self._finish_search(ev[1])
                    finished = True
                elif ev[0] == "sweep_progress":
                    self._on_sweep_progress(*ev[1:])
                elif ev[0] == "sweep_done":
                    self._finish_sweep(ev[1])
                    finished = True
                elif ev[0] == "edge_progress":
                    _, f, n = ev
                    self.edge_progress_var.set(
                        f"fold {f}/{n} · {_hms(time.perf_counter() - st['t0'])} elapsed")
                    self._compute_badge(f"Evaluating edges {f}/{n}")
                elif ev[0] == "edge_done":
                    self._finish_edge_eval(ev[1])
                    finished = True
                else:
                    self.search_progress_var.set(f"failed: {ev[1]}")
                    self.status_var.set(f"Optimize failed: {ev[1]}")
                    log(f"optimize failed: {ev[1]}")
                    finished = True
        except queue.Empty:
            pass
        thread = st.get("thread")
        if not finished and thread is not None and not thread.is_alive() \
                and st["queue"].empty():
            self.search_progress_var.set("failed: the search thread ended silently")
            self.status_var.set("Optimize failed: the search thread ended silently")
            finished = True
        if finished:
            self._search = None
            self._set_search_buttons(running=False)
            self._clear_compute_badge()
            self._refresh_sweep_buttons()
            return
        self.root.after(_SEARCH_PUMP_MS, self._search_pump)

    def _finish_search(self, res):
        st = self._search
        names = st["names"]
        spec = res.spec
        self._search_spec = spec
        self._install_model(res.estimator, names, _TUNED_KIND, spec, preserve_view=True)
        self._refresh_model_readout()
        gain = ""
        if spec.baseline_score is not None:
            gain = f" (baseline {spec.baseline_score:.3f} / {spec.baseline_bacc:.1%})"
        summary = (f"{spec.searcher}: {res.n_trials} trials in {res.elapsed_s:.0f} s"
                   f"{' (stopped early)' if res.stopped else ''} · best: log-loss "
                   f"{spec.cv_score:.3f}, balanced acc {spec.cv_bacc:.1%}{gain} · "
                   f"{spec.layers_text()} · {spec.features_text(len(names))}")
        self.search_progress_var.set(summary)
        log("optimize: " + summary + " -- " + spec.describe(len(names)))
        if res.importances:
            log("optimize: top features (permutation, log-loss): "
                + "  ".join(f"{n}={v:.3f}" for n, v in res.importances[:8]))
        self._refresh_model_strip()
        saved = self._autosave_tuned_model()
        if saved:
            self.search_progress_var.set(summary + f" · saved {saved}")
        # Classify first: it writes its own status line, and the summary
        # (with where the winner was saved) is the one that should stay.
        self._classify()
        self.status_var.set(f"Optimized network on {st['n']} labeled regions - {summary}"
                            + (f" - saved {saved}" if saved else ""))

    def _autosave_tuned_model(self):
        """Pickle the search winner under the session's models folder and
        record it, so the result of a run nobody watched survives: the
        session reloads its most recent recorded model on restore. Returns
        the path, or None (with a log line) when the write failed."""
        folder = self._models_dir
        if not folder or self._clf is None:
            return None
        try:
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, time.strftime("tuned_%Y%m%d_%H%M%S.pkl"))
            self._save_classifier_to(path)
        except Exception as exc:
            log(f"optimize: could not save the winner: {exc}")
            return None
        log(f"optimize: winner saved to {path}")
        return path

    # -- Size sweep ------------------------------------------------------ #
    def _sweep_network(self, sync=False):
        """Score a ladder of architectures, each at its own best settings
        (one fixed-size search per rung), report them, install the best rung
        and classify. Same worker/pump/Cancel as Optimize."""
        if self._search is not None:
            self.status_var.set("A search is already running - Cancel it first.")
            return
        try:
            sizes, per_size = self._sweep_settings()
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        got = self._training_set()
        if got is None:
            return
        X, y, groups, names = got
        _n_trials, timeout_s, seed, feat, backend = self._search_settings()
        try:
            model_search.make_cv(y, groups, seed=seed)
        except ValueError as exc:
            self.status_var.set(f"Cannot sweep: {exc}.")
            return
        schema = self._feature_schema_now()
        searcher = "optuna" if model_search.have_optuna() else "random"
        q = queue.Queue()
        stop = threading.Event()
        space = model_search.SearchSpace(feature_search=feat)

        def progress(k, n, hidden, done, total, row):
            q.put(("sweep_progress", k, n, hidden, done, total, row))

        def work():
            try:
                res = model_search.run_size_sweep(
                    X, y, groups, names, schema, sizes=sizes, trials_per_size=per_size,
                    timeout_s=timeout_s or None, seed=seed,
                    max_iter=model_search.SEARCH_MAX_ITER,
                    patience=model_search.SEARCH_PATIENCE, refit_max_iter=_MLP_MAX_ITER,
                    space=space, progress_cb=progress, stop_event=stop,
                    searcher=searcher, backend=backend)
                q.put(("sweep_done", res))
            except Exception as exc:            # reported, never raised off-thread
                q.put(("error", f"{type(exc).__name__}: {exc}"))

        self._sweep = None
        self._sweep_rows = []
        self._refresh_sweep_report([])
        self._search = {"queue": q, "stop": stop, "thread": None, "names": names,
                        "n": int(len(y)), "t0": time.perf_counter(), "kind": "sweep",
                        "sizes": sizes, "per_size": per_size}
        self.model_kind_var.set(_TUNED_KIND)
        self._set_search_buttons(running=True)
        self.search_progress_var.set(
            f"size sweep: {len(sizes)} rung(s) x {per_size} trials on {len(y)} labeled "
            f"regions, {len(names)} features…")
        self._compute_badge("Sweeping sizes")
        log(f"size sweep: {[model_search.size_text(h) for h in sizes]}, {per_size} "
            f"trials/rung, {searcher}, {model_search.backend_label(backend)}, "
            f"{timeout_s or 'no'} s limit, seed {seed}; {len(y)} rows, {len(names)} features")
        if sync:
            work()
            self._search_pump()
            return
        t = threading.Thread(target=work, name="mscoupon-size-sweep", daemon=True)
        self._search["thread"] = t
        t.start()
        self.root.after(_SEARCH_PUMP_MS, self._search_pump)

    def _on_sweep_progress(self, k, n, hidden, done, total, row):
        st = self._search
        size = model_search.size_text(hidden)
        if row is not None:
            self._sweep_rows.append(row)
            self._sweep = model_search.SweepResult(
                rows=list(self._sweep_rows), names=list(st["names"]), n_classes=0,
                elapsed_s=time.perf_counter() - st["t0"])
            self._refresh_sweep_report(self._sweep_rows)
            log(f"size sweep: {size}: log-loss {row.cv_score:.3f}, bal. acc "
                f"{row.cv_bacc:.1%}, {row.n_params:,} params, {row.n_trials} trials, "
                f"{row.elapsed_s:.0f} s -- {row.spec.settings_text(len(st['names']))}")
        elapsed = time.perf_counter() - st["t0"]
        text = f"size {k + 1}/{n} ({size}): trial {done}/{total}"
        rungs_done = len(self._sweep_rows)
        if rungs_done and elapsed > 1.0:
            per_rung = elapsed / rungs_done
            text += (f" · {_hms(elapsed)} elapsed, ~{_hms(per_rung)}/rung, "
                     f"~{_hms(per_rung * max(0, n - rungs_done))} left")
        if self._sweep_rows:
            b = self._sweep_rows[min(range(len(self._sweep_rows)),
                                     key=lambda i: self._sweep_rows[i].cv_score)]
            text += f" · best so far {b.size} (log-loss {b.cv_score:.3f})"
        self.search_progress_var.set(text)
        self._compute_badge(f"Sweeping {k + 1}/{n} · {done}/{total}")

    def _finish_sweep(self, res):
        st = self._search
        self._sweep = res
        self._sweep_rows = list(res.rows)
        self._refresh_sweep_report(res.rows, stopped=res.stopped)
        lines = res.report_lines()
        log("size sweep report:\n  " + "\n  ".join(lines))
        # Install the best rung (the Optimize convention); a smaller rung is
        # one click away in the table.
        best = res.rows[res.best_index()]
        self._install_sweep_row(best, announce=False)
        saved = self._autosave_tuned_model()
        report_path = self._save_sweep_report(res, saved)
        summary = (f"{len(res.rows)} rung(s) in {_hms(res.elapsed_s)}"
                   f"{' (stopped early)' if res.stopped else ''} · {res.summary()}")
        self.search_progress_var.set(summary + (f" · saved {saved}" if saved else ""))
        self._classify()
        self.status_var.set(f"Size sweep on {st['n']} labeled regions - {summary}"
                            + (f" - report {report_path}" if report_path else ""))

    def _install_sweep_row(self, row, announce=True):
        names = list(self._search["names"] if self._search is not None
                     else (self._sweep.names if self._sweep is not None else []))
        self._search_spec = row.spec
        self._install_model(row.estimator, names, _TUNED_KIND, row.spec, preserve_view=True)
        self._refresh_model_readout()
        self._refresh_model_strip()
        if announce:
            self.status_var.set(f"Installed {row.size} ({row.n_params:,} params, CV log-loss "
                                f"{row.cv_score:.3f}, bal. acc {row.cv_bacc:.1%})")

    def _sweep_install_selected(self):
        """Make the selected rung the model, save it and classify."""
        if self._sweep is None or self._search is not None:
            return
        sel = self.sweep_tree.selection()
        if not sel:
            return
        try:
            row = self._sweep.rows[int(sel[0])]
        except (ValueError, IndexError):
            return
        self._install_sweep_row(row)
        saved = self._autosave_tuned_model()
        status = self.status_var.get()
        self._classify()
        self.status_var.set(status + (f" - saved {saved}" if saved else ""))

    def _save_sweep_report(self, res, model_path=None):
        """Write the sweep (rows, specs, the text table) as JSON beside the
        saved models, so the numbers outlive the session. Returns the path
        or None."""
        folder = self._models_dir
        if not folder:
            return None
        try:
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, time.strftime("sweep_%Y%m%d_%H%M%S.json"))
            doc = res.to_dict()
            doc["installed_model"] = model_path
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
        except Exception as exc:
            log(f"size sweep: could not save the report: {exc}")
            return None
        log(f"size sweep: report saved to {path}")
        return path

    # -- Evaluate edges -------------------------------------------------- #
    def _evaluate_edges(self, sync=False):
        """Leave-slices-out report for the edge model on top of the CURRENT
        base kind: per fold the base is refit on the training slices, the
        pair model on their labeled edges, both scored on the held-out slices,
        then voting before/after. Same worker / pump / Cancel as Optimize."""
        if self._search is not None:
            self.status_var.set("A search is already running - Cancel it first.")
            return
        if self._clf is None or not self._has_embedding(self._clf):
            self.status_var.set("Evaluate edges needs a trained dense base - Train a dense "
                                "or '-> edges' kind first.")
            return
        names = list(self._clf_names)
        data = self._edge_training_data(names)
        if data is None:
            return
        X_all, cls, grp, ext, edges, _n = data
        _trials, _timeout, seed, _feat, _backend = self._search_settings()
        try:
            model_search.make_cv(cls[cls > 0], grp[cls > 0], seed=seed)
        except ValueError as exc:
            self.status_var.set(f"Cannot evaluate edges: {exc}.")
            return
        if not (edges["both"] & edges["diff"]).any():
            self.status_var.set("Cannot evaluate edges: no labeled edge crosses classes - "
                                "label two touching regions of different classes.")
            return
        from sklearn.base import clone
        template = clone(self._clf)
        spec = self._edge_spec_from_ui()
        q = queue.Queue()
        stop = threading.Event()

        def work():
            try:
                rep = edge_model.evaluate_edges(
                    lambda: clone(template), X_all, cls, grp, edges, ext, spec, seed=seed,
                    progress_cb=lambda f, n: q.put(("edge_progress", f, n)), stop_event=stop)
                q.put(("edge_done", rep))
            except Exception as exc:            # reported, never raised off-thread
                q.put(("error", f"{type(exc).__name__}: {exc}"))

        self._search = {"queue": q, "stop": stop, "thread": None, "names": names,
                        "n": int((cls > 0).sum()), "t0": time.perf_counter(), "mode": "edges"}
        self._set_search_buttons(running=True)
        self.edge_progress_var.set(f"evaluating: {int(edges['both'].sum()):,} labeled pairs "
                                   f"({int(edges['diff'].sum()):,} boundaries) on "
                                   f"{len(set(grp[cls > 0].tolist()))} slice(s)…")
        self._compute_badge("Evaluating edges")
        log(f"evaluate edges: {spec.describe()}; {len(edges['a'])} edges, "
            f"{int(edges['both'].sum())} labeled pairs")
        if sync:
            work()
            self._search_pump()
            return
        t = threading.Thread(target=work, name="mscoupon-edge-eval", daemon=True)
        self._search["thread"] = t
        t.start()
        self.root.after(_SEARCH_PUMP_MS, self._search_pump)

    def _finish_edge_eval(self, report):
        if self._edge_model is not None:
            self._edge_model.report = report
        self._fill_edge_report(report)
        lines = edge_model.report_lines(report)
        log("evaluate edges:\n  " + "\n  ".join(lines))
        learned = report.get("edge", {}).get("learned")
        base = report.get("edge", {}).get("argmax differs", {})
        reg = report.get("region", {})
        if learned:
            summary = (f"learned {learned['diff_recall']:.0%}/{learned['diff_precision']:.0%} "
                       f"boundary recall/precision vs argmax {base.get('diff_recall', 0):.0%}/"
                       f"{base.get('diff_precision', 0):.0%}")
            if reg.get("before") and reg.get("after"):
                summary += (f" · held-out region errors {reg['before']['errors']} -> "
                            f"{reg['after']['errors']} with voting")
        else:
            summary = "no fold could be scored: " + "; ".join(report.get("skipped", [])[:2])
        summary += (f" · {report.get('n_folds', 0)}-fold {report.get('cv_kind', '')} CV in "
                    f"{_hms(report.get('elapsed_s', 0))}")
        if report.get("stopped"):
            summary += " (stopped early)"
        self.edge_progress_var.set(summary)
        self._refresh_model_readout()
        self._refresh_model_strip()
        self._refresh_edge_readout()
        self.status_var.set("Evaluated edges - " + summary)

    def _cancel_search(self):
        st = self._search
        if st is None:
            return
        st["stop"].set()
        self.search_progress_var.set("stopping after the current trial…")
        self.cancel_search_btn.config(state="disabled")

    def _set_search_buttons(self, running):
        for name, state in (("optimize_btn", "disabled" if running else "normal"),
                            ("sweep_btn", "disabled" if running else "normal"),
                            ("edge_eval_btn", "disabled" if running else "normal"),
                            ("cancel_search_btn", "normal" if running else "disabled")):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.config(state=state)

    def _on_classify_key(self, _e=None):
        """'C': classify/reclassify with the current model."""
        if self._typing() or self._clf is None:
            return
        self._classify()

    # -- computing badge on the image plane (top-left canvas HUD) -------- #
    def _compute_badge(self, text):
        if self.viewer is not None:
            self.viewer.set_hud("busy", text)
            try:
                self.root.update_idletasks()   # paint before the blocking fit
            except tk.TclError:
                pass

    def _clear_compute_badge(self):
        self._update_busy()      # restores busy/stale/none per engine state

    # -- model <-> profile compatibility --------------------------------- #
    def _expected_feature_names(self):
        """The feature set the ACTIVE profile's statistics produce -- a pure
        schema call, no priming. None when the compiled extension is absent
        (the fallback field list would falsely block)."""
        try:
            from msseg import mscoupon as ext
            if not getattr(ext, "_HAVE_EXTENSION", False):
                return None
        except ImportError:
            return None
        fields = config_io.query_fields(self._params_json())
        return [n for n in fields if n not in _NON_FEATURE_FIELDS]

    def _check_model_compat(self, names, context):
        """None when `names` matches the active profile's statistics schema;
        else a blocking message naming the exact mismatch. Compared as SETS:
        the feature matrix is assembled by name, so order never matters."""
        expected = self._expected_feature_names()
        if expected is None:
            log(f"model compatibility check skipped ({context}): "
                "compiled extension not available")
            return None
        want, have = set(names), set(expected)
        if want == have:
            return None
        prof = "?"
        if 0 <= self.active_profile_idx < len(self.profiles):
            prof = self.profiles[self.active_profile_idx]["name"]
        missing = sorted(want - have)
        extra = sorted(have - want)
        parts = []
        if missing:
            parts.append("model needs: " + ", ".join(missing[:6])
                         + ("…" if len(missing) > 6 else ""))
        if extra:
            parts.append("profile adds: " + ", ".join(extra[:6])
                         + ("…" if len(extra) > 6 else ""))
        return (f"Model does not match profile '{prof}' statistics "
                f"({context}) - " + "; ".join(parts)
                + ". Switch profiles or retrain.")

    # -- model provenance ------------------------------------------------ #
    def _stats_brief(self, stats):
        """One line for a statistics block's channels, e.g. `base,blur(0.7,1.5)
        x4` -- the two numbers (channels, reductions) that decide how wide the
        model's per-feature row is."""
        try:
            doc = config_io.statistics_from_json(stats or {})
        except Exception:
            return "?"
        parts = []
        for c in doc["channels"]:
            sig = c.get("sigmas") or []
            parts.append(c["kind"] + (("(" + ",".join(f"{s:g}" for s in sig) + ")")
                                      if sig else ""))
        return ",".join(parts) + f" x{len(doc['reductions'])}"

    def _refresh_model_strip(self):
        """Repaint the provenance line above Train/Classify. The mismatch check
        is inlined rather than routed through _check_model_compat: this runs on
        every profile switch and must not log."""
        var = getattr(self, "model_strip_var", None)
        if var is None:
            return
        if self._clf is None:
            var.set("no model")
            return
        names = list(self._clf_names or [])
        stats = None
        if self.models:
            last = self.models[-1]
            if list(last.get("fingerprint") or []) == names:
                stats = last.get("statistics") or None
        if stats is None and 0 <= self.active_profile_idx < len(self.profiles):
            stats = self.profiles[self.active_profile_idx].get("statistics")
        bits = [self._clf_kind, f"{len(names)} feats", self._stats_brief(stats)]
        if self._clf_spec is not None:
            bits[1] = self._clf_spec.brief(len(names))
        expected = self._expected_feature_names()
        if expected is not None and set(expected) != set(names):
            bits.append("⚠ profile mismatch")
        if self._edge_model is not None and not _is_edge_kind(self._clf_kind):
            bits.insert(1, "-> edges")
        var.set(" · ".join(bits))
        self._refresh_hints()

    def _switch_profile(self, idx):
        super()._switch_profile(idx)
        self._refresh_model_strip()

    def _profile_from_model(self, path, statistics):
        """Append a profile that keeps the active one's filters/MSC/selection
        but MEASURES what the model was trained on, and activate it.

        A new profile rather than an edit of the active one: profiles have no
        undo stack, and _switch_profile already drops the primed data -- which
        a statistics change needs, since the per-slice feature table is baked
        at prime time (a selection rerun would not rebuild it)."""
        self._snapshot_active_profile()
        base = {}
        if 0 <= self.active_profile_idx < len(self.profiles):
            base = json.loads(json.dumps(self.profiles[self.active_profile_idx]))
        stats = json.loads(json.dumps(statistics))
        base["statistics"] = stats
        # profile_from_json takes the radius from msc, not from the statistics
        # block, so carry the model's across or the round-trip would reset it.
        base.setdefault("msc", {})["extremum_sample_radius"] = max(
            0, int(stats.get("extremum_sample_radius") or 0))
        base["name"] = session.dedupe_profile_name(
            f"from {os.path.basename(path)}", [p["name"] for p in self.profiles])
        notes = []
        # The round-trip re-validates feature_filters against the field
        # universe the NEW statistics produce, dropping the stale ones.
        self.profiles.append(session.profile_from_json(base, notes))
        for msg in notes:
            log(msg)
        self._switch_profile(len(self.profiles) - 1)

    def _classify(self):
        """Predict a class for EVERY region of every computed slice and show
        the result as a translucent layer under the drawn labels. BLOCKS with
        a message when the model's features don't match the active profile
        (replacing the old silent per-slice skip)."""
        if self._clf is None:
            self.status_var.set("Train first.")
            return
        msg = self._check_model_compat(self._clf_names, "classify")
        if msg:
            messagebox.showerror("mscoupon labeler", msg)
            self.status_var.set(msg)
            return
        import numpy as np
        slices = self._all_stat_slices("Preparing classification")
        if slices is None:
            return
        count = 0
        self._compute_badge("Classifying")
        t0 = time.perf_counter()
        try:
            for si, li, key, rec, table in slices:
                if self._predict_slice(si, li, rec, np) is None:
                    self._pred.clear()
                    self.status_var.set(
                        f"Classification stopped: incomplete statistics on "
                        f"slice {si}:{li}.")
                    return
                count += 1
        finally:
            self._clear_compute_badge()
        self._refresh_region_modes()
        self._refresh_confusion()
        self._refresh_render()
        self._refresh_edge_readout()
        self.status_var.set(f"Classified {count} slice(s) in "
                            f"{1e3 * (time.perf_counter() - t0):.0f} ms - "
                            "predictions shown under your labels")

    def _predict_slice(self, si, li, rec, np):
        """The slice's region->class predictions at rec's commit, computing
        and caching them when a model is loaded. Callers gate schema
        compatibility (this only checks that a model exists).

        Caches ``(commit, region_class, region_proba)``: the class
        probabilities come out of the same forward pass as the hard label
        (which is their argmax), and the regions coloring modes and any later
        confidence read need them per region, not per feature row."""
        pr = self._pred.get((si, li))
        if pr is not None and pr[0] == rec.get("commit"):
            return pr[1]
        if self._clf is None:
            return None
        table = rec.get("stats")
        if getattr(table, "values", None) is None:
            return None
        mat = self._feature_matrix(table, self._clf_names, np)
        fids = table.column("feature_id")
        if mat is None or fids is None:
            return None
        # Column order is the estimator's own classes_, NOT 1..N: both kinds
        # sit behind a Pipeline and neither promises contiguous class ids.
        proba = np.asarray(self._clf.predict_proba(mat), np.float32)
        classes = np.asarray(self._clf.classes_, int)
        pred = classes[proba.argmax(1)].astype(np.uint8)
        labels = rec["labels"]
        K = max(int(labels.max()) + 1 if labels.size else 1, 1)
        region_class = np.zeros(K, np.uint8)
        # MAX_CLASSES-wide, not n_classes-wide: the class count can change
        # under a cached slice, and indexing by class id keeps the shape.
        region_proba = np.zeros((K, MAX_CLASSES), np.float32)
        fid = fids.astype(int)
        ok = (fid >= 0) & (fid < K)
        region_class[fid[ok]] = pred[ok]
        keep = (classes >= 0) & (classes < MAX_CLASSES)
        region_proba[fid[ok][:, None], classes[keep][None, :]] = \
            proba[ok][:, keep]
        # The edge model: p(diff) per arc of the RECORD's graph (so the magic
        # fill and the coloring modes index it without a row mapping), and
        # neighbour voting when an edge kind is selected. `raw` keeps the base
        # net's answer so a kind flip re-votes without a forward pass.
        aux = None
        final = region_class
        edge = self._edge_model
        if edge is not None:
            arcs = self._record_arcs(rec, np)
            if arcs is not None and len(arcs.get("a", ())):
                try:
                    ia, ib, keep_e = magic_fill.index_arcs(arcs, fid, np)
                    names = list(self._clf_names)
                    ext = (mat[:, names.index("ext_filtered")]
                           if "ext_filtered" in names else None)
                    sad = arcs.get("saddle")
                    sad = None if sad is None else np.asarray(sad, np.float64)[keep_e]
                    pd_rows = edge_model.predict_pdiff(edge, self._clf, mat, ia, ib, sad, ext)
                    pdiff = np.full(len(arcs["a"]), np.nan, np.float32)
                    pdiff[keep_e] = pd_rows
                    spec = self._edge_spec_from_ui()
                    aux = {"raw": region_class.copy(), "pdiff": pdiff, "keep": keep_e,
                           "la": np.asarray(arcs["a"]), "lb": np.asarray(arcs["b"]),
                           "lam": spec.lam, "rounds": spec.rounds, "flips": 0,
                           "active": False, "source": arcs.get("source")}
                    if _is_edge_kind(self.model_kind_var.get()):
                        final, flips = self._vote_entry(region_proba, aux, spec.lam,
                                                        spec.rounds, np)
                        aux["flips"], aux["active"] = flips, True
                except (ValueError, IndexError) as exc:
                    log(f"edge model unusable on slice {si}:{li}: {exc}")
                    aux = None
        entry = ((rec.get("commit"), final, region_proba) if aux is None
                 else (rec.get("commit"), final, region_proba, aux))
        self._pred[(si, li)] = entry
        return final

    # -- training-set export --------------------------------------------- #
    def _ensure_slice_record(self, si, li):
        """The slice's record at the current commit, computing it SYNCHRONOUSLY
        when the lazy per-slice tier hasn't visited it yet (the training-set
        export needs every slice, not just the browsed ones). Caller must
        ensure no assembly worker is running (the pipes are stateful)."""
        rec = self.engine.record(si, li)
        if rec is not None and rec.get("labels") is not None:
            return rec
        try:
            import numpy as np
            from msseg import mscoupon as ext
        except ImportError:
            return None
        params = dict(self._assembly_params(si, "slice", li))
        params["commit"] = self.engine.commit_id
        tm = {"persist": 0.0, "labels": 0.0, "stats": 0.0,
              "query": 0.0, "rasters": 0.0}
        try:
            rec = self.engine._slice_result(si, li, params, ext, np, tm)
        except Exception as exc:
            log(f"slice ({si},{li}) record failed: {exc}")
            return None
        self.engine.slices[(si, li)] = rec
        return rec

    def _make_training_set(self):
        """Pick a folder; write `train/` (the raw input TIFFs) and `labels/`
        (per-pixel class-id masks from the classifier, with user annotations
        winning where they disagree) -- the raw material for a UNet-style
        image model later."""
        if not self.primed:
            self.status_var.set("Run first - the masks need computed regions.")
            return
        if self.engine.asm_running or self.engine.asm_pending is not None:
            self.status_var.set("Busy computing - try again in a moment.")
            return
        out = filedialog.askdirectory(title="Choose a folder for the training set")
        if not out:
            return
        written, skipped = self._write_training_set(out)
        msg = (f"Training set: {written} image(s) -> {os.path.join(out, 'train')} "
               f"+ masks -> {os.path.join(out, 'labels')}")
        if skipped:
            msg += f" ({skipped} slice(s) skipped - no labels or predictions)"
        self.status_var.set(msg)

    def _write_training_set(self, out_dir):
        import shutil
        import numpy as np
        from PIL import Image
        train_dir = os.path.join(out_dir, "train")
        labels_dir = os.path.join(out_dir, "labels")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        # Gate the model ONCE: under a mismatched profile the masks fall back
        # to annotations alone rather than silently wrong predictions.
        use_model = (self._clf is not None and
                     self._check_model_compat(self._clf_names,
                                              "training-set export") is None)
        written = skipped = 0
        try:
            for si, p in enumerate(self.primed):
                for li in range(len(p["pipes"])):
                    self._compute_badge(f"Exporting {si}:{li}")
                    rec = self._ensure_slice_record(si, li)
                    key = self._slice_key(si, li)
                    if rec is None or rec.get("labels") is None or key is None:
                        skipped += 1
                        continue
                    labels = rec["labels"]
                    K = int(labels.max()) + 1 if labels.size else 1
                    combined = np.zeros(max(K, 1), np.uint8)
                    if use_model:
                        pred = self._predict_slice(si, li, rec, np)
                        if pred is not None:
                            combined = pred.copy()
                    user = resolve_slice(self.store.for_slice(key), labels, np)
                    combined[user > 0] = user[user > 0]    # annotations win
                    if not combined.any():
                        skipped += 1                       # nothing to teach
                        continue
                    mask = np.zeros(labels.shape, np.uint8)
                    valid = labels >= 0
                    mask[valid] = combined[labels[valid]]
                    src = p["files"][li]
                    folder = "seq"
                    if si < len(self.subsequences):
                        folder = self.subsequences[si].get("folder") or "seq"
                    safe = folder.replace("/", "_").replace("\\", "_")
                    base = os.path.basename(src)
                    stem, _ext = os.path.splitext(base)
                    try:
                        shutil.copy2(src, os.path.join(train_dir, f"{safe}__{base}"))
                    except OSError as exc:
                        log(f"training set: could not copy {src}: {exc}")
                        skipped += 1
                        continue
                    Image.fromarray(mask).save(
                        os.path.join(labels_dir, f"{safe}__{stem}.tiff"))
                    written += 1
        finally:
            self._clear_compute_badge()
        return written, skipped

    # -- classifier persistence (pickle: the sklearn-native format) ------ #
    def _save_classifier(self):
        if self._clf is None:
            self.status_var.set("Train first.")
            return
        path = filedialog.asksaveasfilename(title="Save classifier",
                                            defaultextension=".pkl",
                                            initialfile="classifier.pkl",
                                            filetypes=[("Pickle", "*.pkl")])
        if not path:
            return
        self._save_classifier_to(path)
        self.status_var.set(f"Wrote {path}")

    def _save_classifier_to(self, path):
        """Pickle v3: model + feature names + kind + the statistics block it
        was trained under (the fingerprint the session records) + the tuned
        spec (None for the other kinds). v1 (no kind/statistics) and v2 (no
        spec) pickles still load."""
        import pickle
        self._snapshot_active_profile()
        stats = dict(self.profiles[self.active_profile_idx].get("statistics") or {})
        spec = None if self._clf_spec is None else self._clf_spec.to_dict()
        # v4: the edge model on top (None for plain kinds) and the stack's
        # settings, so a loaded '-> edges' model can be retrained as is.
        edge = None if self._edge_model is None else self._edge_model.to_dict()
        stack = {"custom_hidden": self.custom_hidden_var.get(),
                 "edge_spec": self._edge_spec_from_ui().to_dict()}
        with open(path, "wb") as f:
            pickle.dump({"app": "mscoupon-labeler-classifier", "version": 4,
                         "kind": self._clf_kind, "names": self._clf_names,
                         "model": self._clf, "statistics": stats, "spec": spec,
                         "edge": edge, "stack": stack}, f)
        self._record_model(path, stats)

    def _record_model(self, path, statistics):
        """Register a saved/loaded model on the session (deduped by path)."""
        entry = {"path": os.path.abspath(path),
                 "fingerprint": list(self._clf_names or []),
                 "kind": self._clf_kind,
                 "statistics": dict(statistics or {}),
                 "spec": None if self._clf_spec is None else self._clf_spec.to_dict(),
                 "edge": self._edge_model is not None}
        self.models = [m for m in self.models if m.get("path") != entry["path"]]
        self.models.append(entry)

    def _load_classifier(self):
        path = filedialog.askopenfilename(title="Load classifier",
                                          filetypes=[("Pickle", "*.pkl")])
        if not path:
            return
        try:
            self._load_classifier_from(path, interactive=True)
        except Exception as exc:     # missing sklearn, wrong file, incompat
            messagebox.showerror("mscoupon labeler", str(exc))
            self.status_var.set(f"Could not load classifier: {exc}")
            return
        self.status_var.set(f"Loaded {self._clf_kind} from {path} "
                            f"({len(self._clf_names)} features)")

    def _load_classifier_from(self, path, interactive=False):
        """Install a pickled model. `interactive` is opt-in: session restore
        and the selftest call this headlessly, where a modal would hang."""
        import pickle
        with open(path, "rb") as f:
            doc = pickle.load(f)
        if (not isinstance(doc, dict)
                or doc.get("app") != "mscoupon-labeler-classifier"
                or "model" not in doc or not doc.get("names")):
            raise ValueError("not a labeler classifier file")
        # The compatibility gate: a model trained under different statistics
        # is refused OUTRIGHT (per-feature values would silently mean the
        # wrong thing), before anything is installed. Interactively -- and only
        # for a v2 pickle, which carries the statistics it was trained under --
        # the user is first offered a profile built from those statistics.
        msg = self._check_model_compat(doc["names"], "load")
        if msg and interactive and doc.get("statistics"):
            if not messagebox.askyesno(
                    "mscoupon labeler",
                    msg + "\n\nCreate a profile from the model's own statistics "
                          "and switch to it?"):
                raise ValueError(msg)
            self._profile_from_model(path, doc["statistics"])
            # The new profile can still miss: feature_fields may resolve
            # differently here than in the build that saved the pickle.
            msg = self._check_model_compat(doc["names"], "load")
        if msg:
            raise ValueError(msg)
        self._pred.clear()               # predictions belong to the old model
        self._clf = doc["model"]
        self._clf_names = list(doc["names"])
        self._clf_kind = str(doc.get("kind") or "random forest")
        spec = doc.get("spec")
        self._clf_spec = (model_search.ModelSpec.from_dict(spec)
                          if isinstance(spec, dict) else None)
        if self._clf_spec is not None and _base_kind(self._clf_kind) == _TUNED_KIND:
            self._search_spec = self._clf_spec   # Train rebuilds what was loaded
        # v4: the edge model on top, only when it was fit over these features.
        self._edge_model = None
        edge = doc.get("edge")
        if isinstance(edge, dict):
            try:
                em = edge_model.EdgeModel.from_dict(edge)
                if em.names_hash == edge_model.names_hash(self._clf_names):
                    self._edge_model = em
                else:
                    log("edge model in the pickle was fit over other features - dropped")
            except Exception as exc:
                log(f"edge model in the pickle not restored: {exc}")
        stack = doc.get("stack") or {}
        if isinstance(stack.get("custom_hidden"), str):
            try:
                model_search.parse_sizes(stack["custom_hidden"])
                self.custom_hidden_var.set(stack["custom_hidden"])
            except ValueError:
                pass
        if isinstance(stack.get("edge_spec"), dict):
            self._apply_edge_spec(edge_model.EdgeSpec.from_dict(stack["edge_spec"]))
        if self._clf_kind in _MODEL_KINDS:
            self.model_kind_var.set(self._clf_kind)
        self._refresh_model_readout()
        self.classify_btn.config(state="normal")
        self._record_model(path, doc.get("statistics") or {})
        self._refresh_model_strip()
        self._refresh_edge_readout()

    def _export_csv(self):
        """Resolved region -> class table: one row per living MSC region of
        every slice whose labels are cached at the current commit (class 0 =
        unlabeled, so the file carries negatives for training too)."""
        if not self.primed:
            self.status_var.set("Nothing primed - Run first, then export.")
            return
        path = filedialog.asksaveasfilename(title="Export resolved labels as CSV",
                                            defaultextension=".csv",
                                            initialfile="labels.csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        covered, skipped = self._write_labels_csv(path)
        msg = f"Wrote {path}: {covered} slice(s)"
        if skipped:
            msg += (f", {skipped} skipped (labels not computed at this commit - "
                    "browse them or Rerun first)")
        self.status_var.set(msg)

    def _write_labels_csv(self, path):
        import numpy as np
        covered = skipped = 0
        # Every statistics column the spec produced rides along (the feature
        # row IS the region's future design vector). The schema is spec-driven
        # and identical across slices of a run; the first cached table fixes
        # the column order, later tables are aligned by name.
        stat_names = None
        rows = []           # (slice_key, region_id, class, predicted, table, row_idx)
        for si, p in enumerate(self.primed):
            for li in range(len(p["pipes"])):
                rec = self.engine.record(si, li)
                key = self._slice_key(si, li)
                if rec is None or rec.get("labels") is None or key is None:
                    skipped += 1
                    continue
                labels = rec["labels"]
                rc = resolve_slice(self.store.for_slice(key), labels, np)
                # Classifier predictions ride along when they exist for this
                # commit; blank otherwise (a prediction of class 0 never
                # happens -- the model only knows labeled classes).
                pr = self._pred.get((si, li))
                pred = pr[1] if pr is not None and pr[0] == rec.get("commit") else None
                table = rec.get("stats")
                if getattr(table, "values", None) is None:
                    table = None
                row_of = {}
                if table is not None:
                    if stat_names is None:
                        stat_names = [n for n in table.names if n != "feature_id"]
                    fids = table.column("feature_id")
                    if fids is not None:
                        row_of = {int(v): r for r, v in enumerate(fids)}
                ids = np.unique(labels)
                for i in ids[ids >= 0]:
                    p_val = ("" if pred is None or i >= len(pred)
                             else str(int(pred[i])))
                    rows.append((key, int(i), int(rc[i]), p_val, table,
                                 row_of.get(int(i))))
                covered += 1
        lines = ["slice,region_id,class,predicted" +
                 ("," + ",".join(stat_names) if stat_names else "")]
        for key, rid, cls, p_val, table, r in rows:
            line = f"{key},{rid},{cls},{p_val}"
            if stat_names:
                vals = []
                for n in stat_names:
                    col = table.column(n) if (table is not None and r is not None) else None
                    vals.append("" if col is None else f"{float(col[r]):.10g}")
                line += "," + ",".join(vals)
            lines.append(line)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(lines) + "\n")
        return covered, skipped

    def _view_state(self):
        d = super()._view_state()
        d["tool"] = self.tool_var.get()
        d["magic"] = {"metric": self.magic_metric_var.get(),
                      "mode": self.magic_mode_var.get(),
                      "channels": self.magic_channels_var.get(),
                      "ring": self.blob_ring_var.get(),
                      # Parsed floats, not the raw entry text: what the fill
                      # actually used, re-rendered with :g on restore.
                      "hop_gain": _bounded_float(self.magic_gain_var.get(),
                                                 _DEFAULT_HOP_GAIN, *_HOP_GAIN_RANGE),
                      "drag_px": _bounded_float(self.magic_drag_var.get(),
                                                _DEFAULT_DRAG_PX, *_DRAG_PX_RANGE)}
        d["center_tab"] = self._center_tab_name()
        trials, timeout_s, seed, feat, backend = self._search_settings()
        d["model_search"] = {"trials": trials, "timeout_s": timeout_s, "seed": seed,
                             "feature_search": feat, "backend": backend,
                             "sweep_sizes": self.sweep_sizes_var.get(),
                             "sweep_trials": self._sweep_settings_trials()}
        d["neighbours"] = {"custom_hidden": self.custom_hidden_var.get(),
                           "freeze_base": bool(self.freeze_base_var.get()),
                           "edge_spec": self._edge_spec_from_ui().to_dict()}
        return d

    def _apply_neighbours_view(self, d):
        if not isinstance(d, dict):
            return
        if isinstance(d.get("custom_hidden"), str):
            try:
                model_search.parse_sizes(d["custom_hidden"])
                self.custom_hidden_var.set(d["custom_hidden"])
            except ValueError:
                pass
        if isinstance(d.get("freeze_base"), bool):
            self.freeze_base_var.set(d["freeze_base"])
        if isinstance(d.get("edge_spec"), dict):
            self._apply_edge_spec(edge_model.EdgeSpec.from_dict(d["edge_spec"]))

    def _apply_magic_view(self, magic):
        """Restore the Magic rows from a session view dict, field by field --
        an unknown or malformed value leaves that setting as it is."""
        if not isinstance(magic, dict):
            return
        if magic.get("metric") in magic_fill.METRICS:
            self.magic_metric_var.set(magic["metric"])
        if magic.get("mode") in magic_fill.MODES:
            self.magic_mode_var.set(magic["mode"])
        if isinstance(magic.get("channels"), str) and magic["channels"].strip():
            self.magic_channels_var.set(magic["channels"])
        if str(magic.get("ring")) in _RING_CHOICES:
            self.blob_ring_var.set(str(magic["ring"]))
        for key, var, (lo, hi) in (("hop_gain", self.magic_gain_var, _HOP_GAIN_RANGE),
                                   ("drag_px", self.magic_drag_var, _DRAG_PX_RANGE)):
            val = magic.get(key)
            if (isinstance(val, (int, float)) and not isinstance(val, bool)
                    and lo <= val <= hi):
                var.set(f"{float(val):g}")

    def _session_doc(self):
        doc = super()._session_doc()
        doc["annotations"] = self.store.to_json()   # rides the 4s autosave
        doc["models"] = [dict(m) for m in self.models]
        return doc

    def _apply_session_doc(self, doc, source="session", notes=None):
        notes = super()._apply_session_doc(doc, source,
                                           notes if notes is not None else [])
        sdoc = session.session_doc_from_json(doc)
        # The store installs AFTER sequences exist, so rebind sees them (and
        # migrates legacy bare-basename keys against the qualified identity).
        if sdoc.get("annotations"):
            try:
                self._install_store(
                    LabelStore.from_json(sdoc["annotations"]))
            except Exception as exc:
                notes.append(f"labels not restored: {exc}")
        else:
            self._install_store(LabelStore())
        self.models = list(sdoc.get("models") or [])
        # Lazily reload the most recent model whose pickle still exists; a
        # failure (moved file, incompatible profile) is a note, never fatal.
        for entry in reversed(self.models):
            if os.path.isfile(entry.get("path", "")):
                try:
                    self._load_classifier_from(entry["path"])
                except Exception as exc:
                    notes.append(f"model not reloaded: {exc}")
                break
        view = sdoc.get("view") or {}
        if view.get("tool") in _UI_TOOLS:
            self.tool_var.set(view["tool"])
        self._apply_magic_view(view.get("magic"))
        self._apply_search_view(view.get("model_search"))
        self._apply_neighbours_view(view.get("neighbours"))
        # Keep the regions toggle in sync with whatever seg_source restored to.
        self.show_regions_var.set(self.seg_source_var.get() == "msc")
        # The center tab, by name; an unknown or missing value leaves it alone.
        tab = self._center_tabs.get(view.get("center_tab"))
        if tab is not None:
            try:
                self.center.select(tab)
            except tk.TclError:
                pass
        return notes

    def _write_configs(self, out_dir):
        paths = super()._write_configs(out_dir)
        path = os.path.join(out_dir, "annotations.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.store.to_json(), f, indent=2)
        paths.append(path)
        return paths


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def main():
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        return _selftest()
    initial = args[0] if args else None
    root = tk.Tk()
    LabelerApp(root, initial)
    root.mainloop()


class _FakeEvent:
    """Enough of a Tk event for the canvas's click-vs-drag arithmetic and
    the drawing tool's modifier check."""

    def __init__(self, x, y, state=0):
        self.x, self.y = x, y
        self.x_root, self.y_root = x, y
        self.state = state


def _selftest():
    """Exercise the labeler's wiring headlessly (no engine/render needed);
    the pure geometry/ordering/LUT math is covered by tests/test_labeling.py."""
    import tempfile
    import numpy as np
    from . import labeling

    root = tk.Tk()
    root.withdraw()
    # autosave=False: this builds a real app, and a test must not overwrite the
    # user's saved session.
    app = LabelerApp(root, autosave=False)

    stats0 = config_io.statistics_from_json(app.profiles[0]["statistics"])
    assert not stats0["relevance"], "new labeler profiles default relevance off"
    assert app.rerun_btn.cget("text") == "update region simplification"
    assert app.show_pred_var.get(), "classification rendering is on by default"

    # Window layout: a tabbed center. The left pane keeps the profile PICKER,
    # the session and Run; the profile is EDITED on the Processing tab (two
    # columns); the View tab is `app.right`; the model kind lives on the Model
    # tab and is gone from the classifier panel.
    def _under(w, ancestor):
        while w is not None:
            if w is ancestor:
                return True
            w = getattr(w, "master", None)
        return False

    tabs = [str(app.center.tab(t, "text")) for t in app.center.tabs()]
    assert tabs == list(_CENTER_TABS) == ["Processing", "View", "Model", "Analysis"], tabs
    assert app.center.nametowidget(app.center.select()) is app.right, \
        "the View tab is selected at start"
    assert app._center_tab_name() == "View"
    for w in (app.filters_frame, app.base_frame, app.msc_frame, app.stats_frame,
              app.profile_load_btn):
        assert _under(w, app.processing_tab), w
    assert app.filters_frame.master is app.base_frame.master is app.proc_col_a
    assert app.msc_frame.master is app.stats_frame.master is app.proc_col_b
    assert _under(app.profile_combo, app.left) and _under(app.run_btn, app.left_pane)
    assert not _under(app.profile_load_btn, app.left), "profile tools moved"
    # Left pane: Run pinned at the bottom (packed first, side=bottom), the
    # session's three lists as the panes of a vertical paned window.
    assert _under(app.run_frame, app.left_bottom) and not _under(app.run_btn, app.left)
    assert app.left_bottom.pack_info()["side"] == "bottom"
    assert app.left_pane.pack_slaves()[0] is app.left_bottom, "Run claims its space first"
    assert len(app.session_paned.panes()) == 3
    assert [str(app._session_groups[n]) for n in ("folders", "files", "sequences")] == \
        [str(p) for p in app.session_paned.panes()]
    assert _under(app.folder_list, app._session_groups["folders"])
    assert _under(app.file_list, app._session_groups["files"])
    assert _under(app.subseq_list, app._session_groups["sequences"])
    for btn, lst in ((app.folder_btn_row, app.folder_list), (app.make_seq_btn, app.file_list),
                     (app.seq_btn_row, app.subseq_list)):
        slaves = lst.master.master.pack_slaves()
        assert btn.pack_info()["side"] == "bottom"
        assert slaves.index(btn) < slaves.index(lst.master), "buttons packed before the list"
        assert lst.master.pack_info()["expand"] in (1, "1", True)
    assert _under(app.model_kind_combo, app.model_tab)
    assert not _under(app.model_kind_combo, app.label_pane)
    ml_rows = [c for row in app.confusion_holder.master.winfo_children()
               for c in row.winfo_children()]
    assert not any(isinstance(c, ttk.Combobox) for c in ml_rows), \
        "no kind combobox left in the classifier panel"
    # The notebook never binds plain Tab, so the overlay hotkey still fires
    # (and still consumes the key) from every tab.
    assert "<Key-Tab>" not in app.center.bind_class("TNotebook")
    assert app._on_tab_toggle() == "break"
    assert not app.show_overlay_var.get()
    app._on_tab_toggle()
    assert app.show_overlay_var.get()
    # The architecture readout follows the kind, formatted from the constants
    # _make_model builds with.
    for kind in _MODEL_KINDS:
        app.model_kind_var.set(kind)
        assert app.model_arch_var.get() == app._model_description_now(kind), kind
    assert str(_FOREST_TREES) in _model_description("random forest")
    assert str(_MLP_HIDDEN) in _model_description("dense FC")
    assert str(_DENSE_TOP_N["dense-top-16"]) in _model_description("dense-top-16")
    app.model_kind_var.set("dense FC")

    # Toolbar hints: the selected workflow as a compact chain and the active
    # model, each a click away from its tab.
    app._refresh_hints()
    assert app.workflow_hint_var.get().splitlines() == \
        ["topo field: base→msc(asc, 10%)", "stats: base→1ch×4"], app.workflow_hint_var.get()
    assert app.model_hint_var.get() == "model: dense FC · not trained"
    app._on_filter_op_change(0, "blur"); app.filter_cards[0]["params"]["sigma"] = 1.5
    app._on_filter_op_change(1, "edges"); app.filter_cards[1]["params"]["sigma"] = 0.7
    app._refresh_hints()
    assert "base→b(1.5)→e(0.7)" in app.workflow_hint_var.get(), app.workflow_hint_var.get()
    app.filter_cards = [app._new_filter_card()]
    app._rebuild_filter_cards()
    app._refresh_hints()
    assert app.workflow_hint_var.get().splitlines()[0] == "topo field: base→msc(asc, 10%)"
    app._show_center_tab("Processing")
    assert app._center_tab_name() == "Processing"
    app._show_center_tab("Model")
    assert app._center_tab_name() == "Model"
    app._show_center_tab("nope")
    assert app._center_tab_name() == "Model", "an unknown name is ignored"
    app.center.select(app.right)
    app._clf, app._clf_names, app._clf_kind = object(), ["a", "b"], "random forest"
    assert app._model_hint_text().startswith("model: random forest · 2 feats")
    app._clf = None
    app._refresh_hints()
    assert app.model_hint_var.get() == "model: dense FC · not trained"
    # Placement: the workflow hint heads the Run section; the model hint
    # heads the classifier section, above Train/Classify.
    assert app.workflow_hint.master is app.run_frame
    assert app.run_frame.pack_slaves()[0] is app.workflow_hint
    ml = app.confusion_holder.master
    assert app.model_hint.master is ml and ml.pack_slaves()[0] is app.model_hint
    train_row = ml.pack_slaves()[1]
    assert any(isinstance(w, ttk.Button) and str(w.cget("text")).startswith("Train")
               for w in train_row.winfo_children()), "Train/Classify right under the hint"

    # `app.right` IS the View tab, so the scan below reads unchanged.
    right_rows = list(app.right.winfo_children())
    assert not any(isinstance(w, ttk.LabelFrame)
                   and str(w.cget("text")) == "Live parameters"
                   for w in right_rows), "live controls must not have a subpanel"
    direct_labels = [child for row in right_rows
                     for child in row.winfo_children()
                     if isinstance(child, ttk.Label)]
    assert any(str(w.cget("text")) == "Overlay alpha:" for w in direct_labels)
    direct_checks = [child for row in right_rows
                     for child in row.winfo_children()
                     if isinstance(child, ttk.Checkbutton)]
    assert any(str(w.cget("text")) == "Show Classification"
               for w in direct_checks)
    assert any(str(w.cget("text")) == "Show GT" for w in direct_checks)
    overlay_children = list(app.overlay_master_check.master.winfo_children())
    master_idx = overlay_children.index(app.overlay_master_check)
    assert isinstance(overlay_children[master_idx + 1], ttk.Separator), \
        "a vertical separator must follow the overlay master"
    assert overlay_children[master_idx + 2] is app.show_regions_check
    app.vmin_var.set(0.2); app.vmax_var.set(0.8)
    assert app._image_window("base") == (0.2, 0.8)
    assert app._image_window("filtered") == (0.2, 0.8)
    assert app._image_window("edges_s1") == (0.2, 0.8)
    app.vmin_var.set(0.0); app.vmax_var.set(1.0)

    # The labeler never needs more than the per-slice tier.
    assert app._needed_level() == "slice", "regions view must stay on the slice tier"
    app.show_regions_var.set(False); app._on_regions_toggle()
    assert app._needed_level() == "slice"
    app.show_regions_var.set(True); app._on_regions_toggle()

    # Fake one primed slice: 4 blocks with SPARSE living ids, -1 border. The
    # session has one (nonexistent) folder; slice identity is folder-qualified.
    lab = np.full((20, 20), -1, np.int32)
    lab[2:10, 2:10] = 0; lab[2:10, 10:18] = 2
    lab[10:18, 2:10] = 5; lab[10:18, 10:18] = 9
    data_dir = r"C:\labdata"
    files = [os.path.join(data_dir, "s0.tiff")]
    app.folders = [{"path": data_dir, "name": "data"}]
    app.active_folder_idx = 0
    app._refresh_folder_list()
    app.subsequences = [{"name": "seq1", "folder": "data", "files": files}]
    zeros = np.zeros((20, 20), np.float32)
    app.primed = [{"files": files, "base": [zeros], "filtered": [zeros],
                   "pipes": [None], "normalizers": [[]]}]
    app._rebuild_flat_slices()
    from .common import FeatureTable
    table = FeatureTable(["feature_id", "area", "mean_base"],
                         np.array([[0.0, 64.0, 1.5], [2.0, 80.0, 2.5],
                                   [5.0, 96.0, 3.5], [9.0, 112.0, 4.5]]))
    rec = {"commit": app._commit_id, "labels": lab, "stats": table,
           "kept": set(), "cc": None, "n_feat": 4}
    app._slices[(0, 0)] = rec

    # Gestures commit through the same path the DrawController uses.
    app.active_class_var.set(1)
    app._commit_interaction("box", [(3.0, 3.0), (16.0, 16.0)])      # all 4 -> 1
    app.active_class_var.set(2)
    app._commit_interaction("squiggle", [(3.0, 5.0), (15.0, 5.0)])  # {0,2} -> 2
    assert [it.uid for it in app.store.interactions] == [1, 2]
    assert app.store.interactions[0].slice_key == "data/s0.tiff", \
        "slice identity is folder-qualified"
    # The sequence tree's columns track priming + per-slice annotations.
    app._refresh_subseq_list()
    assert tuple(app.subseq_list.item("q0:0", "values")) == ("Y", "2")
    assert tuple(app.subseq_list.item("q0", "values")) == ("Y", "2")

    # The class panels list ON-SLICE interactions only (they swap with the
    # slice); the titles carry ALL-slice annot/region totals.
    other = app.store.add("squiggle", [(0.0, 0.0)], 1, "data/other.tiff")
    app._rebuild_class_panels()
    assert len(app._visible_interactions()) == 2, "on-slice interactions only"
    t1 = str(app._class_title_labels[1].cget("text"))
    assert t1 == "1 · 2a · 2r", t1            # box + off-slice; {5,9}
    t2 = str(app._class_title_labels[2].cget("text"))
    assert t2 == "2 · 1a · 2r", t2            # squiggle; {0,2}
    app.store.remove(other.uid)
    app._rebuild_class_panels()

    # User-picked class colors: reach the LUT (rev bump rebuilds the cache),
    # ride the store, and undo like any other edit.
    app._push_history()
    app.store.set_color(1, "#123456")
    app._rebuild_class_panels()
    assert app._class_color_hex(1) == "#123456"
    lut_c = app._class_lut_for(0, 0, rec, np)
    assert tuple(lut_c[5]) == (0x12, 0x34, 0x56, 255), "picked color in the LUT"
    assert app.store.to_json()["classes"][0]["color"] == "#123456"
    app._undo()
    assert app._class_color_hex(1) != "#123456", "color change is undoable"

    # The swatch is the arm control (the per-class "draw" radiobutton is gone),
    # and the trace rings exactly the armed one.
    for k, frame in app._class_panels.items():
        assert not [w for w in frame.winfo_children()
                    if w.winfo_class() == "Radiobutton"], \
            f"class {k} still has a draw radiobutton"
    app._class_swatches[2].event_generate("<ButtonRelease-1>")
    assert app.active_class_var.get() == 2, "swatch left-click arms the class"
    armed = [k for k, w in app._class_swatches.items()
             if str(w.cget("relief")) == "sunken"]
    assert armed == [2], armed
    app.active_class_var.set(1)          # the trace, not the swatch, repaints
    armed = [k for k, w in app._class_swatches.items()
             if str(w.cget("relief")) == "sunken"]
    assert armed == [1], armed
    app._rebuild_class_panels()          # a rebuild restores the ring
    assert str(app._class_swatches[1].cget("relief")) == "sunken"

    # Resolution + LUT: later interaction painted over the earlier one.
    lut = app._class_lut_for(0, 0, rec, np)
    assert lut is not None and lut.shape == (10, 4)
    assert tuple(lut[0]) == labeling.CLASS_COLORS[2]     # repainted by squiggle
    assert tuple(lut[2]) == labeling.CLASS_COLORS[2]
    assert tuple(lut[5]) == labeling.CLASS_COLORS[1]     # box only
    assert tuple(lut[9]) == labeling.CLASS_COLORS[1]
    assert lut[1, 3] == 0, "id 1 is not a living id -> transparent"

    # Memoization keys on (commit, store.rev).
    assert app._class_lut_for(0, 0, rec, np) is lut, "cache hit expected"
    app._move_interaction(2, 2)          # same class: no rev bump, still cached
    assert app._class_lut_for(0, 0, rec, np) is lut
    app.store.set_class(1, 2)            # mutation -> rev bump -> rebuilt
    lut2 = app._class_lut_for(0, 0, rec, np)
    assert lut2 is not lut and tuple(lut2[9]) == labeling.CLASS_COLORS[2]
    app.engine.commit_selection()        # Rerun path: fresh commit, fresh labels
    rec2 = dict(rec, commit=app._commit_id)
    lut3 = app._class_lut_for(0, 0, rec2, np)
    assert lut3 is not lut2, "a new commit must re-resolve"

    # The overlay stack: dimmed region layer under the opaque class layer.
    from msseg.viz import min_colors
    ovs = app._seg_overlays(0, 0, rec2, None, np, min_colors)
    assert len(ovs) == 2
    assert int(ovs[0]["lut"][:, 3].max()) <= _REGION_ALPHA
    assert int(ovs[1]["lut"][:, 3].max()) == 255
    app.show_gt_var.set(False)
    assert len(app._seg_overlays(0, 0, rec2, None, np, min_colors)) == 1, \
        "Show GT controls the drawn-label layer independently"
    app.show_gt_var.set(True)
    # The master overlay switch (Tab) blanks the whole stack.
    app._on_tab_toggle()
    assert not app.show_overlay_var.get()
    assert all(str(w.cget("state")) == "disabled"
               for w, _state in app._overlay_dependents)
    assert app._seg_overlays(0, 0, rec2, None, np, min_colors) == []
    app._on_tab_toggle()
    assert app.show_overlay_var.get()
    assert str(app.region_mode_combo.cget("state")) == "readonly"
    assert all(str(w.cget("state")) == state
               for w, state in app._overlay_dependents)

    # Class-count change clamps orphans; arm state resets when it vanishes.
    app.active_class_var.set(2)
    app.n_classes_var.set(2)
    app._on_n_classes_change()
    assert all(it.class_id == 1 for it in app.store.interactions)
    assert app.active_class_var.get() == 0

    # Session round-trip: the store rides the v2 session doc, and so does
    # the selected center tab (by name).
    app.center.select(app.model_tab)
    sdoc = app._session_doc()
    assert sdoc["view"]["center_tab"] == "Model"
    app.center.select(app.right)
    assert "labels" not in sdoc, "the gesture geometry is 'annotations' now"
    assert sdoc["annotations"]["n_classes"] == 2
    assert len(sdoc["annotations"]["interactions"]) == 2
    assert sdoc["sequences"][0]["folder"] == "data"
    app.store = LabelStore()             # clobber
    app._apply_session_doc(sdoc, "test")
    assert app._center_tab_name() == "Model", "the center tab restores by name"
    app.center.select(app.right)
    assert len(app.store.interactions) == 2
    assert all(it.bound for it in app.store.interactions), \
        "rebind by folder-qualified key"

    # ...and a session written BEFORE the rename still restores: the key moved,
    # the document under it did not.
    legacy = {k: v for k, v in sdoc.items() if k != "annotations"}
    legacy["labels"] = sdoc["annotations"]
    legacy["view"] = dict(sdoc["view"], center_tab="bogus")   # unknown -> untouched
    app.store = LabelStore()
    app._apply_session_doc(legacy, "legacy key")
    assert app._center_tab_name() == "View", "an unknown tab name is ignored"
    assert len(app.store.interactions) == 2, \
        "a pre-rename session's 'labels' key still loads"

    # Export writes annotations.json alongside the config(s).
    with tempfile.TemporaryDirectory() as td:
        paths = app._write_configs(td)
        assert os.path.basename(paths[-1]) == "annotations.json"
        with open(paths[-1], encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["version"] == 2 and len(doc["interactions"]) == 2

    # The round-trip's apply reset the engine; re-fake the primed slice
    # for the interactive-behavior checks below.
    app.primed = [{"files": files, "base": [zeros], "filtered": [zeros],
                   "pipes": [None], "normalizers": [[]]}]
    app._rebuild_flat_slices()
    rec3 = dict(rec, commit=app._commit_id)
    app._slices[(0, 0)] = rec3

    # Undo/redo: snapshots cover adds, class moves, and class-count changes;
    # a fresh edit wipes the redo branch.
    n0 = len(app.store.interactions)
    app.active_class_var.set(1)
    app._commit_interaction("squiggle", [(12.0, 12.0)])        # single-tap gesture
    assert len(app.store.interactions) == n0 + 1
    assert app.store.interactions[-1].points == [(12.0, 12.0)]
    app._undo()
    assert len(app.store.interactions) == n0, "undo removes the tap"
    app._redo()
    assert len(app.store.interactions) == n0 + 1, "redo restores it"
    app._undo()
    app._commit_interaction("squiggle", [(5.0, 12.0)])
    assert not app._redo_stack, "a new edit wipes the redo branch"
    app._undo()
    assert len(app.store.interactions) == n0
    app.n_classes_var.set(3)
    app._on_n_classes_change()
    uid0 = app.store.interactions[0].uid
    app._move_interaction(uid0, 2)
    assert app.store.get(uid0).class_id == 2
    app._undo()
    assert app.store.get(uid0).class_id == 1, "undo covers class moves"
    app._undo()
    assert app.store.n_classes == 2, "undo covers class-count changes"

    # Hover shows the gesture's geometry; a click recenters at constant zoom.
    if app.viewer is not None:
        app._show_interaction_geometry(uid0)
        assert app.viewer.canvas.find_withtag("ihover"), "hover geometry drawn"
        app._hide_interaction_geometry()
        assert not app.viewer.canvas.find_withtag("ihover")
        # A view change (zoom/pan) must re-project the geometry, not drop it.
        app._show_interaction_geometry(uid0)
        app._redraw_hover_geometry()
        assert app.viewer.canvas.find_withtag("ihover"), \
            "geometry survives a view change"
        app._hide_interaction_geometry()
        s0 = app.viewer.scale
        app._on_row_click(uid0)
        assert app.viewer.scale == s0, "recenter keeps the zoom level"
        it0 = app.store.get(uid0)
        cx = sum(x for x, _y in it0.points) / len(it0.points)
        w = max(app.viewer.canvas.winfo_width(), 1)
        assert abs(app.viewer.view_x - (cx - (w / 2) * s0)) < 1e-6
        # Hovering a LABELED pixel draws every gesture touching its region;
        # a background pixel clears again.
        app._on_hover(5, 5)                  # inside region 0
        assert app.viewer.canvas.find_withtag("ihover"), "pixel hover draws gestures"
        app._on_hover(0, 0)                  # -1 background
        assert not app.viewer.canvas.find_withtag("ihover")

        # The persistent annotation view: every on-slice gesture at once, on
        # its own tag so an overlay repaint (which drops "ihover") keeps it.
        assert not app.viewer.canvas.find_withtag("ipersist")
        app.show_annot_var.set(True)
        app._refresh_annotation_layer()
        n_persist = len(app.viewer.canvas.find_withtag("ipersist"))
        assert n_persist >= len(app._visible_interactions()) > 0, n_persist
        from msseg.viz import min_colors as _mc0
        app._seg_overlays(0, 0, rec, None, np, _mc0)     # drops "ihover" only
        assert len(app.viewer.canvas.find_withtag("ipersist")) == n_persist
        app._redraw_hover_geometry()                     # zoom/pan re-projects
        assert len(app.viewer.canvas.find_withtag("ipersist")) == n_persist
        app.show_annot_var.set(False)
        app._refresh_annotation_layer()
        assert not app.viewer.canvas.find_withtag("ipersist")

        # Right-CLICK on the image plane resolves the annotation under it;
        # a right-DRAG is a pan, so it must NOT reach the menu.
        assert app._interaction_at(5, 5) is not None, "region 0 is annotated"
        assert app._interaction_at(0, 0) is None, "background offers no menu"
        popped = []
        app._interaction_menu = lambda e, uid: popped.append(uid)
        canvas = app.viewer               # the SliceCanvas, not its tk.Canvas
        canvas.on_context = app._canvas_menu
        # Pin the view so screen == image: a withdrawn root has a 1x1 canvas,
        # and _on_row_click just recentred on that.
        canvas.view_x, canvas.view_y, canvas.scale = 0.0, 0.0, 1.0
        sx, sy = 5, 5                        # inside region 0
        want = app._interaction_at(*canvas.screen_to_image(sx, sy))
        assert want is not None, canvas.screen_to_image(sx, sy)
        canvas._context_press(_FakeEvent(sx, sy))
        canvas._context_release(_FakeEvent(sx + 40, sy))     # dragged -> pan
        assert not popped, "a right-drag pans, it does not open a menu"
        canvas._context_press(_FakeEvent(sx, sy))
        canvas._context_release(_FakeEvent(sx + 1, sy))      # a click
        assert popped == [want], (popped, want)
        del app._interaction_menu                            # back to the real one
        assert app.viewer.on_context is not None

    # Export as CSV: one row per living region (class 0 = unlabeled), carrying
    # every statistics column the spec produced for the region.
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "labels.csv")
        covered, skipped = app._write_labels_csv(csv_path)
        assert (covered, skipped) == (1, 0)
        with open(csv_path, encoding="utf-8") as f:
            rows = f.read().strip().splitlines()
        assert rows[0] == "slice,region_id,class,predicted,area,mean_base"
        assert len(rows) == 5
        assert rows[1] == "data/s0.tiff,0,1,,64,1.5"   # blank pre-classify
        assert rows[4] == "data/s0.tiff,9,1,,112,4.5"

    # Train + classify (skipped when scikit-learn isn't installed).
    try:
        import sklearn  # noqa: F401
        have_sklearn = True
    except ImportError:
        have_sklearn = False
        print("labeler selftest: scikit-learn not installed - classifier "
              "checks skipped")
    if have_sklearn:
        from msseg.viz import min_colors as _mc
        from unittest import mock as _mock
        # The fake FeatureTable's schema is not the real profile's, so pin the
        # expected-feature source to it (the real _expected_feature_names is a
        # pure schema call; the check logic below is what's under test).
        app._expected_feature_names = lambda: ["area", "mean_base"]
        assert app.model_kind_var.get() == "dense FC", "dense FC is the default"
        app.model_kind_var.set("random forest")
        app.n_classes_var.set(3)
        app._on_n_classes_change()
        app.active_class_var.set(2)
        app._commit_interaction("squiggle", [(12.0, 12.0), (15.0, 15.0)])  # {9} -> 2

        # Model operations are stack-wide even when only the visible slice has
        # a cached record. Simulate a Rerun (all old records stale), then let a
        # fake synchronous materializer stand in for the compiled pipe.
        file1 = os.path.join(data_dir, "s1.tiff")
        pair_files = [files[0], file1]
        app.subsequences[0]["files"] = pair_files
        app.primed = [{"files": pair_files, "base": [zeros, zeros],
                       "filtered": [zeros, zeros], "pipes": [None, None],
                       "normalizers": [[], []]}]
        app._rebuild_flat_slices()
        second_uids = [
            app.store.add("box", [(3.0, 3.0), (16.0, 16.0)], 1,
                          "data/s1.tiff", 0, 1).uid,
            app.store.add("squiggle", [(12.0, 12.0), (15.0, 15.0)], 2,
                          "data/s1.tiff", 0, 1).uid,
        ]
        original_ensure = app._ensure_slice_record
        app.engine.commit_selection()
        materialized = []
        templates = [rec3, rec3]

        def _materialize(si, li):
            materialized.append((si, li))
            fresh = dict(templates[li], commit=app._commit_id)
            app._slices[(si, li)] = fresh
            return fresh

        app._ensure_slice_record = _materialize
        app._train_classifier()
        assert app._clf is not None, "training must produce a model"
        assert app._clf_kind == "random forest"
        assert "min_x" not in app._clf_names and "ext_x" not in app._clf_names
        assert set(materialized) == {(0, 0), (0, 1)}, \
            "training must materialize every primed slice after a new commit"
        assert "on 8 labeled regions" in app.status_var.get(), app.status_var.get()
        rec3 = app.engine.record(0, 0)
        materialized.clear()
        app._classify()
        pr = app._pred.get((0, 0))
        pr_second = app._pred.get((0, 1))
        assert pr is not None and pr[0] == app._commit_id
        assert pr_second is not None and pr_second[0] == app._commit_id
        assert set(materialized) == {(0, 0), (0, 1)}, \
            "classification must score every primed slice"
        assert pr[1].shape == (10,)
        assert set(int(v) for v in pr[1][[0, 2, 5, 9]]) <= {1, 2}
        assert app.show_pred_var.get()
        if app.viewer is not None:
            assert app.viewer._hud_mode is None, "computing badge cleared"
        ovs = app._seg_overlays(0, 0, rec3, None, np, _mc)
        assert len(ovs) == 3, "regions + prediction layer + drawn labels"
        assert int(ovs[1]["lut"][:, 3].max()) < 255, "prediction layer is translucent"

        # Probabilities ride the same cache; the hard label IS their argmax.
        assert len(pr) == 3, "_pred caches (commit, region_class, region_proba)"
        proba = pr[2]
        assert proba.shape == (10, labeling.MAX_CLASSES)
        for r in (0, 2, 5, 9):
            assert abs(float(proba[r].sum()) - 1.0) < 1e-5, r
            assert int(proba[r].argmax()) == int(pr[1][r]), "hard label = argmax"
        assert float(proba[1].sum()) == 0.0, "id 1 is not a living region"
        app._hover_ctx = {"si": 0, "li": 0, "base": zeros,
                          "filt": zeros, "data": None}
        app._on_hover(3, 3)
        hover = app.hover_var.get()
        assert "P(class 1)=" in hover and "P(class 2)=" in hover, hover
        assert all(name not in hover for name in ("MSC=", "CC=", "global=", "mask=")), hover
        app._on_hover(0, 0)
        assert "class probabilities: -" in app.hover_var.get()

        # Regions coloring modes appear only once probabilities exist.
        app._refresh_region_modes()
        modes = list(app.region_mode_combo.cget("values"))
        assert modes[0] == "label id" and "uncertainty" in modes, modes
        assert "P(class 1)" in modes and "P(class 2)" in modes, modes
        for mode in ("P(class 1)", "uncertainty"):
            app.region_mode_var.set(mode)
            ovs_m = app._seg_overlays(0, 0, rec3, None, np, _mc)
            # The scalar layer REPLACES the id layer, so the stack is the same
            # height; it is the bottom one, and only living regions are opaque.
            assert len(ovs_m) == 3, mode
            lut_m = ovs_m[0]["lut"]
            assert lut_m.shape == (10, 4)
            assert int(lut_m[[0, 2, 5, 9], 3].min()) > 0, mode
            assert int(lut_m[1, 3]) == 0, "unscored region stays invisible"
        app.region_mode_var.set("label id")
        app._pred.clear()
        app._refresh_region_modes()
        assert list(app.region_mode_combo.cget("values")) == ("label id",) or \
            list(app.region_mode_combo.cget("values")) == ["label id"], \
            "modes collapse when the cache is dropped"
        app._classify()
        pr = app._pred.get((0, 0))

        # -- "-> edges" kinds: an edge model on top of a dense base --------- #
        # The two fake slices carry the block raster (ids 0,2 -> class 2 and
        # 5,9 -> class 1), so pixel adjacency gives four arcs per slice: two
        # same-class, two crossing. Train fits the custom base, then the edges.
        kind_before = app.model_kind_var.get()
        assert _CUSTOM_KIND in _MODEL_KINDS and _CUSTOM_EDGE_KIND in _MODEL_KINDS
        assert _base_kind(_CUSTOM_EDGE_KIND) == _CUSTOM_KIND
        assert _edge_kind_of(_TUNED_KIND) == _TUNED_EDGE_KIND and _edge_kind_of("dense FC") is None
        assert _under(app.edge_eval_btn, app.model_tab) and _under(app.edge_tree, app.model_tab)
        assert _under(app.edge_readout, app.confusion_holder.master)
        app.custom_hidden_var.set("4")
        app.model_kind_var.set(_CUSTOM_EDGE_KIND)
        assert "not trained yet" in app.model_arch_var.get(), app.model_arch_var.get()
        assert "not trained" in app.edge_readout_var.get()
        app._train_classifier()
        assert app._clf_kind == _CUSTOM_EDGE_KIND and app._clf_spec is not None
        assert tuple(app._clf_spec.hidden) == (4,)
        edge = app._edge_model
        assert edge is not None, app.status_var.get()
        assert edge.n_edges == 8 and edge.n_diff == 4, (edge.n_edges, edge.n_diff)
        assert not edge.used_saddle and edge.width == 4 and edge.n_in == 10
        assert "-> edges" in app.status_var.get(), app.status_var.get()
        assert "-> edges" in app.model_strip_var.get() and "-> edges" in app.model_hint_var.get()
        assert "edges: logistic" in app.model_arch_var.get()
        app._classify()
        pr = app._pred.get((0, 0))
        assert len(pr) == 4, "the cache entry grows an aux dict with an edge model"
        aux = app._pred_aux(pr)
        assert aux is not None and aux["pdiff"].shape == (4,) and np.isfinite(aux["pdiff"]).all()
        assert aux["source"] == "pixels" and aux["active"] and aux["raw"].shape == (10,)
        assert set(int(v) for v in pr[1][[0, 2, 5, 9]]) <= {1, 2}
        assert int(pr[1][1]) == 0, "a non-living id stays class 0 after voting"
        assert "voting on" in app.edge_readout_var.get() and "flipped" in app.edge_readout_var.get()
        # Coloring modes for the edge model.
        app._refresh_region_modes()
        modes = list(app.region_mode_combo.cget("values"))
        assert _MODE_FLIPPED in modes and _MODE_PDIFF in modes, modes
        for mode in (_MODE_FLIPPED, _MODE_PDIFF):
            app.region_mode_var.set(mode)
            vals, mask = app._region_scalar(pr, np)
            assert vals.shape == (10,) and bool(mask[0]) and not bool(mask[1]), mode
        app.region_mode_var.set(_MODE_PDIFF)
        vals, _m = app._region_scalar(pr, np)
        touching0 = [float(aux["pdiff"][i]) for i in range(4)
                     if 0 in (int(aux["la"][i]), int(aux["lb"][i]))]
        assert np.isclose(float(vals[0]), max(touching0))
        app.region_mode_var.set("label id")
        # N flips to the base kind: raw predictions, voting off, no forward pass.
        app._on_edge_key()
        assert app.model_kind_var.get() == _CUSTOM_KIND
        pr0 = app._pred.get((0, 0))
        assert np.array_equal(pr0[1], app._pred_aux(pr0)["raw"]) and not app._pred_aux(pr0)["active"]
        assert "voting off" in app.edge_readout_var.get()
        app._on_edge_key()
        assert app.model_kind_var.get() == _CUSTOM_EDGE_KIND
        assert app._pred_aux(app._pred[(0, 0)])["active"]
        # lambda 0 is the base answer even with voting on; the entry re-votes in place.
        app.edge_lam_var.set("0")
        app._on_edge_settings_change()
        pr0 = app._pred[(0, 0)]
        assert np.array_equal(pr0[1], app._pred_aux(pr0)["raw"]) and app._pred_aux(pr0)["lam"] == 0.0
        app.edge_lam_var.set("1")
        app._on_edge_settings_change()
        # The learned magic metric: the flood reads p(diff) from the cache and
        # is refused without it.
        tool = app.viewer.tool if app.viewer is not None else None
        if tool is not None:
            app.magic_metric_var.set("learned")
            tool_before, cls_before = app.tool_var.get(), app.active_class_var.get()
            app.tool_var.set("magic")
            app.active_class_var.set(1)
            assert tool.on_press(_FakeEvent(5, 5)), app.status_var.get()
            assert tool.magic._s["ladder"].metric == "learned"
            assert tool.magic._s["ladder"].n_reach == 4
            assert tool.magic.cancel()
            saved_pred = app._pred
            app._pred = {}
            assert not tool.on_press(_FakeEvent(5, 5))
            assert "Classify" in app.status_var.get(), app.status_var.get()
            app._pred = saved_pred
            app.magic_metric_var.set("mean")
            app.tool_var.set(tool_before)
            app.active_class_var.set(cls_before)
        # Freeze base: Train keeps the base net and refits only the edges.
        clf_id = id(app._clf)
        app.freeze_base_var.set(True)
        app._train_classifier()
        assert id(app._clf) == clf_id and app._edge_model is not None and app._edge_model is not edge
        assert "(base frozen)" in app.status_var.get(), app.status_var.get()
        app.freeze_base_var.set(False)
        # Pickle v4 carries the edge model; the session record flags it.
        with tempfile.TemporaryDirectory() as td:
            p4 = os.path.join(td, "edges.pkl")
            app._save_classifier_to(p4)
            assert app.models[-1]["edge"] is True and app.models[-1]["kind"] == _CUSTOM_EDGE_KIND
            sdoc4 = session.session_doc_from_json(app._session_doc())
            assert sdoc4["models"][-1]["edge"] is True
            app._edge_model = None
            app.model_kind_var.set("dense FC")
            app._load_classifier_from(p4)
            assert app._edge_model is not None and app._clf_kind == _CUSTOM_EDGE_KIND
            assert app.model_kind_var.get() == _CUSTOM_EDGE_KIND
            assert app.custom_hidden_var.get() == "4"
        # Evaluate edges runs inline; with two slices every fold is stratified
        # inside them, so no fold has training edges and all are skipped -- the
        # report still lands and the buttons come back.
        app._evaluate_edges(sync=True)
        assert app._search is None and app.edge_progress_var.get(), app.edge_progress_var.get()
        assert str(app.edge_eval_btn.cget("state")) == "normal"
        # The stack settings ride the session view and are validated on the way in.
        v = app._view_state()["neighbours"]
        assert v["custom_hidden"] == "4" and v["edge_spec"]["lam"] == 1.0 and v["freeze_base"] is False
        app._apply_neighbours_view({"custom_hidden": "8-4", "freeze_base": True,
                                    "edge_spec": {"lam": 0.5, "rounds": 2, "features": ["absdiff"]}})
        assert app._custom_hidden() == (8, 4) and app.freeze_base_var.get()
        assert app._edge_spec_from_ui().lam == 0.5 and app._edge_spec_from_ui().features == ("absdiff",)
        app._apply_neighbours_view({"custom_hidden": "bad!"})
        assert app.custom_hidden_var.get() == "8-4", "an unreadable ladder is ignored"
        app.freeze_base_var.set(False)
        app.custom_hidden_var.set(_DEFAULT_CUSTOM_HIDDEN)
        app._apply_edge_spec(edge_model.EdgeSpec())
        # The region list behind a confusion cell, and double-click navigation.
        app._classify()                                # the pickle load cleared _pred
        app._refresh_confusion()
        counts = app._cm_counts["all"]
        cell = max(counts, key=counts.get)
        app._on_confusion_open(*cell)
        assert app._center_tab_name() == "Analysis" and app._cm_cell == cell
        assert len(app.errors_tree.get_children()) == counts[cell] == len(app._error_rows)
        assert str(cell[0]) in app.errors_header_var.get()
        assert all(d["true"] == cell[0] and d["pred"] == cell[1] for d in app._error_rows)
        first = app._error_rows[0]
        assert first["area"] >= app._error_rows[-1]["area"], "largest first within a slice"
        app.errors_tree.selection_set("0")
        app._on_error_row_open()
        app.root.update_idletasks()                    # the deferred centring
        assert app._center_tab_name() == "View"
        assert app._current() == (first["si"], first["li"])
        assert app._hover_key == (first["si"], first["li"], first["region"])
        assert app.status_var.get().startswith(f"region {first['region']}")
        assert not app._goto_region(7, 7, 0), "an unprimed slice is refused"
        app._on_confusion_click(*cell)              # toggles the cell off
        assert app._cm_cell is None and not app.errors_tree.get_children()
        assert "No confusion cell" in app.errors_header_var.get()
        # Back to the plain kind: the edge model goes with the base it sat on.
        app.model_kind_var.set(kind_before)
        app._train_classifier()
        app._classify()
        pr = app._pred.get((0, 0))
        assert app._edge_model is None and len(pr) == 3

        # Confusion matrix: frozen predictions against live labels.
        app._refresh_confusion()
        counts = dict(app._cm_counts["current"])
        counts_all = dict(app._cm_counts["all"])
        truth = app._truth_from_cache(0, 0, rec3, np)
        expect = {}
        for r in (0, 2, 5, 9):
            t, p = int(truth[r]), int(pr[1][r])
            if t >= 1 and p >= 1:
                expect[(t, p)] = expect.get((t, p), 0) + 1
        assert counts == expect, (counts, expect)
        assert sum(counts.values()) == 4, counts
        assert sum(counts_all.values()) == 8, counts_all
        assert str(app._cm_cells["current"][(1, 1)].cget("text")) == \
            str(counts.get((1, 1), 0))
        assert str(app._cm_cells["all"][(1, 1)].cget("text")) == \
            str(counts_all.get((1, 1), 0))
        # A new label moves the TRUE axis without re-Classify.
        moved = next(iter(expect))
        app.active_class_var.set(3 if app.store.n_classes > 3 else 2)
        before = {scope: dict(values) for scope, values in app._cm_counts.items()}
        assert before["current"], before
        app._commit_interaction("squiggle", [(2.0, 2.0)])     # repaints id 0
        assert app._cm_counts["current"] != before["current"], \
            "labeling moves the matrix with the predictions frozen"
        assert sum(app._cm_counts["current"].values()) == \
            sum(before["current"].values()), \
            "the same regions, redistributed across the true axis"
        assert sum(app._cm_counts["all"].values()) == sum(before["all"].values()), \
            "the global table retains every slice while truth moves"
        assert app._pred.get((0, 0))[1] is pr[1], "predictions stay frozen"
        app._undo()
        assert app._cm_counts == before, "undo restores both tables"

        # Clicking either shared cell highlights the same regions/current cell
        # in both grids and reports current/global counts.
        app._on_confusion_click(*moved)
        assert app._cm_cell == moved
        assert app._cm_cells["all"][moved].cget("background") == "#cde8ff"
        assert app._cm_cells["current"][moved].cget("background") == "#cde8ff"
        hits = app._confusion_hits()
        assert hits == {r for r in (0, 2, 5, 9)
                        if (int(truth[r]), int(pr[1][r])) == moved}, hits
        assert (f"{before['current'].get(moved, 0)} on this slice / "
                f"{before['all'].get(moved, 0)} total") in app.status_var.get()
        ovs_h = app._seg_overlays(0, 0, rec3, None, np, _mc)
        assert len(ovs_h) == 4, "highlight rides on top"
        assert set(np.nonzero(ovs_h[-1]["lut"][:, 3])[0].tolist()) == hits
        app._on_confusion_click(*moved)                       # click again clears
        assert app._cm_cell is None
        assert len(app._seg_overlays(0, 0, rec3, None, np, _mc)) == 3

        # Navigating changes only the current-slice table; the all-slice table
        # remains the aggregate over both records.
        all_before_nav = dict(app._cm_counts["all"])
        app._goto_slice(1)
        rec_second = app.engine.record(0, 1)
        truth_second = app._truth_from_cache(0, 1, rec_second, np)
        expect_second = {}
        for r in (0, 2, 5, 9):
            t, p = int(truth_second[r]), int(pr_second[1][r])
            if t >= 1 and p >= 1:
                expect_second[(t, p)] = expect_second.get((t, p), 0) + 1
        assert app._cm_counts["current"] == expect_second
        assert app._cm_counts["all"] == all_before_nav
        app._goto_slice(0)

        # Return the rest of the broad selftest to its original one-slice fixture.
        for uid in second_uids:
            app.store.remove(uid)
        app.subsequences[0]["files"] = files
        app.primed = [{"files": files, "base": [zeros], "filtered": [zeros],
                       "pipes": [None], "normalizers": [[]]}]
        app._slices = {(0, 0): rec3}
        app._pred = {(0, 0): pr}
        app._ensure_slice_record = original_ensure
        app._rebuild_flat_slices()
        app._rebuild_class_panels()

        # Dense FC kind: same buttons, scaler inside the pickled pipeline.
        app.model_kind_var.set("dense FC")
        app._train_classifier()
        assert app._clf_kind == "dense FC"
        app._classify()
        assert app._pred, "dense FC classifies"

        # RF-selected dense variants consume the full profile fingerprint but
        # train the MLP on only the selected dimensions. This two-column
        # fixture also exercises the top-N cap for profiles narrower than N.
        assert set(_DENSE_TOP_N) <= set(_MODEL_KINDS)
        assert app._make_model("dense-top-16", 8, 40).named_steps[
            "select"].max_features == 16
        assert app._make_model("dense-top-32", 8, 40).named_steps[
            "select"].max_features == 32
        for dense_kind in ("dense-top-16", "dense-top-32"):
            app.model_kind_var.set(dense_kind)
            app._train_classifier()
            assert app._clf_kind == dense_kind
            selector = app._clf.named_steps["select"]
            assert int(selector.get_support().sum()) == 2
            assert selector.max_features == 2
            assert app._clf_names == ["area", "mean_base"], \
                "the full profile fingerprint must survive selection"
            app._classify()
            before_load = app._pred[(0, 0)][1].copy()
            with tempfile.TemporaryDirectory() as td:
                dense_path = os.path.join(td, dense_kind + ".pkl")
                app._save_classifier_to(dense_path)
                app._clf = None
                app._clf_names = None
                app.classify_btn.config(state="disabled")
                app._load_classifier_from(dense_path)
                assert app._clf_kind == dense_kind
                assert app.model_kind_var.get() == dense_kind
                app._classify()
                assert np.array_equal(app._pred[(0, 0)][1], before_load), \
                    "loaded dense-top model must preserve predictions"
        # "dense (tuned)": Train with no search yet builds the baseline behind a
        # FeatureSubset; Optimize (run inline here) installs the search winner,
        # whose spec rides the pickle and the session model record.
        assert _TUNED_KIND in _MODEL_KINDS
        assert _under(app.optimize_btn, app.model_tab)
        assert str(app.cancel_search_btn.cget("state")) == "disabled"
        app.model_kind_var.set(_TUNED_KIND)
        assert "No search yet" in app.model_arch_var.get()
        app._train_classifier()
        assert app._clf_kind == _TUNED_KIND and app._clf_spec is None
        assert app._clf.named_steps["dense"].hidden_layer_sizes == _MLP_HIDDEN
        assert app._clf_names == ["area", "mean_base"], "full fingerprint kept"
        app._classify()
        assert app._pred, "tuned baseline classifies"
        # Unscorable labels (one class) refuse with a reason, not a traceback.
        original_training_set = app._training_set
        app._training_set = lambda: (np.zeros((6, 2)), np.ones(6, int),
                                     np.arange(6), ["area", "mean_base"])
        app._optimize_network(sync=True)
        assert app._search is None and "Cannot optimize" in app.status_var.get()
        # A synthetic labeled set -- six slices, three classes -- runs the
        # search to completion inline, grouped by slice.
        rng = np.random.RandomState(0)
        Xs = rng.normal(size=(90, 2))
        ys = rng.randint(1, 4, size=90)
        gs = rng.randint(0, 6, size=90)
        Xs[:, 1] += 2.0 * ys
        app._training_set = lambda: (Xs, ys, gs, ["area", "mean_base"])
        app.search_trials_var.set("3")
        app.search_timeout_var.set("60")            # minutes
        models_td = tempfile.mkdtemp()
        app._models_dir = models_td                 # never the user's folder
        app._optimize_network(sync=True)
        assert app._search is None, "the pump drains a sync search to completion"
        # The winner is pickled + recorded on finish, so a restore finds it.
        saved = [f for f in os.listdir(models_td) if f.startswith("tuned_")]
        assert len(saved) == 1, saved
        assert app.models[-1]["path"] == os.path.join(models_td, saved[0])
        assert app.models[-1]["kind"] == _TUNED_KIND
        assert "saved" in app.status_var.get()
        assert app._clf_spec.max_iter == _MLP_MAX_ITER, "refit with the full budget"
        assert str(app.optimize_btn.cget("state")) == "normal"
        assert app._clf_kind == _TUNED_KIND and app._clf_spec is not None
        spec = app._clf_spec
        assert spec.n_trials == 3 and spec.cv_kind == "slices" and spec.n_groups == 6
        assert spec.cv_score is not None and spec.cv_score <= spec.baseline_score
        assert app._search_spec is spec
        assert app.model_arch_var.get() == _model_description(_TUNED_KIND, spec, 2)
        assert "CV bacc" in app.model_hint_var.get(), app.model_hint_var.get()
        assert "CV bacc" in app.model_strip_var.get(), app.model_strip_var.get()
        assert app._pred, "the winner is classified immediately"
        assert "trials" in app.search_progress_var.get()
        # Train on the tuned kind rebuilds the same spec on the current labels.
        app._pred.clear()
        app._train_classifier()
        assert app._clf_spec == spec and app._clf_kind == _TUNED_KIND
        # The spec rides the pickle (v3) and the session's model record.
        with tempfile.TemporaryDirectory() as td:
            tuned_path = os.path.join(td, "tuned.pkl")
            app._save_classifier_to(tuned_path)
            assert app.models[-1]["kind"] == _TUNED_KIND
            assert app.models[-1]["spec"]["hidden"] == list(spec.hidden)
            sdoc_t = session.session_doc_from_json(app._session_doc())
            assert sdoc_t["models"][-1]["spec"]["hidden"] == list(spec.hidden)
            app._clf = None
            app._clf_spec = None
            app._search_spec = None
            app.model_kind_var.set("dense FC")
            app._load_classifier_from(tuned_path)
            assert app._clf_kind == _TUNED_KIND
            assert app._clf_spec == spec and app._search_spec == spec
            assert app.model_kind_var.get() == _TUNED_KIND
            assert app.model_arch_var.get() == _model_description(_TUNED_KIND, spec, 2)
        # The search settings ride the session view, clamped on the way in.
        assert app._view_state()["model_search"] == \
            {"trials": 3, "timeout_s": 3600, "seed": 0, "feature_search": True,
             "backend": "auto",
             "sweep_sizes": "64-32, 32-16, 16-8, 8-4, 4", "sweep_trials": _SWEEP_TRIALS}, \
            "the entry is minutes, the session seconds"
        app._apply_search_view({"trials": 12, "timeout_s": 5, "seed": 7,
                                "feature_search": False, "backend": "sklearn", "junk": 1})
        assert app._search_settings() == (12, 5, 7, False, "sklearn")
        app.search_timeout_var.set("0.5")
        assert app._search_settings()[1] == 30, "fractional minutes"
        app.search_timeout_var.set("junk")
        assert app._search_settings()[1] == _SEARCH_TIMEOUT_S, "malformed = default"
        # Size sweep: one fixed-size search per rung, a report row per rung,
        # the best rung installed + saved, any rung installable from the table.
        app.search_timeout_var.set("60")
        assert _under(app.sweep_tree, app.analysis_tab) and _under(app.sweep_btn, app.analysis_tab)
        assert _under(app.errors_tree, app.analysis_tab)
        assert model_search.parse_sizes(" 64-32, 16x8;4 ") == [(64, 32), (16, 8), (4,)]
        assert model_search.n_params((4,), 2, 3) == (2 + 1) * 4 + (4 + 1) * 3
        app.sweep_sizes_var.set("8-4, 4")
        app.sweep_trials_var.set("2")
        assert app._sweep_settings() == ([(8, 4), (4,)], 2)
        app.sweep_sizes_var.set("8-4, banana")
        app._sweep_network(sync=True)
        assert app._search is None and "cannot read" in app.status_var.get()
        app.sweep_sizes_var.set("8-4, 4")
        n_models_before = len([f for f in os.listdir(models_td) if f.startswith("tuned_")])
        app._sweep_network(sync=True)
        assert app._search is None and app._sweep is not None
        res = app._sweep
        assert [r.size for r in res.rows] == ["8-4", "4"] and all(r.n_trials == 2 for r in res.rows)
        assert all(r.spec.hidden == r.hidden and r.spec.baseline_hidden == r.hidden
                   for r in res.rows), "each rung searched at its own fixed size"
        assert all(r.spec.max_iter == _MLP_MAX_ITER for r in res.rows)
        assert len(app.sweep_tree.get_children()) == 2
        assert app.sweep_tree.item("0", "values")[0] == "8-4"
        assert app.sweep_summary_var.get().startswith("best: ")
        assert "rung" in app.search_progress_var.get()
        best = res.rows[res.best_index()]
        assert app._clf_spec == best.spec and app._clf_kind == _TUNED_KIND
        assert app._clf is best.estimator and app._pred, "best rung installed + classified"
        assert len([f for f in os.listdir(models_td) if f.startswith("tuned_")]) == n_models_before + 1
        assert len([f for f in os.listdir(models_td) if f.startswith("sweep_")]) == 1
        lines = res.report_lines()
        assert lines[0].startswith("size") and len(lines) == 2 + len(res.rows)
        assert res.to_dict()["rows"][0]["hidden"] == [8, 4]
        # Install the OTHER rung from the table.
        other = 1 - res.best_index()
        app.sweep_tree.selection_set(str(other))
        app._refresh_sweep_buttons()
        assert str(app.sweep_install_btn.cget("state")) == "normal"
        app._sweep_install_selected()
        assert app._clf is res.rows[other].estimator and app._clf_spec == res.rows[other].spec
        assert app._search_spec == res.rows[other].spec
        assert "Installed " + res.rows[other].size in app.status_var.get()
        assert app.model_arch_var.get() == _model_description(_TUNED_KIND, res.rows[other].spec, 2)
        # The sweep settings ride the session view and are validated on the way in.
        app._apply_search_view({"sweep_sizes": "nope!", "sweep_trials": 7})
        assert app.sweep_sizes_var.get() == "8-4, 4" and app._sweep_settings()[1] == 7
        app._apply_search_view({"sweep_sizes": "16-8, 4"})
        assert app.sweep_sizes_var.get() == "16-8, 4"
        app.sweep_sizes_var.set(", ".join(model_search.size_text(h)
                                          for h in model_search.DEFAULT_SIZE_LADDER))
        app.sweep_trials_var.set(str(_SWEEP_TRIALS))
        app._sweep = None
        app._refresh_sweep_report([])
        app._apply_search_view({"backend": "nonsense"})
        assert app._search_settings()[4] == "sklearn", "unknown backend ignored"
        # A forced sklearn backend builds sklearn's MLP even when torch exists;
        # the spec records the resolved backend and the readout names it.
        app.search_trials_var.set("2")
        app._optimize_network(sync=True)
        assert app._search is None and app._clf_spec.backend == "sklearn"
        assert type(app._clf.named_steps["dense"]).__name__ == "MLPClassifier"
        assert "sklearn (cpu)" in app.model_arch_var.get()
        app.search_backend_var.set("auto")
        app.search_trials_var.set("abc")
        assert app._search_settings()[0] == _SEARCH_TRIALS, "malformed = default"
        app.search_trials_var.set(str(_SEARCH_TRIALS))
        app.search_timeout_var.set(str(_SEARCH_TIMEOUT_MIN))
        import shutil
        shutil.rmtree(models_td, ignore_errors=True)
        app._models_dir = None
        app.search_seed_var.set("0")
        app.search_features_var.set(True)
        app.search_backend_var.set("auto")
        app._training_set = original_training_set
        app._search_spec = None
        app.model_kind_var.set("random forest")

        # 'R' = train + immediate reclassify in one call, without resetting
        # visualization choices during the transient empty prediction cache.
        app.region_mode_var.set(_MODE_UNCERTAINTY)
        app.show_pred_var.set(False)
        app._cm_cell = (1, 1)
        app._pred.clear()
        app._train_and_classify()
        assert app._clf_kind == "random forest" and app._pred, \
            "'R' retrains and reclassifies"
        assert app.region_mode_var.get() == _MODE_UNCERTAINTY
        assert not app.show_pred_var.get(), "R preserves Show Classification"
        assert app._cm_cell == (1, 1), "R preserves the confusion selection"

        # 'C' = classify with the current model and also leaves visibility alone.
        app._pred.clear()
        app._on_classify_key()
        assert app._pred, "'C' reclassifies"
        assert not app.show_pred_var.get(), "Classify does not force its layer on"
        app.show_pred_var.set(True)

        # SHIFT-accept: the predictions under a box become one "taps"
        # interaction per predicted class -- geometric, and ONE undo step.
        n_before = len(app.store.interactions)
        app._accept_predictions([(2.0, 2.0), (17.0, 17.0)])   # covers all 4
        added = app.store.interactions[n_before:]
        assert added and all(it.tool == "taps" for it in added)
        assert sum(len(it.points) for it in added) == 4, "all regions accepted"
        rc_now = labeling.resolve_slice(app.store.for_slice("data/s0.tiff"),
                                        lab, np)
        pr_now = app._pred[(0, 0)][1]
        for r in (0, 2, 5, 9):
            assert rc_now[r] == pr_now[r], "accepted labels match predictions"
        app._undo()
        assert len(app.store.interactions) == n_before, "batch accept = one undo"

        # Classifier save/load round trip, session model refs, and the CSV.
        with tempfile.TemporaryDirectory() as td:
            clf_path = os.path.join(td, "classifier.pkl")
            app._save_classifier_to(clf_path)
            assert app.models and app.models[-1]["fingerprint"] == ["area", "mean_base"]
            assert app.models[-1]["kind"] == "random forest"
            sdoc2 = app._session_doc()
            assert sdoc2["models"] and sdoc2["models"][-1]["path"] == \
                os.path.abspath(clf_path)
            app._clf = None
            app._clf_names = None
            app.classify_btn.config(state="disabled")
            # The compatibility gate: an incompatible profile refuses the load
            # outright, and Classify blocks with a message instead of silently
            # skipping slices.
            app._expected_feature_names = lambda: ["area", "mean_base", "std_blur_s1.5"]
            try:
                app._load_classifier_from(clf_path)
                raise AssertionError("load must refuse an incompatible model")
            except ValueError as exc:
                assert "std_blur_s1.5" in str(exc), exc
            assert app._clf is None, "nothing installed on refusal"

            # Track A: interactively, a v2 pickle's OWN statistics can be
            # applied as a new profile instead of that flat refusal.
            def _expected_from_profile():
                """Stand-in for the real schema call (which needs the compiled
                extension): derive the field set from the ACTIVE profile's
                statistics, so applying the model's own really changes it."""
                doc = config_io.statistics_from_json(
                    app.profiles[app.active_profile_idx].get("statistics"))
                out = ["area"]
                for c in doc["channels"]:
                    for s in (c.get("sigmas") or [None]):
                        out.append("mean_" + c["kind"]
                                   + ("" if s is None else f"_s{s:g}"))
                return out

            def _reprime():
                """The engine.reset() inside the profile switch drops the fake
                primed stack; put it back for the rest of the selftest."""
                app.primed = [{"files": files, "base": [zeros],
                               "filtered": [zeros], "pipes": [None],
                               "normalizers": [[]]}]
                app._rebuild_flat_slices()
                rec["commit"] = app._commit_id
                app._slices[(0, 0)] = rec

            app._expected_feature_names = _expected_from_profile
            # The pickle was saved under base-only statistics; move the ACTIVE
            # profile off them so the load really mismatches.
            app.stat_kind_vars["blur"][0].set(True)
            app.stat_kind_vars["blur"][1].set("1.5")
            app._snapshot_active_profile()
            assert app._check_model_compat(["area", "mean_base"], "t") is not None
            n_prof = len(app.profiles)
            with _mock.patch.object(messagebox, "askyesno", return_value=False):
                try:
                    app._load_classifier_from(clf_path, interactive=True)
                    raise AssertionError("declining the offer must still refuse")
                except ValueError:
                    pass
            assert len(app.profiles) == n_prof, "declining creates no profile"
            assert app._clf is None, "nothing installed when the offer is declined"
            with _mock.patch.object(messagebox, "askyesno", return_value=True):
                app._load_classifier_from(clf_path, interactive=True)
            assert len(app.profiles) == n_prof + 1, "accepting adds a profile"
            assert app.active_profile_idx == n_prof, "and switches to it"
            assert app.profiles[-1]["name"] == "from classifier.pkl"
            assert not app.stat_kind_vars["blur"][0].get(), \
                "the model's statistics reached the stats panel"
            assert app._clf is not None and not app._pred, \
                "model installed, predictions dropped with the old parameters"
            strip = app.model_strip_var.get()
            assert "random forest" in strip and "2 feats" in strip, strip
            assert "mismatch" not in strip, strip
            _reprime()
            # Headless (no compiled extension) the offer never fires: the real
            # gate returns None, so the load succeeds as before.
            app._expected_feature_names = lambda: ["area", "mean_base"]
            app._load_classifier_from(clf_path)
            assert app._clf is not None and app._clf_names == ["area", "mean_base"]
            assert str(app.classify_btn.cget("state")) == "normal"
            app._expected_feature_names = lambda: ["area", "mean_base", "extra"]
            msg = app._check_model_compat(app._clf_names, "test")
            assert msg is not None and "extra" in msg
            with _mock.patch.object(messagebox, "showerror") as _err:
                app._classify()
                assert _err.called, "classify must block on mismatch"
            app._expected_feature_names = lambda: ["area", "mean_base"]
            app._classify()
            csv2 = os.path.join(td, "labels2.csv")
            covered2, _sk = app._write_labels_csv(csv2)
            assert covered2 == 1
            with open(csv2, encoding="utf-8") as f:
                rows2 = f.read().strip().splitlines()
            preds = [r.split(",")[3] for r in rows2[1:]]
            assert all(v in ("1", "2") for v in preds), "predicted column filled"

            # Training-set export: raw TIFF into train/, class-id mask into
            # labels/ (classifier predictions, user annotations winning).
            from PIL import Image as _Img
            src_dir = os.path.join(td, "src")
            os.makedirs(src_dir)
            src_tif = os.path.join(src_dir, "s0.tiff")
            _Img.fromarray(zeros).save(src_tif)
            app.folders = [{"path": src_dir, "name": "data"}]
            app.subsequences = [{"name": "seq1", "folder": "data",
                                 "files": [src_tif]}]
            app.primed = [{"files": [src_tif], "base": [zeros],
                           "filtered": [zeros], "pipes": [None],
                           "normalizers": [[]]}]
            app._rebuild_flat_slices()
            rec5 = dict(rec3, commit=app._commit_id)
            app._slices[(0, 0)] = rec5
            out_dir = os.path.join(td, "tset")
            written, skipped_ts = app._write_training_set(out_dir)
            assert (written, skipped_ts) == (1, 0), (written, skipped_ts)
            assert os.path.isfile(os.path.join(out_dir, "train", "data__s0.tiff"))
            mask = np.asarray(_Img.open(
                os.path.join(out_dir, "labels", "data__s0.tiff")))
            assert mask.shape == (20, 20) and mask.dtype == np.uint8
            assert mask[0, 0] == 0, "background (-1) stays class 0"
            assert mask[5, 5] > 0, "labeled region carries its class id"
            # User annotations win over predictions where they disagree.
            rc_user = labeling.resolve_slice(
                app.store.for_slice("data/s0.tiff"), lab, np)
            for r, px in ((0, (5, 5)), (9, (12, 12))):
                if rc_user[r] > 0:
                    assert mask[px] == rc_user[r]

    # -- Drawing-tool state machine: gesture previews + magic fill ----------- #
    # Fresh one-slice fixture at the current commit (earlier sections swapped
    # the record). No engine: adjacency comes from the pixel fallback and the
    # table has mean_base only (so the extremum points fall back to the first
    # pixel of each region).
    rec_m = {"commit": app._commit_id, "labels": lab, "stats": table,
             "kept": set(), "cc": None, "n_feat": 4}
    app._slices[(0, 0)] = rec_m
    app._pred = {}
    v = app.viewer
    v.view_x, v.view_y, v.scale = 0.0, 0.0, 1.0
    ctrl = v.tool
    assert isinstance(ctrl, DrawController)
    n_before = len(app.store.interactions)

    def lit():
        return set(np.flatnonzero(v._transient[2][:, 3] > 0).tolist())

    # Squiggle: incremental, exact -- from region 0 into region 2.
    app.tool_var.set("squiggle"); app.active_class_var.set(1)
    assert ctrl.on_press(_FakeEvent(5, 5))
    assert v._transient is not None and app._hover_suppressed
    assert lit() == {0}, lit()
    assert ctrl.on_move(_FakeEvent(15, 5))
    assert lit() == {0, 2}, lit()
    assert ctrl.on_release(_FakeEvent(15, 5))
    assert v._transient is None and not app._hover_suppressed
    assert len(app.store.interactions) == n_before + 1
    assert app.store.interactions[-1].tool == "squiggle"

    # Box across all four blocks, then Escape: nothing committed, class kept.
    app.tool_var.set("box")
    assert ctrl.on_press(_FakeEvent(3, 3))
    assert ctrl.on_move(_FakeEvent(16, 16))
    assert lit() == {0, 2, 5, 9}, lit()
    app._on_escape()
    assert v._transient is None and ctrl._pts is None
    assert len(app.store.interactions) == n_before + 1, "Escape commits nothing"
    assert app.active_class_var.get() == 1, "Escape mid-gesture keeps the class"
    app._on_escape()
    assert app.active_class_var.get() == 0, "Escape with nothing in flight disarms"

    # SHIFT-accept box: the preview shows what WOULD be accepted, each region
    # in its predicted class color; unpredicted regions stay dark.
    app.active_class_var.set(1)
    k_hi = app.store.n_classes - 1
    pred_rc = np.zeros(10, np.uint8); pred_rc[0] = 1; pred_rc[9] = k_hi
    app._pred[(0, 0)] = (app._commit_id, pred_rc,
                         np.zeros((10, MAX_CLASSES), np.float32))
    assert ctrl.on_press(_FakeEvent(3, 3, state=0x0001))
    assert ctrl.on_move(_FakeEvent(16, 16, state=0x0001))
    assert lit() == {0, 9}, lit()
    pure = np.asarray(app.store.rgba(k_hi)[:3], np.int32)
    shown = v._transient[2][9, :3].astype(np.int32)
    assert (shown >= pure).all() and v._transient[2][9, 3] == 255, (shown, pure)
    app._on_escape()
    app._pred = {}

    # Magic fill: seed region 0, mean/anchor on base. The ladder is
    # d = [0, 1, 2, 3] / std(mean_base), so a long drag up sweeps all four
    # regions and a drag down leaves the seed alone.
    app.tool_var.set("magic"); app.active_class_var.set(1)
    assert app.magic_metric_var.get() == "mean"
    assert ctrl.on_press(_FakeEvent(5, 5))
    assert ctrl.magic.active and v._transient is not None
    assert rec_m.get("arcs") is not None and rec_m["arcs"]["source"] == "pixels"
    assert set(zip(rec_m["arcs"]["a"].tolist(), rec_m["arcs"]["b"].tolist())) == \
        {(0, 2), (0, 5), (2, 9), (5, 9)}
    assert v._hud_mode == "info" and "magic" in v._hud_text
    assert lit() == {0}, lit()
    assert ctrl.on_move(_FakeEvent(5, 5 - 400))
    assert lit() == {0, 2, 5, 9}, lit()
    assert ctrl.on_move(_FakeEvent(5, 5 + 400))
    assert lit() == {0}, lit()
    assert ctrl.on_move(_FakeEvent(5, 5 - 400))
    assert ctrl.on_release(_FakeEvent(5, 5 - 400))
    assert v._transient is None and v._hud_mode is None and not ctrl.magic.active
    it_m = app.store.interactions[-1]
    assert it_m.tool == "taps" and it_m.meta and it_m.meta["tool"] == "magic"
    assert it_m.meta["n_regions"] == 4 and it_m.meta["seed_id"] == 0
    assert len(it_m.points) == 4
    rc_m = labeling.resolve_slice([it_m], lab, np)
    assert all(rc_m[r] == 1 for r in (0, 2, 5, 9))
    # Provenance in the session doc; the options ride the view state; undo
    # removes the whole fill in one step.
    app._rebuild_class_panels()
    doc_m = app._session_doc()
    assert doc_m["view"]["magic"]["metric"] == "mean"
    assert any(d.get("meta", {}).get("tool") == "magic"
               for d in doc_m["annotations"]["interactions"])
    n_now = len(app.store.interactions)
    app._undo()
    assert len(app.store.interactions) == n_now - 1
    assert not any(it.meta for it in app.store.interactions)
    # A second press starts from the threshold last released (all four).
    app.active_class_var.set(1)
    assert ctrl.on_press(_FakeEvent(5, 5))
    assert lit() == {0, 2, 5, 9}, lit()
    assert ctrl.magic.cancel()
    assert v._transient is None
    # Escape mid-fill abandons it and keeps the class armed; a press on the
    # background is refused (falls through to the pan).
    assert ctrl.on_press(_FakeEvent(12, 12))
    app._on_escape()
    assert not ctrl.magic.active and app.active_class_var.get() == 1
    assert not ctrl.on_press(_FakeEvent(0, 0))
    # The chain mode and bhattacharyya need std_ columns: a table without
    # them reports instead of raising out of the press.
    app.magic_metric_var.set("bhattacharyya")
    assert not ctrl.on_press(_FakeEvent(5, 5))
    assert "std_base" in app.status_var.get(), app.status_var.get()
    app.magic_metric_var.set("mean")

    # Blobber: the magic core in the active class, its immediate neighbours
    # in the ring class ("next" -> class 2 for class 1). The last released
    # threshold (all four) is remembered, so drag DOWN to the seed alone.
    if app.store.n_classes < 3:
        app.store.set_n_classes(3); app.n_classes_var.set(3)
    app.tool_var.set("blobber"); app.active_class_var.set(1)
    assert ctrl.magic.ring_class(1) == 2
    app.blob_ring_var.set("1")                 # same as the active class -> next
    assert ctrl.magic.ring_class(1) == 2
    app.blob_ring_var.set(str(app.store.n_classes - 1))
    assert ctrl.magic.ring_class(1) == app.store.n_classes - 1
    app.blob_ring_var.set("next")
    assert ctrl.on_press(_FakeEvent(5, 5))
    assert ctrl.magic.active and "ring" in v._hud_text
    assert ctrl.on_move(_FakeEvent(5, 5 + 400))
    assert lit() == {0, 2, 5}, lit()           # core {0}, ring = its neighbours
    lut = v._transient[2]
    assert lut[0].tolist() == list(app.store.rgba(1)), "seed keeps the pure core color"
    c1 = np.asarray(app.store.rgba(1)[:3], int); c2 = np.asarray(app.store.rgba(2)[:3], int)
    for r in (2, 5):
        d1 = np.abs(lut[r, :3].astype(int) - c1).sum()
        d2 = np.abs(lut[r, :3].astype(int) - c2).sum()
        assert d2 < d1, "ring regions preview in the ring class color"
    n0 = len(app.store.interactions)
    assert ctrl.on_release(_FakeEvent(5, 5 + 400))
    assert len(app.store.interactions) == n0 + 2, "ring + core, two interactions"
    ring_it, core_it = app.store.interactions[-2], app.store.interactions[-1]
    assert ring_it.class_id == 2 and ring_it.meta["part"] == "ring" \
        and ring_it.meta["tool"] == "blobber" and len(ring_it.points) == 2
    assert core_it.class_id == 1 and core_it.meta["part"] == "core" \
        and len(core_it.points) == 1
    rc_b = labeling.resolve_slice([ring_it, core_it], lab, np)
    assert rc_b[0] == 1 and rc_b[2] == 2 and rc_b[5] == 2 and rc_b[9] == 0
    app._rebuild_class_panels()
    assert app._session_doc()["view"]["magic"]["ring"] == "next"
    app._undo()
    assert len(app.store.interactions) == n0, "one undo step removes ring + core"
    # Grown to everything: no ring left, only the core commits.
    assert ctrl.on_press(_FakeEvent(5, 5))
    assert ctrl.on_move(_FakeEvent(5, 5 - 400))
    assert lit() == {0, 2, 5, 9}, lit()
    assert ctrl.on_release(_FakeEvent(5, 5 - 400))
    assert len(app.store.interactions) == n0 + 1
    assert app.store.interactions[-1].meta["part"] == "core"
    app._undo()

    # Hop gain + drag entries: lenient parsing, clamped, session round-trip
    # as floats, restored with validation; the gain reaches the ladder, the
    # HUD, the provenance and the last-threshold key.
    app.tool_var.set("magic")
    app.magic_gain_var.set("abc"); assert ctrl.magic.hop_gain() == _DEFAULT_HOP_GAIN
    app.magic_gain_var.set("0.5"); assert ctrl.magic.hop_gain() == 1.0
    app.magic_gain_var.set("9");   assert ctrl.magic.hop_gain() == 2.0
    app.magic_drag_var.set("x");   assert ctrl.magic.drag_px() == _DEFAULT_DRAG_PX
    app.magic_gain_var.set("1.5"); app.magic_drag_var.set("8")
    vs = app._session_doc()["view"]["magic"]
    assert vs["hop_gain"] == 1.5 and vs["drag_px"] == 8.0
    n0 = len(app.store.interactions)
    assert ctrl.on_press(_FakeEvent(5, 5))
    assert "g1.5" in v._hud_text, v._hud_text
    assert ctrl.magic._s["ladder"].hop_gain == 1.5
    assert ctrl.magic._s["drag_px"] == 8.0
    assert ctrl.on_move(_FakeEvent(5, 5 - 400))
    assert ctrl.on_release(_FakeEvent(5, 5 - 400))
    assert len(app.store.interactions) == n0 + 1
    assert app.store.interactions[-1].meta["hop_gain"] == 1.5
    assert any(k[3] == 1.5 for k in ctrl.magic._last), list(ctrl.magic._last)
    app._undo()
    app._apply_magic_view({"hop_gain": 1.25, "drag_px": 6, "metric": "cosine",
                           "mode": "chain"})
    assert app.magic_gain_var.get() == "1.25" and app.magic_drag_var.get() == "6"
    assert app.magic_metric_var.get() == "cosine" and app.magic_mode_var.get() == "chain"
    app._apply_magic_view({"hop_gain": "abc", "drag_px": 0.5, "metric": "nope"})
    assert app.magic_gain_var.get() == "1.25" and app.magic_drag_var.get() == "6"
    assert app.magic_metric_var.get() == "cosine"
    # cosine runs on the whole row (here area + mean_base) without channels.
    app.magic_gain_var.set("1")
    assert ctrl.on_press(_FakeEvent(5, 5))
    assert ctrl.magic._s["ladder"].metric == "cosine"
    assert ctrl.magic.cancel()
    app.magic_mode_var.set("anchor")

    # proba: refused without predictions or with stale ones; with a fake
    # prediction the seed's class joins first, unscored regions last.
    app.magic_metric_var.set("proba")
    app._pred = {}
    assert not ctrl.on_press(_FakeEvent(5, 5))
    assert "Classify first" in app.status_var.get(), app.status_var.get()
    proba = np.zeros((10, MAX_CLASSES), np.float32)
    proba[0, 1] = 1.0; proba[2, 1] = 1.0; proba[5, 2] = 1.0      # 9 unscored
    rc_p = np.zeros(10, np.uint8)
    app._pred[(0, 0)] = (app._commit_id + 1, rc_p, proba)          # stale
    assert not ctrl.on_press(_FakeEvent(5, 5))
    app._pred[(0, 0)] = (app._commit_id, rc_p, proba)
    assert ctrl.on_press(_FakeEvent(5, 5))
    lad_p = ctrl.magic._s["ladder"]
    assert lad_p.ids[lad_p.order].tolist() == [0, 2, 5, 9], lad_p.ids[lad_p.order].tolist()
    assert np.allclose(lad_p.join, [0.0, 0.0, 1.0, 1.0]), lad_p.join
    assert ctrl.magic.cancel()
    app._pred = {}
    app.magic_metric_var.set("mean")
    app.magic_gain_var.set(f"{_DEFAULT_HOP_GAIN:g}")
    app.magic_drag_var.set(f"{_DEFAULT_DRAG_PX:g}")
    app.tool_var.set("blobber")
    # Two-class store: no ring class exists, the press is refused.
    saved_n = app.store.n_classes
    app.store.set_n_classes(2); app.n_classes_var.set(2)
    assert not ctrl.on_press(_FakeEvent(5, 5))
    assert "second class" in app.status_var.get()
    app.store.set_n_classes(saved_n); app.n_classes_var.set(saved_n)
    app.active_class_var.set(0)
    app.tool_var.set("squiggle")

    # A re-prime (engine "done") bumps the commit, so every commit-keyed cache
    # (per-slice records, class LUTs, predictions) self-invalidates -- the
    # "stale overlays after adding a folder and re-running" regression.
    c0 = app._commit_id
    app.engine.work_q.put(("done", []))
    app.engine.poll()
    assert app._commit_id == c0 + 1, "re-prime must bump the commit"

    # The labeler's session file never collides with the viewer's.
    assert config_io.session_path(app=app.SESSION_APP) != config_io.session_path()

    root.destroy()
    print("labeler selftest OK: tiers, gestures, resolution order, LUT cache, "
          "overlay stack, class-count clamp, session round-trip (qualified "
          "keys), export, undo/redo, tap-squiggle, hover geometry, "
          "click-to-center, CSV, model kinds + compat gate + 'R', "
          "profile-from-model + provenance strip, swatch arming, "
          "proba cache + coloring modes, confusion matrix + highlight, "
          "persistent outlines + canvas right-click, gesture previews, "
          "magic fill, blobber, hop gain + drag + cosine/proba metrics, "
          "center notebook + model tab, toolbar hints, optimize network, size sweep, "
          "edge kinds, analysis tab + region list")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
