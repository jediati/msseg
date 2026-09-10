"""The View tab's controls and layers: the image dropdown, overlay toggles and
region coloring modes, the class/prediction overlay stack and its caches."""
from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .. import edge_model, magic_fill, model_search, fields
from .. import bundle as model_bundle
from ..labeling import (LabelStore, MAX_CLASSES, TOOLS, resolve_slice, resolve_sets,
                          touched_sets, class_lut, scalar_lut, line_pixels, polygon_mask,
                          preview_lut)
from ..training import TrainingSetBuilder, TrainingProblem
from ..widgets import ScrollFrame, attach_tooltip
from ..defaults import *  # noqa: F401,F403
from ..tools import DrawController, MagicFillController, _extremum_points


class ViewControlsMixin:

    def _refresh_render(self):
        viewer = self.viewer
        first = viewer is not None and not viewer.has_base
        super()._refresh_render()
        if first and viewer is not None and viewer.has_base:
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
        self._set_region_layer_visible(self.show_regions_var.get())

    def _on_overlay_toggle(self):
        """Apply the master overlay state to every control in its row."""
        enabled = self.show_overlay_var.get()
        for widget, active_state in self._overlay_dependents:
            widget.configure(state=active_state if enabled else "disabled")
        self._refresh_annotation_layer()
        self._refresh_render()

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
            entry = self._pred.get(self.catalogue.key_of(si, li))
            if entry is not None and entry[0] == rec.get("commit"):
                scalar = self._region_scalar(entry, np)
        if scalar is not None:
            # A coloring mode REPLACES the id LUT rather than tinting it: the
            # ids and the scalar are two readings of the same regions, and
            # stacking them would just muddy both.
            values, mask = scalar
            overlays = [o for o in overlays if "lut" not in o]
            overlays.append(self._region_overlay(
                rec["labels"], scalar_lut(values, np, _SCALAR_ALPHA, mask), np))
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
                pr = self._pred.get(self.catalogue.key_of(si, li))
                if pr is not None and pr[0] == rec.get("commit"):
                    plut = class_lut(pr[1], np,
                                     self._class_colors_rgba(np)).copy()
                    plut[:, 3] = (plut[:, 3].astype(np.uint16)
                                  * _PRED_ALPHA // 255).astype(np.uint8)
                    overlays.append(self._region_overlay(rec["labels"], plut, np))
            if self.show_gt_var.get():
                lut = self._class_lut_for(si, li, rec, np)
                if lut is not None:
                    overlays.append(self._region_overlay(rec["labels"], lut, np))
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
                    overlays.append(self._region_overlay(rec["labels"], hl, np))
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
