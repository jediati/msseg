"""The two link-labels that say what is in effect (the active profile and the
active model) and the centre notebook's tab bookkeeping."""
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


class HintsMixin:

    # -- hints: the selected workflow and the active model ----------------- #
    # With the profile edited on one tab and the model on another, the panels
    # that USE them need a reminder: the workflow (as a compact chain,
    # _workflow_summary) sits in the Run box above the run settings,
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
            self._workflow_hint_tooltip())
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
        return self._workflow_summary(profile)

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
