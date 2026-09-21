"""The Analysis tab: predictions vs annotations (the regions behind a confusion
cell, with go-to) and the size-sweep report."""
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


class AnalysisPanelMixin:

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
        for key, entry in self._pred.items():
            idx = self.catalogue.index_of(key)
            rec = self.regions.record(key) if idx is not None else None
            if rec is None or rec.get("labels") is None or entry[0] != rec.get("commit"):
                continue
            si, li = idx
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
        """Show slice (si, li) centred on `region` (its seeding extremum),
        with the gestures touching it outlined and the selected confusion cell
        still highlighting it. The tab does not change: the canvas is above
        the notebook, so the Analysis list stays open beside the region it
        just sent you to."""
        try:
            idx = self.flat_slices.index((si, li))
        except ValueError:
            self.status_var.set(f"slice {si}:{li} is not primed")
            return False
        self._goto_slice(idx)
        rec = self.regions.record(self.catalogue.key_of(si, li))
        pt = None
        if rec is not None and rec.get("labels") is not None:
            import numpy as np
            res = _extremum_points(rec["labels"], [int(region)], rec.get("stats"), np,
                                   self.FIELDS, self._region_placement())
            pt = res.get(int(region)) if isinstance(res, dict) else (res[0] if res else None)
            if pt is not None and (pt[0] is None or pt[1] is None):
                pt = None
        self.status_var.set(f"region {int(region)} on {self._slice_key(si, li)}"
                            + ("" if pt is not None else " (no position known)"))
        if pt is None:
            return True

        def go():
            # Runs AFTER the repaint _goto_slice queued (which may refit the
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
