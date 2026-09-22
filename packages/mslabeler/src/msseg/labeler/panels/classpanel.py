"""The annotation pane: class panels with their interaction rows (drag between
classes, context menus), the confusion matrix, gesture geometry on the canvas
and the hover readout."""
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


class ClassPanelMixin:

    # ------------------------------------------------------------------ #
    # The label panel (the Annotation tab)
    # ------------------------------------------------------------------ #
    def _build_label_panel(self):
        # A plain frame, not a ScrollFrame: the class subpanels must DIVIDE the
        # available height between them (each scrolls its own interaction list),
        # which needs the holder to fill the tab rather than grow it. This was
        # the window's third PANE until the tabs moved into it.
        self.label_pane = ttk.Frame(self.annot_tab)
        self.label_pane.pack(fill="both", expand=True)
        panel = self.label_pane

        # Two titled halves: what the user DRAWS, and what the model does
        # with it. The classifier half is fixed-height and packs FIRST
        # (side="bottom"), so the annotation half takes every remaining pixel
        # for the class stack -- the one thing here that wants more room.
        ml = ttk.LabelFrame(panel, text="ML Region Classifier")
        ml.pack(side="bottom", fill="x", padx=4, pady=(2, 4))
        # The seam tools (boundaries between regions) sit between the class
        # stack and the classifier; packed bottom so the stack keeps the rest.
        self._build_seam_panel(panel)
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

    def _class_panel_signature(self):
        """What the panels' STRUCTURE depends on: how many class subpanels
        there are and what colour each one's swatch and outlines are. Adding,
        deleting or moving an interaction changes neither."""
        return (int(self.store.n_classes),
                tuple(self._class_color_hex(k)
                      for k in range(1, self.store.n_classes)))

    def _rebuild_class_panels(self):
        """Bring the class panels in line with the store -- the single "labels
        changed" signal, called from eleven places. Also repaints the sequence
        tree, whose "annot" column counts per-slice interactions.

        It DIFFS; it does not rebuild. Destroying the holder's children and
        recreating every frame, ScrollFrame, swatch, tooltip and row is what a
        commit used to cost: Tk erases the destroyed area at once and the
        rebuild repaints it top-down, so *releasing a gesture flashed the whole
        right pane black and filled it back in from the top*. Nothing about
        that work was needed -- the structure depends only on the class count
        and the class colours (see _class_panel_signature), which a commit does
        not touch, and the row set changes by exactly one.

        The subpanels split the holder's height equally (uniform grid rows);
        each one scrolls its own interaction list, so a class with many
        gestures never pushes the others off screen."""
        sig = self._class_panel_signature()
        if sig != getattr(self, "_class_panel_sig", None):
            self._build_class_frames()
            self._class_panel_sig = sig
        self._sync_interaction_rows()
        self._after_class_panels()

    def _build_class_frames(self):
        """(Re)create the per-class subpanels. Only on a structural change."""
        for w in list(self.classes_holder.winfo_children()):
            w.destroy()
        self._class_panels = {}
        self._class_title_labels = {}
        self._class_swatches = {}
        self._class_clear_buttons = {}
        self._class_lists = {}
        self._row_widgets = {}           # uid -> row record (see _sync_...)
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
            attach_tooltip(swatch, (f"{self.store.name(k)}\n" if self.store.has_name(k) else "")
                                   + f"Left-click: draw with class {k} (key {k})\n"
                                   f"Right-click: change color")
            swatch.pack(side="left", padx=(0, 4))
            self._class_swatches[k] = swatch
            # The counts label; double-click names the class (the task's
            # vocabulary: "gland" / "not gland" rather than 1 / 2).
            self._class_title_labels[k] = ttk.Label(title, text=f"Class {k}", cursor="xterm")
            self._class_title_labels[k].pack(side="left")
            self._class_title_labels[k].bind("<Double-Button-1>",
                                             lambda e, k=k: self._rename_class(k))
            attach_tooltip(self._class_title_labels[k],
                           "k · annotations · regions\nDouble-click: name this class")
            # Clear the class: every one of its annotations on every item,
            # after asking -- the per-row context menu is the finer tool.
            clear = ttk.Button(title, text="clear", width=6,
                               command=lambda k=k: self._clear_class_guarded(k))
            clear.pack(side="left", padx=(8, 0))
            attach_tooltip(clear, f"Delete every class {k} annotation on every "
                                  f"{self.ITEM_NOUN} (asks first; Ctrl+Z undoes it)")
            self._class_clear_buttons[k] = clear
            frame.configure(labelwidget=title)
            lst = ScrollFrame(frame, width=240, canvas_width=224,
                              background="white")
            # Small minimum height: the grid's equal weights own the real size.
            lst.canvas.configure(height=48)
            lst.pack(side="top", fill="both", expand=True, padx=2, pady=(0, 2))
            self._class_lists[k] = lst
        # Rows past the last class keep their weight otherwise, so a later
        # regrow would hand a stale row a share of the height.
        for k in range(self.store.n_classes - 1, MAX_CLASSES):
            self.classes_holder.rowconfigure(k, weight=0, uniform="")

    # Pack options for one interaction row; shared by creation and reordering.
    _ROW_PACK = {"side": "top", "fill": "x", "padx": 4, "pady": 0}

    def _sync_interaction_rows(self):
        """Make each class list hold exactly the visible interactions of that
        class, in order, touching as few widgets as possible: a commit creates
        ONE row and leaves every other widget alone.

        A row cannot be reparented in Tk, so an interaction that changed class
        is destroyed and rebuilt -- the only case that flickers, and it moves
        one row rather than the pane."""
        rows = self._row_widgets
        visible = self._visible_interactions()
        # Out-of-range classes are dropped, exactly as the full rebuild's
        # `range(1, n_classes)` loop dropped them: a shrunk class count can
        # leave an interaction pointing past the last panel.
        want = {it.uid for it in visible
                if 1 <= int(it.class_id) < self.store.n_classes}
        for uid in [u for u in rows if u not in want]:
            rows.pop(uid)["frame"].destroy()
        for k in range(1, self.store.n_classes):
            lst = self._class_lists.get(k)
            if lst is None:
                continue
            order = []
            for it in visible:
                if it.class_id != k:
                    continue
                row = rows.get(it.uid)
                if row is not None and row["class"] != k:
                    rows.pop(it.uid)["frame"].destroy()
                    row = None
                if row is None:
                    row = rows[it.uid] = self._build_interaction_row(lst.inner, it)
                else:
                    self._update_interaction_row(row, it)
                order.append(row["frame"])
            # Pack order is creation order, so an appended gesture is already
            # right; only a class move or an undo can disturb it. Chaining each
            # row after its predecessor fixes the whole list without unmapping
            # anything (order[0] ends up first because every other row is
            # explicitly placed after one of them).
            if lst.inner.pack_slaves() != order:
                for i, w in enumerate(order):
                    if i == 0:
                        w.pack_configure(**self._ROW_PACK)
                    else:
                        w.pack_configure(after=order[i - 1], **self._ROW_PACK)

    def _after_class_panels(self):
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
        """Lay out synchronized all-slice and current-slice matrix grids.

        Only when the LAYOUT changed. The grid's shape and its header colours
        are the class count and the class colours -- the same signature the
        subpanels use -- while the counts in the cells are written in place by
        _refresh_confusion, which runs right after this on the same signal. A
        commit changes only the counts, so rebuilding here just made the ML
        panel flash black along with the annotation panel."""
        holder = getattr(self, "confusion_holder", None)
        if holder is None:
            return
        sig = self._class_panel_signature()
        if sig == getattr(self, "_cm_grid_sig", None) and getattr(self, "_cm_cells", None):
            return
        self._cm_grid_sig = sig
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
        current = self._current_key() if scope == "current" else None
        items = ([(current, self._pred.get(current))] if current is not None
                 else []) if scope == "current" else self._pred.items()
        for key, entry in items:
            if entry is None:
                continue
            idx = self.catalogue.index_of(key)
            rec = self.regions.record(key) if idx is not None else None
            if rec is None:
                continue
            si, li = idx
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
        rec = self.regions.record(self.catalogue.key_of(si, li))
        entry = self._pred.get(self.catalogue.key_of(si, li))
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
        return self._gestures_for(*cur)

    def _class_totals(self):
        """ALL-slice totals per class: (annotations, labeled regions). Region
        counts come from the per-slice resolution caches, so only slices that
        carry interactions AND a computed record contribute (the rest have
        nothing to count yet)."""
        import numpy as np
        annot = {}
        for it in self.store.interactions:
            annot[it.class_id] = annot.get(it.class_id, 0) + 1
        # Per ITEM (a gesture is the slide's, and every item that sees it
        # resolves it on its own decomposition), over the items that have a
        # record and see at least one gesture.
        regions = {}
        for key in self.catalogue.keys():
            pos = self.catalogue.index_of(key)
            if pos is None:
                continue
            si, li = pos
            rec = self.regions.record(key)
            if rec is None or rec.get("labels") is None or not self._gestures_for_key(key):
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
                # A named class shows its name after the id.
                head = f"{k} {self.store.name(k)}" if self.store.has_name(k) else f"{k}"
                lbl.configure(text=f"{head} · {annot.get(k, 0)}a · "
                                   f"{regions.get(k, 0)}r")
            except tk.TclError:
                pass

    def _clear_class_guarded(self, k):
        """Delete every annotation of class `k`, across every item, after
        asking. One undo step."""
        its = self.store.for_class(k)
        if not its:
            self.status_var.set(f"Class {k} has no annotations.")
            return False
        n_items = len({it.slice_key for it in its})
        noun = self.ITEM_NOUN
        if not messagebox.askyesno(
                self.APP_TITLE,
                f"Delete all {len(its)} class {k} annotation(s), on {n_items} "
                f"{noun}(s)?\n\nCtrl+Z brings them back."):
            return False
        n = self._remove_interactions([it.uid for it in its])
        self.status_var.set(f"Class {k}: deleted {n} annotation(s) on {n_items} "
                            f"{noun}(s) (Ctrl+Z restores them)")
        return True

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

    def _rename_class(self, k, name=None):
        """Name class `k` in the active task's vocabulary (display only: ids
        stay the wire format). `name` given -> no dialog (headless); an
        empty name clears it. Undoable, like a colour change."""
        if name is None:
            from tkinter import simpledialog
            current = self.store.names.get(int(k), "")
            name = simpledialog.askstring(self.APP_TITLE, f"Name for class {k}:",
                                          initialvalue=current, parent=self.root)
            if name is None:
                return False
        name = str(name).strip()
        if (self.store.names.get(int(k)) or "") == name:
            return False
        self._push_history()
        self.store.set_name(k, name)
        self._rebuild_class_panels()
        return True

    def _interaction_row_text(self, it):
        """The row's label. The rows normally all belong to the current slice,
        so the key is noise -- except when it is NOT this slice's (nothing on
        screen, so _visible_interactions lists everything) or failed to bind at
        all. Depends on the current slice, so a surviving row can need it
        rewritten even though the interaction did not change."""
        cur = self._current()
        if not it.bound:
            where = f"  [{it.slice_key} (unbound)]"
        elif cur is None or not self._gesture_on_item(it, *cur):
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
        return f"#{it.uid} {name}{where}"

    def _update_interaction_row(self, row, it):
        """Refresh a surviving row in place (text and bound-ness only -- the
        uid its callbacks close over cannot change)."""
        text, fg = self._interaction_row_text(it), ("#000" if it.bound else "#888")
        if (text, fg) != (row["text"], row["fg"]):
            row["label"].configure(text=text, foreground=fg)
            row["text"], row["fg"] = text, fg

    def _build_interaction_row(self, parent, it):
        """One interaction row, as a record ``{frame, label, class, text, fg}``
        so _sync_interaction_rows can keep it instead of rebuilding it."""
        # Plain-tk widgets so the rows share the list's white background.
        row = tk.Frame(parent, background="white")
        row.pack(**self._ROW_PACK)
        # Delete on the LEFT: a narrow pane truncates the label, and a
        # right-packed ✕ is the first thing to disappear with it.
        tk.Button(row, text="✕", width=2, relief="flat", background="white",
                  activebackground="#ddd", padx=0, pady=0,
                  command=lambda uid=it.uid: self._delete_interaction(uid)
                  ).pack(side="left")
        text, fg = self._interaction_row_text(it), ("#000" if it.bound else "#888")
        lbl = tk.Label(row, text=text, foreground=fg,
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
        return {"frame": row, "label": lbl, "class": int(it.class_id),
                "text": text, "fg": fg}

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
        rec = self.regions.record(self.catalogue.key_of(si, li))
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
        for it in self._gestures_for(si, li):
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
        cur = self._current()
        if cur is None or not self._gesture_on_item(it, *cur):
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
        for it in self._visible_seam_gestures():
            self._draw_seam_geometry(it, tags=("draw", "ipersist"))
        # Every one of these is a Tk canvas item that the next window redraw
        # has to walk -- a magic fill commits one tap per region, so the count
        # is worth seeing beside the frame time (see labeler/perf.py).
        v.perf.set("annot_items", len(v.canvas.find_withtag("ipersist")))

    def _redraw_hover_geometry(self):
        """Re-project whatever hover geometry is on screen after a zoom/pan
        (the items are drawn in screen coordinates). Runs once per motion
        event, NOT once per painted frame, so its cost is charged to the
        profiler's between-frames bucket (see labeler/perf.py)."""
        v = self.viewer
        if v is None:
            return
        with v.perf.span("annot.persist"):
            self._refresh_annotation_layer()
            tool = getattr(v, "tool", None)
            if tool is not None and hasattr(tool, "redraw"):
                tool.redraw()          # the trace in flight, in screen space
        if self._hover_uid is None and self._hover_key is None:
            return
        with v.perf.span("annot.hover"):
            v.canvas.delete("ihover")
            if self._hover_uid is not None:
                self._draw_interaction_geometry(self.store.get(self._hover_uid))
                return
            si, li, region = self._hover_key
            rec = self.regions.record(self.catalogue.key_of(si, li))
            if rec is None or rec.get("labels") is None:
                return
            import numpy as np
            touch = self._touch_map_for(si, li, rec, np)
            for it in self._gestures_for(si, li):
                ids = touch.get(it.uid)
                if ids and region in ids:
                    self._draw_interaction_geometry(it)

    def _on_hover(self, ix, iy=None):
        """Show image values/probabilities and highlight annotations for a region."""
        super()._on_hover(ix, iy)
        v = self.viewer
        if v is None:
            return
        # A trace in flight owns the pointer: the path from its anchor to the
        # seam point under the cursor, instead of the region outlines.
        tc = getattr(v.tool, "trace", None)
        if tc is not None and tc.active:
            tc.on_hover(ix, iy)
            return
        cur = self._current()
        region = None
        rec = None
        if ix is not None and iy is not None and cur is not None:
            rec = self.regions.record(self.catalogue.key_of(*cur))
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
                        pred = self._pred.get(self.catalogue.key_of(*cur))
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
        for it in self._gestures_for(*cur):
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
        cur = self._current()
        if cur is None or not self._gesture_on_item(it, *cur):
            # Its hints name the slide's first row (the overview), which
            # always covers the gesture.
            try:
                idx = self.flat_slices.index((it.si, it.li))
            except ValueError:
                return
            self._goto_slice(idx)
        cx = sum(x for x, _y in it.points) / len(it.points)
        cy = sum(y for _x, y in it.points) / len(it.points)
        w = max(v.canvas.winfo_width(), 1)
        h = max(v.canvas.winfo_height(), 1)
        v.set_view(cx - (w / 2) * v.scale, cy - (h / 2) * v.scale)   # zoom stays as-is
        v.render()
        self._show_interaction_geometry(uid)  # redraw at the new view
