"""``AnnotationShell``: the generic labeler layer over a ``ViewerShell``.

Composed as ``class MyLabeler(AnnotationShell, MyViewer)``, it adds the
annotation store with undo/redo, the drawing tools and their previews, the
three-pane layout with the tabbed centre, hotkeys, the classifier lifecycle
and the session / New-session additions -- everything that does not depend
on what an item or a region is. The clusters live in the panel mixins and
``classifier.py``; this class holds the state, the layout hooks, the
annotation glue and the session document. The hooks at the end are what a
derived labeler implements; ``msseg.mscoupon.labeler.LabelerApp`` is the
reference.
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
from .tools import DrawController, MagicFillController, _extremum_points
from .classifier import ClassifierMixin
from .panels.hints import HintsMixin
from .panels.model import ModelPanelMixin
from .panels.analysis import AnalysisPanelMixin
from .panels.view import ViewControlsMixin
from .panels.classpanel import ClassPanelMixin


class AnnotationShell(HintsMixin, ModelPanelMixin, AnalysisPanelMixin, ViewControlsMixin,
                      ClassPanelMixin, ClassifierMixin):
    SESSION_APP = "labeler"
    APP_TITLE = "labeler"
    WINDOW_TITLE = "labeler"
    MODEL_APP_TAG = model_bundle.DEFAULT_APP_TAG     # the classifier pickle "app" tag

    def __init__(self, root, initial=None, autosave=True):
        # Labeler state first: the base __init__ calls the overridden build
        # methods, which read these.
        self.store = LabelStore()
        self._training_builder = TrainingSetBuilder(self.FIELDS)
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
        self._models_dir = os.path.join(self.SESSION_IO.app_data_dir(self.SESSION_APP), "models")
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
        self._class_lists = {}          # class_id -> ScrollFrame of its rows
        self._row_widgets = {}          # interaction uid -> row record
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
        root.title(self.WINDOW_TITLE)
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

    def _annotation_count(self, si, li):
        key = self._slice_key(si, li)
        return len(self.store.for_slice(key)) if key else 0

    def _slice_key(self, si, li):
        """The item key of slice (si, li): the catalogue's folder-qualified
        "folder/basename" (see adapters.SequenceCatalogue)."""
        return self.catalogue.key_of(si, li)

    def _current_key(self):
        cur = self._current()
        return self.catalogue.key_of(*cur) if cur is not None else None

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
        v.invalidate()

    def _end_preview(self):
        self._hover_suppressed = False
        v = self.viewer
        if v is not None:
            v.set_transient(None)
            v.invalidate()

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
        rec = self.regions.record(self.catalogue.key_of(si, li))
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
        rec = self.regions.record(self.catalogue.key_of(si, li))
        pr = self._pred.get(self.catalogue.key_of(si, li))
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
        doc = self.SESSION_IO.read_json_file(path)
        if doc is None:
            self.status_var.set(f"Could not read {path}")
            return
        try:
            store = LabelStore.from_json(doc)
        except Exception as exc:
            self.status_var.set(f"Not an annotations.json: {exc}")
            return
        self._install_store(store)

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
        # The PICKED kind, apart from the trained model: a plain Train writes
        # no pickle, so a restore that reloads the newest saved model would
        # otherwise land on that pickle's kind (e.g. the last Optimize winner).
        d["model_kind"] = self.model_kind_var.get()
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

    # -- new session ---------------------------------------------------- #
    def _new_session_options(self):
        return super()._new_session_options() + [(
            "model",
            "Keep model selection & settings (kind, loaded model, edge and search settings)",
            "The picked kind, the model in memory and its saved-model records, the "
            "custom base / edge settings and the Optimize settings carry over. Off: no "
            "model, dense FC selected, default settings.")]

    def _new_session_blurb(self):
        return ("Start a new session: no folders, no sequences, nothing computed, "
                "no annotations (class count and colours stay).")

    def _new_session_doc(self, keep):
        doc = super()._new_session_doc(keep)
        empty = self.store.to_json()
        empty["interactions"] = []            # classes + colours, no gestures
        doc["annotations"] = empty
        keep_model = keep.get("model", True)
        doc["models"] = [dict(m) for m in self.models] if keep_model else []
        if not keep_model:
            for key in ("model_search", "neighbours", "model_kind"):
                doc["view"].pop(key, None)
        # The in-memory model is not in the document (a plain Train writes
        # no pickle): stash it so the apply's pickle reload cannot replace it.
        self._new_session_stash = ((self._clf, self._clf_names, self._clf_kind,
                                    self._clf_spec, self._edge_model, self._search_spec)
                                   if keep_model else None)
        return doc

    def _after_new_session(self, keep):
        stash = getattr(self, "_new_session_stash", None)
        self._new_session_stash = None
        if keep.get("model", True):
            if stash is not None and stash[0] is not None:
                (self._clf, self._clf_names, self._clf_kind, self._clf_spec,
                 self._edge_model, self._search_spec) = stash
                self.classify_btn.config(state="normal")
        else:
            self._reset_model_selection()
        self._pred.clear()
        self._cm_cell = None
        self._refresh_region_modes()
        self._refresh_confusion()
        self._refresh_model_readout()
        self._refresh_model_strip()
        self._refresh_edge_readout()
        self._fill_error_list(None)

    def _reset_model_selection(self):
        """No model, the default kind, default edge / search settings."""
        self._clf = None
        self._clf_names = None
        self._clf_kind = "dense FC"
        self._clf_spec = None
        self._edge_model = None
        self._search_spec = None
        self.models = []
        self.model_kind_var.set("dense FC")
        self.custom_hidden_var.set(_DEFAULT_CUSTOM_HIDDEN)
        self.freeze_base_var.set(False)
        self._apply_edge_spec(edge_model.EdgeSpec())
        self.search_trials_var.set(str(_SEARCH_TRIALS))
        self.search_timeout_var.set(str(_SEARCH_TIMEOUT_MIN))
        self.search_seed_var.set("0")
        self.search_features_var.set(True)
        self.search_backend_var.set("auto")
        self.classify_btn.config(state="disabled")

    def _apply_session_doc(self, doc, source="session", notes=None):
        notes = super()._apply_session_doc(doc, source,
                                           notes if notes is not None else [])
        sdoc = self._session_doc_from_json(doc, [])
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
        # The picked kind wins over the reloaded pickle's kind (the model
        # reload above set model_kind_var from the pickle).
        if view.get("model_kind") in _MODEL_KINDS:
            self.model_kind_var.set(view["model_kind"])
        # Keep the regions toggle in sync with whatever seg_source restored to.
        self.show_regions_var.set(self._region_layer_visible())
        # The center tab, by name; an unknown or missing value leaves it alone.
        tab = self._center_tabs.get(view.get("center_tab"))
        if tab is not None:
            try:
                self.center.select(tab)
            except tk.TclError:
                pass
        return notes

    # ------------------------------------------------------------------ #
    # Hooks with defaults (what the app's profile and compute look like)
    # ------------------------------------------------------------------ #
    FIELDS = fields.DEFAULT           # column conventions of the statistics table

    def _workflow_summary(self, profile):
        """The active profile as one or two compact lines for the Run hint."""
        return str(profile.get("name", "?"))

    def _workflow_hint_tooltip(self):
        return "The active profile. Click to open the Processing tab."

    def _set_region_layer_visible(self, on):
        """Show or hide the region layer under the class layer (the app's)."""

    def _region_layer_visible(self):
        return True

    def _on_regions_toggle(self):
        self._set_region_layer_visible(self.show_regions_var.get())
        self._refresh_render()

    def _stats_brief(self, stats):
        """One line describing a statistics block (for the model strip)."""
        return "?" if not stats else f"{len(stats)} keys"

    def _expected_feature_names(self):
        """The feature names the ACTIVE profile produces, or None to skip the
        compatibility gate (the app knows its statistics schema)."""
        return None

    def _feature_schema_now(self):
        """[{name, channel, reduction}] for the active profile, or None."""
        return None

    def _profile_from_model(self, path, statistics):
        """Offer a profile built from a loaded model's statistics (the app's)."""
