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

from . import context, edge_model, magic_fill, model_search, fields
from . import bundle as model_bundle
from .labeling import (LabelStore, MAX_CLASSES, TOOLS, resolve_slice, resolve_sets,
                          touched_sets, class_lut, scalar_lut, line_pixels, polygon_mask,
                          preview_lut)
from .training import TrainingSetBuilder, TrainingProblem
from .widgets import ScrollFrame, Collapsible, attach_tooltip
from .defaults import *  # noqa: F401,F403
from .tools import DrawController, MagicFillController, _extremum_points
from .classifier import ClassifierMixin
from .panels.hints import HintsMixin
from .panels.model import ModelPanelMixin
from .panels.analysis import AnalysisPanelMixin
from .panels.view import ViewControlsMixin
from .panels.classpanel import ClassPanelMixin
from .panels.seams import SeamPanelMixin
from .seam_classifier import SeamModelMixin
from .seams import SEAM_BOUNDARY


class AnnotationShell(HintsMixin, ModelPanelMixin, AnalysisPanelMixin, ViewControlsMixin,
                      ClassPanelMixin, SeamPanelMixin, SeamModelMixin, ClassifierMixin):
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
        self._clf_scope = None          # the regime it was fitted in (see _feature_scope)
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
        self.edge_feat_vars = {f: tk.BooleanVar(master=root, value=f in edge_model.DEFAULT_FEATURES)
                               for f in edge_model.FEATURE_KINDS}
        self.edge_model_var = tk.StringVar(master=root, value="logistic")
        self.edge_c_var = tk.StringVar(master=root, value="1")
        self.edge_lam_var = tk.StringVar(master=root, value="1")
        self.edge_rounds_var = tk.StringVar(master=root, value="3")
        self.edge_readout_var = tk.StringVar(master=root, value="")
        self.edge_progress_var = tk.StringVar(master=root, value="")
        # Neighbourhood context (context.py): the columns the CURRENT model was
        # fit over ride with it (predictions and the gate rebuild exactly
        # those); the Model tab's controls describe the NEXT model.
        self._clf_context = context.ContextSpec()
        self.context_kind_vars = {k: tk.BooleanVar(master=root, value=False)
                                  for k in context.KINDS}
        self.context_weight_vars = {w: tk.BooleanVar(master=root, value=(w == "uniform"))
                                    for w in context.WEIGHTS}
        self.context_source_var = tk.StringVar(master=root, value="all")
        # The latent-ring head (context.LatentContextModel) fit on top of the
        # base net, and the controls that describe the next one.
        self._context_model = None
        self.context_latent_var = tk.BooleanVar(master=root, value=False)
        self.context_latent_weight_var = tk.StringVar(master=root, value="uniform")
        self.context_latent_layer_var = tk.StringVar(master=root, value="last")
        self.context_latent_h0_var = tk.BooleanVar(master=root, value=True)
        # Labels as context (context.LabelSpec): the ring's annotated classes
        # as columns, with a training-time dropout.
        self.context_labels_var = tk.BooleanVar(master=root, value=False)
        self.context_label_dropout_var = tk.StringVar(master=root, value="0.3")
        self._pred_store_rev = None
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
        # Seam tools (docs/seam_labeling.md): the class a trace / scope
        # paints, the trace's toll and channels, the overlay and its colouring,
        # and the per-item caches the overlay and the tools share.
        self.seam_class_var = tk.IntVar(master=root, value=SEAM_BOUNDARY)
        self.seam_toll_var = tk.StringVar(master=root, value="feature")
        self.seam_channels_var = tk.StringVar(master=root, value="base")
        self.show_seams_var = tk.BooleanVar(master=root, value=True)
        self.seam_color_var = tk.StringVar(master=root, value=_SEAM_MODE_CLASS)
        self._seam_caches = {}      # (si, li) -> (commit, rev, class[S], sets, graph)
        self._seam_rasters = {}     # (si, li) -> (commit, graph, seam-index raster)
        self._seam_pred = {}        # key -> (commit, boundaryness[S])
        self._seam_model = None     # seam_model.SeamModel, or None
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
        # Set while the first render of a run happened on an unmapped canvas:
        # the base fits the image only on that first render, and an unmapped
        # canvas is 1x1, so the fit is redone once it is on screen.
        self._fit_pending = False
        # Which Processing groups are folded shut. Read by _group as each is
        # built, so a session restored before the panel exists still lands.
        self._proc_open = {}
        self._proc_groups = {}
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
    # Three columns: data | picture | tabs
    # ------------------------------------------------------------------ #
    # Left is data navigation and processing SELECTION, the middle is the
    # slice (the inherited `self.right`, still a plain frame, so every
    # viewer-area builder packs into it unchanged), and the right is a
    # notebook of everything you EDIT: the compute profile, the annotations
    # and their classes, the model, the analyses. Roughly 1:3:2, and ONLY the
    # middle carries a weight -- window growth goes to the picture, and the
    # two side columns keep the width they were given.
    #
    # The profile and the picture used to be sibling tabs, which made judging
    # a filter chain a round trip: edit, switch, squint, switch back. The
    # chain IS judged by looking, so the picture is never the thing that is
    # hidden; the sashes (and F9) decide how much of it there is.
    def _build_center(self):
        self.right = ttk.Frame(self.paned, width=900)
        self.paned.add(self.right, weight=1)
        self.center = ttk.Notebook(self.paned, width=420)
        self.paned.add(self.center, weight=0)
        self.processing_tab = ttk.Frame(self.center)
        # Annotation: what the user draws and the classifier that learns it.
        # Filled by _build_label_panel after the base constructor returns.
        self.annot_tab = ttk.Frame(self.center)
        self.model_tab = ttk.Frame(self.center)
        # Analysis: what the models DO with the annotations -- the region
        # list behind a confusion cell, the size sweep -- and, later, plots.
        self.analysis_tab = ttk.Frame(self.center)
        self._center_tabs = {"Processing": self.processing_tab,
                             "Annotation": self.annot_tab,
                             "Model": self.model_tab,
                             "Analysis": self.analysis_tab}
        for name in _CENTER_TABS:
            self.center.add(self._center_tabs[name], text=name)

        # Processing: the profile management rows on top (anchored, not
        # filled -- the base rows pack fill="x" and would stretch across the
        # whole tab), then the parameter sections as ONE scrolling column of
        # collapsible groups. Two columns needed a tab as wide as the window;
        # a column that folds needs only the group being edited.
        self.profile_tools = ttk.Frame(self.processing_tab)
        self.profile_tools.pack(side="top", anchor="w", padx=6, pady=(4, 0))
        self.proc_scroll = ScrollFrame(self.processing_tab, width=420,
                                       canvas_width=400)
        self.proc_scroll.pack(side="top", fill="both", expand=True)
        self.proc_col = self.proc_scroll.inner
        self._proc_groups = {}               # key -> Collapsible

        self._build_model_tab(self.model_tab)
        self._build_analysis_tab(self.analysis_tab)
        self.center.select(self.processing_tab)
        # Bound AFTER the initial select: <<NotebookTabChanged>> fires
        # synchronously on select(), and the viewer does not exist yet.
        self.center.bind("<<NotebookTabChanged>>", self._on_center_tab_changed)
        # No tab change is guaranteed to arrive any more, so the canvas being
        # mapped is what finishes a fit deferred while the window was not.
        self.right.bind("<Map>", self._flush_pending_fit, add="+")

    _DEFAULT_PANES = _LABELER_PANES          # 1:3:2

    def _tab_scroll(self, parent):
        """A scrolling body for a tab whose panels stack taller than the
        column they now live in."""
        sf = ScrollFrame(parent, width=420, canvas_width=400)
        sf.pack(fill="both", expand=True)
        return sf.inner

    def _group(self, parent, text, key=None):
        """A collapsible group: the Processing tab is one tall column, so a
        section you are not editing folds away behind its title."""
        key = key or text
        c = Collapsible(parent, text=text, open=self._proc_open.get(key, True),
                        on_toggle=lambda shown, k=key: self._proc_open.__setitem__(k, shown))
        c.pack(fill="x", padx=4, pady=(0, 2))
        self._proc_groups[key] = c
        return c.body

    def _proc_open_state(self):
        """{group key: open} for the groups that are SHUT -- the default is
        open, so a session only records what was folded away."""
        return {k: False for k, c in self._proc_groups.items() if not c.is_open()}

    def _apply_proc_open(self, state):
        if not isinstance(state, dict):
            return
        for key, shown in state.items():
            self._proc_open[key] = bool(shown)
            group = self._proc_groups.get(key)
            if group is not None:
                group.set_open(bool(shown))

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
        # Only what the session section actually packed: pack_configure on a
        # widget that was never packed would PACK it, and an app whose session
        # browser is a single list of files (mspath) keeps the shell's folder
        # and file lists as unshown bookkeeping widgets.
        def packed(w):
            try:
                return w.winfo_manager() == "pack"
            except tk.TclError:
                return False
        for lb in (self.folder_list, self.file_list, self.subseq_list):
            if packed(lb.master):
                lb.master.pack_configure(fill="both", expand=True)
        for btn, above in ((self.folder_btn_row, self.folder_list.master),
                           (self.make_seq_btn, self.file_list.master),
                           (self.seq_btn_row, self.subseq_list.master)):
            if packed(btn) and packed(above):
                btn.pack_configure(side="bottom", before=above)
        self._build_workflow_hint()

    def _processing_parent(self, section):
        return self.proc_col

    def _handle_event(self, ev):
        if ev[0] == "primed":
            # The commit bump already invalidates these; dropping them outright
            # also frees the old run's arrays.
            self._class_luts.clear()
            self._clear_seam_caches()
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

    def _commit_seam(self, tool, points, class_id, meta=None):
        """Store a seam gesture (a trace or a scope) on the current item."""
        from .seams import SEAM_CLASSES
        cur = self._current()
        if cur is None:
            return
        si, li = cur
        slice_key = self._slice_key(si, li)
        if slice_key is None or int(class_id) < 1:
            return
        self._push_history()
        it = self.store.add_seam(tool, points, int(class_id), slice_key, si, li, meta=meta)
        self._rebuild_class_panels()
        self._refresh_render()
        name = SEAM_CLASSES[it.class_id] if it.class_id < len(SEAM_CLASSES) else str(it.class_id)
        n = (meta or {}).get("seams")
        self.status_var.set(f"#{it.uid} {tool} -> {name}"
                            + (f" ({n} seams)" if n is not None else "") + f" ({slice_key})")

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
        v.set_transient(self._region_overlay(
            labels, preview_lut(K, ids, colors, np, emphasize=emphasize), np))
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
            pts = _extremum_points(labels, ids, table, np, self.FIELDS,
                                   self._region_placement())
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

    def _remove_interactions(self, uids):
        """Delete several gestures as ONE undo step. Returns the count."""
        uids = [u for u in uids if self.store.get(u) is not None]
        if not uids:
            return 0
        self._push_history()
        n = self.store.remove_many(uids)
        self._rebuild_class_panels()
        self._refresh_render()
        return n

    # -- the sequence tree: annotations ride with their rows -------------- #
    def _row_interactions(self, si, li):
        """Every interaction on a tree row (an item's own; all of a
        sequence's items' for a sequence row -- and, where the app says so,
        those on items of the sequence that are no longer listed)."""
        return [it for it in self.store.interactions + self.store.seams
                if self._row_owns_key(si, li, it.slice_key)]

    def _seq_tree_menu_entries(self, si, li):
        entries = super()._seq_tree_menu_entries(si, li)
        n = len(self._row_interactions(si, li))
        clear = (f"Clear annotations… ({n})" if n else "Clear annotations",
                 lambda: self._clear_row_annotations_guarded(si, li), n > 0)
        entries.insert(len(entries) - 1, clear)      # ahead of Remove, which is last
        return entries

    def _clear_row_annotations_guarded(self, si, li):
        """Delete every annotation on a row (the row itself stays), after
        asking. One undo step."""
        its = self._row_interactions(si, li)
        desc = self._row_description(si, li)
        if not its:
            self.status_var.set(f"No annotations on {desc}.")
            return False
        if not messagebox.askyesno(
                self.APP_TITLE,
                f"Delete the {len(its)} annotation(s) on {desc}?\n\n"
                "Ctrl+Z brings them back."):
            return False
        n = self._remove_interactions([it.uid for it in its])
        self.status_var.set(f"Deleted {n} annotation(s) on {desc} (Ctrl+Z restores them)")
        return True

    def _key_survives(self, key, rows):
        """True when a row NOT in `rows` still owns `key` -- the coupon can
        hold one slice in two sequences, and the key is the slice, so the
        annotations stay as long as either sequence does."""
        gone_seqs = {si for si, li in rows if li is None}
        gone_items = {(si, li) for si, li in rows if li is not None}
        for si in range(len(self.subsequences)):
            if si in gone_seqs:
                continue
            for li in range(len(self._sequence_item_labels(si))):
                if (si, li) not in gone_items and self._row_owns_key(si, li, key):
                    return True
        return False

    def _doomed_interactions(self, rows):
        """The interactions a removal takes with it: those on the rows whose
        item no longer exists afterwards."""
        out, seen = [], set()
        for it in self.store.interactions + self.store.seams:
            if it.uid in seen:
                continue
            if any(self._row_owns_key(si, li, it.slice_key) for si, li in rows):
                seen.add(it.uid)
                if not self._key_survives(it.slice_key, rows):
                    out.append(it)
        return out

    def _remove_rows_message(self, rows):
        msg = super()._remove_rows_message(rows)
        n = len(self._doomed_interactions(rows))
        if not n:
            return msg + "\n\nNo annotations are on it."
        return (msg + f"\n\nThe {n} annotation(s) on it go too (Ctrl+Z brings the "
                      "annotations back, greyed until the data is added again).")

    def _remove_rows(self, rows):
        doomed = [it.uid for it in self._doomed_interactions(rows)]
        if doomed:
            self._push_history()
            self.store.remove_many(doomed)
        n = super()._remove_rows(rows)
        if doomed:
            self.status_var.set(f"Removed {n} row(s) and {len(doomed)} annotation(s) "
                                "(Ctrl+Z restores the annotations)")
        return n

    def _after_rows_removed(self, cur_key, pos):
        # Indices shifted under every (si, li)-keyed cache and hint: the LUTs
        # go, the store's hints rebind through the catalogue once the rebuild
        # has refreshed it, and the predictions keep only the keys that still
        # exist.
        self._class_luts.clear()
        self._clear_seam_caches()
        self._hover_key = None
        self._hover_uid = None
        super()._after_rows_removed(cur_key, pos)
        self.store.rebind(self.subsequences, resolve=self.catalogue.index_of)
        for key in [k for k in self._pred if self.catalogue.index_of(k) is None]:
            self._pred.pop(key, None)
        self._rebuild_class_panels()

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
        # The file lists bind "folder/basename" keys; the catalogue binds
        # whatever else an app calls an item (a slide's "...@level#rect").
        unbound = store.rebind(self.subsequences, resolve=self.catalogue.index_of)
        self._class_luts.clear()
        self._clear_seam_caches()
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
        # Seam tools: T trace, S scope, E seam layer on/off, Enter commits the
        # trace in flight, BackSpace drops its last leg.
        self.root.bind("t", self._on_trace_key)
        self.root.bind("T", self._on_trace_key)
        self.root.bind("s", self._on_scope_key)
        self.root.bind("S", self._on_scope_key)
        self.root.bind("e", self._on_seams_toggle_key)
        self.root.bind("E", self._on_seams_toggle_key)
        self.root.bind("<Return>", self._on_trace_commit_key)
        self.root.bind("<BackSpace>", self._on_trace_back_key)
        # F9: the canvas takes the whole center, and back. A function key
        # needs no _typing() guard -- and every letter is already taken.
        self.root.bind("<F9>", self._toggle_center_tabs)

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

    # _unfocus_entries and _typing live on ViewerShell (the viewer's own
    # hotkey needs them too) and are inherited here.

    def _on_magic_key(self, _e=None):
        if self._typing():
            return
        self.tool_var.set("magic")

    def _on_blob_key(self, _e=None):
        if self._typing():
            return
        self.tool_var.set("blobber")

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
        d["seams"] = self._seams_view_state()
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
        # The sashes as FRACTIONS, not pixels: a session restored into a
        # differently sized window should divide it the same way. The folded
        # Processing groups ride along -- the panel is one column, so which
        # groups are out of the way is part of how it is set up.
        d["panes"] = self._pane_fractions()
        d["proc_open"] = self._proc_open_state()
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
        # The PICKED context (what the next Train builds), apart from the
        # trained model's, which rides its pickle.
        d["context"] = self._context_spec_from_ui().to_dict()
        return d

    def _apply_context_view(self, d):
        if isinstance(d, dict):
            self._apply_context_spec(context.ContextSpec.from_dict(d))

    def _rebuild_class_panels(self):
        # The single "labels changed" signal: with labels in the model's
        # context the cached predictions are stale (ClassifierMixin).
        super()._rebuild_class_panels()
        self._labels_changed()
        self._refresh_seam_panel()

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
        empty.pop("seams", None)              # ...and no seam gestures: a v2 doc
        empty["version"] = 2
        doc["annotations"] = empty
        keep_model = keep.get("model", True)
        doc["models"] = [dict(m) for m in self.models] if keep_model else []
        if not keep_model:
            for key in ("model_search", "neighbours", "model_kind", "context"):
                doc["view"].pop(key, None)
        # The in-memory model is not in the document (a plain Train writes
        # no pickle): stash it so the apply's pickle reload cannot replace it.
        self._new_session_stash = ((self._clf, self._clf_names, self._clf_kind,
                                    self._clf_spec, self._edge_model, self._search_spec,
                                    self._clf_scope, self._clf_context, self._context_model)
                                   if keep_model else None)
        self._new_session_seam_stash = self._seam_model if keep_model else None
        return doc

    def _after_new_session(self, keep):
        stash = getattr(self, "_new_session_stash", None)
        self._new_session_stash = None
        if keep.get("model", True):
            if stash is not None and stash[0] is not None:
                (self._clf, self._clf_names, self._clf_kind, self._clf_spec,
                 self._edge_model, self._search_spec, self._clf_scope,
                 self._clf_context, self._context_model) = stash
                self.classify_btn.config(state="normal")
            self._seam_model = getattr(self, "_new_session_seam_stash", None)
        else:
            self._reset_model_selection()
        self._pred.clear()
        self._seam_pred.clear()
        if not keep.get("model", True):
            self._seam_model = None
        self._cm_cell = None
        self._refresh_region_modes()
        self._refresh_confusion()
        self._refresh_model_readout()
        self._refresh_model_strip()
        self._refresh_edge_readout()
        self._fill_error_list(None)
        self._refresh_seam_panel()

    def _reset_model_selection(self):
        """No model, the default kind, default edge / search settings."""
        self._clf = None
        self._clf_names = None
        self._clf_scope = None
        self._clf_kind = "dense FC"
        self._clf_spec = None
        self._edge_model = None
        self._search_spec = None
        self._clf_context = context.ContextSpec()
        self._context_model = None
        self.models = []
        self.model_kind_var.set("dense FC")
        self.custom_hidden_var.set(_DEFAULT_CUSTOM_HIDDEN)
        self.freeze_base_var.set(False)
        self._apply_edge_spec(edge_model.EdgeSpec())
        self._apply_context_spec(context.ContextSpec())
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
        self._apply_seams_view(view.get("seams"))
        self._apply_search_view(view.get("model_search"))
        self._apply_neighbours_view(view.get("neighbours"))
        # The picked context wins over the reloaded pickle's (as the kind does).
        self._apply_context_view(view.get("context"))
        # The picked kind wins over the reloaded pickle's kind (the model
        # reload above set model_kind_var from the pickle).
        if view.get("model_kind") in _MODEL_KINDS:
            self.model_kind_var.set(view["model_kind"])
        # Keep the regions toggle in sync with whatever seg_source restored to.
        self.show_regions_var.set(self._region_layer_visible())
        # The center tab, by name; an unknown or missing value leaves it alone
        # (which is also how a pre-split session's "View" is handled).
        if isinstance(view.get("panes"), (list, tuple)):
            self._apply_pane_fractions(view["panes"])
        self._apply_proc_open(view.get("proc_open"))
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
