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
from . import session_doc
from .labeling import (LabelStore, MAX_CLASSES, TOOLS, resolve_slice, resolve_sets,
                          touched_sets, class_lut, scalar_lut, line_pixels, polygon_mask,
                          preview_lut, gesture_meets)
from .training import TrainingSetBuilder, TrainingProblem
from .widgets import ScrollFrame, Collapsible, attach_tooltip
from .defaults import *  # noqa: F401,F403
from .tools import DrawController, MagicFillController, _extremum_points
from .classifier import ClassifierMixin
from .panels.hints import HintsMixin
from .panels.stages import StagesMixin
from .panels.model import ModelPanelMixin
from .panels.analysis import AnalysisPanelMixin
from .panels.view import ViewControlsMixin
from .panels.classpanel import ClassPanelMixin
from .panels.seams import SeamPanelMixin
from .seam_classifier import SeamModelMixin
from .seams import SEAM_BOUNDARY
from .task import TASK_VIEW_KEYS, ModelStack, Task, TaskCaches, dedupe_task_name


# --------------------------------------------------------------------------- #
# The active task's state, under the names every mixin has always read.
#
# A session holds several tasks (msseg.labeler.task) and exactly one is
# active. Rather than thread a task through ~150 call sites in the mixins,
# the tools and the selftests, the shell keeps the attribute names they use
# -- ``self.store``, ``self._clf``, ``self._pred``, ... -- as properties that
# read and write the active task. A task switch is then one assignment to
# ``self._task``; nothing downstream knows there is more than one.
# --------------------------------------------------------------------------- #
_UNSET = object()


def _task_attr(field, doc):
    def get(self):
        return getattr(self._task, field)

    def put(self, value):
        setattr(self._task, field, value)
    return property(get, put, doc=doc)


def _stack_attr(field, doc):
    def get(self):
        return getattr(self._task.model, field)

    def put(self, value):
        setattr(self._task.model, field, value)
    return property(get, put, doc=doc)


def _cache_attr(field, doc):
    def get(self):
        return getattr(self._task.caches, field)

    def put(self, value):
        setattr(self._task.caches, field, value)
    return property(get, put, doc=doc)


class AnnotationShell(StagesMixin, HintsMixin, ModelPanelMixin, AnalysisPanelMixin,
                      ViewControlsMixin, ClassPanelMixin, SeamPanelMixin, SeamModelMixin,
                      ClassifierMixin):
    SESSION_APP = "labeler"
    APP_TITLE = "labeler"
    WINDOW_TITLE = "labeler"
    MODEL_APP_TAG = model_bundle.DEFAULT_APP_TAG     # the classifier pickle "app" tag
    # The left column's profile section: in a labeler a profile is the active
    # task's WORKFLOW (the Combobox shows and rebinds it), so say so.
    PROFILE_SECTION_TITLE = "0. Workflow"

    # -- the active task's state (see the factories above) ------------------ #
    store = _task_attr("store", "The active task's LabelStore (gestures, seams, vocabulary).")
    models = _task_attr("models", "The active task's saved-model records, newest last.")
    _undo_stack = _task_attr("undo", "Store snapshots behind the active task's Ctrl+Z.")
    _redo_stack = _task_attr("redo", "Store snapshots ahead of it.")
    _clf = _stack_attr("clf", "The active task's fitted region pipeline.")
    _clf_names = _stack_attr("names", "Its feature-column order.")
    _clf_kind = _stack_attr("kind", "Its model kind.")
    _clf_spec = _stack_attr("spec", "Its tuned ModelSpec (dense (tuned) kinds).")
    _clf_scope = _stack_attr("scope", "The regime it was fitted in (see _feature_scope).")
    _clf_context = _stack_attr("context", "The ContextSpec its feature row was built with.")
    _context_model = _stack_attr("latent", "The latent-ring head fit on top of it.")
    _edge_model = _stack_attr("edge", "The edge model fit on top of it.")
    _search_spec = _stack_attr("search_spec", "The last Optimize winner (what Train rebuilds).")
    _seam_model = _stack_attr("seam", "The active task's seam model.")
    _pred = _cache_attr("pred", "(si, li) -> (commit, region_class, region_proba, aux).")
    _seam_pred = _cache_attr("seam_pred", "item key -> (commit, boundaryness[S]).")
    _pred_store_rev = _cache_attr("pred_store_rev", "The store rev the predictions were made under.")
    _cm_cell = _cache_attr("cm_cell", "The selected confusion cell (true, pred).")

    @property
    def _models_dir(self):
        """Where a finished search saves its winner so a restart (or the
        morning after an overnight run) finds it -- per task, so two tasks'
        autosaves never interleave. The selftest points this at a temp dir,
        and at None to switch the autosave off, so an explicit assignment
        wins over the per-task default."""
        if self._models_dir_override is not _UNSET:
            return self._models_dir_override
        return os.path.join(self.SESSION_IO.app_data_dir(self.SESSION_APP), "models",
                            self._task.uid)

    @_models_dir.setter
    def _models_dir(self, value):
        self._models_dir_override = value

    def __init__(self, root, initial=None, autosave=True):
        # The task first: every piece of labeler state below that belongs to
        # a detector -- store, vocabulary, model stack, prediction caches,
        # undo history -- lives on the active task, reached through the
        # properties above. The base __init__ calls the overridden build
        # methods, which read these. The first task's workflow binds once
        # the profiles exist (_build_task_section).
        self.tasks = [Task.new("task 1")]
        self._task = self.tasks[0]
        if self.ENROLMENT:
            # An app with enrolment starts every task working nothing: what a
            # task works is chosen (enrol / cut an ROI), never inherited.
            self._task.enrolled = {}
        self._models_dir_override = _UNSET
        self._task_rows_syncing = False
        self._training_builder = TrainingSetBuilder(self.FIELDS)
        # (si, li) -> (commit, store.rev, lut|None, {uid: touched id set});
        # one rasterization pass serves both the class layer and the
        # which-interactions-touch-this-region hover lookup. Keyed on the
        # store's rev with no task discriminator, so a task switch clears it.
        self._class_luts = {}
        # Likewise the context-column memo (_context_table): bounded to two
        # entries, keyed on the store's rev.
        self._ctx_cache = {}
        self._hover_key = None     # (si, li, region) whose geometry is on screen
        self._hover_uid = None     # row-hovered interaction whose geometry shows
        self.model_kind_var = tk.StringVar(master=root, value="dense FC")
        # The in-flight Optimize / sweep / evaluate worker: one per window,
        # not per task -- its finish installs into the ACTIVE task, which is
        # why _activate_task refuses while it runs.
        self._search = None            # {"queue","stop","thread","names","n","t0"}
        self.search_trials_var = tk.StringVar(master=root, value=str(_SEARCH_TRIALS))
        self.search_timeout_var = tk.StringVar(master=root, value=str(_SEARCH_TIMEOUT_MIN))
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
        # classifier pickle; on the task's stack), the custom base's hidden
        # sizes, whether Train keeps the base (to try edge variants on top),
        # and the edge spec.
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
        # fit over ride with it on the task's stack (predictions and the gate
        # rebuild exactly those); the Model tab's controls describe the NEXT
        # model.
        self.context_kind_vars = {k: tk.BooleanVar(master=root, value=False)
                                  for k in context.KINDS}
        self.context_weight_vars = {w: tk.BooleanVar(master=root, value=(w == "uniform"))
                                    for w in context.WEIGHTS}
        self.context_source_var = tk.StringVar(master=root, value="all")
        # The latent-ring head (context.LatentContextModel) fit on top of the
        # base net rides the task's stack; these controls describe the next one.
        self.context_latent_var = tk.BooleanVar(master=root, value=False)
        self.context_latent_weight_var = tk.StringVar(master=root, value="uniform")
        self.context_latent_layer_var = tk.StringVar(master=root, value="last")
        self.context_latent_h0_var = tk.BooleanVar(master=root, value=True)
        # Labels as context (context.LabelSpec): the ring's annotated classes
        # as columns, with a training-time dropout.
        self.context_labels_var = tk.BooleanVar(master=root, value=False)
        self.context_label_dropout_var = tk.StringVar(master=root, value="0.3")
        # Provenance line above the classifier controls: which model is loaded,
        # how wide its feature vector is, and whether it still agrees with the
        # active profile (a mismatch blocks Classify, so say so up front).
        self.model_strip_var = tk.StringVar(master=root, value="no model")
        # The saved-model records (`self.models`) and the per-item prediction
        # caches (`self._pred`) live on the task.
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
        # A released fill is an extent (derive.py) unless this is off.
        self.magic_extent_var = tk.BooleanVar(master=root, value=_DEFAULT_MAGIC_EXTENT)
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
        # (`_seam_pred` and `_seam_model` are the task's.)
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
        # whole-store snapshots on the task (`_undo_stack` / `_redo_stack`):
        # the store is small and a snapshot restore reuses the load path, so
        # history can never drift from reality.
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
        self.features_hint_var = tk.StringVar(master=root, value="")
        self._hint_after = None
        super().__init__(root, initial=initial, autosave=autosave)
        # The profiles exist now: the first task's workflow is the active one.
        if self._task.workflow is None and 0 <= self.active_profile_idx < len(self.profiles):
            self._task.workflow = self.profiles[self.active_profile_idx]["name"]
        self._update_task_rows()
        root.title(self.WINDOW_TITLE)
        self._build_label_panel()
        if self.viewer is not None:
            self.viewer.tool = DrawController(self)
            # Zoom/pan invalidates screen-space annotation geometry.
            self.viewer.on_view_changed = self._redraw_hover_geometry
            # Right-CLICK (not right-drag, which still pans) on the image
            # plane offers the same menu as an interaction row.
            self.viewer.on_context = self._canvas_menu
            # The stage strip's boxes open the tab that edits them.
            self.viewer.on_stage_click = self._on_stage_click
        self._bind_hotkeys()
        # A task kind offers its own tools; any other request (a hotkey, a
        # restored view, a test) falls back to the kind's first tool.
        self._coercing_tool = False
        self.tool_var.trace_add("write", self._coerce_tool)
        self._apply_task_kind()
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
        # Features: the statistics each region is measured by. Apart from
        # Processing because an edit here costs a re-measure of the item on
        # screen (the MSC is kept), where an edit there needs a Run.
        self.features_tab = ttk.Frame(self.center)
        # Annotation: what the user draws and the classifier that learns it.
        # Filled by _build_label_panel after the base constructor returns.
        self.annot_tab = ttk.Frame(self.center)
        self.model_tab = ttk.Frame(self.center)
        # Analysis: what the models DO with the annotations -- the region
        # list behind a confusion cell, the size sweep -- and, later, plots.
        self.analysis_tab = ttk.Frame(self.center)
        self._center_tabs = {"Processing": self.processing_tab,
                             "Features": self.features_tab,
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
        ttk.Label(self.features_tab, foreground="#666", wraplength=400, justify="left",
                  text="What each region is measured by: the base channel the "
                       "statistics are read from, and the statistics. An edit "
                       "re-measures the item on screen -- the regions stay, only "
                       "their statistics are rebuilt; no Run needed.").pack(side="top", anchor="w",
                                                        padx=6, pady=(4, 0))
        self.feat_scroll = ScrollFrame(self.features_tab, width=420, canvas_width=400)
        self.feat_scroll.pack(side="top", fill="both", expand=True)
        self.feat_col = self.feat_scroll.inner

        # The Model and Analysis tabs hold one body per task kind; the active
        # task's kind decides which is packed (_apply_task_kind).
        self._model_region = ttk.Frame(self.model_tab)
        self._model_region.pack(fill="both", expand=True)
        self._build_model_tab(self._model_region)
        self._model_polyline = ttk.Frame(self.model_tab)
        ttk.Label(self._model_polyline, foreground="#666", wraplength=380, justify="left",
                  text="The seam model is trained and evaluated from the Annotation "
                       "tab (Train seams / Evaluate).").pack(anchor="w", padx=8, pady=8)
        self._analysis_region = ttk.Frame(self.analysis_tab)
        self._analysis_region.pack(fill="both", expand=True)
        self._build_analysis_tab(self._analysis_region)
        self._analysis_polyline = ttk.Frame(self.analysis_tab)
        ttk.Label(self._analysis_polyline, foreground="#666", wraplength=380,
                  justify="left", text="No seam analyses yet.").pack(anchor="w", padx=8,
                                                                     pady=8)
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
        # The Tasks list sits between the workflow box and the session lists:
        # data below, what is being detected in it above.
        self._build_task_section()
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

    # The sections that measure rather than build the field: they go on the
    # Features tab.
    _FEATURE_SECTIONS = ("stats", "base")

    def _processing_parent(self, section):
        if section in self._FEATURE_SECTIONS:
            return self.feat_col
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
        if ev[0] in ("primed", "item_primed"):
            self._classify_on_arrival()       # the item on screen, if it can be

    def _update_busy(self):
        """The app's HUD line, then the stage strip (every engine event,
        preview and badge ends here, so the strip follows them all)."""
        super()._update_busy()
        self._refresh_stages()

    def _goto_slice(self, idx):
        # By ITEM key: two items of one slide share a gesture key, but list
        # different gestures (those meeting each item's rect).
        before = self._current_key()
        super()._goto_slice(idx)
        self._refresh_stages()
        after = self._current_key()
        if after != before:
            # The class panels list the ON-ITEM interactions only; swap them
            # with the item.
            self._rebuild_class_panels()
            self._refresh_edge_readout()
            cur = self._current()
            if cur is not None:
                self._coarse_notice(*cur)
        # Selected = classified (when a model is loaded and the item is
        # computed; an uncomputed item is classified when its prime lands).
        self._classify_on_arrival()

    # -- what a gesture is keyed by, and which gestures an item sees ------- #
    # A gesture is a statement about tissue at a location, so it is keyed by
    # the SLIDE (coupon: the slice file, which is its own slide) and every
    # item covering that location -- an ROI, the overview, the same rect at
    # another level -- queries it by rect (docs/design_multi_model_tasks.md
    # §8). The catalogue says what the binding is (`binding_of`); a catalogue
    # without the method binds by item, as before.
    def _binding_of(self, key):
        fn = getattr(self.catalogue, "binding_of", None)
        if key is None or fn is None:
            return key, None
        return fn(key)

    def _rebase_key(self, key):
        fn = getattr(self.catalogue, "rebase", None)
        return None if fn is None else fn(key)

    def _slice_key(self, si, li):
        """The key gestures drawn on item (si, li) are stored under: its
        slide (coupon: the folder-qualified "folder/basename", which is the
        item key too)."""
        return self._binding_of(self.catalogue.key_of(si, li))[0]

    def _current_key(self):
        cur = self._current()
        return self.catalogue.key_of(*cur) if cur is not None else None

    def _gestures_for_key(self, key):
        """The region gestures item `key` sees, in creation order."""
        slide, rect = self._binding_of(key)
        return self.store.for_item(slide, rect) if slide is not None else []

    def _seam_gestures_for_key(self, key):
        slide, rect = self._binding_of(key)
        return self.store.for_item_seams(slide, rect) if slide is not None else []

    def _gestures_for(self, si, li):
        return self._gestures_for_key(self.catalogue.key_of(si, li))

    def _seam_gestures_for(self, si, li):
        return self._seam_gestures_for_key(self.catalogue.key_of(si, li))

    def _gesture_on_item(self, it, si, li):
        """Whether a gesture belongs to item (si, li): same slide, and its
        extent meets the item's place. This replaces the old
        ``(it.si, it.li) == current`` test everywhere -- the hints now name
        the slide's first row, which is the same for every item of a slide."""
        slide, rect = self._binding_of(self.catalogue.key_of(si, li))
        return slide is not None and it.slice_key == slide and gesture_meets(it, rect)

    def _rebind_store(self, store):
        """Bind a store's gestures against the session: file keys through
        the sequences, item keys of an older store rebased to their slide
        (`rebase`), anything else through the catalogue (`resolve`). The one
        place the three callbacks are wired. Returns the unbound count."""
        return store.rebind(self.subsequences, resolve=self.catalogue.index_of,
                            rebase=self._rebase_key)

    def _annotation_count(self, si, li):
        return len(self._gestures_for(si, li))

    def _sequence_annotation_count(self, si, counts):
        """Distinct gestures over the sequence's items: a gesture inside an
        ROI is seen by the ROI and the overview, and counts once."""
        seen = set()
        for li in range(len(self._sequence_item_labels(si))):
            seen.update(it.uid for it in self._gestures_for(si, li))
        return len(seen)

    # Levels coarser than the item's at which a gesture counts as drawn
    # "much coarser": its scale of intent is >= 4x the item's pixel.
    COARSE_LEVELS = 2

    def _coarse_gestures(self, si, li):
        """The item's gestures drawn at least ``COARSE_LEVELS`` levels coarser
        than it works at -- a stroke meant at gland scale applied to a
        decomposition where the lumen is its own regions. Empty for an app
        without levels (no `_draw_meta`)."""
        draw = self._draw_meta(si, li)
        if not draw or draw.get("level") is None:
            return []
        here = int(draw["level"])
        out = []
        for it in self._gestures_for(si, li) + self._seam_gestures_for(si, li):
            lvl = (it.meta or {}).get("level")
            if isinstance(lvl, (int, float)) and not isinstance(lvl, bool) \
                    and int(lvl) - here >= self.COARSE_LEVELS:
                out.append(it)
        return out

    def _annotation_mark(self, si, li, count):
        mark = super()._annotation_mark(si, li, count)
        if count and self._coarse_gestures(si, li):
            mark += "!"
        return mark

    def _coarse_notice(self, si, li):
        """Once per item and task: say that some of what it shows was drawn
        much coarser (a warning, never a refusal -- whether *gland* at level
        4 means *gland including lumen* at level 0 is the task's call)."""
        key = self.catalogue.key_of(si, li)
        if key is None or key in self._task.caches.coarse_noticed:
            return
        coarse = self._coarse_gestures(si, li)
        if not coarse:
            return
        self._task.caches.coarse_noticed.add(key)
        levels = sorted({int((it.meta or {}).get("level")) for it in coarse})
        self._notify(f"{len(coarse)} annotation(s) here were drawn at level "
                     f"{'/'.join(str(l) for l in levels)}, much coarser than this item "
                     "- they mark the swath the user saw, not this level's regions.")

    # ------------------------------------------------------------------ #
    # Interactions
    # ------------------------------------------------------------------ #
    def _hints_for(self, slice_key, si, li):
        """The (si, li) hints a new gesture stores: where its key binds
        (mspath: the slide's first row), so a gesture drawn on an ROI
        carries the same hints before and after a save / load."""
        found = self.catalogue.index_of(slice_key)
        return (int(found[0]), int(found[1])) if found is not None else (si, li)

    def _with_draw_meta(self, meta, si, li):
        """`meta` plus the item's scale of intent (`_draw_meta`: level, scale,
        px) for the keys it does not set itself; None when there is nothing
        -- a labeler without levels keeps writing meta-less gestures."""
        out = dict(meta or {})
        draw = self._draw_meta(si, li) or {}
        for k, v in draw.items():
            out.setdefault(k, v)
        return out or None

    def _commit_interaction(self, tool, points, meta=None):
        cur = self._current()
        if cur is None:
            return
        si, li = cur
        slice_key = self._slice_key(si, li)
        cls = int(self.active_class_var.get())
        if slice_key is None or not (1 <= cls < self.store.n_classes):
            return
        self._push_history()
        hsi, hli = self._hints_for(slice_key, si, li)
        it = self.store.add(tool, points, cls, slice_key, hsi, hli,
                            meta=self._with_draw_meta(meta, si, li))
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
        hsi, hli = self._hints_for(slice_key, si, li)
        it = self.store.add_seam(tool, points, int(class_id), slice_key, hsi, hli,
                                 meta=self._with_draw_meta(meta, si, li))
        self._rebuild_class_panels()
        self._refresh_render()
        name = SEAM_CLASSES[it.class_id] if it.class_id < len(SEAM_CLASSES) else str(it.class_id)
        n = (meta or {}).get("seams")
        self.status_var.set(f"#{it.uid} {tool} -> {name}"
                            + (f" ({n} seams)" if n is not None else "") + f" ({slice_key})")
        return it

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
        from .extents import default_ext, outline_of_ids
        rec = self.regions.record(self.catalogue.key_of(si, li))
        table = rec.get("stats") if rec is not None else None
        place = self._region_placement()
        added = []
        for ids, cls, meta in parts:
            ids = [int(i) for i in ids]
            if not ids or not (1 <= int(cls) < self.store.n_classes):
                continue
            pts = _extremum_points(labels, ids, table, np, self.FIELDS, place)
            if not pts:
                continue
            if not added:
                self._push_history()
            hsi, hli = self._hints_for(slice_key, si, li)
            # The set IS the object (an extent): its outline, in image
            # coordinates, is what crosses a level; the seeds re-resolve here.
            meta = dict(meta or {})
            try:
                meta["outline"] = outline_of_ids(labels, ids, np, place, ext=default_ext())
            except Exception as exc:               # never lose the fill over its outline
                self._log(f"outline not recorded: {type(exc).__name__}: {exc}")
            added.append(self.store.add("taps", pts, int(cls), slice_key, hsi, hli,
                                        meta=self._with_draw_meta(meta, si, li)))
        if not added:
            return
        self._rebuild_class_panels()
        self._refresh_render()
        what = " + ".join(f"#{it.uid} {(it.meta or {}).get('part', 'fill')} "
                          f"-> class {it.class_id} ({len(it.points)})"
                          for it in added)
        tool = (added[-1].meta or {}).get("tool", "magic")
        t = (added[-1].meta or {}).get("threshold")
        self.status_var.set(f"{tool}: {what}" + (f" at t={t:.3g}" if t is not None else ""))

    def _commit_outline(self, si, li, ids, cls, extra=None):
        """A region task's closed outline: the regions it encloses become ONE
        extent of class `cls`, stored the way a fill is (taps at the seeding
        extrema + the outline, derive.py reads the extent) -- one gesture,
        one undo step, no seam gesture."""
        rec = self.regions.record(self.catalogue.key_of(si, li))
        if rec is None or rec.get("labels") is None:
            self.status_var.set("outline not stored: the regions are gone")
            return
        ids = [int(i) for i in ids]
        meta = {"tool": "outline", "extent": True, "n_regions": len(ids)}
        meta.update(extra or {})
        self._commit_blob(si, li, rec["labels"], [(ids, int(cls), meta)])

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
        # The box arrives in image coordinates; the raster may be a placed
        # item (an ROI at an origin, an overview at 1/16), so both the
        # indexing and the recorded points go through the placement.
        place = self._region_placement()
        (x0, y0), (x1, y1) = place.to_raster(*pts[0]), place.to_raster(*pts[-1])
        xa, xb = sorted((int(round(x0)), int(round(x1))))
        ya, yb = sorted((int(round(y0)), int(round(y1))))
        xa, xb = max(xa, 0), min(xb, w - 1)
        ya, yb = max(ya, 0), min(yb, h - 1)
        if xa > xb or ya > yb:
            return
        sub = labels[ya:yb + 1, xa:xb + 1]
        by_class = {}
        ids_by_class = {}
        for r in np.unique(sub):
            r = int(r)
            if r < 0 or r >= len(region_class):
                continue
            k = int(region_class[r])
            if not (1 <= k < self.store.n_classes):
                continue
            yy, xx = np.argwhere(sub == r)[0]     # a pixel of r inside the box
            px, py = place.to_image(float(xa + xx), float(ya + yy))
            by_class.setdefault(k, []).append((float(px), float(py)))
            ids_by_class.setdefault(k, []).append(r)
        if not by_class:
            self.status_var.set("No predicted classes under the box.")
            return
        slice_key = self._slice_key(si, li)
        self._push_history()
        n = 0
        hsi, hli = self._hints_for(slice_key, si, li)
        from .extents import default_ext, outline_of_ids
        for k in sorted(by_class):
            # The accepted set is an extent: its outline crosses a level.
            meta = {"tool": "accept"}
            try:
                meta["outline"] = outline_of_ids(labels, ids_by_class[k], np, place,
                                                 ext=default_ext())
            except Exception as exc:
                self._log(f"outline not recorded: {type(exc).__name__}: {exc}")
            self.store.add("taps", by_class[k], k, slice_key, hsi, hli,
                           meta=self._with_draw_meta(meta, si, li))
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
        if li is None:
            return [it for it in self.store.interactions + self.store.seams
                    if self._row_owns_key(si, li, it.slice_key)]
        return self._gestures_for(si, li) + self._seam_gestures_for(si, li)

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
            # An ITEM row whose gestures survive it (an ROI: they are the
            # slide's, visible on the overview and back on a re-cut). A
            # sequence row keeps the old wording -- a coupon slice held by a
            # second sequence is that sequence's business.
            on = {it.uid for si, li in rows if li is not None
                  for it in self._row_interactions(si, li)}
            if on:
                return msg + (f"\n\nThe {len(on)} annotation(s) on it are kept: they "
                              "belong to the slide, not to this row.")
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
        self._rebind_store(self.store)
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
        unbound = self._rebind_store(store)
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
        region, polyline = self._for_kind("region"), self._for_kind("polyline")
        self.root.bind("m", region(self._on_magic_key))
        self.root.bind("M", region(self._on_magic_key))
        self.root.bind("b", region(self._on_blob_key))
        self.root.bind("B", region(self._on_blob_key))
        self.root.bind("<Control-z>", self._on_undo_key)
        self.root.bind("<Control-y>", self._on_undo_key)
        self.root.bind("<Tab>", self._on_tab_toggle)
        # 'R': one keystroke = train + immediate reclassify. 'C': classify.
        # R / C train / classify the ACTIVE task's model, whichever kind.
        self.root.bind("r", self._on_train_hotkey)
        self.root.bind("R", self._on_train_hotkey)
        self.root.bind("c", self._on_classify_hotkey)
        self.root.bind("C", self._on_classify_hotkey)
        # 'O': optimize the dense network (search + install + classify).
        self.root.bind("o", region(self._on_optimize_key))
        self.root.bind("O", region(self._on_optimize_key))
        # 'N': neighbours on/off -- flip between an edge kind and its base.
        self.root.bind("n", region(self._on_edge_key))
        self.root.bind("N", region(self._on_edge_key))
        # Polyline tools: T trace, S scope, E seam layer on/off; Enter commits
        # the livewire in flight (a trace, or a region task's outline) and
        # BackSpace drops its last leg.
        self.root.bind("t", polyline(self._on_trace_key))
        self.root.bind("T", polyline(self._on_trace_key))
        self.root.bind("s", polyline(self._on_scope_key))
        self.root.bind("S", polyline(self._on_scope_key))
        self.root.bind("e", polyline(self._on_seams_toggle_key))
        self.root.bind("E", polyline(self._on_seams_toggle_key))
        self.root.bind("<Return>", self._on_trace_commit_key)
        self.root.bind("<BackSpace>", self._on_trace_back_key)
        # F9: the canvas takes the whole center, and back. A function key
        # needs no _typing() guard -- and every letter is already taken.
        self.root.bind("<F9>", self._toggle_center_tabs)

    # ------------------------------------------------------------------ #
    # Task kinds: what the tabs, tools and keys offer
    # ------------------------------------------------------------------ #
    _KIND_GLYPH = {"region": "\u25a6", "polyline": "\u2307"}

    def _task_kind(self):
        return getattr(getattr(self, "_task", None), "kind", "region")

    def _allowed_tools(self):
        return _POLYLINE_TOOLS if self._task_kind() == "polyline" else _REGION_TOOLS

    def _coerce_tool(self, *_a):
        """tool_var's guard: a tool the active kind does not offer falls back
        to the kind's first (squiggle / trace)."""
        if self._coercing_tool:
            return
        allowed = self._allowed_tools()
        if self.tool_var.get() not in allowed:
            self._coercing_tool = True
            try:
                self.tool_var.set(allowed[0])
            finally:
                self._coercing_tool = False

    def _for_kind(self, kind):
        """Wrap a hotkey handler so it acts only in a task of `kind`."""
        def wrap(handler):
            def run(e=None):
                if self._task_kind() != kind:
                    return None
                return handler(e)
            return run
        return wrap

    def _apply_task_kind(self):
        """Show the active task kind's controls: the Annotation tab's region
        frames (tools + classes, ML Region Classifier) or the seam frame, the
        Model and Analysis bodies, and a tool the kind offers. A trace or
        outline in flight is abandoned first."""
        if not hasattr(self, "annot_frame"):
            return
        poly = self._task_kind() == "polyline"
        tool = self.viewer.tool if self.viewer is not None else None
        trace = getattr(tool, "trace", None)
        if trace is not None and trace.active:
            trace.cancel("abandoned: the task changed")
        for w in (self.annot_frame, self.region_ml_frame, self.seam_frame):
            w.pack_forget()
        if poly:
            self.seam_frame.pack(side="top", fill="both", expand=True, padx=4, pady=4)
        else:
            self.region_ml_frame.pack(side="bottom", fill="x", padx=4, pady=(2, 4))
            self.annot_frame.pack(side="top", fill="both", expand=True, padx=4, pady=(4, 2))
        for region_w, poly_w in ((self._model_region, self._model_polyline),
                                 (self._analysis_region, self._analysis_polyline)):
            region_w.pack_forget()
            poly_w.pack_forget()
            (poly_w if poly else region_w).pack(fill="both", expand=True)
        self._coerce_tool()

    def _on_train_hotkey(self, e=None):
        if self._task_kind() == "polyline":
            if self._typing():
                return None
            self._train_seam_model()
            return None
        return self._train_and_classify(e)

    def _on_classify_hotkey(self, e=None):
        if self._task_kind() == "polyline":
            if self._typing():
                return None
            self._classify_seams()
            return None
        return self._on_classify_key(e)

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
                                                _DEFAULT_DRAG_PX, *_DRAG_PX_RANGE),
                      "extent": bool(self.magic_extent_var.get())}
        d["center_tab"] = self._center_tab_name()
        # The sashes as FRACTIONS, not pixels: a session restored into a
        # differently sized window should divide it the same way. The folded
        # Processing groups ride along -- the panel is one column, so which
        # groups are out of the way is part of how it is set up.
        d["panes"] = self._pane_fractions()
        d["proc_open"] = self._proc_open_state()
        # The Model-tab settings (kind, search, neighbours, context) are the
        # TASK's view, not the window's: see _task_view_from_ui.
        return d

    # -- the task's view: what the Model tab describes -------------------- #
    def _task_view_from_ui(self):
        """The Model-tab settings that belong to the active task
        (``TASK_VIEW_KEYS``): the PICKED kind, apart from the trained model
        (a plain Train writes no pickle, so a restore that reloads the newest
        saved model would otherwise land on that pickle's kind, e.g. the last
        Optimize winner); the Optimize settings; the custom base / edge
        settings; and the PICKED context (what the next Train builds), apart
        from the trained model's, which rides its pickle."""
        trials, timeout_s, seed, feat, backend = self._search_settings()
        return {"model_kind": self.model_kind_var.get(),
                "model_search": {"trials": trials, "timeout_s": timeout_s, "seed": seed,
                                 "feature_search": feat, "backend": backend,
                                 "sweep_sizes": self.sweep_sizes_var.get(),
                                 "sweep_trials": self._sweep_settings_trials()},
                "neighbours": {"custom_hidden": self.custom_hidden_var.get(),
                               "freeze_base": bool(self.freeze_base_var.get()),
                               "edge_spec": self._edge_spec_from_ui().to_dict()},
                "context": self._context_spec_from_ui().to_dict()}

    def _stash_task_view(self, task):
        """Record the Model tab's current settings on `task` (the one they
        describe) -- before a switch, a save, or a New session."""
        task.view = self._task_view_from_ui()

    def _apply_task_view(self, view):
        """Push a task's view onto the Model tab. A key that is absent leaves
        the control as it is, so a task created without settings inherits
        the ones on screen (and records them at its first stash). The kind
        and the context win over a reloaded pickle's, so this runs AFTER a
        model reload."""
        view = view if isinstance(view, dict) else {}
        self._apply_search_view(view.get("model_search"))
        self._apply_neighbours_view(view.get("neighbours"))
        self._apply_context_view(view.get("context"))
        if view.get("model_kind") in _MODEL_KINDS:
            self.model_kind_var.set(view["model_kind"])

    def _apply_context_view(self, d):
        if isinstance(d, dict):
            self._apply_context_spec(context.ContextSpec.from_dict(d))

    def _rebuild_class_panels(self):
        # The single "labels changed" signal: with labels in the model's
        # context the cached predictions are stale (ClassifierMixin).
        super()._rebuild_class_panels()
        self._labels_changed()
        self._refresh_seam_panel()
        self._update_task_rows()          # the gesture count column
        self._refresh_stages()            # an edit can make the model stale

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
        if isinstance(magic.get("extent"), bool):
            self.magic_extent_var.set(magic["extent"])

    def _session_doc_kwargs(self):
        """The tasks (the document becomes v3): every task's gestures, saved
        model records and Model-tab view, the active one's view read from
        the controls first. Rides the 4 s autosave."""
        self._stash_task_view(self._task)
        return {"tasks": [t.to_doc() for t in self.tasks],
                "active_task": self._task.uid}

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
        """The tasks survive a New session -- names, workflows, vocabularies
        (class count, colours, names) and, with the model option, their
        saved-model records and Model-tab views -- with every store emptied.
        The old session was auto-saved first, so nothing is lost."""
        doc = super()._new_session_doc(keep)
        keep_model = keep.get("model", True)
        self._stash_task_view(self._task)
        tasks = []
        for t in self.tasks:
            td = t.to_doc()
            empty = td["annotations"]
            empty["interactions"] = []        # classes + colours + names, no gestures
            empty.pop("seams", None)          # ...and no seam gestures: a v2 doc
            empty["version"] = 2
            if not keep_model:
                td["models"] = []
                td["view"] = {}
            if self.ENROLMENT:
                td["enrolled"] = {}           # no slides left to work
            tasks.append(td)
        doc.pop("annotations", None)
        doc.pop("models", None)
        doc["tasks"] = tasks
        doc["active_task"] = self._task.uid
        doc["session_version"] = session_doc.SESSION_DOC_VERSION_TASKS
        # The in-memory models are not in the document (a plain Train writes
        # no pickle): stash every task's stack, by uid, so the apply's pickle
        # reload cannot replace one with an older saved state.
        self._new_session_stash = ({t.uid: t.model for t in self.tasks}
                                   if keep_model else None)
        return doc

    def _after_new_session(self, keep):
        stash = getattr(self, "_new_session_stash", None) or {}
        self._new_session_stash = None
        if keep.get("model", True):
            for t in self.tasks:
                if t.uid in stash:
                    t.model = stash[t.uid]
                    t.model_pending = False
        else:
            for t in self.tasks:
                t.model = ModelStack()
            self._reset_model_selection()
        for t in self.tasks:
            t.caches.clear()
        self._set_classify_enabled()
        self._refresh_model_panels()
        self._update_task_rows()

    def _refresh_model_panels(self):
        """Repaint everything that shows the active task's model and
        predictions: after a New session, a task switch, a model reset."""
        self._refresh_region_modes()
        self._refresh_confusion()
        self._refresh_model_readout()
        self._refresh_model_strip()
        self._refresh_edge_readout()
        self._fill_error_list(None)
        self._refresh_seam_panel()

    def _set_classify_enabled(self):
        btn = getattr(self, "classify_btn", None)
        if btn is not None:
            btn.config(state="normal" if self._clf is not None else "disabled")

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
        notes = notes if notes is not None else []
        # A labeler document whose tasks do not say their kind is refused
        # whole, before anything of it is applied: nothing to migrate.
        why = session_doc.labeler_refusal(doc)
        if why is not None:
            notes.append(why)
            self._log(f"{source}: {why}")
            self._notify(why)
            return notes
        # (An app's override may return None rather than the list it was
        # given; the notes it appended are in ours either way.)
        returned = super()._apply_session_doc(doc, source, notes)
        if returned is not None:
            notes = returned
        # The base already parsed (and noted) folders / sequences / profiles;
        # only the task notes of this second parse are new.
        rnotes = []
        sdoc = self._session_doc_from_json(doc, rnotes)
        notes.extend(n for n in rnotes if n.startswith("task "))
        # The tasks install AFTER sequences exist, so rebind sees them (and
        # migrates legacy bare-basename keys against the qualified identity).
        self.tasks = [Task.from_doc(td, notes) for td in sdoc["tasks"]]
        for t in self.tasks:
            self._normalize_enrolment(t, notes)
        for t in self.tasks:
            unbound = self._rebind_store(t.store)
            if unbound:
                notes.append(f"task {t.name!r}: {unbound} annotation(s) reference "
                             f"{self.ITEM_NOUN}s not in the session (kept, greyed)")
        active = next((t for t in self.tasks if t.uid == sdoc["active_task"]), self.tasks[0])
        # Activation switches to the task's workflow, reloads its newest saved
        # pickle (a failure -- moved file, incompatible profile -- is a note,
        # never fatal) and pushes its Model-tab view, which wins over the
        # pickle's kind and context.
        self._activate_task(active, initial=True)
        if active.load_note:
            notes.append(active.load_note)
        view = sdoc.get("view") or {}
        if view.get("tool") in _UI_TOOLS:
            self.tool_var.set(view["tool"])
        self._apply_magic_view(view.get("magic"))
        self._apply_seams_view(view.get("seams"))
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
    # Tasks: several named detectors, one active
    # ------------------------------------------------------------------ #
    def _profile_index(self, name):
        for i, p in enumerate(self.profiles):
            if p["name"] == name:
                return i
        return None

    def _bind_workflow(self, name):
        """Make the profile called `name` the active task's workflow (and
        the active profile). The bindings' `_profile_from_model` hooks use
        this after appending a profile, instead of poking the index."""
        idx = self._profile_index(name)
        if idx is not None:
            self._switch_profile(idx)

    def _switch_profile(self, idx):
        """The active profile IS the active task's workflow: whichever way it
        changes -- the Combobox, New / Duplicate / Load / Delete profile, a
        profile built from a model -- the task follows."""
        super()._switch_profile(idx)
        if 0 <= self.active_profile_idx < len(self.profiles):
            self._task.workflow = self.profiles[self.active_profile_idx]["name"]
        self._update_task_rows()

    def _profile_rename(self):
        old = self.profiles[self.active_profile_idx]["name"]
        super()._profile_rename()
        new = self.profiles[self.active_profile_idx]["name"]
        if new != old:
            for t in self.tasks:
                if t.workflow == old:
                    t.workflow = new
            self._update_task_rows()

    def _profile_delete(self):
        old = self.profiles[self.active_profile_idx]["name"]
        super()._profile_delete()
        names = [p["name"] for p in self.profiles]
        if old not in names:
            # The active task was re-stamped by the switch inside; every
            # other task on the deleted workflow moves to the survivor too.
            survivor = names[self.active_profile_idx]
            for t in self.tasks:
                if t.workflow == old:
                    t.workflow = survivor
            self._update_task_rows()

    def _activate_task(self, task, initial=False):
        """Make `task` the active one: its workflow becomes the active
        profile, its store / model / caches are what every control reads,
        its Model-tab view is on screen. Returns False when refused.

        Refused while an Optimize / sweep / evaluate worker runs: its finish
        installs into the ACTIVE task, so a switch mid-search would land the
        winner in the wrong task. On the initial activation of a session
        load there is no previous task to refuse in favour of."""
        if task is self._task and not initial:
            return True
        if self._search is not None and not initial:
            self._notify("A search is running - Cancel it before switching tasks.")
            self._update_task_rows()
            return False
        if not initial:
            self._stash_task_view(self._task)
            # Stamp the outgoing task's cache signature too: whatever it
            # enrolled or saw while active is what its caches are valid for.
            self._task.caches.keys_sig = self._keys_signature()
        # Caches keyed on the store's rev with no task discriminator: two
        # stores can share a rev with different content, so they clear (each
        # rebuilds in one rasterization pass per visible item).
        self._class_luts.clear()
        self._clear_seam_caches()
        self._ctx_cache.clear()
        self._hover_key = None
        self._hover_uid = None
        self._task = task
        idx = self._profile_index(task.workflow)
        if idx is None:
            if task.workflow is not None:
                self._log(f"task {task.name!r}: workflow {task.workflow!r} is not in "
                          f"the session - using the active profile")
            idx = max(0, self.active_profile_idx)
        # A no-op (primes kept) when it is already the active profile; a
        # different workflow drops them, as a profile switch always has.
        self._switch_profile(idx)
        if 0 <= self.active_profile_idx < len(self.profiles):
            task.workflow = self.profiles[self.active_profile_idx]["name"]
        # The saved pickle loads on activation, not at session restore: the
        # load runs the compatibility gate against the ACTIVE workflow.
        if task.model_pending and task.model.empty:
            self._reload_task_model(task)
        self._apply_task_view(task.view)
        # The items the task works may differ from the previous task's (an
        # app with enrolment): the navigation, the catalogue and the tree
        # follow before anything reads them. Never a prime.
        self._enrolment_changed()
        # Rows may have come or gone while the task was inactive: rebind its
        # gestures, and drop predictions whose item is gone or whose row
        # moved (a prediction is keyed by item, but the confusion cell and
        # the error rows hold (si, li)).
        self._rebind_store(task.store)
        sig = self._keys_signature()
        old = task.caches.keys_sig
        if old is not None and old != sig:
            live = set(sig[0])
            for cache in (task.caches.pred, task.caches.seam_pred):
                for k in [k for k in cache if k not in live]:
                    cache.pop(k, None)
            task.caches.cm_cell = None
        task.caches.keys_sig = sig
        self.n_classes_var.set(task.store.n_classes)
        if self.active_class_var.get() >= task.store.n_classes:
            self.active_class_var.set(0)
        self._set_classify_enabled()
        self._apply_task_kind()
        self._rebuild_class_panels()
        self._refresh_model_panels()
        self._update_task_rows()
        try:
            self._refresh_render()
        except Exception as exc:
            self._log(f"redraw after task switch failed: {exc}")
        return True

    def _keys_signature(self):
        """What a task's per-item caches are valid for: the catalogue's item
        keys and the row addresses they sit at."""
        return (tuple(self.catalogue.keys()), tuple(tuple(p) for p in self.flat_slices))

    def _enrolment_changed(self):
        """The active task's items changed (a switch, an enrol / unenrol).
        An app without enrolment has nothing to redo; mspath rebuilds its
        navigation, catalogue and tree without priming anything."""

    def _normalize_enrolment(self, task, notes=None):
        """After a session's tasks install: bring a task's enrolment to the
        app's rules (mspath: a task written before enrolment works its
        places, never the overview). Nothing to do without enrolment."""

    # ------------------------------------------------------------------ #
    # Statistics edits re-measure; they never re-prime
    # ------------------------------------------------------------------ #
    def _on_stat_spec_change(self):
        """The Features tab's statistics changed. The viewer's answer is a
        badge and a Rerun; the labeler's is to settle (a sigma typed digit by
        digit is ONE edit) and re-measure the item on screen -- the MSC, its
        labels and arcs are kept, only the rows are rebuilt -- so the next
        Train or Classify reads the new columns without a Run."""
        super()._on_stat_spec_change()
        after = getattr(self, "_stat_edit_after", None)
        if after is not None:
            try:
                self.root.after_cancel(after)
            except tk.TclError:
                pass
        self._stat_edit_after = None
        try:
            self._stat_edit_after = self.root.after(_PREVIEW_SETTLE_MS,
                                                    self._stat_edit_settled)
        except (tk.TclError, AttributeError):
            pass

    def _stat_edit_settled(self):
        self._stat_edit_after = None
        try:
            self._remeasure_current()
        except Exception as exc:
            self._log(f"re-measure after a statistics edit failed: {exc}")

    def _preview_edit_settled(self):
        """Every settled profile edit (the chain cards report here): the
        viewer repaints a preview; the labeler also asks what the edit costs.
        A field edit (the topology chain, the MSC) stays a preview with its
        "Run to re-prime" badge; a measurement edit (the base chain, the
        statistics) re-measures the item on screen now."""
        super()._preview_edit_settled()
        try:
            if self._measurement_moved():
                self._remeasure_current()
        except Exception as exc:
            self._log(f"re-measure after an edit failed: {exc}")

    def _reload_task_model(self, task):
        """Load the newest of the task's recorded pickles that still exists
        into its stack (it must be the active task: the loader writes
        through the properties). A failure is a note on the task."""
        assert task is self._task
        task.model_pending = False
        task.load_note = None
        for entry in reversed(task.models):
            if os.path.isfile(entry.get("path", "")):
                try:
                    self._load_classifier_from(entry["path"])
                except Exception as exc:
                    task.load_note = f"task {task.name!r}: model not reloaded: {exc}"
                    self._log(task.load_note)
                break

    # -- the Tasks list ------------------------------------------------- #
    def _build_task_section(self):
        """The task list, above the session lists: one row per task (name,
        workflow, gesture count, model), the active one selected; New /
        Dup / Rename… / Delete underneath. Built inside the base __init__,
        after the profiles exist, so the first task binds its workflow here."""
        c = ttk.LabelFrame(self._left_section_parent("tasks"), text="Tasks")
        c.pack(fill="x", padx=6, pady=4, before=self.session_frame)
        self.task_frame = c
        tree = ttk.Treeview(c, columns=("kind", "workflow", "annot", "model"),
                            show="tree headings", height=3, selectmode="browse")
        tree.heading("#0", text="task")
        tree.column("#0", width=110, minwidth=60, stretch=True)
        tree.heading("kind", text="")
        tree.column("kind", width=24, minwidth=20, stretch=False, anchor="center")
        tree.heading("workflow", text="workflow")
        tree.column("workflow", width=90, minwidth=40, stretch=True)
        tree.heading("annot", text="annot")
        tree.column("annot", width=46, minwidth=30, stretch=False, anchor="e")
        tree.heading("model", text="model")
        tree.column("model", width=96, minwidth=40, stretch=True)
        tree.pack(fill="x", padx=4, pady=(4, 2))
        tree.bind("<<TreeviewSelect>>", self._on_task_select)
        tree.bind("<Double-1>", lambda _e: self._task_rename())
        attach_tooltip(tree, "The detectors this session trains, each with its own "
                             "classes, annotations, workflow and model.\n"
                             "Click: switch (a different workflow drops the primed "
                             "data, as a profile switch does). Double-click: rename.")
        self.task_tree = tree
        row = ttk.Frame(c)
        row.pack(fill="x", padx=4, pady=(0, 4))
        self.task_btn_row = row
        new = ttk.Menubutton(row, text="New", width=5)
        menu = tk.Menu(new, tearoff=False)
        menu.add_command(label="Region task (classify regions)",
                         command=lambda: self._task_new(kind="region"))
        menu.add_command(label="Polyline task (classify boundaries)",
                         command=lambda: self._task_new(kind="polyline"))
        new.config(menu=menu)
        new.pack(side="left")
        attach_tooltip(new, "A region task labels regions (squiggle, lasso, magic, "
                            "outline ...); a polyline task labels the boundaries "
                            "between them (trace, scope). The kind is fixed.")
        b = ttk.Button(row, text="Dup", width=5, command=self._task_duplicate)
        b.pack(side="left", padx=2)
        attach_tooltip(b, "A new task with this one's classes, colours, names, "
                          "workflow and Model-tab settings -- no gestures, no model")
        ttk.Button(row, text="Rename…", command=self._task_rename).pack(side="left", padx=2)
        ttk.Button(row, text="Delete", command=self._task_delete).pack(side="left", padx=2)
        if self._task.workflow is None and 0 <= self.active_profile_idx < len(self.profiles):
            self._task.workflow = self.profiles[self.active_profile_idx]["name"]
        self._paint_task_rows(tree)

    def _task_row_values(self, task):
        n = task.gesture_count
        if task.model.clf is not None or task.model.seam is not None:
            model = task.model.kind if task.model.clf is not None else "seam model"
        elif task.model_pending and task.models:
            model = f"{task.models[-1].get('kind', '?')} (saved)"
        else:
            model = ""
        return (self._KIND_GLYPH.get(task.kind, "?"), task.workflow or "-",
                str(n) if n else "", model)

    def _paint_task_rows(self, tree):
        """Repaint in place (the sequence tree's pattern): rows are keyed by
        uid, so a rename or a count change never rebuilds the list, and the
        selection -- the active task -- is set under a guard so the
        Treeview's own select event does not re-activate it."""
        want = [t.uid for t in self.tasks]
        self._task_rows_syncing = True
        try:
            if list(tree.get_children("")) != want:
                tree.delete(*tree.get_children(""))
                for t in self.tasks:
                    tree.insert("", "end", iid=t.uid, text=t.name,
                                values=self._task_row_values(t))
            else:
                for t in self.tasks:
                    tree.item(t.uid, text=t.name, values=self._task_row_values(t))
            if tuple(tree.selection()) != (self._task.uid,):
                tree.selection_set(self._task.uid)
            tree.see(self._task.uid)
        except tk.TclError:
            pass
        finally:
            self._task_rows_syncing = False

    def _update_task_rows(self):
        """Repaint the Tasks list from `self.tasks` (a no-op before the list
        exists)."""
        tree = getattr(self, "task_tree", None)
        if tree is None:
            return
        self._paint_task_rows(tree)

    def _on_task_select(self, _event=None):
        if self._task_rows_syncing:
            return
        sel = self.task_tree.selection()
        if not sel:
            return
        task = next((t for t in self.tasks if t.uid == sel[0]), None)
        if task is not None:
            self._activate_task(task)

    def _task_by_name(self, name):
        return next((t for t in self.tasks if t.name == name), None)

    def _task_new(self, name=None, kind="region"):
        """A task of `kind` ("region" | "polyline", fixed from here on) on
        the active workflow with the current class count and default
        colours, activated. `name` given -> no dialog (headless)."""
        taken = [t.name for t in self.tasks]
        name = dedupe_task_name(str(name or ("walls" if kind == "polyline" else "task")), taken)
        workflow = None
        if 0 <= self.active_profile_idx < len(self.profiles):
            workflow = self.profiles[self.active_profile_idx]["name"]
        task = Task.new(name, workflow=workflow, n_classes=self.store.n_classes,
                        taken=[t.uid for t in self.tasks], kind=kind)
        if self.ENROLMENT:
            task.enrolled = {}             # a new task works nothing yet
        self.tasks.append(task)
        if not self._activate_task(task):
            self.tasks.remove(task)
            return None
        self.status_var.set(f"Task '{task.name}' created on workflow '{task.workflow}'.")
        return task

    def _task_duplicate(self, name=None):
        src = self._task
        taken = [t.name for t in self.tasks]
        name = dedupe_task_name(str(name or src.name), taken)
        self._stash_task_view(src)
        task = src.duplicate(name, taken=[t.uid for t in self.tasks])
        self.tasks.append(task)
        if not self._activate_task(task):
            self.tasks.remove(task)
            return None
        self.status_var.set(f"Task '{task.name}' duplicated from '{src.name}' "
                            "(classes and settings; no gestures, no model).")
        return task

    def _task_rename(self, name=None):
        current = self._task.name
        if name is None:
            from tkinter import simpledialog
            name = simpledialog.askstring(self.APP_TITLE, "Task name:",
                                          initialvalue=current, parent=self.root)
        name = str(name or "").strip()
        if not name or name == current:
            return False
        name = dedupe_task_name(name, [t.name for t in self.tasks if t is not self._task])
        self._task.name = name
        self._update_task_rows()
        return True

    def _task_delete(self, confirm=None):
        """Delete the active task. Refused for the last one. Asks when the
        task has gestures or a model unless `confirm` says (headless)."""
        task = self._task
        if len(self.tasks) <= 1:
            self.status_var.set("A session keeps at least one task.")
            return False
        loaded = not task.model.empty
        if confirm is None and (task.gesture_count or loaded):
            what = " and ".join(p for p, on in ((f"{task.gesture_count} annotation(s)",
                                                 task.gesture_count),
                                                ("its model", loaded)) if on)
            confirm = messagebox.askyesno(
                self.APP_TITLE, f"Delete task '{task.name}' with {what}?\n\n"
                                "Its saved pickles stay on disk.")
        if confirm is False:
            return False
        idx = self.tasks.index(task)
        neighbour = self.tasks[idx - 1] if idx > 0 else self.tasks[1]
        if not self._activate_task(neighbour):
            return False
        self.tasks.remove(task)
        self._update_task_rows()
        self.status_var.set(f"Task '{task.name}' deleted.")
        return True

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
