"""The generic Tk viewer shell every MSSeg labeler is built on.

``ViewerShell`` owns what does not depend on the data or the compute: the
window (toolbar, a horizontal paned window with a left column, a centre pane
and the layout hooks a subclass re-lays it with), the session browser (data
folders, their image files, the sequences made from them), named profiles
(opaque dicts the app edits through hooks), item navigation over an
``ItemCatalogue``, the work-queue pump over a ``RegionProvider``, and the
session document with its save/load/restore/auto-save/New-session flow.

Everything domain-specific is a hook with a documented default: what a
profile is (`_default_profile`, `_profile_from_ui`, `_apply_profile_to_ui`,
`_profile_to_file_doc`, `_profile_from_file_doc`), what the compute state is
(`_init_compute`, `_make_catalogue`, `_make_region_provider`,
`_reset_compute`, `_enumerate_items`, `_handle_compute_event`), what the app
draws (`_refresh_render`, `_build_processing_sections`, `_build_run_section`,
`_build_segmentation_controls`, `_build_live_panel`) and what rides the
session beside the generic view state (`_view_state`, `_apply_view_state`,
`_run_settings`, `_apply_run_settings`). The coupon viewer
(`msseg.mscoupon.app.MscouponApp`) is the reference implementation.
"""
from __future__ import annotations

import json
import os
import re
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from . import session_doc
from .widgets import ScrollFrame, jump_scale, scrolled_listbox, attach_tooltip

IMAGE_EXTENSIONS = (".tif", ".tiff")


def natural_key(path):
    """Natural sort key (so asdf_2 < asdf_10)."""
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_files(folder, extensions=IMAGE_EXTENSIONS):
    """Naturally-sorted files with one of `extensions` in `folder`."""
    try:
        entries = [os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(tuple(extensions))]
    except OSError:
        return []
    return sorted(entries, key=natural_key)


class ViewerShell:
    # Per-app identity: the session file (`SESSION_IO.session_path(app=...)`),
    # dialog titles, the window title and the log prefix. Subclasses override.
    SESSION_APP = "labeler"
    APP_TITLE = "labeler"
    WINDOW_TITLE = "labeler"
    LOG_PREFIX = "labeler"
    # The module the session file goes through (session_path, read_json_file,
    # serialize_session, write_session_text, rotate_session_backups). An app
    # that re-exports these through its own module names that instead, so a
    # monkeypatch on its module keeps working.
    SESSION_IO = session_doc
    # True while a priming run is in flight (gates Load/Restore/New session);
    # an app with an engine mirrors the engine's flag as a property.
    _run_active = False

    def _log(self, msg):
        print(f"[{self.LOG_PREFIX}] {msg}", flush=True)

    def _default_profile(self, name="default"):
        """Factory hook: a new profile. The shell knows only its name."""
        return {"name": name}

    def _make_catalogue(self):
        """Factory hook: the ItemCatalogue over this app's data."""
        raise NotImplementedError("a ViewerShell subclass must provide _make_catalogue")

    def _make_region_provider(self):
        """Factory hook: the RegionProvider over this app's compute."""
        raise NotImplementedError("a ViewerShell subclass must provide _make_region_provider")

    def __init__(self, root, initial=None, autosave=True):
        self.root = root
        self.root.title(self.WINDOW_TITLE)

        # --- data model -------------------------------------------------- #
        # A SESSION is the top-level object: data folders, the sequences made
        # from them, named profiles (one active), run settings, and -- in a
        # labeler -- annotations and model references.
        self.folders = []                        # [{"path": str, "name": str}]
        self.active_folder_idx = None            # index into self.folders
        self.all_files = []                      # ACTIVE folder's image paths
        self.subsequences = []                   # [{"name","folder","files":[abs paths]}]
        self.profiles = [self._default_profile()]
        self.active_profile_idx = 0
        # The app's compute state (its engine, parameter cards) is a hook, and
        # the framework's seams sit over it (see protocols.py): items are what
        # can be annotated, regions their current decomposition.
        self._init_compute()
        self.catalogue = self._make_catalogue()
        self.regions = self._make_region_provider()
        self.flat_slices = []                    # [(subseq_idx, local_idx)] linearized
        self._pump_started = False

        # --- tk variables ------------------------------------------------ #
        self._init_variables()                   # the app's own come first
        self.slice_var = tk.IntVar(value=0)
        self.alpha_var = tk.DoubleVar(value=0.5)
        self.vmin_var = tk.DoubleVar(value=0.0)          # base window (fractions
        self.vmax_var = tk.DoubleVar(value=1.0)          #  of the base channel's range)
        self.status_var = tk.StringVar(value="Ready.")
        self.hover_var = tk.StringVar(value="")
        self._hover_ctx = None                           # cached arrays for the hover readout
        self.autosave_var = tk.BooleanVar(value=bool(autosave))

        # --- layout ------------------------------------------------------ #
        # The toolbar packs first: self.paned takes the cavity with expand=True,
        # so anything packed "top" after it would be stacked underneath.
        self._build_toolbar(root)

        self.paned = ttk.PanedWindow(root, orient="horizontal")
        self.paned.pack(fill="both", expand=True)

        # The left panel's sections outgrow the window as soon as a few
        # parameter cards are added, so it scrolls: a canvas carries the real
        # panel and `self.left` IS that inner frame, which leaves _build_left()
        # and every section below it untouched.
        self._build_left_shell()

        self.paned.add(self.left_pane, weight=0)
        # The center pane is a hook: the viewer wants one frame (`self.right`),
        # the labeler a tabbed notebook whose View tab IS `self.right`.
        self._build_center()
        self._build_left()
        self._build_right()
        self._after_layout()

        ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w").pack(
            side="bottom", fill="x")

        if initial and os.path.isdir(initial):
            self._add_folder_path(initial)

        # Auto-save on a timer rather than a trace per widget: a trace has to be
        # remembered every time a control is added, and the one that is forgotten
        # is the one whose value is lost. Disabled for --selftest, which builds a
        # real app and must not touch the user's saved session.
        self._autosave_last = ""         # last session text actually written
        self._autosave_after = None      # pending root.after id
        # Auto-save OWNS the saved session only once this instance has opened
        # it (Restore last / Load session…) or explicitly written one (Save
        # session…). A freshly launched app is empty, and the timer below fires
        # 4 s later: without this gate that empty state is serialized straight
        # over the real session, before the user can even reach "Restore last".
        self._session_owned = False
        if autosave:
            try:
                self.root.protocol("WM_DELETE_WINDOW", self._on_close)
            except tk.TclError:
                pass
            self._schedule_autosave()

    # ------------------------------------------------------------------ #
    # Hooks with defaults (the app's compute, parameters and drawing)
    # ------------------------------------------------------------------ #
    def _init_compute(self):
        """Create the app's compute state (engine, parameter cards) before the
        catalogue and region provider are asked for."""

    def _init_variables(self):
        """Create the app's own Tk variables (before the shell's and the layout)."""

    def _after_layout(self):
        """Runs once the window is built, before the status bar."""

    def _build_processing_sections(self):
        """The app's parameter sections in the left column (between the
        session browser and Run)."""

    def _build_run_section(self):
        c = ttk.LabelFrame(self._left_section_parent("run"), text="Run")
        c.pack(fill="x", padx=6, pady=4)
        self.run_frame = c
        self.run_btn = ttk.Button(c, text="Run", command=self._run)
        self.run_btn.pack(fill="x", padx=4, pady=4)

    def _run(self):
        self.status_var.set("Nothing to run.")

    def _build_segmentation_controls(self, chan):
        """Controls after the Image dropdown (the coupon adds its segmentation
        radios and the mask toggle)."""

    def _build_live_panel(self, parent):
        live = ttk.LabelFrame(parent, text="View")
        live.pack(side="bottom", fill="x", padx=6, pady=4)
        row = ttk.Frame(live); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Slice:").pack(side="left")
        self._build_slice_nav(row)
        row = ttk.Frame(live); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Base Min/Max:", width=14).pack(side="left")
        self._scale(row, from_=0.0, to=1.0, variable=self.vmin_var, orient="horizontal",
                    command=lambda *_: self._refresh_render()).pack(side="left", fill="x", expand=True)
        self._scale(row, from_=0.0, to=1.0, variable=self.vmax_var, orient="horizontal",
                    command=lambda *_: self._refresh_render()).pack(side="left", fill="x", expand=True)
        row = ttk.Frame(live); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Overlay alpha:").pack(side="left")
        self._scale(row, from_=0.0, to=1.0, variable=self.alpha_var, orient="horizontal",
                    command=lambda *_: self._refresh_render()).pack(side="left", fill="x", expand=True)

    def _bind_preview(self):
        """Bind a click/drag over the file list to a preview (the app's)."""

    def _preview_file(self, path):
        """Show `path` without computing anything (the app's)."""

    def _list_files(self, folder):
        return list_files(folder, IMAGE_EXTENSIONS)

    def _slice_msc_mark(self, si, li):
        """The sequence tree's "computed" mark for one slice ("Y" or "")."""
        return ""

    def _slice_nav_text(self, si, li):
        key = self.catalogue.key_of(si, li)
        return key if key is not None else f"[{si}:{li}]"

    def _enumerate_items(self):
        """(si, li) for every navigable item, in order."""
        for key in self.regions.keys():
            idx = self.catalogue.index_of(key)
            if idx is not None:
                yield idx

    def _reset_compute(self):
        """Drop computed state: a profile switch or a session load replaces
        the parameters that produced it."""
        self._hover_ctx = None

    def _settle_controls(self):
        try:
            self.run_btn.config(state="normal")
        except (tk.TclError, AttributeError):
            pass

    def _update_busy(self):
        """Refresh the busy/stale indication (the app's)."""

    def _region_overlay(self, labels, lut, np, visible=True):
        """One overlay dict for a raster of the CURRENT item's region ids.

        Every region layer -- the id colouring, the class layer, the
        predictions, a confusion highlight, a gesture preview -- goes through
        here, because where that raster sits is the app's business, not the
        caller's. For an in-memory item the raster IS the image and the default
        hands it over as it stands. For an item that covers part of a larger
        image (a whole-slide ROI, or a coarse pyramid level of one) the app
        overrides this to wrap the raster in a ``LabelLayer`` that places it,
        so the canvas can compose it in the image's coordinates instead of
        drawing it at the origin at 1:1.
        """
        return {"labels": labels, "lut": lut, "visible": bool(visible)}

    def _refresh_render(self):
        """Repaint the canvas for the current item (the app's)."""

    def _handle_compute_event(self, ev):
        """An engine event the shell does not know ("progress" and "error"
        are handled here)."""

    def _run_settings(self):
        """The session-level run settings block (`run` in the session doc)."""
        return {}

    def _apply_run_settings(self, run, setvar, notes):
        """Push a loaded `run` block onto the widgets."""

    def _apply_view_state(self, view, setvar, notes):
        """Push the app's own view-state keys (beyond background/alpha/window)."""

    def _session_doc_from_json(self, doc, notes):
        return session_doc.session_doc_from_json(doc, notes, profile_reader=self._profile_from_file_doc,
                                                 default_profile=self._default_profile)

    def _import_legacy_docs(self, docs, source):
        """Documents that are not v2 sessions; the default applies the first
        readable one as if it were."""
        good = [(p, d) for p, d in docs if d is not None]
        if good:
            self._apply_session_doc(good[0][1], f"{source} (imported)", [])

    def _profile_to_file_doc(self, profile):
        return dict(profile)

    def _profile_from_file_doc(self, doc, notes):
        return session_doc._passthrough_profile(doc, notes)

    def _profile_from_ui(self):
        """Snapshot the panel into a profile dict (the app's)."""
        return dict(self.profiles[self.active_profile_idx])

    def _apply_profile_to_ui(self, profile, setvar, notes):
        """Push a profile onto the widgets (the app's)."""

    def _build_left(self):
        """The left column: the profile picker, the session browser (folders,
        files, sequences), the app's processing sections, then Run."""
        self._build_profile_section()
        self._build_session_section()
        self._build_processing_sections()
        self._build_run_section()


    def _build_profile_section(self):
        # 0. Compute profile: the named parameter set the panel below edits.
        c = ttk.LabelFrame(self._left_section_parent("profile"), text="0. Compute profile")
        c.pack(fill="x", padx=6, pady=4)
        self.profile_frame = c
        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=2)
        self.profile_var = tk.StringVar(value=self.profiles[0]["name"])
        self.profile_combo = ttk.Combobox(row, textvariable=self.profile_var,
                                          state="readonly", width=22)
        self.profile_combo.pack(side="left", fill="x", expand=True)
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_profile_selected)
        # The management rows go wherever the profile is EDITED (the labeler
        # moves them next to the parameter sections); the picker stays here.
        tools = self._profile_tools_parent(c)
        row = ttk.Frame(tools); row.pack(fill="x", padx=4, pady=2)
        ttk.Button(row, text="New", command=self._profile_new, width=5).pack(side="left")
        ttk.Button(row, text="Dup", command=self._profile_duplicate, width=5).pack(side="left", padx=2)
        ttk.Button(row, text="Rename…", command=self._profile_rename, width=8).pack(side="left", padx=2)
        ttk.Button(row, text="Delete", command=self._profile_delete, width=6).pack(side="left", padx=2)
        row = ttk.Frame(tools); row.pack(fill="x", padx=4, pady=2)
        ttk.Button(row, text="Save profile…",
                   command=self._save_profile).pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.profile_load_btn = ttk.Button(row, text="Load profile…",
                                           command=self._load_profile)
        self.profile_load_btn.pack(side="left", fill="x", expand=True, padx=(2, 0))
        self._refresh_profile_combo()

    def _build_session_section(self):
        # 1. Session: data folders and the sequences made from them.
        c = ttk.LabelFrame(self._left_section_parent("session"), text="1. Session")
        c.pack(fill="x", padx=6, pady=4)
        self.session_frame = c
        # Three groups (folders / files / sequences); each one's buttons are
        # named so a subclass can re-pack them against a resizable list.
        g = self._session_group("folders", c)
        ttk.Label(g, text="Folders:").pack(anchor="w", padx=4)
        self.folder_list = self._scrolled_listbox(g, height=4, exportselection=False)
        self.folder_list.bind("<<ListboxSelect>>", self._on_folder_selected)
        row = ttk.Frame(g); row.pack(fill="x", padx=4, pady=2)
        self.folder_btn_row = row
        ttk.Button(row, text="Add folder…", command=self._add_folder).pack(side="left")
        ttk.Button(row, text="Remove", command=self._remove_folder).pack(side="left", padx=4)
        # Production folders hold thousands of files -> scrollable list.
        # Clicking / dragging over the list previews the image on the right.
        g = self._session_group("files", c)
        self.file_list = self._scrolled_listbox(g, selectmode="extended", height=10,
                                                exportselection=False)
        self._bind_preview()
        self.make_seq_btn = ttk.Button(g, text="Make sequence from selection",
                                       command=self._make_subsequence)
        self.make_seq_btn.pack(fill="x", padx=4, pady=2)
        g = self._session_group("sequences", c)
        ttk.Label(g, text="Sequences:").pack(anchor="w", padx=4)
        holder = ttk.Frame(g); holder.pack(fill="x", padx=4, pady=2)
        self.subseq_list = ttk.Treeview(holder, columns=("msc", "annot"),
                                        height=7, selectmode="extended")
        self.subseq_list.heading("#0", text="sequence / image")
        self.subseq_list.heading("msc", text="msc")
        self.subseq_list.heading("annot", text="annot")
        self.subseq_list.column("#0", width=190, stretch=True)
        self.subseq_list.column("msc", width=38, anchor="center", stretch=False)
        self.subseq_list.column("annot", width=44, anchor="center", stretch=False)
        sb = ttk.Scrollbar(holder, orient="vertical",
                           command=self.subseq_list.yview)
        self.subseq_list.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.subseq_list.pack(side="left", fill="both", expand=True)
        # Clicking a TIFF row under a sequence navigates the viewer to it.
        self.subseq_list.bind("<<TreeviewSelect>>", self._on_seq_tree_select)
        row = ttk.Frame(g); row.pack(fill="x", padx=4, pady=2)
        self.seq_btn_row = row
        ttk.Button(row, text="Remove", command=self._remove_subsequence).pack(side="left")
        ttk.Button(row, text="Clear all", command=self._clear_subsequences).pack(side="left", padx=4)


    def _build_image_controls(self, parent):
        # Background image, drawn fully opaque. A dropdown rather than two radios
        # because every measurement channel is displayable -- looking at the
        # scale-space response a query is thresholding is the point of having it.
        chan = ttk.Frame(parent); chan.pack(fill="x")
        ttk.Label(chan, text="Image:").pack(side="left", padx=(4, 0))
        self.background_var = tk.StringVar(value="base")
        self.background_combo = ttk.Combobox(chan, textvariable=self.background_var,
                                             values=["base", "filtered"], state="readonly",
                                             width=18)
        self.background_combo.pack(side="left", padx=4)
        self.background_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_render())
        self._build_segmentation_controls(chan)

    # ------------------------------------------------------------------ #
    # Toolbar
    # ------------------------------------------------------------------ #
    def _build_toolbar(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(side="top", fill="x")
        self.toolbar = bar
        self.new_session_btn = ttk.Button(bar, text="New session…",
                                          command=self._new_session)
        self.new_session_btn.pack(side="left", padx=(6, 2), pady=3)
        attach_tooltip(self.new_session_btn,
                       "Start over with no folders, sequences or results, keeping "
                       "what you tick in the dialog (the parameters, and in the "
                       "labeler the model selection). The session you leave is "
                       "auto-saved first.")
        self.save_session_btn = ttk.Button(bar, text="Save session…",
                                           command=self._save_session_as)
        self.save_session_btn.pack(side="left", padx=2, pady=3)
        self.load_btn = ttk.Button(bar, text="Load session…",
                                   command=self._load_session)
        self.load_btn.pack(side="left", padx=2)
        self.restore_btn = ttk.Button(bar, text="Restore last",
                                      command=self._restore_last)
        self.restore_btn.pack(side="left", padx=2)
        ttk.Checkbutton(bar, text="auto-save", variable=self.autosave_var,
                        command=self._on_autosave_toggle).pack(side="left", padx=(10, 2))
        self.autosave_label = ttk.Label(bar, text="", foreground="#555")
        self.autosave_label.pack(side="left", padx=4)

    def _set_load_enabled(self, enabled):
        """Gate Load/Restore on the priming worker.

        The engine's "primed" event overwrites `self.primed` with no epoch guard, so a
        load landing mid-run would leave a stack primed under the parameters the
        load just replaced.
        """
        state = "normal" if enabled else "disabled"
        for btn in (getattr(self, "load_btn", None),
                    getattr(self, "restore_btn", None),
                    getattr(self, "profile_load_btn", None)):
            if btn is not None:
                try:
                    btn.config(state=state)
                except tk.TclError:
                    pass

    # ------------------------------------------------------------------ #
    # Layout hooks
    # ------------------------------------------------------------------ #
    # Three template methods decide WHERE the shared sections land, so a
    # subclass can re-lay the window without forking the builders. The
    # defaults reproduce the viewer's tree exactly: one center frame, and
    # every section in the scrolling left panel.
    def _build_left_shell(self):
        """Create the left pane (`self.left_pane`, added to the paned window by
        the caller) and the frame the sections pack into (`self.left`).

        The viewer's six sections outgrow the window as soon as a few filter /
        query cards are added, so its panel scrolls: a canvas carries the real
        panel and `self.left` IS that inner frame."""
        self.left_pane = ScrollFrame(self.paned, width=376, canvas_width=360)
        self.left = self.left_pane.inner

    def _left_section_parent(self, section):
        """Parent for a left-panel section: "profile", "session" or "run"."""
        return self.left

    def _session_group(self, name, section):
        """Container for one group of the Session section -- "folders",
        "files" or "sequences" -- inside the section's LabelFrame. A plain
        frame here; the labeler makes them the panes of a vertical paned
        window so the lists can be resized against each other."""
        f = ttk.Frame(section)
        f.pack(fill="x")
        return f

    def _build_center(self):
        """Create the center pane (`self.right`) and add it to the paned window."""
        self.right = ttk.Frame(self.paned, width=900)
        self.paned.add(self.right, weight=1)

    def _profile_tools_parent(self, section):
        """Parent for the profile New/Dup/Rename/Delete and Save/Load rows;
        `section` is the compute-profile LabelFrame that holds the picker."""
        return section

    def _processing_parent(self, section):
        """Parent for one of the parameter sections: "filters", "base",
        "msc" or "stats"."""
        return self.left

    # ------------------------------------------------------------------ #
    # Right panel
    # ------------------------------------------------------------------ #
    def _scale(self, parent, **kw):
        """A ttk.Scale that jumps to the click/drag position (see jump_scale)."""
        return jump_scale(parent, **kw)

    @staticmethod
    def _scrolled_listbox(parent, **kw):
        return scrolled_listbox(parent, **kw)

    def _build_right(self):
        # Three template methods so a subclass (the labeler) can replace the
        # controls row or the live panel without touching the viewer area.
        self._build_viewer_area(self.right)
        self._build_image_controls(self.render_frame)
        # hover readout (values under the cursor)
        ttk.Label(self.render_frame, textvariable=self.hover_var, anchor="w",
                  font=("TkFixedFont", 8)).pack(fill="x", padx=4)
        self._build_live_panel(self.right)

    def _build_viewer_area(self, parent):
        self.render_frame = ttk.Frame(parent)
        self.render_frame.pack(side="top", fill="both", expand=True)
        self.canvas_holder = ttk.Frame(self.render_frame)
        self.canvas_holder.pack(fill="both", expand=True)
        self.viewer = None
        try:
            from msseg.labeler.canvas import SliceCanvas
            self.viewer = SliceCanvas(self.canvas_holder)
            self.viewer.pack(fill="both", expand=True)
            self.viewer.on_hover = self._on_hover
        except Exception as exc:  # numpy/PIL unavailable -> no live render
            ttk.Label(self.canvas_holder,
                      text=f"(renderer unavailable: {exc})").pack(padx=8, pady=8)

    # ------------------------------------------------------------------ #
    # Session browser: folders and sequences
    # ------------------------------------------------------------------ #
    def _add_folder(self):
        folder = filedialog.askdirectory(title="Add a data folder to the session")
        if folder:
            self._add_folder_path(folder)

    def _add_folder_path(self, path):
        """Add one folder to the session (dialog-free; the selftest and the
        `initial` ctor argument use this directly). Returns its index, or None
        when the path is already in the session."""
        norm = os.path.normpath(path)
        for i, f in enumerate(self.folders):
            if os.path.normpath(f["path"]) == norm:
                self.status_var.set(f"Folder already in session: {f['name']}")
                self._set_active_folder(i)
                return None
        name = session_doc.folder_display_name(path, [f["name"] for f in self.folders])
        self.folders.append({"path": path, "name": name})
        self._refresh_folder_list()
        idx = len(self.folders) - 1
        self._set_active_folder(idx)
        return idx

    def _remove_folder(self):
        sel = list(self.folder_list.curselection())
        if not sel:
            return
        idx = sel[0]
        name = self.folders[idx]["name"]
        referencing = [s for s in self.subsequences if s.get("folder") == name]
        if referencing:
            if not messagebox.askyesno(
                    self.APP_TITLE, f"Folder '{name}' is used by {len(referencing)} "
                    "sequence(s), which will be removed too. Continue?"):
                return
            self.subsequences = [s for s in self.subsequences
                                 if s.get("folder") != name]
            self._refresh_subseq_list()
        del self.folders[idx]
        self._refresh_folder_list()
        if not self.folders:
            self.active_folder_idx = None
            self.all_files = []
            self.file_list.delete(0, "end")
        else:
            self._set_active_folder(min(idx, len(self.folders) - 1))

    def _refresh_folder_list(self):
        self.folder_list.delete(0, "end")
        for f in self.folders:
            self.folder_list.insert("end", f["name"])
        if self.active_folder_idx is not None and self.folders:
            self.folder_list.selection_clear(0, "end")
            self.folder_list.selection_set(self.active_folder_idx)

    def _on_folder_selected(self, _event=None):
        sel = list(self.folder_list.curselection())
        if sel:
            self._set_active_folder(sel[0])

    def _set_active_folder(self, idx):
        """Make folders[idx] the browsed folder: its TIFFs fill file_list
        (positional index there ⇄ self.all_files index, as before)."""
        if not (0 <= idx < len(self.folders)):
            return
        self.active_folder_idx = idx
        folder = self.folders[idx]
        self.all_files = self._list_files(folder["path"])
        self.file_list.delete(0, "end")
        for f in self.all_files:
            self.file_list.insert("end", os.path.basename(f))
        self._refresh_folder_list()
        self.status_var.set(f"{len(self.all_files)} TIFFs in {folder['name']} "
                            f"({folder['path']})")

    def _folder_by_name(self, name):
        for f in self.folders:
            if f["name"] == name:
                return f
        return None

    def _refresh_subseq_list(self):
        """Repaint the sequence tree from self.subsequences, which is the only
        writer: one top-level row per sequence, its TIFFs as children, with
        per-slice "msc" (primed) and "annot" (labeler interaction count)
        columns.

        Tries an in-place update first. This runs on every annotation commit
        (the "annot" column counts interactions) and delete-and-reinsert
        repaints the whole tree, flickers the left pane and drops the
        selection -- while the rows themselves almost never change, only their
        two values do."""
        if self._update_subseq_values():
            return
        tree = self.subseq_list
        open_seqs = {iid for iid in tree.get_children()
                     if tree.item(iid, "open")}
        tree.delete(*tree.get_children())
        for si, s in enumerate(self.subsequences):
            marks = [(self._slice_msc_mark(si, li), self._annotation_count(si, li))
                     for li in range(len(s.get("files") or []))]
            seq_msc = "Y" if marks and all(m[0] == "Y" for m in marks) else ""
            seq_annot = sum(m[1] for m in marks)
            iid = f"q{si}"
            tree.insert("", "end", iid=iid, text=self._sequence_row_text(s),
                        values=(seq_msc, str(seq_annot) if seq_annot else ""),
                        open=iid in open_seqs)
            for li, path in enumerate(s.get("files") or []):
                msc, annot = marks[li]
                tree.insert(iid, "end", iid=f"q{si}:{li}",
                            text=os.path.basename(path),
                            values=(msc, str(annot) if annot else ""))

    def _update_subseq_values(self):
        """Rewrite the tree's "msc"/"annot" values in place, or return False if
        the rows no longer match self.subsequences (a sequence was added,
        removed, renamed or re-filed -- then the caller rebuilds).

        `tree.set` only touches the one cell, so an unchanged value costs
        nothing on screen and the selection, scroll position and expanded rows
        all survive."""
        tree = self.subseq_list
        seq_iids = list(tree.get_children())
        if seq_iids != [f"q{si}" for si in range(len(self.subsequences))]:
            return False
        for si, s in enumerate(self.subsequences):
            iid = f"q{si}"
            files = s.get("files") or []
            if list(tree.get_children(iid)) != [f"q{si}:{li}" for li in range(len(files))]:
                return False
            if tree.item(iid, "text") != self._sequence_row_text(s):
                return False
            marks = [(self._slice_msc_mark(si, li), self._annotation_count(si, li))
                     for li in range(len(files))]
            seq_msc = "Y" if marks and all(m[0] == "Y" for m in marks) else ""
            seq_annot = sum(m[1] for m in marks)
            self._set_row_values(iid, seq_msc, str(seq_annot) if seq_annot else "")
            for li, (msc, annot) in enumerate(marks):
                child = f"q{si}:{li}"
                # A sequence can swap a file without changing its length, so
                # the row's own name is checked, not just the count of rows.
                if tree.item(child, "text") != os.path.basename(files[li]):
                    return False
                self._set_row_values(child, msc, str(annot) if annot else "")
        return True

    def _set_row_values(self, iid, msc, annot):
        tree = self.subseq_list
        if tree.set(iid, "msc") != msc:
            tree.set(iid, "msc", msc)
        if tree.set(iid, "annot") != annot:
            tree.set(iid, "annot", annot)

    def _annotation_count(self, si, li):
        """Interactions on one slice; the labeler overrides this (the viewer
        has no annotations, so its column stays blank)."""
        return 0

    def _on_seq_tree_select(self, _event=None):
        sel = self.subseq_list.selection()
        if not sel or ":" not in sel[0]:
            return                        # a sequence row: selection only
        left, li = sel[0].split(":")
        try:
            si, li = int(left[1:]), int(li)
        except ValueError:
            return
        try:
            idx = self.flat_slices.index((si, li))
        except ValueError:
            # Not primed (yet): preview the slice as a click in the file list
            # would -- a sequence is browsable before any Run.
            try:
                path = self.subsequences[si]["files"][li]
            except (IndexError, KeyError, TypeError):
                return
            self._preview_file(path)
            return
        if idx != int(round(float(self.slice_var.get()))):
            self._goto_slice(idx)

    @staticmethod
    def _sequence_row_text(s):
        return session_doc.sequence_row_text(s)

    def _make_subsequence(self):
        if self.active_folder_idx is None:
            self.status_var.set("Add a folder first.")
            return
        sel = list(self.file_list.curselection())
        if not sel:
            return
        files = [self.all_files[i] for i in sel]
        folder = self.folders[self.active_folder_idx]["name"]
        stem = lambda p: os.path.splitext(os.path.basename(p))[0]
        name = (f"{folder} {stem(files[0])}" if len(files) == 1
                else f"{folder} {stem(files[0])}-{stem(files[-1])}")
        self.subsequences.append({"name": name, "folder": folder, "files": files})
        self._refresh_subseq_list()

    def _remove_subsequence(self):
        # A selected TIFF child counts as its sequence.
        sis = {int(iid.split(":")[0][1:]) for iid in self.subseq_list.selection()}
        for si in sorted(sis, reverse=True):
            if 0 <= si < len(self.subsequences):
                del self.subsequences[si]
        self._refresh_subseq_list()

    def _clear_subsequences(self):
        self.subsequences.clear()
        self._refresh_subseq_list()

    def _snapshot_active_profile(self):
        if 0 <= self.active_profile_idx < len(self.profiles):
            self.profiles[self.active_profile_idx] = self._profile_from_ui()

    def _refresh_profile_combo(self):
        combo = getattr(self, "profile_combo", None)
        if combo is None:
            return
        names = [p["name"] for p in self.profiles]
        try:
            combo.config(values=names)
        except tk.TclError:
            return
        if 0 <= self.active_profile_idx < len(names):
            self.profile_var.set(names[self.active_profile_idx])

    def _switch_profile(self, idx):
        """Activate profiles[idx]. Primed data belongs to the previous
        profile's parameters, so it is dropped -- sequences, folders and (in
        the labeler) labels are untouched."""
        if not (0 <= idx < len(self.profiles)) or idx == self.active_profile_idx:
            self._refresh_profile_combo()
            return
        self._snapshot_active_profile()
        self._reset_compute()
        self.active_profile_idx = idx
        notes = []

        def setvar(var, value):
            try:
                var.set(value)
            except (tk.TclError, ValueError, TypeError):
                notes.append(f"ignored unusable value {value!r}")

        self._apply_profile_to_ui(self.profiles[idx], setvar, notes)
        self._refresh_profile_combo()
        self._rebuild_flat_slices()
        self._settle_controls()
        try:
            self._update_busy()
            self._refresh_render()
        except Exception as exc:
            self._log(f"redraw after profile switch failed: {exc}")
        for msg in notes:
            self._log(msg)
        self.status_var.set(f"Profile '{self.profiles[idx]['name']}' active - "
                            "Run to prime.")

    def _on_profile_selected(self, _event=None):
        name = self.profile_var.get()
        for i, p in enumerate(self.profiles):
            if p["name"] == name:
                self._switch_profile(i)
                return

    def _profile_new(self):
        self._snapshot_active_profile()
        name = session_doc.dedupe_profile_name("profile",
                                           [p["name"] for p in self.profiles])
        self.profiles.append(self._default_profile(name))
        self._switch_profile(len(self.profiles) - 1)

    def _profile_duplicate(self):
        self._snapshot_active_profile()
        src = self.profiles[self.active_profile_idx]
        dup = json.loads(json.dumps(src))
        dup["name"] = session_doc.dedupe_profile_name(src["name"],
                                                  [p["name"] for p in self.profiles])
        self.profiles.append(dup)
        self._switch_profile(len(self.profiles) - 1)

    def _profile_rename(self):
        from tkinter import simpledialog
        current = self.profiles[self.active_profile_idx]["name"]
        name = simpledialog.askstring(self.APP_TITLE, "Profile name:",
                                      initialvalue=current, parent=self.root)
        if not name or name == current:
            return
        name = session_doc.dedupe_profile_name(
            name, [p["name"] for i, p in enumerate(self.profiles)
                   if i != self.active_profile_idx])
        self.profiles[self.active_profile_idx]["name"] = name
        self._refresh_profile_combo()

    def _profile_delete(self):
        if len(self.profiles) <= 1:
            self.status_var.set("A session keeps at least one profile.")
            return
        idx = self.active_profile_idx
        name = self.profiles[idx]["name"]
        if not messagebox.askyesno(self.APP_TITLE, f"Delete profile '{name}'?"):
            return
        del self.profiles[idx]
        self.active_profile_idx = -1          # force the switch to re-apply
        self._switch_profile(min(idx, len(self.profiles) - 1))

    def _save_profile(self):
        self._snapshot_active_profile()
        profile = self.profiles[self.active_profile_idx]
        path = filedialog.asksaveasfilename(
            title="Save compute profile", defaultextension=".json",
            initialfile=f"{profile['name']}.profile.json",
            filetypes=[("JSON", "*.json")])
        if not path:
            return
        blob = self.SESSION_IO.serialize_session(self._profile_to_file_doc(profile))
        if self.SESSION_IO.write_session_text(blob, path):
            self.status_var.set(f"Wrote {path}")
        else:
            self.status_var.set(f"Could not write {path}")

    def _load_profile(self):
        path = filedialog.askopenfilename(title="Load compute profile",
                                          filetypes=[("JSON", "*.json")])
        if not path:
            return
        doc = self.SESSION_IO.read_json_file(path)
        if doc is None:
            self.status_var.set(f"Could not read {path}")
            return
        notes = []
        profile = self._profile_from_file_doc(doc, notes)
        self._snapshot_active_profile()
        profile["name"] = session_doc.dedupe_profile_name(
            profile["name"], [p["name"] for p in self.profiles])
        self.profiles.append(profile)
        self._switch_profile(len(self.profiles) - 1)
        for msg in notes:
            self._log(msg)
        if notes:
            self.status_var.set(f"Loaded profile '{profile['name']}' - "
                                + "; ".join(notes[:2]))

    def _ensure_pump(self):
        """Start the work-queue pump if it isn't already running."""
        if not self._pump_started:
            self._pump_started = True
            self.root.after(80, self._pump)

    def _pump(self):
        for ev in self.regions.poll():
            self._handle_event(ev)
        # Keep pumping while any async work (priming or assembly) is outstanding.
        if self.regions.pending():
            self.root.after(80, self._pump)
        else:
            self._pump_started = False

    def _handle_event(self, ev):
        """UI half of one engine event (the engine bookkeeping already ran in
        regions.poll())."""
        kind = ev[0]
        if kind == "progress":
            done, total = ev[1]
            self.status_var.set(f"Priming slice {done}/{total}…")
        elif kind == "error":
            self.run_btn.config(state="normal")
            self._set_load_enabled(True)
            self.status_var.set(f"Error: {ev[1]}")
            messagebox.showerror(self.APP_TITLE, str(ev[1]))
        else:
            self._handle_compute_event(ev)

    def _build_slice_nav(self, row):
        """Slice navigation: back/forward buttons side by side, then a
        (scrollable) dropdown of the input images, replacing the old slider."""
        ttk.Button(row, text="<", width=3,
                   command=lambda: self._step_slice(-1)).pack(side="left", padx=(4, 1))
        ttk.Button(row, text=">", width=3,
                   command=lambda: self._step_slice(+1)).pack(side="left", padx=(1, 2))
        self.slice_combo = ttk.Combobox(row, state="readonly", width=34)
        self.slice_combo.pack(side="left", fill="x", expand=True, padx=(2, 4))
        self.slice_combo.bind("<<ComboboxSelected>>", self._on_slice_combo)

    def _sync_slice_combo(self):
        combo = getattr(self, "slice_combo", None)
        if combo is None:
            return
        idx = int(round(float(self.slice_var.get())))
        try:
            if 0 <= idx < len(self.flat_slices):
                combo.current(idx)
            else:
                combo.set("")
        except tk.TclError:
            pass

    def _on_slice_combo(self, _event=None):
        idx = self.slice_combo.current()
        if idx >= 0:
            self._goto_slice(idx)

    def _step_slice(self, delta):
        idx = int(round(float(self.slice_var.get()))) + delta
        if 0 <= idx < len(self.flat_slices):
            self._goto_slice(idx)

    def _goto_slice(self, idx):
        """Navigate to flat slice `idx`: THE one entry point for slice moves
        (dropdown, buttons, tree clicks, row clicks)."""
        if not (0 <= idx < len(self.flat_slices)):
            return
        self.slice_var.set(idx)
        self._sync_slice_combo()
        si, li = self.flat_slices[idx]
        # Only what this view needs, for this slice: browsing the stack with
        # the MSC overlay costs one slice's work, not the whole 3D assembly.
        self.regions.request(self.catalogue.key_of(si, li))
        self._refresh_render()

    def _rebuild_flat_slices(self):
        self.flat_slices = list(self._enumerate_items())
        self.catalogue.refresh()
        combo = getattr(self, "slice_combo", None)
        if combo is not None:
            try:
                combo.config(values=[self._slice_nav_text(si, li)
                                     for si, li in self.flat_slices])
            except tk.TclError:
                pass
        self.slice_var.set(0)
        self._sync_slice_combo()

    def _current(self):
        """Return (subseq_idx, local_idx) for the current global slice, or None."""
        idx = int(round(float(self.slice_var.get())))
        if 0 <= idx < len(self.flat_slices):
            return self.flat_slices[idx]
        return None

    # ------------------------------------------------------------------ #
    # Session save/load/restore/auto-save
    #
    # All of this is best-effort. A session may be hand-edited, may name a
    # folder that has since moved, or may predate a schema change, and none of
    # that may raise inside a Tk callback -- there it surfaces as a traceback
    # on stderr and a button that appeared to do nothing. Problems are
    # collected as notes and shown in the status bar (in full via self._log())
    # rather than as dialogs.
    # ------------------------------------------------------------------ #
    _AUTOSAVE_MS = 4000

    def _view_state(self):
        """The transient view half of a session (what neither a profile nor
        the folder/sequence model expresses); subclasses extend the dict.

        Carries no timestamp on purpose -- the auto-save decides whether to
        write by comparing this text against the last text written, and a
        clock would differ on every tick and write forever.
        """
        return {
            "background": self.background_var.get(),
            "alpha": float(self.alpha_var.get()),
            "vmin": float(self.vmin_var.get()),
            "vmax": float(self.vmax_var.get()),
        }

    def _session_doc(self):
        """The whole session as one v2 document (folders, sequences, every
        profile, run settings, view state; subclasses add labels/models)."""
        self._snapshot_active_profile()
        active = "default"
        if 0 <= self.active_profile_idx < len(self.profiles):
            active = self.profiles[self.active_profile_idx]["name"]
        return session_doc.build_session_doc(
            app=self.SESSION_APP,
            folders=self.folders,
            sequences=self.subsequences,
            profiles=self.profiles,
            active_profile=active,
            run=self._run_settings(),
            view=self._view_state())

    def _apply_session_doc(self, doc, source="session", notes=None):
        """THE apply path: push a session document onto the whole app.

        The order mirrors the old _apply_state and is load-bearing: drop the
        compute state FIRST (the parameters that produced it are being
        replaced), then run settings, folders, sequences, profiles (the active
        one lands on the widgets), view state, and finally settle the UI.
        """
        notes = notes if notes is not None else []
        sdoc = self._session_doc_from_json(doc, notes)

        def setvar(var, value):
            try:
                var.set(value)
            except (tk.TclError, ValueError, TypeError):
                notes.append(f"ignored unusable value {value!r}")

        # 1. Compute state first. (The coupon's engine.reset() leaves a running assembly
        # worker running: its result lands as not-accepted via the bumped
        # token, and the pipes are not re-entrant.)
        self._reset_compute()

        # 2. Session-level run settings.
        self._apply_run_settings(sdoc.get("run") or {}, setvar, notes)

        # 3. Folders. A missing folder is kept (with a note): the session
        # still describes it, and a network share may come back.
        self.folders = []
        for f in sdoc.get("folders") or []:
            if not os.path.isdir(f["path"]):
                notes.append(f"folder not found: {f['path']}")
            self.folders.append(dict(f))
        self.active_folder_idx = 0 if self.folders else None
        if self.active_folder_idx is not None:
            self._set_active_folder(0)
        else:
            self.all_files = []
            try:
                self.file_list.delete(0, "end")
            except tk.TclError:
                pass
        self._refresh_folder_list()

        # 4. Sequences, resolved basenames -> absolute paths per folder.
        by_name = {f["name"]: f for f in self.folders}
        self.subsequences = []
        for s in sdoc.get("sequences") or []:
            files = session_doc.resolve_sequence_files(s, by_name, notes)
            files = self._existing_files(files, len(self.subsequences) + 1, notes)
            if not files:
                notes.append(f"sequence {s.get('name')!r}: no files found - skipped")
                continue
            stem = lambda p: os.path.splitext(os.path.basename(p))[0]
            name = s.get("name") or f"{s['folder']} {stem(files[0])}-{stem(files[-1])}"
            self.subsequences.append({"name": name, "folder": s["folder"],
                                      "files": files})
        self._refresh_subseq_list()

        # 5. Profiles; the active one lands on the widgets.
        self.profiles = sdoc["profiles"]
        names = [p["name"] for p in self.profiles]
        self.active_profile_idx = names.index(sdoc["active_profile"])
        self._refresh_profile_combo()
        self._apply_profile_to_ui(self.profiles[self.active_profile_idx],
                                  setvar, notes)

        # 6. View state, all optional.
        view = sdoc.get("view") or {}
        for key, var in (("alpha", self.alpha_var), ("vmin", self.vmin_var),
                         ("vmax", self.vmax_var)):
            if view.get(key) is not None:
                try:
                    setvar(var, float(view[key]))
                except (TypeError, ValueError):
                    notes.append(f"ignored {key}={view[key]!r}")
        if view.get("background"):
            setvar(self.background_var, str(view["background"]))
        self._apply_view_state(view, setvar, notes)

        # 7. Settle the UI. Nothing is primed, so Rerun has nothing to redo --
        # the next step is Run.
        self._rebuild_flat_slices()
        self._settle_controls()
        try:
            self._update_busy()
            self._refresh_render()
        except Exception as exc:            # a stale render must not eat the load
            self._log(f"redraw after load failed: {exc}")

        # This instance now HOLDS a session, so auto-save may continue it.
        # Until a document has been opened (or one explicitly saved), writing
        # would replace a session nobody in this window has seen.
        self._session_owned = True

        for msg in notes:
            self._log(msg)
        summary = f"Loaded {source}: {len(self.subsequences)} sequence(s), " \
                  f"{len(self.profiles)} profile(s)"
        if notes:
            summary += " — " + "; ".join(notes[:2])
            if len(notes) > 2:
                summary += f" (+{len(notes) - 2} more, see the log)"
        self.status_var.set(summary)
        return notes

    def _apply_session_docs(self, docs, source):
        """Route parsed documents to the right apply path: a single v2 session
        applies directly; anything else (a v1 session, a multi-select of
        exported config_N.json) imports through the legacy converter."""
        good = [(p, d) for p, d in docs if d is not None]
        if not good:
            for p, _d in docs:
                self._log(f"could not read {os.path.basename(str(p))}")
            self.status_var.set("Could not read any file; nothing changed.")
            return
        if len(good) == 1 and session_doc.is_session_doc(good[0][1]):
            self._apply_session_doc(good[0][1], source)
        else:
            self._import_legacy_docs(docs, source)

    @staticmethod
    def _existing_files(files, index, notes):
        """Drop files that are no longer on disk.

        When the containing folder itself is gone the per-file check is skipped:
        stat()ing thousands of paths under a dead network share takes minutes,
        and the answer is already known.
        """
        if not files:
            return []
        parent = os.path.dirname(files[0])
        if parent and not os.path.isdir(parent):
            notes.append(f"seq{index}: {parent} is not reachable - files kept as listed")
            return list(files)
        present = [f for f in files if os.path.isfile(f)]
        if len(present) != len(files):
            notes.append(f"seq{index}: {len(files) - len(present)} of {len(files)} "
                         f"files missing")
        return present

    def _load_session(self):
        if self._run_active:
            self.status_var.set("Busy priming — wait for the run to finish.")
            return
        paths = filedialog.askopenfilenames(
            title="Load session (or legacy config.json files)",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        paths = list(paths or [])
        if not paths:
            return
        self._apply_session_docs([(p, self.SESSION_IO.read_json_file(p)) for p in paths],
                                 f"{len(paths)} file(s)")

    def _save_session_as(self):
        path = filedialog.asksaveasfilename(
            title="Save session", defaultextension=".json",
            initialfile="session.json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        blob = self.SESSION_IO.serialize_session(self._session_doc())
        if self.SESSION_IO.write_session_text(blob, path):
            # Writing a session on purpose is the other way to take ownership:
            # auto-save continuing from here can no longer surprise anyone.
            self._session_owned = True
            self.status_var.set(f"Wrote {path}")
        else:
            self.status_var.set(f"Could not write {path}")

    def _restore_last(self):
        if self._run_active:
            self.status_var.set("Busy priming — wait for the run to finish.")
            return
        path = self.SESSION_IO.session_path(app=self.SESSION_APP)
        doc = self.SESSION_IO.read_json_file(path)
        if doc is None:
            self.status_var.set(f"No saved session at {path}")
            return
        self._apply_session_docs([(path, doc)], "last session")

    # -- new session ---------------------------------------------------- #
    def _new_session_options(self):
        """``[(key, label, tooltip)]`` the New session dialog offers, every
        one ticked by default. Subclasses extend (the labeler adds the
        model selection)."""
        return [("profiles",
                 "Keep compute profiles (filter chains, MSC parameters, statistics)",
                 "The named parameter sets carry over; only folders, sequences and "
                 "computed results are dropped. Off: one fresh default profile.")]

    def _new_session_blurb(self):
        return ("Start a new session: no folders, no sequences, nothing computed.")

    def _new_session_dialog(self):
        """Modal: which parts to carry over. ``{key: bool}``, or None."""
        opts = self._new_session_options()
        dlg = tk.Toplevel(self.root)
        dlg.title("New session")
        dlg.transient(self.root)
        dlg.resizable(False, False)
        ttk.Label(dlg, text=self._new_session_blurb(), wraplength=440,
                  justify="left").pack(anchor="w", padx=12, pady=(12, 6))
        flags = {}
        for key, label, tip in opts:
            var = tk.BooleanVar(master=self.root, value=True)
            flags[key] = var
            cb = ttk.Checkbutton(dlg, text=label, variable=var)
            cb.pack(anchor="w", padx=16, pady=2)
            attach_tooltip(cb, tip)
        ttk.Label(dlg, wraplength=440, justify="left", foreground="#555",
                  text="The session you are leaving is auto-saved first and kept as "
                       "last_session.1.json; Restore last brings the NEW session back. "
                       "To keep the old one under a name, save it now.").pack(
            anchor="w", padx=12, pady=(8, 6))
        result = {}

        def create():
            result.update({k: bool(v.get()) for k, v in flags.items()})
            dlg.destroy()

        row = ttk.Frame(dlg); row.pack(fill="x", padx=12, pady=(4, 12))
        ttk.Button(row, text="Save current session…",
                   command=self._save_session_as).pack(side="left")
        ttk.Button(row, text="Cancel", command=dlg.destroy).pack(side="right")
        ttk.Button(row, text="Create", command=create).pack(side="right", padx=(0, 6))
        dlg.bind("<Escape>", lambda e: dlg.destroy())
        dlg.bind("<Return>", lambda e: create())
        try:
            dlg.update_idletasks()
            x = self.root.winfo_rootx() + (self.root.winfo_width() - dlg.winfo_width()) // 2
            y = self.root.winfo_rooty() + (self.root.winfo_height() - dlg.winfo_height()) // 3
            dlg.geometry(f"+{max(x, 0)}+{max(y, 0)}")
            dlg.grab_set()
            dlg.focus_set()
            self.root.wait_window(dlg)
        except tk.TclError:
            return None
        return result or None

    def _new_session(self, keep=None):
        """A session with no data -- no folders, no sequences, nothing
        computed (the labeler also drops annotations) -- carrying over what
        `keep` says: the dialog's answer, or a ``{key: bool}`` from a
        headless caller. Returns True when a new session was made."""
        if self._run_active:
            self.status_var.set("Busy priming — wait for the run to finish.")
            return False
        if keep is None:
            keep = self._new_session_dialog()
            if keep is None:
                return False
        # The session being left goes to disk first (and to the .1 backup
        # through rotate_session_backups), so it can be brought back.
        if self._session_owned and self.autosave_var.get():
            try:
                self._autosave_now()
            except Exception as exc:
                self._log(f"auto-save before the new session skipped: {exc}")
        doc = self._new_session_doc(keep)
        self._apply_session_doc(doc, "new session")
        self._after_new_session(keep)
        kept = [label.split(" (")[0].replace("Keep ", "") for key, label, _t in
                self._new_session_options() if keep.get(key, True)]
        self.status_var.set("New session" + (" - kept " + ", ".join(kept) if kept
                                              else " - nothing kept"))
        return True

    def _new_session_doc(self, keep):
        """The empty session document, with the profiles carried over or
        replaced by one default; the view state (windowing, tool choices)
        always carries over."""
        self._snapshot_active_profile()
        if keep.get("profiles", True) and self.profiles:
            profiles = [json.loads(json.dumps(p)) for p in self.profiles]
            idx = self.active_profile_idx
            active = profiles[idx]["name"] if 0 <= idx < len(profiles) else profiles[0]["name"]
        else:
            profiles = [self._default_profile()]
            active = profiles[0]["name"]
        return session_doc.build_session_doc(
            app=self.SESSION_APP, folders=[], sequences=[], profiles=profiles,
            active_profile=active,
            run=self._run_settings(),
            view=self._view_state())

    def _after_new_session(self, keep):
        """Hook for what the document cannot express (the labeler resets or
        keeps the in-memory model here)."""

    # -- auto-save ------------------------------------------------------ #
    def _schedule_autosave(self):
        if self._autosave_after is None:
            try:
                self._autosave_after = self.root.after(self._AUTOSAVE_MS,
                                                       self._autosave_tick)
            except tk.TclError:                 # window is going away
                pass

    def _autosave_tick(self):
        self._autosave_after = None
        try:
            if self.autosave_var.get():
                self._autosave_now()
        except Exception as exc:      # a failed save must never stop the timer
            self._log(f"auto-save skipped: {exc}")
        self._schedule_autosave()

    def _autosave_now(self):
        """Write the session, but only when it actually differs.

        Safe while a run or an assembly is in flight: this reads only state the
        UI thread owns (subsequences, the card lists, the Tk vars) and never
        primed/_slices/_assembly, which the workers write.
        """
        if not self._session_owned:
            # Say so rather than failing silently: "auto-save" is checked, and
            # a user building a session from scratch has to know it is not
            # being written yet.
            self.autosave_label.config(text="auto-save held — load or save a session")
            return
        blob = self.SESSION_IO.serialize_session(self._session_doc())
        if blob == self._autosave_last:
            return
        path = self.SESSION_IO.session_path(app=self.SESSION_APP)
        self.SESSION_IO.rotate_session_backups(path)
        if self.SESSION_IO.write_session_text(blob, path):
            self._autosave_last = blob
            self.autosave_label.config(text=f"saved {time.strftime('%H:%M:%S')}")
        else:
            # Unwritable app-data directory: say so once and stop, rather than
            # retrying every few seconds for the rest of the session.
            self.autosave_label.config(text="auto-save unavailable")
            self.autosave_var.set(False)

    def _on_autosave_toggle(self):
        if self.autosave_var.get():
            try:
                self._autosave_now()
            except Exception as exc:
                self._log(f"auto-save skipped: {exc}")
        else:
            self.autosave_label.config(text="auto-save off")

    def _on_close(self):
        try:
            if self.autosave_var.get():
                self._autosave_now()
        except Exception as exc:
            self._log(f"auto-save on close skipped: {exc}")
        try:
            if self._autosave_after is not None:
                self.root.after_cancel(self._autosave_after)
        except Exception:
            pass
        # The workers are daemon threads, so there is nothing to join.
        self.root.destroy()
