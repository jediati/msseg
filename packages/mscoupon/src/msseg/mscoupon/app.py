"""mscoupon interactive Tkinter viewer.

Left panel (controls):
    1. Sequences  -- browse a folder of TIFFs; ctrl-select runs into named
       subsequences (each becomes its own 3D stack).
    2. Filter chain -- an extendable stack of filter cards applied in order.
    3. MSC params -- persistence (percent/abs) + ascending/descending 2-manifold.
    4. Run -- discard prior runs; prime each subsequence per slice (cache the
       filter-chained field + the MSC base decomposition / statistics tree).
    5. Export config.json -- serialize the workflow + selection so the C++ CLI
       reproduces the same output (one config per subsequence).

Right panel:
    Top    -- one slice at a time with toggleable overlay channels, brightness/
              contrast and overlay alpha (large_image-backed pyramidal canvas).
    Bottom -- a global (linearized-over-all-subsequences) slice slider, a live
              persistence slider, an extendable feature-query chain, and the
              on-the-fly 3D assembly.

Heavy dependencies (numpy, the compiled msseg.mscoupon engine, large_image, PIL,
matplotlib, msseg.viz) are imported lazily so the pure-Python control logic stays
importable/testable in headless environments (see `--selftest`).

Run:  mscoupon-gui [folder]     |     mscoupon-gui --selftest
"""
from __future__ import annotations

import os
import sys
import re
import json
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from . import config_io
from .config_io import FILTER_SCHEMA, FILTER_OPERATIONS, QUERY_OPS, query_fields
# Shared helpers live in common.py (re-exported here so existing imports of
# `msseg.mscoupon.app` keep working) and the small reusable widgets in
# widgets.py -- both are shared with the labeler app.
from .common import (log, natural_key, list_tiffs, _wheel_delta,
                     _bind_click_to_value, _id_lut, FeatureTable,
                     _parse_sigmas, _format_sigmas, group_contiguous)
from .widgets import ScrollFrame, jump_scale, scrolled_listbox
from .engine import ComputeEngine


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
class MscouponApp:
    # The per-app session-file identity (config_io.session_path(app=...));
    # subclasses (the labeler) override it so their sessions never collide.
    SESSION_APP = "mscoupon"

    def __init__(self, root, initial=None, autosave=True):
        self.root = root
        self.root.title("mscoupon viewer")

        # --- data model -------------------------------------------------- #
        self.folder = ""
        self.all_files = []                      # files in the browsed folder
        self.subsequences = []                   # [{"name": str, "files": [paths]}]
        self.filter_cards = [self._new_filter_card()]   # trailing "none" card appended
        # Base-channel chain (typically a single `normalize` stage). Statistics
        # and pixel thresholds are measured against its output, while
        # `filter_cards` builds the topology field the MSC runs on. Both derive
        # from the raw slice, so an empty base chain behaves exactly as before.
        self.base_cards = [self._new_filter_card()]
        self._normalize_readouts = []            # StringVars, one per normalize card
        self.query_cards = [self._new_query_card()]     # per-slice selection (2D) cards
        self.pixel_cards = [self._new_pixel_card()]     # pixel intensity trim cards
        # The UI-free compute core: primed stacks, per-slice cache, assemblies,
        # worker threads. The engine's state is also reachable through the
        # delegating properties below (self.primed, self._slices, ...) so the
        # rest of the class -- and the selftest -- reads as before.
        self.engine = ComputeEngine(self._assembly_params)
        self.flat_slices = []                    # [(subseq_idx, local_idx)] linearized
        self._pump_started = False

        # --- tk variables ------------------------------------------------ #
        self.persist_pct_var = tk.StringVar(value="10")
        self.manifold_var = tk.StringVar(value="ascending")
        self.accurate_var = tk.BooleanVar(value=False)
        self.ext_radius_var = tk.StringVar(value="0")
        self.min_area_var = tk.StringVar(value="")
        self.connectivity_var = tk.IntVar(value=6)
        self.seg_source_var = tk.StringVar(value="global")  # none|msc|cc|global
        self._selection_dirty = False   # selection params changed since last assembly
        self.cores_per_slice_var = tk.IntVar(value=max(1, (os.cpu_count() or 2) // 2))  # ~physical cores
        self.concurrent_slices_var = tk.IntVar(value=1)   # slices computed at once
        self.slice_var = tk.IntVar(value=0)
        self.persist_live_var = tk.StringVar(value="10")   # live persistence % (numeric entry)
        self.alpha_var = tk.DoubleVar(value=0.5)
        self.vmin_var = tk.DoubleVar(value=0.0)          # base window (fractions
        self.vmax_var = tk.DoubleVar(value=1.0)          #  of the base channel's range)
        self.vmin_filt_var = tk.DoubleVar(value=0.0)     # filtered window (fractions
        self.vmax_filt_var = tk.DoubleVar(value=1.0)     #  of the filtered channel's range)
        # Measurement channels (`statistics.channels[]`): the two rasters the
        # pipeline already builds, plus derived scale-space responses measured on
        # the base channel. One card per kind; the sigma list is the cross-product
        # that makes a multi-scale stack one line of config instead of many.
        self.stat_base_var = tk.BooleanVar(value=True)
        self.stat_filtered_var = tk.BooleanVar(value=False)
        self.stat_kind_vars = {}       # kind -> (BooleanVar, StringVar sigmas)
        self.stat_reduction_vars = {}  # reduction -> BooleanVar
        self.stat_extremum_var = tk.BooleanVar(value=True)
        # Rasters for a derived channel are computed on demand for the slice on
        # screen and cached by (subsequence, slice, channel). Holding the whole
        # stack would cost one float32 raster per channel per primed slice.
        self._chan_cache = {}
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

        # The left panel's six sections outgrow the window as soon as a few
        # filter/query cards are added, so it scrolls: a canvas carries the real
        # panel and `self.left` IS that inner frame, which leaves _build_left()
        # and every section below it untouched.
        self.left_pane = ScrollFrame(self.paned, width=376, canvas_width=360)
        self.left = self.left_pane.inner

        self.right = ttk.Frame(self.paned, width=900)
        self.paned.add(self.left_pane, weight=0)
        self.paned.add(self.right, weight=1)
        self._build_left()
        self._build_right()

        ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w").pack(
            side="bottom", fill="x")

        if initial and os.path.isdir(initial):
            self._set_folder(initial)

        # Auto-save on a timer rather than a trace per widget: a trace has to be
        # remembered every time a control is added, and the one that is forgotten
        # is the one whose value is lost. Disabled for --selftest, which builds a
        # real app and must not touch the user's saved session.
        self._autosave_last = ""         # last session text actually written
        self._autosave_after = None      # pending root.after id
        if autosave:
            try:
                self.root.protocol("WM_DELETE_WINDOW", self._on_close)
            except tk.TclError:
                pass
            self._schedule_autosave()

    # ------------------------------------------------------------------ #
    # Data-model factories
    # ------------------------------------------------------------------ #
    @staticmethod
    def _new_filter_card():
        return {"operation": "none", "params": {}}

    @staticmethod
    def _new_query_card():
        return {"field": "", "op": "gt", "value": 0.0, "value2": 0.0}

    @staticmethod
    def _new_pixel_card():
        return {"channel": "", "mode": "keep", "op": "gt", "value": 0.0}

    # ------------------------------------------------------------------ #
    # Engine state (delegates)
    # ------------------------------------------------------------------ #
    # The compute state lives in ComputeEngine; these properties keep the
    # historical attribute names addressable (the selftest and this class both
    # read and assign them).
    @property
    def primed(self):
        return self.engine.primed

    @primed.setter
    def primed(self, value):
        self.engine.primed = value

    @property
    def _slices(self):
        return self.engine.slices

    @_slices.setter
    def _slices(self, value):
        self.engine.slices = value

    @property
    def _assembly(self):
        return self.engine.assembly

    @_assembly.setter
    def _assembly(self, value):
        self.engine.assembly = value

    @property
    def _commit_id(self):
        return self.engine.commit_id

    @_commit_id.setter
    def _commit_id(self, value):
        self.engine.commit_id = value

    @property
    def _asm_token(self):
        return self.engine.asm_token

    @_asm_token.setter
    def _asm_token(self, value):
        self.engine.asm_token = value

    @property
    def _asm_running(self):
        return self.engine.asm_running

    @_asm_running.setter
    def _asm_running(self, value):
        self.engine.asm_running = value

    @property
    def _asm_running_si(self):
        return self.engine.asm_running_si

    @_asm_running_si.setter
    def _asm_running_si(self, value):
        self.engine.asm_running_si = value

    @property
    def _asm_pending(self):
        return self.engine.asm_pending

    @_asm_pending.setter
    def _asm_pending(self, value):
        self.engine.asm_pending = value

    @property
    def _run_active(self):
        return self.engine.run_active

    @_run_active.setter
    def _run_active(self, value):
        self.engine.run_active = value

    @property
    def _work_q(self):
        return self.engine.work_q

    # ------------------------------------------------------------------ #
    # Toolbar
    # ------------------------------------------------------------------ #
    def _build_toolbar(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(side="top", fill="x")
        self.load_btn = ttk.Button(bar, text="Load config.json…",
                                   command=self._load_config)
        self.load_btn.pack(side="left", padx=(6, 2), pady=3)
        self.restore_btn = ttk.Button(bar, text="Restore last",
                                      command=self._restore_last)
        self.restore_btn.pack(side="left", padx=2)
        ttk.Checkbutton(bar, text="auto-save", variable=self.autosave_var,
                        command=self._on_autosave_toggle).pack(side="left", padx=(10, 2))
        self.autosave_label = ttk.Label(bar, text="", foreground="#555")
        self.autosave_label.pack(side="left", padx=4)

    def _build_statistics_panel(self):
        """Which channels a feature is measured on, and with which reductions.

        A derived channel is a Gaussian-derivative response computed on the base
        raster; naming several sigmas on one row is the cross-product, so a
        scale-space stack is one line rather than one row per (kind, sigma). They
        are measure-only: the topology field is still `filters`, and the seeding
        extremum is still located on it.
        """
        c = ttk.LabelFrame(self.left, text="5. Statistics channels")
        c.pack(fill="x", padx=6, pady=4)

        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=2)
        ttk.Checkbutton(row, text="base", variable=self.stat_base_var,
                        command=self._on_stat_spec_change).pack(side="left")
        ttk.Checkbutton(row, text="filtered", variable=self.stat_filtered_var,
                        command=self._on_stat_spec_change).pack(side="left", padx=8)

        for kind in config_io.DERIVED_CHANNEL_KINDS:
            row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=1)
            on = tk.BooleanVar(value=False)
            sigmas = tk.StringVar(value="0.7, 1.5, 3.0")
            ttk.Checkbutton(row, text=kind, variable=on, width=10,
                            command=self._on_stat_spec_change).pack(side="left")
            ttk.Label(row, text="sigmas:").pack(side="left")
            entry = ttk.Entry(row, textvariable=sigmas, width=16)
            entry.pack(side="left", padx=2)
            # Commit on Enter / focus-out only: every keystroke would re-resolve
            # the schema and rebuild the query dropdowns mid-typing.
            entry.bind("<Return>", lambda e: self._on_stat_spec_change())
            entry.bind("<FocusOut>", lambda e: self._on_stat_spec_change())
            if kind == "hessian":
                ttk.Label(row, text="(largest + smallest)").pack(side="left")
            self.stat_kind_vars[kind] = (on, sigmas)

        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=(4, 2))
        ttk.Label(row, text="reductions:").pack(side="left")
        for reduction in config_io.STAT_REDUCTIONS:
            var = tk.BooleanVar(value=True)
            self.stat_reduction_vars[reduction] = var
            ttk.Checkbutton(row, text=reduction, variable=var,
                            command=self._on_stat_spec_change).pack(side="left", padx=2)
        ttk.Checkbutton(c, text="seeding extremum (ext_* per channel)",
                        variable=self.stat_extremum_var,
                        command=self._on_stat_spec_change).pack(anchor="w", padx=4)
        self.stat_summary_var = tk.StringVar(value="")
        ttk.Label(c, textvariable=self.stat_summary_var, foreground="#555",
                  wraplength=330, justify="left").pack(anchor="w", padx=4, pady=(0, 3))
        self._refresh_stat_summary()

    def _apply_stat_state(self, state, setvar):
        """Restore the `statistics` block into the panel's controls.

        A config may name the same kind more than once (e.g. two `blur` entries
        with different sigmas); the panel has one row per kind, so their sigma
        lists are merged. That is lossless for what the panel can express and is
        the only case where a reload does not reproduce the document verbatim.
        """
        channels = state.get("stat_channels")
        if channels is None:
            return
        setvar(self.stat_base_var, any(c.get("kind") == "base" for c in channels))
        setvar(self.stat_filtered_var, any(c.get("kind") == "filtered" for c in channels))
        by_kind = {}
        for card in channels:
            kind = card.get("kind")
            if kind in ("base", "filtered") or kind not in self.stat_kind_vars:
                continue
            for sigma in card.get("sigmas") or []:
                by_kind.setdefault(kind, [])
                if sigma not in by_kind[kind]:
                    by_kind[kind].append(float(sigma))
        for kind, (on, sigmas) in self.stat_kind_vars.items():
            values = by_kind.get(kind)
            setvar(on, bool(values))
            if values:
                setvar(sigmas, _format_sigmas(sorted(values)))

        reductions = state.get("stat_reductions")
        if reductions is not None:
            for name, var in self.stat_reduction_vars.items():
                setvar(var, name in reductions)
        if state.get("stat_extremum") is not None:
            setvar(self.stat_extremum_var, bool(state["stat_extremum"]))
        self._chan_cache.clear()

    def _refresh_channel_picker(self):
        """Offer exactly the channels the current spec resolves to, keeping the
        selection when it survives the change."""
        combo = getattr(self, "background_combo", None)
        if combo is None:
            return
        names = self._stat_channel_names()
        # `filtered` is always displayable -- it is the topology field, whether or
        # not a workflow also measures aggregates on it.
        if "filtered" not in names:
            names = names + ["filtered"]
        try:
            combo.config(values=names)
        except tk.TclError:
            return
        if self.background_var.get() not in names:
            self.background_var.set(names[0] if names else "base")

    def _channel_raster(self, si, li, name, np):
        """The named measurement channel for one slice, computed on demand.

        Derived channels are not cached with the primed slice: twelve float32
        rasters per slice would dominate a primed subsequence. They are cheap to
        recompute for the one slice on screen, and the result is memoised per
        (subsequence, slice, channel) so panning the persistence entry does not
        redo it."""
        # `primed` is a list indexed by subsequence, and a slice index can outrun
        # a subsequence that was re-primed shorter, so both are bounds-checked:
        # this runs from a render callback, where an IndexError surfaces as a
        # Tkinter traceback rather than a failed call.
        if si is None or si >= len(self.primed):
            return None
        p = self.primed[si]
        if li is None or li >= len(p.get("base") or ()):
            return None
        if name in ("", "base"):
            return p["base"][li]
        if name == "filtered":
            return p["filtered"][li]
        key = (si, li, name)
        hit = self._chan_cache.get(key)
        if hit is not None:
            return hit
        try:
            from msseg import mscoupon as engine
        except Exception:
            return p["base"][li]
        if not hasattr(engine, "stat_channel_images"):
            return p["base"][li]
        try:
            names, imgs = engine.stat_channel_images(
                np.asarray(p["base"][li], dtype=np.float32),
                np.asarray(p["filtered"][li], dtype=np.float32),
                self._params_json())
        except Exception as exc:
            log(f"channel '{name}' unavailable: {exc}")
            return p["base"][li]
        names = list(names)
        if name not in names:
            return p["base"][li]
        # Keep only the requested plane; holding all C would defeat the point.
        raster = np.array(imgs[names.index(name)], copy=True)
        if len(self._chan_cache) > 8:
            self._chan_cache.clear()
        self._chan_cache[key] = raster
        return raster

    def _refresh_stat_summary(self):
        """Show the resolved channel count and field count -- the two numbers that
        decide how wide every per-feature row is."""
        try:
            params = self._params_json()
            n_ch = len(config_io.stat_channels(params))
            n_fields = len(config_io.query_fields(params))
        except Exception:
            n_ch, n_fields = 0, 0
        self.stat_summary_var.set(f"{n_ch} channels -> {n_fields} selectable fields")

    def _on_stat_spec_change(self):
        """The measurement spec changed: the field universe moved with it, so the
        query cards and the channel pickers must be rebuilt, and anything primed
        under the old spec is stale."""
        self._chan_cache.clear()
        self._refresh_stat_summary()
        self._refresh_channel_picker()
        self._rebuild_query_cards()
        self._rebuild_pixel_cards()
        self._mark_selection_dirty()

    def _set_load_enabled(self, enabled):
        """Gate Load/Restore on the priming worker.

        The engine's "primed" event overwrites `self.primed` with no epoch guard, so a
        load landing mid-run would leave a stack primed under the parameters the
        load just replaced.
        """
        state = "normal" if enabled else "disabled"
        for btn in (getattr(self, "load_btn", None), getattr(self, "restore_btn", None)):
            if btn is not None:
                try:
                    btn.config(state=state)
                except tk.TclError:
                    pass

    # ------------------------------------------------------------------ #
    # Left panel
    # ------------------------------------------------------------------ #
    def _build_left(self):
        # 1. Sequences
        c = ttk.LabelFrame(self.left, text="1. Sequences")
        c.pack(fill="x", padx=6, pady=4)
        ttk.Button(c, text="Browse folder…", command=self._browse_folder).pack(fill="x", padx=4, pady=2)
        # Production folders hold thousands of files -> scrollable list.
        self.file_list = self._scrolled_listbox(c, selectmode="extended", height=10,
                                                exportselection=False)
        ttk.Button(c, text="Make subsequence from selection",
                   command=self._make_subsequence).pack(fill="x", padx=4, pady=2)
        ttk.Label(c, text="Subsequences:").pack(anchor="w", padx=4)
        self.subseq_list = self._scrolled_listbox(c, height=4, exportselection=False)
        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=2)
        ttk.Button(row, text="Remove", command=self._remove_subsequence).pack(side="left")
        ttk.Button(row, text="Clear all", command=self._clear_subsequences).pack(side="left", padx=4)

        # 2. Filter chain
        self.filters_frame = ttk.LabelFrame(self.left, text="2. Filter chain (topology field)")
        self.filters_frame.pack(fill="x", padx=6, pady=4)
        self._rebuild_filter_cards()

        # 3. Base channel: 2-point normalization
        self.base_frame = ttk.LabelFrame(self.left, text="3. Base channel (2-point normalization)")
        self.base_frame.pack(fill="x", padx=6, pady=4)
        ttk.Label(self.base_frame, wraplength=330, justify="left",
                  text="Add a 'normalize' stage to put region statistics and pixel "
                       "thresholds on a 0..1 scale between two measured landmarks. "
                       "A threshold of 0.7 then means 0.3*low + 0.7*high on every "
                       "slice, so one value holds across a drifting stack."
                  ).pack(anchor="w", padx=4, pady=(2, 4))
        self._rebuild_filter_cards("base")

        # 4. MSC params
        c = ttk.LabelFrame(self.left, text="4. MSC parameters")
        c.pack(fill="x", padx=6, pady=4)
        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Max persistence %:").pack(side="left")
        ttk.Entry(row, textvariable=self.persist_pct_var, width=8).pack(side="left", padx=4)
        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=2)
        ttk.Radiobutton(row, text="ascending", variable=self.manifold_var,
                        value="ascending").pack(side="left")
        ttk.Radiobutton(row, text="descending", variable=self.manifold_var,
                        value="descending").pack(side="left", padx=6)
        ttk.Checkbutton(c, text="accurate gradient (slower, more memory)",
                        variable=self.accurate_var).pack(anchor="w", padx=4)
        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="ext sample radius:").pack(side="left")
        ttk.Entry(row, textvariable=self.ext_radius_var, width=8).pack(side="left", padx=4)
        ttk.Label(row, text="(0 = the critical pixel)").pack(side="left")
        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Per-slice min area:").pack(side="left")
        ttk.Entry(row, textvariable=self.min_area_var, width=8).pack(side="left", padx=4)

        # 5. Statistics channels
        self._build_statistics_panel()

        # 6. Run
        c = ttk.LabelFrame(self.left, text="6. Run")
        c.pack(fill="x", padx=6, pady=4)
        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Cores/slice:").pack(side="left")
        ttk.Entry(row, textvariable=self.cores_per_slice_var, width=5).pack(side="left", padx=(2, 10))
        ttk.Label(row, text="Concurrent slices:").pack(side="left")
        ttk.Entry(row, textvariable=self.concurrent_slices_var, width=5).pack(side="left", padx=2)
        self.run_btn = ttk.Button(c, text="Run with selected", command=self._run)
        self.run_btn.pack(fill="x", padx=4, pady=4)

        # 5. Export
        c = ttk.LabelFrame(self.left, text="7. Export")
        c.pack(fill="x", padx=6, pady=4)
        ttk.Button(c, text="Export config.json…", command=self._export_config).pack(
            fill="x", padx=4, pady=4)

    def _chain(self, chain):
        """(card list, containing frame) for one of the two filter chains.

        "topo" builds the field the MSC runs on; "base" preprocesses the channel
        that statistics and pixel thresholds are read from. They share all the
        card machinery below -- only the list and the frame differ.
        """
        if chain == "base":
            return self.base_cards, self.base_frame
        return self.filter_cards, self.filters_frame

    def _set_chain_cards(self, chain, cards):
        if chain == "base":
            self.base_cards = cards
        else:
            self.filter_cards = cards

    def _rebuild_filter_cards(self, chain="topo"):
        cards, frame = self._chain(chain)
        for w in list(frame.winfo_children()):
            # The base frame carries an explanatory label above the cards.
            if isinstance(w, ttk.Label):
                continue
            w.destroy()
        if chain == "base":
            self._normalize_readouts = []
        for idx, card in enumerate(cards):
            self._build_filter_card(idx, card, chain)

    def _build_filter_card(self, idx, card, chain="topo"):
        cards, parent = self._chain(chain)
        frame = ttk.Frame(parent, relief="groove", borderwidth=1)
        frame.pack(fill="x", padx=4, pady=2)
        top = ttk.Frame(frame); top.pack(fill="x")
        op_var = tk.StringVar(value=card["operation"])
        combo = ttk.Combobox(top, textvariable=op_var, values=FILTER_OPERATIONS,
                             state="readonly", width=20)
        combo.pack(side="left", padx=2, pady=2)
        combo.bind("<<ComboboxSelected>>",
                   lambda e, i=idx, v=op_var, c=chain: self._on_filter_op_change(i, v.get(), c))
        if idx < len(cards) - 1 or card["operation"] != "none":
            ttk.Button(top, text="✕", width=3,
                       command=lambda i=idx, c=chain: self._remove_filter_card(i, c)
                       ).pack(side="right", padx=2)
        # param widgets for the selected operation
        for pname, kind, default in FILTER_SCHEMA.get(card["operation"], []):
            self._build_param_row(frame, card["params"], pname, kind, default)
        if card["operation"] == "normalize":
            self._build_normalize_readout(frame, card)

    def _build_normalize_readout(self, frame, card):
        """Show the landmarks measured for the slice currently on screen.

        Blank until Run has primed the stack -- the landmarks are per slice, so
        there is nothing meaningful to display before the measure has run.
        """
        row = ttk.Frame(frame); row.pack(fill="x", padx=6, pady=(2, 3))
        var = tk.StringVar(value="landmarks: (run to measure)")
        ttk.Label(row, textvariable=var, foreground="#555").pack(side="left")
        self._normalize_readouts.append(var)
        self._refresh_normalize_readouts()

    def _refresh_normalize_readouts(self):
        """Update every normalize card with the current slice's landmarks."""
        if not getattr(self, "_normalize_readouts", None):
            return
        measured = []
        current = self._current() if self.primed else None
        if current is not None:
            si, li = current
            if si < len(self.primed):
                norms = self.primed[si].get("normalizers") or []
                if li < len(norms):
                    measured = norms[li]
        for i, var in enumerate(self._normalize_readouts):
            if i < len(measured):
                tp = measured[i]
                var.set(f"landmarks: low={tp.low:.6g}  high={tp.high:.6g}")
            else:
                var.set("landmarks: (run to measure)")

    def _build_param_row(self, parent, params, pname, kind, default):
        row = ttk.Frame(parent); row.pack(fill="x", padx=6, pady=1)
        ttk.Label(row, text=pname, width=16).pack(side="left")
        if pname not in params:
            params[pname] = default
        if kind == "bool":
            var = tk.BooleanVar(value=bool(params[pname]))
            var.trace_add("write", lambda *_: params.__setitem__(pname, var.get()))
            ttk.Checkbutton(row, variable=var).pack(side="left")
        elif kind.startswith("choice:"):
            choices = kind.split(":", 1)[1].split(",")
            var = tk.StringVar(value=str(params[pname]))
            var.trace_add("write", lambda *_: params.__setitem__(pname, var.get()))
            ttk.Combobox(row, textvariable=var, values=choices, state="readonly",
                         width=12).pack(side="left")
        elif kind == "str":
            var = tk.StringVar(value=str(params[pname]))
            var.trace_add("write", lambda *_: params.__setitem__(pname, var.get()))
            ttk.Entry(row, textvariable=var, width=14).pack(side="left")
        elif kind in ("optfloat", "nullfloat"):
            # Blank means "not set". For optfloat it is dropped on export, so an
            # unset optional bound cannot be mistaken for a real 0.0; for
            # nullfloat it exports as an explicit null, because "keep every
            # pixel" is a real setting that must not fall back to the default.
            var = tk.StringVar(value=str(params[pname]))
            def commit_opt(*_, p=pname, v=var):
                text = v.get().strip()
                if not text:
                    params[p] = ""
                    return
                try:
                    params[p] = float(text)
                except ValueError:
                    pass
            var.trace_add("write", commit_opt)
            ttk.Entry(row, textvariable=var, width=10).pack(side="left")
        else:  # float | int
            var = tk.StringVar(value=str(params[pname]))
            def commit(*_, p=pname, k=kind, v=var):
                try:
                    params[p] = int(v.get()) if k == "int" else float(v.get())
                except ValueError:
                    pass
            var.trace_add("write", commit)
            ttk.Entry(row, textvariable=var, width=10).pack(side="left")

    def _on_filter_op_change(self, idx, op, chain="topo"):
        cards, _ = self._chain(chain)
        cards[idx]["operation"] = op
        cards[idx]["params"] = {}
        # keep exactly one trailing "none" card so the user can always add more
        cards = [c for c in cards if c["operation"] != "none"]
        cards.append(self._new_filter_card())
        self._set_chain_cards(chain, cards)
        self._rebuild_filter_cards(chain)

    def _remove_filter_card(self, idx, chain="topo"):
        cards, _ = self._chain(chain)
        if 0 <= idx < len(cards):
            del cards[idx]
        if not cards or cards[-1]["operation"] != "none":
            cards.append(self._new_filter_card())
        self._set_chain_cards(chain, cards)
        self._rebuild_filter_cards(chain)

    # ------------------------------------------------------------------ #
    # Sequence browser
    # ------------------------------------------------------------------ #
    def _browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self._set_folder(folder)

    def _set_folder(self, folder):
        self.folder = folder
        self.all_files = list_tiffs(folder)
        self.file_list.delete(0, "end")
        for f in self.all_files:
            self.file_list.insert("end", os.path.basename(f))
        self.status_var.set(f"{len(self.all_files)} TIFFs in {folder}")

    def _refresh_subseq_list(self):
        """Repaint subseq_list from self.subsequences, which is the only writer.

        The piecewise insert/delete the browser used to do has no inverse, and
        loading a config replaces the whole list at once.
        """
        self.subseq_list.delete(0, "end")
        for s in self.subsequences:
            self.subseq_list.insert("end", s["name"])

    def _make_subsequence(self):
        sel = list(self.file_list.curselection())
        if not sel:
            return
        files = [self.all_files[i] for i in sel]
        name = f"seq{len(self.subsequences) + 1} ({len(files)})"
        self.subsequences.append({"name": name, "files": files})
        self._refresh_subseq_list()

    def _remove_subsequence(self):
        sel = list(self.subseq_list.curselection())
        for i in reversed(sel):
            del self.subsequences[i]
        self._refresh_subseq_list()

    def _clear_subsequences(self):
        self.subsequences.clear()
        self._refresh_subseq_list()

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
            from .viewer_canvas import SliceCanvas
            self.viewer = SliceCanvas(self.canvas_holder)
            self.viewer.pack(fill="both", expand=True)
            self.viewer.on_hover = self._on_hover
        except Exception as exc:  # numpy/PIL unavailable -> no live render
            ttk.Label(self.canvas_holder,
                      text=f"(renderer unavailable: {exc})").pack(padx=8, pady=8)

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
        ttk.Label(chan, text="   Segmentation:").pack(side="left")
        # The four rasters are the pipeline's stages in order: every MSC region,
        # those surviving the per-slice selection, those after the pixel trim +
        # in-plane CC, and finally the cross-slice 3D features.
        for src, txt in (("none", "none"), ("msc", "MSC"), ("msc_kept", "MSC filtered"),
                         ("cc", "per-slice CC"), ("global", "global CC")):
            ttk.Radiobutton(chan, text=txt, variable=self.seg_source_var, value=src,
                            command=self._on_seg_source_change).pack(side="left", padx=2)
        self.mask_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(chan, text="mask", variable=self.mask_var,
                        command=self._on_seg_source_change).pack(side="left", padx=(8, 4))

    def _build_live_panel(self, parent):
        live = ttk.LabelFrame(parent, text="Live parameters")
        live.pack(side="bottom", fill="x", padx=6, pady=4)

        # slice slider (global, linearized over all subsequences)
        row = ttk.Frame(live); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Slice:").pack(side="left")
        self.slice_scale = self._scale(row, from_=0, to=0, orient="horizontal",
                                       command=self._on_slice_change)
        self.slice_scale.pack(side="left", fill="x", expand=True, padx=4)
        self.slice_label = ttk.Label(row, text="-")
        self.slice_label.pack(side="left")

        # per-channel windowing: each pair maps 0->1 onto that channel's own min->max
        row = ttk.Frame(live); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Base Min/Max:", width=14).pack(side="left")
        self._scale(row, from_=0.0, to=1.0, variable=self.vmin_var, orient="horizontal",
                    command=lambda *_: self._refresh_render()).pack(side="left", fill="x", expand=True)
        self._scale(row, from_=0.0, to=1.0, variable=self.vmax_var, orient="horizontal",
                    command=lambda *_: self._refresh_render()).pack(side="left", fill="x", expand=True)
        row = ttk.Frame(live); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Filtered Min/Max:", width=14).pack(side="left")
        self._scale(row, from_=0.0, to=1.0, variable=self.vmin_filt_var, orient="horizontal",
                    command=lambda *_: self._refresh_render()).pack(side="left", fill="x", expand=True)
        self._scale(row, from_=0.0, to=1.0, variable=self.vmax_filt_var, orient="horizontal",
                    command=lambda *_: self._refresh_render()).pack(side="left", fill="x", expand=True)
        row = ttk.Frame(live); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Overlay alpha:").pack(side="left")
        self._scale(row, from_=0.0, to=1.0, variable=self.alpha_var, orient="horizontal",
                    command=lambda *_: self._refresh_render()).pack(side="left", fill="x", expand=True)

        # persistence (live simplification): numeric input -- recompute (refilter)
        # is chunky, so commit only on Enter / focus-out, not on every keystroke.
        row = ttk.Frame(live); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Persistence %:").pack(side="left")
        self.persist_entry = ttk.Entry(row, textvariable=self.persist_live_var, width=8)
        self.persist_entry.pack(side="left", padx=4)
        self.persist_entry.bind("<Return>", self._on_persistence_change)
        self.persist_entry.bind("<FocusOut>", self._on_persistence_change)
        self.persist_value_label = ttk.Label(row, text="")
        self.persist_value_label.pack(side="left", padx=4)

        # per-slice selection (queries on 2D merged-region stats)
        self.queries_frame = ttk.LabelFrame(live, text="Per-slice selection")
        self.queries_frame.pack(fill="x", padx=4, pady=4)
        self._rebuild_query_cards()

        # pixel intensity trim (per-pixel keep/omit by base/filtered value)
        self.pixels_frame = ttk.LabelFrame(live, text="Pixel filter (trim)")
        self.pixels_frame.pack(fill="x", padx=4, pady=4)
        self._rebuild_pixel_cards()

        row = ttk.Frame(live); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Connectivity:").pack(side="left")
        for c in (6, 18, 26):
            ttk.Radiobutton(row, text=str(c), variable=self.connectivity_var,
                            value=c, command=self._on_connectivity_change).pack(side="left")

        # Assembly is chunky, so it is NOT live: persistence / selection / pixel /
        # connectivity changes only enable this button; clicking it re-assembles.
        self.rerun_btn = ttk.Button(live, text="Rerun selection", state="disabled",
                                    command=self._rerun_selection)
        self.rerun_btn.pack(fill="x", padx=4, pady=(2, 4))

    def _rebuild_query_cards(self):
        if getattr(self, "queries_frame", None) is None:
            return       # a subclass without the per-slice selection panel
        for w in list(self.queries_frame.winfo_children()):
            w.destroy()
        # Per-card labels showing "surviving before -> after" for that card, plus
        # the hovered feature's value for its field (appended in idx order by
        # _build_query_card below).
        self.query_stat_labels = []
        for idx, card in enumerate(self.query_cards):
            self._build_query_card(idx, card)
        # Populate straight away so a rebuilt chain shows its counts without
        # waiting for a hover or a slider move.
        try:
            self._update_query_stat_labels(None)
        except Exception:
            pass

    def _query_fields(self):
        """Selectable statistic names, from the C++ schema so the dropdown offers
        exactly what an exported config will validate against.

        The live params JSON is passed, so switching a channel or a reduction on
        changes what the dropdowns offer -- previously this asked for the DEFAULT
        spec's fields and could never reflect the statistics block at all."""
        return config_io.query_fields(self._params_json())

    def _query_pickers(self):
        """(channel order, reductions per channel) for the two-level field picker.

        A twelve-channel stack is ~60 fields; one flat combobox of that length is
        unusable, and the pair is how the fields are actually organised. The split
        comes from the C++ schema rather than from parsing names, so `min_x` is
        never read as the `min` reduction of a channel called `x`."""
        return config_io.channels_and_reductions(self._params_json())

    def _build_query_card(self, idx, card):
        frame = ttk.Frame(self.queries_frame); frame.pack(fill="x", padx=2, pady=1)
        order, by_channel = self._query_pickers()
        channel, reduction = config_io.split_field(card["field"], self._params_json())
        if not card["field"]:
            channel, reduction = "", ""

        chan_var = tk.StringVar(value=channel)
        red_var = tk.StringVar(value=reduction)
        chan_combo = ttk.Combobox(frame, textvariable=chan_var, values=[""] + order,
                                  state="readonly", width=13)
        chan_combo.pack(side="left", padx=1)
        red_combo = ttk.Combobox(frame, textvariable=red_var,
                                 values=by_channel.get(channel, []),
                                 state="readonly", width=7)
        red_combo.pack(side="left", padx=1)

        def on_channel(_e=None, i=idx, cv=chan_var, rv=red_var, rc=red_combo):
            choices = self._query_pickers()[1].get(cv.get(), [])
            rc.config(values=choices)
            # Keep the reduction if the new channel still offers it (switching
            # sigma should not reset "max" back to nothing), else take the first.
            if rv.get() not in choices:
                rv.set(choices[0] if choices else "")
            self._on_query_field_change(
                i, config_io.compose_field(cv.get(), rv.get()) if cv.get() and rv.get() else "")

        def on_reduction(_e=None, i=idx, cv=chan_var, rv=red_var):
            self._on_query_field_change(
                i, config_io.compose_field(cv.get(), rv.get()) if cv.get() and rv.get() else "")

        chan_combo.bind("<<ComboboxSelected>>", on_channel)
        red_combo.bind("<<ComboboxSelected>>", on_reduction)
        op_var = tk.StringVar(value=card["op"])
        op_var.trace_add("write", lambda *_, i=idx, v=op_var: self.query_cards[i].__setitem__("op", v.get()))
        ttk.Combobox(frame, textvariable=op_var, values=QUERY_OPS, state="readonly",
                     width=7).pack(side="left", padx=1)
        val_var = tk.StringVar(value=str(card["value"]))
        def commit(*_, i=idx, v=val_var):
            try:
                self.query_cards[i]["value"] = float(v.get())
            except ValueError:
                pass
            self._on_selection_change()
        val_var.trace_add("write", commit)
        ttk.Entry(frame, textvariable=val_var, width=8).pack(side="left", padx=1)
        if card["field"]:
            ttk.Button(frame, text="✕", width=3,
                       command=lambda i=idx: self._remove_query_card(i)).pack(side="left")
        # "surviving before -> after" for this card, plus the hovered region's
        # value for its field (updated in _update_query_stat_labels).
        stat = ttk.Label(frame, text="", width=30, anchor="w", font=("TkFixedFont", 8))
        stat.pack(side="left", padx=4)
        self.query_stat_labels.append(stat)

    def _on_query_field_change(self, idx, field):
        self.query_cards[idx]["field"] = field
        self.query_cards = [c for c in self.query_cards if c["field"]]
        self.query_cards.append(self._new_query_card())
        self._rebuild_query_cards()
        self._on_selection_change()

    def _remove_query_card(self, idx):
        if 0 <= idx < len(self.query_cards):
            del self.query_cards[idx]
        if not self.query_cards or self.query_cards[-1]["field"]:
            self.query_cards.append(self._new_query_card())
        self._rebuild_query_cards()
        self._on_selection_change()

    # -- pixel intensity trim cards ------------------------------------- #
    def _rebuild_pixel_cards(self):
        if getattr(self, "pixels_frame", None) is None:
            return       # a subclass without the pixel-trim panel
        for w in list(self.pixels_frame.winfo_children()):
            w.destroy()
        for idx, card in enumerate(self.pixel_cards):
            self._build_pixel_card(idx, card)

    def _build_pixel_card(self, idx, card):
        from .config_io import PIXEL_CHANNELS, PIXEL_MODES, PIXEL_OPS
        frame = ttk.Frame(self.pixels_frame); frame.pack(fill="x", padx=2, pady=1)
        chan_var = tk.StringVar(value=card["channel"])
        combo = ttk.Combobox(frame, textvariable=chan_var, values=[""] + PIXEL_CHANNELS,
                             state="readonly", width=9)
        combo.pack(side="left", padx=1)
        combo.bind("<<ComboboxSelected>>",
                   lambda e, i=idx, v=chan_var: self._on_pixel_channel_change(i, v.get()))
        mode_var = tk.StringVar(value=card["mode"])
        mode_var.trace_add("write", lambda *_, i=idx, v=mode_var: self.pixel_cards[i].__setitem__("mode", v.get()))
        ttk.Combobox(frame, textvariable=mode_var, values=PIXEL_MODES, state="readonly",
                     width=6).pack(side="left", padx=1)
        op_var = tk.StringVar(value=card["op"])
        op_var.trace_add("write", lambda *_, i=idx, v=op_var: self.pixel_cards[i].__setitem__("op", v.get()))
        ttk.Combobox(frame, textvariable=op_var, values=PIXEL_OPS, state="readonly",
                     width=5).pack(side="left", padx=1)
        val_var = tk.StringVar(value=str(card["value"]))
        def commit(*_, i=idx, v=val_var):
            try:
                self.pixel_cards[i]["value"] = float(v.get())
            except ValueError:
                pass
            self._on_selection_change()
        val_var.trace_add("write", commit)
        ttk.Entry(frame, textvariable=val_var, width=8).pack(side="left", padx=1)
        if card["channel"]:
            ttk.Button(frame, text="✕", width=3,
                       command=lambda i=idx: self._remove_pixel_card(i)).pack(side="left")

    def _on_pixel_channel_change(self, idx, channel):
        self.pixel_cards[idx]["channel"] = channel
        self.pixel_cards = [c for c in self.pixel_cards if c["channel"]]
        self.pixel_cards.append(self._new_pixel_card())
        self._rebuild_pixel_cards()
        self._on_selection_change()

    def _remove_pixel_card(self, idx):
        if 0 <= idx < len(self.pixel_cards):
            del self.pixel_cards[idx]
        if not self.pixel_cards or self.pixel_cards[-1]["channel"]:
            self.pixel_cards.append(self._new_pixel_card())
        self._rebuild_pixel_cards()
        self._on_selection_change()

    # ------------------------------------------------------------------ #
    # Run (prime) + live recompute -- wired to the compiled engine
    # ------------------------------------------------------------------ #
    def _cores_per_slice(self):
        try:
            return max(1, int(self.cores_per_slice_var.get()))
        except (ValueError, tk.TclError):
            return 1

    def _concurrent_slices(self):
        try:
            return max(1, int(self.concurrent_slices_var.get()))
        except (ValueError, tk.TclError):
            return 1

    def _ext_radius(self):
        try:
            return max(0, int(self.ext_radius_var.get()))
        except (ValueError, tk.TclError):
            return 0

    def _params_json(self, cores=None):
        try:
            pct = float(self.persist_pct_var.get())
        except ValueError:
            pct = 10.0
        if cores is None:
            cores = self._cores_per_slice()
        msc = {"manifold": self.manifold_var.get(),
               "persistence_percent": pct,
               "accurate_ascending": self.accurate_var.get(),
               "accurate_descending": self.accurate_var.get()}
        radius = self._ext_radius()
        if radius > 0:
            msc["extremum_sample_radius"] = radius
        if cores > 1:
            # Partitioned builder + N-way parallelism drives MSCEER's discrete
            # gradient, partitioned MSC construction, and manifold labeling
            # (msc2d.cpp compute_with_algorithm). filter/omp threads are governed
            # separately (only in the CLI, via execution.threads_per_slice).
            msc["compute_algorithm"] = "partitioned"
            msc["requested_parallelism"] = cores
        return json.dumps({
            "filters": config_io.filters_to_json(self.filter_cards),
            "base_filters": config_io.filters_to_json(self.base_cards),
            "msc": msc,
            # Drives which per-feature fields exist, so the query dropdown, the
            # primed pipelines and an exported config all agree.
            "statistics": config_io.statistics_to_json(
                self._stat_channel_cards(), self._stat_reductions(),
                self.stat_extremum_var.get(), radius),
        })

    def _stat_channel_cards(self):
        """The `statistics.channels[]` model, in slot order."""
        cards = []
        if self.stat_base_var.get():
            cards.append({"kind": "base"})
        if self.stat_filtered_var.get():
            cards.append({"kind": "filtered"})
        for kind, (on, sigmas) in self.stat_kind_vars.items():
            if not on.get():
                continue
            values = _parse_sigmas(sigmas.get())
            if values:
                cards.append({"kind": kind, "sigmas": values})
        return cards

    def _stat_reductions(self):
        return [r for r, v in self.stat_reduction_vars.items() if v.get()]

    def _stat_channel_names(self):
        """Resolved channel names for the current spec, for the pickers."""
        try:
            return [c["name"] for c in config_io.stat_channels(self._params_json())]
        except Exception:
            return ["base"]

    def _run(self):
        if not self.subsequences:
            messagebox.showinfo("mscoupon", "Define at least one subsequence first.")
            return
        self._set_load_enabled(False)
        self.run_btn.config(state="disabled")
        self.status_var.set("Priming…")
        params = self._params_json()
        subseqs = [dict(s) for s in self.subsequences]
        # The concurrency numbers ride along for the worker's log line, so it
        # never reads Tk state off the UI thread.
        self.engine.start_run(subseqs, params,
                              {"cores_per_slice": self._cores_per_slice(),
                               "concurrent_slices": self._concurrent_slices()})
        self._ensure_pump()

    def _ensure_pump(self):
        """Start the work-queue pump if it isn't already running."""
        if not self._pump_started:
            self._pump_started = True
            self.root.after(80, self._pump)

    def _pump(self):
        for ev in self.engine.poll():
            self._handle_event(ev)
        # Keep pumping while any async work (priming or assembly) is outstanding.
        if self.engine.pending_work():
            self.root.after(80, self._pump)
        else:
            self._pump_started = False

    def _handle_event(self, ev):
        """UI half of one engine event (the engine bookkeeping already ran in
        engine.poll())."""
        kind = ev[0]
        if kind == "progress":
            done, total = ev[1]
            self.status_var.set(f"Priming slice {done}/{total}…")
        elif kind == "error":
            self.run_btn.config(state="normal")
            self._set_load_enabled(True)
            self.status_var.set(f"Error: {ev[1]}")
            messagebox.showerror("mscoupon", str(ev[1]))
        elif kind == "primed":
            self.run_btn.config(state="normal")
            self._set_load_enabled(True)
            self._rebuild_flat_slices()
            self.status_var.set(f"Primed {len(self.primed)} subsequence(s), "
                                f"{len(self.flat_slices)} slices.")
            cur = self._current()
            if cur is not None:
                self._request_assembly(cur[0])   # off-thread; UI stays responsive
            self._refresh_render()
        elif kind == "assembly_done":
            self._on_assembly_done(ev[1], ev[2])

    def _rebuild_flat_slices(self):
        self.flat_slices = []
        for si, p in enumerate(self.primed):
            for li in range(len(p["pipes"])):
                self.flat_slices.append((si, li))
        n = max(0, len(self.flat_slices) - 1)
        self.slice_scale.config(to=n)
        self.slice_var.set(0)

    def _current(self):
        """Return (subseq_idx, local_idx) for the current global slice, or None."""
        idx = int(round(float(self.slice_var.get())))
        if 0 <= idx < len(self.flat_slices):
            return self.flat_slices[idx]
        return None

    def _on_seg_source_change(self):
        """Switching the overlay (or the mask) can demand a tier that has not been
        computed -- selecting `global CC` is what actually triggers the 3D
        assembly. Request it, then repaint with whatever is available now."""
        cur = self._current()
        if cur is not None:
            self._request_assembly(cur[0])
        self._refresh_render()

    def _on_slice_change(self, _value=None):
        # The Scale isn't bound to slice_var directly, so store the slider's value
        # here before reading _current() (which reads slice_var).
        if _value is not None:
            try:
                self.slice_var.set(int(round(float(_value))))
            except (ValueError, tk.TclError):
                pass
        cur = self._current()
        if cur is None:
            return
        si, li = cur
        self.slice_label.config(text=f"{self.subsequences[si]['name']} [{li}]")
        # Only what this view needs, for this slice: browsing the stack with the
        # MSC overlay costs one slice's work, not the whole 3D assembly.
        self._request_assembly(si)
        self._refresh_render()

    def _min_area(self):
        s = self.min_area_var.get().strip()
        try:
            return int(s) if s else None
        except ValueError:
            return None

    def _max_persist_pct(self):
        """The build-time 'Max persistence %' -- the cap the cancellation hierarchy
        was primed to. Live persistence can't exceed it without re-priming."""
        try:
            return float(self.persist_pct_var.get())
        except (ValueError, tk.TclError):
            return 10.0

    def _persist_pct(self):
        # Live persistence, clamped to the primed cap (beyond it MSCEER just
        # saturates at the coarsest complex; raise Max persistence % and re-Run).
        try:
            live = float(self.persist_live_var.get())
        except (ValueError, tk.TclError):
            return min(10.0, self._max_persist_pct())
        return min(live, self._max_persist_pct())

    def _on_persistence_change(self, _event=None):
        # Persistence feeds the (chunky) assembly -> defer to the Rerun button.
        self._mark_selection_dirty()
        self._refresh_render()   # updates the persistence readout; render uses cache

    def _update_persist_label(self):
        """Show the current persistence: the % entry converted to the absolute
        value applied to the current slice (persistence is abs = %*value_range)."""
        cur = self._current()
        if cur is None or not self.primed:
            self.persist_value_label.config(text="")
            return
        si, li = cur
        try:
            vr = float(self.primed[si]["pipes"][li].value_range())
        except Exception:
            self.persist_value_label.config(text="")
            return
        pabs = vr * self._persist_pct() / 100.0
        self.persist_value_label.config(text=f"= {pabs:.4g} abs  (range {vr:.4g})")

    def _on_connectivity_change(self):
        # Connectivity changes both the in-plane CC and the cross-slice stencil.
        self._mark_selection_dirty()

    def _on_selection_change(self):
        # Per-slice selection or pixel trim changed.
        self._mark_selection_dirty()
        # Refresh the pass counters against the cached stats, so the effect of an
        # edit is visible before it is committed with Rerun selection.
        try:
            self._update_query_stat_labels(None)
        except Exception:      # a half-built card chain during a rebuild
            pass

    def _mark_selection_dirty(self):
        """A selection parameter (persistence / per-slice selection / pixel trim /
        connectivity) changed. Assembly is expensive, so don't recompute now --
        just enable the Rerun button. The current view stays until the user reruns."""
        if not self.primed:
            return
        self._selection_dirty = True
        if getattr(self, "rerun_btn", None) is not None:
            self.rerun_btn.config(state="normal")
        self.status_var.set("Selection changed - click 'Rerun selection' to re-assemble.")
        self._update_busy()   # show the 'Out of date' badge on the canvas

    def _rerun_selection(self):
        """Commit the current selection parameters and re-assemble the current
        subsequence (the chunky step, off the UI thread). The old view stays visible
        (with a 'Recomputing' spinner) until the new result lands, and other
        subsequences reassemble lazily when navigated to (commit-id mismatch)."""
        self._selection_dirty = False
        if getattr(self, "rerun_btn", None) is not None:
            self.rerun_btn.config(state="disabled")
        # Bumps the committed parameter generation and prunes the now-stale
        # per-slice records (they are keyed by commit).
        self.engine.commit_selection()
        cur = self._current()
        if cur is not None:
            self._request_assembly(cur[0])
        self._refresh_render()

    # -- assembly (off the UI thread; single-flight over the stateful pipes) --- #
    #
    # Work is tiered by what is actually on screen. The 3D assembly is ~98% of a
    # selection re-run and touches every slice in the stack, but only the "global
    # CC" overlay and the mask need it; browsing MSC regions on one slice needs
    # that one slice. The tiers are ordered, each a superset of the last:
    #
    #   "slice"  visible slice only: persistence, labels, stats, selection
    #   "cc"     + that slice's pixel trim and in-plane connected components
    #   "global" every slice + the cross-slice 3D assembly (the old behaviour)
    _LEVEL_ORDER = {"slice": 0, "cc": 1, "global": 2}

    def _needed_level(self):
        """The cheapest tier that can draw what is currently selected."""
        if self.mask_var.get():
            return "global"          # the mask is painted from global ids
        src = self.seg_source_var.get()
        if src == "global":
            return "global"
        if src == "cc":
            return "cc"
        return "slice"               # none / msc / msc_kept

    def _slice_ready(self, si, li, level):
        """True iff slice `li` is cached at `level` for the current commit."""
        return self.engine.slice_ready(si, li, level)

    def _assembly_params(self, si, level, li):
        """The parameter snapshot for one assembly work item. Called by the
        engine on the UI thread when the item actually LAUNCHES, so the worker
        measures against the spec the user had when they hit Rerun, not
        whatever the panel says by the time it runs. The engine stamps
        commit/level/li itself."""
        return {
            "pct": self._persist_pct(),
            "queries": config_io.queries_to_json(self.query_cards),
            "pixels": config_io.pixel_filters_to_json(self.pixel_cards),
            "connectivity": int(self.connectivity_var.get()),
            "min_area": self._min_area(),
            # Decides which side a 3D feature's seeding extremum comes from.
            "manifold": self.manifold_var.get(),
            "json": self._params_json(),
            "reductions": self._stat_reductions(),
            "extremum": bool(self.stat_extremum_var.get()),
            "name": self.subsequences[si]["name"],
        }

    def _announce_assembly(self, si, level, li):
        what = self.subsequences[si]["name"]
        self.status_var.set(f"Assembling {what}…" if level == "global"
                            else f"Updating {what} [{li}]…")

    def _request_assembly(self, si, level=None):
        """Queue off-thread work for subsequence `si` at the current parameters
        (see ComputeEngine.request_assembly for the single-flight semantics)."""
        if not self.primed or si is None:
            return
        cur = self._current()
        li = cur[1] if cur is not None and cur[0] == si else 0
        if level is None:
            level = self._needed_level()
        launched = self.engine.request_assembly(si, li, level)
        if launched is not None:
            self._announce_assembly(*launched)
        self._ensure_pump()
        self._update_busy()

    def _is_current_busy(self):
        """True iff an assembly for the currently-viewed subsequence is in flight."""
        cur = self._current()
        if cur is None:
            return False
        si = cur[0]
        return ((self._asm_running and self._asm_running_si == si) or
                (self._asm_pending is not None and self._asm_pending[1] == si))

    def _update_busy(self):
        """Drive the canvas HUD: an animated 'Recomputing' spinner while an assembly
        for the current view is in flight, a static 'Out of date' badge when the
        selection changed but hasn't been rerun, else nothing."""
        if self.viewer is None:
            return
        if self._is_current_busy():
            self.viewer.set_hud("busy", "Recomputing")
        elif self._selection_dirty:
            self.viewer.set_hud("stale", "Out of date - click Rerun")
        else:
            self.viewer.set_hud(None)

    def _on_assembly_done(self, si, accepted):
        """UI half of an "assembly_done" event: the result (if accepted) is
        already stored in the engine's caches; refresh the view if it concerns
        the slice on screen, then let the engine start any superseding request."""
        if accepted:
            cur = self._current()
            if cur is not None and cur[0] == si:
                data = self._assembly.get(si)
                if data is not None and data.get("_commit") == self._commit_id:
                    self.status_var.set(f"{self.subsequences[si]['name']}: "
                                        f"{data.get('n_global', 0)} global features")
                else:
                    rec = self._slices.get((si, cur[1])) or {}
                    self.status_var.set(f"{self.subsequences[si]['name']} [{cur[1]}]: "
                                        f"{len(rec.get('kept', ()))} regions kept")
                self._refresh_render()
        launched = self.engine.launch_pending()   # process any newer request
        if launched is not None:
            self._announce_assembly(*launched)
        self._update_busy()               # clear the spinner if nothing is pending

    # -- rendering (reads only cached numpy rasters; never the live pipes) ---- #
    def _seg_overlays(self, si, li, rec, data, np, min_colors):
        """Build the overlay list for one slice (a subclass hook: the labeler
        appends its class layer here).

        Segmentation overlay: recolor the selected source's label raster by id.
        MSC / MSC filtered / per-slice CC come from the per-slice cache, so they
        render without the stack-wide 3D assembly ever running; only global CC
        and the mask read `data`. `rec` and `data` are already commit-checked
        (None when stale)."""
        overlays = []
        src = self.seg_source_var.get()
        raster = None
        if src == "msc" and rec is not None:
            raster = rec["labels"]
        elif src == "msc_kept" and rec is not None:
            # The MSC regions that passed the per-slice selection, before the
            # pixel trim and CC. Masked on the fly rather than cached: a LUT
            # gather over the label raster is cheaper than holding another
            # per-slice int32 raster for the whole stack.
            from . import assembly as asm_mod
            labels = rec["labels"]
            keep = asm_mod.selection_mask(labels, rec["kept"])
            raster = np.where(keep, labels, -1)
        elif src == "cc" and rec is not None and rec.get("cc") is not None:
            raster = rec["cc"]
        elif src == "global" and data is not None:
            raster = data["global_labels"][li]
        if raster is not None:
            overlays.append({"labels": raster, "lut": _id_lut(raster, min_colors, np),
                             "visible": True})
        if self.mask_var.get() and data is not None:
            glob = data["global_labels"][li]                 # -1 bg, >=0 = kept feature
            K = int(glob.max()) + 1 if glob.size else 1
            mlut = np.zeros((max(K, 1), 4), np.uint8)
            mlut[:, 0] = 255; mlut[:, 1] = 255; mlut[:, 3] = 255   # yellow where global>=0
            overlays.append({"labels": glob, "lut": mlut, "visible": True})
        return overlays

    def _refresh_render(self):
        self._update_persist_label()
        self._update_busy()
        # Landmarks and the selection pass counts are both per slice, so both
        # follow the slider.
        self._refresh_normalize_readouts()
        self._update_query_stat_labels(None)
        if self.viewer is None:
            return
        cur = self._current()
        if cur is None or not self.primed:
            return
        try:
            import numpy as np
            from msseg.viz import min_colors
        except Exception:
            return
        si, li = cur
        p = self.primed[si]
        base = np.asarray(p["base"][li], dtype=np.float32)
        filt = np.asarray(p["filtered"][li], dtype=np.float32)
        data = self._assembly.get(si)
        if data is not None and data.get("_commit") != self._commit_id:
            data = None               # stale 3D result: a Rerun superseded it
        self._hover_ctx = {"si": si, "li": li, "base": base, "filt": filt, "data": data}

        rec = self._slices.get((si, li))
        if rec is not None and rec.get("commit") != self._commit_id:
            rec = None                    # stale: a Rerun superseded it
        overlays = self._seg_overlays(si, li, rec, data, np, min_colors)

        first = self.viewer._base is None and self.viewer._source is None
        channel = self.background_var.get()
        if channel == "filtered":
            self.viewer.set_base(array=filt, path=None)
            self.viewer.set_window(self.vmin_filt_var.get(), self.vmax_filt_var.get())
        elif channel in ("", "base"):
            self.viewer.set_base(array=base, path=p["files"][li])
            self.viewer.set_window(self.vmin_var.get(), self.vmax_var.get())
        else:
            # A derived scale-space channel: its range has nothing to do with the
            # base channel's, so the window opens full rather than reusing either
            # of the two hand-kept window pairs.
            raster = self._channel_raster(si, li, channel, np)
            if raster is None:
                # The channel is not available for this slice (nothing primed yet,
                # or an extension that cannot build it). Show the base rather than
                # blanking the canvas.
                self.viewer.set_base(array=base, path=p["files"][li])
                self.viewer.set_window(self.vmin_var.get(), self.vmax_var.get())
            else:
                self.viewer.set_base(array=raster, path=None)
                self.viewer.set_window(0.0, 1.0)
        self.viewer.set_overlays(overlays)
        self.viewer.set_alpha(self.alpha_var.get())
        if first:
            self.viewer.fit()
        else:
            self.viewer.render()

    def _on_hover(self, ix, iy=None):
        """Format the values under the cursor: coords, base/filtered value, and the
        MSC / per-slice-CC / global ids + mask (all from the cached rasters)."""
        ctx = self._hover_ctx
        if ix is None or ctx is None:
            self.hover_var.set("")
            self._update_query_stat_labels(None)
            return
        base, filt, data = ctx["base"], ctx["filt"], ctx["data"]
        h, w = base.shape[:2]
        if not (0 <= ix < w and 0 <= iy < h):
            self.hover_var.set("")
            self._update_query_stat_labels(None)
            return
        li = ctx["li"]
        # MSC id/stats come from the per-slice cache; the CC and global ids only
        # exist once those tiers have run, and read "-" rather than a stale value.
        si = ctx.get("si")
        rec = self._slices.get((si, li))
        if rec is not None and rec.get("commit") != self._commit_id:
            rec = None
        fid = ccid = gid = -1
        stat2d = None
        if rec is not None:
            fid = int(rec["labels"][iy, ix])
            stat2d = rec["stats"].row_of_feature(fid) if rec.get("stats") else None
            if rec.get("cc") is not None:
                ccid = int(rec["cc"][iy, ix])
        if data is not None:
            gid = int(data["global_labels"][li][iy, ix])
        cc_txt = ccid if (rec is not None and rec.get("cc") is not None) else "-"
        gl_txt = gid if data is not None else "-"
        mask_txt = (1 if gid >= 0 else 0) if data is not None else "-"
        self.hover_var.set(
            f"x={ix} y={iy}  |  base={float(base[iy, ix]):.4g} "
            f"filtered={float(filt[iy, ix]):.4g}  |  MSC={fid} CC={cc_txt} "
            f"global={gl_txt}  |  mask={mask_txt}")
        # Per-slice selection cards evaluate on the hovered pixel's 2D MSC region;
        # show that region's per-field value next to each card.
        self._update_query_stat_labels(stat2d)

    def _selection_counts(self):
        """Cumulative surviving-feature count through the selection chain, for the
        slice on screen: [N, after card 0, after cards 0-1, ...].

        Counted on the 2D MSC regions BEFORE the pixel trim and in-plane CC, which
        is the stage the cards actually gate -- so card i turning A into B says
        exactly how much that one predicate costs, given the ones above it.

        Uses the stats cached by the last assembly, so edits show their effect
        before you commit them with Rerun selection."""
        cur = self._current()
        if cur is None:
            return None
        si, li = cur
        rec = self._slices.get((si, li))
        if rec is None or rec.get("commit") != self._commit_id:
            return None
        table = rec["stats"]
        if table is None or not table.n_rows:
            return None

        active = [c for c in self.query_cards if c.get("field")]
        counts = [table.n_rows]
        if not active:
            return counts
        try:
            from msseg import mscoupon as engine
        except ImportError:
            return counts
        # Evaluate progressively longer prefixes of the chain. Cheap next to the
        # assembly itself (a few thousand rows), and it reuses the same evaluator
        # the CLI runs, so the numbers cannot disagree with a batch run.
        for k in range(1, len(active) + 1):
            qjson = json.dumps(config_io.queries_to_json(active[:k]))
            if hasattr(engine, "evaluate_queries_table"):
                flags = engine.evaluate_queries_table(table.names, table.values, qjson)
            else:
                flags = engine.evaluate_queries(table.rows(), qjson)
            counts.append(int(sum(1 for f in flags if f)))
        return counts

    def _update_query_stat_labels(self, stat2d):
        """Label each per-slice-selection card with how many features survive it
        ("before -> after"), plus the hovered region's value for that field."""
        counts = self._selection_counts()
        active_idx = [i for i, c in enumerate(self.query_cards) if c.get("field")]
        for idx, lbl in enumerate(getattr(self, "query_stat_labels", [])):
            field = self.query_cards[idx]["field"] if idx < len(self.query_cards) else ""
            if not field:
                lbl.config(text="")
                continue
            parts = []
            if counts is not None and idx in active_idx:
                k = active_idx.index(idx)
                if k + 1 < len(counts):
                    parts.append(f"{counts[k]} -> {counts[k + 1]}")
            if stat2d is not None:
                if field in stat2d:
                    v = stat2d[field]
                    parts.append(f"={v:.4g}" if isinstance(v, float) else f"={v}")
                else:
                    parts.append("=n/a")
            lbl.config(text="  ".join(parts))

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def _export_config(self):
        if not self.subsequences:
            messagebox.showinfo("mscoupon", "Define at least one subsequence first.")
            return
        out_dir = filedialog.askdirectory(title="Choose an output folder for config(s)")
        if not out_dir:
            return
        paths = self._write_configs(out_dir)
        self.status_var.set(f"Wrote {len(paths)} config(s) to {out_dir}")
        messagebox.showinfo("mscoupon", "Exported:\n" + "\n".join(os.path.basename(p) for p in paths))

    def _config_for(self, files, output_folder, folder=None):
        """The AppConfig dict for one file list.

        Single source of truth for both the exported config_N.json and the
        auto-saved session, so the two cannot drift apart.
        """
        try:
            pct = float(self.persist_pct_var.get())
        except ValueError:
            pct = 10.0
        min_area = None
        if self.min_area_var.get().strip():
            try:
                min_area = int(self.min_area_var.get())
            except ValueError:
                min_area = None
        return config_io.build_config(
            files=files,
            output_folder=output_folder,
            filters=self.filter_cards,
            base_filters=self.base_cards,
            persistence_percent=pct,
            manifold=self.manifold_var.get(),
            accurate=self.accurate_var.get(),
            extremum_sample_radius=self._ext_radius(),
            min_area=min_area,
            feature_filters=self.query_cards,
            pixel_filters=self.pixel_cards,
            connectivity=self.connectivity_var.get(),
            cores_per_slice=self._cores_per_slice(),
            concurrent_slices=self._concurrent_slices(),
            stat_channels=self._stat_channel_cards(),
            stat_reductions=self._stat_reductions(),
            stat_extremum=self.stat_extremum_var.get(),
            folder=folder,
        )

    def _write_configs(self, out_dir):
        """Write one config.json per subsequence; returns the written paths."""
        paths = []
        for i, s in enumerate(self.subsequences):
            cfg = self._config_for(s["files"], os.path.join(out_dir, f"out_{i}"))
            path = os.path.join(out_dir, f"config_{i}.json")
            config_io.dump_config(cfg, path)
            paths.append(path)
        return paths

    # ------------------------------------------------------------------ #
    # Load a config / restore the last session / auto-save
    #
    # All of this is best-effort. A config may be hand-edited, may name a folder
    # that has since moved, or may predate a schema change, and none of that may
    # raise inside a Tk callback -- there it surfaces as a traceback on stderr
    # and a button that appeared to do nothing. Problems are collected as notes
    # and shown in the status bar (in full via log()) rather than as dialogs: one
    # dialog per skipped file in a 20-file multi-select is unusable.
    # ------------------------------------------------------------------ #
    _AUTOSAVE_MS = 4000

    def _gui_state(self):
        """The GUI-only half of a session: what AppConfig cannot express.

        Carries no timestamp on purpose -- the auto-save decides whether to write
        by comparing this text against the last text written, and a clock would
        differ on every tick and write forever.
        """
        return {
            "version": config_io.SESSION_VERSION,
            "folder": self.folder,
            "subsequences": [{"name": s["name"], "files": list(s["files"])}
                             for s in self.subsequences],
            "persist_live": self.persist_live_var.get(),
            "seg_source": self.seg_source_var.get(),
            "background": self.background_var.get(),
            "mask": bool(self.mask_var.get()),
            "alpha": float(self.alpha_var.get()),
            "vmin": float(self.vmin_var.get()),
            "vmax": float(self.vmax_var.get()),
            "vmin_filt": float(self.vmin_filt_var.get()),
            "vmax_filt": float(self.vmax_filt_var.get()),
        }

    def _session_state(self):
        """A valid AppConfig for the FIRST subsequence -- so the saved session is
        still `mscoupon --config`-runnable -- plus the GUI half under "_gui"."""
        files = list(self.subsequences[0]["files"]) if self.subsequences else []
        folder = self.folder or (os.path.dirname(files[0]) if files else ".")
        cfg = self._config_for(files, os.path.join(folder, "out_0"), folder=folder)
        return config_io.build_session(cfg, self._gui_state())

    def _apply_state(self, state, gui=None, notes=None):
        """Push a `config_io.config_to_state()` dict onto the widgets.

        Each step is guarded on its own, so a value one widget rejects costs that
        control and not the rest of the load.
        """
        gui = gui or {}
        notes = notes if notes is not None else []

        def setvar(var, value):
            try:
                var.set(value)
            except (tk.TclError, ValueError, TypeError):
                notes.append(f"ignored unusable value {value!r}")

        # 1. Drop the compute state FIRST, before anything rebuilds against it:
        # the parameters that produced it are being replaced, so every primed
        # stack, cached slice, assembly and hover context is now stale.
        # (engine.reset() leaves a running assembly worker running: its result
        # lands as not-accepted via the bumped token, and the pipes are not
        # re-entrant, so a second worker must not start beside it.)
        self.engine.reset()
        self._hover_ctx = None
        self._selection_dirty = False

        # 2. Scalars.
        if state.get("manifold"):
            setvar(self.manifold_var, state["manifold"])
        setvar(self.accurate_var, bool(state.get("accurate")))
        setvar(self.ext_radius_var, str(int(state.get("extremum_sample_radius") or 0)))
        min_area = state.get("min_area")
        setvar(self.min_area_var, "" if min_area is None else str(int(min_area)))
        if state.get("connectivity"):
            setvar(self.connectivity_var, int(state["connectivity"]))
        if state.get("cores_per_slice"):
            setvar(self.cores_per_slice_var, int(state["cores_per_slice"]))
        if state.get("concurrent_slices"):
            setvar(self.concurrent_slices_var, int(state["concurrent_slices"]))
        # The cap and the live value are independent vars and _persist_pct clamps
        # live against the cap, so restoring a 30% cap while live still read 10%
        # would render at a threshold the config never asked for.
        pct = state.get("persistence_percent")
        if pct is not None:
            setvar(self.persist_pct_var, f"{float(pct):g}")
            setvar(self.persist_live_var, f"{float(pct):g}")
        if gui.get("persist_live") is not None:
            setvar(self.persist_live_var, str(gui["persist_live"]))

        # 2b. The measurement spec, restored BEFORE the query cards: their
        # dropdowns are generated from the field universe it defines, so
        # rebuilding them first would offer the previous spec's channels.
        self._apply_stat_state(state, setvar)

        # 3./4. Card chains, each with the trailing add-row the GUI expects, then
        # rebuilt in _build_left's order: the BASE rebuild is what resets
        # _normalize_readouts, so running it first would leave the topo chain's
        # readouts bound to destroyed widgets.
        self.filter_cards = list(state.get("filters") or []) + [self._new_filter_card()]
        self.base_cards = list(state.get("base_filters") or []) + [self._new_filter_card()]
        self.query_cards = list(state.get("feature_filters") or []) + [self._new_query_card()]
        self.pixel_cards = list(state.get("pixel_filters") or []) + [self._new_pixel_card()]
        self._rebuild_filter_cards()
        self._rebuild_filter_cards("base")
        self._rebuild_query_cards()
        self._rebuild_pixel_cards()
        self._refresh_channel_picker()
        self._refresh_stat_summary()

        # 5. Input side. A folder that has moved is not fatal -- the subsequences
        # carry absolute paths of their own.
        folder = gui.get("folder") or state.get("folder") or ""
        if folder and os.path.isdir(folder):
            self._set_folder(folder)
        elif folder:
            notes.append(f"folder not found: {folder}")

        # 6. Subsequences, resolved by the caller (it sees every document).
        if state.get("subsequences") is not None:
            self.subsequences = list(state["subsequences"])
            self._refresh_subseq_list()

        # 7. View state, all optional.
        for key, var in (("alpha", self.alpha_var), ("vmin", self.vmin_var),
                         ("vmax", self.vmax_var), ("vmin_filt", self.vmin_filt_var),
                         ("vmax_filt", self.vmax_filt_var)):
            if gui.get(key) is not None:
                try:
                    setvar(var, float(gui[key]))
                except (TypeError, ValueError):
                    notes.append(f"ignored {key}={gui[key]!r}")
        if gui.get("seg_source"):
            setvar(self.seg_source_var, str(gui["seg_source"]))
        if gui.get("background"):
            setvar(self.background_var, str(gui["background"]))
        if gui.get("mask") is not None:
            setvar(self.mask_var, bool(gui["mask"]))

        # 8. Settle the UI. Nothing is primed, so Rerun has nothing to redo --
        # the next step is Run.
        self._rebuild_flat_slices()
        try:
            self.rerun_btn.config(state="disabled")
            if not self._run_active:
                self.run_btn.config(state="normal")
        except tk.TclError:
            pass
        try:
            self._update_busy()
            self._refresh_render()
        except Exception as exc:            # a stale render must not eat the load
            log(f"redraw after load failed: {exc}")
        return notes

    def _apply_documents(self, docs, source):
        """Apply already-parsed config documents: `[(path, doc_or_None), ...]`.

        Parameters come from the FIRST readable document. Subsequences come from
        the `_gui` half when a single session file carries them, and otherwise
        one per document -- which is what makes a multi-select of the
        `config_0..config_N` that Export wrote reconstruct all N stacks.
        """
        notes = []
        good = []
        for path, doc in docs:
            if doc is None:
                notes.append(f"could not read {os.path.basename(path)}")
            else:
                good.append((path, doc))
        if not good:
            for msg in notes:
                log(msg)
            self.status_var.set("Could not read any config; nothing changed.")
            return

        cfg0, gui0 = config_io.split_session(good[0][1])
        state = config_io.config_to_state(cfg0, fields=self._query_fields(), notes=notes)

        saved = gui0.get("subsequences")
        if len(good) == 1 and isinstance(saved, list) and saved:
            raw = [(s.get("name"), s.get("files")) for s in saved
                   if isinstance(s, dict)]
        else:
            raw = []
            for _path, doc in good:
                cfg, _gui = config_io.split_session(doc)
                one = config_io.config_to_state(cfg)
                raw.append((None, one["files"]))

        subs = []
        for name, files in raw:
            files = [f for f in (files or []) if isinstance(f, str) and f]
            files = self._existing_files(files, len(subs) + 1, notes)
            if not files:
                notes.append(f"seq{len(subs) + 1}: no files found - skipped")
                continue
            subs.append({"name": name or f"seq{len(subs) + 1} ({len(files)})",
                         "files": files})
        state["subsequences"] = subs

        self._apply_state(state, gui0, notes)
        for msg in notes:
            log(msg)
        summary = f"Loaded {source}: {len(subs)} subsequence(s)"
        if notes:
            summary += " — " + "; ".join(notes[:2])
            if len(notes) > 2:
                summary += f" (+{len(notes) - 2} more, see the log)"
        self.status_var.set(summary)

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

    def _load_config(self):
        if self._run_active:
            self.status_var.set("Busy priming — wait for the run to finish.")
            return
        paths = filedialog.askopenfilenames(
            title="Load config.json (multi-select = one subsequence per file)",
            filetypes=[("Config JSON", "*.json"), ("All files", "*.*")])
        paths = list(paths or [])
        if not paths:
            return
        self._apply_documents([(p, config_io.read_json_file(p)) for p in paths],
                              f"{len(paths)} config(s)")

    def _restore_last(self):
        if self._run_active:
            self.status_var.set("Busy priming — wait for the run to finish.")
            return
        path = config_io.session_path(app=self.SESSION_APP)
        doc = config_io.read_json_file(path)
        if doc is None:
            self.status_var.set(f"No saved session at {path}")
            return
        self._apply_documents([(path, doc)], "last session")

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
            log(f"auto-save skipped: {exc}")
        self._schedule_autosave()

    def _autosave_now(self):
        """Write the session, but only when it actually differs.

        Safe while a run or an assembly is in flight: this reads only state the
        UI thread owns (subsequences, the card lists, the Tk vars) and never
        primed/_slices/_assembly, which the workers write.
        """
        blob = config_io.serialize_session(self._session_state())
        if blob == self._autosave_last:
            return
        if config_io.write_session_text(blob, config_io.session_path(app=self.SESSION_APP)):
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
                log(f"auto-save skipped: {exc}")
        else:
            self.autosave_label.config(text="auto-save off")

    def _on_close(self):
        try:
            if self.autosave_var.get():
                self._autosave_now()
        except Exception as exc:
            log(f"auto-save on close skipped: {exc}")
        try:
            if self._autosave_after is not None:
                self.root.after_cancel(self._autosave_after)
        except Exception:
            pass
        # The workers are daemon threads, so there is nothing to join.
        self.root.destroy()


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def main():
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        return _selftest()
    initial = args[0] if args else None
    root = tk.Tk()
    MscouponApp(root, initial)
    root.mainloop()


def _selftest():
    """Exercise the pure-Python control logic headlessly (no engine/render)."""
    import tempfile
    root = tk.Tk()
    root.withdraw()
    # autosave=False: this builds a real app, and a test must not overwrite the
    # user's saved session.
    app = MscouponApp(root, autosave=False)

    # Synthetic file browser state -> subsequence grouping.
    app.all_files = [f"/data/asdf_{i:04d}.tiff" for i in [11, 12, 13, 25, 26]]
    app.file_list.delete(0, "end")
    for f in app.all_files:
        app.file_list.insert("end", os.path.basename(f))
    runs = group_contiguous([0, 1, 2, 3, 4])
    assert runs == [[0, 1, 2, 3, 4]], runs
    assert group_contiguous([0, 1, 2, 4, 5]) == [[0, 1, 2], [4, 5]]

    # Two subsequences from two contiguous selections.
    app.file_list.selection_set(0, 2); app._make_subsequence()
    app.file_list.selection_clear(0, "end")
    app.file_list.selection_set(3, 4); app._make_subsequence()
    assert len(app.subsequences) == 2, app.subsequences
    assert [len(s["files"]) for s in app.subsequences] == [3, 2]

    # Filter chain: choose blur -> a trailing none card is appended.
    app._on_filter_op_change(0, "blur")
    assert app.filter_cards[0]["operation"] == "blur"
    assert app.filter_cards[-1]["operation"] == "none"
    app.filter_cards[0]["params"]["sigma"] = 2.0

    # Base channel: a normalize stage, on its own independent chain.
    app._on_filter_op_change(0, "normalize", "base")
    assert app.base_cards[0]["operation"] == "normalize"
    assert app.base_cards[-1]["operation"] == "none"
    assert app.filter_cards[0]["operation"] == "blur", "chains must stay independent"
    app.base_cards[0]["params"].update({"method": "gmm", "low_from": "", "high_from": ""})

    # Landmark readouts against a primed stack. This path only runs once a run
    # finishes, so nothing above reaches it -- and it is driven from
    # _refresh_render, where a bad slice accessor surfaces as a Tkinter callback
    # traceback rather than a failed assertion.
    from .normalize import TwoPoint
    saved_flat, saved_primed = app.flat_slices, app.primed
    app.flat_slices = [(0, 0), (0, 1)]
    app.primed = [{"files": ["a.tif", "b.tif"], "base": [], "filtered": [], "pipes": [],
                   "normalizers": [[TwoPoint(0.25, 0.75)], []]}]
    assert app._normalize_readouts, "normalize card should own a readout var"
    app.slice_var.set(0)
    app._refresh_normalize_readouts()
    assert "0.25" in app._normalize_readouts[0].get(), app._normalize_readouts[0].get()
    app.slice_var.set(1)          # primed, but this slice measured nothing
    app._refresh_normalize_readouts()
    assert "run to measure" in app._normalize_readouts[0].get()
    app.slice_var.set(99)         # out of range must not raise
    app._refresh_normalize_readouts()
    app.flat_slices, app.primed = saved_flat, saved_primed

    # Assembly tiers: the overlay choice decides how much work a Rerun does, so a
    # mis-mapped level silently reintroduces the full 3D assembly on every edit.
    app.mask_var.set(False)
    for src, want in (("none", "slice"), ("msc", "slice"), ("msc_kept", "slice"),
                      ("cc", "cc"), ("global", "global")):
        app.seg_source_var.set(src)
        got = app._needed_level()
        assert got == want, f"seg source {src!r} -> {got!r}, expected {want!r}"
    app.seg_source_var.set("msc")
    app.mask_var.set(True)
    assert app._needed_level() == "global", "the mask is painted from global ids"
    app.mask_var.set(False)
    app.seg_source_var.set("msc")

    # _slice_ready gates on commit and on tier, so a stale or too-cheap cache
    # entry does not satisfy a request.
    app._slices[(0, 0)] = {"commit": app._commit_id, "labels": None, "stats": {},
                           "kept": set(), "cc": None}
    assert app._slice_ready(0, 0, "slice"), "fresh slice record satisfies the slice tier"
    assert not app._slice_ready(0, 0, "cc"), "no CC cached -> cc tier not ready"
    assert not app._slice_ready(0, 0, "global"), "no 3D result -> global tier not ready"
    app._slices[(0, 0)]["commit"] = app._commit_id - 1
    assert not app._slice_ready(0, 0, "slice"), "stale commit is not ready"
    app._slices.clear()

    # Per-slice selection chain (2D merged-region queries).
    app._on_query_field_change(0, "area")
    app.query_cards[0]["op"] = "ge"; app.query_cards[0]["value"] = 50.0
    assert app.query_cards[-1]["field"] == ""

    # Pixel intensity trim chain: choose a base channel -> trailing card appended.
    app._on_pixel_channel_change(0, "filtered")
    app.pixel_cards[0]["mode"] = "omit"; app.pixel_cards[0]["op"] = "lt"
    app.pixel_cards[0]["value"] = 0.1
    assert app.pixel_cards[-1]["channel"] == ""

    # Config export matches the C++ schema (per-slice selection + pixel trim).
    with tempfile.TemporaryDirectory() as d:
        paths = app._write_configs(d)
        assert len(paths) == 2
        cfg = json.load(open(paths[0]))
        assert cfg["input"]["files"] == app.subsequences[0]["files"]
        assert cfg["filters"] == [{"operation": "blur", "params": {"sigma": 2.0}}]
        assert cfg["msc"]["manifold"] == "ascending"
        assert cfg["feature_filters"] == [{"field": "area", "op": "ge", "value": 50.0}]
        assert cfg["pixel_filters"] == [
            {"channel": "filtered", "mode": "omit", "op": "lt", "value": 0.1}]
        assert cfg["assembly"]["connectivity"] == 6
        # Blank landmark names must be dropped, not exported as "", so the CLI
        # falls back to the method's default pair instead of failing a lookup.
        assert cfg["base_filters"] == [{"operation": "normalize", "params": {
            "method": "gmm", "omit_value": 0.0, "downsample_factor": 1, "clamp": False,
        }}], cfg["base_filters"]

        # A blank no-data sentinel must reach the CLI as an explicit null: it
        # means "keep every pixel", and dropping the key would silently restore
        # the default of 0 instead.
        app.base_cards[0]["params"]["omit_value"] = ""
        blanked = json.load(open(app._write_configs(d)[0]))
        assert blanked["base_filters"][0]["params"]["omit_value"] is None,             blanked["base_filters"][0]["params"]
        app.base_cards[0]["params"]["omit_value"] = 43.0
        sentinel = json.load(open(app._write_configs(d)[0]))
        assert sentinel["base_filters"][0]["params"]["omit_value"] == 43.0

        # Session round-trip: loading what the viewer saved must put every
        # control back, and re-exporting must reproduce the same config.
        before = json.load(open(app._write_configs(d)[0]))
        doc = app._session_state()
        assert "_gui" in doc, "the session carries the GUI half"
        assert len(doc["_gui"]["subsequences"]) == 2
        cfg2, gui2 = config_io.split_session(doc)
        assert "_gui" not in cfg2, "the wrapper must not leak into the AppConfig"

        # Clobber every widget-backed value, then restore from the session.
        app._clear_subsequences()
        app.filter_cards = [app._new_filter_card()]; app._rebuild_filter_cards()
        app.base_cards = [app._new_filter_card()]; app._rebuild_filter_cards("base")
        app.query_cards = [app._new_query_card()]; app._rebuild_query_cards()
        app.pixel_cards = [app._new_pixel_card()]; app._rebuild_pixel_cards()
        app.persist_pct_var.set("99"); app.connectivity_var.set(26)
        app.manifold_var.set("descending"); app.min_area_var.set("7")
        app._apply_documents([(os.path.join(d, "last_session.json"), doc)], "test")

        assert len(app.subsequences) == 2, app.subsequences
        assert app.subseq_list.size() == 2, "the listbox must follow the model"
        assert app.subsequences[0]["files"] == before["input"]["files"]
        assert app.filter_cards[0]["operation"] == "blur"
        assert app.filter_cards[0]["params"]["sigma"] == 2.0
        assert app.filter_cards[-1]["operation"] == "none", "trailing add-card"
        assert app.base_cards[0]["operation"] == "normalize"
        assert app.base_cards[0]["params"]["omit_value"] == 43.0
        assert app.base_cards[0]["params"]["low_from"] == "", "blank landmark stays blank"
        assert app.base_cards[-1]["operation"] == "none"
        assert app.query_cards[0]["field"] == "area"
        assert app.query_cards[0]["value"] == 50.0
        assert app.query_cards[-1]["field"] == ""
        assert app.pixel_cards[0]["channel"] == "filtered"
        assert app.pixel_cards[-1]["channel"] == ""
        assert float(app.persist_pct_var.get()) == 10.0
        assert app.manifold_var.get() == "ascending"
        assert app.connectivity_var.get() == 6
        assert app.min_area_var.get() == "", "a config without segments clears the gate"
        # A load must leave no primed/assembly state behind -- the parameters
        # that produced it were just replaced.
        assert app.primed == [] and app.flat_slices == [] and not app._slices
        assert app._asm_pending is None and app._selection_dirty is False
        assert str(app.rerun_btn.cget("state")) == "disabled"

        after = json.load(open(app._write_configs(d)[0]))
        assert after == before, (before, after)

        # A blank no-data sentinel must come back BLANK, not as the default 0.0.
        app.base_cards[0]["params"]["omit_value"] = ""
        doc_b = app._session_state()
        cfg_b, gui_b = config_io.split_session(doc_b)
        app._apply_state(config_io.config_to_state(cfg_b), gui_b)
        assert app.base_cards[0]["params"]["omit_value"] == "",             app.base_cards[0]["params"]
        app.base_cards[0]["params"]["omit_value"] = 43.0

        # Junk must not raise into the mainloop, and an unreadable document must
        # leave the state alone rather than half-applying it.
        app._apply_state(config_io.config_to_state({"msc": "nonsense", "filters": 7}))
        kept = [dict(s) for s in app.subsequences]
        app._apply_documents([(os.path.join(d, "gone.json"), None)], "missing")
        assert app.subsequences == kept, "a failed load changes nothing"
        assert isinstance(config_io.session_path(), str) and config_io.session_path()

        # Restore the chains the remaining checks below expect.
        app.filter_cards = [{"operation": "blur", "params": {"sigma": 2.0}},
                            app._new_filter_card()]

        # A workflow with no base chain must export exactly as it did before.
        app.base_cards = [app._new_filter_card()]
        plain = json.load(open(app._write_configs(d)[0]))
        assert "base_filters" not in plain
        # ... and neither does a workflow that never touched the statistics
        # panel: the default spec is not emitted at all.
        assert "statistics" not in plain, plain.get("statistics")

        # --- measurement channels ---------------------------------------- #
        # A scale-space stack: three sigmas x three kinds, hessian splitting into
        # largest/smallest, so 3 + 3 + 6 derived channels on top of base.
        app.stat_kind_vars["blur"][0].set(True)
        app.stat_kind_vars["blur"][1].set("0.7, 1.5, 3.0")
        app.stat_kind_vars["edges"][0].set(True)
        app.stat_kind_vars["edges"][1].set("0.7, 1.5, 3.0")
        app.stat_kind_vars["hessian"][0].set(True)
        app.stat_kind_vars["hessian"][1].set("0.7, 1.5, 3.0")
        app.stat_reduction_vars["std"].set(False)
        cards = app._stat_channel_cards()
        assert cards[0] == {"kind": "base"}, cards
        assert {"kind": "blur", "sigmas": [0.7, 1.5, 3.0]} in cards, cards
        assert app._stat_reductions() == ["mean", "min", "max"], app._stat_reductions()

        block = json.loads(app._params_json())["statistics"]
        assert block["reductions"] == ["mean", "min", "max"], block
        assert "statistics" in json.load(open(app._write_configs(d)[0]))

        # The spec must survive a session round trip -- the whole block used to
        # be dropped on load except extremum_sample_radius.
        doc_s = app._session_state()
        cfg_s, gui_s = config_io.split_session(doc_s)
        state_s = config_io.config_to_state(cfg_s)
        assert state_s["stat_reductions"] == ["mean", "min", "max"], state_s
        app.stat_kind_vars["blur"][0].set(False)          # perturb, then restore
        app.stat_reduction_vars["std"].set(True)
        app._apply_state(state_s, gui_s)
        assert app.stat_kind_vars["blur"][0].get(), "blur channel restored"
        assert app._stat_reductions() == ["mean", "min", "max"], app._stat_reductions()
        assert _parse_sigmas(app.stat_kind_vars["hessian"][1].get()) == [0.7, 1.5, 3.0]

        # The two-level picker must cover every field, and must not mistake the
        # geometry columns for reductions of a channel.
        order, by_channel = app._query_pickers()
        assert config_io.GEOMETRY_CHANNEL in by_channel
        assert "min_x" in by_channel[config_io.GEOMETRY_CHANNEL], by_channel
        if "blur_s1.5" in by_channel:      # only with the extension built
            assert "mean" in by_channel["blur_s1.5"], by_channel["blur_s1.5"]
            assert config_io.compose_field("blur_s1.5", "mean") == "mean_blur_s1.5"
            assert config_io.split_field("mean_blur_s1.5",
                                         app._params_json()) == ("blur_s1.5", "mean")
            fields = app._query_fields()
            assert "mean_blur_s1.5" in fields and "std_blur_s1.5" not in fields, fields
            # Every offered (channel, reduction) pair must name a real field.
            for channel in order:
                for reduction in by_channel[channel]:
                    assert config_io.compose_field(channel, reduction) in fields

        # The background dropdown resolves a channel NAME against the primed
        # stack. `primed` is a list indexed by subsequence, and the render
        # callback is the only caller -- so an accessor slip here shows up as a
        # Tkinter traceback and a picture that never changes, not as an error.
        import numpy as _np
        saved_primed2 = app.primed
        app.primed = []
        assert app._channel_raster(0, 0, "base", _np) is None, "nothing primed -> None"
        b0 = _np.full((4, 4), 1.0, _np.float32)
        f0 = _np.full((4, 4), 2.0, _np.float32)
        app.primed = [{"files": ["a.tif"], "base": [b0], "filtered": [f0],
                       "pipes": [], "normalizers": []}]
        assert app._channel_raster(0, 0, "base", _np) is b0
        assert app._channel_raster(0, 0, "", _np) is b0, "blank name reads as base"
        assert app._channel_raster(0, 0, "filtered", _np) is f0
        assert app._channel_raster(1, 0, "base", _np) is None, "subsequence out of range"
        assert app._channel_raster(0, 9, "base", _np) is None, "slice out of range"
        # An unknown channel must degrade to the base raster, never to None: the
        # renderer would otherwise blank the canvas.
        assert app._channel_raster(0, 0, "no_such_channel", _np) is not None
        app.primed = saved_primed2

        # Restore the default spec so nothing below inherits a wide channel set.
        for on, _sig in app.stat_kind_vars.values():
            on.set(False)
        app.stat_reduction_vars["std"].set(True)
    print("selftest OK: sequences, filters, base chain, stat channels, assembly tiers, "
          "per-slice selection, pixel trim, export, config load + session round-trip")
    root.destroy()


if __name__ == "__main__":
    main()
