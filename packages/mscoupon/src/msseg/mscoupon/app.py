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
from .config_io import (FILTER_SCHEMA, FILTER_OPERATIONS, COLOR_METHODS, QUERY_OPS,
                        filter_param_schema, query_fields)
# Shared helpers live in common.py (re-exported here so existing imports of
# `msseg.mscoupon.app` keep working) and the small reusable widgets in
# widgets.py -- both are shared with the labeler app.
from .common import (log, natural_key, list_tiffs, _id_lut, FeatureTable, _parse_sigmas, _format_sigmas, group_contiguous)
from msseg.labeler.widgets import (ScrollFrame, jump_scale, scrolled_listbox, attach_tooltip,
                                   _wheel_delta, _bind_click_to_value)
from .engine import ComputeEngine, preview_job, preview_raster
from msseg.labeler.shell import ViewerShell
from msseg.labeler.defaults import _PREVIEW_PUMP_MS
from . import session


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
# Byte budget for channels computed on previews (a 3232^2 float32 raster is
# ~42 MB): the base chain (a GMM normalize is ~5 s) and a wide blur (~3.5 s at
# sigma 64) must survive a few channel switches.
_PREVIEW_CHAN_BUDGET = 640 * 1024 * 1024
# ...but once a stack is primed its own per-slice base/filtered rasters are
# resident too, and the live preview keeps filling this beside them.
_PREVIEW_CHAN_BUDGET_PRIMED = 256 * 1024 * 1024


class _RasterCache:
    """LRU of numpy rasters under a byte budget. Evicting by count was the
    wrong unit: a stack of derived channels is a handful of ~40 MB planes,
    and clearing the lot on the fifth entry re-ran a 5 s normalize."""

    def __init__(self, budget):
        from collections import OrderedDict
        self.budget = int(budget)
        self._d = OrderedDict()
        self.nbytes = 0

    def get(self, key):
        hit = self._d.get(key)
        if hit is not None:
            self._d.move_to_end(key)
        return hit

    def put(self, key, raster):
        old = self._d.pop(key, None)
        if old is not None:
            self.nbytes -= int(old.nbytes)
        self._d[key] = raster
        self.nbytes += int(raster.nbytes)
        while self.nbytes > self.budget and len(self._d) > 1:
            _k, v = self._d.popitem(last=False)
            self.nbytes -= int(v.nbytes)

    def clear(self):
        self._d.clear()
        self.nbytes = 0

    def __len__(self):
        return len(self._d)


def parse_hist_ranges(text):
    """`"*: 0, 1; hessian_largest_s1.5: -0.2, 0.2"` -> {name: [lo, hi]}."""
    out = {}
    for part in str(text or "").split(";"):
        if ":" not in part:
            continue
        name, _, pair = part.partition(":")
        nums = [t for t in pair.replace(",", " ").split() if t]
        try:
            out[name.strip()] = [float(nums[0]), float(nums[1])]
        except (IndexError, ValueError):
            continue
    return out


def format_hist_ranges(ranges):
    """Inverse of parse_hist_ranges."""
    return "; ".join(f"{k}: {v[0]:g}, {v[1]:g}" for k, v in ranges.items() if len(v) == 2)


def single_channel_params(params_json, name):
    """`params_json` with its statistics block cut down to the ONE derived
    channel `name` (its kind at its sigma, keeping the card's other keys such
    as hessian's sort_by_absolute_value), or None when `name` is not a
    derived channel of the spec.

    stat_channel_images builds every channel a spec names; asking for one
    plane out of a twelve-channel spec used to pay for all twelve -- and a
    blur at sigma 64 is ~3.5 s at 3232^2 on its own. Each channel's response
    is independent of the others, so a one-channel spec yields the same
    raster."""
    try:
        card = next(c for c in config_io.stat_channels(params_json)
                    if c.get("name") == name)
    except StopIteration:
        return None
    kind = card.get("kind")
    if kind in ("base", "filtered", "color") or not kind:
        return None
    source = card.get("source") or "base"
    doc = json.loads(params_json)
    stats = dict(doc.get("statistics") or {})
    keep = {}
    for c in stats.get("channels") or []:
        if (isinstance(c, dict) and c.get("kind") == kind
                and (c.get("source") or ("color" if kind in config_io.COLOR_ONLY_KINDS
                                         else "base")) == source):
            keep = {k: v for k, v in c.items() if k != "sigmas"}
            break
    keep["kind"] = kind
    keep["sigmas"] = [float(card.get("sigma", 0.0))]
    stats["channels"] = [keep]
    # A histogram names channels this one-channel spec no longer has, and a
    # preview raster needs no bins anyway.
    stats.pop("histogram", None)
    doc["statistics"] = stats
    return json.dumps(doc)


class MscouponApp(ViewerShell):
    # The per-app session-file identity (config_io.session_path(app=...));
    # subclasses (the labeler) override it so their sessions never collide.
    SESSION_APP = "mscoupon"
    # The framework shell's identity/dialog strings and the session-file I/O
    # module (config_io re-exports the framework's; routing through it keeps
    # `config_io.session_path` monkeypatchable, which the selftest relies on).
    APP_TITLE = "mscoupon"
    WINDOW_TITLE = "mscoupon viewer"
    LOG_PREFIX = "mscoupon"
    SESSION_IO = config_io

    def _default_profile(self, name="default"):
        """Factory hook for tools with a different lean statistics default."""
        return session.default_profile(name)

    def _make_catalogue(self):
        """Factory hook: the framework ItemCatalogue over this app's data."""
        from .adapters import SequenceCatalogue
        return SequenceCatalogue(self)

    def _make_region_provider(self):
        """Factory hook: the framework RegionProvider over this app's compute."""
        from .adapters import EngineRegionProvider
        return EngineRegionProvider(self)

    def _init_compute(self):
        """The coupon compute model: the parameter cards and the engine
        (the shell asks for the catalogue and region provider after this)."""
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

    def _init_variables(self):
        """The coupon Tk variables (the shell owns slice/alpha/window/status)."""
        # --- tk variables ------------------------------------------------ #
        self.persist_pct_var = tk.StringVar(value="10")
        self.manifold_var = tk.StringVar(value="ascending")
        self.accurate_var = tk.BooleanVar(value=False)
        self.gpu_var = tk.BooleanVar(value=False)   # msc.use_gpu_gradient
        # msc.simplification: "merge_forest" (extremum network, the default
        # since 2026-09-02) | "msc" (the cancellation hierarchy). The default
        # lives in FOUR places that cannot import each other -- Msc2DParams,
        # mscoupon::Config, session.default_profile and this var -- and this
        # one WINS, because _profile_from_ui writes it into the params JSON
        # unconditionally. It was the one missed when the default flipped, so a
        # GUI-saved session pinned "msc" over every other default.
        self.simplification_var = tk.StringVar(value=session.DEFAULT_SIMPLIFICATION)
        self.ext_radius_var = tk.StringVar(value="0")
        self.min_area_var = tk.StringVar(value="")
        self.connectivity_var = tk.IntVar(value=6)
        self.seg_source_var = tk.StringVar(value="global")  # none|msc|cc|global
        self._selection_dirty = False   # selection params changed since last assembly
        self.cores_per_slice_var = tk.IntVar(value=max(1, (os.cpu_count() or 2) // 2))  # ~physical cores
        self.concurrent_slices_var = tk.IntVar(value=1)   # slices computed at once
        self.persist_live_var = tk.StringVar(value="10")   # live persistence % (numeric entry)
        # Measurement channels (`statistics.channels[]`): the two rasters the
        # pipeline already builds, plus derived scale-space responses measured on
        # the base channel. One card per kind; the sigma list is the cross-product
        # that makes a multi-scale stack one line of config instead of many.
        self.stat_base_var = tk.BooleanVar(value=True)
        self.stat_color_var = tk.BooleanVar(value=False)   # the raw colour planes
        # Per-region histograms: K bins over a fixed range per channel.
        self.hist_on_var = tk.BooleanVar(value=False)
        self.hist_bins_var = tk.StringVar(value="16")
        self.hist_channels_var = tk.StringVar(value="base")
        self.hist_ranges_var = tk.StringVar(value="*: 0, 1")
        # How a multi-sample TIFF is read (profile `input.color`): alpha policy,
        # the default colour->scalar method, and the plane count the statistics
        # schema is resolved for (filled from the slice on screen).
        self.color_alpha_var = tk.StringVar(value="drop")
        self.color_default_var = tk.StringVar(value="luminance")
        self.color_channels_var = tk.IntVar(value=0)
        self.color_planes_text = tk.StringVar(value="planes: - (grayscale)")
        self.stat_filtered_var = tk.BooleanVar(value=False)
        self.stat_kind_vars = {}       # kind -> (BooleanVar, StringVar sigmas)
        self.stat_reduction_vars = {}  # reduction -> BooleanVar
        self.stat_extremum_var = tk.BooleanVar(value=True)
        # Rasters for a derived channel are computed on demand for the slice on
        # screen and cached by (subsequence, slice, channel). Holding the whole
        # stack would cost one float32 raster per channel per primed slice.
        self._chan_cache = {}
        # Image preview (click / drag over the TIFF list, before any Run):
        # transient, last-writer-wins against _refresh_render, never touches
        # the engine. Loads are debounced and LRU-cached (~4 float32 slices).
        self._preview_after = None
        self._preview_path = None
        self._preview_cache = {}                          # path -> float32 array (LRU)
        # Channels computed ON a preview (base chain / topology chain / derived
        # scale-space responses), keyed by (path, channel, params): the Image
        # dropdown works before any Run, through the same chains a run applies.
        self._preview_chan_cache = _RasterCache(_PREVIEW_CHAN_BUDGET)
        # Live preview of the chain (a parameter edit, no Run): the worker the
        # recompute runs on (built lazily -- the root exists only once the
        # shell has laid the window out), what the pump is waiting for, the key
        # of the raster now painted, and the (path, channel, raster) standing
        # in for a primed view whose chain has since been edited.
        self._preview_worker = None
        self._preview_pending = None
        self._preview_shown_key = None
        self._preview_override = None
        self._preview_sync = False       # selftests run the worker inline
        # The chains the primed stack was built with. A live preview compares
        # itself against this to know whether it is showing something the
        # primed overlays still describe.
        self._primed_chain = None

    def _after_layout(self):
        # Offer the profile's channels from the start: a preview (no Run) can
        # show any of them.
        self._refresh_channel_picker()

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

    def _build_statistics_panel(self):
        """Which channels a feature is measured on, and with which reductions.

        A derived channel is a Gaussian-derivative response computed on the base
        raster; naming several sigmas on one row is the cross-product, so a
        scale-space stack is one line rather than one row per (kind, sigma). They
        are measure-only: the topology field is still `filters`, and the seeding
        extremum is still located on it.
        """
        c = self._group(self._processing_parent("stats"), "5. Statistics channels",
                        key="stats")
        self.stats_frame = c

        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=2)
        ttk.Checkbutton(row, text="base", variable=self.stat_base_var,
                        command=self._on_stat_spec_change).pack(side="left")
        ttk.Checkbutton(row, text="filtered", variable=self.stat_filtered_var,
                        command=self._on_stat_spec_change).pack(side="left", padx=8)
        # The raw input planes (color_c0, ...) of a colour slice.
        ttk.Checkbutton(row, text="color planes", variable=self.stat_color_var,
                        command=self._on_stat_spec_change).pack(side="left", padx=8)

        for kind in config_io.DERIVED_CHANNEL_KINDS:
            row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=1)
            on = tk.BooleanVar(value=False)
            sigmas = tk.StringVar(value="0.7, 1.5, 3.0")
            color_only = kind in config_io.COLOR_ONLY_KINDS
            ttk.Checkbutton(row, text=kind, variable=on, width=10,
                            command=self._on_stat_spec_change).pack(side="left")
            ttk.Label(row, text="sigmas:").pack(side="left")
            entry = ttk.Entry(row, textvariable=sigmas, width=16)
            entry.pack(side="left", padx=2)
            # Commit on Enter / focus-out only: every keystroke would re-resolve
            # the schema and rebuild the query dropdowns mid-typing.
            entry.bind("<Return>", lambda e: self._on_stat_spec_change())
            entry.bind("<FocusOut>", lambda e: self._on_stat_spec_change())
            # What the response is computed on: the base scalar, or the colour
            # planes (one response per plane; the cross-channel kinds only).
            source = tk.StringVar(value="color" if color_only else "base")
            src = ttk.Combobox(row, textvariable=source, values=config_io.STAT_SOURCES,
                               state="disabled" if color_only else "readonly", width=6)
            src.pack(side="left", padx=2)
            src.bind("<<ComboboxSelected>>", lambda e: self._on_stat_spec_change())
            if kind in config_io.TWO_SLOT_KINDS:
                ttk.Label(row, text="(largest + smallest)").pack(side="left")
            self.stat_kind_vars[kind] = (on, sigmas, source)

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
        # Histograms: a fixed range per channel (or "*" for all), so the bins
        # add across slices and mean the same thing on every slice.
        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=(4, 1))
        ttk.Checkbutton(row, text="histogram", variable=self.hist_on_var, width=10,
                        command=self._on_stat_spec_change).pack(side="left")
        ttk.Label(row, text="bins:").pack(side="left")
        for var, width in ((self.hist_bins_var, 4), (self.hist_channels_var, 14)):
            e = ttk.Entry(row, textvariable=var, width=width); e.pack(side="left", padx=2)
            e.bind("<Return>", lambda ev: self._on_stat_spec_change())
            e.bind("<FocusOut>", lambda ev: self._on_stat_spec_change())
            if var is self.hist_bins_var:
                ttk.Label(row, text="channels:").pack(side="left")
        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=1)
        ttk.Label(row, text="ranges (name: lo, hi; ...):").pack(side="left")
        e = ttk.Entry(row, textvariable=self.hist_ranges_var, width=22); e.pack(side="left", padx=2)
        e.bind("<Return>", lambda ev: self._on_stat_spec_change())
        e.bind("<FocusOut>", lambda ev: self._on_stat_spec_change())
        ttk.Button(row, text="measure", width=8,
                   command=self._measure_hist_ranges).pack(side="left", padx=2)
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
        setvar(self.stat_color_var, any(c.get("kind") == "color" for c in channels))
        by_kind = {}
        by_source = {}
        for card in channels:
            kind = card.get("kind")
            if kind in ("base", "filtered", "color") or kind not in self.stat_kind_vars:
                continue
            # One row per kind: a config naming a kind on both sources keeps
            # the first source it saw (the other lossy merge, besides sigmas).
            by_source.setdefault(kind, card.get("source") or "base")
            for sigma in card.get("sigmas") or []:
                by_kind.setdefault(kind, [])
                if sigma not in by_kind[kind]:
                    by_kind[kind].append(float(sigma))
        for kind, (on, sigmas, source) in self.stat_kind_vars.items():
            values = by_kind.get(kind)
            setvar(on, bool(values))
            if values:
                setvar(sigmas, _format_sigmas(sorted(values)))
            setvar(source, "color" if kind in config_io.COLOR_ONLY_KINDS
                   else by_source.get(kind, "base"))

        reductions = state.get("stat_reductions")
        if reductions is not None:
            for name, var in self.stat_reduction_vars.items():
                setvar(var, name in reductions)
        if state.get("stat_extremum") is not None:
            setvar(self.stat_extremum_var, bool(state["stat_extremum"]))
        hist = state.get("stat_histogram")
        if hist is not None:
            setvar(self.hist_on_var, bool(hist.get("bins")))
            if hist.get("bins"):
                setvar(self.hist_bins_var, str(int(hist["bins"])))
                setvar(self.hist_channels_var, ", ".join(hist.get("channels") or ["base"]))
                setvar(self.hist_ranges_var, format_hist_ranges(hist.get("ranges") or {}))
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
        # A colour slice also offers its composite and each raw plane.
        n_color = self._current_color_count()
        if n_color:
            extra = ["color"] + [f"color_c{i}" for i in range(n_color)]
            names = [n for n in extra if n not in names] + names
            # The plane count the statistics schema is resolved for.
            if self.color_channels_var.get() != n_color:
                self.color_channels_var.set(n_color)
            self.color_planes_text.set(f"planes: {n_color} (the slice on screen)")
        else:
            self.color_planes_text.set("planes: - (grayscale slice on screen)")
        self._picker_color_count = n_color
        try:
            combo.config(values=names)
        except tk.TclError:
            return
        if self.background_var.get() not in names:
            self.background_var.set(names[0] if names else "base")

    def _current_color_count(self):
        """Planes of the slice on screen (primed, else the preview), 0 for gray."""
        try:
            cur = self._current() if self.primed else None
        except Exception:
            cur = None
        if cur is not None:
            si, li = cur
            if si < len(self.primed):
                planes = self.primed[si].get("color") or []
                if li < len(planes) and planes[li] is not None:
                    return int(getattr(planes[li], "shape", (0,))[0])
            return 0
        arr = self._preview_cache.get(getattr(self, "_preview_path", None))
        if arr is not None and getattr(arr, "ndim", 2) == 3:
            return int(arr.shape[0])
        return 0

    @staticmethod
    def _color_plane(planes, name, fallback):
        """`color` -> the planes; `color_c<i>` -> one plane; else `fallback`."""
        if planes is None:
            return fallback
        if name == "color":
            return planes
        try:
            idx = int(name[len("color_c"):])
        except ValueError:
            return fallback
        return planes[idx] if 0 <= idx < planes.shape[0] else fallback

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
        if name == "color" or name.startswith("color_c"):
            planes = (p.get("color") or [])
            planes = planes[li] if li < len(planes) else None
            return self._color_plane(None if planes is None else np.asarray(planes), name,
                                     p["base"][li])
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
        # One-channel spec: the bank builds every channel it is asked for,
        # and only this plane is wanted.
        params = single_channel_params(self._params_json(), name)
        if params is None:
            return p["base"][li]
        t0 = time.perf_counter()
        try:
            from .engine import color_planes, stat_images
            names, imgs = stat_images(engine, np.asarray(p["base"][li], dtype=np.float32),
                                      np.asarray(p["filtered"][li], dtype=np.float32), params,
                                      color_planes(p, li, np))
        except Exception as exc:
            log(f"channel '{name}' unavailable: {exc}")
            return p["base"][li]
        names = list(names)
        log(f"channel {name} for slice {si}:{li}: "
            f"{1e3 * (time.perf_counter() - t0):.0f}ms ({len(names)} plane(s))")
        if name not in names:
            return p["base"][li]
        if len(self._chan_cache) > 8:
            self._chan_cache.clear()
        # A kind can yield several planes per sigma (hessian: largest +
        # smallest); keep them all, they came for free.
        for k, n in enumerate(names):
            self._chan_cache[(si, li, n)] = np.array(imgs[k], copy=True)
        return self._chan_cache[key]

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

    def _build_processing_sections(self):
        """The compute parameters: colour input, the two filter chains, the
        MSC parameters and the statistics channels (sections 1b-5)."""
        # 1b. Colour input: how a multi-sample TIFF's planes are read. The
        # conversion itself is a `color` stage at the head of each chain.
        c = self._group(self._processing_parent("filters"), "1b. Colour input",
                        key="color")
        self.color_frame = c
        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="alpha:").pack(side="left")
        cb = ttk.Combobox(row, textvariable=self.color_alpha_var, values=["drop", "keep"],
                          state="readonly", width=6)
        cb.pack(side="left", padx=(2, 8))
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_color_input_change(reload=True))
        ttk.Label(row, text="default method:").pack(side="left")
        cb = ttk.Combobox(row, textvariable=self.color_default_var, values=COLOR_METHODS,
                          state="readonly", width=14)
        cb.pack(side="left", padx=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_color_input_change())
        ttk.Label(c, textvariable=self.color_planes_text, foreground="#555").pack(
            anchor="w", padx=4, pady=(0, 3))

        # 2. Filter chain
        self.filters_frame = self._group(self._processing_parent("filters"),
                                         "2. Filter chain (topology field)",
                                         key="filters")
        self._rebuild_filter_cards()

        # 3. Base channel: 2-point normalization
        self.base_frame = self._group(self._processing_parent("base"),
                                      "3. Base channel (2-point normalization)",
                                      key="base")
        ttk.Label(self.base_frame, wraplength=330, justify="left",
                  text="Add a 'normalize' stage to put region statistics and pixel "
                       "thresholds on a 0..1 scale between two measured landmarks. "
                       "A threshold of 0.7 then means 0.3*low + 0.7*high on every "
                       "slice, so one value holds across a drifting stack."
                  ).pack(anchor="w", padx=4, pady=(2, 4))
        self._rebuild_filter_cards("base")

        # 4. MSC params
        c = self._group(self._processing_parent("msc"), "4. MSC parameters", key="msc")
        self.msc_frame = c
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
        ttk.Checkbutton(c, text="GPU gradient (CUDA; bit-identical results)",
                        variable=self.gpu_var).pack(anchor="w", padx=4)
        # Simplification is a SEPARATE axis from the GPU gradient (which only
        # decides where the discrete gradient is computed): this picks what is
        # built on top of it. Visible because the two were easy to confuse
        # while this one had no control at all -- a profile could sit on the
        # slower branch for weeks with nothing on screen or in the log saying so.
        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Simplification:").pack(side="left")
        ttk.Radiobutton(row, text="merge forest", variable=self.simplification_var,
                        value="merge_forest").pack(side="left", padx=4)
        ttk.Radiobutton(row, text="MSC hierarchy", variable=self.simplification_var,
                        value="msc").pack(side="left", padx=4)
        ttk.Label(c, text="merge forest: no MSC is built while priming (much faster); "
                          "differs from the MSC only in that it also merges away\n"
                          "corner minima, which the MSC keeps alive at every "
                          "persistence. MSC arcs are still built on demand.",
                  foreground="#666").pack(anchor="w", padx=4)
        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="ext sample radius:").pack(side="left")
        ttk.Entry(row, textvariable=self.ext_radius_var, width=8).pack(side="left", padx=4)
        ttk.Label(row, text="(0 = the critical pixel)").pack(side="left")
        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Per-slice min area:").pack(side="left")
        ttk.Entry(row, textvariable=self.min_area_var, width=8).pack(side="left", padx=4)

        # 5. Statistics channels
        self._build_statistics_panel()

    def _build_run_section(self):
        # 6. Run
        c = ttk.LabelFrame(self._left_section_parent("run"), text="6. Run")
        c.pack(fill="x", padx=6, pady=4)
        self.run_frame = c
        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Cores/slice:").pack(side="left")
        ttk.Entry(row, textvariable=self.cores_per_slice_var, width=5).pack(side="left", padx=(2, 10))
        ttk.Label(row, text="Concurrent slices:").pack(side="left")
        ttk.Entry(row, textvariable=self.concurrent_slices_var, width=5).pack(side="left", padx=2)
        self.run_btn = ttk.Button(c, text="Run with selected", command=self._run)
        self.run_btn.pack(fill="x", padx=4, pady=4)
        # (config.json export is on hold -- it will return as a bundle of the
        # compute profile AND the trained model. _config_for/_write_configs stay
        # as its dormant seed and as the selftest's AppConfig-equivalence probe.)

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
        # The conversion a multi-plane input gets when its chain does not start
        # with one is DERIVED: re-planned on every rebuild, never stored in
        # `cards`, so it cannot reach filters_to_json, a session or a config.
        # Drawing it is the whole of Stage 1 -- it has always run, but only
        # inside the runner, where nothing could see it.
        try:
            plan = config_io.chain_plan(cards, self._current_color_count() or 1,
                                        self._default_color_method())
            head = plan["stages"][0] if plan["stages"] else None
        except Exception:
            head = None
        if head is not None and head["synthesized"]:
            self._build_auto_color_card(frame, head, chain)
        for idx, card in enumerate(cards):
            self._build_filter_card(idx, card, chain)

    def _plan_record(self, chain, idx):
        """The planned arity of config stage `idx`, or None if it cannot be
        planned (a chain mid-edit often cannot)."""
        try:
            cards, _frame = self._chain(chain)
            plan = config_io.chain_plan(cards, self._current_color_count() or 1,
                                        self._default_color_method())
            for rec in plan["stages"]:
                if rec["index"] == idx:
                    return rec
        except Exception:
            return None
        return None

    def _build_auto_color_card(self, parent, stage, chain):
        """Draw the conversion the runner inserts, greyed and read-only.

        It is not one of `self.filter_cards`, so it is not exported, not saved
        and not editable in place. "Pin" makes it a real card at the head of the
        chain, which is the only way its method becomes editable -- and, later,
        the only way it becomes movable."""
        frame = ttk.Frame(parent, relief="groove", borderwidth=1)
        frame.pack(fill="x", padx=4, pady=2)
        top = ttk.Frame(frame); top.pack(fill="x")
        method = str(stage["params"].get("method", "luminance"))
        ttk.Label(top, text=f"(auto) color / {method}", foreground="#777",
                  width=22).pack(side="left", padx=2, pady=2)
        ttk.Label(top, text=f"{stage['in']}→{stage['out']}",
                  foreground="#777").pack(side="left", padx=2)
        ttk.Button(top, text="Pin", width=5,
                   command=lambda c=chain, m=method: self._pin_auto_color(c, m)
                   ).pack(side="right", padx=2)

    def _pin_auto_color(self, chain, method):
        """Make the synthesized conversion an ordinary card at index 0."""
        cards, _frame = self._chain(chain)
        cards.insert(0, {"operation": "color", "params": {"method": method}})
        self._set_chain_cards(chain, cards)
        self._rebuild_filter_cards(chain)
        self._notify_profile_edit()

    def _build_filter_card(self, idx, card, chain="topo"):
        cards, parent = self._chain(chain)
        frame = ttk.Frame(parent, relief="groove", borderwidth=1)
        frame.pack(fill="x", padx=4, pady=2)
        top = ttk.Frame(frame); top.pack(fill="x")
        op_var = tk.StringVar(value=card["operation"])
        # `color` consumes the input planes, so only the head of a chain may be one.
        ops = config_io.filter_operations_at(idx)
        combo = ttk.Combobox(top, textvariable=op_var, values=ops,
                             state="readonly", width=20)
        combo.pack(side="left", padx=2, pady=2)
        combo.bind("<<ComboboxSelected>>",
                   lambda e, i=idx, v=op_var, c=chain: self._on_filter_op_change(i, v.get(), c))
        # The plan's arity for this stage, so lifting is never silent: a card
        # reading `3->3 (per plane)` is the whole reason automatic lifting is
        # acceptable where the leading reduction was not.
        rec = self._plan_record(chain, idx)
        if rec is not None and (rec["in"] != 1 or rec["out"] != 1):
            text = f"{rec['in']}→{rec['out']}" + (" (per plane)" if rec.get("lifted") else "")
            ttk.Label(top, text=text, foreground="#777").pack(side="left", padx=4)
        if idx < len(cards) - 1 or card["operation"] != "none":
            ttk.Button(top, text="✕", width=3,
                       command=lambda i=idx, c=chain: self._remove_filter_card(i, c)
                       ).pack(side="right", padx=2)
        # param widgets for the selected operation
        if card["operation"] == "color":
            self._build_color_method_row(frame, card, chain)
            for pname, kind, default in filter_param_schema("color", card["params"])[1:]:
                self._build_param_row(frame, card["params"], pname, kind, default)
        else:
            for pname, kind, default in FILTER_SCHEMA.get(card["operation"], []):
                self._build_param_row(frame, card["params"], pname, kind, default)
        if card["operation"] == "normalize":
            self._build_normalize_readout(frame, card)

    def _build_color_method_row(self, frame, card, chain):
        """The `method` picker of a color card. Switching it swaps the card's
        parameter rows, so the card is rebuilt with only the new method's keys."""
        params = card["params"]
        if params.get("method") not in COLOR_METHODS:
            params["method"] = "luminance"
        row = ttk.Frame(frame); row.pack(fill="x", padx=6, pady=1)
        ttk.Label(row, text="method", width=16).pack(side="left")
        var = tk.StringVar(value=params["method"])
        combo = ttk.Combobox(row, textvariable=var, values=COLOR_METHODS, state="readonly", width=16)
        combo.pack(side="left")

        def on_method(_e=None, c=card, v=var, ch=chain):
            method = v.get()
            if method == c["params"].get("method"):
                return
            c["params"] = {"method": method}
            self._rebuild_filter_cards(ch)
            self._notify_profile_edit()
        combo.bind("<<ComboboxSelected>>", on_method)

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
        # Every commit reports the edit: the live preview repaints the chain's
        # output, debounced, so a chain is judged by looking at it rather than
        # by priming. _notify_profile_edit is cheap and idempotent.
        def _set(p, value):
            params[p] = value
            self._notify_profile_edit()

        if kind == "bool":
            var = tk.BooleanVar(value=bool(params[pname]))
            var.trace_add("write", lambda *_: _set(pname, var.get()))
            ttk.Checkbutton(row, variable=var).pack(side="left")
        elif kind.startswith("choice:"):
            choices = kind.split(":", 1)[1].split(",")
            var = tk.StringVar(value=str(params[pname]))
            var.trace_add("write", lambda *_: _set(pname, var.get()))
            ttk.Combobox(row, textvariable=var, values=choices, state="readonly",
                         width=12).pack(side="left")
        elif kind in ("str", "floats", "numstr"):
            var = tk.StringVar(value=str(params[pname]))
            var.trace_add("write", lambda *_: _set(pname, var.get()))
            ttk.Entry(row, textvariable=var, width=18 if kind == "floats" else 14).pack(side="left")
        elif kind in ("optfloat", "nullfloat"):
            # Blank means "not set". For optfloat it is dropped on export, so an
            # unset optional bound cannot be mistaken for a real 0.0; for
            # nullfloat it exports as an explicit null, because "keep every
            # pixel" is a real setting that must not fall back to the default.
            var = tk.StringVar(value=str(params[pname]))
            def commit_opt(*_, p=pname, v=var):
                text = v.get().strip()
                if not text:
                    _set(p, "")
                    return
                try:
                    _set(p, float(text))
                except ValueError:
                    pass
            var.trace_add("write", commit_opt)
            ttk.Entry(row, textvariable=var, width=10).pack(side="left")
        else:  # float | int
            var = tk.StringVar(value=str(params[pname]))
            def commit(*_, p=pname, k=kind, v=var):
                try:
                    _set(p, int(v.get()) if k == "int" else float(v.get()))
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
        self._notify_profile_edit()

    def _remove_filter_card(self, idx, chain="topo"):
        cards, _ = self._chain(chain)
        if 0 <= idx < len(cards):
            del cards[idx]
        if not cards or cards[-1]["operation"] != "none":
            cards.append(self._new_filter_card())
        self._set_chain_cards(chain, cards)
        self._rebuild_filter_cards(chain)
        self._notify_profile_edit()

    def _slice_msc_mark(self, si, li):
        """"Y" when the slice has a primed MSC under the current sequences."""
        try:
            p = self.primed[si]
            if (p["files"] == self.subsequences[si]["files"]
                    and li < len(p["pipes"])):
                return "Y"
        except (IndexError, KeyError, TypeError):
            pass
        return ""

    # ------------------------------------------------------------------ #
    # Image preview (no priming): click / drag over the TIFF list
    # ------------------------------------------------------------------ #
    _PREVIEW_CACHE_MAX = 4        # float32 slices (~40 MB each on 3232^2)

    def _bind_preview(self):
        # add="+" keeps Tk's extended-selectmode rubber-band working, so the
        # same drag that previews also selects files for "Make sequence".
        self.file_list.bind("<Button-1>", self._on_filelist_click, add="+")
        self.file_list.bind("<B1-Motion>", self._on_filelist_click, add="+")

    def _on_filelist_click(self, event):
        idx = self.file_list.nearest(event.y)
        if not (0 <= idx < len(self.all_files)):
            return
        # Debounce: dragging fires per motion event, but a ~40 MB TIFF load
        # should happen once per rest, not per pixel.
        if self._preview_after is not None:
            try:
                self.root.after_cancel(self._preview_after)
            except tk.TclError:
                pass
        self._preview_after = self.root.after(40, self._preview_apply, idx)

    def _preview_apply(self, idx):
        self._preview_after = None
        if self.viewer is None or not (0 <= idx < len(self.all_files)):
            return
        self._preview_file(self.all_files[idx])

    def _preview_array(self, path):
        """The raw slice at `path` from the LRU, reading it when absent; None
        when it cannot be read (the caller falls back to the pyramid).

        Shared with the live preview, which needs the same pixels and must not
        re-read a ~40 MB file because a sigma moved."""
        if not path:
            return None
        arr = self._preview_cache.get(path)
        if arr is not None:
            self._preview_cache[path] = self._preview_cache.pop(path)   # LRU
            return arr
        try:
            from .common import load_slice
            arr = load_slice(path, self._color_alpha(), log=log)
        except Exception as exc:
            log(f"preview load failed for {os.path.basename(path)}: {exc}")
            return None
        self._preview_cache[path] = arr
        while len(self._preview_cache) > self._PREVIEW_CACHE_MAX:
            self._preview_cache.pop(next(iter(self._preview_cache)))
        return arr

    def _preview_file(self, path):
        """Show one TIFF without priming: the Image dropdown's channel of it."""
        if self.viewer is None:
            return
        arr = self._preview_array(path)
        if arr is None:
            # Fall back to the pyramidal path-only source (large_image);
            # reset_array drops any stale in-memory base first.
            first = not self.viewer.has_base
            self.viewer.set_base(array=None, path=path, reset_array=True)
            self._preview_path = path
            self._preview_shown_key = None
            self.viewer.set_overlays([])
            self.viewer.set_window(*self._window_for("base"))
            self.viewer.fit() if first else self.viewer.render()
            self.status_var.set(f"preview: {os.path.basename(path)}")
            return
        first = not self.viewer.has_base
        self._preview_path = path
        if self._current_color_count() != getattr(self, "_picker_color_count", 0):
            self._refresh_channel_picker()
        self._render_preview(first)

    def _color_alpha(self):
        """The alpha policy colour files load under (`input.color.alpha`)."""
        try:
            return self.color_alpha_var.get() or "drop"
        except tk.TclError:
            return "drop"

    def _default_color_method(self):
        """The conversion a chain without a leading `color` stage gets."""
        try:
            return self.color_default_var.get() or "luminance"
        except tk.TclError:
            return "luminance"

    def _on_color_input_change(self, reload=False):
        """The colour input block changed. The default method is part of every
        preview key already; a new alpha policy needs the files read again."""
        if reload:
            self._preview_cache.clear()
            self._preview_chan_cache.clear()
        self._refresh_stat_summary()
        self._repreview_if_active()

    def _render_preview(self, first=False):
        """Paint the active preview in the Image dropdown's channel."""
        path = self._preview_path
        arr = self._preview_cache.get(path)
        if arr is None:
            # Path-only (pyramidal) preview: no array to compute channels on.
            self.viewer.set_window(*self._window_for("base"))
            self.viewer.fit() if first else self.viewer.render()
            return
        channel = self.background_var.get()
        raster = self._preview_channel(arr, path, channel)
        # The path doubles as the pyramidal source only for the raw base.
        self.viewer.set_base(array=raster, path=path if raster is arr else None)
        self.viewer.set_overlays([])
        # The raw slice stands in for a channel it cannot build: window it as
        # the original, not under the missing channel's name.
        shown = channel if raster is not arr else self._original_channel()
        self.viewer.set_window(*self._window_for(shown))
        if first:
            self.viewer.fit()
        else:
            self.viewer.render()
        # Replaces any "computing …" line _preview_channel left behind.
        shown = "" if raster is arr else f"  [{channel}]"
        self.status_var.set(f"preview: {os.path.basename(path)}{shown}")

    # ------------------------------------------------------------------ #
    # Preview channels: what the chain produces, before anything is primed
    # ------------------------------------------------------------------ #
    def _preview_plan(self, arr, path, channel, params=None):
        """How to compute `channel` on preview slice `path`, or None when
        there is nothing to compute (the raw slice already IS the channel, or
        the name is not one of the spec's).

        The plan carries the CACHE KEY, and that key is also the dependency
        set: "base" is keyed on `base_filters`, "filtered" on `filters`, a
        derived channel on its own one-channel spec -- and on `filters` too
        when the card says it is measured on the filtered field. So "did
        anything this channel depends on change?" is answered by comparing
        keys; there is no second dependency table to drift out of step.
        """
        if params is None:
            params = self._params_json()
        doc = json.loads(params)
        base_filters = doc.get("base_filters") or []
        filters = doc.get("filters") or []
        default_method = self._default_color_method()
        planar = getattr(arr, "ndim", 2) == 3
        label = os.path.basename(path or "")
        keys = {"base": (path, "base", json.dumps({"chain": base_filters,
                                                   "default": default_method},
                                                  sort_keys=True)),
                "filtered": (path, "filtered", json.dumps({"chain": filters,
                                                           "default": default_method},
                                                          sort_keys=True))}
        plan = {"path": path, "keys": keys, "spec_key": None, "planar": planar,
                "base_filters": base_filters, "filters": filters}
        if channel in ("", "base"):
            # A colour slice always goes through its leading colour stage
            # (explicit or the default), so an empty chain still converts.
            if not base_filters and not planar:
                return None
            plan["key"] = keys["base"]
            plan["job"] = preview_job("base", base_filters, filters, default_method,
                                      planar=planar, label=label)
            return plan
        if channel == "filtered":
            if not filters and not planar:
                return None
            plan["key"] = keys["filtered"]
            plan["job"] = preview_job("filtered", base_filters, filters, default_method,
                                      planar=planar, label=label)
            return plan
        try:
            from msseg import mscoupon as engine
        except Exception:
            return None
        if not hasattr(engine, "stat_channel_images"):
            return None
        single = single_channel_params(params, channel)
        if single is None:
            return None
        card = ((json.loads(single).get("statistics") or {}).get("channels") or [{}])[0]
        source = card.get("source") or "base"
        # The derived channel depends on the base chain and on its own spec,
        # and on NOTHING else: a statistics source is base or colour (the core
        # refuses "filtered"), so the topology chain cannot move it. Which is
        # why editing `filters` while a derived channel is shown is free.
        spec = {"base": base_filters, "stat": json.loads(single).get("statistics")}
        plan["spec_key"] = json.dumps(spec, sort_keys=True)
        plan["key"] = (path, channel, plan["spec_key"])
        plan["job"] = preview_job("derived", base_filters, filters, default_method,
                                  single=single, source=source, planar=planar,
                                  label=label)
        return plan

    def _preview_have(self, arr, plan):
        """Hand the job whatever it needs and the cache already holds. A
        derived channel is measured on the base and filtered rasters, and
        both are usually one channel switch old."""
        job = plan["job"]
        if job["kind"] != "derived":
            return
        for name, chain in (("base", plan["base_filters"]),
                            ("filtered", plan["filters"])):
            if not chain and not plan["planar"]:
                job["have"][name] = arr            # the chain is the identity
                continue
            hit = self._preview_chan_cache.get(plan["keys"][name])
            if hit is not None:
                job["have"][name] = hit

    def _preview_store(self, plan, out):
        """File every raster a compute produced and return the one asked for.

        Called on the Tk thread only, whether the compute ran here or on the
        worker -- _RasterCache is a bare OrderedDict whose reads reorder it,
        so it must never be touched from two threads."""
        cache = self._preview_chan_cache
        for name in ("base", "filtered"):
            if out.get(name) is not None:
                cache.put(plan["keys"][name], out[name])
        for name, raster in out.get("channels") or []:
            cache.put((plan["path"], name, plan["spec_key"]), raster)
        return cache.get(plan["key"])

    def _preview_channel(self, arr, path, channel):
        """The named channel of a preview slice, computed on the spot.

        "base" is the raw slice through the base chain (the raw array itself
        when that chain is empty), "filtered" is the topology chain's output,
        and a derived name is the scale-space response the statistics spec
        defines -- each through the SAME calls a run makes (engine.filter_slice
        / _apply_base_chain / stat_channel_images), so what is previewed is
        what priming will measure. Any failure (no extension, unknown name)
        falls back to the raw slice.

        Memoised per path under a byte budget, on exactly the parameters that
        produce each raster (see _preview_plan). Keying everything on the
        whole params JSON made a sigma edit re-run the base chain's GMM (~5 s
        at 3232^2), and computing the spec's whole bank for one plane charged
        every other channel's sigma to each switch. Every step is timed to the
        log.

        This is the SYNCHRONOUS path, kept for the callers that need a raster
        in hand (the histogram range measurement, the selftests). A parameter
        edit goes through _launch_preview instead, which runs the same compute
        on a worker thread.
        """
        try:
            import numpy as np                      # noqa: F401
            from msseg import mscoupon as engine
        except Exception:
            return arr
        try:
            if channel == "color" or channel.startswith("color_c"):
                planar = getattr(arr, "ndim", 2) == 3
                return self._color_plane(arr if planar else None, channel, arr)
            plan = self._preview_plan(arr, path, channel)
            if plan is None:
                return arr
            hit = self._preview_chan_cache.get(plan["key"])
            if hit is not None:
                return hit
            self._preview_have(arr, plan)
            got = self._preview_store(plan, preview_raster(engine, arr, plan["job"], log))
            return arr if got is None else got
        except Exception as exc:
            log(f"preview channel '{channel}' unavailable: {exc}")
            return arr

    def _repreview_if_active(self):
        """Repaint an active preview. The window sliders, the Image dropdown
        and the parameter panels all call _refresh_render, which early-returns
        while nothing is primed -- this keeps them live for previews too."""
        if self.viewer is not None and self._preview_path is not None:
            self._render_preview()

    # ------------------------------------------------------------------ #
    # The live preview: a parameter edit repaints the chain output
    # ------------------------------------------------------------------ #
    def _live_preview_slice(self):
        """(path, raw array) the live preview computes on, or (None, None).

        Before any Run that is the slice being previewed from the file list.
        Once a stack is primed it is the raw file behind the slice on screen:
        the primed rasters were built by the chain as it WAS, so a preview of
        the chain as it is now has to start from the pixels again. Read
        through the same small LRU, so re-editing a sigma does not re-read the
        file."""
        path = self._preview_path
        if path is None and self.primed:
            cur = self._current()
            if cur is not None:
                try:
                    path = self.primed[cur[0]]["files"][cur[1]]
                except (IndexError, KeyError, TypeError):
                    path = None
        if path is None:
            return None, None
        return path, self._preview_array(path)

    def _launch_preview(self):
        """A chain parameter settled: repaint the shown channel, off-thread.

        Returns the token submitted, or None when there was nothing to do --
        which is the common case, because a channel cache key IS its
        dependency set (_preview_plan), so editing the topology chain while
        the dropdown shows `base` costs one tuple comparison."""
        if self.viewer is None:
            return None
        path, arr = self._live_preview_slice()
        if arr is None:
            self._preview_shown_key = None   # no slice, or a pyramid-only one
            return None
        channel = self.background_var.get()
        plan = self._preview_plan(arr, path, channel)
        if plan is None:
            # The raw slice already IS this channel (an empty chain, the
            # colour planes): repaint from it and drop any override.
            self._preview_shown_key = None
            self._clear_preview_override()
            self._paint_live(path, channel, self._preview_channel(arr, path, channel))
            return None
        if plan["key"] == self._preview_shown_key:
            return None                      # this channel did not move
        hit = self._preview_chan_cache.get(plan["key"])
        if hit is not None:                  # retyping the old sigma: instant
            self._preview_shown_key = plan["key"]
            self._paint_live(path, channel, hit)
            return None
        try:
            from msseg import mscoupon as ext
        except Exception:
            return None
        self._preview_have(arr, plan)
        self._preview_token += 1
        token, job = self._preview_token, plan["job"]
        self._preview_pending = (path, channel, plan)
        self.viewer.set_hud("busy", f"Previewing {channel}")
        self.status_var.set(f"preview: computing {channel} on "
                            f"{os.path.basename(path)}...")
        self._worker().submit(token,
                              lambda stop: preview_raster(ext, arr, job, log, stop),
                              sync=self._preview_sync)
        return token

    def _worker(self):
        if self._preview_worker is None:
            from msseg.labeler.preview import PreviewWorker
            self._preview_worker = PreviewWorker(
                self.root, self._on_preview_result, on_error=self._on_preview_error,
                log=log, pump_ms=_PREVIEW_PUMP_MS, name="mscoupon-preview")
        return self._preview_worker

    def _on_preview_result(self, token, out):
        """A chain recompute landed. Everything that touches a cache or a
        widget happens HERE, on the Tk thread."""
        if token != self._preview_token or self._preview_pending is None:
            return
        path, channel, plan = self._preview_pending
        self._preview_pending = None
        raster = self._preview_store(plan, out or {})
        if raster is None:                   # stopped, or the channel vanished
            self._update_busy()
            return
        self._preview_shown_key = plan["key"]
        self._paint_live(path, channel, raster)

    def _on_preview_error(self, token, msg):
        if token == self._preview_token:
            self._preview_pending = None
        log(f"preview failed: {msg}")
        self.status_var.set(f"preview failed: {msg}")
        self._update_busy()                  # the canvas keeps the last raster

    def _paint_live(self, path, channel, raster):
        """Put a live raster on the canvas, keeping zoom and pan.

        `set_base` never touches the viewport -- only fit() and center_on do --
        and the path key is held constant, so the pyramid behind the slice is
        not dropped and reopened per keystroke. While a stack is primed this
        also records an override: the region overlays came from the chain as
        it was, and boundaries drawn over a differently filtered field are not
        slightly stale but simply wrong, so they come off."""
        if self.viewer is None or raster is None:
            return
        # `_primed_chain` is None until a prime reports one; "unknown" must not
        # read as "stale", or a faked stack would hide its own overlays.
        if (self.primed and self._primed_chain is not None
                and self._chain_fingerprint() != self._primed_chain):
            self._preview_override = (path, channel, raster)
        self.viewer.set_base(array=raster, path=None)
        self.viewer.set_overlays([])
        self.viewer.set_window(*self._window_for(channel))
        self.viewer.render()
        self._update_busy()
        self.status_var.set(f"preview: {os.path.basename(path or '')}  [{channel}]")

    def _clear_preview_override(self):
        """Back to what was primed: the overlays mean something again."""
        if self._preview_override is not None:
            self._preview_override = None
            self._update_busy()

    def _build_segmentation_controls(self, chan):
        """The segmentation-source radios and the mask toggle, after the
        Image dropdown."""
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
        self._build_slice_nav(row)

        # One slider pair for the channel on screen; every channel keeps its
        # own window (ViewerShell._window_for), so switching channels swaps
        # the pair rather than sharing it.
        row = ttk.Frame(live); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Image Min/Max:", width=14).pack(side="left")
        self._scale(row, from_=0.0, to=1.0, variable=self.vmin_var, orient="horizontal",
                    command=self._on_window_change).pack(side="left", fill="x", expand=True)
        self._scale(row, from_=0.0, to=1.0, variable=self.vmax_var, orient="horizontal",
                    command=self._on_window_change).pack(side="left", fill="x", expand=True)
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
        """The priming params for the ACTIVE profile as edited right now, plus
        the session-level core count (cores > 1 selects MSCEER's partitioned
        builder -- see session.profile_params_json). Drives which per-feature
        fields exist, so the query dropdown, the primed pipelines and a saved
        profile all agree."""
        if cores is None:
            cores = self._cores_per_slice()
        return session.profile_params_json(self._profile_from_ui(), cores,
                                           self._current_color_count() or None)

    # ------------------------------------------------------------------ #
    # Compute profiles: the left panel edits the ACTIVE one
    # ------------------------------------------------------------------ #
    def _profile_from_ui(self):
        """Snapshot the panel into a profile dict (JSON/writer-shaped)."""
        try:
            pct = float(self.persist_pct_var.get())
        except (ValueError, tk.TclError):
            pct = 10.0
        radius = self._ext_radius()
        name = "default"
        relevance = True
        if 0 <= self.active_profile_idx < len(self.profiles):
            name = self.profiles[self.active_profile_idx].get("name", name)
            relevance = config_io.statistics_from_json(
                self.profiles[self.active_profile_idx].get("statistics"))["relevance"]
        return {
            "name": name,
            # Every key `color_input_from_json` normalizes to has to be here,
            # or the first load adds it and a profile file round trip is lossy.
            "input": {"color": {"alpha": self._color_alpha(),
                                "default_method": self._default_color_method(),
                                "channels": max(0, int(self.color_channels_var.get())),
                                "reduce_at": self._reduce_at()}},
            "filters": config_io.filters_to_json(self.filter_cards),
            "base_filters": config_io.filters_to_json(self.base_cards),
            "msc": {"manifold": self.manifold_var.get(),
                    "persistence_percent": pct,
                    "accurate": bool(self.accurate_var.get()),
                    "extremum_sample_radius": radius,
                    "use_gpu_gradient": bool(self.gpu_var.get()),
                    "simplification": self.simplification_var.get()},
            "statistics": config_io.statistics_to_json(
                self._stat_channel_cards(), self._stat_reductions(),
                self.stat_extremum_var.get(), radius, relevance,
                self._stat_histogram(), self._stat_sources()),
            "selection": {
                "feature_filters": config_io.queries_to_json(self.query_cards),
                "pixel_filters": config_io.pixel_filters_to_json(self.pixel_cards),
                "connectivity": int(self.connectivity_var.get()),
                "min_area": self._min_area(),
            },
        }

    def _apply_profile_to_ui(self, profile, setvar, notes):
        """Push one profile onto the panel. The ORDER mirrors the old
        _apply_state and is load-bearing: scalars, then the statistics spec
        (the query dropdowns are generated from the field universe it defines),
        then the card chains -- topo before base, because the base rebuild is
        what resets _normalize_readouts."""
        col = session.color_input_from_json(profile.get("input"))
        setvar(self.color_alpha_var, col["alpha"])
        setvar(self.color_default_var, col["default_method"])
        setvar(self.color_channels_var, int(col["channels"]))
        msc = profile.get("msc") or {}
        if msc.get("manifold"):
            setvar(self.manifold_var, msc["manifold"])
        setvar(self.accurate_var, bool(msc.get("accurate")))
        setvar(self.gpu_var, bool(msc.get("use_gpu_gradient")))
        setvar(self.simplification_var,
               str(msc.get("simplification") or session.DEFAULT_SIMPLIFICATION))
        setvar(self.ext_radius_var, str(int(msc.get("extremum_sample_radius") or 0)))
        sel = profile.get("selection") or {}
        min_area = sel.get("min_area")
        setvar(self.min_area_var, "" if min_area is None else str(int(min_area)))
        if sel.get("connectivity"):
            setvar(self.connectivity_var, int(sel["connectivity"]))
        # The cap and the live value are independent vars and _persist_pct
        # clamps live against the cap, so both follow the profile (a session's
        # view block may override the live value afterwards).
        pct = msc.get("persistence_percent")
        if pct is not None:
            setvar(self.persist_pct_var, f"{float(pct):g}")
            setvar(self.persist_live_var, f"{float(pct):g}")

        stats = config_io.statistics_from_json(profile.get("statistics"), notes)
        self._apply_stat_state({"stat_channels": stats["channels"],
                                "stat_reductions": stats["reductions"],
                                "stat_extremum": stats["extremum"],
                                "stat_histogram": stats.get("histogram")}, setvar)

        fields = config_io.query_fields(
            json.dumps({"statistics": profile.get("statistics") or {}}))
        self.filter_cards = (config_io.filters_from_json(profile.get("filters"), notes)
                             + [self._new_filter_card()])
        self.base_cards = (config_io.filters_from_json(profile.get("base_filters"), notes)
                           + [self._new_filter_card()])
        self.query_cards = (config_io.queries_from_json(sel.get("feature_filters"),
                                                        fields, notes)
                            + [self._new_query_card()])
        self.pixel_cards = (config_io.pixel_filters_from_json(sel.get("pixel_filters"),
                                                              notes)
                            + [self._new_pixel_card()])
        self._rebuild_filter_cards()
        self._rebuild_filter_cards("base")
        self._rebuild_query_cards()
        self._rebuild_pixel_cards()
        self._refresh_channel_picker()
        self._refresh_stat_summary()

    def _reduce_at(self):
        """Where the conversion goes when the chain does not reduce itself:
        `front` (the head, as always) or `end` (append it, so the chain lifts
        through and an RGB intermediate exists). Carried from the active
        profile -- there is no control for it yet."""
        try:
            prof = self.profiles[self.active_profile_idx]
        except (AttributeError, IndexError):
            return "front"
        col = ((prof.get("input") or {}).get("color") or {})
        return "end" if col.get("reduce_at") == "end" else "front"

    def _stat_sources(self):
        """The active profile's `statistics.sources`, carried through unchanged.

        There is no editor for them yet: a source is a filter chain a config or
        another tool declares. But the statistics block is rebuilt from UI state
        on every export, so without this a profile that declares a source would
        lose it the first time the GUI wrote the profile back -- and its channels
        would then stop resolving. Preserving what a profile brought is the whole
        contract until there is a panel."""
        try:
            prof = self.profiles[self.active_profile_idx]
        except (AttributeError, IndexError):
            return None
        stats = prof.get("statistics")
        if not isinstance(stats, dict):
            return None
        src = stats.get("sources")
        return src if isinstance(src, dict) and src else None

    def _stat_channel_cards(self):
        """The `statistics.channels[]` model, in slot order."""
        cards = []
        if self.stat_base_var.get():
            cards.append({"kind": "base"})
        if self.stat_filtered_var.get():
            cards.append({"kind": "filtered"})
        if self.stat_color_var.get():
            cards.append({"kind": "color"})
        for kind, (on, sigmas, source) in self.stat_kind_vars.items():
            if not on.get():
                continue
            values = _parse_sigmas(sigmas.get())
            if values:
                card = {"kind": kind, "sigmas": values}
                if kind in config_io.COLOR_ONLY_KINDS or source.get() == "color":
                    card["source"] = "color"
                cards.append(card)
        return cards

    def _stat_reductions(self):
        return [r for r, v in self.stat_reduction_vars.items() if v.get()]

    def _hist_channel_names(self):
        return [c.strip() for c in self.hist_channels_var.get().replace(";", ",").split(",")
                if c.strip()]

    def _stat_histogram(self):
        """The `statistics.histogram` model: {bins, channels, ranges} or None."""
        if not self.hist_on_var.get():
            return None
        try:
            bins = int(float(self.hist_bins_var.get()))
        except (ValueError, tk.TclError):
            bins = 16
        return {"bins": bins, "channels": self._hist_channel_names(),
                "ranges": parse_hist_ranges(self.hist_ranges_var.get())}

    def _measure_hist_ranges(self):
        """Fill the "*" range from the slice on screen: the min/max over every
        histogrammed channel's raster, so the bins cover what the data spans
        (the C++ clamps anything outside into the end bins)."""
        try:
            import numpy as np
        except Exception:
            return
        lo, hi = np.inf, -np.inf
        cur = self._current() if self.primed else None
        path = getattr(self, "_preview_path", None)
        for name in self._hist_channel_names() or ["base"]:
            raster = None
            if cur is not None:
                raster = self._channel_raster(cur[0], cur[1], name, np)
            elif path in self._preview_cache:
                raster = self._preview_channel(self._preview_cache[path], path, name)
            if raster is None:
                continue
            r = np.asarray(raster, dtype=np.float64)
            r = r[np.isfinite(r)]
            if r.size:
                lo, hi = min(lo, float(r.min())), max(hi, float(r.max()))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            self.status_var.set("histogram ranges: nothing on screen to measure")
            return
        self.hist_ranges_var.set(f"*: {lo:.6g}, {hi:.6g}")
        self._on_stat_spec_change()

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
        params = self._params_json()
        subseqs = [dict(s) for s in self.subsequences]
        # Incremental priming: a sequence already primed under the SAME compute
        # parameters (cores and the GPU-gradient flag excluded -- neither
        # changes results, the GPU pairing is bit-identical) and the same files
        # is reused instead of recomputed, so adding a sequence never re-primes
        # the rest. Any parameter/statistics change misses the fingerprint and
        # re-primes everything.
        fp_doc = json.loads(self._params_json(cores=1))
        fp_doc.get("msc", {}).pop("use_gpu_gradient", None)
        fingerprint = json.dumps(fp_doc, sort_keys=True)
        reused = 0
        if fingerprint == getattr(self, "_primed_fingerprint", None) and self.primed:
            by_files = {tuple(p["files"]): p for p in self.primed}
            for s in subseqs:
                hit = by_files.get(tuple(s["files"]))
                if hit is not None:
                    s["_reuse"] = hit
                    reused += 1
        self._pending_fingerprint = fingerprint
        need = len(subseqs) - reused
        self.status_var.set(f"Priming… ({reused} of {len(subseqs)} sequence(s) "
                            f"reused)" if reused else "Priming…")
        log(f"run: {need} sequence(s) to prime, {reused} reused")
        # The concurrency numbers ride along for the worker's log line, so it
        # never reads Tk state off the UI thread.
        self.engine.start_run(subseqs, params,
                              {"cores_per_slice": self._cores_per_slice(),
                               "concurrent_slices": self._concurrent_slices()})
        self._ensure_pump()

    def _handle_compute_event(self, ev):
        """The coupon engine's own events: a finished priming run and a
        finished assembly (the shell handles progress and errors)."""
        kind = ev[0]
        if kind == "primed":
            self.run_btn.config(state="normal")
            self._set_load_enabled(True)
            self._chan_cache.clear()      # derived rasters belong to the old run
            # This priming's parameters are now the reuse fingerprint; its
            # chains are what a live preview compares itself against, and the
            # primed rasters are the truth again, so any override goes.
            self._primed_fingerprint = getattr(self, "_pending_fingerprint", None)
            self._primed_chain = self._chain_fingerprint()
            self._preview_override = None
            self._preview_shown_key = None
            # The stack's own base/filtered rasters are resident now, so the
            # preview cache gives up most of its budget.
            self._preview_chan_cache.budget = _PREVIEW_CHAN_BUDGET_PRIMED
            self._rebuild_flat_slices()
            self._refresh_subseq_list()   # "msc" column follows the primed set
            self.status_var.set(f"Primed {len(self.primed)} subsequence(s), "
                                f"{len(self.flat_slices)} slices.")
            cur = self._current()
            if cur is not None:
                self._request_assembly(cur[0])   # off-thread; UI stays responsive
            self._refresh_render()
        elif kind == "assembly_done":
            self._on_assembly_done(ev[1], ev[2])

    def _slice_nav_text(self, si, li):
        s = self.subsequences[si] if si < len(self.subsequences) else None
        folder = s.get("folder", "?") if s else "?"
        try:
            base = os.path.basename(self.primed[si]["files"][li])
        except (IndexError, KeyError, TypeError):
            base = f"[{li}]"
        return f"{folder}/{base}"

    def _on_seg_source_change(self):
        """Switching the overlay (or the mask) can demand a tier that has not been
        computed -- selecting `global CC` is what actually triggers the 3D
        assembly. Request it, then repaint with whatever is available now."""
        cur = self._current()
        if cur is not None:
            self._request_assembly(cur[0])
        self._refresh_render()

    def _on_slice_change(self, _value=None):
        """Compat shim for older callers: navigate by flat index (or refresh
        the current slice when called without one)."""
        if _value is not None:
            try:
                self._goto_slice(int(round(float(_value))))
                return
            except (ValueError, tk.TclError):
                return
        idx = int(round(float(self.slice_var.get())))
        self._goto_slice(idx)

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
        """Drive the canvas HUD, most urgent first: an animated spinner while
        an assembly or a chain preview is in flight, then the two stale
        badges -- a live preview of a chain the primed stack was not built
        with (Run), a selection edit not yet re-assembled (Rerun) -- else
        nothing."""
        if self.viewer is None:
            return
        if self._is_current_busy():
            self.viewer.set_hud("busy", "Recomputing")
        elif self._preview_pending is not None:
            self.viewer.set_hud("busy", f"Previewing {self._preview_pending[1]}")
        elif self._preview_override is not None:
            self.viewer.set_hud("stale", "Preview - filters changed, Run to re-prime")
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
    def _original_channel(self):
        """What F flips back to: the colour planes when the slice on screen
        has them, else the base channel."""
        return "color" if self._current_color_count() > 0 else "base"

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
            overlays.append(self._region_overlay(
                raster, _id_lut(raster, min_colors, np), np))
        if self.mask_var.get() and data is not None:
            glob = data["global_labels"][li]                 # -1 bg, >=0 = kept feature
            K = int(glob.max()) + 1 if glob.size else 1
            mlut = np.zeros((max(K, 1), 4), np.uint8)
            mlut[:, 0] = 255; mlut[:, 1] = 255; mlut[:, 3] = 255   # yellow where global>=0
            overlays.append(self._region_overlay(glob, mlut, np))
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
            self._repreview_if_active()   # window sliders stay live for previews
            return
        if self._preview_override is not None:
            # A chain edit since the prime: the live raster is what is true,
            # and the primed overlays are not about this field at all.
            self._paint_live(*self._preview_override)
            return
        try:
            import numpy as np
            from msseg.viz import min_colors
        except Exception:
            return
        si, li = cur
        p = self.primed[si]
        if self._current_color_count() != getattr(self, "_picker_color_count", 0):
            self._refresh_channel_picker()
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

        first = not self.viewer.has_base
        channel = self.background_var.get()
        if channel == "filtered":
            array, path, shown = filt, None, "filtered"
        elif channel in ("", "base"):
            array, path, shown = base, p["files"][li], "base"
        else:
            # A derived scale-space channel (or the colour planes).
            raster = self._channel_raster(si, li, channel, np)
            if raster is None:
                # The channel is not available for this slice (nothing primed yet,
                # or an extension that cannot build it). Show the base rather than
                # blanking the canvas -- windowed as the base, since that is what
                # is on screen.
                array, path, shown = base, p["files"][li], "base"
            else:
                array, path, shown = raster, None, channel
        self.viewer.set_base(array=array, path=path)
        # After set_base: a channel seen for the first time takes its window
        # from the source now on the canvas.
        self.viewer.set_window(*self._window_for(shown))
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
    def _config_for(self, files, output_folder, profile=None, folder=None):
        """The AppConfig dict for one file list under one profile (the active
        one by default).

        DORMANT UI-wise: config.json export is on hold until it returns as a
        bundle of the compute profile AND the trained model. Kept because it is
        the seed of that export and the selftest's AppConfig-equivalence probe.
        """
        if profile is None:
            profile = self._profile_from_ui()
        msc = profile.get("msc") or {}
        sel = profile.get("selection") or {}
        stats = config_io.statistics_from_json(profile.get("statistics"))
        return config_io.build_config(
            files=files,
            output_folder=output_folder,
            filters=profile.get("filters") or [],
            base_filters=profile.get("base_filters") or [],
            persistence_percent=float(msc.get("persistence_percent") or 10.0),
            manifold=str(msc.get("manifold") or "ascending"),
            accurate=bool(msc.get("accurate")),
            extremum_sample_radius=int(msc.get("extremum_sample_radius") or 0),
            min_area=sel.get("min_area"),
            feature_filters=sel.get("feature_filters") or [],
            pixel_filters=sel.get("pixel_filters") or [],
            connectivity=int(sel.get("connectivity") or 6),
            cores_per_slice=self._cores_per_slice(),
            concurrent_slices=self._concurrent_slices(),
            stat_channels=stats["channels"],
            stat_reductions=stats["reductions"],
            stat_extremum=stats["extremum"],
            stat_relevance=stats["relevance"],
            folder=folder,
        )

    def _write_configs(self, out_dir):
        """Write one config.json per sequence; returns the written paths.
        (Dormant, see _config_for.)"""
        paths = []
        for i, s in enumerate(self.subsequences):
            cfg = self._config_for(s["files"], os.path.join(out_dir, f"out_{i}"))
            path = os.path.join(out_dir, f"config_{i}.json")
            config_io.dump_config(cfg, path)
            paths.append(path)
        return paths

    def _view_state(self):
        view = super()._view_state()
        view.update({
            "persist_live": self.persist_live_var.get(),
            "seg_source": self.seg_source_var.get(),
            "mask": bool(self.mask_var.get()),
        })
        return view

    # ------------------------------------------------------------------ #
    # ViewerShell hooks: the coupon compute engine behind the generic shell
    # ------------------------------------------------------------------ #
    def _reset_compute(self):
        """Drop the primed data and everything derived from it (a profile
        switch or a session load replaces the parameters that produced it)."""
        self.engine.reset()
        self._hover_ctx = None
        self._selection_dirty = False
        # Nothing is primed, so there is nothing for a preview to be stale
        # against, and the cache gets its full budget back.
        self._primed_chain = None
        self._preview_override = None
        self._preview_shown_key = None
        self._preview_chan_cache.budget = _PREVIEW_CHAN_BUDGET

    def _settle_controls(self):
        try:
            self.rerun_btn.config(state="disabled")
            if not self._run_active:
                self.run_btn.config(state="normal")
        except tk.TclError:
            pass

    def _enumerate_items(self):
        """(si, li) for every primed slice, in stack order."""
        for si, p in enumerate(self.primed):
            for li in range(len(p["pipes"])):
                yield (si, li)

    # -- removing rows: the primed data goes with them --------------------- #
    # The primed list is positional and parallel to the sequences, so a
    # removed sequence or slice is dropped from it at the same index -- when
    # that entry really is this sequence's (the files match; after an old
    # remove-without-drop they might not, and then the stale entry is left
    # for the next Run's reuse check to sort out).
    def _remove_item_at(self, si, li):
        if self._slice_msc_mark(si, li) == "Y":
            self.engine.drop_slice(si, li)
        super()._remove_item_at(si, li)
        self._forget_derived()

    def _remove_sequence_at(self, si):
        try:
            aligned = self.primed[si]["files"] == self.subsequences[si]["files"]
        except (IndexError, KeyError, TypeError):
            aligned = False
        if aligned:
            self.engine.drop_sequence(si)
        super()._remove_sequence_at(si)
        self._forget_derived()

    def _forget_derived(self):
        """Caches keyed by (si, li) are meaningless once indices shift."""
        self._chan_cache.clear()
        self._hover_ctx = None

    def _run_settings(self):
        return {"cores_per_slice": self._cores_per_slice(),
                "concurrent_slices": self._concurrent_slices()}

    def _apply_run_settings(self, run, setvar, notes):
        if run.get("cores_per_slice"):
            setvar(self.cores_per_slice_var, int(run["cores_per_slice"]))
        if run.get("concurrent_slices"):
            setvar(self.concurrent_slices_var, int(run["concurrent_slices"]))

    def _apply_view_state(self, view, setvar, notes):
        if view.get("persist_live") is not None:
            setvar(self.persist_live_var, str(view["persist_live"]))
        # An older session's separate filtered pair becomes that channel's
        # window (only when it had been moved off the default).
        if not isinstance(view.get("windows"), dict):
            self._seed_window("filtered", view.get("vmin_filt"), view.get("vmax_filt"))
        if view.get("seg_source"):
            setvar(self.seg_source_var, str(view["seg_source"]))
        if view.get("mask") is not None:
            setvar(self.mask_var, bool(view["mask"]))

    def _session_doc_from_json(self, doc, notes):
        return session.session_doc_from_json(doc, notes)

    def _import_legacy_docs(self, docs, source):
        """v1 sessions and bare config.json exports go through the legacy
        converter (one v2 session out of many documents)."""
        notes = []
        doc = session.legacy_docs_to_session(docs, notes)
        self._apply_session_doc(doc, f"{source} (imported)", notes)

    def _profile_to_file_doc(self, profile):
        return session.profile_file_doc(profile)

    def _profile_from_file_doc(self, doc, notes):
        return session.profile_from_json(doc, notes)




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

    # Synthetic session state: one folder (nonexistent on purpose; every path
    # below must stay total), its TIFF list faked positionally.
    data_dir = r"C:\data"
    app.folders = [{"path": data_dir, "name": "data"}]
    app.active_folder_idx = 0
    app._refresh_folder_list()
    assert app.folder_list.size() == 1
    app.all_files = [os.path.join(data_dir, f"asdf_{i:04d}.tiff")
                     for i in [11, 12, 13, 25, 26]]
    app.file_list.delete(0, "end")
    for f in app.all_files:
        app.file_list.insert("end", os.path.basename(f))
    runs = group_contiguous([0, 1, 2, 3, 4])
    assert runs == [[0, 1, 2, 3, 4]], runs
    assert group_contiguous([0, 1, 2, 4, 5]) == [[0, 1, 2], [4, 5]]

    # Folder naming: collisions qualify with trailing path parts.
    assert session.folder_display_name(r"D:\other\data", ["data"]) == "other/data"

    # Two sequences from two contiguous selections; each is stamped with the
    # active folder and renders folder-qualified with its start-end span.
    app.file_list.selection_set(0, 2); app._make_subsequence()
    app.file_list.selection_clear(0, "end")
    app.file_list.selection_set(3, 4); app._make_subsequence()
    assert len(app.subsequences) == 2, app.subsequences
    assert [len(s["files"]) for s in app.subsequences] == [3, 2]
    assert all(s["folder"] == "data" for s in app.subsequences)
    assert app.subsequences[0]["name"] == "data asdf_0011-asdf_0013"
    assert app._sequence_row_text(app.subsequences[0]) == \
        "data  [asdf_0011 – asdf_0013] (3)"
    # The sequence view is a tree: TIFF children under each sequence, with
    # msc/annot columns (nothing primed and no annotations -> blank).
    assert app.subseq_list.item("q1", "text") == "data  [asdf_0025 – asdf_0026] (2)"
    assert len(app.subseq_list.get_children("q0")) == 3
    assert app.subseq_list.item("q0:0", "text") == "asdf_0011.tiff"
    assert tuple(app.subseq_list.item("q0:0", "values")) == ("", "")
    # Selecting a TIFF row of a sequence that is not primed previews it (the
    # file does not exist, so this exercises the path-only fallback).
    app._preview_path = None
    app.subseq_list.selection_set("q1:1")
    app._on_seq_tree_select()
    if app.viewer is not None:
        assert app._preview_path == app.subsequences[1]["files"][1]
    app.subseq_list.selection_remove("q1:1")

    # Preview plumbing (pure parts): a click schedules a debounced load; a
    # missing file falls back without raising and still records the preview.
    class _Ev:
        y = 5
    app._on_filelist_click(_Ev())
    if app._preview_after is not None:
        app.root.after_cancel(app._preview_after)
        app._preview_after = None
    app._preview_apply(0)               # C:\data doesn't exist -> fallback path
    if app.viewer is not None:
        assert app._preview_path == app.all_files[0]

    # Filter chain: choose blur -> a trailing none card is appended.
    app._on_filter_op_change(0, "blur")
    assert app.filter_cards[0]["operation"] == "blur"
    assert app.filter_cards[-1]["operation"] == "none"
    app.filter_cards[0]["params"]["sigma"] = 2.0

    # The synthesized colour conversion is DERIVED and must stay out of the
    # config. A blur-only chain on RGB plans a `color` head that is not one of
    # `filter_cards`, so filters_to_json cannot see it; pinning makes it a real
    # card, and only then does it export. This is the byte-compat gate the
    # design note names: an auto card that leaked here would change config bytes
    # for every colour workflow.
    exported = [f["operation"] for f in config_io.filters_to_json(app.filter_cards)]
    assert "color" not in exported, exported
    plan = config_io.chain_plan(app.filter_cards, 3, "luminance")
    head = plan["stages"][0]
    assert head["synthesized"] and head["operation"] == "color", plan
    assert (head["in"], head["out"]) == (3, 1)
    assert config_io.chain_plan(app.filter_cards, 1, "luminance")["stages"][0]["operation"] == "blur",         "one plane plans no conversion"
    app._pin_auto_color("topo", "mean")
    assert app.filter_cards[0]["operation"] == "color"
    assert app.filter_cards[0]["params"] == {"method": "mean"}
    pinned = config_io.filters_to_json(app.filter_cards)
    assert pinned[0]["operation"] == "color" and pinned[0]["params"]["method"] == "mean", pinned
    assert not config_io.chain_plan(app.filter_cards, 3, "luminance")["stages"][0]["synthesized"],         "a pinned conversion is the caller's own"
    app._remove_filter_card(0, "topo")   # back to the blur-only chain
    assert app.filter_cards[0]["operation"] == "blur", app.filter_cards

    # Channels on a PREVIEW (nothing primed): the topology chain and the
    # derived channels are computed on the spot from the raw array through
    # the run's own calls, memoised per (path, channel, params), and an
    # unknown name falls back to the raw slice.
    try:
        import numpy as np
        from msseg import mscoupon as _ext   # noqa: F401
        have_ext = True
    except Exception:
        have_ext = False
    if have_ext and app.viewer is not None:
        raw = np.random.default_rng(0).random((32, 32), dtype=np.float32)
        fake = os.path.join(data_dir, "fake.tiff")
        app._preview_cache[fake] = raw
        app._preview_path = fake
        assert app._preview_channel(raw, fake, "base") is raw, "empty base chain -> raw"
        filt = app._preview_channel(raw, fake, "filtered")
        assert filt.shape == raw.shape and not np.array_equal(filt, raw), "blur applied"
        assert app._preview_channel(raw, fake, "filtered") is filt, "memoised"
        app.stat_kind_vars["edges"][0].set(True)
        app._on_stat_spec_change()
        derived = [n for n in app._stat_channel_names() if n.startswith("edges_")]
        assert derived, app._stat_channel_names()
        assert derived[0] in app.background_combo.cget("values"), "picker offers it before a Run"
        e = app._preview_channel(raw, fake, derived[0])
        assert e.shape == raw.shape and e is not raw
        assert app._preview_channel(raw, fake, derived[0]) is e, "memoised"
        assert app._preview_channel(raw, fake, "no_such_channel") is raw
        # A derived request is cut down to a one-channel spec (the bank would
        # otherwise build every channel of the profile for one plane), and
        # the memo keys are the parameters that PRODUCE each raster: editing
        # another kind's sigmas must not invalidate the filtered field.
        one = json.loads(single_channel_params(app._params_json(), derived[0]))
        assert one["statistics"]["channels"] == [{"kind": "edges", "sigmas": [0.7]}], one
        assert single_channel_params(app._params_json(), "base") is None
        assert single_channel_params(app._params_json(), "nope") is None
        app.stat_kind_vars["blur"][0].set(True)
        app.stat_kind_vars["blur"][1].set("2, 4")
        app._on_stat_spec_change()
        assert app._preview_channel(raw, fake, "filtered") is filt, "unaffected by the spec"
        assert app._preview_channel(raw, fake, derived[0]) is e, "unaffected by another kind"
        h = json.loads(single_channel_params(
            app._params_json(), "blur_s4"))
        assert h["statistics"]["channels"] == [{"kind": "blur", "sigmas": [4.0]}], h
        app.stat_kind_vars["blur"][0].set(False)
        app._on_stat_spec_change()
        # The cache is byte-budgeted, not counted: the most recent entries
        # survive and the total stays under budget.
        small = _RasterCache(3 * raw.nbytes)
        for i in range(5):
            small.put(("k", i), raw.copy())
        assert len(small) == 3 and small.nbytes <= 3 * raw.nbytes
        assert small.get(("k", 4)) is not None and small.get(("k", 0)) is None
        app.background_var.set(derived[0])
        app._refresh_render()                  # the dropdown's path, unprimed
        assert app.viewer._base is not None and app.viewer._base.shape == raw.shape
        assert np.array_equal(app.viewer._base, e), "the preview shows the derived channel"
        # a channel seen for the first time takes its window from its raster
        assert app._window_channel == derived[0] and derived[0] in app._channel_windows
        lo_d, hi_d = app._channel_windows[derived[0]]
        assert 0.0 <= lo_d < hi_d <= 1.0 and float(app.vmin_var.get()) == lo_d
        app.background_var.set("filtered")
        app._refresh_render()
        app._preview_shown_key = None

        # --- the live preview: an edit repaints, without a Run ------------ #
        # Run the worker inline so the whole loop is one call, and watch the
        # token: it moves only when the SHOWN channel depends on what changed.
        app._preview_sync = True
        zoom_before = (app.viewer.scale, app.viewer.view_x, app.viewer.view_y)
        app.viewer.scale = zoom_before[0] * 0.5
        before = np.array(app.viewer._base, copy=True)
        app.filter_cards[0]["params"]["sigma"] = 6.0
        tok = app._launch_preview()
        assert tok == app._preview_token and tok > 0, "a filter edit recomputes"
        assert not np.array_equal(app.viewer._base, before), "and repaints"
        assert app.viewer.scale == zoom_before[0] * 0.5, "zoom survives an edit"
        assert (app.viewer.view_x, app.viewer.view_y) == zoom_before[1:], "so does pan"
        assert app._preview_pending is None and app.viewer._hud_mode is None
        # Nothing moved: no work, no repaint.
        assert app._launch_preview() is None and app._preview_token == tok
        # Retyping the old sigma comes back from the cache, still without a
        # submission.
        app.filter_cards[0]["params"]["sigma"] = 2.0
        assert app._launch_preview() is None, "a chain already computed is a cache hit"
        assert np.array_equal(app.viewer._base, before), "and the old raster returns"
        # The dependency gate: with `base` shown, the topology chain is free.
        app.background_var.set("base")
        app._launch_preview()
        tok = app._preview_token
        app.filter_cards[0]["params"]["sigma"] = 9.0
        assert app._launch_preview() is None and app._preview_token == tok, \
            "editing the topology chain while `base` is shown costs nothing"
        # ...but the base chain is not.
        app.base_cards[0]["operation"] = "blur"
        app.base_cards[0]["params"] = {"sigma": 1.0}
        assert app._launch_preview() == tok + 1, "the base chain is what `base` shows"
        app.base_cards[0]["operation"] = "none"
        app.base_cards[0]["params"] = {}
        # The signal itself: a card edit settles into exactly one launch.
        app.filter_cards[0]["params"]["sigma"] = 2.0
        app.background_var.set("filtered")
        app._launch_preview()
        tok = app._preview_token
        app._build_param_row(app.filters_frame, app.filter_cards[0]["params"],
                             "sigma", "float", 2.0)
        row = app.filters_frame.winfo_children()[-1]
        entry = [w for w in row.winfo_children() if isinstance(w, ttk.Entry)][0]
        for text in ("3", "3.", "3.5"):        # typing, keystroke by keystroke
            entry.delete(0, "end"); entry.insert(0, text)
        assert app.filter_cards[0]["params"]["sigma"] == 3.5, "the field committed"
        assert app._preview_edit_after is not None, "one settle timer, not three"
        assert app._preview_token == tok, "and nothing has run yet"
        app.root.after_cancel(app._preview_edit_after)
        app._preview_edit_after = None
        app._preview_edit_settled()
        assert app._preview_token == tok + 1, "the settle launches once"
        # A failing compute keeps the last raster and clears the spinner.
        shown = np.array(app.viewer._base, copy=True)
        app._preview_pending = (fake, "filtered", {})
        app._on_preview_error(app._preview_token, "boom")
        assert np.array_equal(app.viewer._base, shown) and app.viewer._hud_mode is None
        assert "boom" in app.status_var.get()
        app._preview_sync = False
        app.filter_cards[0]["params"]["sigma"] = 2.0
        app.background_var.set("base")
        app.stat_kind_vars["edges"][0].set(False)
        app._on_stat_spec_change()
        app._preview_cache.pop(fake, None)

        # A COLOUR preview: planar (C,h,w) planes. The picker offers the
        # composite and each plane; `base` goes through the default colour
        # method (luminance) or the chain's own `color` card; the statistics
        # panel can read the planes, and the params JSON declares their count.
        planes = np.stack([raw, 2 * raw, 0.5 * raw]).astype(np.float32)
        rgb = os.path.join(data_dir, "rgb.tiff")
        app._preview_cache[rgb] = planes
        app._preview_path = rgb
        app._refresh_channel_picker()
        values = list(app.background_combo.cget("values"))
        assert values[:4] == ["color", "color_c0", "color_c1", "color_c2"], values
        assert app.color_channels_var.get() == 3, "plane count follows the slice on screen"
        assert app._preview_channel(planes, rgb, "color") is planes
        assert np.array_equal(app._preview_channel(planes, rgb, "color_c1"), 2 * raw)
        lum = app._preview_channel(planes, rgb, "base")
        assert lum.shape == raw.shape and np.allclose(lum, (0.2126 + 2 * 0.7152 + 0.5 * 0.0722) * raw)
        app.base_cards = [{"operation": "color", "params": {"method": "pick", "channel": 2}},
                          app._new_filter_card()]
        app._rebuild_filter_cards("base")
        assert np.allclose(app._preview_channel(planes, rgb, "base"), 0.5 * raw), "a color card at index 0 wins"
        app.background_var.set("color")
        app._refresh_render()
        assert app.viewer.base_is_rgb and app.viewer._base.shape == (32, 32, 3), "the canvas shows RGB"

        # -- F flips the original (the planes, here) <-> the derived channel
        # last shown; every channel keeps its own brightness window ---------- #
        assert app._original_channel() == "color", "a colour slice's original is its planes"
        lo_c, hi_c = app._channel_windows["color"]
        assert 0.0 <= lo_c < hi_c <= 1.0 and app._window_channel == "color"
        app.background_combo.focus_set()
        app.background_var.set("color_c1"); app._on_image_channel_change()
        assert not app._typing(), "the dropdown hands the keyboard back"
        assert app._window_channel == "color_c1" and "color_c1" in app._channel_windows
        app.vmin_var.set(0.3); app._on_window_change()
        assert app._channel_windows["color_c1"][0] == 0.3, "the slider edits the channel on screen"
        app._on_swap_key()
        assert app.background_var.get() == "color" and app._swap_channel == "color_c1"
        assert (float(app.vmin_var.get()), float(app.vmax_var.get())) == (lo_c, hi_c),             "the original's own window is back on the sliders"
        app._on_swap_key()
        assert app.background_var.get() == "color_c1" and float(app.vmin_var.get()) == 0.3,             "a moved window is kept for its channel"
        assert "(F: color)" in app.status_var.get(), app.status_var.get()
        app._on_swap_key()
        assert app.background_var.get() == "color"
        # the windows ride the session; an older session's one pair seeds the
        # base (and the coupon's filtered pair its channel) only when moved
        vdoc = app._session_doc()["view"]
        assert vdoc["windows"]["color_c1"][0] == 0.3 and vdoc["swap_channel"] == "color_c1"
        notes = []
        app._apply_windows_view({"vmin": 0.2, "vmax": 0.8}, notes)
        assert app._channel_windows == {"base": (0.2, 0.8)} and app._swap_channel is None
        app._apply_view_state({"vmin_filt": 0.1, "vmax_filt": 0.9}, lambda v, x: v.set(x), notes)
        assert app._channel_windows["filtered"] == (0.1, 0.9)
        app._apply_windows_view({"vmin": 0.0, "vmax": 1.0}, notes)
        assert app._channel_windows == {}, "the default pair seeds nothing: the percentiles win"
        app._apply_windows_view({"windows": {"base": [0.1, 0.5], "bad": [1, 0]},
                                 "swap_channel": "edges_s1"}, notes)
        assert app._channel_windows == {"base": (0.1, 0.5)} and app._swap_channel == "edges_s1"
        assert notes and "bad" in notes[-1], notes
        app._apply_windows_view(vdoc, [])
        assert app._channel_windows["color_c1"] == (0.3, vdoc["windows"]["color_c1"][1])
        app._refresh_render()
        app.stat_color_var.set(True)
        app.stat_kind_vars["dizenzo"][0].set(True)
        app.stat_kind_vars["blur"][0].set(True)
        app.stat_kind_vars["blur"][1].set("0.7")
        app.stat_kind_vars["blur"][2].set("color")
        app.hist_on_var.set(True)
        app.hist_channels_var.set("base, color_c0")
        app._on_stat_spec_change()
        doc = json.loads(app._params_json())
        assert doc["input"]["color"]["channels"] == 3, doc.get("input")
        names = app._stat_channel_names()
        assert "color_c2" in names and "dizenzo_largest_s0.7" in names and "blur_c1_s0.7" in names, names
        assert "hist00_color_c0" in config_io.query_fields(app._params_json())
        app._measure_hist_ranges()
        assert app.hist_ranges_var.get().startswith("*: 0, ") or app.hist_ranges_var.get().startswith("*: "),             app.hist_ranges_var.get()
        d = app._preview_channel(planes, rgb, "dizenzo_largest_s0.7")
        assert d.shape == raw.shape and d is not planes, "a colour-sourced channel previews"
        one = json.loads(single_channel_params(app._params_json(), "blur_c1_s0.7"))
        assert one["statistics"]["channels"] == [{"kind": "blur", "sigmas": [0.7], "source": "color"}], one
        prof = app._profile_from_ui()
        assert prof["input"]["color"]["channels"] == 3 and prof["statistics"]["histogram"]["bins"] == 16
        assert session.profile_summary(prof).splitlines()[1].endswith("+16h×2"), session.profile_summary(prof)
        app.hist_on_var.set(False)
        app.stat_color_var.set(False)
        app.stat_kind_vars["dizenzo"][0].set(False)
        app.stat_kind_vars["blur"][0].set(False)
        app.stat_kind_vars["blur"][2].set("base")
        app.base_cards = [app._new_filter_card()]
        app._rebuild_filter_cards("base")
        app._on_stat_spec_change()
        app.background_var.set("base")
        app._preview_cache.pop(rgb, None)
        app._preview_path = fake
        app._preview_chan_cache.clear()
        app._preview_path = None

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

        # Session round-trip (v2 doc): loading what the viewer saved must put
        # every control back, and re-exporting must reproduce the same config.
        before = json.load(open(app._write_configs(d)[0]))
        doc = app._session_doc()
        assert session.is_session_doc(doc)
        assert len(doc["sequences"]) == 2
        assert doc["sequences"][0]["files"] == [
            "asdf_0011.tiff", "asdf_0012.tiff", "asdf_0013.tiff"], \
            "the doc stores basenames under the folder reference"
        assert doc["folders"] == [{"path": data_dir, "name": "data"}]

        # Clobber every widget-backed value, then restore from the session.
        app._clear_subsequences()
        app.folders = []; app.active_folder_idx = None; app._refresh_folder_list()
        app.filter_cards = [app._new_filter_card()]; app._rebuild_filter_cards()
        app.base_cards = [app._new_filter_card()]; app._rebuild_filter_cards("base")
        app.query_cards = [app._new_query_card()]; app._rebuild_query_cards()
        app.pixel_cards = [app._new_pixel_card()]; app._rebuild_pixel_cards()
        app.persist_pct_var.set("99"); app.connectivity_var.set(26)
        app.manifold_var.set("descending"); app.min_area_var.set("7")
        app._apply_session_docs([(os.path.join(d, "last_session.json"), doc)], "test")

        assert len(app.subsequences) == 2, app.subsequences
        assert len(app.subseq_list.get_children()) == 2, \
            "the tree must follow the model"
        assert app.folders == [{"path": data_dir, "name": "data"}]
        assert app.subsequences[0]["folder"] == "data"
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
        assert app.min_area_var.get() == "", "a profile without min_area clears the gate"
        # A load must leave no primed/assembly state behind -- the parameters
        # that produced it were just replaced.
        assert app.primed == [] and app.flat_slices == [] and not app._slices
        assert app._asm_pending is None and app._selection_dirty is False
        assert str(app.rerun_btn.cget("state")) == "disabled"

        after = json.load(open(app._write_configs(d)[0]))
        assert after == before, (before, after)

        # A blank no-data sentinel must come back BLANK, not as the default 0.0.
        def _setvar(var, value):
            var.set(value)
        app.base_cards[0]["params"]["omit_value"] = ""
        app._apply_profile_to_ui(session.profile_from_json(app._profile_from_ui()),
                                 _setvar, [])
        assert app.base_cards[0]["params"]["omit_value"] == "", \
            app.base_cards[0]["params"]
        app.base_cards[0]["params"]["omit_value"] = 43.0

        # Junk must not raise into the mainloop (a garbage doc reads as an
        # empty session), and an unreadable document must leave the state
        # alone rather than half-applying it.
        app._apply_session_doc({"profiles": "junk", "sequences": 7,
                                "folders": "nope"})
        assert app.subsequences == [] and len(app.profiles) == 1
        app._apply_session_docs([(os.path.join(d, "gone.json"), None)], "missing")
        assert app.subsequences == [], "a failed load changes nothing"
        assert isinstance(config_io.session_path(), str) and config_io.session_path()

        # --- auto-save ownership gate ------------------------------------ #
        # A freshly launched app is EMPTY and its timer fires 4 s later; without
        # this gate that empty state lands on top of a real saved session
        # before the user can reach "Restore last".
        fresh_root = tk.Tk(); fresh_root.withdraw()
        fresh = MscouponApp(fresh_root, autosave=False)
        assert fresh._session_owned is False, "a new window owns nothing"
        sess = os.path.join(d, "owned_session.json")
        config_io.write_session_text(config_io.serialize_session(doc), sess)
        keep = open(sess, encoding="utf-8").read()
        _real_path = config_io.session_path
        config_io.session_path = lambda app=None, name=None: sess
        try:
            fresh._autosave_now()            # the empty app must NOT write
            assert open(sess, encoding="utf-8").read() == keep, \
                "auto-save must not overwrite a session this window never opened"
            assert "held" in str(fresh.autosave_label.cget("text")), \
                fresh.autosave_label.cget("text")
            # Opening one takes ownership; auto-save then continues it.
            fresh._apply_session_docs([(sess, doc)], "test")
            assert fresh._session_owned is True
            fresh._clear_subsequences()
            fresh._autosave_now()
            assert open(sess, encoding="utf-8").read() != keep, \
                "auto-save writes once the session is owned"
            # ...and the previous generation survives as .1.
            root_, ext_ = os.path.splitext(sess)
            assert open(f"{root_}.1{ext_}", encoding="utf-8").read() == keep, \
                "the overwritten session is recoverable"
        finally:
            config_io.session_path = _real_path
            fresh_root.destroy()

        # Rotation is depth-3 and never removes the live file.
        rot = os.path.join(d, "rot.json")
        root_, ext_ = os.path.splitext(rot)
        for gen in ("a", "b", "c", "d"):
            config_io.write_session_text(gen, rot)
            config_io.rotate_session_backups(rot)
        assert open(rot, encoding="utf-8").read() == "d", "live file kept"
        assert open(f"{root_}.1{ext_}", encoding="utf-8").read() == "d"
        assert open(f"{root_}.2{ext_}", encoding="utf-8").read() == "c"
        assert open(f"{root_}.3{ext_}", encoding="utf-8").read() == "b"
        assert not os.path.exists(f"{root_}.4{ext_}"), "depth is capped at 3"
        config_io.rotate_session_backups(os.path.join(d, "nope.json"))   # no file

        # Restore the real session for everything below.
        app._apply_session_docs([(os.path.join(d, "last_session.json"), doc)], "test")
        assert len(app.subsequences) == 2

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

        # The spec must survive a profile round trip -- the whole block used to
        # be dropped on load except extremum_sample_radius.
        prof_s = session.profile_from_json(app._profile_from_ui())
        stats_s = config_io.statistics_from_json(prof_s["statistics"])
        assert stats_s["reductions"] == ["mean", "min", "max"], stats_s
        app.stat_kind_vars["blur"][0].set(False)          # perturb, then restore
        app.stat_reduction_vars["std"].set(True)
        app._apply_profile_to_ui(prof_s, _setvar, [])
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

        # --- a live preview over a primed stack --------------------------- #
        # An edit after a Run shows the NEW chain, and the primed overlays come
        # off with it: they describe a different field, so drawing them over
        # this raster would not be slightly stale but simply wrong. The badge
        # says which button fixes it, and re-priming clears the whole thing.
        if have_ext and app.viewer is not None:
            app.flat_slices = [(0, 0)]
            app.slice_var.set(0)
            app._preview_path = None            # browsing a primed slice
            app._preview_cache[os.path.join(data_dir, "a.tif")] = raw
            app.primed[0]["files"] = [os.path.join(data_dir, "a.tif")]
            app._primed_chain = app._chain_fingerprint()
            app._preview_sync = True
            app.background_var.set("filtered")
            app._preview_shown_key = None
            app._launch_preview()               # the chain as primed
            assert app._preview_override is None, "the primed chain is not stale"
            assert app._launch_preview() is None, "and asking again is free"
            app.filter_cards[0]["operation"] = "blur"
            app.filter_cards[0]["params"] = {"sigma": 2.5}
            assert app._launch_preview() == app._preview_token
            assert app._preview_override is not None, "an edit after a Run overrides"
            assert app._preview_override[1] == "filtered"
            assert app.viewer._hud_mode == "stale" and "Run" in app.viewer._hud_text
            assert app.viewer._overlays == [], "the primed overlays come off"
            # Every repaint keeps showing the live raster while it stands.
            app._refresh_render()
            assert _np.array_equal(app.viewer._base, app._preview_override[2])
            # A Run clears it: the primed rasters are the truth again, and the
            # cache steps back to the smaller budget.
            app._pending_fingerprint = "fp"
            app._handle_compute_event(("primed",))
            assert app._preview_override is None and app._preview_shown_key is None
            assert app._primed_chain == app._chain_fingerprint()
            assert app._preview_chan_cache.budget == _PREVIEW_CHAN_BUDGET_PRIMED
            app._preview_sync = False
            app.filter_cards[0]["operation"] = "blur"
            app.filter_cards[0]["params"] = {"sigma": 2.0}
            app._primed_chain = None
            app.flat_slices = []
        app.primed = saved_primed2
        app._preview_chan_cache.budget = _PREVIEW_CHAN_BUDGET

        # Restore the default spec so nothing below inherits a wide channel set.
        for on, *_rest in app.stat_kind_vars.values():
            on.set(False)
        app.stat_reduction_vars["std"].set(True)

        # --- compute profiles ---------------------------------------------- #
        # Switching snapshots the edited profile verbatim, restores the other
        # one's widgets, and drops the (fake) primed state.
        app._snapshot_active_profile()
        prof_a = json.loads(json.dumps(app.profiles[app.active_profile_idx]))
        app._profile_new()
        assert len(app.profiles) == 2 and app.active_profile_idx == 1
        assert app.filter_cards[0]["operation"] == "none", "new profile = defaults"
        assert app.primed == [] and app.flat_slices == []
        app._switch_profile(0)
        app._snapshot_active_profile()
        assert app.profiles[0] == prof_a, "switch snapshots the profile verbatim"
        assert app.filter_cards[0]["operation"] == "blur", "profile A restored"
        assert app.profiles[1]["name"] == "profile"
        assert app.profile_var.get() == prof_a["name"]

        # Profile file round trip through the on-disk format.
        prof_path = os.path.join(d, "a.profile.json")
        blob = config_io.serialize_session(session.profile_file_doc(app.profiles[0]))
        assert config_io.write_session_text(blob, prof_path)
        loaded = session.profile_from_json(config_io.read_json_file(prof_path))
        loaded["name"] = app.profiles[0]["name"]     # file doc keeps the name; be explicit
        assert loaded == app.profiles[0], "profile file round trip is lossless"

        # --- incremental priming -------------------------------------------- #
        # A subsequence carrying "_reuse" skips the worker entirely, so adding
        # a sequence never re-primes the ones already computed.
        fake_primed = {"files": ["x.tif"], "base": [], "filtered": [],
                       "pipes": [None], "normalizers": [[]]}
        eng2 = ComputeEngine(lambda si, level, li: {})
        eng2._run_worker([{"name": "s", "files": ["x.tif"],
                           "_reuse": fake_primed}], app._params_json(), {})
        evs = [e[0] for e in eng2.poll()]
        assert "primed" in evs and eng2.primed == [fake_primed]

        # --- legacy v1 import ----------------------------------------------- #
        # An old AppConfig+_gui session imports as folders + sequences + ONE
        # "imported" profile (replacing the profile list).
        cfg_v1 = app._config_for(app.subsequences[0]["files"], "out")
        gui_v1 = {"version": 1, "folder": data_dir,
                  "subsequences": [{"name": s["name"], "files": list(s["files"])}
                                   for s in app.subsequences],
                  "persist_live": "5", "alpha": 0.25}
        v1 = config_io.build_session(cfg_v1, gui_v1)
        app._apply_session_docs([("last_session_v1.json", v1)], "legacy")
        assert len(app.subsequences) == 2
        assert app.subsequences[0]["folder"] == "data"
        assert [p["name"] for p in app.profiles] == ["imported"]
        assert app.filter_cards[0]["operation"] == "blur", "v1 filters imported"
        assert float(app.alpha_var.get()) == 0.25
        assert app.persist_live_var.get() == "5"
    print("selftest OK: session (folders/sequences/preview + preview channels + colour"
          " + live preview on an edit), filters, base chain, "
          "stat channels, assembly tiers, per-slice selection, pixel trim, "
          "profiles + switch + file round-trip, session v2 round-trip, legacy import")
    # New session: folders, sequences and computed results go; the profiles
    # stay (or reset) as asked; ownership passes to the new session.
    names_before = [p["name"] for p in app.profiles]
    assert app._new_session_options()[0][0] == "profiles"
    assert app._new_session(keep={"profiles": True})
    assert app.folders == [] and app.subsequences == [] and not app.primed
    assert [p["name"] for p in app.profiles] == names_before
    assert app._session_owned and app.status_var.get().startswith("New session")
    assert app._new_session(keep={"profiles": False})
    assert len(app.profiles) == 1 and app.profiles[0]["name"] == "default"
    app._run_active = True
    assert not app._new_session(keep={"profiles": True}), "refused while priming"
    app._run_active = False

    root.destroy()


if __name__ == "__main__":
    main()
