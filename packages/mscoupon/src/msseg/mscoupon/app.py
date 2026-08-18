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
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from . import config_io
from .config_io import FILTER_SCHEMA, FILTER_OPERATIONS, QUERY_OPS, query_fields


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def log(msg):
    """Command-line stage logging (alongside MSCEER's own stdout, which is left
    verbose on purpose -- it reports the data's critical-point/cancellation
    structure). Launch mscoupon-gui from a terminal to see all of it."""
    print(f"[mscoupon] {msg}", flush=True)


def natural_key(path: str):
    """Natural sort key (so asdf_2 < asdf_10)."""
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_tiffs(folder: str):
    """Naturally-sorted .tif/.tiff files in `folder`."""
    try:
        entries = [os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".tif", ".tiff"))]
    except OSError:
        return []
    return sorted(entries, key=natural_key)


def _wheel_delta(event):
    """Scroll units for one wheel event, normalized across platforms.

    Windows/macOS report a signed `delta` (120 per notch); X11 sends Button-4/5
    with no delta at all. Positive result = scroll down.
    """
    num = getattr(event, "num", 0)
    if num == 4:
        return -1
    if num == 5:
        return 1
    return int(-getattr(event, "delta", 0) / 120) or 0


def _bind_click_to_value(scale):
    """Make a ttk.Scale jump to the clicked/dragged position (absolute) instead of
    stepping toward it by a page increment. Approximates the trough as the full
    widget width; the endpoints clamp to from/to."""
    def jump(event):
        w = scale.winfo_width()
        if w <= 1:
            return None
        frac = min(max(event.x / w, 0.0), 1.0)
        lo = float(scale.cget("from"))
        hi = float(scale.cget("to"))
        scale.set(lo + frac * (hi - lo))
        return "break"
    scale.bind("<Button-1>", jump)
    scale.bind("<B1-Motion>", jump)


def _id_lut(raster, min_colors, np):
    """RGBA color LUT indexed by non-negative label id (the canvas treats id<0 as
    transparent background). Colors come from the shared min_colors palette."""
    K = int(raster.max()) + 1 if raster.size else 1
    lut = np.zeros((max(K, 1), 4), np.uint8)
    ids = np.arange(max(K, 1))
    lut[:, :3] = (min_colors(ids) * 255).astype(np.uint8)
    lut[:, 3] = 255
    return lut


def group_contiguous(indices):
    """Group a sorted list of indices into contiguous runs -> list of lists."""
    runs, cur = [], []
    for i in sorted(indices):
        if cur and i == cur[-1] + 1:
            cur.append(i)
        else:
            if cur:
                runs.append(cur)
            cur = [i]
    if cur:
        runs.append(cur)
    return runs


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
class MscouponApp:
    def __init__(self, root, initial=None):
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
        # primed[subseq_idx] = {"files":[...], "pipes":[pipe|None], "base":[arr],
        #                       "filtered":[arr]}  (populated by Run)
        self.primed = []
        self.flat_slices = []                    # [(subseq_idx, local_idx)] linearized
        # subseq_idx -> 3D assembly result (cc/global label rasters + global table).
        # Only populated at the "global" level -- see _needed_level().
        self._assembly = {}
        # (subseq_idx, local_idx) -> per-slice result at some commit:
        #   {commit, labels, stats, kept, cc (optional)}
        # The per-slice tiers write here; the global tier fills it for every slice
        # as a by-product, so navigating after a full assembly costs nothing.
        self._slices = {}
        self._work_q = queue.Queue()
        # Async assembly (off the UI thread): pipes are stateful, so only ONE
        # assembly worker runs at a time (single-flight); newer requests supersede.
        self._asm_token = 0
        self._asm_running = False
        self._asm_running_si = None              # subsequence the worker is assembling
        self._asm_pending = None                 # (token, si, level, li) latest requested
        self._commit_id = 0                       # committed parameter generation
        self._pump_started = False
        self._run_active = False                 # priming worker in flight

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
        self.status_var = tk.StringVar(value="Ready.")
        self.hover_var = tk.StringVar(value="")
        self._hover_ctx = None                           # cached arrays for the hover readout

        # --- layout ------------------------------------------------------ #
        self.paned = ttk.PanedWindow(root, orient="horizontal")
        self.paned.pack(fill="both", expand=True)

        # The left panel's six sections outgrow the window as soon as a few
        # filter/query cards are added, so it scrolls: a canvas carries the real
        # panel and `self.left` IS that inner frame, which leaves _build_left()
        # and every section below it untouched.
        self.left_pane = ttk.Frame(self.paned, width=376)
        self.left_canvas = tk.Canvas(self.left_pane, width=360, highlightthickness=0,
                                     borderwidth=0, takefocus=0)
        left_scroll = ttk.Scrollbar(self.left_pane, orient="vertical",
                                    command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=left_scroll.set)
        left_scroll.pack(side="right", fill="y")
        self.left_canvas.pack(side="left", fill="both", expand=True)

        self.left = ttk.Frame(self.left_canvas)
        self._left_window = self.left_canvas.create_window(
            (0, 0), window=self.left, anchor="nw")
        self.left.bind("<Configure>", self._on_left_content_resize)
        self.left_canvas.bind("<Configure>", self._on_left_canvas_resize)
        # Wheel scrolling is armed only while the pointer is over the panel: the
        # slice canvas binds the wheel to zoom (viewer_canvas.py), so a permanent
        # bind_all would hijack it.
        self.left_canvas.bind("<Enter>", self._bind_left_wheel)
        self.left_canvas.bind("<Leave>", self._unbind_left_wheel)

        self.right = ttk.Frame(self.paned, width=900)
        self.paned.add(self.left_pane, weight=0)
        self.paned.add(self.right, weight=1)
        self._build_left()
        self._build_right()

        ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w").pack(
            side="bottom", fill="x")

        if initial and os.path.isdir(initial):
            self._set_folder(initial)

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
    # Left-panel scrolling
    # ------------------------------------------------------------------ #
    def _on_left_content_resize(self, _event=None):
        """Sections were added/removed -> refresh the scrollable extent."""
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

    def _on_left_canvas_resize(self, event):
        """Hold the inner frame at the canvas width so `fill="x"` still spans."""
        self.left_canvas.itemconfigure(self._left_window, width=event.width)

    def _left_scrollable(self):
        box = self.left_canvas.bbox("all")
        return bool(box) and box[3] > self.left_canvas.winfo_height()

    def _bind_left_wheel(self, _event=None):
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.left_canvas.bind_all(seq, self._on_left_wheel)

    def _unbind_left_wheel(self, _event=None):
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.left_canvas.unbind_all(seq)

    def _on_left_wheel(self, event):
        # Widgets with their own wheel binding (the sequence listbox) return
        # "break" before this fires, so they scroll instead of the panel.
        if self._left_scrollable():
            self.left_canvas.yview_scroll(_wheel_delta(event), "units")

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

        # 4. Run
        c = ttk.LabelFrame(self.left, text="5. Run")
        c.pack(fill="x", padx=6, pady=4)
        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Cores/slice:").pack(side="left")
        ttk.Entry(row, textvariable=self.cores_per_slice_var, width=5).pack(side="left", padx=(2, 10))
        ttk.Label(row, text="Concurrent slices:").pack(side="left")
        ttk.Entry(row, textvariable=self.concurrent_slices_var, width=5).pack(side="left", padx=2)
        self.run_btn = ttk.Button(c, text="Run with selected", command=self._run)
        self.run_btn.pack(fill="x", padx=4, pady=4)

        # 5. Export
        c = ttk.LabelFrame(self.left, text="6. Export")
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

    def _make_subsequence(self):
        sel = list(self.file_list.curselection())
        if not sel:
            return
        files = [self.all_files[i] for i in sel]
        name = f"seq{len(self.subsequences) + 1} ({len(files)})"
        self.subsequences.append({"name": name, "files": files})
        self.subseq_list.insert("end", name)

    def _remove_subsequence(self):
        sel = list(self.subseq_list.curselection())
        for i in reversed(sel):
            del self.subsequences[i]
            self.subseq_list.delete(i)

    def _clear_subsequences(self):
        self.subsequences.clear()
        self.subseq_list.delete(0, "end")

    # ------------------------------------------------------------------ #
    # Right panel
    # ------------------------------------------------------------------ #
    def _scale(self, parent, **kw):
        """A ttk.Scale that jumps to the click/drag position (see _bind_click_to_value)."""
        s = ttk.Scale(parent, **kw)
        _bind_click_to_value(s)
        return s

    @staticmethod
    def _scrolled_listbox(parent, **kw):
        """A Listbox with a vertical scrollbar + mouse-wheel scrolling (production
        folders have thousands of files)."""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=4, pady=2)
        lb = tk.Listbox(frame, **kw)
        sb = ttk.Scrollbar(frame, orient="vertical", command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        lb.pack(side="left", fill="both", expand=True)
        # "break" keeps the wheel here instead of also scrolling the left panel.
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):   # Button-4/5 = X11
            lb.bind(seq, lambda e, w=lb: (w.yview_scroll(_wheel_delta(e), "units"), "break")[1])
        return lb

    def _build_right(self):
        self.render_frame = ttk.Frame(self.right)
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

        # background image (base | filtered, drawn fully opaque) + overlay toggles
        chan = ttk.Frame(self.render_frame); chan.pack(fill="x")
        ttk.Label(chan, text="Image:").pack(side="left", padx=(4, 0))
        self.background_var = tk.StringVar(value="base")
        for ch in ("base", "filtered"):
            ttk.Radiobutton(chan, text=ch, variable=self.background_var, value=ch,
                            command=self._refresh_render).pack(side="left", padx=4)
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

        # hover readout (values under the cursor)
        ttk.Label(self.render_frame, textvariable=self.hover_var, anchor="w",
                  font=("TkFixedFont", 8)).pack(fill="x", padx=4)

        live = ttk.LabelFrame(self.right, text="Live parameters")
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
        exactly what an exported config will validate against."""
        return config_io.query_fields()

    def _build_query_card(self, idx, card):
        frame = ttk.Frame(self.queries_frame); frame.pack(fill="x", padx=2, pady=1)
        field_var = tk.StringVar(value=card["field"])
        combo = ttk.Combobox(frame, textvariable=field_var, values=[""] + self._query_fields(),
                             state="readonly", width=13)
        combo.pack(side="left", padx=1)
        combo.bind("<<ComboboxSelected>>",
                   lambda e, i=idx, v=field_var: self._on_query_field_change(i, v.get()))
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
        })

    def _run(self):
        if not self.subsequences:
            messagebox.showinfo("mscoupon", "Define at least one subsequence first.")
            return
        self.run_btn.config(state="disabled")
        self.status_var.set("Priming…")
        params = self._params_json()
        subseqs = [dict(s) for s in self.subsequences]
        self._run_active = True
        t = threading.Thread(target=self._run_worker, args=(subseqs, params), daemon=True)
        t.start()
        self._ensure_pump()

    @staticmethod
    def _apply_base_chain(arr, base_filters, engine, log):
        """Run the base-channel chain, returning (raster, measured landmarks).

        A `normalize` stage is measured here rather than inside
        ``engine.filter_slice`` so the GUI can show the landmarks it resolved --
        it is the same C++ measure and the same affine map, so the exported
        config reproduces this exactly.
        """
        import numpy as np
        from msseg.mscoupon.normalize import measure_two_point

        cur = arr
        measured = []
        for i, f in enumerate(base_filters):
            if f.get("operation") == "normalize":
                params = dict(f.get("params", {}))
                tp = measure_two_point(cur, **params)
                cur = tp.apply(cur, clamp=bool(params.get("clamp", False)))
                measured.append(tp)
                log(f"  base[{i}] normalize({params.get('method', 'gmm')}) "
                    f"-> low={tp.low:.6g} high={tp.high:.6g} "
                    f"[{cur.min():.4g}, {cur.max():.4g}]")
            else:
                cur = engine.filter_slice(cur, json.dumps({"filter": f}))
                log(f"  base[{i}] {f['operation']}({f.get('params', {})}) "
                    f"-> min={cur.min():.4g} max={cur.max():.4g}")
        return np.ascontiguousarray(cur, dtype=np.float32), measured

    def _run_worker(self, subseqs, params):
        try:
            import numpy as np
            from msseg import mscoupon as engine
            from PIL import Image
            p = json.loads(params)
            filters = p.get("filters", [])
            base_filters = p.get("base_filters", [])
            msc = p.get("msc", {})
            # Serial-MSC variant of the params, used as a graceful fallback if the
            # linked MSCEER lacks the partitioned ComputeOptions surface.
            msc_serial = {k: v for k, v in msc.items()
                          if k not in ("compute_algorithm", "requested_parallelism")}
            params_serial = json.dumps({"filters": filters, "msc": msc_serial})
            use_serial = "compute_algorithm" not in msc
            total = sum(len(s["files"]) for s in subseqs)
            log("=" * 60)
            log(f"RUN: {len(subseqs)} subsequence(s), {total} slices")
            log(f"  filters: {[f['operation'] for f in filters] or ['(none)']}")
            log(f"  base_filters: {[f['operation'] for f in base_filters] or ['(none)']}")
            log(f"  msc: manifold={msc.get('manifold')} "
                f"persistence_percent={msc.get('persistence_percent')} "
                f"accurate={msc.get('accurate_ascending')} "
                f"algorithm={msc.get('compute_algorithm', 'serial')} "
                f"requested_parallelism={msc.get('requested_parallelism', 0)}")
            # Concurrency picture: priming still walks slices serially in ONE daemon
            # worker thread, but each slice's MSC (discrete gradient / partitioned
            # construction / manifold labeling) runs cores/slice-way parallel inside
            # MSCEER when compute_algorithm=partitioned. "Concurrent slices" (running
            # whole slices at once) is honored by the exported CLI config, which has
            # the full lane pipeline; the live GUI would additionally need the pybind
            # bindings to release the GIL to overlap slices.
            log(f"  concurrency: worker_thread={threading.current_thread().name!r} "
                f"os.cpu_count={os.cpu_count()} "
                f"cores/slice(MSC)={self._cores_per_slice()} "
                f"concurrent_slices(export)={self._concurrent_slices()} "
                f"scheduling=serial(1 slice at a time)")
            done = 0
            primed = []
            for s in subseqs:
                log(f"subsequence: {os.path.basename(s['files'][0])} .. "
                    f"({len(s['files'])} slices)")
                base_slices, filt_slices, pipes, norms = [], [], [], []
                for path in s["files"]:
                    t_slice = time.perf_counter()
                    arr = np.asarray(Image.open(path), dtype=np.float32)
                    if arr.ndim == 3:
                        arr = arr.mean(axis=2).astype(np.float32)
                    arr = np.ascontiguousarray(arr)
                    t_load = time.perf_counter()
                    log(f"slice {done + 1}/{total} {os.path.basename(path)}: "
                        f"shape={arr.shape} min={arr.min():.4g} max={arr.max():.4g} "
                        f"mean={arr.mean():.4g}")
                    # Apply the filter chain step by step so each stage's params +
                    # output range are logged (functionally == filter_chain).
                    cur = arr
                    for i, f in enumerate(filters):
                        cur = engine.filter_slice(cur, json.dumps({"filter": f}))
                        log(f"  filter[{i}] {f['operation']}({f.get('params', {})}) "
                            f"-> min={cur.min():.4g} max={cur.max():.4g}")
                    filt = np.ascontiguousarray(cur, dtype=np.float32)
                    # Base channel: the raster statistics and pixel thresholds are
                    # read from. Derived from the raw slice like `filters`, not
                    # chained onto it, matching the C++ pipeline.
                    base, slice_norms = self._apply_base_chain(arr, base_filters, engine, log)
                    t_filter = time.perf_counter()
                    if use_serial:
                        pipe = engine.prime_slice(base, filt, params_serial)
                    else:
                        try:
                            pipe = engine.prime_slice(base, filt, params)
                        except RuntimeError as pe:
                            msg = str(pe)
                            if any(t in msg for t in ("BuilderMode", "ComputeOptions", "partitioned")):
                                log(f"  WARN: partitioned MSC unavailable ({msg}); "
                                    "falling back to serial MSC for remaining slices")
                                use_serial = True
                                pipe = engine.prime_slice(base, filt, params_serial)
                            else:
                                raise
                    t_prime = time.perf_counter()
                    n_at_build = len(pipe.feature_stats())
                    log(f"  MSC primed: value_range={pipe.value_range():.4g} "
                        f"regions@{msc.get('persistence_percent')}%={n_at_build}")
                    log(f"  slice timings [thread={threading.current_thread().name!r}]: "
                        f"load={1e3 * (t_load - t_slice):.0f}ms "
                        f"filters={1e3 * (t_filter - t_load):.0f}ms "
                        f"prime={1e3 * (t_prime - t_filter):.0f}ms "
                        f"total={1e3 * (t_prime - t_slice):.0f}ms")
                    base_slices.append(base); filt_slices.append(filt); pipes.append(pipe)
                    norms.append(slice_norms)
                    done += 1
                    self._work_q.put(("progress", (done, total)))
                primed.append({"files": s["files"], "base": base_slices,
                               "filtered": filt_slices, "pipes": pipes,
                               "normalizers": norms})
            log(f"RUN complete: primed {total} slices")
            self._work_q.put(("done", primed))
        except Exception as exc:  # surfaced on the UI thread
            log(f"ERROR: {exc}")
            self._work_q.put(("error", exc))

    def _ensure_pump(self):
        """Start the work-queue pump if it isn't already running."""
        if not self._pump_started:
            self._pump_started = True
            self.root.after(80, self._pump)

    def _pump(self):
        try:
            while True:
                kind, payload = self._work_q.get_nowait()
                self._handle_work(kind, payload)
        except queue.Empty:
            pass
        # Keep pumping while any async work (priming or assembly) is outstanding.
        if self._run_active or self._asm_running or self._asm_pending is not None:
            self.root.after(80, self._pump)
        else:
            self._pump_started = False

    def _handle_work(self, kind, payload):
        if kind == "progress":
            done, total = payload
            self.status_var.set(f"Priming slice {done}/{total}…")
        elif kind == "error":
            self._run_active = False
            self.run_btn.config(state="normal")
            self.status_var.set(f"Error: {payload}")
            messagebox.showerror("mscoupon", str(payload))
        elif kind == "done":
            self._run_active = False
            self.run_btn.config(state="normal")
            self.primed = payload
            self._assembly.clear()
            self._slices.clear()
            self._rebuild_flat_slices()
            self.status_var.set(f"Primed {len(self.primed)} subsequence(s), "
                                f"{len(self.flat_slices)} slices.")
            cur = self._current()
            if cur is not None:
                self._request_assembly(cur[0])   # off-thread; UI stays responsive
            self._refresh_render()
        elif kind == "assembly":
            self._on_assembly_done(*payload)

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
        self._commit_id += 1        # bumps the committed parameter generation
        cur = self._current()
        if cur is not None:
            self._request_assembly(cur[0])
        # Cached per-slice records are keyed by commit, so they fall out of date
        # automatically; drop them so memory does not grow one stack per Rerun.
        self._slices = {k: v for k, v in self._slices.items()
                        if v.get("commit") == self._commit_id}
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
        rec = self._slices.get((si, li))
        if rec is None or rec.get("commit") != self._commit_id:
            return False
        if level == "cc" and rec.get("cc") is None:
            return False
        if level == "global":
            data = self._assembly.get(si)
            return data is not None and data.get("_commit") == self._commit_id
        return True

    def _request_assembly(self, si, level=None):
        """Queue off-thread work for subsequence `si` at the current parameters.
        Pipes are stateful (select_persistence mutates them), so only ONE worker
        runs at a time; a newer request supersedes an in-flight one."""
        if not self.primed or si is None:
            return
        cur = self._current()
        li = cur[1] if cur is not None and cur[0] == si else 0
        if level is None:
            level = self._needed_level()
        if self._slice_ready(si, li, level):
            return                    # already have it at this tier
        self._asm_token += 1
        self._asm_pending = (self._asm_token, si, level, li)
        self._launch_pending_assembly()
        self._ensure_pump()
        self._update_busy()

    def _launch_pending_assembly(self):
        if self._asm_running or self._asm_pending is None:
            return
        token, si, level, li = self._asm_pending
        self._asm_pending = None
        self._asm_running = True
        self._asm_running_si = si
        params = {
            "pct": self._persist_pct(),
            "queries": config_io.queries_to_json(self.query_cards),
            "pixels": config_io.pixel_filters_to_json(self.pixel_cards),
            "connectivity": int(self.connectivity_var.get()),
            "min_area": self._min_area(),
            # Decides which side a 3D feature's seeding extremum comes from.
            "manifold": self.manifold_var.get(),
            "commit": self._commit_id,
            "level": level,
            "li": li,
        }
        what = self.subsequences[si]["name"]
        self.status_var.set(f"Assembling {what}…" if level == "global"
                            else f"Updating {what} [{li}]…")
        threading.Thread(target=self._assemble_worker, args=(token, si, params),
                         daemon=True).start()
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

    def _slice_result(self, si, li, params, engine, np, tm):
        """Per-slice work for one slice: re-threshold, labels, stats, selection.
        Returns the record cached in self._slices."""
        p = self.primed[si]
        pipe = p["pipes"][li]
        pct, queries, min_area = params["pct"], params["queries"], params["min_area"]
        qjson = json.dumps(queries)

        # Re-thresholding is the dominant per-slice cost (MSCEER cancellation), and
        # a Rerun triggered by a filter edit does not move persistence at all.
        # Track what each pipe is already at and skip the no-op -- comparing the
        # requested percentage rather than current_persistence(), because
        # select_persistence clamps to the build-time cap, so a request above the
        # cap would never compare equal.
        applied = p.setdefault("_applied_pct", [None] * len(p["pipes"]))
        t = time.perf_counter()
        if applied[li] != pct:
            pipe.select_persistence(pipe.value_range() * pct / 100.0)
            applied[li] = pct
        tm["persist"] += time.perf_counter() - t

        t = time.perf_counter()
        lab = np.asarray(pipe.labels())
        tm["labels"] += time.perf_counter() - t

        t = time.perf_counter()
        stats = pipe.feature_stats()               # 2D region stats
        sd = {int(f["feature_id"]): f for f in stats}
        tm["stats"] += time.perf_counter() - t

        t = time.perf_counter()
        flags = engine.evaluate_queries(stats, qjson) if queries else [True] * len(stats)
        kept = set()
        for f, ok in zip(stats, flags):
            if not ok:
                continue
            if min_area is not None and f.get("area", 0) < min_area:
                continue
            kept.add(int(f["feature_id"]))
        tm["query"] += time.perf_counter() - t

        return {"commit": params.get("commit", 0), "labels": lab, "stats": sd,
                "kept": kept, "cc": None, "n_feat": len(stats)}

    def _assemble_worker(self, token, si, params):
        """Worker, tiered by params["level"] -- see _needed_level().

        "slice"/"cc" touch only the visible slice; "global" does every slice and
        the cross-slice assembly. Posts the result to the UI thread."""
        try:
            import numpy as np
            from msseg import mscoupon as engine
            from . import assembly as asm_mod
            p = self.primed[si]
            pct, queries = params["pct"], params["queries"]
            pixels, conn = params["pixels"], params["connectivity"]
            level, li0 = params["level"], params["li"]
            ascending = params.get("manifold", "ascending") != "descending"
            t0 = time.perf_counter()
            # Per-stage wall clock. When a re-run feels slow the log has to say
            # which stage owns it, not just the total.
            tm = {"persist": 0.0, "labels": 0.0, "stats": 0.0, "query": 0.0, "rasters": 0.0}
            name = self.subsequences[si]["name"]

            if level != "global":
                # --- cheap tiers: the visible slice only --------------------- #
                rec = self._slice_result(si, li0, params, engine, np, tm)
                if level == "cc":
                    t = time.perf_counter()
                    base = np.asarray(p["base"][li0], dtype=np.float32)
                    filt = np.asarray(p["filtered"][li0], dtype=np.float32)
                    tm["rasters"] += time.perf_counter() - t
                    t = time.perf_counter()
                    mask = asm_mod.selection_mask(rec["labels"], rec["kept"])
                    mask = asm_mod.apply_pixel_filters(mask, base, filt, pixels)
                    lbl, n = asm_mod.per_slice_cc(mask, conn)
                    rec["cc"] = np.where(lbl > 0, lbl - 1, -1)
                    tm["cc"] = time.perf_counter() - t
                total = time.perf_counter() - t0
                log(f"slice '{name}' [{li0}] level={level}: persistence={pct:.2f}% "
                    f"selection={len(queries)} -> {len(rec['kept'])}/{rec['n_feat']} kept "
                    f"({1e3 * total:.0f}ms)")
                log("  stages: " + "  ".join(f"{k}={1e3 * v:.0f}ms"
                                             for k, v in tm.items() if v > 0.0))
                self._work_q.put(("assembly", (token, si, {"_level": level, "_li": li0,
                                                           "_commit": params.get("commit", 0),
                                                           "_slice": rec})))
                return

            # --- global: every slice + the cross-slice 3D assembly ----------- #
            merged_labels, merged_stats, base_list, filt_list, kept_list = [], [], [], [], []
            recs, n_feat = [], 0
            for li in range(len(p["pipes"])):
                rec = self._slice_result(si, li, params, engine, np, tm)
                recs.append(rec)
                n_feat += rec["n_feat"]
                merged_labels.append(rec["labels"])
                merged_stats.append(rec["stats"])
                kept_list.append(rec["kept"])
                t = time.perf_counter()
                base_list.append(np.asarray(p["base"][li], dtype=np.float32))
                filt_list.append(np.asarray(p["filtered"][li], dtype=np.float32))
                tm["rasters"] += time.perf_counter() - t

            t_sel = time.perf_counter()
            asm_timing = {}
            out = asm_mod.assemble_cc(merged_labels, kept_list, base_list, filt_list,
                                      pixel_rules=pixels, connectivity=conn,
                                      ascending=ascending, timing=asm_timing)
            tm["assemble"] = time.perf_counter() - t_sel
            out["merged_labels"] = merged_labels
            out["merged_stats"] = merged_stats
            out["kept_list"] = kept_list
            out["_commit"] = params.get("commit", 0)
            out["_level"] = "global"
            # Hand back the per-slice records too, so navigating after a full
            # assembly is free rather than re-deriving each slice on arrival.
            out["_slices"] = recs
            for li, rec in enumerate(recs):
                rec["cc"] = out["cc_labels"][li]
            total = time.perf_counter() - t0
            log(f"assemble '{name}': persistence={pct:.2f}% "
                f"conn={conn} selection={len(queries)} pixel_rules={len(pixels)} "
                f"-> {out['n_global']} global features "
                f"({1e3 * total:.0f}ms)")
            log("  stages: " + "  ".join(f"{k}={1e3 * v:.0f}ms" for k, v in tm.items())
                + f"   [{len(p['pipes'])} slices, {n_feat} features]")
            if asm_timing:
                log("  assemble: " + "  ".join(f"{k}={1e3 * v:.0f}ms"
                                               for k, v in asm_timing.items()))
            self._work_q.put(("assembly", (token, si, out)))
        except Exception as exc:
            log(f"ASSEMBLY ERROR: {exc}")
            self._work_q.put(("assembly", (token, si, None)))

    def _on_assembly_done(self, token, si, out):
        self._asm_running = False
        self._asm_running_si = None
        if out is not None and token == self._asm_token:
            if out.get("_level") == "global":
                self._assembly[si] = out
                # The global tier also produced every slice's record; cache them
                # so navigating the stack afterwards needs no further work.
                for li, rec in enumerate(out.pop("_slices", [])):
                    self._slices[(si, li)] = rec
            else:
                self._slices[(si, out["_li"])] = out["_slice"]
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
        self._launch_pending_assembly()   # process any newer request
        self._update_busy()               # clear the spinner if nothing is pending

    # -- rendering (reads only cached numpy rasters; never the live pipes) ---- #
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

        # Segmentation overlay: recolor the selected source's label raster by id.
        # MSC / MSC filtered / per-slice CC come from the per-slice cache, so they
        # render without the stack-wide 3D assembly ever running; only global CC
        # and the mask read `data`.
        overlays = []
        src = self.seg_source_var.get()
        rec = self._slices.get((si, li))
        if rec is not None and rec.get("commit") != self._commit_id:
            rec = None                    # stale: a Rerun superseded it
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

        first = self.viewer._base is None and self.viewer._source is None
        if self.background_var.get() == "filtered":
            self.viewer.set_base(array=filt, path=None)
            self.viewer.set_window(self.vmin_filt_var.get(), self.vmax_filt_var.get())
        else:
            self.viewer.set_base(array=base, path=p["files"][li])
            self.viewer.set_window(self.vmin_var.get(), self.vmax_var.get())
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
            stat2d = rec["stats"].get(fid)
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
        rows = list(rec["stats"].values())
        if not rows:
            return None

        active = [c for c in self.query_cards if c.get("field")]
        counts = [len(rows)]
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
            flags = engine.evaluate_queries(rows, qjson)
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

    def _write_configs(self, out_dir):
        """Write one config.json per subsequence; returns the written paths."""
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
        paths = []
        for i, s in enumerate(self.subsequences):
            cfg = config_io.build_config(
                files=s["files"],
                output_folder=os.path.join(out_dir, f"out_{i}"),
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
            )
            path = os.path.join(out_dir, f"config_{i}.json")
            config_io.dump_config(cfg, path)
            paths.append(path)
        return paths


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
    app = MscouponApp(root)

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

        # A workflow with no base chain must export exactly as it did before.
        app.base_cards = [app._new_filter_card()]
        plain = json.load(open(app._write_configs(d)[0]))
        assert "base_filters" not in plain
    print("selftest OK: sequences, filters, base chain, assembly tiers, "
          "per-slice selection, pixel trim, export")
    root.destroy()


if __name__ == "__main__":
    main()
