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
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from . import config_io
from .config_io import FILTER_SCHEMA, FILTER_OPERATIONS, QUERY_FIELDS, QUERY_OPS


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
        self.query_cards = [self._new_query_card()]     # trailing empty card appended
        # primed[subseq_idx] = {"files":[...], "pipes":[pipe|None], "base":[arr],
        #                       "filtered":[arr]}  (populated by Run)
        self.primed = []
        self.flat_slices = []                    # [(subseq_idx, local_idx)] linearized
        self._assembly = {}                      # subseq_idx -> {global_of, features_3d, kept_global}
        self._work_q = queue.Queue()

        # --- tk variables ------------------------------------------------ #
        self.persist_pct_var = tk.StringVar(value="10")
        self.manifold_var = tk.StringVar(value="ascending")
        self.accurate_var = tk.BooleanVar(value=False)
        self.min_area_var = tk.StringVar(value="")
        self.connectivity_var = tk.IntVar(value=26)
        self.slice_var = tk.IntVar(value=0)
        self.persist_live_var = tk.DoubleVar(value=10.0)
        self.alpha_var = tk.DoubleVar(value=0.5)
        self.vmin_var = tk.DoubleVar(value=0.0)
        self.vmax_var = tk.DoubleVar(value=1.0)
        self.status_var = tk.StringVar(value="Ready.")

        # --- layout ------------------------------------------------------ #
        self.paned = ttk.PanedWindow(root, orient="horizontal")
        self.paned.pack(fill="both", expand=True)
        self.left = ttk.Frame(self.paned, width=360)
        self.right = ttk.Frame(self.paned, width=900)
        self.paned.add(self.left, weight=0)
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

    # ------------------------------------------------------------------ #
    # Left panel
    # ------------------------------------------------------------------ #
    def _build_left(self):
        # 1. Sequences
        c = ttk.LabelFrame(self.left, text="1. Sequences")
        c.pack(fill="x", padx=6, pady=4)
        ttk.Button(c, text="Browse folder…", command=self._browse_folder).pack(fill="x", padx=4, pady=2)
        self.file_list = tk.Listbox(c, selectmode="extended", height=7, exportselection=False)
        self.file_list.pack(fill="x", padx=4, pady=2)
        ttk.Button(c, text="Make subsequence from selection",
                   command=self._make_subsequence).pack(fill="x", padx=4, pady=2)
        ttk.Label(c, text="Subsequences:").pack(anchor="w", padx=4)
        self.subseq_list = tk.Listbox(c, height=4, exportselection=False)
        self.subseq_list.pack(fill="x", padx=4, pady=2)
        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=2)
        ttk.Button(row, text="Remove", command=self._remove_subsequence).pack(side="left")
        ttk.Button(row, text="Clear all", command=self._clear_subsequences).pack(side="left", padx=4)

        # 2. Filter chain
        self.filters_frame = ttk.LabelFrame(self.left, text="2. Filter chain")
        self.filters_frame.pack(fill="x", padx=6, pady=4)
        self._rebuild_filter_cards()

        # 3. MSC params
        c = ttk.LabelFrame(self.left, text="3. MSC parameters")
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
        ttk.Label(row, text="Per-slice min area:").pack(side="left")
        ttk.Entry(row, textvariable=self.min_area_var, width=8).pack(side="left", padx=4)

        # 4. Run
        c = ttk.LabelFrame(self.left, text="4. Run")
        c.pack(fill="x", padx=6, pady=4)
        self.run_btn = ttk.Button(c, text="Run with selected", command=self._run)
        self.run_btn.pack(fill="x", padx=4, pady=4)

        # 5. Export
        c = ttk.LabelFrame(self.left, text="5. Export")
        c.pack(fill="x", padx=6, pady=4)
        ttk.Button(c, text="Export config.json…", command=self._export_config).pack(
            fill="x", padx=4, pady=4)

    def _rebuild_filter_cards(self):
        for w in list(self.filters_frame.winfo_children()):
            w.destroy()
        for idx, card in enumerate(self.filter_cards):
            self._build_filter_card(idx, card)

    def _build_filter_card(self, idx, card):
        frame = ttk.Frame(self.filters_frame, relief="groove", borderwidth=1)
        frame.pack(fill="x", padx=4, pady=2)
        top = ttk.Frame(frame); top.pack(fill="x")
        op_var = tk.StringVar(value=card["operation"])
        combo = ttk.Combobox(top, textvariable=op_var, values=FILTER_OPERATIONS,
                             state="readonly", width=20)
        combo.pack(side="left", padx=2, pady=2)
        combo.bind("<<ComboboxSelected>>",
                   lambda e, i=idx, v=op_var: self._on_filter_op_change(i, v.get()))
        if idx < len(self.filter_cards) - 1 or card["operation"] != "none":
            ttk.Button(top, text="✕", width=3,
                       command=lambda i=idx: self._remove_filter_card(i)).pack(side="right", padx=2)
        # param widgets for the selected operation
        for pname, kind, default in FILTER_SCHEMA.get(card["operation"], []):
            self._build_param_row(frame, card["params"], pname, kind, default)

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
        else:  # float | int
            var = tk.StringVar(value=str(params[pname]))
            def commit(*_, p=pname, k=kind, v=var):
                try:
                    params[p] = int(v.get()) if k == "int" else float(v.get())
                except ValueError:
                    pass
            var.trace_add("write", commit)
            ttk.Entry(row, textvariable=var, width=10).pack(side="left")

    def _on_filter_op_change(self, idx, op):
        self.filter_cards[idx]["operation"] = op
        self.filter_cards[idx]["params"] = {}
        # keep exactly one trailing "none" card so the user can always add more
        self.filter_cards = [c for c in self.filter_cards if c["operation"] != "none"]
        self.filter_cards.append(self._new_filter_card())
        self._rebuild_filter_cards()

    def _remove_filter_card(self, idx):
        if 0 <= idx < len(self.filter_cards):
            del self.filter_cards[idx]
        if not self.filter_cards or self.filter_cards[-1]["operation"] != "none":
            self.filter_cards.append(self._new_filter_card())
        self._rebuild_filter_cards()

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
        except Exception as exc:  # numpy/PIL unavailable -> no live render
            ttk.Label(self.canvas_holder,
                      text=f"(renderer unavailable: {exc})").pack(padx=8, pady=8)

        # channel toggles
        chan = ttk.Frame(self.render_frame); chan.pack(fill="x")
        self.channel_vars = {}
        for ch in ("base", "filtered", "segmentation", "mask"):
            v = tk.BooleanVar(value=(ch in ("base", "segmentation")))
            self.channel_vars[ch] = v
            ttk.Checkbutton(chan, text=ch, variable=v,
                            command=self._refresh_render).pack(side="left", padx=4)

        live = ttk.LabelFrame(self.right, text="Live parameters")
        live.pack(side="bottom", fill="x", padx=6, pady=4)

        # slice slider (global, linearized over all subsequences)
        row = ttk.Frame(live); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Slice:").pack(side="left")
        self.slice_scale = ttk.Scale(row, from_=0, to=0, orient="horizontal",
                                     command=self._on_slice_change)
        self.slice_scale.pack(side="left", fill="x", expand=True, padx=4)
        self.slice_label = ttk.Label(row, text="-")
        self.slice_label.pack(side="left")

        # brightness / contrast + overlay alpha
        row = ttk.Frame(live); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Min/Max:").pack(side="left")
        ttk.Scale(row, from_=0.0, to=1.0, variable=self.vmin_var, orient="horizontal",
                  command=lambda *_: self._refresh_render()).pack(side="left", fill="x", expand=True)
        ttk.Scale(row, from_=0.0, to=1.0, variable=self.vmax_var, orient="horizontal",
                  command=lambda *_: self._refresh_render()).pack(side="left", fill="x", expand=True)
        row = ttk.Frame(live); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Overlay alpha:").pack(side="left")
        ttk.Scale(row, from_=0.0, to=1.0, variable=self.alpha_var, orient="horizontal",
                  command=lambda *_: self._refresh_render()).pack(side="left", fill="x", expand=True)

        # persistence (live simplification)
        row = ttk.Frame(live); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Persistence %:").pack(side="left")
        self.persist_scale = ttk.Scale(row, from_=0.0, to=100.0, variable=self.persist_live_var,
                                       orient="horizontal", command=self._on_persistence_change)
        self.persist_scale.pack(side="left", fill="x", expand=True, padx=4)

        # feature-query chain
        self.queries_frame = ttk.LabelFrame(live, text="Feature queries")
        self.queries_frame.pack(fill="x", padx=4, pady=4)
        self._rebuild_query_cards()

        row = ttk.Frame(live); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="3D connectivity:").pack(side="left")
        for c in (6, 18, 26):
            ttk.Radiobutton(row, text=str(c), variable=self.connectivity_var,
                            value=c, command=self._on_connectivity_change).pack(side="left")
        ttk.Button(live, text="Show merge tree", command=self._show_tree).pack(
            fill="x", padx=4, pady=2)

    def _rebuild_query_cards(self):
        for w in list(self.queries_frame.winfo_children()):
            w.destroy()
        for idx, card in enumerate(self.query_cards):
            self._build_query_card(idx, card)

    def _build_query_card(self, idx, card):
        frame = ttk.Frame(self.queries_frame); frame.pack(fill="x", padx=2, pady=1)
        field_var = tk.StringVar(value=card["field"])
        combo = ttk.Combobox(frame, textvariable=field_var, values=[""] + QUERY_FIELDS,
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
            self._apply_queries()
        val_var.trace_add("write", commit)
        ttk.Entry(frame, textvariable=val_var, width=8).pack(side="left", padx=1)
        if card["field"]:
            ttk.Button(frame, text="✕", width=3,
                       command=lambda i=idx: self._remove_query_card(i)).pack(side="left")

    def _on_query_field_change(self, idx, field):
        self.query_cards[idx]["field"] = field
        self.query_cards = [c for c in self.query_cards if c["field"]]
        self.query_cards.append(self._new_query_card())
        self._rebuild_query_cards()
        self._apply_queries()

    def _remove_query_card(self, idx):
        if 0 <= idx < len(self.query_cards):
            del self.query_cards[idx]
        if not self.query_cards or self.query_cards[-1]["field"]:
            self.query_cards.append(self._new_query_card())
        self._rebuild_query_cards()
        self._apply_queries()

    # ------------------------------------------------------------------ #
    # Run (prime) + live recompute -- wired to the compiled engine
    # ------------------------------------------------------------------ #
    def _params_json(self):
        try:
            pct = float(self.persist_pct_var.get())
        except ValueError:
            pct = 10.0
        return json.dumps({
            "filters": config_io.filters_to_json(self.filter_cards),
            "msc": {"manifold": self.manifold_var.get(),
                    "persistence_percent": pct,
                    "accurate_ascending": self.accurate_var.get(),
                    "accurate_descending": self.accurate_var.get()},
        })

    def _run(self):
        if not self.subsequences:
            messagebox.showinfo("mscoupon", "Define at least one subsequence first.")
            return
        self.run_btn.config(state="disabled")
        self.status_var.set("Priming…")
        params = self._params_json()
        subseqs = [dict(s) for s in self.subsequences]
        t = threading.Thread(target=self._run_worker, args=(subseqs, params), daemon=True)
        t.start()
        self.root.after(80, self._pump)

    def _run_worker(self, subseqs, params):
        try:
            import numpy as np
            from msseg import mscoupon as engine
            from PIL import Image
            p = json.loads(params)
            filters = p.get("filters", [])
            msc = p.get("msc", {})
            total = sum(len(s["files"]) for s in subseqs)
            log("=" * 60)
            log(f"RUN: {len(subseqs)} subsequence(s), {total} slices")
            log(f"  filters: {[f['operation'] for f in filters] or ['(none)']}")
            log(f"  msc: manifold={msc.get('manifold')} "
                f"persistence_percent={msc.get('persistence_percent')} "
                f"accurate={msc.get('accurate_ascending')}")
            done = 0
            primed = []
            for s in subseqs:
                log(f"subsequence: {os.path.basename(s['files'][0])} .. "
                    f"({len(s['files'])} slices)")
                base_slices, filt_slices, pipes = [], [], []
                for path in s["files"]:
                    arr = np.asarray(Image.open(path), dtype=np.float32)
                    if arr.ndim == 3:
                        arr = arr.mean(axis=2).astype(np.float32)
                    arr = np.ascontiguousarray(arr)
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
                    pipe = engine.prime_slice(arr, filt, params)
                    n_at_build = len(pipe.feature_stats())
                    log(f"  MSC primed: value_range={pipe.value_range():.4g} "
                        f"regions@{msc.get('persistence_percent')}%={n_at_build}")
                    base_slices.append(arr); filt_slices.append(filt); pipes.append(pipe)
                    done += 1
                    self._work_q.put(("progress", (done, total)))
                primed.append({"files": s["files"], "base": base_slices,
                               "filtered": filt_slices, "pipes": pipes})
            log(f"RUN complete: primed {total} slices")
            self._work_q.put(("done", primed))
        except Exception as exc:  # surfaced on the UI thread
            log(f"ERROR: {exc}")
            self._work_q.put(("error", exc))

    def _pump(self):
        try:
            while True:
                kind, payload = self._work_q.get_nowait()
                if kind == "progress":
                    done, total = payload
                    self.status_var.set(f"Priming slice {done}/{total}…")
                    continue
                if kind == "error":
                    self.run_btn.config(state="normal")
                    self.status_var.set(f"Error: {payload}")
                    messagebox.showerror("mscoupon", str(payload))
                    return
                # kind == "done"
                self.run_btn.config(state="normal")
                self.primed = payload
                self._assembly.clear()
                self._rebuild_flat_slices()
                self.status_var.set(f"Primed {len(self.primed)} subsequence(s), "
                                    f"{len(self.flat_slices)} slices.")
                cur = self._current()
                if cur is not None:
                    self._ensure_assembly(cur[0])
                self._refresh_render()
                return
        except queue.Empty:
            self.root.after(80, self._pump)

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

    def _on_slice_change(self, _value):
        cur = self._current()
        if cur is None:
            return
        si, li = cur
        self.slice_label.config(text=f"{self.subsequences[si]['name']} [{li}]")
        self._ensure_assembly(si)
        self._refresh_render()

    def _min_area(self):
        s = self.min_area_var.get().strip()
        try:
            return int(s) if s else None
        except ValueError:
            return None

    def _persist_pct(self):
        try:
            return float(self.persist_live_var.get())
        except (ValueError, tk.TclError):
            return 10.0

    def _on_persistence_change(self, _value):
        # Persistence is global -> all cached assemblies are stale.
        self._assembly.clear()
        cur = self._current()
        if cur is not None:
            self._ensure_assembly(cur[0])
        self._refresh_render()

    def _on_connectivity_change(self):
        # Connectivity changes the cross-slice linking -> re-assemble.
        self._assembly.clear()
        cur = self._current()
        if cur is not None:
            self._ensure_assembly(cur[0])
        self._refresh_render()

    def _apply_queries(self):
        cur = self._current()
        if cur is not None and cur[0] in self._assembly:
            self._refilter_queries(cur[0])   # cheap: re-filter without re-assembling
        elif cur is not None:
            self._ensure_assembly(cur[0])
        self._refresh_render()

    # -- 3D assembly ---------------------------------------------------- #
    def _ensure_assembly(self, si):
        """Assemble subsequence `si` at the current persistence (cached)."""
        if si in self._assembly or not self.primed:
            return
        try:
            import numpy as np
            from . import assembly as asm_mod
        except Exception:
            return
        p = self.primed[si]
        pct = self._persist_pct()
        min_area = self._min_area()
        name = self.subsequences[si]["name"]
        log(f"assemble '{name}': persistence={pct:.1f}% conn={self.connectivity_var.get()} "
            f"min_area={min_area}")
        labels_list, kept_list, stats_list = [], [], []
        total_2d = kept_2d = 0
        for li, pipe in enumerate(p["pipes"]):
            pabs = pipe.value_range() * pct / 100.0
            pipe.select_persistence(pabs)
            labels_list.append(np.asarray(pipe.labels()))
            sd = {int(f["feature_id"]): f for f in pipe.feature_stats()}
            stats_list.append(sd)
            kept = {fid for fid, f in sd.items()
                    if min_area is None or f.get("area", 0) >= min_area}
            kept_list.append(kept)
            total_2d += len(sd); kept_2d += len(kept)
        log(f"  2D features: {total_2d} total, {kept_2d} pass size gate "
            f"(across {len(p['pipes'])} slices)")
        global_of, feats3d = asm_mod.assemble(
            labels_list, kept_list, stats_list, self.connectivity_var.get())
        log(f"  3D assembly: {len(feats3d)} features")
        self._assembly[si] = {"global_of": global_of, "features_3d": feats3d,
                              "kept_global": None}
        self._refilter_queries(si)

    def _refilter_queries(self, si):
        """(Re)compute the kept 3D-feature set for subsequence `si` from the
        current feature-query chain (cheap; no re-assembly)."""
        data = self._assembly.get(si)
        if data is None:
            return
        feats = data["features_3d"]
        queries = config_io.queries_to_json(self.query_cards)
        if not queries:
            data["kept_global"] = set(f["global_id"] for f in feats)
        else:
            try:
                from msseg import mscoupon as engine
                flags = engine.evaluate_queries(feats, json.dumps(queries))
                data["kept_global"] = {f["global_id"] for f, k in zip(feats, flags) if k}
            except Exception:
                data["kept_global"] = set(f["global_id"] for f in feats)
        qdesc = ", ".join(f"{q['field']} {q['op']} {q['value']}" for q in queries) or "(none)"
        log(f"  queries [{qdesc}]: {len(data['kept_global'])}/{len(feats)} 3D features kept")
        self.status_var.set(
            f"{self.subsequences[si]['name']}: {len(feats)} 3D features, "
            f"{len(data['kept_global'])} kept")

    # -- rendering ------------------------------------------------------ #
    def _refresh_render(self):
        if self.viewer is None:
            return
        cur = self._current()
        if cur is None or not self.primed:
            return
        try:
            import numpy as np
            from msseg.viz import min_color
        except Exception:
            return
        si, li = cur
        p = self.primed[si]
        base = np.asarray(p["base"][li], dtype=np.float32)
        filt = np.asarray(p["filtered"][li], dtype=np.float32)
        labels = np.asarray(p["pipes"][li].labels())
        h, w = base.shape[:2]
        data = self._assembly.get(si)
        global_of = data["global_of"] if data else None
        kept_global = data["kept_global"] if data else None

        overlays = []
        if self.channel_vars["filtered"].get():
            fmin, fmax = float(filt.min()), float(filt.max())
            norm = np.clip((filt - fmin) / ((fmax - fmin) or 1.0), 0, 1)
            rgba = np.zeros((h, w, 4), np.uint8)
            rgba[..., 0] = (norm * 255).astype(np.uint8)
            rgba[..., 2] = ((1 - norm) * 255).astype(np.uint8)
            rgba[..., 3] = 160
            overlays.append({"rgba": rgba, "visible": True})

        want_seg = self.channel_vars["segmentation"].get()
        want_mask = self.channel_vars["mask"].get()
        if want_seg or want_mask:
            seg = np.zeros((h, w, 4), np.uint8)
            mask = np.zeros((h, w, 4), np.uint8)
            for fid in (int(v) for v in np.unique(labels) if v >= 0):
                gid = global_of.get((li, fid)) if global_of else fid
                if kept_global is not None and gid is not None and gid not in kept_global:
                    continue
                sel = labels == fid
                c = min_color(gid if gid is not None else fid)
                seg[sel] = [int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255]
                mask[sel] = [255, 255, 0, 255]
            if want_seg:
                overlays.append({"rgba": seg, "visible": True})
            if want_mask:
                overlays.append({"rgba": mask, "visible": True})

        first = self.viewer._base is None and self.viewer._source is None
        self.viewer.set_base(array=base, path=p["files"][li])
        self.viewer.set_overlays(overlays)
        self.viewer.set_window(self.vmin_var.get(), self.vmax_var.get())
        self.viewer.set_alpha(self.alpha_var.get())
        if first:
            self.viewer.fit()
        else:
            self.viewer.render()

    def _show_tree(self):
        cur = self._current()
        if cur is None or not self.primed:
            return
        si, li = cur
        try:
            from .tree_window import TreeWindow
        except Exception as exc:
            messagebox.showinfo("mscoupon", f"Merge-tree view unavailable: {exc}")
            return
        TreeWindow(self.root, self.primed[si]["pipes"][li].merge_tree_json())

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
                persistence_percent=pct,
                manifold=self.manifold_var.get(),
                accurate=self.accurate_var.get(),
                min_area=min_area,
                feature_filters=self.query_cards,
                connectivity=self.connectivity_var.get(),
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

    # Feature query chain.
    app._on_query_field_change(0, "area")
    app.query_cards[0]["op"] = "ge"; app.query_cards[0]["value"] = 50.0
    assert app.query_cards[-1]["field"] == ""

    # Config export matches the C++ schema.
    with tempfile.TemporaryDirectory() as d:
        paths = app._write_configs(d)
        assert len(paths) == 2
        cfg = json.load(open(paths[0]))
        assert cfg["input"]["files"] == app.subsequences[0]["files"]
        assert cfg["filters"] == [{"operation": "blur", "params": {"sigma": 2.0}}]
        assert cfg["msc"]["manifold"] == "ascending"
        assert cfg["feature_filters"] == [{"field": "area", "op": "ge", "value": 50.0}]
        assert cfg["assembly"]["connectivity"] == 26
    print("selftest OK: sequences, filter chain, query chain, config export")
    root.destroy()


if __name__ == "__main__":
    main()
