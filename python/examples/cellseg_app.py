"""cellseg interactive Tkinter application.

Left panel:  data selection, heavy-lift parameters, heavy compute + show-tree,
             Phase-B thresholds (recompute on entry/release), save buttons.
Right panel: render controls (fixed height) + resizable tri-planar slice views
             (XY/Z, XZ/Y, YZ/X) with segmentation overlays and crosshairs.

The tree window (Show tree) draws the voxel-count merge-tree icicle with a hover
readout + horizontal guide line.

Run:  python cellseg_app.py [optional_stack.tif_or_XxYxZ.raw]
"""
import os
import re
import sys
import json
import queue
import threading

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tifffile
from msseg import cellseg

# 16-entry LUT for the seg8 bit-flag overlay, 20-entry for labels.
_TAB20 = matplotlib.colormaps["tab20"]
_BIT_LUT = _TAB20(np.linspace(0, 1, 16))[:, :3]
_LAB_LUT = _TAB20(np.linspace(0, 1, 20))[:, :3]


# --------------------------------------------------------------------------- #
# Data loading helpers
# --------------------------------------------------------------------------- #
def parse_raw_dims(path):
    """(X, Y, Z) from a '<name>_XxYxZ.raw' filename, or None."""
    m = re.search(r"_(\d+)x(\d+)x(\d+)(?:\.[A-Za-z0-9]+)?$", os.path.basename(path))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def reduce_channels(a, channel="max"):
    """Collapse an RGB/RGBA channel axis (size 3/4) to one scalar per voxel."""
    if a.ndim >= 3 and a.shape[-1] in (3, 4):
        axis = a.ndim - 1
    elif a.ndim >= 4 and a.shape[1] in (3, 4):
        axis = 1
    else:
        return a
    if isinstance(channel, int):
        return np.take(a, channel, axis=axis)
    return a.mean(axis=axis) if channel == "mean" else a.max(axis=axis)


def load_volume(path, channel="max"):
    """Load a .tif/.tiff stack or a '<name>_XxYxZ.raw' into float32 (z, y, x)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".raw":
        dims = parse_raw_dims(path)
        if dims is None:
            raise ValueError("raw files must be named '<name>_XxYxZ.raw' to deduce dimensions")
        x, y, z = dims
        data = np.fromfile(path, dtype="<f4")
        if data.size != x * y * z:
            raise ValueError(f"{path}: expected {x*y*z} float32, found {data.size}")
        return np.ascontiguousarray(data.reshape(z, y, x), dtype=np.float32)

    raw = np.squeeze(np.asarray(tifffile.imread(path)))
    raw = reduce_channels(raw, channel)
    while raw.ndim > 3:
        raw = raw[0]
    if raw.ndim == 2:
        raw = raw[None, ...]
    if raw.ndim != 3:
        raise ValueError(f"expected a 2D/3D volume, got shape {raw.shape}")
    return np.ascontiguousarray(raw, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Merge-tree icicle (shared with the demo script)
# --------------------------------------------------------------------------- #
def draw_icicle(ax, tree, seed=0):
    """Draw the voxel-count merge-tree icicle on `ax`. Returns (ymin, ymax)."""
    rng = np.random.default_rng(seed)

    def rand_color():
        h = float(rng.random())
        s = 0.45 + 0.40 * float(rng.random())
        v = 0.75 + 0.20 * float(rng.random())
        import colorsys
        return colorsys.hsv_to_rgb(h, s, v)

    vals = []

    def collect_vals(n):
        vals.append(n["value"])
        for c in n.get("children", []):
            collect_vals(c)

    def set_lowest(n):
        kids = n.get("children", [])
        n["_lo"] = n["value"] if not kids else min(set_lowest(c) for c in kids)
        return n["_lo"]

    for r in tree["roots"]:
        collect_vals(r)
        set_lowest(r)
    vmin, vmax = min(vals), max(vals)
    vspan = (vmax - vmin) or 1.0
    root_top = vmax + 0.05 * vspan

    boxes = []

    def layout(n, x0, color, parent_value):
        boxes.append((x0, n["value"], float(n["voxel_count"]), parent_value - n["value"], color))
        kids = n.get("children", [])
        largest = max(kids, key=lambda c: c["voxel_count"], default=None)
        cursor = x0
        for c in sorted(kids, key=lambda c: c["_lo"]):
            layout(c, cursor, color if c is largest else rand_color(), n["value"])
            cursor += float(c["voxel_count"])

    cursor = 0.0
    for r in sorted(tree["roots"], key=lambda r: r["_lo"]):
        layout(r, cursor, rand_color(), root_top)
        cursor += float(r["voxel_count"])

    ax.clear()
    for x0, y0, w, h, color in boxes:
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=color, edgecolor="white", linewidth=1.0))
    ax.set_xlim(0, cursor or 1)
    ax.set_ylim(vmin - 0.02 * vspan, root_top)
    ax.set_xlabel("voxel count (feature size)")
    ax.set_ylabel("function value")
    return vmin - 0.02 * vspan, root_top


# --------------------------------------------------------------------------- #
# Tree window (icicle + hover guide)
# --------------------------------------------------------------------------- #
class TreeWindow:
    def __init__(self, master, app):
        self.app = app
        self.top = tk.Toplevel(master)
        self.top.title("cellseg merge tree")
        self.top.geometry("900x600")
        self.fig = Figure(figsize=(8, 5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.hline = None
        self.text = None
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.top.protocol("WM_DELETE_WINDOW", self._on_close)
        self.refresh()

    def refresh(self):
        if self.app.pipe is None:
            return
        tree = json.loads(self.app.pipe.merge_tree_json())
        draw_icicle(self.ax, tree)
        self.ax.set_title(f"merge tree — persistence {self.app.pipe.current_persistence():.4g} "
                          f"(hover for value; cut = {self.app.get_float('cut'):.4g})")
        self.hline = self.ax.axhline(0, ls=":", color="black", lw=0.9, visible=False)
        self.text = self.ax.text(0.99, 0.99, "", transform=self.ax.transAxes,
                                 ha="right", va="top", fontsize=9,
                                 bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))
        self.canvas.draw_idle()

    def _on_move(self, event):
        if self.hline is None:
            return
        if event.inaxes != self.ax or event.ydata is None:
            self.hline.set_visible(False)
            self.text.set_text("")
        else:
            self.hline.set_ydata([event.ydata, event.ydata])
            self.hline.set_visible(True)
            self.text.set_text(f"value = {event.ydata:.4g}")
        self.canvas.draw_idle()

    def _on_close(self):
        self.app.tree_win = None
        self.top.destroy()


# --------------------------------------------------------------------------- #
# One tri-planar slice view (figure + canvas + slider + events)
# --------------------------------------------------------------------------- #
class SlicePane:
    def __init__(self, parent, app, plane, label, slider_label, slider_max_getter):
        self.app = app
        self.plane = plane
        self.slider_max_getter = slider_max_getter
        self.frame = ttk.Frame(parent)

        self.fig = Figure(figsize=(4, 3))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect("equal", adjustable="datalim")   # 1:1 with voxel size
        self.im_base = None
        self.im_over = None
        self.contour = None          # live under-mouse contour (QuadContourSet)
        self.pan = None              # active pan drag state
        self.view_initialized = False
        self.vline = self.ax.axvline(0, ls=":", color="yellow", lw=0.8, alpha=0.7, visible=False)
        self.hline = self.ax.axhline(0, ls=":", color="yellow", lw=0.8, alpha=0.7, visible=False)
        self.title = self.ax.text(0.01, 0.99, label, transform=self.ax.transAxes,
                                  ha="left", va="top", color="white", fontsize=9,
                                  bbox=dict(boxstyle="round", fc="black", alpha=0.4))

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", lambda e: app.on_press(self, e))
        self.canvas.mpl_connect("button_release_event", lambda e: app.on_release(self, e))
        self.canvas.mpl_connect("motion_notify_event", lambda e: app.on_motion(self, e))
        self.canvas.mpl_connect("scroll_event", lambda e: app.on_scroll(self, e))
        self.canvas.mpl_connect("axes_leave_event", lambda e: app.on_leave(self, e))

        row = ttk.Frame(self.frame)
        row.pack(fill=tk.X)
        ttk.Label(row, text=slider_label, width=3).pack(side=tk.LEFT)
        self.var = tk.IntVar(value=0)
        self.slider = ttk.Scale(row, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.var,
                                command=self._on_slider)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.val_label = ttk.Label(row, text="0", width=5)
        self.val_label.pack(side=tk.LEFT)

    def configure_slider(self, maxv, value):
        self.slider.configure(to=max(1, maxv))
        self.var.set(value)
        self.val_label.configure(text=str(value))

    def clear_contour(self):
        if self.contour is not None:
            try:
                self.contour.remove()
            except Exception:  # noqa: BLE001
                pass
            self.contour = None

    def _on_slider(self, _evt=None):
        v = int(float(self.var.get()))
        self.val_label.configure(text=str(v))
        self.app.on_index_change(self.plane, v)


# --------------------------------------------------------------------------- #
# Main application
# --------------------------------------------------------------------------- #
class CellsegApp:
    def __init__(self, root, initial_path=None):
        self.root = root
        self.root.title("cellseg")
        self.root.geometry("1360x880")

        self.volume = None          # original (z, y, x) float32
        self.pipe = None            # cellseg.CellPipeline
        self.transformed = None     # filtered volume from the pipeline
        self.seg8 = None
        self.ids = None
        self.d = self.h = self.w = 0
        self.cz = self.cy = self.cx = 0
        self.tree_win = None
        self._busy = False
        self._result_q = queue.Queue()

        self._build_ui()
        self.root.after(80, self._pump)
        if initial_path:
            self._load_path(initial_path)

    # ----- UI construction -------------------------------------------------- #
    def _build_ui(self):
        outer = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(outer, width=340)
        right = ttk.Frame(outer)
        outer.add(left, weight=0)
        outer.add(right, weight=1)

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, left):
        # Data card
        data = ttk.LabelFrame(left, text="Data")
        data.pack(fill=tk.X, padx=6, pady=4)
        self.path_var = tk.StringVar()
        ttk.Entry(data, textvariable=self.path_var).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(data, text="Browse…", command=self._browse).pack(anchor="w", padx=4, pady=2)
        self.shape_label = ttk.Label(data, text="(no data loaded)")
        self.shape_label.pack(anchor="w", padx=4, pady=2)

        # Heavy params card
        heavy = ttk.LabelFrame(left, text="Heavy-lift parameters")
        heavy.pack(fill=tk.X, padx=6, pady=4)
        self.blur_var = tk.StringVar(value="2.0")
        self.pers_pct_var = tk.StringVar(value="5.0")
        self._labeled_entry(heavy, "Blur sigma", self.blur_var)
        self._labeled_entry(heavy, "Persistence %", self.pers_pct_var)
        self.heavy_btn = ttk.Button(heavy, text="Heavy compute", command=self._start_heavy)
        self.heavy_btn.pack(fill=tk.X, padx=4, pady=(4, 2))
        self.tree_btn = ttk.Button(heavy, text="Show tree", command=self._show_tree, state=tk.DISABLED)
        self.tree_btn.pack(fill=tk.X, padx=4, pady=2)

        # Phase-B thresholds card
        thr = ttk.LabelFrame(left, text="Phase-B thresholds (recompute on Enter)")
        thr.pack(fill=tk.X, padx=6, pady=4)
        self.persel_var = tk.StringVar(value="")
        self.cut_var = tk.StringVar(value="0.0")
        self.bg_var = tk.StringVar(value="0.0")
        for lbl, var in [("Persistence", self.persel_var), ("Cut threshold", self.cut_var),
                         ("Background", self.bg_var)]:
            e = self._labeled_entry(thr, lbl, var)
            e.bind("<Return>", lambda _e: self._recompute_phaseb())
            e.bind("<FocusOut>", lambda _e: self._recompute_phaseb())

        save = ttk.Frame(left)
        save.pack(fill=tk.X, padx=6, pady=6)
        self.save_lbl_btn = ttk.Button(save, text="Save labels", command=lambda: self._save("labels"),
                                       state=tk.DISABLED)
        self.save_bit_btn = ttk.Button(save, text="Save bitfield", command=lambda: self._save("bitfield"),
                                       state=tk.DISABLED)
        self.save_lbl_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.save_bit_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.status = ttk.Label(left, text="Ready", relief=tk.SUNKEN, anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _labeled_entry(self, parent, label, var):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(row, text=label, width=13).pack(side=tk.LEFT)
        e = ttk.Entry(row, textvariable=var)
        e.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return e

    def _build_right(self, right):
        # Render controls (fixed height)
        controls = ttk.LabelFrame(right, text="Render controls")
        controls.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        self.show_slice_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Show slice", variable=self.show_slice_var,
                        command=self.render_all).grid(row=0, column=0, sticky="w", padx=4)
        self.base_src_var = tk.StringVar(value="original")
        ttk.Radiobutton(controls, text="original", variable=self.base_src_var, value="original",
                        command=self.render_all).grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(controls, text="transformed", variable=self.base_src_var, value="transformed",
                        command=self.render_all).grid(row=0, column=2, sticky="w")

        ttk.Label(controls, text="Segmentation:").grid(row=1, column=0, sticky="w", padx=4)
        self.seg_mode_var = tk.StringVar(value="none")
        for i, mode in enumerate(("none", "bitfield", "labels")):
            ttk.Radiobutton(controls, text=mode, variable=self.seg_mode_var, value=mode,
                            command=self.render_all).grid(row=1, column=1 + i, sticky="w")

        ttk.Label(controls, text="Overlay alpha:").grid(row=2, column=0, sticky="w", padx=4)
        self.alpha_var = tk.DoubleVar(value=0.5)
        ttk.Scale(controls, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.alpha_var,
                  command=lambda _e: self.render_all()).grid(row=2, column=1, columnspan=3,
                                                             sticky="ew", padx=4)

        self.show_contour_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Show contour (at mouse value)",
                        variable=self.show_contour_var,
                        command=self._on_contour_toggle).grid(row=3, column=0, columnspan=3,
                                                              sticky="w", padx=4, pady=(2, 0))
        ttk.Button(controls, text="Reset view", command=self.reset_views).grid(
            row=3, column=3, sticky="e", padx=4)

        self.coord_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.coord_var, anchor="e",
                  font=("TkFixedFont", 9)).grid(row=4, column=0, columnspan=4, sticky="ew", padx=4)
        controls.columnconfigure(3, weight=1)

        # Resizable stack of the three slice views
        panes = ttk.PanedWindow(right, orient=tk.VERTICAL)
        panes.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)

        self.pane_xy = SlicePane(panes, self, "xy", "XY  (Z slider)", "Z", lambda: self.d - 1)
        self.pane_xz = SlicePane(panes, self, "xz", "XZ  (Y slider, Z vertical)", "Y", lambda: self.h - 1)
        self.pane_yz = SlicePane(panes, self, "yz", "YZ  (X slider, Z vertical)", "X", lambda: self.w - 1)
        for p in (self.pane_xy, self.pane_xz, self.pane_yz):
            panes.add(p.frame, weight=1)

    # ----- status / task pump ---------------------------------------------- #
    def set_status(self, msg):
        self.status.configure(text=msg)
        self.status.update_idletasks()

    def _pump(self):
        try:
            while True:
                kind, payload = self._result_q.get_nowait()
                if kind == "heavy_ok":
                    self._heavy_done(payload)
                elif kind == "heavy_err":
                    self._busy = False
                    self.heavy_btn.configure(state=tk.NORMAL)
                    messagebox.showerror("Heavy compute failed", str(payload))
                    self.set_status("Heavy compute failed")
        except queue.Empty:
            pass
        self.root.after(80, self._pump)

    # ----- data loading ----------------------------------------------------- #
    def _browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("Volumes", "*.tif *.tiff *.raw"), ("TIFF", "*.tif *.tiff"),
                       ("RAW", "*.raw"), ("All", "*.*")])
        if path:
            self._load_path(path)

    def _load_path(self, path):
        try:
            self.volume = load_volume(path)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        self.path_var.set(path)
        self.d, self.h, self.w = self.volume.shape
        self.cz, self.cy, self.cx = self.d // 2, self.h // 2, self.w // 2
        self.pipe = self.transformed = self.seg8 = self.ids = None
        self.tree_btn.configure(state=tk.DISABLED)
        self.save_lbl_btn.configure(state=tk.DISABLED)
        self.save_bit_btn.configure(state=tk.DISABLED)
        self.base_src_var.set("original")
        self.shape_label.configure(
            text=f"shape (z,y,x) = {self.volume.shape}\nrange [{self.volume.min():.4g}, "
                 f"{self.volume.max():.4g}]")
        self.pane_xy.configure_slider(self.d - 1, self.cz)
        self.pane_xz.configure_slider(self.h - 1, self.cy)
        self.pane_yz.configure_slider(self.w - 1, self.cx)
        self._reset_images()
        self.render_all()
        self.set_status(f"Loaded {os.path.basename(path)}")

    def _reset_images(self):
        for p in (self.pane_xy, self.pane_xz, self.pane_yz):
            for im in list(p.ax.images):
                im.remove()
            p.clear_contour()
            p.im_base = None
            p.im_over = None
            p.view_initialized = False

    # ----- heavy compute ---------------------------------------------------- #
    def _start_heavy(self):
        if self.volume is None:
            messagebox.showinfo("No data", "Load a volume first.")
            return
        if self._busy:
            return
        try:
            params = json.dumps({"blur_sigma": float(self.blur_var.get()),
                                 "persistence_percent": float(self.pers_pct_var.get())})
        except ValueError:
            messagebox.showerror("Bad parameter", "Blur sigma / persistence % must be numbers.")
            return
        self._busy = True
        self.heavy_btn.configure(state=tk.DISABLED)
        self.set_status("Heavy compute… (this runs once)")
        vol = self.volume

        def worker():
            try:
                pipe = cellseg.heavy_lift(vol, params)
                self._result_q.put(("heavy_ok", pipe))
            except Exception as exc:  # noqa: BLE001
                self._result_q.put(("heavy_err", exc))

        threading.Thread(target=worker, daemon=True).start()

    def _heavy_done(self, pipe):
        self.pipe = pipe
        self.transformed = np.asarray(pipe.filtered(), dtype=np.float32)
        self._busy = False
        self.heavy_btn.configure(state=tk.NORMAL)
        self.tree_btn.configure(state=tk.NORMAL)
        self.persel_var.set(f"{pipe.heavy_persistence():.6g}")
        self.set_status(f"Heavy done: value range {pipe.value_range():.4g}, "
                        f"persistence {pipe.heavy_persistence():.4g}")
        self._recompute_phaseb()

    # ----- Phase-B recompute ------------------------------------------------ #
    def get_float(self, which, default=0.0):
        var = {"cut": self.cut_var, "bg": self.bg_var, "persel": self.persel_var}[which]
        try:
            return float(var.get())
        except ValueError:
            return default

    def _recompute_phaseb(self):
        if self.pipe is None:
            return
        try:
            persel = float(self.persel_var.get())
        except ValueError:
            persel = self.pipe.heavy_persistence()
        cut = self.get_float("cut")
        bg = self.get_float("bg")
        self.set_status("Segmenting…")
        try:
            if abs(persel - self.pipe.current_persistence()) > 1e-9:
                self.pipe.set_persistence(persel)
                if self.tree_win is not None:
                    self.tree_win.refresh()
            seg8, ids = self.pipe.segment(cut, bg)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Segmentation failed", str(exc))
            self.set_status("Segmentation failed")
            return
        self.seg8 = np.asarray(seg8)
        self.ids = np.asarray(ids)
        self.save_lbl_btn.configure(state=tk.NORMAL)
        self.save_bit_btn.configure(state=tk.NORMAL)
        self.set_status(f"Segmented: cut={cut:.4g} bg={bg:.4g} "
                        f"(membrane={int((self.seg8 & 2 > 0).sum())} vox)")
        self.render_all()

    def _show_tree(self):
        if self.pipe is None:
            return
        if self.tree_win is None:
            self.tree_win = TreeWindow(self.root, self)
        else:
            self.tree_win.top.lift()
            self.tree_win.refresh()

    # ----- saving ----------------------------------------------------------- #
    def _save(self, which):
        arr = self.ids if which == "labels" else self.seg8
        if arr is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".tif",
                                            filetypes=[("TIFF", "*.tif"), ("RAW", "*.raw")])
        if not path:
            return
        out = arr.astype(np.int32 if which == "labels" else np.uint8)
        try:
            if path.lower().endswith(".raw"):
                out.tofile(path)
                with open(path + ".dat", "w") as fh:
                    fh.write(f"{self.w} {self.h} {self.d}\n")
            else:
                tifffile.imwrite(path, out)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save failed", str(exc))
            return
        self.set_status(f"Saved {which} -> {os.path.basename(path)}")

    # ----- interaction ------------------------------------------------------ #
    def on_index_change(self, plane, value):
        if plane == "xy":
            self.cz = value
        elif plane == "xz":
            self.cy = value
        elif plane == "yz":
            self.cx = value
        self.render_all()

    def on_double_click(self, plane, xdata, ydata):
        xi, yi = int(round(xdata)), int(round(ydata))
        if plane == "xy":            # sets x (YZ) and y (XZ)
            self.cx = np.clip(xi, 0, self.w - 1)
            self.cy = np.clip(yi, 0, self.h - 1)
        elif plane == "xz":          # sets x (YZ) and z (XY)
            self.cx = np.clip(xi, 0, self.w - 1)
            self.cz = np.clip(yi, 0, self.d - 1)
        elif plane == "yz":          # sets y (XZ) and z (XY)
            self.cy = np.clip(xi, 0, self.h - 1)
            self.cz = np.clip(yi, 0, self.d - 1)
        self.pane_xy.configure_slider(self.d - 1, self.cz)
        self.pane_xz.configure_slider(self.h - 1, self.cy)
        self.pane_yz.configure_slider(self.w - 1, self.cx)
        self.render_all()

    def on_press(self, pane, event):
        if event.inaxes != pane.ax:
            return
        if event.dblclick:
            if event.xdata is not None and event.ydata is not None:
                self.on_double_click(pane.plane, event.xdata, event.ydata)
            pane.pan = None
            return
        if event.button == 1 and event.x is not None:   # begin pan
            pane.pan = (event.x, event.y, pane.ax.get_xlim(), pane.ax.get_ylim(),
                        pane.ax.get_window_extent())

    def on_release(self, pane, _event):
        pane.pan = None

    def on_motion(self, pane, event):
        if pane.pan is not None and event.x is not None:      # pan drag
            px, py, (x0, x1), (y0, y1), bbox = pane.pan
            if bbox.width and bbox.height:
                dx = (event.x - px) * (x1 - x0) / bbox.width
                dy = (event.y - py) * (y1 - y0) / bbox.height
                pane.ax.set_xlim(x0 - dx, x1 - dx)
                pane.ax.set_ylim(y0 - dy, y1 - dy)
                pane.canvas.draw_idle()
        self._update_readout(pane, event)

    def on_scroll(self, pane, event):                         # wheel zoom about cursor
        if event.inaxes != pane.ax or event.xdata is None:
            return
        scale = 1.0 / 1.2 if event.button == "up" else 1.2
        x0, x1 = pane.ax.get_xlim()
        y0, y1 = pane.ax.get_ylim()
        xd, yd = event.xdata, event.ydata
        pane.ax.set_xlim(xd + (x0 - xd) * scale, xd + (x1 - xd) * scale)
        pane.ax.set_ylim(yd + (y0 - yd) * scale, yd + (y1 - yd) * scale)
        pane.canvas.draw_idle()

    def on_leave(self, pane, _event):
        self.coord_var.set("")
        if self.show_contour_var.get():
            pane.clear_contour()
            pane.canvas.draw_idle()

    def _coord_for(self, plane, col, rowi):
        if plane == "xy":
            return col, rowi, self.cz
        if plane == "xz":
            return col, self.cy, rowi
        return self.cx, col, rowi

    def _update_readout(self, pane, event):
        if self.volume is None or event.inaxes != pane.ax or event.xdata is None:
            return
        base2d = self._base_slice(pane.plane, self._base_volume())
        col, rowi = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= rowi < base2d.shape[0] and 0 <= col < base2d.shape[1]:
            val = float(base2d[rowi, col])
            x, y, z = self._coord_for(pane.plane, col, rowi)
            self.coord_var.set(f"value {val:.4g}    (x={x}, y={y}, z={z})")
            if self.show_contour_var.get() and pane.pan is None:
                self._update_contour(pane, base2d, val)
        else:
            self.coord_var.set("")

    def _update_contour(self, pane, base2d, level):
        pane.clear_contour()
        try:
            pane.contour = pane.ax.contour(base2d, levels=[level], colors="red",
                                           linestyles=":", linewidths=0.9, alpha=0.7)
        except Exception:  # noqa: BLE001
            pane.contour = None
        pane.canvas.draw_idle()

    def _on_contour_toggle(self):
        if not self.show_contour_var.get():
            for p in (self.pane_xy, self.pane_xz, self.pane_yz):
                p.clear_contour()
                p.canvas.draw_idle()

    def _home_extent(self, plane):
        if plane == "xy":
            nrow, ncol = self.h, self.w
        elif plane == "xz":
            nrow, ncol = self.d, self.w
        else:
            nrow, ncol = self.d, self.h
        return (-0.5, ncol - 0.5), (nrow - 0.5, -0.5)

    def _apply_home(self, pane):
        (x0, x1), (y0, y1) = self._home_extent(pane.plane)
        pane.ax.set_xlim(x0, x1)
        pane.ax.set_ylim(y0, y1)

    def reset_views(self):
        if self.volume is None:
            return
        for p in (self.pane_xy, self.pane_xz, self.pane_yz):
            p.clear_contour()
            self._apply_home(p)
            p.canvas.draw_idle()

    # ----- rendering -------------------------------------------------------- #
    def _base_volume(self):
        if self.base_src_var.get() == "transformed" and self.transformed is not None:
            return self.transformed
        return self.volume

    def _base_slice(self, plane, vol):
        if plane == "xy":
            return vol[self.cz, :, :]
        if plane == "xz":
            return vol[:, self.cy, :]
        return vol[:, :, self.cx]

    def _overlay_rgba(self, plane):
        mode = self.seg_mode_var.get()
        if mode == "none" or self.pipe is None:
            return None
        src = self.seg8 if mode == "bitfield" else self.ids
        if src is None:
            return None
        s2d = self._base_slice(plane, src)
        alpha = float(self.alpha_var.get())
        if mode == "bitfield":
            idx = np.clip(s2d.astype(int), 0, 15)
            rgb = _BIT_LUT[idx]
        else:
            idx = np.mod(s2d.astype(int), 20)
            rgb = _LAB_LUT[idx]
        a = np.where(s2d > 0, alpha, 0.0)
        return np.dstack([rgb, a])

    def _crosshair(self, plane):
        # returns (vline_x, hline_y) in the plane's data coordinates
        if plane == "xy":
            return self.cx, self.cy
        if plane == "xz":
            return self.cx, self.cz
        return self.cy, self.cz

    def render_all(self):
        if self.volume is None:
            return
        for p in (self.pane_xy, self.pane_xz, self.pane_yz):
            self._render_pane(p)

    def _render_pane(self, pane):
        vol = self._base_volume()
        base2d = self._base_slice(pane.plane, vol)
        pane.clear_contour()          # stale (belongs to the previous slice)
        show = self.show_slice_var.get()
        vmin, vmax = float(vol.min()), float(vol.max())

        if pane.im_base is None:
            pane.im_base = pane.ax.imshow(base2d, cmap="gray", vmin=vmin, vmax=vmax,
                                          interpolation="nearest", aspect="auto", zorder=0)
        else:
            pane.im_base.set_data(base2d)
            pane.im_base.set_clim(vmin, vmax)
        pane.im_base.set_visible(show)

        rgba = self._overlay_rgba(pane.plane)
        if rgba is None:
            if pane.im_over is not None:
                pane.im_over.set_visible(False)
        else:
            if pane.im_over is None:
                pane.im_over = pane.ax.imshow(rgba, interpolation="nearest", aspect="auto", zorder=1)
            else:
                pane.im_over.set_data(rgba)
            pane.im_over.set_visible(True)

        vx, hy = self._crosshair(pane.plane)
        pane.vline.set_xdata([vx, vx]); pane.vline.set_visible(True)
        pane.hline.set_ydata([hy, hy]); pane.hline.set_visible(True)
        # keep lines and title on top
        for artist in (pane.vline, pane.hline):
            artist.set_zorder(3)
        if not pane.view_initialized:          # fit 1:1 on first draw; preserve zoom/pan after
            self._apply_home(pane)
            pane.view_initialized = True
        pane.canvas.draw_idle()


def main():
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    if initial == "--selftest":
        return _selftest()
    root = tk.Tk()
    CellsegApp(root, initial)
    root.mainloop()


def _selftest():
    """Build the app, load a synthetic stack, run compute+render, no mainloop."""
    import tempfile
    z, y, x = 24, 40, 40
    zz, yy, xx = np.mgrid[0:z, 0:y, 0:x].astype(np.float32)
    vol = np.zeros((z, y, x), np.float32)
    for cz, cy, cx in [(12, 12, 12), (12, 28, 26)]:
        r = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
        vol += np.exp(-((r - 7.0) ** 2) / (2 * 1.5 ** 2))
    tmp = os.path.join(tempfile.gettempdir(), "selftest_%dx%dx%d.raw" % (x, y, z))
    vol.astype("<f4").tofile(tmp)

    root = tk.Tk()
    root.withdraw()
    app = CellsegApp(root, tmp)
    app.blur_var.set("1.5")
    app.pers_pct_var.set("5.0")
    pipe = cellseg.heavy_lift(app.volume, json.dumps({"blur_sigma": 1.5, "persistence_percent": 5.0}))
    app._heavy_done(pipe)               # exercises filtered(), phase-B, render
    app.cut_var.set("0.0"); app.bg_var.set("0.3")
    app._recompute_phaseb()
    app.seg_mode_var.set("bitfield"); app.render_all()
    app.on_double_click("xy", 20, 18)
    app._show_tree()
    app.tree_win.refresh()

    # exercise the interaction handlers with synthetic mouse events
    class _E:
        pass

    def evt(pane, xd, yd, **kw):
        e = _E()
        e.inaxes = pane.ax if xd is not None else None
        e.xdata, e.ydata, e.x, e.y = xd, yd, 100, 100
        e.dblclick, e.button = False, 1
        for k, v in kw.items():
            setattr(e, k, v)
        return e

    p = app.pane_xy
    p.canvas.draw()                          # ensure a renderer for get_window_extent
    app.show_contour_var.set(True)
    app.on_motion(p, evt(p, 20, 18))         # readout + live contour
    assert app.coord_var.get().startswith("value")
    assert p.contour is not None
    app.on_scroll(p, evt(p, 20, 18, button="up"))    # zoom in
    app.on_press(p, evt(p, 20, 18))          # begin pan
    app.on_motion(p, evt(p, 24, 22))         # pan drag
    app.on_release(p, evt(p, 24, 22))
    app.reset_views()
    app.on_leave(p, evt(p, None, None))

    assert app.seg8 is not None and app.ids is not None
    assert app.transformed is not None and app.transformed.shape == app.volume.shape
    print("selftest OK: seg8", app.seg8.shape, "ids", app.ids.shape,
          "membrane", int((app.seg8 & 2 > 0).sum()), "| readout+zoom+pan+contour+reset exercised")
    root.destroy()


if __name__ == "__main__":
    main()
