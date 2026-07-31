"""cellseg interactive Tkinter application.

Left panel:  data selection, heavy-lift parameters, heavy compute + show-tree,
             Phase-B thresholds (recompute on entry/release), save buttons.
Right panel: a resizable tri-planar 2x2 grid (XY | YZ over XZ | controls) with
             physical per-plane aspect, visible axes, segmentation overlays,
             crosshairs, and a control quadrant (base/seg/alpha/contour, Min/Max
             brightness, normalized Z/Y/X sliders, voxel spacing, legend, readout).

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

# Tree traversals are iterative and the tree JSON is flat, so deep trees no
# longer recurse; a modest bump is kept only as a safety margin.
sys.setrecursionlimit(10000)


def _stage(msg):
    """Print a pipeline-stage marker (flushed) for progress/crash localization."""
    print(f"[app] {msg}", flush=True)

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tifffile
from msseg import cellseg

# 16-entry LUT for the seg8 bit-flag overlay.
_TAB20 = matplotlib.colormaps["tab20"]
_BIT_LUT = _TAB20(np.linspace(0, 1, 16))[:, :3]

# Per-minimum palette + merge-tree icicle now live in the shared msseg.viz package.
from msseg.viz import min_color, draw_icicle, _MIN_LUT, _MIN_K  # noqa: F401

_PLANE_META = {"xy": ("XY", "X", "Y", "z"), "xz": ("XZ", "X", "Z", "y"), "yz": ("YZ", "Y", "Z", "x")}


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


def read_tiff_spacing(path):
    """Best-effort (sx, sy, sz) physical voxel spacing from TIFF tags; 1,1,1 fallback."""
    sx = sy = sz = 1.0
    try:
        with tifffile.TiffFile(path) as tf:
            ij = tf.imagej_metadata or {}
            if ij.get("spacing"):
                sz = float(ij["spacing"])
            page = tf.pages[0]

            def _pixsize(tagname):
                tag = page.tags.get(tagname)
                if tag is None:
                    return None
                val = tag.value
                if isinstance(val, tuple) and len(val) == 2 and val[0]:
                    return float(val[1]) / float(val[0])   # (num, den) pixels/unit -> unit/pixel
                try:
                    return 1.0 / float(val) if float(val) else None
                except (TypeError, ValueError):
                    return None

            px, py = _pixsize("XResolution"), _pixsize("YResolution")
            if px and px > 0:
                sx = px
            if py and py > 0:
                sy = py
    except Exception:  # noqa: BLE001
        pass
    return sx, sy, sz


def normalized_index(norm, size):
    """Map a normalized slider value [0,1] to a safe index."""
    if size <= 1:
        return 0
    return int(min(1.0, max(0.0, float(norm))) * (size - 1))


def norm_from_index(idx, size):
    """Map an index to a normalized [0,1] slider value."""
    if size <= 1:
        return 0.0
    return float(max(0, min(size - 1, int(idx)))) / float(size - 1)


def compute_aspects(sx, sy, sz):
    """Per-plane physical aspect (imshow aspect = y-unit / x-unit)."""
    sx, sy, sz = (max(float(v), 1e-8) for v in (sx, sy, sz))
    return {"xy": sy / sx, "xz": sz / sx, "yz": sz / sy}


# --------------------------------------------------------------------------- #
# Tree window (icicle + hover guide)  —  draw_icicle imported from msseg.viz
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
        # Pan / box-zoom / home-reset via the standard toolbar; the huge
        # background branch dwarfs everything, so zoom is how you reach the
        # small features. Scroll-wheel adds cursor-centered zoom on top.
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.top, pack_toolbar=True)
        self.toolbar.update()
        self.hline = None
        self.text = None
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
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
        # Make the toolbar's Home button reset to this freshly-drawn full view.
        self.toolbar.update()
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        # Cursor-centered zoom: scroll up = zoom in, down = zoom out, on both
        # axes, keeping the point under the cursor fixed.
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self.ax.set_xlim(event.xdata - (event.xdata - x0) * scale,
                         event.xdata + (x1 - event.xdata) * scale)
        self.ax.set_ylim(event.ydata - (event.ydata - y0) * scale,
                         event.ydata + (y1 - event.ydata) * scale)
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
# One tri-planar slice view (figure + canvas + events; no slider)
# --------------------------------------------------------------------------- #
class SlicePane:
    def __init__(self, parent, app, plane):
        self.app = app
        self.plane = plane
        self.frame = ttk.Frame(parent)

        self.fig = Figure(figsize=(4, 3))
        self.fig.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.14)
        self.ax = self.fig.add_subplot(111)
        self.ax.tick_params(labelsize=8)
        self.im_base = None
        self.im_over = None
        self.contour = None           # live under-mouse contour (QuadContourSet)
        self.pan = None               # active pan drag state
        self.default_limits = None    # (xlim, ylim) captured on first draw, for Reset
        self.vline = self.ax.axvline(0, ls=":", color="yellow", lw=0.8, alpha=0.85, visible=False)
        self.hline = self.ax.axhline(0, ls=":", color="yellow", lw=0.8, alpha=0.85, visible=False)
        self.tip = self.ax.annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
                                    ha="left", va="bottom", fontsize=8, zorder=10, visible=False,
                                    bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.92))

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", lambda e: app.on_press(self, e))
        self.canvas.mpl_connect("button_release_event", lambda e: app.on_release(self, e))
        self.canvas.mpl_connect("motion_notify_event", lambda e: app.on_motion(self, e))
        self.canvas.mpl_connect("scroll_event", lambda e: app.on_scroll(self, e))
        self.canvas.mpl_connect("axes_leave_event", lambda e: app.on_leave(self, e))

    def clear_contour(self):
        if self.contour is not None:
            try:
                self.contour.remove()
            except Exception:  # noqa: BLE001
                pass
            self.contour = None


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
        self.spacing = [1.0, 1.0, 1.0]   # sx, sy, sz
        # per-persistence caches (ascending labels, cancellation persistence, tree)
        self.asc = None
        self.asc_tree = None        # branch-decomposition ("asc tree") labels
        self.cells = None           # post-cut cell-interior labels (cut-scoped)
        self._cells_bg = 0          # background cell label (heaviest) in `cells`
        self._ids_bg = 0            # background cell label (heaviest) in `ids` (extended)
        self._min_pers = None
        self._tree = None
        self._labels_persistence = None
        self._min_value = None
        self._min_first_merger = None
        # per-cut caches (region colors, background, merge-deaths, min->bg)
        self._region_rgb = None
        self._region_bg = -1
        self._region_death = {}
        self._min_is_bg = None
        self.tree_win = None
        self._busy = False
        self._suspend_slider = False
        self._syncing_sash = False
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
        self.pers_abs_var = tk.StringVar(value="")
        self._labeled_entry(heavy, "Blur sigma", self.blur_var)
        self._labeled_entry(heavy, "Persistence cap", self.pers_abs_var)
        self.heavy_btn = ttk.Button(heavy, text="Heavy compute", command=self._start_heavy)
        self.heavy_btn.pack(fill=tk.X, padx=4, pady=(4, 2))

        # Phase-B thresholds card
        thr = ttk.LabelFrame(left, text="Phase-B thresholds (recompute on Enter)")
        thr.pack(fill=tk.X, padx=6, pady=4)
        self.persel_var = tk.StringVar(value="")
        self.cut_var = tk.StringVar(value="0.0")
        self.bg_var = tk.StringVar(value="0.0")
        e = self._labeled_entry(thr, "Persistence", self.persel_var)
        e.bind("<Return>", lambda _e: self._recompute_phaseb())
        e.bind("<FocusOut>", lambda _e: self._recompute_phaseb())
        # Show tree sits right under the persistence input (it shows that persistence).
        self.tree_btn = ttk.Button(thr, text="Show tree", command=self._show_tree, state=tk.DISABLED)
        self.tree_btn.pack(anchor="e", padx=4, pady=(0, 2))
        for lbl, var in [("Cut threshold", self.cut_var), ("Background", self.bg_var)]:
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
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        rows = ttk.PanedWindow(right, orient=tk.VERTICAL)
        rows.grid(row=0, column=0, sticky="nsew", padx=6, pady=4)
        top_row = ttk.Frame(rows)
        bottom_row = ttk.Frame(rows)
        rows.add(top_row, weight=3)
        rows.add(bottom_row, weight=2)
        for r in (top_row, bottom_row):
            r.rowconfigure(0, weight=1)
            r.columnconfigure(0, weight=1)

        self.top_pane = ttk.PanedWindow(top_row, orient=tk.HORIZONTAL)
        self.top_pane.grid(row=0, column=0, sticky="nsew")
        self.bottom_pane = ttk.PanedWindow(bottom_row, orient=tk.HORIZONTAL)
        self.bottom_pane.grid(row=0, column=0, sticky="nsew")

        self.pane_xy = SlicePane(self.top_pane, self, "xy")
        self.pane_yz = SlicePane(self.top_pane, self, "yz")
        self.pane_xz = SlicePane(self.bottom_pane, self, "xz")
        self.top_pane.add(self.pane_xy.frame, weight=3)
        self.top_pane.add(self.pane_yz.frame, weight=2)
        self.bottom_pane.add(self.pane_xz.frame, weight=3)

        quad = ttk.Frame(self.bottom_pane)
        self.bottom_pane.add(quad, weight=2)
        self._build_controls(quad)

        # Keep the two rows' column dividers aligned (a real 2x2 grid).
        for src, dst in ((self.top_pane, self.bottom_pane), (self.bottom_pane, self.top_pane)):
            self.top_pane.bind("<B1-Motion>", lambda _e: self._mirror_sash(self.top_pane, self.bottom_pane))
            self.bottom_pane.bind("<B1-Motion>", lambda _e: self._mirror_sash(self.bottom_pane, self.top_pane))
            self.top_pane.bind("<ButtonRelease-1>", lambda _e: self._mirror_sash(self.top_pane, self.bottom_pane))
            self.bottom_pane.bind("<ButtonRelease-1>", lambda _e: self._mirror_sash(self.bottom_pane, self.top_pane))
        self.root.after_idle(self._sync_columns_initial)

    def _build_controls(self, quad):
        # base source
        r = ttk.Frame(quad); r.pack(fill=tk.X, padx=4, pady=(6, 2))
        self.show_slice_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(r, text="Show slice", variable=self.show_slice_var,
                        command=self.render_all).pack(side=tk.LEFT)
        self.base_src_var = tk.StringVar(value="original")
        ttk.Radiobutton(r, text="orig", variable=self.base_src_var, value="original",
                        command=self.render_all).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Radiobutton(r, text="transformed", variable=self.base_src_var, value="transformed",
                        command=self.render_all).pack(side=tk.LEFT)

        # segmentation mode
        r = ttk.Frame(quad); r.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(r, text="Seg:").pack(side=tk.LEFT)
        self.seg_mode_var = tk.StringVar(value="none")
        for m, txt in (("none", "none"), ("bitfield", "bitfield"),
                       ("ascman", "asc man"), ("asctree", "asc tree"),
                       ("cells", "cell-interior"), ("labels", "labels")):
            ttk.Radiobutton(r, text=txt, variable=self.seg_mode_var, value=m,
                            command=self._on_seg_mode).pack(side=tk.LEFT, padx=(4, 0))

        # overlay alpha
        r = ttk.Frame(quad); r.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(r, text="Alpha", width=6).pack(side=tk.LEFT)
        self.alpha_var = tk.DoubleVar(value=0.5)
        ttk.Scale(r, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.alpha_var,
                  command=lambda _e: self.render_all()).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # contour + reset
        r = ttk.Frame(quad); r.pack(fill=tk.X, padx=4, pady=2)
        self.show_contour_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r, text="Contour@mouse", variable=self.show_contour_var,
                        command=self._on_contour_toggle).pack(side=tk.LEFT)
        ttk.Button(r, text="Reset view", command=self.reset_views).pack(side=tk.RIGHT)

        # min / max brightness
        self.vmin_var = tk.DoubleVar(value=0.0)
        self.vmax_var = tk.DoubleVar(value=1.0)
        r = ttk.Frame(quad); r.pack(fill=tk.X, padx=4, pady=(4, 1))
        ttk.Label(r, text="Min", width=4).pack(side=tk.LEFT)
        self.vmin_scale = ttk.Scale(r, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                    variable=self.vmin_var, command=lambda _e: self.render_all())
        self.vmin_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        r = ttk.Frame(quad); r.pack(fill=tk.X, padx=4, pady=1)
        ttk.Label(r, text="Max", width=4).pack(side=tk.LEFT)
        self.vmax_scale = ttk.Scale(r, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                    variable=self.vmax_var, command=lambda _e: self.render_all())
        self.vmax_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # normalized Z / Y / X sliders
        self.z_norm = tk.DoubleVar(value=0.5)
        self.y_norm = tk.DoubleVar(value=0.5)
        self.x_norm = tk.DoubleVar(value=0.5)
        self.z_scale = self._slice_slider(quad, "Z", self.z_norm)
        self.y_scale = self._slice_slider(quad, "Y", self.y_norm)
        self.x_scale = self._slice_slider(quad, "X", self.x_norm)

        # voxel spacing (aspect)
        r = ttk.Frame(quad); r.pack(fill=tk.X, padx=4, pady=(4, 2))
        ttk.Label(r, text="Spacing x/y/z").pack(side=tk.LEFT)
        self.sx_var = tk.StringVar(value="1.0")
        self.sy_var = tk.StringVar(value="1.0")
        self.sz_var = tk.StringVar(value="1.0")
        for var in (self.sx_var, self.sy_var, self.sz_var):
            e = ttk.Entry(r, textvariable=var, width=5)
            e.pack(side=tk.LEFT, padx=(4, 0))
            e.bind("<Return>", lambda _e: self._on_spacing_change())
            e.bind("<FocusOut>", lambda _e: self._on_spacing_change())

        # bitfield legend
        self.legend_frame = ttk.Frame(quad)
        self.legend_frame.pack(fill=tk.X, padx=4, pady=(4, 2))

        # value/coord readout
        self.coord_var = tk.StringVar(value="")
        ttk.Label(quad, textvariable=self.coord_var, anchor="e",
                  font=("TkFixedFont", 8)).pack(fill=tk.X, padx=4, pady=(2, 4))

    def _slice_slider(self, quad, label, var):
        r = ttk.Frame(quad); r.pack(fill=tk.X, padx=4, pady=1)
        ttk.Label(r, text=label, width=2).pack(side=tk.LEFT)
        s = ttk.Scale(r, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=var,
                      command=self._on_slice_slider)
        s.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return s

    # ----- column-divider sync (keeps the 2x2 grid aligned) ----------------- #
    def _mirror_sash(self, src, dst):
        if self._syncing_sash:
            return
        try:
            pos = src.sashpos(0)
        except Exception:  # noqa: BLE001
            return
        self._syncing_sash = True
        try:
            dst.sashpos(0, pos)
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._syncing_sash = False

    def _sync_columns_initial(self, attempt=0):
        tw = int(self.top_pane.winfo_width() or 0)
        bw = int(self.bottom_pane.winfo_width() or 0)
        if tw <= 1 or bw <= 1:
            if attempt < 8:
                self.root.after(60, lambda: self._sync_columns_initial(attempt + 1))
            return
        try:
            pos = int(self.top_pane.sashpos(0))
        except Exception:  # noqa: BLE001
            pos = 0
        if pos <= 1:
            pos = int(0.62 * tw)
        self._syncing_sash = True
        try:
            self.top_pane.sashpos(0, pos)
            self.bottom_pane.sashpos(0, pos)
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._syncing_sash = False

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
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Load failed", str(exc))
            return
        self.path_var.set(path)
        self.d, self.h, self.w = self.volume.shape
        self.cz, self.cy, self.cx = self.d // 2, self.h // 2, self.w // 2
        self.pipe = self.transformed = self.seg8 = self.ids = None
        self.asc = self.asc_tree = self.cells = self._tree = self._min_pers = self._region_rgb = None
        self._labels_persistence = None
        self.tree_btn.configure(state=tk.DISABLED)
        self.save_lbl_btn.configure(state=tk.DISABLED)
        self.save_bit_btn.configure(state=tk.DISABLED)
        self.base_src_var.set("original")

        # spacing (TIFF metadata when present)
        if os.path.splitext(path)[1].lower() in (".tif", ".tiff"):
            sx, sy, sz = read_tiff_spacing(path)
        else:
            sx = sy = sz = 1.0
        self.spacing = [sx, sy, sz]
        self.sx_var.set(f"{sx:g}"); self.sy_var.set(f"{sy:g}"); self.sz_var.set(f"{sz:g}")

        # brightness slider ranges from the volume min/max
        vmn, vmx = float(self.volume.min()), float(self.volume.max())
        if vmx <= vmn:
            vmx = vmn + 1.0
        self.vmin_scale.configure(from_=vmn, to=vmx)
        self.vmax_scale.configure(from_=vmn, to=vmx)
        self.vmin_var.set(vmn)
        self.vmax_var.set(vmx)

        self.shape_label.configure(
            text=f"shape (z,y,x) = {self.volume.shape}\nrange [{vmn:.4g}, {vmx:.4g}]  "
                 f"spacing {sx:g}/{sy:g}/{sz:g}")
        self._sync_sliders_from_indices()
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
            p.default_limits = None

    # ----- heavy compute ---------------------------------------------------- #
    def _start_heavy(self):
        if self.volume is None:
            messagebox.showinfo("No data", "Load a volume first.")
            return
        if self._busy:
            return
        try:
            blur = float(self.blur_var.get())
            pers = float(self.pers_abs_var.get())
        except ValueError:
            messagebox.showerror("Bad parameter",
                                 "Blur sigma and Persistence (abs) are required numbers.")
            return
        params = json.dumps({"blur_sigma": blur, "persistence_absolute": pers})
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
        # selection can't exceed the hierarchy cap (the heavy-lift persistence)
        cap = self.pipe.heavy_persistence()
        if persel > cap:
            persel = cap
            self.persel_var.set(f"{cap:.6g}")
        cut = self.get_float("cut")
        bg = self.get_float("bg")
        self.set_status("Segmenting…")
        try:
            persistence_changed = abs(persel - self.pipe.current_persistence()) > 1e-9
            if persistence_changed:
                _stage("set_persistence %.4g" % persel)
                self.pipe.set_persistence(persel)
            _stage("segment (cut=%.4g bg=%.4g)" % (cut, bg))
            seg8, ids = self.pipe.segment(cut, bg)
            if persistence_changed or self.asc is None:
                # persistence-scoped data: ascending labels, per-node cancellation
                # persistence, and the merge tree (all change only with persistence)
                _stage("fetch ascending_labels")
                self.asc = np.asarray(self.pipe.ascending_labels())
                _stage("fetch ascending_tree_labels")
                self.asc_tree = np.asarray(self.pipe.ascending_tree_labels())
                _stage("fetch node_cancellation_persistence")
                self._min_pers = np.asarray(self.pipe.node_cancellation_persistence())
                _stage("fetch + parse merge_tree_json")
                self._tree = json.loads(self.pipe.merge_tree_json())
                _stage("prep tree caches")
                self._prep_tree_caches()
                self._labels_persistence = self.pipe.current_persistence()
                if self.tree_win is not None:
                    self.tree_win.refresh()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Segmentation failed", str(exc))
            self.set_status("Segmentation failed")
            return
        self.seg8 = np.asarray(seg8)
        self.ids = np.asarray(ids)                            # cell interiors extended thru membrane
        self._seg8_present = np.unique(self.seg8)
        self._ids_bg = self._heaviest_label(self.ids)         # background cell in the extended ids
        _stage("fetch cell_labels")
        self.cells = np.asarray(self.pipe.cell_labels(cut))   # cell interiors only (cut-scoped)
        self._cells_bg = self._heaviest_label(self.cells)
        _stage("rebuild labels (re-derive cut)")
        self._rebuild_labels(cut)     # per-cut region colors / deaths / background
        _stage("phase-B done, rendering")
        self.save_lbl_btn.configure(state=tk.NORMAL)
        self.save_bit_btn.configure(state=tk.NORMAL)
        self.set_status(f"Segmented: cut={cut:.4g} bg={bg:.4g} "
                        f"(membrane={int((self.seg8 & 2 > 0).sum())} vox)")
        self._update_legend()
        self.render_all()

    # ----- label palette / cut re-derivation (shared with the tree) --------- #
    def _prep_tree_caches(self):
        """From the flat merge tree: per-min value and the first-merger-above
        value (for the (u) persistence estimate on never-cancelled minima).
        Iterative DFS over the flat node list."""
        nodes = self._tree["nodes"]
        val, first = {}, {}
        stack = [(nid, None) for nid in self._tree["roots"]]
        while stack:
            nid, parent_val = stack.pop()
            nd = nodes[nid]
            kids = nd["children"]
            if not kids:
                val[nd["node_id"]] = nd["value"]
                first[nd["node_id"]] = parent_val
            else:
                for c in kids:
                    stack.append((c, nd["value"]))
        n = (max(val) + 1) if val else 1
        self._min_value = np.full(n, np.nan, np.float32)
        self._min_first_merger = np.full(n, np.nan, np.float32)
        for nid, v in val.items():
            self._min_value[nid] = v
        for nid, fv in first.items():
            if fv is not None:
                self._min_first_merger[nid] = fv

    @staticmethod
    def _branch_survivors(tree, cut, select):
        """Branch decomposition + relabel, mirroring C++ branch_survivors. Each
        minimum-leaf owns a branch that dies at the merger where a deeper sibling
        first appears (leaf-to-merge persistence = merger value - leaf value); the
        branch stays its own feature only when that death is a *barrier* (merger
        value > cut AND persistence >= select), else it folds into the deeper
        branch. Fully iterative. Returns (survivor_of_min, death_of_survivor)
        keyed by minimum node_id (death is the barrier merger value, or None)."""
        nodes = tree["nodes"]
        n = len(nodes)
        lo = [0.0] * n          # deepest leaf value in subtree
        rep = [-1] * n          # node index of that deepest leaf (branch rep)
        st = [(r, False) for r in tree["roots"]]     # post-order
        while st:
            idx, done = st.pop()
            kids = nodes[idx]["children"]
            if not kids:
                lo[idx] = nodes[idx]["value"]; rep[idx] = idx
            elif done:
                best = kids[0]
                for c in kids:
                    if lo[c] < lo[best]:
                        best = c
                lo[idx] = lo[best]; rep[idx] = rep[best]
            else:
                st.append((idx, True))
                for c in kids:
                    st.append((c, False))

        parent_leaf = [-1] * n       # leaf idx -> leaf idx it dies into
        surviving = [False] * n
        death_val = [None] * n       # barrier survivors: the death merger value
        for r in tree["roots"]:
            surviving[rep[r]] = True                 # root branch never dies
        st = list(tree["roots"])                     # pre-order
        while st:
            idx = st.pop()
            kids = nodes[idx]["children"]
            if kids:
                mainrep = rep[idx]
                for c in kids:
                    leafL = rep[c]
                    if leafL != mainrep:             # side branch dies at idx
                        pers = nodes[idx]["value"] - lo[c]
                        parent_leaf[leafL] = mainrep
                        if nodes[idx]["value"] > cut and pers >= select:
                            surviving[leafL] = True
                            death_val[leafL] = nodes[idx]["value"]
                    st.append(c)

        memo = [-1] * n
        survivor_of_min, death_of_survivor = {}, {}
        for i in range(n):
            if nodes[i]["children"]:
                continue
            path, x = [], i
            while not surviving[x]:
                if memo[x] != -1:
                    x = memo[x]; break
                if parent_leaf[x] == -1:
                    break
                path.append(x); x = parent_leaf[x]
            for p in path:
                memo[p] = x
            mn, svmin = nodes[i]["node_id"], nodes[x]["node_id"]
            survivor_of_min[mn] = svmin
            death_of_survivor[svmin] = death_val[x]
        return survivor_of_min, death_of_survivor

    @staticmethod
    def _rederive_cut(tree, cut, select):
        """Relabel-then-cut on the flat tree so region ids match the C++ ids
        volume: fold every sub-`select`-persistence branch into its parent, then
        number the surviving branches into regions in the same pre-order
        first-reference order as C++ cut_regions. Returns region_of_min,
        region_death, region_rep(deepest min), region_voxels, num."""
        nodes = tree["nodes"]
        survivor_of_min, death_of_survivor = CellsegApp._branch_survivors(tree, cut, select)
        region_of_survivor, region_of_min = {}, {}
        region_vox, region_death, region_rep = {}, {}, {}
        counter = 0
        stack = list(reversed(tree["roots"]))
        while stack:
            nid = stack.pop()
            nd = nodes[nid]
            kids = nd["children"]
            if not kids:                            # leaf
                mn = nd["node_id"]
                sv = survivor_of_min.get(mn, mn)
                rid = region_of_survivor.get(sv)
                if rid is None:
                    rid = counter
                    counter += 1
                    region_of_survivor[sv] = rid
                    region_rep[rid] = sv            # survivor = deepest min of the feature
                    region_death[rid] = death_of_survivor.get(sv)
                region_of_min[mn] = rid
                region_vox[rid] = region_vox.get(rid, 0) + nd["voxel_count"]
            else:
                for c in reversed(kids):
                    stack.append(c)
        return region_of_min, region_death, region_rep, region_vox, counter

    @staticmethod
    def _heaviest_label(vol):
        """The most common non-zero label in a volume (the background cell)."""
        flat = vol.ravel()
        nz = flat[flat > 0]
        return int(np.bincount(nz).argmax()) if nz.size else 0

    def _rebuild_labels(self, cut):
        if self._tree is None:
            return
        # relabel threshold = the persistence this view/tree was taken at
        select = self._labels_persistence if self._labels_persistence is not None else 0.0
        region_of_min, region_death, region_rep, region_vox, num = \
            self._rederive_cut(self._tree, cut, select)
        # background = largest region by summed voxel count (matches C++)
        bg, best = -1, -1
        for r, v in region_vox.items():
            if v > best:
                best, bg = v, r
        self._region_bg = bg
        self._region_death = region_death
        # region -> color via its deepest (representative) minimum
        rgb = np.full((max(num, 1), 3), 0.5, np.float32)
        for r, rep in region_rep.items():
            rgb[r] = _MIN_LUT[int(rep) % _MIN_K]
        self._region_rgb = rgb
        # min NodeId -> in background region (to suppress it in the asc-man overlay)
        n = self._min_value.shape[0] if self._min_value is not None else 1
        mbg = np.zeros(n, bool)
        for nid, r in region_of_min.items():
            if nid < n and r == bg:
                mbg[nid] = True
        self._min_is_bg = mbg

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

    # ----- slider / spacing / interaction ----------------------------------- #
    def _on_slice_slider(self, _v=None):
        if self._suspend_slider or self.volume is None:
            return
        self.cz = normalized_index(self.z_norm.get(), self.d)
        self.cy = normalized_index(self.y_norm.get(), self.h)
        self.cx = normalized_index(self.x_norm.get(), self.w)
        self.render_all()

    def _sync_sliders_from_indices(self):
        self._suspend_slider = True
        try:
            self.z_norm.set(norm_from_index(self.cz, self.d))
            self.y_norm.set(norm_from_index(self.cy, self.h))
            self.x_norm.set(norm_from_index(self.cx, self.w))
        finally:
            self._suspend_slider = False

    def _on_spacing_change(self):
        def f(var, cur):
            try:
                return max(1e-6, float(var.get()))
            except ValueError:
                return cur
        self.spacing = [f(self.sx_var, self.spacing[0]), f(self.sy_var, self.spacing[1]),
                        f(self.sz_var, self.spacing[2])]
        self.render_all()

    def _plane_aspect(self, plane):
        return compute_aspects(*self.spacing)[plane]

    def on_double_click(self, plane, xdata, ydata):
        xi, yi = int(round(xdata)), int(round(ydata))
        if plane == "xy":            # sets x (YZ) and y (XZ)
            self.cx = int(np.clip(xi, 0, self.w - 1))
            self.cy = int(np.clip(yi, 0, self.h - 1))
        elif plane == "xz":          # sets x (YZ) and z (XY)
            self.cx = int(np.clip(xi, 0, self.w - 1))
            self.cz = int(np.clip(yi, 0, self.d - 1))
        elif plane == "yz":          # sets y (XZ) and z (XY)
            self.cy = int(np.clip(xi, 0, self.h - 1))
            self.cz = int(np.clip(yi, 0, self.d - 1))
        self._sync_sliders_from_indices()
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
        pane.tip.set_visible(False)
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
            x, y, z = self._coord_for(pane.plane, col, rowi)
            self.coord_var.set(f"(x={x}, y={y}, z={z})")
            pane.tip.xy = (event.xdata, event.ydata)
            pane.tip.set_text(self._tip_text(x, y, z))
            pane.tip.set_visible(True)
            if self.show_contour_var.get() and pane.pan is None:
                self._update_contour(pane, base2d, float(base2d[rowi, col]))
            pane.canvas.draw_idle()
        else:
            self.coord_var.set("")
            pane.tip.set_visible(False)
            pane.canvas.draw_idle()

    def _tip_text(self, x, y, z):
        lines = [f"orig={float(self.volume[z, y, x]):.4g}"]
        if self.transformed is not None:
            lines.append(f"xform={float(self.transformed[z, y, x]):.4g}")
        mode = self.seg_mode_var.get()
        if mode == "bitfield" and self.seg8 is not None:
            v = int(self.seg8[z, y, x])
            lines.append(f"bits={v} ({self._bit_label(v)})")
        elif mode == "labels" and self.ids is not None:
            lab = int(self.ids[z, y, x])
            if lab <= 0:
                lines.append("cell=—")
            else:
                lines.append(f"cell={lab - 1}" + ("  (bg)" if lab == self._ids_bg else ""))
        elif mode == "ascman" and self.asc is not None:
            lab = int(self.asc[z, y, x])
            if lab <= 0:
                lines.append("min=—")
            else:
                nid = lab - 1
                p = (self._min_pers[nid] if self._min_pers is not None
                     and nid < self._min_pers.shape[0] else np.nan)
                if np.isfinite(p):
                    lines.append(f"min={nid}  pers={float(p):.4g}")
                else:                                   # never cancelled -> estimate (u)
                    est = np.nan
                    if (self._min_first_merger is not None and nid < self._min_first_merger.shape[0]):
                        fm, mv = self._min_first_merger[nid], self._min_value[nid]
                        if np.isfinite(fm) and np.isfinite(mv):
                            est = fm - mv
                    lines.append(f"min={nid}  pers={est:.4g} (u)" if np.isfinite(est)
                                 else f"min={nid}  pers=— (u)")
        elif mode == "asctree" and self.asc_tree is not None:
            lab = int(self.asc_tree[z, y, x])
            lines.append("branch=—" if lab <= 0 else f"branch={lab - 1}")
        elif mode == "cells" and self.cells is not None:
            lab = int(self.cells[z, y, x])
            lines.append("cell=—" if lab <= 0 else f"cell={lab - 1}")
        return "\n".join(lines)

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

    def reset_views(self):
        if self.volume is None:
            return
        for p in (self.pane_xy, self.pane_xz, self.pane_yz):
            p.clear_contour()
            if p.default_limits is not None:
                p.ax.set_xlim(p.default_limits[0])
                p.ax.set_ylim(p.default_limits[1])
            p.canvas.draw_idle()

    # ----- bitfield legend -------------------------------------------------- #
    @staticmethod
    def _bit_label(v):
        names = []
        if v & 1:
            names.append("asc")
        if v & 2:
            names.append("clean-mem")
        if v & 4:
            names.append("mem-region")
        if v & 8:
            names.append("fg")
        return "+".join(names) if names else "none"

    def _on_seg_mode(self):
        self._update_legend()
        self.render_all()

    def _update_legend(self):
        for w in self.legend_frame.winfo_children():
            w.destroy()
        if self.seg_mode_var.get() != "bitfield" or self.seg8 is None:
            return
        vals = getattr(self, "_seg8_present", None)
        if vals is None:
            vals = np.unique(self.seg8)
        ttk.Label(self.legend_frame, text="bitfield:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        col, rowi = 0, 0
        for v in vals:
            v = int(v)
            if v == 0:
                continue
            rgb = _BIT_LUT[min(v, 15)]
            hexc = "#%02x%02x%02x" % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            tk.Label(self.legend_frame, text="  ", background=hexc, relief="solid",
                     borderwidth=1).grid(row=rowi + 1, column=col * 2, sticky="w", padx=(0, 2), pady=1)
            ttk.Label(self.legend_frame, text=f"{v}: {self._bit_label(v)}",
                      font=("TkDefaultFont", 8)).grid(row=rowi + 1, column=col * 2 + 1,
                                                      sticky="w", padx=(0, 8))
            col += 1
            if col >= 2:          # 2 swatches per row (narrow quadrant)
                col, rowi = 0, rowi + 1

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
        alpha = float(self.alpha_var.get())

        if mode == "bitfield":
            if self.seg8 is None:
                return None
            s2d = self._base_slice(plane, self.seg8)
            rgb = _BIT_LUT[np.clip(s2d.astype(int), 0, 15)]
            a = np.where(s2d > 0, alpha, 0.0)

        elif mode == "labels":                       # cell interiors EXTENDED thru membranes
            if self.ids is None:
                return None
            s2d = self._base_slice(plane, self.ids)            # extended cell rep NodeId + 1
            nid = np.maximum(s2d.astype(int) - 1, 0)           # shared per-minimum palette
            rgb = _MIN_LUT[nid % _MIN_K]
            a = np.where((s2d > 0) & (s2d != self._ids_bg), alpha, 0.0)  # background transparent

        elif mode == "ascman":                       # per-minimum color (full decomposition)
            if self.asc is None:
                return None
            s2d = self._base_slice(plane, self.asc)            # min NodeId + 1
            nid = np.maximum(s2d.astype(int) - 1, 0)           # wrap (not clip): NodeIds far exceed _MIN_K
            rgb = _MIN_LUT[nid % _MIN_K]
            a = np.where(s2d > 0, alpha, 0.0)                  # only unlabeled voxels transparent

        elif mode == "asctree":                      # branch-decomposition color (relabeled)
            if self.asc_tree is None:
                return None
            s2d = self._base_slice(plane, self.asc_tree)       # surviving-branch min NodeId + 1
            nid = np.maximum(s2d.astype(int) - 1, 0)           # same palette as the tree / asc man
            rgb = _MIN_LUT[nid % _MIN_K]
            a = np.where(s2d > 0, alpha, 0.0)

        elif mode == "cells":                        # post-cut cell absorption (shared palette)
            if self.cells is None:
                return None
            s2d = self._base_slice(plane, self.cells)          # final cell's deepest-min NodeId + 1
            nid = np.maximum(s2d.astype(int) - 1, 0)
            rgb = _MIN_LUT[nid % _MIN_K]
            a = np.where(s2d > 0, alpha, 0.0)

        else:
            return None
        return np.dstack([rgb, a])

    def _crosshair(self, plane):
        # (vline_x, hline_y) in the plane's data coordinates
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
        vmin = float(self.vmin_var.get())
        vmax = max(vmin + 1e-6, float(self.vmax_var.get()))

        if pane.im_base is None:
            pane.im_base = pane.ax.imshow(base2d, cmap="gray", vmin=vmin, vmax=vmax,
                                          origin="lower", interpolation="nearest",
                                          aspect="auto", zorder=0)
            pane.default_limits = (pane.ax.get_xlim(), pane.ax.get_ylim())
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
                pane.im_over = pane.ax.imshow(rgba, origin="lower", interpolation="nearest",
                                              aspect="auto", zorder=1)
            else:
                pane.im_over.set_data(rgba)
            pane.im_over.set_visible(True)

        vx, hy = self._crosshair(pane.plane)
        pane.vline.set_xdata([vx, vx]); pane.vline.set_visible(True); pane.vline.set_zorder(3)
        pane.hline.set_ydata([hy, hy]); pane.hline.set_visible(True); pane.hline.set_zorder(3)

        title, xl, yl, il = _PLANE_META[pane.plane]
        idx = {"xy": self.cz, "xz": self.cy, "yz": self.cx}[pane.plane]
        pane.ax.set_title(f"{title}  {il}={idx}", fontsize=9)
        pane.ax.set_xlabel(xl, fontsize=8)
        pane.ax.set_ylabel(yl, fontsize=8)
        pane.ax.set_aspect(self._plane_aspect(pane.plane), adjustable="box")
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
    app.pers_abs_var.set("0.05")
    pipe = cellseg.heavy_lift(app.volume, json.dumps({"blur_sigma": 1.5, "persistence_absolute": 0.05}))
    app._heavy_done(pipe)               # exercises filtered(), asc labels, cancellation pers, render
    app.cut_var.set("0.0"); app.bg_var.set("0.3")
    app._recompute_phaseb()
    app.seg_mode_var.set("bitfield"); app._on_seg_mode()
    assert app.legend_frame.winfo_children(), "bitfield legend should be populated"

    # unified palette + asc-man mode + hover-tip text
    assert app.asc is not None and app._min_pers is not None and app._region_rgb is not None
    app.seg_mode_var.set("labels"); app._on_seg_mode()
    assert app._overlay_rgba("xy") is not None
    app.seg_mode_var.set("ascman"); app._on_seg_mode()
    assert app._overlay_rgba("xy") is not None
    tip = app._tip_text(app.cx, app.cy, app.cz)
    assert "orig=" in tip and "xform=" in tip and "min=" in tip
    app.seg_mode_var.set("labels")
    assert "cell=" in app._tip_text(app.cx, app.cy, app.cz)   # labels now uses the cells labeling

    # branch-decomposition "asc tree" overlay + tip
    assert app.asc_tree is not None
    app.seg_mode_var.set("asctree"); app._on_seg_mode()
    assert app._overlay_rgba("xy") is not None
    assert "branch=" in app._tip_text(app.cx, app.cy, app.cz)

    # post-cut "cells" absorption overlay + tip
    assert app.cells is not None
    app.seg_mode_var.set("cells"); app._on_seg_mode()
    assert app._overlay_rgba("xy") is not None
    assert "cell=" in app._tip_text(app.cx, app.cy, app.cz)

    # exercise the new layout controls
    app.sx_var.set("1"); app.sy_var.set("1"); app.sz_var.set("3"); app._on_spacing_change()
    assert abs(app._plane_aspect("xz") - 3.0) < 1e-6
    app.vmin_var.set(0.1); app.vmax_var.set(0.9); app.render_all()
    app.z_norm.set(0.3); app._on_slice_slider()
    assert app.cz == normalized_index(0.3, app.d)
    app.on_double_click("xy", 20, 18)
    assert app.pane_xy.default_limits is not None
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
    app.on_motion(p, evt(p, 20, 18))         # readout + hover tip + live contour
    assert app.coord_var.get().startswith("(x=")
    assert p.tip.get_visible() and p.tip.get_text().startswith("orig=")
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
          "membrane", int((app.seg8 & 2 > 0).sum()),
          "| 2x2 grid + aspect + minmax + normalized sliders + interaction exercised")
    root.destroy()


if __name__ == "__main__":
    main()
