"""mscoupon interactive labeler: annotate MSC regions with classes.

A fork of the viewer (``mscoupon-gui``) for fast class annotation: arm a class
(button or numeric hotkey), then draw over the slice --

    squiggle   every living MSC region under the polyline gets the class
    box        every region intersecting the dragged rectangle
    lasso      every region under the filled (auto-closed) drag polygon

Class 0 is "no label"; while it is armed the left mouse button pans as in the
viewer (middle-drag always pans). Each gesture is an *interaction*, listed in
its class's subpanel on the right; interactions resolve in creation order (a
later gesture paints over an earlier one), can be dragged between class
subpanels (or moved via right-click), and are stored as raw geometry -- so
changing persistence and hitting Rerun re-resolves them against the new
regions for free.

The class layer renders through the canvas's label-overlay path: one RGBA LUT
per slice (region id -> class color), rebuilt only when an interaction or the
segmentation commit changes, gathered over the visible crop at render time.

Everything else -- sequences, filter chains, statistics, priming, persistence,
config export, session autosave -- is inherited from the viewer. The exported
folder additionally receives ``labels.json`` (the raw interactions), and the
session autosaves under its own file (``mscoupon-labeler``), never the
viewer's.

Run:  mscoupon-labeler [folder]     |     mscoupon-labeler --selftest
"""
from __future__ import annotations

import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog

from . import config_io
from .app import MscouponApp
from .common import log
from .widgets import ScrollFrame
from .labeling import (LabelStore, MAX_CLASSES, TOOLS, class_color_hex,
                       resolve_slice, class_lut)

# How faint the inherited region overlay is drawn under the class layer
# (0..255); the class colors themselves stay fully opaque in the LUT and are
# scaled by the shared alpha slider like every overlay.
_REGION_ALPHA = 90

_TOOL_LABELS = (("squiggle", "squiggle"), ("box", "box"), ("polygon", "lasso"))

# Focus-widget classes whose keystrokes must not arm classes (typing "1" into
# the persistence entry is not a request to arm class 1).
_TYPING_CLASSES = ("Entry", "TEntry", "Spinbox", "TSpinbox", "TCombobox",
                   "Listbox", "Text")


class DrawController:
    """The canvas drawing tool (SliceCanvas.tool): claims button-1 while a
    class is armed, collects the gesture in IMAGE coordinates (floats -- so a
    box drawn zoomed-out stays accurate), and draws its own rubber-band as
    canvas items tagged "draw" in screen coordinates, recomputed from the
    stored image points each move so zooming mid-drag stays consistent."""

    MIN_SCREEN_PX = 3      # squiggle/lasso point spacing (screen px)

    def __init__(self, app):
        self.app = app
        self._pts = None           # image-coord points of the gesture in flight
        self._last_screen = None

    def _image_pt(self, e):
        v = self.app.viewer
        return (v.view_x + e.x * v.scale, v.view_y + e.y * v.scale)

    def on_press(self, e):
        if self.app.active_class_var.get() <= 0:
            return False           # no class armed -> pan as in the viewer
        if self.app._current() is None:
            return False           # nothing on screen to label
        self._pts = [self._image_pt(e)]
        self._last_screen = (e.x, e.y)
        return True

    def on_move(self, e):
        if self._pts is None:
            return False
        if self.app.tool_var.get() == "box":
            # A box is its two corners: the anchor plus the live corner.
            if len(self._pts) > 1:
                self._pts[-1] = self._image_pt(e)
            else:
                self._pts.append(self._image_pt(e))
        else:
            lx, ly = self._last_screen
            if abs(e.x - lx) + abs(e.y - ly) >= self.MIN_SCREEN_PX:
                self._pts.append(self._image_pt(e))
                self._last_screen = (e.x, e.y)
        self._draw_feedback()
        return True

    def on_release(self, e):
        if self._pts is None:
            return False
        pts, self._pts = self._pts, None
        self.app.viewer.canvas.delete("draw")
        tool = self.app.tool_var.get()
        # A single-click tap IS a squiggle (the polyline's start point is always
        # part of the gesture); box/lasso need an actual drag to mean anything.
        need = 1 if tool == "squiggle" else 2
        if len(pts) >= need:
            self.app._commit_interaction(tool, pts)
        return True

    def _draw_feedback(self):
        v = self.app.viewer
        c = v.canvas
        c.delete("draw")
        cls = self.app.active_class_var.get()
        color = class_color_hex(cls) if 0 < cls < MAX_CLASSES else "#ffffff"
        scr = [((x - v.view_x) / v.scale, (y - v.view_y) / v.scale)
               for x, y in self._pts]
        if len(scr) < 2:
            return
        if self.app.tool_var.get() == "box":
            (x0, y0), (x1, y1) = scr[0], scr[-1]
            c.create_rectangle(x0, y0, x1, y1, outline=color, width=2, tags="draw")
        else:
            flat = [coord for pt in scr for coord in pt]
            c.create_line(*flat, fill=color, width=2, tags="draw")
            if self.app.tool_var.get() == "polygon":
                # Preview the auto-close edge.
                c.create_line(scr[-1][0], scr[-1][1], scr[0][0], scr[0][1],
                              fill=color, width=1, dash=(3, 2), tags="draw")


class LabelerApp(MscouponApp):
    SESSION_APP = "mscoupon-labeler"

    def __init__(self, root, initial=None, autosave=True):
        # Labeler state first: the base __init__ calls the overridden build
        # methods, which read these.
        self.store = LabelStore()
        self._class_luts = {}      # (si, li) -> (commit, store.rev, lut|None)
        self.active_class_var = tk.IntVar(master=root, value=0)
        self.tool_var = tk.StringVar(master=root, value="squiggle")
        self.show_regions_var = tk.BooleanVar(master=root, value=True)
        self.n_classes_var = tk.IntVar(master=root, value=self.store.n_classes)
        self._drag_uid = None      # interaction row being dragged between classes
        self._drag_origin = None   # (x_root, y_root) at press: distinguishes click vs drag
        self._drop_panel = None    # class panel currently highlighted as target
        self._panel_relief = "flat"
        # Undo/redo over store mutations (add/delete/move/class-count), as
        # whole-store snapshots: the store is small and a snapshot restore
        # reuses the load path, so history can never drift from reality.
        self._undo_stack = []
        self._redo_stack = []
        super().__init__(root, initial=initial, autosave=autosave)
        root.title("mscoupon labeler")
        self._build_label_panel()
        if self.viewer is not None:
            self.viewer.tool = DrawController(self)
        self._bind_hotkeys()

    # ------------------------------------------------------------------ #
    # Right-side overrides (viewer area is inherited unchanged)
    # ------------------------------------------------------------------ #
    def _build_image_controls(self, parent):
        chan = ttk.Frame(parent); chan.pack(fill="x")
        ttk.Label(chan, text="Image:").pack(side="left", padx=(4, 0))
        self.background_var = tk.StringVar(value="base")
        self.background_combo = ttk.Combobox(chan, textvariable=self.background_var,
                                             values=["base", "filtered"], state="readonly",
                                             width=18)
        self.background_combo.pack(side="left", padx=4)
        self.background_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_render())
        # One toggle instead of the viewer's five seg sources: the labeler works
        # on the full MSC labeling ("msc"), shown faintly under the class layer.
        # seg_source stays a valid viewer value so _needed_level() is always
        # "slice" and every inherited path keeps working; the mask is never on.
        ttk.Checkbutton(chan, text="regions", variable=self.show_regions_var,
                        command=self._on_regions_toggle).pack(side="left", padx=(12, 4))
        self.mask_var = tk.BooleanVar(value=False)
        self.seg_source_var.set("msc" if self.show_regions_var.get() else "none")

    def _on_regions_toggle(self):
        self.seg_source_var.set("msc" if self.show_regions_var.get() else "none")
        self._on_seg_source_change()

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

        # per-channel windowing (same pairs as the viewer)
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

        # persistence: region identity depends on it, so it stays adjustable;
        # commit is the Rerun button exactly as in the viewer.
        row = ttk.Frame(live); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Persistence %:").pack(side="left")
        self.persist_entry = ttk.Entry(row, textvariable=self.persist_live_var, width=8)
        self.persist_entry.pack(side="left", padx=4)
        self.persist_entry.bind("<Return>", self._on_persistence_change)
        self.persist_entry.bind("<FocusOut>", self._on_persistence_change)
        self.persist_value_label = ttk.Label(row, text="")
        self.persist_value_label.pack(side="left", padx=4)

        # No per-slice selection / pixel trim / connectivity: the labeler works
        # on the full pre-filter MSC labeling (the guarded _rebuild_*_cards
        # no-op without their frames).
        self.rerun_btn = ttk.Button(live, text="Rerun selection", state="disabled",
                                    command=self._rerun_selection)
        self.rerun_btn.pack(fill="x", padx=4, pady=(2, 4))

    # ------------------------------------------------------------------ #
    # Rendering: the class layer
    # ------------------------------------------------------------------ #
    def _seg_overlays(self, si, li, rec, data, np, min_colors):
        overlays = super()._seg_overlays(si, li, rec, data, np, min_colors)
        # The inherited region overlay is orientation, not the point: fade it
        # under the class layer (copy first -- _id_lut results may be shared).
        for o in overlays:
            if "lut" in o:
                o["lut"] = o["lut"].copy()
                o["lut"][:, 3] = np.minimum(o["lut"][:, 3], _REGION_ALPHA)
        if rec is not None:
            lut = self._class_lut_for(si, li, rec, np)
            if lut is not None:
                overlays.append({"labels": rec["labels"], "lut": lut,
                                 "visible": True})
        return overlays

    def _class_lut_for(self, si, li, rec, np):
        """The slice's region->class-color LUT, memoized on (commit, store.rev):
        a gesture bumps rev (rebuild the LUT only), a Rerun bumps the commit (a
        fresh labels raster arrives and the interactions re-resolve against it
        -- which is how annotations survive persistence changes)."""
        key = (si, li)
        commit = rec.get("commit")
        cached = self._class_luts.get(key)
        if cached is not None and cached[0] == commit and cached[1] == self.store.rev:
            return cached[2]
        lut = None
        slice_key = self._slice_key(si, li)
        if slice_key is not None:
            its = self.store.for_slice(slice_key)
            if its:
                lut = class_lut(resolve_slice(its, rec["labels"], np), np)
        self._class_luts[key] = (commit, self.store.rev, lut)
        return lut

    def _slice_key(self, si, li):
        try:
            return os.path.basename(self.subsequences[si]["files"][li])
        except (IndexError, KeyError, TypeError):
            return None

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
        self.root.bind("<Escape>", lambda e: self.active_class_var.set(0))
        self.root.bind("<Control-z>", self._on_undo_key)
        self.root.bind("<Control-y>", self._on_undo_key)

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
    # The label panel (third pane)
    # ------------------------------------------------------------------ #
    def _build_label_panel(self):
        # A plain frame, not a ScrollFrame: the class subpanels must DIVIDE the
        # available height between them (each scrolls its own interaction list),
        # which needs the holder to fill the pane rather than grow it.
        self.label_pane = ttk.Frame(self.paned, width=300)
        self.paned.add(self.label_pane, weight=0)
        panel = self.label_pane

        row = ttk.Frame(panel); row.pack(side="top", fill="x", padx=4, pady=(6, 2))
        ttk.Label(row, text="Classes:").pack(side="left")
        self.n_classes_spin = ttk.Spinbox(row, from_=2, to=MAX_CLASSES,
                                          textvariable=self.n_classes_var,
                                          width=4, state="readonly",
                                          command=self._on_n_classes_change)
        self.n_classes_spin.pack(side="left", padx=4)
        ttk.Label(row, text="(class 0 = no label)").pack(side="left", padx=4)

        row = ttk.Frame(panel); row.pack(side="top", fill="x", padx=4, pady=2)
        ttk.Label(row, text="Tool:").pack(side="left")
        for value, txt in _TOOL_LABELS:
            ttk.Radiobutton(row, text=txt, variable=self.tool_var,
                            value=value).pack(side="left", padx=2)

        # Bottom rows pack first (side="bottom") so the classes holder takes
        # exactly the remaining cavity.
        row = ttk.Frame(panel); row.pack(side="bottom", fill="x", padx=4, pady=(2, 8))
        ttk.Button(row, text="Export as CSV…",
                   command=self._export_csv).pack(fill="x")
        row = ttk.Frame(panel); row.pack(side="bottom", fill="x", padx=4, pady=2)
        ttk.Button(row, text="Save labels…",
                   command=self._save_labels).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(row, text="Load labels…",
                   command=self._load_labels).pack(side="left", fill="x", expand=True, padx=(2, 0))

        self.classes_holder = ttk.Frame(panel)
        self.classes_holder.pack(side="top", fill="both", expand=True, padx=2, pady=4)
        self.classes_holder.columnconfigure(0, weight=1)
        self._class_panels = {}          # class_id -> LabelFrame (drop targets)

        self._rebuild_class_panels()

    def _on_n_classes_change(self):
        try:
            n = int(self.n_classes_var.get())
        except (tk.TclError, ValueError):
            return
        if n == self.store.n_classes:
            return
        self._push_history()
        changed = self.store.set_n_classes(n)
        if self.active_class_var.get() >= n:
            self.active_class_var.set(0)
        if changed:
            self.status_var.set(f"{len(changed)} interaction(s) moved to class {n - 1}")
        self._rebuild_class_panels()
        self._refresh_render()

    def _rebuild_class_panels(self):
        """Full repaint from the store (interaction counts are small).

        The subpanels split the holder's height equally (uniform grid rows);
        each one scrolls its own interaction list, so a class with many
        gestures never pushes the others off screen."""
        for w in list(self.classes_holder.winfo_children()):
            w.destroy()
        self._class_panels = {}
        for k in range(1, self.store.n_classes):
            frame = ttk.LabelFrame(self.classes_holder, text=f"Class {k}")
            self.classes_holder.rowconfigure(k - 1, weight=1, uniform="cls")
            frame.grid(row=k - 1, column=0, sticky="nsew", padx=2, pady=3)
            self._class_panels[k] = frame
            if k == 1:
                self._panel_relief = str(frame.cget("relief"))
            color = class_color_hex(k)
            tk.Radiobutton(frame, text=f"draw (key {k})",
                           variable=self.active_class_var, value=k,
                           indicatoron=False, selectcolor=color,
                           bg="#e8e8e8", activebackground=color,
                           ).pack(side="top", fill="x", padx=4, pady=(2, 4))
            lst = ScrollFrame(frame, width=240, canvas_width=224)
            # Small minimum height: the grid's equal weights own the real size.
            lst.canvas.configure(height=48)
            lst.pack(side="top", fill="both", expand=True, padx=2, pady=(0, 2))
            for it in self.store.interactions:
                if it.class_id == k:
                    self._build_interaction_row(lst.inner, it)

    def _build_interaction_row(self, parent, it):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=6, pady=1)
        where = it.slice_key if it.bound else f"{it.slice_key} (unbound)"
        lbl = ttk.Label(row, text=f"#{it.uid} {it.tool}  [{where}]",
                        foreground=("#000" if it.bound else "#888"),
                        cursor="hand2")
        lbl.pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="✕", width=2,
                   command=lambda uid=it.uid: self._delete_interaction(uid)
                   ).pack(side="right")
        # Drag a row onto another class's subpanel to reassign it; a stationary
        # release is a CLICK, which recenters the view on the gesture.
        # Right-click offers move/delete as a menu (drag fallback, faster in
        # bulk); hovering shows the gesture's geometry on the canvas.
        lbl.bind("<ButtonPress-1>", lambda e, uid=it.uid: self._row_drag_start(e, uid))
        lbl.bind("<B1-Motion>", self._row_drag_motion)
        lbl.bind("<ButtonRelease-1>", self._row_drag_drop)
        lbl.bind("<Button-3>", lambda e, uid=it.uid: self._row_menu(e, uid))
        lbl.bind("<Enter>", lambda e, uid=it.uid: self._show_interaction_geometry(uid))
        lbl.bind("<Leave>", lambda e: self._hide_interaction_geometry())

    # -- row drag-and-drop between class subpanels ---------------------- #
    def _panel_under_pointer(self, e):
        w = self.root.winfo_containing(e.x_root, e.y_root)
        while w is not None:
            for k, frame in self._class_panels.items():
                if w is frame:
                    return k, frame
            w = getattr(w, "master", None)
        return None, None

    def _row_drag_start(self, e, uid):
        self._drag_uid = uid
        self._drag_origin = (e.x_root, e.y_root)

    def _row_drag_motion(self, e):
        if self._drag_uid is None:
            return
        _k, frame = self._panel_under_pointer(e)
        if frame is not self._drop_panel:
            if self._drop_panel is not None:
                try:
                    self._drop_panel.configure(relief=self._panel_relief)
                except tk.TclError:
                    pass
            self._drop_panel = frame
            if frame is not None:
                frame.configure(relief="ridge")

    def _row_drag_drop(self, e):
        uid, self._drag_uid = self._drag_uid, None
        origin, self._drag_origin = self._drag_origin, None
        if self._drop_panel is not None:
            try:
                self._drop_panel.configure(relief=self._panel_relief)
            except tk.TclError:
                pass
        self._drop_panel = None
        if uid is None:
            return
        # A release that never really moved is a click, not a drop.
        if origin is not None and (abs(e.x_root - origin[0]) +
                                   abs(e.y_root - origin[1])) < 5:
            self._on_row_click(uid)
            return
        k, _frame = self._panel_under_pointer(e)
        it = self.store.get(uid)
        if k is not None and it is not None and k != it.class_id:
            self._move_interaction(uid, k)

    def _row_menu(self, e, uid):
        it = self.store.get(uid)
        if it is None:
            return
        menu = tk.Menu(self.root, tearoff=0)
        for k in range(1, self.store.n_classes):
            if k != it.class_id:
                menu.add_command(label=f"Move to class {k}",
                                 command=lambda k=k: self._move_interaction(uid, k))
        menu.add_separator()
        menu.add_command(label="Delete", command=lambda: self._delete_interaction(uid))
        menu.tk_popup(e.x_root, e.y_root)

    # -- interaction geometry on the canvas (hover + click-to-center) ---- #
    def _show_interaction_geometry(self, uid):
        """Draw the gesture's geometry fully opaque over the slice: the
        polyline itself, or the outer boundary of a box / lasso. Only when the
        gesture's slice is the one on screen (clicking the row navigates)."""
        it = self.store.get(uid)
        v = self.viewer
        if it is None or v is None or not it.points or not it.bound:
            return
        if self._current() != (it.si, it.li):
            return
        c = v.canvas
        c.delete("ihover")
        color = class_color_hex(it.class_id)
        scr = [((x - v.view_x) / v.scale, (y - v.view_y) / v.scale)
               for x, y in it.points]
        tags = ("draw", "ihover")        # "draw" keeps it above re-blits
        if it.tool == "box" and len(scr) >= 2:
            (x0, y0), (x1, y1) = scr[0], scr[-1]
            c.create_rectangle(x0, y0, x1, y1, outline=color, width=3, tags=tags)
        elif len(scr) >= 2:
            pts = scr + [scr[0]] if it.tool == "polygon" else scr
            flat = [coord for pt in pts for coord in pt]
            c.create_line(*flat, fill=color, width=3, tags=tags)
        else:                            # a single-tap squiggle
            x, y = scr[0]
            c.create_oval(x - 4, y - 4, x + 4, y + 4, outline=color, width=3,
                          tags=tags)

    def _hide_interaction_geometry(self):
        if self.viewer is not None:
            self.viewer.canvas.delete("ihover")

    def _on_row_click(self, uid):
        """Center the view on the gesture's centroid at the current zoom,
        navigating to its slice first when it lives on another one."""
        it = self.store.get(uid)
        v = self.viewer
        if it is None or v is None or not it.points or not it.bound:
            return
        if self._current() != (it.si, it.li):
            try:
                idx = self.flat_slices.index((it.si, it.li))
            except ValueError:
                return
            # Setting the scale drives _on_slice_change (slider sync, per-slice
            # assembly request, re-render) exactly like a user drag.
            self.slice_scale.set(idx)
        cx = sum(x for x, _y in it.points) / len(it.points)
        cy = sum(y for _x, y in it.points) / len(it.points)
        w = max(v.canvas.winfo_width(), 1)
        h = max(v.canvas.winfo_height(), 1)
        v.view_x = cx - (w / 2) * v.scale     # zoom (v.scale) stays as-is
        v.view_y = cy - (h / 2) * v.scale
        v.render()
        self._show_interaction_geometry(uid)  # redraw at the new view

    # ------------------------------------------------------------------ #
    # Persistence: labels.json + session
    # ------------------------------------------------------------------ #
    def _save_labels(self):
        path = filedialog.asksaveasfilename(title="Save labels.json",
                                            defaultextension=".json",
                                            initialfile="labels.json",
                                            filetypes=[("JSON", "*.json")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.store.to_json(), f, indent=2)
        self.status_var.set(f"Wrote {path}")

    def _load_labels(self):
        path = filedialog.askopenfilename(title="Load labels.json",
                                          filetypes=[("JSON", "*.json")])
        if not path:
            return
        doc = config_io.read_json_file(path)
        if doc is None:
            self.status_var.set(f"Could not read {path}")
            return
        try:
            store = LabelStore.from_json(doc)
        except Exception as exc:
            self.status_var.set(f"Not a labels.json: {exc}")
            return
        self._install_store(store)

    def _export_csv(self):
        """Resolved region -> class table: one row per living MSC region of
        every slice whose labels are cached at the current commit (class 0 =
        unlabeled, so the file carries negatives for training too)."""
        if not self.primed:
            self.status_var.set("Nothing primed - Run first, then export.")
            return
        path = filedialog.asksaveasfilename(title="Export resolved labels as CSV",
                                            defaultextension=".csv",
                                            initialfile="labels.csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        covered, skipped = self._write_labels_csv(path)
        msg = f"Wrote {path}: {covered} slice(s)"
        if skipped:
            msg += (f", {skipped} skipped (labels not computed at this commit - "
                    "browse them or Rerun first)")
        self.status_var.set(msg)

    def _write_labels_csv(self, path):
        import numpy as np
        covered = skipped = 0
        lines = ["slice,region_id,class"]
        for si, p in enumerate(self.primed):
            for li in range(len(p["pipes"])):
                rec = self.engine.record(si, li)
                key = self._slice_key(si, li)
                if rec is None or rec.get("labels") is None or key is None:
                    skipped += 1
                    continue
                labels = rec["labels"]
                rc = resolve_slice(self.store.for_slice(key), labels, np)
                ids = np.unique(labels)
                for i in ids[ids >= 0]:
                    lines.append(f"{key},{int(i)},{int(rc[i])}")
                covered += 1
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(lines) + "\n")
        return covered, skipped

    def _gui_state(self):
        d = super()._gui_state()
        d["labels"] = self.store.to_json()   # rides the inherited 4s autosave
        d["tool"] = self.tool_var.get()
        return d

    def _apply_state(self, state, gui=None, notes=None):
        notes = super()._apply_state(state, gui, notes if notes is not None else [])
        gui = gui or {}
        if gui.get("labels"):
            try:
                self._install_store(LabelStore.from_json(gui["labels"]))
            except Exception as exc:
                notes.append(f"labels not restored: {exc}")
        if gui.get("tool") in TOOLS:
            self.tool_var.set(gui["tool"])
        # Keep the regions toggle in sync with whatever seg_source restored to.
        self.show_regions_var.set(self.seg_source_var.get() == "msc")
        return notes

    def _write_configs(self, out_dir):
        paths = super()._write_configs(out_dir)
        path = os.path.join(out_dir, "labels.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.store.to_json(), f, indent=2)
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
    LabelerApp(root, initial)
    root.mainloop()


def _selftest():
    """Exercise the labeler's wiring headlessly (no engine/render needed);
    the pure geometry/ordering/LUT math is covered by tests/test_labeling.py."""
    import tempfile
    import numpy as np
    from . import labeling

    root = tk.Tk()
    root.withdraw()
    # autosave=False: this builds a real app, and a test must not overwrite the
    # user's saved session.
    app = LabelerApp(root, autosave=False)

    # The labeler never needs more than the per-slice tier.
    assert app._needed_level() == "slice", "regions view must stay on the slice tier"
    app.show_regions_var.set(False); app._on_regions_toggle()
    assert app._needed_level() == "slice"
    app.show_regions_var.set(True); app._on_regions_toggle()

    # Fake one primed slice: 4 blocks with SPARSE living ids, -1 border.
    lab = np.full((20, 20), -1, np.int32)
    lab[2:10, 2:10] = 0; lab[2:10, 10:18] = 2
    lab[10:18, 2:10] = 5; lab[10:18, 10:18] = 9
    files = [os.path.join("/data", "s0.tiff")]
    app.subsequences = [{"name": "seq1", "files": files}]
    zeros = np.zeros((20, 20), np.float32)
    app.primed = [{"files": files, "base": [zeros], "filtered": [zeros],
                   "pipes": [None], "normalizers": [[]]}]
    app._rebuild_flat_slices()
    rec = {"commit": app._commit_id, "labels": lab, "stats": None,
           "kept": set(), "cc": None, "n_feat": 4}
    app._slices[(0, 0)] = rec

    # Gestures commit through the same path the DrawController uses.
    app.active_class_var.set(1)
    app._commit_interaction("box", [(3.0, 3.0), (16.0, 16.0)])      # all 4 -> 1
    app.active_class_var.set(2)
    app._commit_interaction("squiggle", [(3.0, 5.0), (15.0, 5.0)])  # {0,2} -> 2
    assert [it.uid for it in app.store.interactions] == [1, 2]
    assert app.store.interactions[0].slice_key == "s0.tiff"

    # Resolution + LUT: later interaction painted over the earlier one.
    lut = app._class_lut_for(0, 0, rec, np)
    assert lut is not None and lut.shape == (10, 4)
    assert tuple(lut[0]) == labeling.CLASS_COLORS[2]     # repainted by squiggle
    assert tuple(lut[2]) == labeling.CLASS_COLORS[2]
    assert tuple(lut[5]) == labeling.CLASS_COLORS[1]     # box only
    assert tuple(lut[9]) == labeling.CLASS_COLORS[1]
    assert lut[1, 3] == 0, "id 1 is not a living id -> transparent"

    # Memoization keys on (commit, store.rev).
    assert app._class_lut_for(0, 0, rec, np) is lut, "cache hit expected"
    app._move_interaction(2, 2)          # same class: no rev bump, still cached
    assert app._class_lut_for(0, 0, rec, np) is lut
    app.store.set_class(1, 2)            # mutation -> rev bump -> rebuilt
    lut2 = app._class_lut_for(0, 0, rec, np)
    assert lut2 is not lut and tuple(lut2[9]) == labeling.CLASS_COLORS[2]
    app.engine.commit_selection()        # Rerun path: fresh commit, fresh labels
    rec2 = dict(rec, commit=app._commit_id)
    lut3 = app._class_lut_for(0, 0, rec2, np)
    assert lut3 is not lut2, "a new commit must re-resolve"

    # The overlay stack: dimmed region layer under the opaque class layer.
    from msseg.viz import min_colors
    ovs = app._seg_overlays(0, 0, rec2, None, np, min_colors)
    assert len(ovs) == 2
    assert int(ovs[0]["lut"][:, 3].max()) <= _REGION_ALPHA
    assert int(ovs[1]["lut"][:, 3].max()) == 255

    # Class-count change clamps orphans; arm state resets when it vanishes.
    app.active_class_var.set(2)
    app.n_classes_var.set(2)
    app._on_n_classes_change()
    assert all(it.class_id == 1 for it in app.store.interactions)
    assert app.active_class_var.get() == 0

    # Session round-trip: the store rides _gui_state and _apply_state.
    g = app._gui_state()
    assert g["labels"]["n_classes"] == 2 and len(g["labels"]["interactions"]) == 2
    app.store = LabelStore()             # clobber
    state = config_io.config_to_state({}, notes=[])
    state["subsequences"] = app.subsequences
    app._apply_state(state, g)
    assert len(app.store.interactions) == 2
    assert all(it.bound for it in app.store.interactions), "rebind by basename"

    # Export writes labels.json alongside the config(s).
    with tempfile.TemporaryDirectory() as td:
        paths = app._write_configs(td)
        assert os.path.basename(paths[-1]) == "labels.json"
        with open(paths[-1], encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["version"] == 1 and len(doc["interactions"]) == 2

    # The round-trip's _apply_state reset the engine; re-fake the primed slice
    # for the interactive-behavior checks below.
    app.primed = [{"files": files, "base": [zeros], "filtered": [zeros],
                   "pipes": [None], "normalizers": [[]]}]
    app._rebuild_flat_slices()
    rec3 = dict(rec, commit=app._commit_id)
    app._slices[(0, 0)] = rec3

    # Undo/redo: snapshots cover adds, class moves, and class-count changes;
    # a fresh edit wipes the redo branch.
    n0 = len(app.store.interactions)
    app.active_class_var.set(1)
    app._commit_interaction("squiggle", [(12.0, 12.0)])        # single-tap gesture
    assert len(app.store.interactions) == n0 + 1
    assert app.store.interactions[-1].points == [(12.0, 12.0)]
    app._undo()
    assert len(app.store.interactions) == n0, "undo removes the tap"
    app._redo()
    assert len(app.store.interactions) == n0 + 1, "redo restores it"
    app._undo()
    app._commit_interaction("squiggle", [(5.0, 12.0)])
    assert not app._redo_stack, "a new edit wipes the redo branch"
    app._undo()
    assert len(app.store.interactions) == n0
    app.n_classes_var.set(3)
    app._on_n_classes_change()
    uid0 = app.store.interactions[0].uid
    app._move_interaction(uid0, 2)
    assert app.store.get(uid0).class_id == 2
    app._undo()
    assert app.store.get(uid0).class_id == 1, "undo covers class moves"
    app._undo()
    assert app.store.n_classes == 2, "undo covers class-count changes"

    # Hover shows the gesture's geometry; a click recenters at constant zoom.
    if app.viewer is not None:
        app._show_interaction_geometry(uid0)
        assert app.viewer.canvas.find_withtag("ihover"), "hover geometry drawn"
        app._hide_interaction_geometry()
        assert not app.viewer.canvas.find_withtag("ihover")
        s0 = app.viewer.scale
        app._on_row_click(uid0)
        assert app.viewer.scale == s0, "recenter keeps the zoom level"
        it0 = app.store.get(uid0)
        cx = sum(x for x, _y in it0.points) / len(it0.points)
        w = max(app.viewer.canvas.winfo_width(), 1)
        assert abs(app.viewer.view_x - (cx - (w / 2) * s0)) < 1e-6

    # Export as CSV: one row per living region, class 0 = unlabeled.
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "labels.csv")
        covered, skipped = app._write_labels_csv(csv_path)
        assert (covered, skipped) == (1, 0)
        with open(csv_path, encoding="utf-8") as f:
            rows = f.read().strip().splitlines()
        assert rows[0] == "slice,region_id,class" and len(rows) == 5
        assert all(r.startswith("s0.tiff,") for r in rows[1:])

    # The labeler's session file never collides with the viewer's.
    assert config_io.session_path(app=app.SESSION_APP) != config_io.session_path()

    root.destroy()
    print("labeler selftest OK: tiers, gestures, resolution order, LUT cache, "
          "overlay stack, class-count clamp, session round-trip, export, "
          "undo/redo, tap-squiggle, hover geometry, click-to-center, CSV")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
