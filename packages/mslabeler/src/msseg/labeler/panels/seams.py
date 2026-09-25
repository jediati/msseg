"""A polyline task's annotation controls: the trace / scope tool row and the
toll row (in the Annotation frame, where a region task has its tools and the
Magic rows), and the ML Seam Classifier frame (the overlay, Train / Evaluate
/ Export, the readout). The gestures themselves are listed in the class
frames like any task's. Also the per-item caches the overlay and the tools
share -- the seam graph comes from the provider (``regions.seams``), the
pixel raster is rebuilt per commit, the resolved classes per (commit,
store.rev). See docs/seam_labeling.md.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .. import seam_labeling, seam_path
from ..seams import (SEAM_COLORS, nearest_seam_point, seam_class_lut, seam_pixel_raster,
                     seam_scalar_lut)
from ..widgets import attach_tooltip
from ..defaults import *  # noqa: F401,F403


class SeamPanelMixin:
    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def _build_polyline_rows(self, ann):
        """The polyline task's rows of the Annotation frame (packed in place of
        the region tools and the Magic rows by _apply_task_kind): the tools,
        and the livewire's toll."""
        row = ttk.Frame(ann)
        self.poly_tool_row = row
        ttk.Label(row, text="Tool:").pack(side="left")
        for value, txt in _SEAM_TOOL_LABELS:
            rb = ttk.Radiobutton(row, text=txt, variable=self.tool_var, value=value)
            rb.pack(side="left", padx=2)
            if value == "trace":
                attach_tooltip(rb, "Trace (key T): click near a seam to anchor, move to "
                                   "see the cheapest path along the seams, click to add "
                                   "an anchor, Enter or double-click to commit in the "
                                   "armed class, BackSpace drops the last leg, Escape "
                                   "abandons.")
            else:
                attach_tooltip(rb, "Scope (key S): drag a box; every seam fully inside "
                                   "it takes the armed class unless a trace says "
                                   "otherwise. A trace anchored inside a scope stays "
                                   "inside it.")
        ttk.Label(row, text="  (arm a class: its number key or swatch)",
                  foreground="#666").pack(side="left")

        row = ttk.Frame(ann)
        self.trace_row = row
        ttk.Label(row, text="Toll:").pack(side="left")
        cb = ttk.Combobox(row, textvariable=self.seam_toll_var,
                          values=list(seam_path.TOLLS), state="readonly", width=13)
        cb.pack(side="left", padx=2)
        cb.bind("<<ComboboxSelected>>", self._unfocus_entries)
        attach_tooltip(cb, "Cost per crack of walking a seam = eps + (1 - affinity).\n"
                           "geometric: length only. feature: z-scored mean dissimilarity "
                           "of the two flanks over the channels. bhattacharyya: their "
                           "Gaussian overlap. barrier: the arc's saddle depth (MSC arcs). "
                           "edges: the '-> edges' model's p(diff). model: the trained "
                           "seam model's boundaryness.")
        ttk.Label(row, text="on").pack(side="left", padx=(6, 2))
        ent = ttk.Entry(row, textvariable=self.seam_channels_var, width=10)
        ent.pack(side="left")
        ent.bind("<Return>", self._unfocus_entries)
        attach_tooltip(ent, "Channels for the feature / bhattacharyya tolls "
                            "(comma-separated; 'base' by default).")

    def _build_seam_panel(self, parent):
        """The ML Seam Classifier frame (a polyline task's, in place of the ML
        Region Classifier): the overlay, the seam model, the readout."""
        f = ttk.LabelFrame(parent, text="ML Seam Classifier")
        self.seam_frame = f

        row = ttk.Frame(f); row.pack(side="top", fill="x", padx=4, pady=2)
        chk = ttk.Checkbutton(row, text="show seams (E)", variable=self.show_seams_var,
                              command=self._refresh_render)
        chk.pack(side="left")
        ttk.Label(row, text="  colour:").pack(side="left")
        self.seam_mode_combo = ttk.Combobox(row, textvariable=self.seam_color_var,
                                            values=list(_SEAM_MODES), state="readonly",
                                            width=13)
        self.seam_mode_combo.pack(side="left", padx=2)
        self.seam_mode_combo.bind("<<ComboboxSelected>>", self._on_seam_mode_change)

        row = ttk.Frame(f); row.pack(side="top", fill="x", padx=4, pady=2)
        self.seam_train_btn = ttk.Button(row, text="Train (R)", width=11,
                                         command=self._train_seam_model)
        self.seam_train_btn.pack(side="left")
        attach_tooltip(self.seam_train_btn, "Fit the seam model on every labelled seam, "
                                            "then score every seam of every item: the "
                                            "'model' toll and the boundaryness colouring.")
        self.seam_eval_btn = ttk.Button(row, text="Evaluate", width=9,
                                        command=self._evaluate_seams)
        self.seam_eval_btn.pack(side="left", padx=2)
        attach_tooltip(self.seam_eval_btn, "Leave-items-out cross-validation of the seam "
                                           "model (held-out log-loss, balanced accuracy).")
        self.seam_export_btn = ttk.Button(row, text="Export…", width=8,
                                          command=self._export_seams)
        self.seam_export_btn.pack(side="left", padx=2)
        attach_tooltip(self.seam_export_btn, "Write every item's seams with their class "
                                             "and boundaryness (seams_<item>.json + "
                                             "seams_summary.csv).")

        self.seam_readout_var = tk.StringVar(master=self.root, value="")
        lab = ttk.Label(f, textvariable=self.seam_readout_var, justify="left",
                        wraplength=280)
        lab.pack(side="top", fill="x", padx=6, pady=(2, 4))

    def _on_seam_mode_change(self, _e=None):
        self._unfocus_entries()
        self._refresh_render()

    # ------------------------------------------------------------------ #
    # Caches (per item)
    # ------------------------------------------------------------------ #
    def _clear_seam_caches(self):
        self._seam_caches.clear()
        self._seam_rasters.clear()

    def _seam_graph_for(self, si, li, np):
        key = self.catalogue.key_of(si, li)
        return None if key is None else self.regions.seams(key, np)

    def _seam_raster_for(self, si, li, rec, graph, np):
        hit = self._seam_rasters.get((si, li))
        if hit is not None and hit[0] == rec.get("commit") and hit[1] is graph:
            return hit[2]
        raster = seam_pixel_raster(graph, np)
        self._seam_rasters[(si, li)] = (rec.get("commit"), graph, raster)
        return raster

    def _seam_cache_for(self, si, li, rec, graph, np):
        """(commit, store.rev, seam class uint8[S], [(gesture, mask)]) for one
        item, memoized on (commit, rev) like the region class LUTs."""
        hit = self._seam_caches.get((si, li))
        if (hit is not None and hit[0] == rec.get("commit") and hit[1] == self.store.rev
                and hit[4] is graph):
            return hit
        key = self.catalogue.key_of(si, li)
        # Every gesture speaks to the seams (derive.py): region samples label
        # the seam between two labelled flanks, extents their outer seams,
        # and the explicit seam gestures overwrite. The region rasterization
        # is the class layer's own (one pass per item and rev); gestures drawn
        # much coarser than this item are left out, or a swath's edges would
        # read as boundaries.
        from .. import derive
        from ..labeling import resolve_sets
        gestures = self._gestures_for_key(key)
        coarse = {it.uid for it in self._coarse_gestures(si, li)}
        keep = [it for it in gestures if it.uid not in coarse]
        region_class, touch = None, {}
        cache = getattr(self, "_labels_cache_for", None)
        if cache is not None and keep:
            lc = cache(si, li, rec, np)
            touch = lc[3]
            if len(keep) == len(gestures):
                region_class = lc[5]
            else:                                # the same sets, minus the coarse ones
                region_class = resolve_sets([(it, touch.get(it.uid, set())) for it in keep],
                                            rec["labels"], np)
        extents = [(it, touch.get(it.uid, set())) for it in keep if derive.is_extent(it)]
        res = derive.seam_labels(graph, np, region_class, extents,
                                 self._seam_gestures_for_key(key))
        entry = (rec.get("commit"), self.store.rev, res.cls, res.sets, graph, res.derived)
        self._seam_caches[(si, li)] = entry
        return entry

    def _seam_classes_for(self, si, li, np):
        """The resolved seam classes of an item (None without a record)."""
        rec = self.regions.record(self.catalogue.key_of(si, li))
        if rec is None or rec.get("labels") is None:
            return None, None
        graph = self._seam_graph_for(si, li, np)
        if graph is None:
            return None, None
        return graph, self._seam_cache_for(si, li, rec, graph, np)[2]

    # ------------------------------------------------------------------ #
    # Overlay
    # ------------------------------------------------------------------ #
    def _seam_overlay(self, si, li, rec, np):
        """The seam layer for the class stack: seam classes in their colours,
        or boundaryness on the ramp when a seam model has scored this commit."""
        if rec is None or rec.get("labels") is None:
            return None
        graph = self._seam_graph_for(si, li, np)
        if graph is None or graph.n_seams == 0:
            return None
        lut = None
        if self.seam_color_var.get() == _SEAM_MODE_BOUNDARYNESS:
            entry = self._seam_pred.get(self.catalogue.key_of(si, li))
            if entry is not None and entry[0] == rec.get("commit") and len(entry[1]) == graph.n_seams:
                lut = seam_scalar_lut(entry[1], np, _SCALAR_ALPHA)
        if lut is None:
            cls = self._seam_cache_for(si, li, rec, graph, np)[2]
            if not cls.any():
                return None            # nothing labelled: no layer to composite
            lut = seam_class_lut(cls, np, colors=self._class_colors_rgba(np))
        # The raster (both flank pixels of every crack) is built only now, once
        # there is something to show, and cached per commit.
        raster = self._seam_raster_for(si, li, rec, graph, np)
        return self._region_overlay(raster, lut, np)

    # ------------------------------------------------------------------ #
    # Tolls for the trace tool
    # ------------------------------------------------------------------ #
    def _seam_channels(self):
        want = [c.strip() for c in self.seam_channels_var.get().split(",") if c.strip()]
        return want or ["base"]

    def _seam_tolls_for(self, key, rec, graph, np):
        """The toll per seam for the picked toll, or a ValueError naming what
        is missing (shown in the status line; the press is refused)."""
        toll = self.seam_toll_var.get()
        table = rec.get("stats")
        arcs = self.regions.arcs(key, np) if toll in seam_path.ARC_TOLLS else None
        pdiff = None
        boundaryness = None
        if toll == "edges":
            pr = self._pred.get(key)
            aux = self._pred_aux(pr) if pr is not None and pr[0] == rec.get("commit") else None
            pdiff = None if aux is None else aux.get("pdiff")
        elif toll == "model":
            entry = self._seam_pred.get(key)
            if entry is not None and entry[0] == rec.get("commit"):
                boundaryness = entry[1]
        chans = self._seam_channels()
        if table is not None and toll in ("feature", "bhattacharyya"):
            avail = self.FIELDS.channel_names(table)
            chans = [c for c in chans if c in avail] or (["base"] if "base" in avail else avail[:1])
        aff = seam_path.seam_affinity(graph, toll, np, table=table, arcs=arcs, channels=chans,
                                      pdiff=pdiff, boundaryness=boundaryness, conv=self.FIELDS)
        return seam_path.seam_tolls(aff, np)

    def _scope_restrict(self, si, li, rec, graph, seam, np):
        """The union of the scopes containing `seam` as a seam mask, or None
        when the anchor lies in no scope."""
        sets = self._seam_cache_for(si, li, rec, graph, np)[3]
        masks = [m for it, m in sets if it.tool == "scope" and m[seam]]
        if not masks:
            return None
        out = masks[0].copy()
        for m in masks[1:]:
            out |= m
        return out

    # ------------------------------------------------------------------ #
    # Geometry on the canvas
    # ------------------------------------------------------------------ #
    def _seam_color_hex(self, class_id):
        """A seam class's colour: the task's own class colour (a polyline
        task's vocabulary is its seam vocabulary)."""
        if getattr(self, "_task_kind", lambda: "region")() == "polyline":
            return self._class_color_hex(int(class_id))
        k = int(class_id)
        r, g, b, _a = SEAM_COLORS[k] if 0 <= k < len(SEAM_COLORS) else SEAM_COLORS[-1]
        return f"#{r:02x}{g:02x}{b:02x}"

    def _visible_seam_gestures(self):
        cur = self._current()
        if cur is None:
            return list(self.store.seams)
        return self._seam_gestures_for(*cur)

    def _draw_seam_geometry(self, it, tags=("draw", "ihover")):
        """A seam gesture's geometry over the item: a trace as its polyline,
        a scope as its rectangle, in the seam class colour."""
        v = self.viewer
        if it is None or v is None or not it.points or not it.bound:
            return
        cur = self._current()
        if cur is None or not self._gesture_on_item(it, *cur):
            return
        c = v.canvas
        color = self._seam_color_hex(it.class_id)
        scr = [((x - v.view_x) / v.scale, (y - v.view_y) / v.scale) for x, y in it.points]
        if it.tool == "scope" and len(scr) >= 2:
            (x0, y0), (x1, y1) = scr[0], scr[-1]
            c.create_rectangle(x0, y0, x1, y1, outline=color, width=2, dash=(6, 3), tags=tags)
        elif len(scr) >= 2:
            flat = [coord for pt in scr for coord in pt]
            c.create_line(*flat, fill=color, width=3, tags=tags)
        else:
            x, y = scr[0]
            c.create_oval(x - 4, y - 4, x + 4, y + 4, outline=color, width=3, tags=tags)

    # ------------------------------------------------------------------ #
    # Readout + gesture list
    # ------------------------------------------------------------------ #
    def _seam_counts_text(self):
        cur = self._current()
        if cur is None:
            return "no item on screen"
        its = self._visible_seam_gestures()
        n_tr = sum(1 for it in its if it.tool == "trace")
        n_sc = sum(1 for it in its if it.tool == "scope")
        head = f"{n_tr} trace{'s' if n_tr != 1 else ''}, {n_sc} scope{'s' if n_sc != 1 else ''} here"
        try:
            import numpy as np
            graph, cls = self._seam_classes_for(cur[0], cur[1], np)
        except Exception:
            graph, cls = None, None
        if graph is None:
            return head + " · no regions yet"
        import numpy as np
        counts = np.bincount(np.asarray(cls, np.intp), minlength=self.store.n_classes)
        parts = [f"{int(counts[k])} {self._class_name(k)}"
                 for k in range(1, self.store.n_classes)]
        text = (f"{head} · {graph.n_seams} seams: " + ", ".join(parts)
                + f", {int(counts[0])} unknown")
        # Labels the region gestures derived (derive.py), no seam gesture needed.
        entry = self._seam_caches.get((cur[0], cur[1]))
        derived = entry[5] if entry is not None and len(entry) > 5 else None
        if derived is not None and derived.any():
            text += f" ({int(derived.sum())} derived)"
        model = getattr(self, "_seam_model", None)
        text += " · model: " + (model.brief() if model is not None else "none")
        return text

    def _class_name(self, k):
        return self.store.name(k) if self.store.has_name(k) else f"class {k}"

    def _refresh_seam_panel(self):
        var = getattr(self, "seam_readout_var", None)
        if var is None:
            return
        var.set(self._seam_counts_text())

    def _seam_class_totals(self):
        """All-item seams per class: resolved on every item that has a record
        and a seam gesture (the polyline task's counterpart of the region
        counts in the class titles)."""
        import numpy as np
        out = {}
        for key in self.catalogue.keys():
            pos = self.catalogue.index_of(key)
            if pos is None or not self._seam_gestures_for_key(key):
                continue
            rec = self.regions.record(key)
            if rec is None or rec.get("labels") is None:
                continue
            graph = self.regions.seams(key, np)
            if graph is None:
                continue
            cls = self._seam_cache_for(pos[0], pos[1], rec, graph, np)[2]
            counts = np.bincount(np.asarray(cls, np.intp), minlength=self.store.n_classes)
            for k in range(1, self.store.n_classes):
                if counts[k]:
                    out[k] = out.get(k, 0) + int(counts[k])
        return out

    def _seam_gesture_at(self, ix, iy):
        """The uid of the seam gesture that labels the seam nearest an image
        point (the last-drawn one, the resolution order), or None."""
        cur = self._current()
        if cur is None or ix is None or iy is None:
            return None
        import numpy as np
        rec = self.regions.record(self.catalogue.key_of(*cur))
        if rec is None or rec.get("labels") is None:
            return None
        graph = self._seam_graph_for(cur[0], cur[1], np)
        if graph is None or graph.n_seams == 0:
            return None
        rx, ry = self._region_placement().to_raster(ix, iy)
        hit = nearest_seam_point(graph, rec["labels"], rx, ry, np, radius=6)
        if hit is None:
            return None
        sets = self._seam_cache_for(cur[0], cur[1], rec, graph, np)[3]
        best = None
        for it, mask in sets:                 # application order: the last wins
            if mask[hit[0]]:
                best = it.uid
        return best

    # ------------------------------------------------------------------ #
    # Session view state
    # ------------------------------------------------------------------ #
    def _seams_view_state(self):
        return {"toll": self.seam_toll_var.get(),
                "channels": self.seam_channels_var.get(),
                "show": bool(self.show_seams_var.get()),
                "coloring": self.seam_color_var.get()}

    def _apply_seams_view(self, d):
        if not isinstance(d, dict):
            return
        if d.get("toll") in seam_path.TOLLS:
            self.seam_toll_var.set(d["toll"])
        if isinstance(d.get("channels"), str) and d["channels"].strip():
            self.seam_channels_var.set(d["channels"])
        if isinstance(d.get("show"), bool):
            self.show_seams_var.set(d["show"])
        if d.get("coloring") in _SEAM_MODES:
            self.seam_color_var.set(d["coloring"])

    # ------------------------------------------------------------------ #
    # Hotkeys
    # ------------------------------------------------------------------ #
    def _on_trace_key(self, _e=None):
        if self._typing():
            return
        self.tool_var.set("trace")

    def _on_scope_key(self, _e=None):
        if self._typing():
            return
        self.tool_var.set("scope")

    def _on_seams_toggle_key(self, _e=None):
        if self._typing():
            return
        self.show_seams_var.set(not self.show_seams_var.get())
        self._refresh_render()

    def _trace_controller(self):
        tool = self.viewer.tool if self.viewer is not None else None
        return getattr(tool, "trace", None)

    def _on_trace_commit_key(self, e=None):
        """Enter commits the trace in flight. Entries bind Return to
        _unfocus_entries at widget level and this toplevel binding fires
        after focus moved, so the guard reads the EVENT's widget."""
        try:
            if e is not None and e.widget.winfo_class() in _TYPING_CLASSES:
                return
        except (AttributeError, tk.TclError):
            pass
        tc = self._trace_controller()
        if tc is not None and tc.active:
            tc.commit()
            return "break"
        return None

    def _on_trace_back_key(self, e=None):
        try:
            if e is not None and e.widget.winfo_class() in _TYPING_CLASSES:
                return
        except (AttributeError, tk.TclError):
            pass
        tc = self._trace_controller()
        if tc is not None and tc.active:
            tc.drop_last()
            return "break"
        return None
