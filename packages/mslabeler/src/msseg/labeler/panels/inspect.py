"""Inspect: look at the regions' embeddings (``represent``).

Pick an embedding -- the region net's input or a hidden layer, another
task's net, a column family, the statistics row, the class probabilities --
and the canvas recolours every region by how alike it is to an ANCHOR: the
region under the pointer (live hover), the regions pinned by clicking on
the canvas or a class's centroid. **The pin tool is the tab**: while the
Inspect tab is open a click pins (Shift-click adds, Escape clears) instead of
drawing, and the colouring switches to ``similarity``; leaving the tab gives
the drawing tools and the previous colouring back (key I toggles the tab). Pins keep their VECTOR, so with an embedding that means the
same thing on every item (a net layer, raw families, histograms, proba --
``Embedding.cross_item``) they recolour other items too.

The Inspect tab adds: the regions most like the anchors (this item or every
computed item; double-click goes there, "Label as" commits the selection as
one taps gesture per item), a 2-D scatter of the item's embedding (PCA,
coloured by annotation / prediction / similarity; hover links to the canvas,
a drag selects -- "Label as" commits it), one embedding unit as a colouring
mode, and "Evaluate embeddings": how well every option separates the
annotated classes, scored leave-items-out.

Design: ``docs/design_region_embeddings.md``.
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk

from .. import represent
from ..labeling import MAX_CLASSES, scalar_lut
from ..training import TrainingProblem
from ..widgets import attach_tooltip
from ..defaults import *  # noqa: F401,F403

_SCATTER_PX = 380            # the scatter's side
_SCATTER_MAX_POINTS = 60000  # drawn points (seeded sample beyond it)
_NEIGHBOURS_K = 50
_EVAL_PUMP_MS = 120
# Scatter colours: unlabelled / unpredicted points.
_SCATTER_GREY = (150, 150, 150)
_SCATTER_BG = (24, 24, 24)


class InspectMixin:

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def _init_inspect_state(self, root):
        self.inspect_ref = None                      # None = the default (see below)
        self.inspect_label_var = tk.StringVar(master=root, value="")
        self.inspect_sim_var = tk.StringVar(master=root, value="cosine")
        self.inspect_reduce_var = tk.StringVar(master=root, value="max")
        self.inspect_live_var = tk.BooleanVar(master=root, value=True)
        self.inspect_scope_var = tk.StringVar(master=root, value="this item")
        self.inspect_color_var = tk.StringVar(master=root, value="annotation")
        self.inspect_unit_var = tk.StringVar(master=root, value="0")
        self.inspect_status_var = tk.StringVar(master=root, value="")
        self._inspect_labels = {}     # picker label -> EmbeddingRef (last listing)
        self._inspect_pins = []       # [{"key", "region", "vec", "ref", "task", "model", "what"}]
        self._inspect_hover = None    # (key, region) under the pointer
        self._inspect_live_lut = None  # the similarity LUT on the canvas (hover repaint)
        self._inspect_rows = []       # the neighbours list
        self._inspect_scatter = None  # {"P", "rows", "ids", "key", "xy"} of the drawn scatter
        self._inspect_sel = []        # region ids selected on the scatter (current item)
        self._inspect_eval = None     # in-flight evaluation {"q", "thread"}
        self._inspect_eval_rows = []
        self._inspect_was_active = False  # the Inspect tab showing (clicks pin)
        self._inspect_prev_mode = None    # the colouring before the tab opened
        self._inspect_prev_tab = None     # where key I goes back to

    # ------------------------------------------------------------------ #
    # Which embedding
    # ------------------------------------------------------------------ #
    def _inspect_ref_now(self):
        """The picked embedding, or the default: the region net's last hidden
        layer when the task has a dense net, else the statistics row."""
        if self.inspect_ref is not None:
            return self.inspect_ref
        model = getattr(self._task, "model", None)
        if model is not None and model.clf is not None and represent.depth_of(model.clf) > 0:
            return represent.EmbeddingRef(source="net")
        return represent.EmbeddingRef()

    def _inspect_label_of(self, ref):
        return getattr(self, "_inspect_clean", {}).get(ref.key()) or represent.short_label(ref)

    def _refresh_inspect_options(self):
        """Repopulate the picker from ``represent.options`` for the item on
        screen; greyed options are listed with their reason."""
        try:
            opts = self._embedding_options()
        except Exception as exc:                 # noqa: BLE001 -- never break the picker
            self._log(f"inspect: embedding options unavailable: {exc}")
            opts = []
        # Sections with each entry under what it is computed from; headers
        # and greyed lines are not picks.
        rows = represent.menu_rows(opts)
        self._inspect_labels = {t: o.ref for t, o in rows if o is not None and o.usable}
        self._inspect_clean = {o.ref.key(): o.label for o in opts}
        values = [t for t, _o in rows]
        combo = getattr(self, "inspect_combo", None)
        if combo is not None:
            try:
                combo.config(values=values)
            except tk.TclError:
                pass
        self.inspect_label_var.set(self._inspect_label_of(self._inspect_ref_now()))
        return opts

    def _on_inspect_pick(self, _e=None):
        got = self.inspect_label_var.get()
        if got in self._inspect_labels:
            self._set_inspect_embedding(self._inspect_labels[got])
        else:                                    # a greyed entry: keep the pick
            self.inspect_label_var.set(self._inspect_label_of(self._inspect_ref_now()))
        self._unfocus_entries()

    def _set_inspect_embedding(self, ref, similarity=None):
        """Headless: inspect `ref` (None = the default). Pins made under
        another embedding stop counting (their vectors are another space)."""
        self.inspect_ref = ref
        if similarity in represent.SIMILARITIES:
            self.inspect_sim_var.set(similarity)
        self.inspect_label_var.set(self._inspect_label_of(self._inspect_ref_now()))
        self._inspect_scatter = None
        self._inspect_settings_changed()

    def _inspect_settings_changed(self, *_a):
        self._refresh_inspect_status()
        if self.region_mode_var.get() in _INSPECT_MODES:
            self._refresh_render()

    # ------------------------------------------------------------------ #
    # Vectors and anchors
    # ------------------------------------------------------------------ #
    def _inspect_embedding(self, key, rec, np, quiet=False):
        """The inspected embedding on item `key`, or None (the reason in the
        status bar unless `quiet`)."""
        if rec is None or rec.get("stats") is None:
            return None
        ref = self._inspect_ref_now()
        try:
            return self._embedding(key, rec, ref, np)
        except Exception as exc:                 # noqa: BLE001 -- refusals, pending grids
            if not quiet:
                self.status_var.set(f"inspect: {represent.short_label(ref)}: {exc}")
            return None

    def _inspect_model_sig(self, ref):
        """What a pin's vector depends on beyond the ref: the task, and the
        net it came from (a retrain makes old pins another space)."""
        if not ref.uses_model() or ref.provider is not None:
            return self._task.uid if ref.uses_model() else None, None
        stack = self._task.model
        return self._task.uid, represent.signature_of(ref, None, stack)[2:]

    def _inspect_pins_for(self, key, emb):
        """The pins that count on item `key` under `emb`: same embedding,
        same model, the right width, and -- for a per-item embedding -- made
        on this item."""
        ref = emb.ref
        task, model = self._inspect_model_sig(ref)
        out = []
        for p in self._inspect_pins:
            if p["ref"] != ref.key() or p["task"] != task or p["model"] != model:
                continue
            if len(p["vec"]) != emb.dim:
                continue
            if not emb.cross_item and p["key"] != key:
                continue
            out.append(p)
        return out

    def _inspect_anchors(self, key, emb, np):
        """``(A float32[m, d], what)`` the colouring compares against: the
        pins that count, else the hovered region (live hover), else None."""
        pins = self._inspect_pins_for(key, emb)
        if pins:
            return np.stack([p["vec"] for p in pins]), f"{len(pins)} pin(s)"
        hov = self._inspect_hover
        if self.inspect_live_var.get() and hov is not None and hov[0] == key:
            row = emb.row_of(hov[1])
            if row is not None:
                return emb.Z[row:row + 1], f"region {hov[1]} (hover)"
        return None, None

    def _inspect_unit(self, dim):
        try:
            j = int(float(self.inspect_unit_var.get()))
        except (TypeError, ValueError):
            j = 0
        return min(max(j, 0), max(dim - 1, 0))

    def _inspect_scalar(self, si, li, rec, np):
        """``(values[K], mask[K])`` for the Inspect colouring modes on item
        (si, li), or None (no embedding, or nothing to compare against)."""
        if rec is None or rec.get("labels") is None:
            return None
        key = self.catalogue.key_of(si, li)
        emb = self._inspect_embedding(key, rec, np, quiet=True)
        if emb is None or emb.dim == 0:
            return None
        labels = rec["labels"]
        K = int(labels.max()) + 1 if labels.size else 1
        mode = self.region_mode_var.get()
        if mode == _MODE_UNIT:
            col = emb.Z[:, self._inspect_unit(emb.dim)]
            return emb.by_id(represent.to_unit(col, "dot", np), np, K=K)
        A, _what = self._inspect_anchors(key, emb, np)
        if A is None:
            return None
        kind = self.inspect_sim_var.get()
        if kind not in represent.SIMILARITIES:
            kind = "cosine"
        reduce = self.inspect_reduce_var.get()
        s = represent.similarity(emb.Z, A, kind, np, reduce if reduce in represent.REDUCES else "max")
        return emb.by_id(represent.to_unit(s, kind, np), np, K=K)

    def _refresh_inspect_status(self):
        ref = self._inspect_ref_now()
        n = len(self._inspect_pins)
        cross = "same on every item" if ref.cross_item else "per item (pins stay on their item)"
        self.inspect_status_var.set(
            f"{represent.short_label(ref)} -- {cross}; {n} pin(s); "
            f"{self.inspect_sim_var.get()}, {self.inspect_reduce_var.get()} over anchors")

    # ------------------------------------------------------------------ #
    # The canvas: the region under the pointer, pins, hover
    # ------------------------------------------------------------------ #
    def _region_at(self, ix, iy):
        """``(key, region)`` under image point (ix, iy) on the item on screen,
        or None. Placement-correct: the provider's ``LabelLayer`` (an mspath
        ROI sits at its own origin and scale)."""
        cur = self._current()
        if cur is None or ix is None or iy is None:
            return None
        key = self.catalogue.key_of(*cur)
        layer = self.regions.label_layer(key)
        if layer is None:
            return None
        try:
            r = int(layer.id_at(int(ix), int(iy)))
        except (IndexError, ValueError, TypeError):
            return None
        return (key, r) if r >= 0 else None

    def _inspect_press(self, e, add=False):
        """A canvas click while the Inspect tab is open: pin the region under
        it (Shift adds to the pins); a click off the regions pans."""
        v = self.viewer
        x, y = v.view_x + e.x * v.scale, v.view_y + e.y * v.scale
        hit = self._region_at(x, y)
        if hit is None:
            return False
        return self._inspect_pin(hit[0], hit[1], add=add)

    def _inspect_pin(self, key, region, add=False):
        """Pin `region` of item `key` (headless). False with the reason in the
        status bar when it has no vector under the inspected embedding."""
        import numpy as np
        rec = self.regions.record(key)
        emb = self._inspect_embedding(key, rec, np)
        if emb is None:
            return False
        row = emb.row_of(region)
        if row is None:
            self.status_var.set(f"inspect: region {region} has no statistics row")
            return False
        task, model = self._inspect_model_sig(emb.ref)
        pin = {"key": key, "region": int(region), "vec": np.array(emb.Z[row], np.float32),
               "ref": emb.ref.key(), "task": task, "model": model, "what": f"region {region}"}
        if not add:
            self._inspect_pins = []
        self._inspect_pins.append(pin)
        self._inspect_pins_changed()
        return True

    def _inspect_pin_class(self, cls, scope="this item"):
        """Pin the centroid of class `cls`'s annotated regions -- on the item
        on screen, or over every computed item (a per-item embedding stays on
        this item). False when no region of that class is annotated."""
        import numpy as np
        cur = self._current()
        if cur is None:
            return False
        key0 = self.catalogue.key_of(*cur)
        ref = self._inspect_ref_now()
        items = [(cur[0], cur[1], key0, self.regions.record(key0))]
        if scope != "this item" and ref.cross_item:
            items = self._inspect_items()
            if items is None:
                return False
        vecs = []
        for si, li, key, rec in items:
            emb = self._inspect_embedding(key, rec, np, quiet=True)
            truth = self._truth_from_cache(si, li, rec, np) if rec is not None else None
            if emb is None or truth is None:
                continue
            ids = np.asarray(emb.ids, np.intp)
            ok = (ids >= 0) & (ids < len(truth))
            rows = np.flatnonzero(ok)[np.asarray(truth)[ids[ok]] == int(cls)]
            if len(rows):
                vecs.append(emb.Z[rows])
        if not vecs:
            self.status_var.set(f"inspect: no region of class {cls} is annotated here")
            return False
        emb0 = self._inspect_embedding(key0, items[0][3], np, quiet=True)
        task, model = self._inspect_model_sig(ref)
        n = sum(len(v) for v in vecs)
        self._inspect_pins.append({"key": key0, "region": -1,
                                   "vec": np.concatenate(vecs).mean(axis=0).astype(np.float32),
                                   "ref": (emb0.ref if emb0 is not None else ref).key(),
                                   "task": task, "model": model,
                                   "what": f"class {cls} centroid ({n} regions)"})
        self._inspect_pins_changed()
        return True

    def _inspect_clear(self):
        """Drop every pin. False when there were none (Escape then disarms)."""
        if not self._inspect_pins:
            return False
        self._inspect_pins = []
        self._inspect_pins_changed()
        return True

    def _inspect_unpin(self, i):
        if 0 <= i < len(self._inspect_pins):
            del self._inspect_pins[i]
            self._inspect_pins_changed()

    def _inspect_pins_changed(self):
        if self.region_mode_var.get() not in _INSPECT_MODES:
            if _MODE_SIMILARITY not in self._region_modes():
                self._refresh_region_modes()
            self.region_mode_var.set(_MODE_SIMILARITY)
        self._fill_inspect_pins()
        self._refresh_inspect_status()
        self._refresh_render()
        self._fill_inspect_neighbours()

    def _inspect_on_hover(self, ix, iy):
        """Canvas hover: the anchor of a live similarity colouring (when no
        pin is set), and the matching point on the scatter."""
        hit = self._region_at(ix, iy) if ix is not None else None
        if hit == self._inspect_hover:
            return
        self._inspect_hover = hit
        self._scatter_mark(hit)
        if (self.region_mode_var.get() == _MODE_SIMILARITY and self.inspect_live_var.get()
                and not self._inspect_pins and self.show_regions_var.get()):
            self._inspect_repaint()

    def _inspect_repaint(self):
        """Swap the similarity layer's LUT in place (the region raster and
        every other layer stay) and ask for a debounced repaint -- a hover
        tick is one mat-vec and one LUT, not a re-render. Falls back to a full
        render when the layer is not on the canvas."""
        import numpy as np
        v = self.viewer
        cur = self._current()
        old = self._inspect_live_lut
        if v is None or cur is None:
            return
        rec = self.regions.record(self.catalogue.key_of(*cur))
        scalar = self._inspect_scalar(cur[0], cur[1], rec, np)
        overlays = getattr(v, "_overlays", None)
        if scalar is None or old is None or overlays is None:
            self._refresh_render()
            return
        new = scalar_lut(scalar[0], np, _SCALAR_ALPHA, scalar[1])
        for i, t in enumerate(overlays):
            if t[0] == "label" and t[2] is old:
                overlays[i] = (t[0], t[1], new, t[3])
                self._inspect_live_lut = new
                v.invalidate()
                return
        self._refresh_render()

    # ------------------------------------------------------------------ #
    # The tab IS the tool: open = clicks pin, closed = clicks draw
    # ------------------------------------------------------------------ #
    def _inspect_active(self):
        """Whether the Inspect tab is the one showing (canvas clicks pin)."""
        tabs = getattr(self, "_center_tabs", None)
        return bool(tabs) and "Inspect" in tabs and self._center_tab_name() == "Inspect"

    def _inspect_tab_changed(self):
        """Entering the Inspect tab: abandon a gesture in flight and colour by
        similarity; leaving it: the previous colouring back, no hover anchor.
        Idempotent -- called on every tab change."""
        active = self._inspect_active()
        if active == self._inspect_was_active:
            return
        self._inspect_was_active = active
        if active:
            tool = self.viewer.tool if self.viewer is not None else None
            if tool is not None and hasattr(tool, "cancel"):
                tool.cancel()
            self._inspect_prev_mode = self.region_mode_var.get()
            if self._inspect_prev_mode not in _INSPECT_MODES:
                self._refresh_region_modes()
                self.region_mode_var.set(_MODE_SIMILARITY)
            self.status_var.set("Inspect: click a region to pin it (Shift adds, Escape "
                                "clears); leave the tab to draw again")
        else:
            self._inspect_hover = None
            prev = self._inspect_prev_mode
            if self.region_mode_var.get() in _INSPECT_MODES and prev is not None:
                self.region_mode_var.set(prev if prev in self._region_modes() else _MODE_ID)
            self._inspect_prev_mode = None
        self._refresh_render()

    def _on_inspect_key(self, _e=None):
        """I: open the Inspect tab, or go back to the tab you came from."""
        if self._typing():
            return
        if self._inspect_active():
            self._show_center_tab(self._inspect_prev_tab or "Annotation")
        else:
            self._inspect_prev_tab = self._center_tab_name()
            self._show_center_tab("Inspect")
        self._inspect_tab_changed()

    # ------------------------------------------------------------------ #
    # The Inspect tab
    # ------------------------------------------------------------------ #
    def _build_inspect_tab(self, parent):
        box = ttk.LabelFrame(parent, text="Embedding")
        box.pack(side="top", fill="x", padx=6, pady=4)
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=2)
        self.inspect_combo = ttk.Combobox(row, textvariable=self.inspect_label_var,
                                          state="readonly", width=38,
                                          postcommand=self._refresh_inspect_options)
        self.inspect_combo.pack(side="left", padx=2, fill="x", expand=True)
        self.inspect_combo.bind("<<ComboboxSelected>>", self._on_inspect_pick)
        attach_tooltip(self.inspect_combo,
                       "What a region is, as a vector: the region net's scaled input or a "
                       "hidden layer, another task's net, a column family (foundation, "
                       "encoder, context), the statistics row (z-scored per item) or the "
                       "class probabilities. Greyed entries say why they are unavailable.")
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="similarity").pack(side="left")
        cb = ttk.Combobox(row, textvariable=self.inspect_sim_var, state="readonly", width=9,
                          values=list(represent.SIMILARITIES))
        cb.pack(side="left", padx=2)
        cb.bind("<<ComboboxSelected>>", self._inspect_settings_changed)
        ttk.Label(row, text="over anchors").pack(side="left", padx=(6, 0))
        cb = ttk.Combobox(row, textvariable=self.inspect_reduce_var, state="readonly", width=5,
                          values=list(represent.REDUCES))
        cb.pack(side="left", padx=2)
        cb.bind("<<ComboboxSelected>>", self._inspect_settings_changed)
        ttk.Checkbutton(row, text="live hover", variable=self.inspect_live_var,
                        command=self._inspect_settings_changed).pack(side="left", padx=(8, 0))
        ttk.Label(box, textvariable=self.inspect_status_var, foreground="#555",
                  wraplength=400, justify="left").pack(anchor="w", padx=6, pady=(0, 2))
        ttk.Label(box, foreground="#666", wraplength=400, justify="left",
                  text="While this tab is open the canvas colours regions by "
                       "similarity: hover a region, or click to pin it (Shift-click "
                       "adds, Escape clears). Leave the tab to draw again; I toggles "
                       "it.").pack(anchor="w", padx=6, pady=(0, 4))

        box = ttk.LabelFrame(parent, text="Embedding space (this item, PCA)")
        box.pack(side="top", fill="x", padx=6, pady=4)
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=2)
        ttk.Button(row, text="Project", command=self._draw_inspect_scatter).pack(side="left")
        ttk.Label(row, text="colour").pack(side="left", padx=(8, 2))
        cb = ttk.Combobox(row, textvariable=self.inspect_color_var, state="readonly", width=11,
                          values=["annotation", "prediction", "similarity"])
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self._draw_inspect_scatter())
        ttk.Button(row, text="Label selection",
                   command=self._on_inspect_label_selection).pack(side="left", padx=(8, 0))
        ttk.Label(row, text="unit").pack(side="left", padx=(8, 2))
        sp = ttk.Spinbox(row, from_=0, to=4095, width=4, textvariable=self.inspect_unit_var,
                         command=self._inspect_settings_changed)
        sp.pack(side="left")
        attach_tooltip(sp, "The embedding column the 'embedding unit' colouring mode shows.")
        self.inspect_canvas = tk.Canvas(box, width=_SCATTER_PX, height=_SCATTER_PX,
                                        background="#181818", highlightthickness=0)
        self.inspect_canvas.pack(padx=4, pady=(0, 4))
        self.inspect_canvas.bind("<Motion>", self._on_scatter_motion)
        self.inspect_canvas.bind("<ButtonPress-1>", self._on_scatter_press)
        self.inspect_canvas.bind("<B1-Motion>", self._on_scatter_drag)
        self.inspect_canvas.bind("<ButtonRelease-1>", self._on_scatter_release)
        self.inspect_canvas.bind("<Leave>", lambda e: self._on_scatter_motion(None))
        self._scatter_drag = None

        box = ttk.LabelFrame(parent, text="Which embedding separates the classes?")
        box.pack(side="top", fill="x", padx=6, pady=4)
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=2)
        ttk.Button(row, text="Evaluate embeddings",
                   command=self._evaluate_embeddings).pack(side="left")
        self.inspect_eval_var = tk.StringVar(master=self.root, value="")
        ttk.Label(row, textvariable=self.inspect_eval_var, foreground="#555").pack(
            side="left", padx=6)
        cols = ("emb", "dim", "knn", "centroid", "sil")
        self.inspect_eval_tree = ttk.Treeview(box, columns=cols, show="headings", height=6,
                                              selectmode="browse")
        for cid, text, width, anchor in (("emb", "embedding", 200, "w"), ("dim", "dim", 45, "e"),
                                         ("knn", "kNN acc", 65, "e"),
                                         ("centroid", "centroid acc", 80, "e"),
                                         ("sil", "silhouette", 70, "e")):
            self.inspect_eval_tree.heading(cid, text=text)
            self.inspect_eval_tree.column(cid, width=width, anchor=anchor,
                                          stretch=(cid == "emb"))
        self.inspect_eval_tree.pack(fill="x", padx=4, pady=(0, 4))
        self.inspect_eval_tree.bind("<Double-1>", self._on_inspect_eval_open)
        attach_tooltip(self.inspect_eval_tree,
                       "Every embedding on offer, scored on the annotated regions: k-NN "
                       "(k=5) and nearest-centroid accuracy in cosine geometry, leave-"
                       "items-out (one item: two random halves), and the mean silhouette. "
                       "Double-click to inspect that embedding.")
        box = ttk.LabelFrame(parent, text="Anchors")
        box.pack(side="top", fill="x", padx=6, pady=4)
        self.inspect_pins_list = tk.Listbox(box, height=4, exportselection=False)
        self.inspect_pins_list.pack(fill="x", padx=4, pady=2)
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(row, text="Remove", command=self._on_inspect_unpin).pack(side="left")
        ttk.Button(row, text="Clear", command=self._inspect_clear).pack(side="left", padx=4)
        ttk.Label(row, text="class centroid:").pack(side="left", padx=(8, 2))
        self.inspect_class_var = tk.StringVar(master=self.root, value="1")
        ttk.Spinbox(row, from_=1, to=MAX_CLASSES - 1, width=3,
                    textvariable=self.inspect_class_var).pack(side="left")
        ttk.Combobox(row, textvariable=self.inspect_scope_var, state="readonly", width=10,
                     values=["this item", "all items"]).pack(side="left", padx=2)
        ttk.Button(row, text="Pin", command=self._on_inspect_pin_class).pack(side="left")

        box = ttk.LabelFrame(parent, text="Most similar regions")
        box.pack(side="top", fill="x", padx=6, pady=4)
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=2)
        ttk.Button(row, text="Find", command=self._fill_inspect_neighbours).pack(side="left")
        ttk.Label(row, text="Label selected as class").pack(side="left", padx=(8, 2))
        self.inspect_label_cls_var = tk.StringVar(master=self.root, value="1")
        ttk.Spinbox(row, from_=1, to=MAX_CLASSES - 1, width=3,
                    textvariable=self.inspect_label_cls_var).pack(side="left")
        ttk.Button(row, text="Label", command=self._on_inspect_label_rows).pack(side="left", padx=2)
        holder = ttk.Frame(box); holder.pack(fill="x", padx=4, pady=(0, 4))
        cols = ("item", "region", "sim", "true", "pred")
        self.inspect_tree = ttk.Treeview(holder, columns=cols, show="headings", height=7,
                                         selectmode="extended")
        for cid, text, width, anchor in (("item", "item", 170, "w"),
                                         ("region", "region", 60, "e"),
                                         ("sim", "similarity", 70, "e"),
                                         ("true", "annotated", 65, "e"),
                                         ("pred", "predicted", 65, "e")):
            self.inspect_tree.heading(cid, text=text)
            self.inspect_tree.column(cid, width=width, anchor=anchor, stretch=(cid == "item"))
        sb = ttk.Scrollbar(holder, orient="vertical", command=self.inspect_tree.yview)
        self.inspect_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.inspect_tree.pack(side="left", fill="x", expand=True)
        self.inspect_tree.bind("<Double-1>", self._on_inspect_row_open)
        self.inspect_tree.bind("<Return>", self._on_inspect_row_open)
        attach_tooltip(self.inspect_tree,
                       "The regions most like the anchors (the scope is the centroid's: "
                       "this item, or every computed item for an embedding that means the "
                       "same on each). Double-click to go there; select rows and Label to "
                       "annotate them.")

        for var in (self.inspect_unit_var,):
            var.trace_add("write", lambda *_a: self._inspect_settings_changed())
        self._refresh_inspect_status()

    def _inspect_view_state(self):
        d = {"similarity": self.inspect_sim_var.get(), "reduce": self.inspect_reduce_var.get(),
             "live": bool(self.inspect_live_var.get())}
        if self.inspect_ref is not None:
            d["embedding"] = self.inspect_ref.to_dict()
        return d

    def _apply_inspect_view(self, d):
        if not isinstance(d, dict):
            return
        if d.get("similarity") in represent.SIMILARITIES:
            self.inspect_sim_var.set(d["similarity"])
        if d.get("reduce") in represent.REDUCES:
            self.inspect_reduce_var.set(d["reduce"])
        if isinstance(d.get("live"), bool):
            self.inspect_live_var.set(d["live"])
        ref = None
        if isinstance(d.get("embedding"), dict):
            try:
                ref = represent.EmbeddingRef.from_dict(d["embedding"])
            except (TypeError, ValueError):
                ref = None
        self.inspect_ref = ref
        self.inspect_label_var.set(self._inspect_label_of(self._inspect_ref_now()))
        self._refresh_inspect_status()

    # -- anchors list ---------------------------------------------------- #
    def _fill_inspect_pins(self):
        lb = getattr(self, "inspect_pins_list", None)
        if lb is None:
            return
        try:
            lb.delete(0, "end")
            for p in self._inspect_pins:
                where = self._slice_key(*self.catalogue.index_of(p["key"])) \
                    if self.catalogue.index_of(p["key"]) is not None else str(p["key"])
                lb.insert("end", f"{p['what']} on {where}")
        except tk.TclError:
            pass

    def _on_inspect_unpin(self):
        sel = self.inspect_pins_list.curselection()
        if sel:
            self._inspect_unpin(int(sel[0]))

    def _on_inspect_pin_class(self):
        try:
            k = int(self.inspect_class_var.get())
        except ValueError:
            return
        self._inspect_pin_class(k, self.inspect_scope_var.get())

    # -- the items an all-items operation walks -------------------------- #
    def _inspect_items(self):
        """``[(si, li, key, rec)]`` of every computed item, or None (the
        reason in the status bar). Never primes (``_stream_stat_slices``)."""
        it = self._stream_stat_slices("Inspecting")
        if it is None:
            return None
        out = []
        try:
            for si, li, key, rec, _table in it:
                out.append((si, li, key, rec))
        except TrainingProblem as exc:
            self.status_var.set(str(exc))
            return None
        return out

    # -- neighbours ------------------------------------------------------ #
    def _inspect_neighbours(self, k=_NEIGHBOURS_K, scope=None):
        """The `k` regions most like the anchors, best first, as row dicts
        ``{si, li, key, item, region, sim, true, pred}``; [] without anchors.
        `scope` "all items" walks every computed item when the embedding
        means the same on each."""
        import numpy as np
        cur = self._current()
        if cur is None:
            return []
        key0 = self.catalogue.key_of(*cur)
        scope = scope or self.inspect_scope_var.get()
        ref = self._inspect_ref_now()
        if scope != "this item" and ref.cross_item:
            items = self._inspect_items() or []
        else:
            items = [(cur[0], cur[1], key0, self.regions.record(key0))]
        kind = self.inspect_sim_var.get()
        reduce = self.inspect_reduce_var.get()
        # The anchors are the item on screen's (its pins, or the hover); with
        # a cross-item embedding they are compared with every item's rows.
        emb0 = self._inspect_embedding(key0, self.regions.record(key0), np)
        A = None if emb0 is None else self._inspect_anchors(key0, emb0, np)[0]
        if A is None:
            return []
        cands = []
        for si, li, key, rec in items:
            emb = emb0 if key == key0 else self._inspect_embedding(key, rec, np, quiet=True)
            if emb is None or emb.dim != A.shape[1]:
                continue
            exclude = [emb.row_of(p["region"]) for p in self._inspect_pins
                       if p["key"] == key and p["region"] >= 0]
            exclude = [r for r in exclude if r is not None]
            rows, sims = represent.top_k(emb.Z, A, kind, np, k=k, reduce=reduce, exclude=exclude)
            truth = self._truth_from_cache(si, li, rec, np)
            pr = self._pred.get(key)
            pred = pr[1] if pr is not None and pr[0] == rec.get("commit") else None
            for r, s in zip(rows.tolist(), sims.tolist()):
                rid = int(emb.ids[r])
                cands.append({"si": si, "li": li, "key": key,
                              "item": self._slice_key(si, li) or f"{si}:{li}",
                              "region": rid, "sim": float(s),
                              "true": int(truth[rid]) if truth is not None and rid < len(truth) else 0,
                              "pred": int(pred[rid]) if pred is not None and rid < len(pred) else 0})
        cands.sort(key=lambda d: -d["sim"])
        return cands[:k]

    def _fill_inspect_neighbours(self):
        tree = getattr(self, "inspect_tree", None)
        if tree is None:
            return
        rows = self._inspect_neighbours()
        self._inspect_rows = rows
        try:
            tree.delete(*tree.get_children())
            for i, d in enumerate(rows):
                tree.insert("", "end", iid=str(i), values=(
                    d["item"], d["region"], f"{d['sim']:.3f}", d["true"] or "",
                    d["pred"] or ""))
        except tk.TclError:
            pass

    def _on_inspect_row_open(self, _e=None):
        sel = self.inspect_tree.selection()
        if not sel:
            return
        try:
            d = self._inspect_rows[int(sel[0])]
        except (ValueError, IndexError):
            return
        self._goto_region(d["si"], d["li"], d["region"])

    def _inspect_label_regions(self, picks, cls):
        """Commit `picks` (``[(si, li, region)]``) as class `cls`: one taps
        gesture per item, a sample (not an extent)."""
        by_item = {}
        for si, li, r in picks:
            by_item.setdefault((si, li), []).append(int(r))
        ref = self._inspect_ref_now()
        n = 0
        for (si, li), ids in by_item.items():
            rec = self.regions.record(self.catalogue.key_of(si, li))
            if rec is None or rec.get("labels") is None:
                continue
            meta = {"tool": "inspect", "extent": False, "n_regions": len(ids),
                    "embedding": ref.to_dict(), "similarity": self.inspect_sim_var.get()}
            self._commit_blob(si, li, rec["labels"], [(ids, int(cls), meta)])
            n += len(ids)
        return n

    def _on_inspect_label_rows(self):
        try:
            cls = int(self.inspect_label_cls_var.get())
        except ValueError:
            return
        sel = self.inspect_tree.selection()
        picks = []
        for iid in sel:
            try:
                d = self._inspect_rows[int(iid)]
            except (ValueError, IndexError):
                continue
            picks.append((d["si"], d["li"], d["region"]))
        if picks:
            self._inspect_label_regions(picks, cls)

    # -- the scatter ----------------------------------------------------- #
    def _inspect_project(self):
        """The item's embedding on its first two principal axes, fitted to
        the scatter box: ``{"P", "xy", "ids", "key", "rows"}`` or None."""
        import numpy as np
        cur = self._current()
        if cur is None:
            return None
        key = self.catalogue.key_of(*cur)
        rec = self.regions.record(key)
        emb = self._inspect_embedding(key, rec, np)
        if emb is None or len(emb.Z) == 0:
            return None
        P = represent.project_2d(emb.Z, np)
        rows = np.arange(len(P))
        if len(rows) > _SCATTER_MAX_POINTS:
            rows = np.sort(np.random.default_rng(0).choice(len(P), _SCATTER_MAX_POINTS,
                                                           replace=False))
        lo = P.min(axis=0)
        span = np.maximum(P.max(axis=0) - lo, 1e-9)
        m = 8
        xy = np.empty_like(P)
        xy[:, 0] = m + (P[:, 0] - lo[0]) / span[0] * (_SCATTER_PX - 2 * m)
        xy[:, 1] = _SCATTER_PX - m - (P[:, 1] - lo[1]) / span[1] * (_SCATTER_PX - 2 * m)
        return {"P": P, "xy": xy, "ids": np.asarray(emb.ids), "key": key, "rows": rows,
                "si": cur[0], "li": cur[1], "rec": rec, "emb": emb}

    def _scatter_colors(self, sc, np):
        """``uint8[n, 3]`` per point by the colour choice."""
        n = len(sc["ids"])
        out = np.tile(np.asarray(_SCATTER_GREY, np.uint8), (n, 1))
        how = self.inspect_color_var.get()
        ids = np.asarray(sc["ids"], np.intp)
        if how == "similarity":
            A, _w = self._inspect_anchors(sc["key"], sc["emb"], np)
            if A is not None:
                kind = self.inspect_sim_var.get()
                s = represent.similarity(sc["emb"].Z, A, kind, np, self.inspect_reduce_var.get())
                lut = scalar_lut(represent.to_unit(s, kind, np), np)
                return lut[:, :3].copy()
            return out
        if how == "annotation":
            cls = self._truth_from_cache(sc["si"], sc["li"], sc["rec"], np)
        else:
            pr = self._pred.get(sc["key"])
            cls = pr[1] if pr is not None and pr[0] == sc["rec"].get("commit") else None
        if cls is None:
            return out
        cls = np.asarray(cls)
        ok = (ids >= 0) & (ids < len(cls))
        k = np.zeros(n, np.intp)
        k[ok] = cls[ids[ok]]
        colors = self._class_colors_rgba(np)[:, :3]
        has = (k > 0) & (k < len(colors))
        out[has] = colors[k[has]]
        return out

    def _draw_inspect_scatter(self):
        """Project the item's embedding and draw it (an RGB raster through
        PIL, so tens of thousands of points stay one image)."""
        import numpy as np
        cv = getattr(self, "inspect_canvas", None)
        sc = self._inspect_project()
        self._inspect_scatter = sc
        self._inspect_sel = []
        if cv is None or sc is None:
            return sc
        img = np.empty((_SCATTER_PX, _SCATTER_PX, 3), np.uint8)
        img[:] = _SCATTER_BG
        cols = self._scatter_colors(sc, np)
        rows = sc["rows"]
        # Unlabelled grey first, coloured points on top.
        order = rows[np.argsort((cols[rows] != _SCATTER_GREY).any(axis=1), kind="stable")]
        xs = np.clip(sc["xy"][order, 0].astype(int), 1, _SCATTER_PX - 2)
        ys = np.clip(sc["xy"][order, 1].astype(int), 1, _SCATTER_PX - 2)
        for dx in (0, 1):
            for dy in (0, 1):
                img[ys + dy, xs + dx] = cols[order]
        try:
            from PIL import Image, ImageTk
            self._scatter_photo = ImageTk.PhotoImage(Image.fromarray(img, mode="RGB"),
                                                     master=self.root)
            cv.delete("all")
            cv.create_image(0, 0, image=self._scatter_photo, anchor="nw", tags=("pts",))
        except (ImportError, tk.TclError):
            return sc
        return sc

    def _scatter_point_at(self, x, y, radius=6.0):
        import numpy as np
        sc = self._inspect_scatter
        if sc is None:
            return None
        rows = sc["rows"]
        d = (sc["xy"][rows, 0] - x) ** 2 + (sc["xy"][rows, 1] - y) ** 2
        i = int(np.argmin(d)) if len(d) else -1
        if i < 0 or d[i] > radius * radius:
            return None
        return int(sc["ids"][rows[i]])

    def _on_scatter_motion(self, e):
        """Scatter hover: outline the region on the canvas."""
        sc = self._inspect_scatter
        if e is None or sc is None:
            return
        rid = self._scatter_point_at(e.x, e.y)
        cur = self._current()
        if rid is None or cur is None or self.catalogue.key_of(*cur) != sc["key"]:
            return
        key = (cur[0], cur[1], rid)
        if key != self._hover_key:
            self._hover_key = key
            self._hover_uid = None
            self._redraw_hover_geometry()

    def _scatter_mark(self, hit):
        """Ring the scatter point of the canvas region under the pointer."""
        cv = getattr(self, "inspect_canvas", None)
        sc = self._inspect_scatter
        if cv is None or sc is None:
            return
        try:
            cv.delete("mark")
            if hit is None or hit[0] != sc["key"]:
                return
            emb = sc["emb"]
            row = emb.row_of(hit[1])
            if row is None:
                return
            x, y = sc["xy"][row]
            cv.create_oval(x - 5, y - 5, x + 5, y + 5, outline="#ffffff", width=2, tags=("mark",))
        except tk.TclError:
            pass

    def _on_scatter_press(self, e):
        self._scatter_drag = (e.x, e.y)

    def _on_scatter_drag(self, e):
        if self._scatter_drag is None:
            return
        x0, y0 = self._scatter_drag
        try:
            self.inspect_canvas.delete("rubber")
            self.inspect_canvas.create_rectangle(x0, y0, e.x, e.y, outline="#ffff66",
                                                 dash=(3, 2), tags=("rubber",))
        except tk.TclError:
            pass

    def _on_scatter_release(self, e):
        if self._scatter_drag is None:
            return
        x0, y0 = self._scatter_drag
        self._scatter_drag = None
        if abs(e.x - x0) < 3 and abs(e.y - y0) < 3:
            rid = self._scatter_point_at(e.x, e.y)
            sc = self._inspect_scatter
            if rid is not None and sc is not None:
                self._goto_region(sc["si"], sc["li"], rid)
            return
        self._scatter_select_box(x0, y0, e.x, e.y)

    def _scatter_select_box(self, x0, y0, x1, y1):
        """Select the drawn points inside a scatter-box (headless-callable);
        their regions preview on the canvas. Returns the region ids."""
        import numpy as np
        sc = self._inspect_scatter
        if sc is None:
            return []
        lo_x, hi_x = sorted((x0, x1))
        lo_y, hi_y = sorted((y0, y1))
        rows = sc["rows"]
        xy = sc["xy"][rows]
        inside = ((xy[:, 0] >= lo_x) & (xy[:, 0] <= hi_x) & (xy[:, 1] >= lo_y) & (xy[:, 1] <= hi_y))
        ids = [int(i) for i in np.asarray(sc["ids"])[rows[inside]]]
        self._inspect_sel = ids
        rec = sc["rec"]
        if ids and rec is not None and rec.get("labels") is not None and self.viewer is not None:
            labels = rec["labels"]
            K = int(labels.max()) + 1 if labels.size else 1
            self._preview_regions(labels, K, ids, (255, 255, 102, 200))
        self.status_var.set(f"inspect: {len(ids)} region(s) selected on the scatter - "
                            "Label selection to annotate them")
        return ids

    def _on_inspect_label_selection(self):
        sc = self._inspect_scatter
        if sc is None or not self._inspect_sel:
            self.status_var.set("inspect: drag a box on the scatter first")
            return
        try:
            cls = int(self.inspect_label_cls_var.get())
        except ValueError:
            return
        if self.viewer is not None:
            self.viewer.set_transient(None)
            self.viewer.invalidate()
        self._inspect_label_regions([(sc["si"], sc["li"], r) for r in self._inspect_sel], cls)
        self._inspect_sel = []

    # -- Evaluate embeddings --------------------------------------------- #
    def _inspect_eval_data(self):
        """``[(option, Z_labelled, cls, groups)]`` for every usable option,
        gathered on the Tk thread (records and caches live here); only the
        annotated rows are kept."""
        import numpy as np
        items = self._inspect_items()
        if items is None:
            return None
        opts = [o for o in self._embedding_options() if o.usable]
        out = []
        for o in opts:
            Zs, ys, gs = [], [], []
            for n, (si, li, key, rec) in enumerate(items):
                truth = self._truth_from_cache(si, li, rec, np)
                if truth is None:
                    continue
                try:
                    emb = self._embedding(key, rec, o.ref, np)
                except Exception:                # noqa: BLE001 -- an option a item cannot fill
                    continue
                ids = np.asarray(emb.ids, np.intp)
                ok = (ids >= 0) & (ids < len(truth))
                cls = np.zeros(len(ids), int)
                cls[ok] = np.asarray(truth)[ids[ok]]
                lab = cls > 0
                if not lab.any():
                    continue
                Zs.append(emb.Z[lab])
                ys.append(cls[lab])
                gs.append(np.full(int(lab.sum()), n))
            if Zs and len({int(v) for y in ys for v in np.unique(y)}) >= 2:
                out.append((o, np.concatenate(Zs), np.concatenate(ys), np.concatenate(gs)))
        return out

    def _evaluate_embeddings(self, sync=False):
        """Score every embedding on offer (``represent.separability``) on a
        worker thread; `sync` runs inline (the selftests)."""
        import numpy as np
        if self._inspect_eval is not None:
            self.status_var.set("Evaluate embeddings is already running")
            return False
        data = self._inspect_eval_data()
        if not data:
            self.inspect_eval_var.set("needs annotated regions of two classes")
            return False
        q = queue.Queue()

        def work():
            for o, Z, y, g in data:
                try:
                    q.put(("row", o, represent.separability(Z, y, g, np)))
                except Exception as exc:         # noqa: BLE001 -- report, keep going
                    q.put(("error", o, str(exc)))
            q.put(("done", None, None))

        self._inspect_eval_rows = []
        self.inspect_eval_var.set(f"scoring {len(data)} embedding(s)...")
        if sync:
            work()
            self._inspect_eval = {"q": q}
            self._inspect_eval_pump(sync=True)
            return True
        self._inspect_eval = {"q": q, "thread": threading.Thread(target=work, daemon=True)}
        self._inspect_eval["thread"].start()
        self.root.after(_EVAL_PUMP_MS, self._inspect_eval_pump)
        return True

    def _inspect_eval_pump(self, sync=False):
        st = self._inspect_eval
        if st is None:
            return
        done = False
        while True:
            try:
                kind, o, res = st["q"].get_nowait()
            except queue.Empty:
                break
            if kind == "done":
                done = True
                break
            if kind == "row" and res is not None:
                self._inspect_eval_rows.append((o, res))
        self._fill_inspect_eval()
        if done:
            self._inspect_eval = None
            n = len(self._inspect_eval_rows)
            self.inspect_eval_var.set(f"{n} embedding(s) scored, best first")
        elif not sync:
            self.root.after(_EVAL_PUMP_MS, self._inspect_eval_pump)

    def _fill_inspect_eval(self):
        tree = getattr(self, "inspect_eval_tree", None)
        rows = sorted(self._inspect_eval_rows, key=lambda r: -r[1]["knn"])
        self._inspect_eval_rows = rows
        if tree is None:
            return
        try:
            tree.delete(*tree.get_children())
            for i, (o, res) in enumerate(rows):
                tree.insert("", "end", iid=str(i), values=(
                    o.label, o.dim if o.dim is not None else "", f"{res['knn']:.3f}",
                    f"{res['centroid']:.3f}", f"{res['silhouette']:.3f}"))
        except tk.TclError:
            pass

    def _on_inspect_eval_open(self, _e=None):
        sel = self.inspect_eval_tree.selection()
        if not sel:
            return
        try:
            o, _res = self._inspect_eval_rows[int(sel[0])]
        except (ValueError, IndexError):
            return
        self._set_inspect_embedding(o.ref)
