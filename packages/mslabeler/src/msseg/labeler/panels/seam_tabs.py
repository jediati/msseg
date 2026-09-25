"""A polyline task's Model, Analysis and Features content -- the region
tabs' counterparts over seams.

* **Features**: the seam descriptor group (which blocks the seam model reads:
  pair terms over the flank rows or an embedding, the saddle barrier, the
  edge model's p(diff), geometry). The statistics above it are the flank rows.
* **Model**: the seam classifier's kind and settings, and Evaluate with its
  held-out report -- a table per fold plus the mean, and per-class recall /
  precision.
* **Analysis**: the seams behind a confusion cell, on every classified item;
  double-click one to go there (``_goto_seam``, the ``_goto_region`` pattern).

The settings ride the task's view (``seam_spec``), like the region model's.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .. import seam_model
from ..widgets import attach_tooltip
from ..defaults import *  # noqa: F401,F403

_SEAM_KINDS = ("logistic", "mlp")
_SEAM_EMBEDS = ("auto", "features", "net")
_FEATURE_TIPS = {
    "pair": "|difference| and product of the two flank regions' rows (or of their "
            "embedding in the base net's hidden layer, see Model > embed)",
    "barrier": "the arc's saddle depth and the flanks' extremum gap (MSC arcs)",
    "edges": "the '-> edges' model's p(different) for the pair -- needs a region "
             "model with an edge head",
    "geometry": "crack length, its log, the bbox aspect, tortuosity, loop",
}


class SeamTabsMixin:
    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def _init_seam_tab_vars(self, root):
        self.seam_model_kind_var = tk.StringVar(master=root, value="logistic")
        self.seam_C_var = tk.StringVar(master=root, value="1.0")
        self.seam_embed_var = tk.StringVar(master=root, value="auto")
        self.seam_layer_var = tk.StringVar(master=root, value="-1")
        self.seam_feature_vars = {k: tk.BooleanVar(master=root, value=k in seam_model.DEFAULT_FEATURES)
                                  for k in seam_model.FEATURE_KINDS}
        self._seam_error_rows = []

    def _seam_spec(self):
        """The seam model the next Train builds (the Features and Model tabs)."""
        feats = tuple(k for k in seam_model.FEATURE_KINDS
                      if self.seam_feature_vars[k].get()) or seam_model.DEFAULT_FEATURES
        try:
            C = max(1e-4, float(self.seam_C_var.get()))
        except (ValueError, tk.TclError):
            C = 1.0
        try:
            layer = int(self.seam_layer_var.get())
        except (ValueError, tk.TclError):
            layer = -1
        kind = self.seam_model_kind_var.get()
        embed = self.seam_embed_var.get()
        return seam_model.SeamSpec(features=feats,
                                   model=kind if kind in _SEAM_KINDS else "logistic",
                                   C=C, embed=embed if embed in _SEAM_EMBEDS else "auto",
                                   layer=layer)

    def _apply_seam_spec(self, d):
        if not isinstance(d, dict):
            return
        spec = seam_model.SeamSpec.from_dict(d)
        self.seam_model_kind_var.set(spec.model if spec.model in _SEAM_KINDS else "logistic")
        self.seam_C_var.set(f"{spec.C:g}")
        self.seam_embed_var.set(spec.embed if spec.embed in _SEAM_EMBEDS else "auto")
        self.seam_layer_var.set(str(spec.layer))
        for k, var in self.seam_feature_vars.items():
            var.set(k in spec.features)

    # ------------------------------------------------------------------ #
    # Features tab: the seam descriptor
    # ------------------------------------------------------------------ #
    def _build_seam_features_group(self):
        body = self._group(self.feat_col, "6. Seam descriptor", key="seam_descriptor")
        self._seam_desc_group = body.master if body.master is not self.feat_col else body
        ttk.Label(body, foreground="#666", wraplength=380, justify="left",
                  text="What the seam model reads about each seam; the statistics "
                       "above are its two flank regions' rows.").pack(anchor="w", padx=4,
                                                                     pady=(2, 2))
        row = ttk.Frame(body); row.pack(fill="x", padx=4, pady=(0, 4))
        for k in seam_model.FEATURE_KINDS:
            cb = ttk.Checkbutton(row, text=k, variable=self.seam_feature_vars[k])
            cb.pack(side="left", padx=(0, 8))
            attach_tooltip(cb, _FEATURE_TIPS.get(k, k))

    # ------------------------------------------------------------------ #
    # Model tab: the seam classifier and its held-out report
    # ------------------------------------------------------------------ #
    def _build_seam_model_body(self, parent):
        box = ttk.LabelFrame(parent, text="Seam classifier")
        box.pack(side="top", fill="x", padx=6, pady=4)
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=(4, 2))
        ttk.Label(row, text="Kind:").pack(side="left")
        self.seam_kind_combo = ttk.Combobox(row, textvariable=self.seam_model_kind_var,
                                            values=list(_SEAM_KINDS), state="readonly", width=10)
        self.seam_kind_combo.pack(side="left", padx=2)
        ttk.Label(row, text="C").pack(side="left", padx=(8, 2))
        ttk.Entry(row, textvariable=self.seam_C_var, width=6).pack(side="left")
        ttk.Label(row, text="embed").pack(side="left", padx=(8, 2))
        cb = ttk.Combobox(row, textvariable=self.seam_embed_var, values=list(_SEAM_EMBEDS),
                          state="readonly", width=8)
        cb.pack(side="left")
        attach_tooltip(cb, "What the pair terms compare: 'features' = the flank rows "
                           "z-scored; 'net' = the flanks' embedding in a region net's "
                           "hidden layer (a region input, when there is one); 'auto' = "
                           "net when available.")
        ttk.Label(row, text="layer").pack(side="left", padx=(8, 2))
        ttk.Entry(row, textvariable=self.seam_layer_var, width=4).pack(side="left")
        ttk.Label(box, foreground="#666", wraplength=380, justify="left",
                  text="logistic: a balanced multinomial regression; mlp: a (32, 16) "
                       "network. Both over the standardised seam descriptor (Features "
                       "tab). Train (R) on the Annotation tab.").pack(anchor="w", padx=6,
                                                                       pady=(0, 4))

        ev = ttk.LabelFrame(parent, text="Evaluate (held-out)")
        ev.pack(side="top", fill="x", padx=6, pady=4)
        row = ttk.Frame(ev); row.pack(fill="x", padx=4, pady=(4, 2))
        self.seam_eval_btn2 = ttk.Button(row, text="Evaluate seams",
                                         command=self._evaluate_seams)
        self.seam_eval_btn2.pack(side="left")
        attach_tooltip(self.seam_eval_btn2, "Leave-items-out cross-validation of the "
                                            "seam model the next Train builds.")
        self.seam_eval_var = tk.StringVar(master=self.root, value="not evaluated")
        ttk.Label(row, textvariable=self.seam_eval_var, foreground="#555",
                  wraplength=280, justify="left").pack(side="left", padx=6)
        cols = ("fold", "n", "bacc", "auc", "logloss")
        self.seam_report_tree = ttk.Treeview(ev, columns=cols, show="headings", height=6)
        for cid, text, w in (("fold", "fold", 60), ("n", "seams", 60), ("bacc", "bal.acc", 70),
                             ("auc", "boundary AUC", 90), ("logloss", "log-loss", 70)):
            self.seam_report_tree.heading(cid, text=text)
            self.seam_report_tree.column(cid, width=w, anchor="e")
        self.seam_report_tree.pack(fill="x", padx=4, pady=2)
        cols = ("cls", "name", "recall", "precision")
        self.seam_class_tree = ttk.Treeview(ev, columns=cols, show="headings", height=4)
        for cid, text, w, a in (("cls", "class", 50, "e"), ("name", "name", 140, "w"),
                                ("recall", "recall", 70, "e"),
                                ("precision", "precision", 70, "e")):
            self.seam_class_tree.heading(cid, text=text)
            self.seam_class_tree.column(cid, width=w, anchor=a)
        self.seam_class_tree.pack(fill="x", padx=4, pady=(2, 4))

    def _fill_seam_report(self, report):
        tree = getattr(self, "seam_report_tree", None)
        if tree is None:
            return
        for t in (tree, self.seam_class_tree):
            t.delete(*t.get_children())
        if not report:
            self.seam_eval_var.set("not evaluated")
            return

        def f(v, fmt):
            return "" if v is None or v != v else fmt.format(v)
        for k, fd in enumerate(report.get("folds") or [], start=1):
            tree.insert("", "end", values=(k, fd.get("n", ""), f(fd.get("bacc"), "{:.1%}"),
                                           f(fd.get("auc"), "{:.3f}"),
                                           f(fd.get("logloss"), "{:.3f}")))
        m = report.get("mean") or {}
        if m:
            tree.insert("", "end", values=("mean", m.get("n", ""), f(m.get("bacc"), "{:.1%}"),
                                           f(m.get("auc"), "{:.3f}"),
                                           f(m.get("logloss"), "{:.3f}")))
            for c, pc in sorted((m.get("per_class") or {}).items()):
                self.seam_class_tree.insert("", "end", values=(
                    c, self._class_name(int(c)), f(pc.get("recall"), "{:.0%}"),
                    f(pc.get("precision"), "{:.0%}")))
        self.seam_eval_var.set(seam_model.summary(report))

    def _finish_seam_eval(self, report):
        super()._finish_seam_eval(report)
        self._fill_seam_report(report)

    # ------------------------------------------------------------------ #
    # Analysis tab: the seams behind a confusion cell
    # ------------------------------------------------------------------ #
    def _build_seam_analysis_body(self, parent):
        box = ttk.LabelFrame(parent, text="Predictions vs labels: the seams behind a "
                                          "confusion cell")
        box.pack(side="top", fill="x", padx=6, pady=4)
        self.seam_errors_header_var = tk.StringVar(
            master=self.root,
            value="Click a cell of the seam confusion matrix (Annotation tab) to list its "
                  "seams here; double-click a row to go to that seam.")
        ttk.Label(box, textvariable=self.seam_errors_header_var, justify="left",
                  wraplength=760, foreground="#333").pack(anchor="w", padx=6, pady=(4, 2))
        holder = ttk.Frame(box); holder.pack(fill="x", padx=4, pady=(0, 4))
        cols = ("item", "seam", "flanks", "true", "pred", "length", "p_true", "p_pred")
        self.seam_errors_tree = ttk.Treeview(holder, columns=cols, show="headings", height=8,
                                             selectmode="browse")
        for cid, text, width, anchor in (("item", "item", 200, "w"), ("seam", "seam", 60, "e"),
                                         ("flanks", "flanks", 80, "e"),
                                         ("true", "labelled", 70, "e"),
                                         ("pred", "predicted", 70, "e"),
                                         ("length", "length (px)", 80, "e"),
                                         ("p_true", "P(labelled)", 85, "e"),
                                         ("p_pred", "P(predicted)", 85, "e")):
            self.seam_errors_tree.heading(cid, text=text)
            self.seam_errors_tree.column(cid, width=width, anchor=anchor,
                                         stretch=(cid == "item"))
        sb = ttk.Scrollbar(holder, orient="vertical", command=self.seam_errors_tree.yview)
        self.seam_errors_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.seam_errors_tree.pack(side="left", fill="x", expand=True)
        self.seam_errors_tree.bind("<Double-1>", self._on_seam_error_open)
        self.seam_errors_tree.bind("<Return>", self._on_seam_error_open)

    def _seam_cell_rows(self, i, j):
        """The seams counted in confusion cell (labelled i, predicted j) on
        every item classified at its current record, longest first."""
        import numpy as np
        rows = []
        for key, entry in self._seam_pred.items():
            got = self._seam_truth_pred(key, np)
            if got is None:
                continue
            truth, pred, _w = got
            hits = np.nonzero((truth == i) & (pred == j))[0]
            if not len(hits):
                continue
            si, li = self.catalogue.index_of(key)
            graph = self.regions.seams(key, np)
            lengths = graph.lengths(np)
            proba = np.asarray(entry[2], np.float64)
            name = self._slice_key(si, li) or str(key)
            for s in hits.tolist():
                rows.append({"si": si, "li": li, "item": name, "seam": int(s),
                             "flanks": f"{int(graph.a[s])}|{int(graph.b[s])}",
                             "true": int(truth[s]), "pred": int(pred[s]),
                             "length": int(lengths[s]),
                             "p_true": float(proba[s, i]) if i < proba.shape[1] else 0.0,
                             "p_pred": float(proba[s, j]) if j < proba.shape[1] else 0.0})
        rows.sort(key=lambda d: (d["si"], d["li"], -d["length"]))
        return rows

    def _fill_error_list(self, cell):
        if self._task_kind() != "polyline":
            return super()._fill_error_list(cell)
        tree = getattr(self, "seam_errors_tree", None)
        if tree is None:
            return None
        tree.delete(*tree.get_children())
        self._seam_error_rows = []
        if cell is None:
            self.seam_errors_header_var.set("No confusion cell selected.")
            return None
        i, j = cell
        rows = self._seam_cell_rows(i, j)
        self._seam_error_rows = rows
        for k, d in enumerate(rows):
            tree.insert("", "end", iid=str(k), values=(
                d["item"], d["seam"], d["flanks"], d["true"], d["pred"], d["length"],
                f"{d['p_true']:.3f}", f"{d['p_pred']:.3f}"))
        what = "correct" if i == j else "misclassified"
        n_items = len({(d["si"], d["li"]) for d in rows})
        self.seam_errors_header_var.set(
            f"labelled {self._class_name(i)} -> predicted {self._class_name(j)}: {len(rows)} "
            f"{what} seam(s) on {n_items} {self.ITEM_NOUN}(s), longest first -- a fit check "
            f"on the training labels (Evaluate reports held-out numbers). Double-click a "
            f"row to go there.")
        return None

    def _on_seam_error_open(self, _e=None):
        sel = self.seam_errors_tree.selection()
        if not sel:
            return
        try:
            d = self._seam_error_rows[int(sel[0])]
        except (ValueError, IndexError):
            return
        self._goto_seam(d["si"], d["li"], d["seam"])

    def _goto_seam(self, si, li, seam):
        """Show item (si, li) centred on `seam`, drawn highlighted; the tab
        does not change (the list stays beside what it sent you to)."""
        try:
            idx = self.flat_slices.index((si, li))
        except ValueError:
            self.status_var.set(f"{self.ITEM_NOUN} {si}:{li} is not worked")
            return False
        self._goto_slice(idx)
        import numpy as np
        graph = self._seam_graph_for(si, li, np)
        if graph is None or not (0 <= int(seam) < graph.n_seams):
            self.status_var.set(f"seam {seam}: no seam graph here")
            return True
        pts = graph.to_image(graph.seam_points(int(seam)).tolist())
        mid = pts[len(pts) // 2]
        self.status_var.set(f"seam {int(seam)} ({int(graph.a[seam])}|{int(graph.b[seam])}) "
                            f"on {self._slice_key(si, li)}")

        def go():
            v = self.viewer
            if v is None:
                return
            self._fit_pending = False
            v.center_on(mid[0], mid[1])
            v.canvas.delete("ihover")
            flat = []
            for x, y in pts:
                flat.extend(((x - v.view_x) / v.scale, (y - v.view_y) / v.scale))
            if len(flat) >= 4:
                v.canvas.create_line(*flat, fill="#ffffff", width=3, dash=(5, 3),
                                     tags=("draw", "ihover"))
        try:
            self.root.after_idle(go)
        except tk.TclError:
            go()
        return True
