"""The Model tab: the classifier kind, its architecture readout, the edge-model
panel and the provenance strip above Train/Classify."""
from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .. import edge_model, magic_fill, model_search, fields
from .. import bundle as model_bundle
from ..labeling import (LabelStore, MAX_CLASSES, TOOLS, resolve_slice, resolve_sets,
                          touched_sets, class_lut, scalar_lut, line_pixels, polygon_mask,
                          preview_lut)
from ..training import TrainingSetBuilder, TrainingProblem
from ..widgets import ScrollFrame, attach_tooltip
from ..defaults import *  # noqa: F401,F403
from ..tools import DrawController, MagicFillController, _extremum_points


class ModelPanelMixin:

    def _build_model_tab(self, parent):
        """Model design. For now: which estimator kind Train builds, and a
        read-only description of its architecture (formatted from the same
        constants _make_model uses)."""
        box = ttk.LabelFrame(parent, text="Region classifier")
        box.pack(side="top", fill="x", padx=6, pady=4)
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=(4, 2))
        ttk.Label(row, text="Kind:").pack(side="left")
        self.model_kind_combo = ttk.Combobox(row, textvariable=self.model_kind_var,
                                             state="readonly", values=_MODEL_KINDS,
                                             width=22)
        self.model_kind_combo.pack(side="left", padx=4)
        self.model_kind_combo.bind("<<ComboboxSelected>>", self._unfocus_entries)
        self.model_arch_var = tk.StringVar(master=self.root, value="")
        self.model_arch_label = ttk.Label(box, textvariable=self.model_arch_var,
                                          justify="left", wraplength=700,
                                          foreground="#333")
        self.model_arch_label.pack(anchor="w", padx=6, pady=(0, 6))
        # A trace rather than the combobox event: Load classifier… sets the
        # kind programmatically and the readout must follow that too.
        self.model_kind_var.trace_add("write", lambda *_: self._on_kind_change())
        self._refresh_model_readout()
        self._build_edge_panel(parent)

        # -- Optimize network: the search over the dense FC space ------------ #
        srch = ttk.LabelFrame(parent, text="Optimize network (dense FC search)")
        srch.pack(side="top", fill="x", padx=6, pady=4)
        row = ttk.Frame(srch); row.pack(fill="x", padx=4, pady=(4, 2))
        for label, var, width, tip in (
                ("Trials:", self.search_trials_var, 6,
                 "How many architectures to score. Trial 1 is always the "
                 "un-tuned dense FC baseline, so the winner never loses to it."),
                ("Time limit (min):", self.search_timeout_var, 6,
                 "Stop after this many minutes (the current trial finishes); "
                 "0 = no limit. The best so far is installed, saved under the "
                 "session's models folder and classified either way -- leave "
                 "it running overnight."),
                ("Seed:", self.search_seed_var, 4,
                 "Seeds the sampler, the folds and the networks: the same "
                 "labels and seed reproduce the same winner.")):
            ttk.Label(row, text=label).pack(side="left")
            ent = ttk.Entry(row, textvariable=var, width=width)
            ent.pack(side="left", padx=(2, 8))
            attach_tooltip(ent, tip)
        ttk.Label(row, text="Backend:").pack(side="left")
        self.search_backend_combo = ttk.Combobox(
            row, textvariable=self.search_backend_var, state="readonly",
            values=model_search.BACKENDS, width=7)
        self.search_backend_combo.pack(side="left", padx=(2, 8))
        self.search_backend_combo.bind("<<ComboboxSelected>>", self._unfocus_entries)
        attach_tooltip(self.search_backend_combo,
                       "torch: the GPU MLP (every fold of a trial trains as one "
                       "stacked batch; adds dropout to the search); sklearn: "
                       "MLPClassifier on the CPU; auto: torch when installed. "
                       f"Now: {model_search.backend_label('auto')}.")
        chk = ttk.Checkbutton(row, text="search feature subset",
                              variable=self.search_features_var)
        chk.pack(side="left", padx=(0, 8))
        attach_tooltip(chk, "Also search which measurement CHANNELS the network "
                            "sees (all reductions of a channel together); off = "
                            "every non-positional field.")
        row = ttk.Frame(srch); row.pack(fill="x", padx=4, pady=2)
        self.optimize_btn = ttk.Button(row, text="Optimize network (O)",
                                       command=self._optimize_network)
        self.optimize_btn.pack(side="left", padx=(0, 4))
        attach_tooltip(self.optimize_btn,
                       "Cross-validate candidate dense networks -- depth, widths, "
                       "L2 alpha, learning rate, batch size, early stopping, "
                       "feature subset -- leaving whole slices out, then install "
                       "the winner as the 'dense (tuned)' model, save it and "
                       "classify. Runs in the background; labeling stays live.\n"
                       f"Each trial trains at most {model_search.SEARCH_MAX_ITER} "
                       f"epochs (patience {model_search.SEARCH_PATIENCE}); mini-"
                       f"batches under {model_search.TORCH_MIN_BATCH} rows train on "
                       "sklearn's CPU MLP, full/large batches on torch; the winner "
                       f"is refit with the full {_MLP_MAX_ITER}-epoch budget.")
        self.cancel_search_btn = ttk.Button(row, text="Cancel", state="disabled",
                                            command=self._cancel_search)
        self.cancel_search_btn.pack(side="left")
        ttk.Label(srch, textvariable=self.search_progress_var, justify="left",
                  wraplength=700, foreground="#333").pack(anchor="w", padx=6,
                                                          pady=(0, 6))

    def _refresh_model_readout(self):
        var = getattr(self, "model_arch_var", None)
        if var is not None:
            var.set(self._model_description_now(self.model_kind_var.get()))

    def _model_description_now(self, kind):
        """`_model_description` for the app's state: the tuned kind renders the
        last search's winner over the trained model's column count; the custom
        kind its typed sizes; an edge kind adds the stacked pair model."""
        base = _base_kind(kind)
        n = len(self._clf_names) if self._clf_names else None
        if base == _TUNED_KIND:
            spec = self._search_spec
            text = _model_description(_TUNED_KIND, spec, n if spec is not None else None)
        elif base == _CUSTOM_KIND:
            text = (f"custom FC: FeatureSubset(all) -> StandardScaler -> MLP(hidden layers "
                    f"{self._custom_hidden()}, max_iter {_MLP_MAX_ITER}), fit with balanced "
                    f"sample weights; backend {model_search.backend_label(self._search_settings()[4])}.")
        else:
            text = _model_description(kind)
        if _is_edge_kind(kind):
            widths = None
            try:
                widths = edge_model.hidden_widths(self._clf) if self._clf is not None else None
            except TypeError:
                widths = None
            spec = self._edge_spec_from_ui()
            edge = self._edge_model
            if edge is not None:
                text += " -> " + edge.describe(widths)
                if edge.spec != spec:
                    text += " [edge settings changed - Train (R) to apply]"
            else:
                text += " -> " + spec.describe(widths) + " [not trained yet - Train (R)]"
            if self.freeze_base_var.get():
                text += " [base frozen: Train refits only the edges]"
        return text

    def _custom_hidden(self):
        """The custom base's hidden sizes from the Model tab entry (first rung
        of a ladder-style string), (16, 8) when unreadable."""
        try:
            return model_search.parse_sizes(self.custom_hidden_var.get())[0]
        except (ValueError, IndexError):
            return tuple(model_search.parse_sizes(_DEFAULT_CUSTOM_HIDDEN)[0])

    def _edge_spec_from_ui(self):
        layer = edge_model.LAYER_CHOICES.get(self.edge_layer_var.get(), -1)
        feats = tuple(f for f in edge_model.FEATURE_KINDS if self.edge_feat_vars[f].get())
        model = self.edge_model_var.get()
        if model not in edge_model.EDGE_MODELS:
            model = "logistic"
        c = _bounded_float(self.edge_c_var.get(), 1.0, *_EDGE_C_RANGE)
        lam = _bounded_float(self.edge_lam_var.get(), 1.0, *_EDGE_LAM_RANGE)
        rounds = int(round(_bounded_float(self.edge_rounds_var.get(), 3.0, *_EDGE_ROUNDS_RANGE)))
        return edge_model.EdgeSpec(layer=int(layer), features=feats or edge_model.FEATURE_KINDS,
                                   model=model, C=float(c), lam=float(lam), rounds=rounds,
                                   seed=int(self._search_settings()[2]))

    def _apply_edge_spec(self, spec):
        """Push an EdgeSpec onto the Model tab controls."""
        for name, idx in edge_model.LAYER_CHOICES.items():
            if idx == spec.layer:
                self.edge_layer_var.set(name)
        for f, var in self.edge_feat_vars.items():
            var.set(f in spec.features)
        self.edge_model_var.set(spec.model)
        self.edge_c_var.set(f"{spec.C:g}")
        self.edge_lam_var.set(f"{spec.lam:g}")
        self.edge_rounds_var.set(str(int(spec.rounds)))

    def _on_kind_change(self):
        """The kind combobox / N: the readout follows, and switching between an
        edge kind and its base re-votes the cached predictions in place."""
        self._refresh_model_readout()
        if getattr(self, "_edge_model", None) is not None and getattr(self, "_pred", None):
            self._revote_all()
        self._refresh_edge_readout()

    def _build_edge_panel(self, parent):
        """The edge model behind the `-> edges` kinds: the custom base, the pair
        model's settings, the voting weight, an evaluation report."""
        box = ttk.LabelFrame(parent, text="Edge model (-> edges kinds): a pair model over the region graph")
        box.pack(side="top", fill="x", padx=6, pady=4)
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=(4, 2))
        ttk.Label(row, text="base hidden:").pack(side="left")
        en = ttk.Entry(row, textvariable=self.custom_hidden_var, width=8)
        en.pack(side="left", padx=(2, 6))
        en.bind("<Return>", lambda e: (self._unfocus_entries(), self._on_edge_settings_change()))
        en.bind("<FocusOut>", lambda e: self._on_edge_settings_change())
        attach_tooltip(en, "Hidden layer sizes of the custom FC base (e.g. 16-8), used by "
                           "the custom FC / custom FC -> edges kinds.")
        chk = ttk.Checkbutton(row, text="freeze base", variable=self.freeze_base_var,
                              command=self._on_edge_settings_change)
        chk.pack(side="left", padx=(0, 10))
        attach_tooltip(chk, "Train (R) on a -> edges kind keeps the current base net and "
                            "refits only the edge model: try edge variants on the same base.")
        ttk.Label(row, text="layer:").pack(side="left")
        cb = ttk.Combobox(row, textvariable=self.edge_layer_var, state="readonly",
                          values=list(edge_model.LAYER_CHOICES), width=8)
        cb.pack(side="left", padx=(2, 6))
        cb.bind("<<ComboboxSelected>>", lambda e: (self._unfocus_entries(), self._on_edge_settings_change()))
        attach_tooltip(cb, "Which hidden layer of the base net embeds a region: last (the "
                           "narrow one, best in the experiment) or the previous, wider one.")
        ttk.Label(row, text="model:").pack(side="left")
        cb = ttk.Combobox(row, textvariable=self.edge_model_var, state="readonly",
                          values=list(edge_model.EDGE_MODELS), width=8)
        cb.pack(side="left", padx=(2, 6))
        cb.bind("<<ComboboxSelected>>", lambda e: (self._unfocus_entries(), self._on_edge_settings_change()))
        attach_tooltip(cb, "The pair classifier: a balanced logistic regression, or an MLP 32-16.")
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=2)
        lbl = ttk.Label(row, text="pair inputs:")
        lbl.pack(side="left")
        attach_tooltip(lbl, "What the pair model sees for an edge between two regions, "
                            "built from the base net's hidden-layer embedding of each "
                            "(symmetric in the two regions).")
        for f, label, tip in (
                ("absdiff", "|d|", "Absolute difference of the two regions' embeddings, "
                                   "element by element: how far apart they sit."),
                ("prod", "product", "Element-wise product of the two embeddings: which "
                                     "features both regions share."),
                ("barrier", "barrier", "The saddle joining the two regions: its depth "
                                       "above their extrema (saddle - max(ext_a, ext_b)) "
                                       "and |ext_a - ext_b|. Zero on pixel adjacency.")):
            chk = ttk.Checkbutton(row, text=label, variable=self.edge_feat_vars[f],
                                  command=self._on_edge_settings_change)
            chk.pack(side="left", padx=(2, 4))
            attach_tooltip(chk, tip)
        for label, var, width, tip in (
                ("C:", self.edge_c_var, 5, "Logistic inverse regularisation (0.001..1000)."),
                ("lambda:", self.edge_lam_var, 5,
                 "Weight of the edge terms in neighbour voting (0 = the base net alone; "
                 "applies at once to cached predictions)."),
                ("rounds:", self.edge_rounds_var, 4,
                 "Voting rounds (0 = no refinement); applies at once.")):
            ttk.Label(row, text=label).pack(side="left", padx=(8, 0))
            en = ttk.Entry(row, textvariable=var, width=width)
            en.pack(side="left", padx=(2, 2))
            en.bind("<Return>", lambda e: (self._unfocus_entries(), self._on_edge_settings_change()))
            en.bind("<FocusOut>", lambda e: self._on_edge_settings_change())
            attach_tooltip(en, tip)
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=2)
        self.edge_eval_btn = ttk.Button(row, text="Evaluate edges", command=self._evaluate_edges)
        self.edge_eval_btn.pack(side="left", padx=(0, 6))
        attach_tooltip(self.edge_eval_btn,
                       "Leave-slices-out report for the pair model against the base net's own "
                       "answers (argmax differs / 1 - sum P_a P_b), and region errors before and "
                       "after voting. Refits the base per fold on a worker thread; Cancel above.")
        ttk.Label(row, textvariable=self.edge_progress_var, foreground="#333",
                  wraplength=560, justify="left").pack(side="left", fill="x", expand=True)
        holder = ttk.Frame(box); holder.pack(fill="x", padx=4, pady=(0, 4))
        cols = ("model", "n", "bacc", "auc", "logloss", "recall", "prec")
        self.edge_tree = ttk.Treeview(holder, columns=cols, show="headings", height=5)
        for cid, text, width, anchor in (("model", "edge model / region", 190, "w"),
                                         ("n", "n", 60, "e"), ("bacc", "bal. acc", 70, "e"),
                                         ("auc", "AUC", 60, "e"), ("logloss", "log-loss", 70, "e"),
                                         ("recall", "diff recall", 80, "e"),
                                         ("prec", "diff prec", 80, "e")):
            self.edge_tree.heading(cid, text=text)
            self.edge_tree.column(cid, width=width, anchor=anchor, stretch=(cid == "model"))
        self.edge_tree.pack(side="left", fill="x", expand=True)
        attach_tooltip(self.edge_tree,
                       "HELD-OUT numbers: every slice is scored by models that never saw "
                       "it (leave-slices-out folds). Edge rows: how well each answer tells "
                       "same-class from different-class edges (diff recall / precision = "
                       "boundary edges found / boundary calls that were right). Region rows: "
                       "accuracy on the labeled regions before and after neighbour voting. "
                       "The confusion matrix on the right is NOT held-out (the model saw "
                       "those labels), so its errors are fewer.")

    def _on_edge_settings_change(self, *_a):
        """Edge settings edited: lambda / rounds apply to the cached predictions
        at once; the rest wait for Train (the readout says so)."""
        self._refresh_model_readout()
        if self._edge_model is not None and self._pred:
            self._revote_all()
        self._refresh_edge_readout()

    def _fill_edge_report(self, report):
        tree = getattr(self, "edge_tree", None)
        if tree is None:
            return
        try:
            tree.delete(*tree.get_children())
        except tk.TclError:
            return
        for name in edge_model.EVAL_ROWS:
            s = (report or {}).get("edge", {}).get(name)
            if s:
                tree.insert("", "end", values=(name, s["n"], f"{s['bacc']:.3f}", f"{s['auc']:.3f}",
                                               f"{s['logloss']:.3f}", f"{s['diff_recall']:.1%}",
                                               f"{s['diff_precision']:.1%}"))
        reg = (report or {}).get("region", {})
        for key, label in (("before", "region net alone (held-out)"),
                           ("after", "+ neighbour voting (held-out)")):
            r = reg.get(key)
            if r:
                tree.insert("", "end", values=(label, r["n"], f"{r['bacc']:.3f}", "",
                                               f"acc {r['acc']:.3f}",
                                               f"{r['errors']} held-out errors", ""))

    def _refresh_edge_readout(self):
        var = getattr(self, "edge_readout_var", None)
        if var is None:
            return
        edge = self._edge_model
        kind = self.model_kind_var.get()
        if edge is None:
            var.set("edges: not trained - Train (R) on this kind" if _is_edge_kind(kind)
                    else "edges: none - pick a '-> edges' kind and Train (R)")
            return
        text = edge.brief()
        active = _is_edge_kind(kind)
        text += " · voting " + ("on" if active else "off") + " (N)"
        cur = self._current()
        entry = self._pred.get(self.catalogue.key_of(*cur)) if cur is not None else None
        aux = self._pred_aux(entry)
        if active and aux is not None and aux.get("flips") is not None:
            text += f" · {aux['flips']} flipped on this slice"
        var.set(text)

    def _refresh_model_strip(self):
        """Repaint the provenance line above Train/Classify. The mismatch check
        is inlined rather than routed through _check_model_compat: this runs on
        every profile switch and must not log."""
        var = getattr(self, "model_strip_var", None)
        if var is None:
            return
        if self._clf is None:
            var.set("no model")
            return
        names = list(self._clf_names or [])
        stats = None
        if self.models:
            last = self.models[-1]
            if list(last.get("fingerprint") or []) == names:
                stats = last.get("statistics") or None
        if stats is None and 0 <= self.active_profile_idx < len(self.profiles):
            stats = self.profiles[self.active_profile_idx].get("statistics")
        bits = [self._clf_kind, f"{len(names)} feats", self._stats_brief(stats)]
        if self._clf_spec is not None:
            bits[1] = self._clf_spec.brief(len(names))
        expected = self._expected_feature_names()
        if expected is not None and set(expected) != set(names):
            bits.append("⚠ profile mismatch")
        if self._edge_model is not None and not _is_edge_kind(self._clf_kind):
            bits.insert(1, "-> edges")
        var.set(" · ".join(bits))
        self._refresh_hints()

    def _switch_profile(self, idx):
        super()._switch_profile(idx)
        self._refresh_model_strip()
