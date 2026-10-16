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

from .. import context, edge_model, magic_fill, model_search, fields
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
        parent = self._tab_scroll(parent)
        box = ttk.LabelFrame(parent, text="Region classifier")
        box.pack(side="top", fill="x", padx=6, pady=4)
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=(4, 2))
        ttk.Label(row, text="Kind:").pack(side="left")
        self.model_kind_combo = ttk.Combobox(row, textvariable=self.model_kind_var,
                                             state="readonly", values=_MODEL_KINDS,
                                             width=22)
        self.model_kind_combo.pack(side="left", padx=4)
        self.model_kind_combo.bind("<<ComboboxSelected>>", self._unfocus_entries)
        # The edge model is its own choice, beside the base it sits on.
        self.edge_on_chk = ttk.Checkbutton(row, text="+ edge model", variable=self.edge_on_var)
        self.edge_on_chk.pack(side="left", padx=(8, 0))
        attach_tooltip(self.edge_on_chk,
                       "Train (R) also fits an edge model on top of the base: a pair "
                       "classifier over the region graph, on the base net's hidden-layer "
                       "embedding, giving p(different class) per pair of touching regions "
                       "(the 'learned' magic-fill metric, the boundary colourings, and "
                       "neighbour voting when that is on). Any dense base; a forest has no "
                       "hidden layer. Optimize / size-sweep winners get one too.")
        self.edge_on_var.trace_add("write", lambda *_: self._on_edge_toggle())
        self.voting_var.trace_add("write", lambda *_: self._on_voting_toggle())
        # The custom FC's hidden sizes: shown only for the custom kinds
        # (_sync_model_frames), with or without an edge model on top.
        row = self.custom_hidden_row = ttk.Frame(box)
        ttk.Label(row, text="hidden layers:").pack(side="left")
        en = ttk.Entry(row, textvariable=self.custom_hidden_var, width=10)
        en.pack(side="left", padx=4)
        en.bind("<Return>", lambda e: (self._unfocus_entries(), self._on_edge_settings_change()))
        en.bind("<FocusOut>", lambda e: self._on_edge_settings_change())
        attach_tooltip(en, "Hidden layer sizes of the custom FC net (e.g. 16-8).")
        self.model_arch_var = tk.StringVar(master=self.root, value="")
        self.model_arch_label = ttk.Label(box, textvariable=self.model_arch_var,
                                          justify="left", wraplength=360,
                                          foreground="#333")
        self.model_arch_label.pack(anchor="w", padx=6, pady=(0, 6))
        # A trace rather than the combobox event: Load classifier… sets the
        # kind programmatically and the readout must follow that too.
        self.model_kind_var.trace_add("write", lambda *_: self._on_kind_change())
        self._refresh_model_readout()
        self._build_context_panel(parent)
        self._build_edge_panel(parent)

        # -- Optimize network: the search over the dense FC space ------------ #
        srch = self.optimize_frame = ttk.LabelFrame(parent, text="Optimize network (dense FC search)")
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
        self._sync_model_frames()

    def _sync_model_frames(self):
        """Show only the Model-tab frames that apply to the picked kind: the
        custom sizes for a custom FC, the latent ring head for a dense base
        (or while it is ticked, so it can be unticked), the edge frame while
        '+ edge model' is ticked or an edge model exists (voting off keeps it,
        and its lambda / rounds / Evaluate stay reachable). '+ edge model'
        is disabled on a forest unless ticked."""
        opt = getattr(self, "optimize_frame", None)
        if opt is None:
            return
        base = _base_kind(self.model_kind_var.get())
        flat = base in _FLAT_KINDS
        self.edge_on_chk.state(["disabled"] if flat and not self.edge_on_var.get()
                               else ["!disabled"])
        if base == _CUSTOM_KIND:
            if not self.custom_hidden_row.winfo_manager():
                self.custom_hidden_row.pack(fill="x", padx=4, pady=(0, 2),
                                            before=self.model_arch_label)
        else:
            self.custom_hidden_row.pack_forget()
        edge = getattr(self, "_edge_model", None)
        show = ((self.latent_frame, not flat or bool(self.context_latent_var.get())),
                (self.edge_frame, bool(self.edge_on_var.get()) or edge is not None))
        if edge is None:
            title = ("Edge model: not trained yet - Train (R)" if not flat
                     else "Edge model: a forest has no hidden layer - pick a dense kind")
        elif self.edge_on_var.get():
            title = "Edge model: trained on this base"
        else:
            title = "Edge model: trained, but '+ edge model' is off - Train (R) drops it"
        self.edge_frame.configure(text=title)
        # Re-pack in order, each before the Optimize frame, so the visible
        # ones keep their relative order.
        for frame, _on in show:
            frame.pack_forget()
        for frame, on in show:
            if on:
                frame.pack(side="top", fill="x", padx=6, pady=4, before=opt)

    def _refresh_model_readout(self):
        var = getattr(self, "model_arch_var", None)
        if var is not None:
            var.set(self._model_description_now(self.model_kind_var.get()))

    def _model_description_now(self, kind):
        """`_model_description` for the app's state: the tuned kind renders the
        last search's winner over the trained model's column count; the custom
        kind its typed sizes; '+ edge model' adds the stacked pair model."""
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
        if self.edge_on_var.get() and base not in _FLAT_KINDS:
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
        ctx = self._context_spec_from_ui()
        have = getattr(self, "_clf_context", None)
        if not ctx.empty():
            text += " + " + ctx.describe(self._clf_names if self._clf is not None else None)
        latent = getattr(self, "_context_model", None)
        if latent is not None:
            text += " -> " + latent.describe()
        elif self._clf is not None and have is not None and have.latent is not None:
            text += " [latent head not fit - Train (R)]"
        if self._clf is not None and have is not None and have != ctx:
            text += " [context changed - Train (R) to apply]"
        return text

    # -- neighbourhood context ------------------------------------------- #
    def _context_spec_from_ui(self):
        kinds = tuple(k for k in context.KINDS if self.context_kind_vars[k].get())
        weights = tuple(w for w in context.WEIGHTS if self.context_weight_vars[w].get())
        source = self.context_source_var.get()
        if source not in context.SOURCES:
            source = "all"
        latent = None
        if self.context_latent_var.get():
            w = self.context_latent_weight_var.get()
            dist = self.context_latent_distance_var.get()
            latent = context.LatentSpec(
                layer=context.LATENT_LAYERS.get(self.context_latent_layer_var.get(), -1),
                weight=w if w in context.LATENT_WEIGHTS else "uniform",
                h0=bool(self.context_latent_h0_var.get()),
                z=_bounded_float(self.context_latent_z_var.get(), 1.0, -3.0, 10.0),
                distance=dist if dist in context.LATENT_DISTANCES else "euclidean",
                hidden=self._latent_hidden())
        labels = None
        if self.context_labels_var.get():
            labels = context.LabelSpec(
                dropout=_bounded_float(self.context_label_dropout_var.get(), 0.3, 0.0, 0.95),
                seed=int(self._search_settings()[2]))
        return context.ContextSpec(kinds=kinds, weights=weights or ("uniform",), source=source,
                                   latent=latent, labels=labels,
                                   foundation=self._foundation_spec_from_ui())

    def _apply_context_spec(self, spec):
        """Push a ContextSpec onto its controls: the Features tab's Context
        group (columns, labels) and the Model tab's latent ring head."""
        for k, var in self.context_kind_vars.items():
            var.set(k in spec.kinds)
        for w, var in self.context_weight_vars.items():
            var.set(w in spec.weights)
        self.context_source_var.set(spec.source if spec.source in context.SOURCES else "all")
        self.context_latent_var.set(spec.latent is not None)
        if spec.latent is not None:
            self.context_latent_weight_var.set(spec.latent.weight)
            for name, idx in context.LATENT_LAYERS.items():
                if idx == spec.latent.layer:
                    self.context_latent_layer_var.set(name)
            self.context_latent_h0_var.set(bool(spec.latent.h0))
            self.context_latent_distance_var.set(spec.latent.distance)
            self.context_latent_z_var.set(f"{spec.latent.z:g}")
            self.context_latent_hidden_var.set(
                "-".join(str(h) for h in spec.latent.hidden) if spec.latent.hidden else "")
        self.context_labels_var.set(spec.labels is not None)
        if spec.labels is not None:
            self.context_label_dropout_var.set(f"{spec.labels.dropout:g}")
        # The foundation group is shared by both task kinds: a region task's
        # spec rides its context, a polyline task's its seam spec.
        if self._task_kind() != "polyline":
            self._apply_foundation_spec(spec.foundation)
        self._sync_model_frames()
        self._refresh_model_readout()

    def _on_context_settings_change(self, *_a):
        """Context settings edited: they apply at the next Train (the readout
        says so); nothing cached changes."""
        self._refresh_model_readout()
        self._refresh_model_strip()

    CONTEXT_GROUP_TITLE = "9. Context"

    def _build_context_features_group(self):
        """The neighbourhood context columns (context.py) -- which reductions
        over a region's arc neighbours join its feature row, under which
        weighting, over which source columns, and the ring's annotations --
        as the LAST group of the Features tab: they are columns of the row,
        like everything above them, built from the columns above them. A
        region task's (the group hides for a polyline task). The latent ring
        head, which needs a trained net, stays on the Model tab."""
        box = self._group(self.feat_col, self.CONTEXT_GROUP_TITLE, key="context")
        self._context_group = self._proc_groups["context"]
        self.context_frame = box
        ttk.Label(box, foreground="#666", wraplength=380, justify="left",
                  text="Columns built from each region's neighbours. Like the rest "
                       "of this tab they describe the task's NEXT model: they apply "
                       "at Train (R).").pack(anchor="w", padx=4, pady=(2, 0))
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=(4, 2))
        lbl = ttk.Label(row, text="ring:")
        lbl.pack(side="left")
        attach_tooltip(lbl, "Reductions over the regions that touch this one (its arc "
                            "neighbours). Each adds one column per source column; the "
                            "Optimize feature mask can switch each kind off again.")
        for kind, label, tip in (
                ("ring_mean", "mean", "Weighted mean of the neighbours' values."),
                ("ring_contrast", "contrast", "Own value minus the ring mean: how this "
                                              "region differs from what surrounds it."),
                ("ring_min", "min", "Smallest neighbour value."),
                ("ring_max", "max", "Largest neighbour value."),
                ("ring_std", "std", "Spread of the neighbours' values (weighted).")):
            chk = ttk.Checkbutton(row, text=label, variable=self.context_kind_vars[kind],
                                  command=self._on_context_settings_change)
            chk.pack(side="left", padx=(2, 4))
            attach_tooltip(chk, tip)
        ttk.Label(row, text=" ").pack(side="left")
        for kind, label, tip in (
                ("hop2_mean", "2-hop mean", "The weighted mean applied twice: what the "
                                            "neighbours' neighbourhoods look like."),
                ("slice_mean", "slice mean", "The item-wide mean of each source column, "
                                             "the same on every region of the item."),
                ("slice_contrast", "slice contrast", "Own value minus the item-wide mean.")):
            chk = ttk.Checkbutton(row, text=label, variable=self.context_kind_vars[kind],
                                  command=self._on_context_settings_change)
            chk.pack(side="left", padx=(2, 4))
            attach_tooltip(chk, tip)
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=(0, 4))
        lbl = ttk.Label(row, text="weights:")
        lbl.pack(side="left")
        attach_tooltip(lbl, "How much each neighbour counts. More than one gives a "
                            "separate column set per weighting (named ring_mean[contact]__…).")
        for w, label, tip in (
                ("uniform", "uniform", "Every touching region counts once (needs only the arcs)."),
                ("area", "area", "By the neighbour's pixel count."),
                ("contact", "contact", "By the length of the shared boundary, measured on "
                                       "the label raster (derived once per slice, cached).")):
            chk = ttk.Checkbutton(row, text=label, variable=self.context_weight_vars[w],
                                  command=self._on_context_settings_change)
            chk.pack(side="left", padx=(2, 4))
            attach_tooltip(chk, tip)
        ttk.Label(row, text="   source:").pack(side="left")
        combo = ttk.Combobox(row, textvariable=self.context_source_var, state="readonly",
                             values=context.SOURCES, width=6)
        combo.pack(side="left", padx=4)
        combo.bind("<<ComboboxSelected>>",
                   lambda e: (self._unfocus_entries(), self._on_context_settings_change()))
        attach_tooltip(combo, "Which of the row's columns the context is built over: every "
                              "non-positional column, only the ext_* columns (the seeding "
                              "extremum: what the region looks like away from its boundary), "
                              "or only the mean_* columns.")
        # Labels as context: the ring's annotations as columns.
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=(0, 4))
        chk = ttk.Checkbutton(row, text="labels as context", variable=self.context_labels_var,
                              command=self._on_context_settings_change)
        chk.pack(side="left")
        attach_tooltip(chk, "One column per class: the fraction of the ring annotated with "
                            "it (plus the fraction annotated at all). A region's own label "
                            "never enters. Predictions then depend on the annotations, so "
                            "they clear whenever the store changes -- Classify (C) refreshes "
                            "them with everything drawn so far.")
        ttk.Label(row, text=" dropout:").pack(side="left")
        en = ttk.Entry(row, textvariable=self.context_label_dropout_var, width=5)
        en.pack(side="left", padx=2)
        en.bind("<Return>", lambda e: (self._unfocus_entries(), self._on_context_settings_change()))
        en.bind("<FocusOut>", lambda e: self._on_context_settings_change())
        attach_tooltip(en, "At training, each labeled neighbour is hidden with this "
                           "probability (0..0.95), so the net learns to work with a partly "
                           "labeled ring -- what it meets on a fresh slice.")

    def _build_context_panel(self, parent):
        """The Model tab's half of the context: the latent ring head, a second
        net on the trained base net's embedding of the ring. (The context
        COLUMNS are on the Features tab: _build_context_features_group.)"""
        box = ttk.LabelFrame(parent, text="Latent ring head (on the trained net)")
        box.pack(side="top", fill="x", padx=6, pady=4)
        self.latent_frame = box
        # The latent ring: a second head on the base net's embedding of the ring.
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=(4, 4))
        chk = ttk.Checkbutton(row, text="latent ring head", variable=self.context_latent_var,
                              command=lambda: (self._sync_model_frames(),
                                               self._on_context_settings_change()))
        chk.pack(side="left")
        attach_tooltip(chk, "After the base net is fit, embed every region with its hidden "
                            "layer, average the ring's embeddings (what KIND of material "
                            "surrounds this one) and fit a second net on the row plus those "
                            "columns. Needs a dense base (not a forest). These columns come "
                            "from the model, not the profile, so they are not in the "
                            "fingerprint; the head rides the pickle.")
        ttk.Label(row, text=" layer:").pack(side="left")
        combo = ttk.Combobox(row, textvariable=self.context_latent_layer_var, state="readonly",
                             values=tuple(context.LATENT_LAYERS), width=8)
        combo.pack(side="left", padx=2)
        combo.bind("<<ComboboxSelected>>",
                   lambda e: (self._unfocus_entries(), self._on_context_settings_change()))
        attach_tooltip(combo, "Which hidden layer embeds a region (last = the narrow one).")
        ttk.Label(row, text=" weight:").pack(side="left")
        combo = ttk.Combobox(row, textvariable=self.context_latent_weight_var, state="readonly",
                             values=context.LATENT_WEIGHTS, width=8)
        combo.pack(side="left", padx=2)
        combo.bind("<<ComboboxSelected>>",
                   lambda e: (self._unfocus_entries(), self._on_context_settings_change()))
        attach_tooltip(combo, "How the ring's embeddings are averaged: uniform, by area, by "
                              "contact length, or `latent` = softmax over the latent distance "
                              "(neighbours of the region's own kind count more).")
        chk = ttk.Checkbutton(row, text="H0 shape", variable=self.context_latent_h0_var,
                              command=self._on_context_settings_change)
        chk.pack(side="left", padx=(6, 0))
        attach_tooltip(chk, "Four columns from the single-linkage (H0) filtration of the "
                            "region + its ring in latent space: the largest merge, its ratio "
                            "to the second, the region's own attach distance (an outlier "
                            "against its ring reads high) and the component count at "
                            "mean + std of the slice's arc distances (a ring with two kinds "
                            "of neighbour reads 2).")
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Label(row, text="distance:").pack(side="left")
        combo = ttk.Combobox(row, textvariable=self.context_latent_distance_var, state="readonly",
                             values=context.LATENT_DISTANCES, width=9)
        combo.pack(side="left", padx=2)
        combo.bind("<<ComboboxSelected>>",
                   lambda e: (self._unfocus_entries(), self._on_context_settings_change()))
        attach_tooltip(combo, "How far apart two regions sit in latent space -- for the H0 "
                              "shape, the `latent` weighting and the component threshold. "
                              "euclidean: on the raw activations, so a few large units "
                              "dominate and a strongly-activated region is far from "
                              "everything. cosine (1 - cos) / l2n (euclidean between "
                              "length-1 vectors): direction only, 'what kind of thing', "
                              "not how strongly. The ring MEAN is unaffected.")
        ttk.Label(row, text=" z:").pack(side="left")
        en = ttk.Entry(row, textvariable=self.context_latent_z_var, width=5)
        en.pack(side="left", padx=2)
        en.bind("<Return>", lambda e: (self._unfocus_entries(), self._on_context_settings_change()))
        en.bind("<FocusOut>", lambda e: self._on_context_settings_change())
        attach_tooltip(en, "H0 components: a merge counts as a separate component when it "
                           "is longer than mean + z * std of the item's arc distances "
                           "(-3..10; default 1). Lower = more components.")
        ttk.Label(row, text=" head hidden:").pack(side="left")
        en = ttk.Entry(row, textvariable=self.context_latent_hidden_var, width=8)
        en.pack(side="left", padx=2)
        en.bind("<Return>", lambda e: (self._unfocus_entries(), self._on_context_settings_change()))
        en.bind("<FocusOut>", lambda e: self._on_context_settings_change())
        attach_tooltip(en, "Hidden layer sizes of the head net (e.g. 32-16). Empty = the "
                           "base net's own architecture. The head's input is wider than the "
                           "base's (the row + the embedding width + 4 H0 columns), so it may "
                           "want more room than a small base -- or less than a big tuned one.")

    def _custom_hidden(self):
        """The custom base's hidden sizes from the Model tab entry (first rung
        of a ladder-style string), (16, 8) when unreadable."""
        try:
            return model_search.parse_sizes(self.custom_hidden_var.get())[0]
        except (ValueError, IndexError):
            return tuple(model_search.parse_sizes(_DEFAULT_CUSTOM_HIDDEN)[0])

    def _latent_hidden(self):
        """The latent head's own hidden sizes from its entry, or None (empty
        or unreadable) = the base's architecture."""
        text = self.context_latent_hidden_var.get().strip()
        if not text:
            return None
        try:
            sizes = tuple(int(h) for h in model_search.parse_sizes(text)[0])
        except (ValueError, IndexError, TypeError):
            return None
        return sizes if sizes and all(h > 0 for h in sizes) else None

    def _edge_spec_from_ui(self):
        layer = edge_model.LAYER_CHOICES.get(self.edge_layer_var.get(), -1)
        feats = tuple(f for f in edge_model.FEATURE_KINDS if self.edge_feat_vars[f].get())
        model = self.edge_model_var.get()
        if model not in edge_model.EDGE_MODELS:
            model = "logistic"
        c = _bounded_float(self.edge_c_var.get(), 1.0, *_EDGE_C_RANGE)
        lam = _bounded_float(self.edge_lam_var.get(), 1.0, *_EDGE_LAM_RANGE)
        rounds = int(round(_bounded_float(self.edge_rounds_var.get(), 3.0, *_EDGE_ROUNDS_RANGE)))
        return edge_model.EdgeSpec(layer=int(layer), features=feats or edge_model.DEFAULT_FEATURES,
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
        """The kind combobox: the frames and the readout follow (voting is
        its own checkbox, so nothing re-votes)."""
        self._sync_model_frames()
        self._refresh_model_readout()
        self._refresh_edge_readout()

    def _on_edge_toggle(self):
        """'+ edge model': applies at the next Train; the frame and the
        readout follow at once."""
        self._sync_model_frames()
        self._refresh_model_readout()
        self._refresh_edge_readout()

    def _on_voting_toggle(self):
        """The voting checkbox / N: the cached predictions re-vote in place."""
        if getattr(self, "_edge_model", None) is not None and getattr(self, "_pred", None):
            self._revote_all()
        self._refresh_edge_readout()

    def _build_edge_panel(self, parent):
        """The edge model ('+ edge model' beside the kind): voting, the pair
        model's settings, an evaluation report. Its title says whether one is
        trained (_sync_model_frames)."""
        box = self.edge_frame = ttk.LabelFrame(parent, text="Edge model")
        box.pack(side="top", fill="x", padx=6, pady=4)
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=(4, 2))
        self.voting_chk = ttk.Checkbutton(row, text="neighbour voting (N)",
                                          variable=self.voting_var)
        self.voting_chk.pack(side="left", padx=(0, 10))
        attach_tooltip(self.voting_chk,
                       "Refine the predictions with the edge model: regions the pair "
                       "model says are the same class pull each other's class "
                       "probabilities together (lambda, rounds below). Off = the base "
                       "net's own answers; p(diff) is still computed either way. Applies "
                       "at once to cached predictions.")
        chk = ttk.Checkbutton(row, text="freeze base", variable=self.freeze_base_var,
                              command=self._on_edge_settings_change)
        chk.pack(side="left", padx=(0, 10))
        attach_tooltip(chk, "Train (R) keeps the current base net and refits only the edge "
                            "model: try edge variants on the same base.")
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
                                       "and |ext_a - ext_b|. Zero on pixel adjacency."),
                ("contact", "contact", "log(1 + shared boundary length) of the two regions, "
                                       "measured on the label raster (saddle-free geometry; "
                                       "with barrier off the pair model reads no saddle).")):
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
        # Every edge-model install / load / drop lands here: the edge frame's
        # visibility follows it.
        self._sync_model_frames()
        var = getattr(self, "edge_readout_var", None)
        if var is None:
            return
        edge = self._edge_model
        if edge is None:
            var.set("edges: not trained - Train (R)" if self.edge_on_var.get()
                    else "edges: none - tick '+ edge model' (Model tab) and Train (R)")
            return
        text = edge.brief()
        active = self._voting_on()
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
        ctx = getattr(self, "_clf_context", None)
        if ctx is not None and not ctx.empty():
            bits.append(ctx.brief())
        # The model's own context columns are expected too (the gate's view).
        expected = self._expected_names_for(None)
        if expected is not None and set(expected) != set(names):
            bits.append("⚠ profile mismatch")
        if self._edge_model is not None:
            bits.insert(1, "-> edges")
        var.set(" · ".join(bits))
        self._refresh_hints()

    def _switch_profile(self, idx):
        super()._switch_profile(idx)
        self._refresh_model_strip()
