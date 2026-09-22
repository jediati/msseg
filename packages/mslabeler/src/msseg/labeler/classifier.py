"""Train / Classify / Optimize / size sweep / edge evaluation: the region
classifier's whole lifecycle on top of the headless training and bundle
modules, run on a worker thread drained by a Tk pump."""
from __future__ import annotations

import dataclasses
import json
import math
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from . import seam_model
from . import context, edge_model, magic_fill, model_search, fields
from . import bundle as model_bundle
from .labeling import (LabelStore, MAX_CLASSES, TOOLS, resolve_slice, resolve_sets,
                          touched_sets, class_lut, scalar_lut, line_pixels, polygon_mask,
                          preview_lut)
from .training import TrainingSetBuilder, TrainingProblem
from .widgets import ScrollFrame, attach_tooltip
from .defaults import *  # noqa: F401,F403
from .tools import DrawController, MagicFillController, _extremum_points


class ClassifierMixin:

    # ------------------------------------------------------------------ #
    # Classifier: train on the labeled regions, predict every region
    # ------------------------------------------------------------------ #
    def _stream_stat_slices(self, action, spec=None, training=False):
        """Yield ``(si, li, key, rec, table)`` for every primed item, one at a
        time.

        Same contract as ``_all_stat_slices`` -- model operations are
        stack-wide and must never silently degrade to the subset the lazy
        viewer happened to visit -- but the records are not all held alive at
        once. That is the difference between a stack of coupon slices and a
        session of whole-slide items: each record carries a label raster and a
        statistics table, and every consumer here (the training set, the edge
        set, classification) walks them exactly once.

        `spec` is the ``ContextSpec`` whose columns the yielded table carries
        on top of the record's own (see `_context_table`); None or an empty
        spec yields ``rec["stats"]`` itself. `training` says the rows are for
        a fit (the labels block then applies its dropout).

        None when the operation must abort before it starts; ``TrainingProblem``
        when an item cannot be computed part-way through, which is what the
        callers already handle.
        """
        keys = self.regions.keys()
        if not keys:
            self.status_var.set("Run first - no slices are primed.")
            return None
        if self.regions.pending():
            self.status_var.set("Busy computing - try again in a moment.")
            return None
        return self._stat_slice_iter(action, list(keys), spec, training)

    def _stat_slice_iter(self, action, keys, spec=None, training=False):
        import numpy as np
        total = len(keys)
        self._compute_badge(f"{action} 0/{total}")
        try:
            for n, key in enumerate(keys, start=1):
                si, li = self.catalogue.index_of(key)
                self._compute_badge(f"{action} {n}/{total}")
                rec = self._ensure_slice_record(si, li)
                table = None if rec is None else rec.get("stats")
                if (rec is None or rec.get("labels") is None
                        or getattr(table, "values", None) is None):
                    raise TrainingProblem(
                        f"{action} stopped: could not compute slice {si}:{li}.")
                try:
                    table = self._context_table(key, rec, spec, np, training)
                except ValueError as exc:
                    raise TrainingProblem(f"{action} stopped: {exc} (slice {si}:{li}).")
                yield si, li, key, rec, table
        finally:
            self._clear_compute_badge()

    def _context_table(self, key, rec, spec, np, training=False):
        """The record's statistics table plus the context columns `spec` asks
        for (``context.augment``), or ``rec["stats"]`` itself for None / an
        empty spec. Built from the provider's arcs (contact lengths derived
        from the label raster and cached on them only when a weighting needs
        them); the labels block reads the store's classes for the item (with
        a seeded dropout draw when `training`); a small bounded cache keyed
        on the record's commit, the spec and (for labels) the store's
        revision keeps Train-then-Classify from rebuilding the slice on
        screen. Raises ValueError when the spec cannot be built here."""
        table = rec.get("stats")
        if spec is None or spec.empty():
            return table
        cache = getattr(self, "_ctx_cache", None)
        if cache is None:
            cache = self._ctx_cache = {}
        with_labels = spec.labels is not None
        ck = (key, rec.get("commit"), spec.key(),
              (bool(training), self.store.rev) if with_labels else None)
        hit = cache.get(ck)
        if hit is not None and hit[0] is table:
            return hit[1]
        arcs = self.regions.arcs(key, np)
        labels = rec.get("labels")
        extra = None
        if with_labels:
            ids = table.column(self.FIELDS.id_field)
            rc = self._training_builder.row_classes(
                self._gestures_for_key(key), labels, ids, np, self.regions.label_layer(key))
            rng = None
            if training:
                import zlib
                salt = zlib.crc32(str(key).encode("utf-8"))
                rng = np.random.default_rng([int(spec.labels.seed),
                                             int(rec.get("commit") or 0), int(salt)])
            extra = {"row_classes": rc, "rng": rng}
        aug = context.augment(table, arcs, spec, np, conv=self.FIELDS, labels=labels,
                              extra=extra)
        if len(cache) >= 2:
            cache.pop(next(iter(cache)))
        cache[ck] = (table, aug)
        return aug

    def _labels_changed(self):
        """The store changed. With labels in the loaded model's context the
        cached predictions were made from the previous annotations, so they
        go (the overlay empties until Classify); everything else keeps."""
        ctx = getattr(self, "_clf_context", None)
        if ctx is None or ctx.labels is None or not getattr(self, "_pred", None):
            return
        if getattr(self, "_pred_store_rev", None) == self.store.rev:
            return
        self._pred.clear()
        self._cm_cell = None
        self._refresh_region_modes()
        self._refresh_confusion()

    def _context_spec_from_ui(self):
        """The ContextSpec the Model tab describes (headless: empty)."""
        return context.ContextSpec()

    def _apply_context_spec(self, spec):
        """Push a ContextSpec onto the Model tab controls (headless: no-op)."""

    def _all_stat_slices(self, action, spec=None):
        """``_stream_stat_slices`` drained into a list, for a caller that needs
        random access (the edge-pairs experiment). None means abort."""
        stream = self._stream_stat_slices(action, spec)
        if stream is None:
            return None
        try:
            return list(stream)
        except TrainingProblem as problem:
            self.status_var.set(str(problem))
            return None

    def _iter_stat_slices(self):
        """Cached current records only; callers needing completeness use
        _all_stat_slices()."""
        for key in self.regions.keys():
            idx = self.catalogue.index_of(key)
            rec = self.regions.record(key)
            if idx is None or rec is None or rec.get("labels") is None:
                continue
            table = rec.get("stats")
            if getattr(table, "values", None) is None:
                continue
            si, li = idx
            yield si, li, key, rec, table

    _feature_matrix = staticmethod(TrainingSetBuilder.feature_matrix)

    def _train_classifier(self, preserve_view=False):
        """Fit the selected model kind on the labeled regions' statistics rows.

        Random forest is the default for exactly this data shape: a few
        hundred labels, tens of features of which only a handful matter (trees
        select thresholds per feature, so the noise dimensions that dragged
        k-means across the label boundary are simply never split on), no
        scaling sensitivity, and millisecond retrains -- with
        feature_importances_ naming the dimensions that matter (logged).
        "dense FC" is a small MLP behind an in-pipeline StandardScaler for
        when the boundary is not axis-aligned. The dense-top-N variants first
        fit a balanced forest and retain its N most important dimensions.
        "dense (tuned)" rebuilds the last Optimize winner (or the baseline
        while there is none) with balanced sample weights."""
        got = self._training_set()
        if got is None:
            return
        X, y, _groups, names = got
        ctx = self._context_spec_from_ui()
        kind = self.model_kind_var.get()
        base = _base_kind(kind)
        spec = self._search_spec if base == _TUNED_KIND else None
        build_spec = spec
        if base == _TUNED_KIND and spec is None:     # baseline on the picked backend
            build_spec = model_search.ModelSpec(max_iter=_MLP_MAX_ITER,
                                                backend=self._search_settings()[4])
        if base == _CUSTOM_KIND:
            spec = build_spec = model_search.ModelSpec(
                hidden=tuple(self._custom_hidden()), max_iter=_MLP_MAX_ITER,
                backend=self._search_settings()[4])
        # A frozen base: an edge kind keeps the current base net (same
        # features, has an embedding) and refits only the edges on top.
        want_frozen = _is_edge_kind(kind) and bool(self.freeze_base_var.get())
        frozen = (want_frozen
                  and self._clf is not None and list(self._clf_names or []) == list(names)
                  and self._has_embedding(self._clf))
        # The context columns are part of `names`: a changed context spec
        # means a different design matrix, so the base cannot be kept.
        context_changed = (want_frozen and not frozen and self._clf is not None
                           and getattr(self, "_clf_context", None) != ctx)
        t0 = time.perf_counter()
        if frozen:
            clf = self._clf
            spec = self._clf_spec
        else:
            clf = self._make_model(base, len(y), len(names), names=names, spec=build_spec)
            self._compute_badge("Training")
            try:
                if base in (_TUNED_KIND, _CUSTOM_KIND):
                    model_search.fit_estimator(clf, X, y)
                else:
                    clf.fit(X, y)
            finally:
                self._clear_compute_badge()
        dt_ms = 1e3 * (time.perf_counter() - t0)
        edge, edge_note = (None, "")
        if _is_edge_kind(kind):
            edge, edge_note = self._fit_edge_model_now(clf, names, ctx)
        latent, latent_note = (None, "")
        if ctx.latent is not None:
            latent, latent_note = self._fit_latent_head_now(clf, names, ctx, spec)
        edge_note += latent_note
        self._install_model(clf, names, kind, spec, preserve_view, edge=edge, ctx=ctx,
                            latent=latent)
        if base in (_TUNED_KIND, _CUSTOM_KIND):
            acc = f", train acc {clf.score(X, y):.1%}"
            hint = "" if spec is None else " - " + spec.brief(len(names))
            if frozen:
                hint += " (base frozen)"
            elif context_changed:
                hint += " (base refit: context changed)"
        elif kind in _DENSE_TOP_N:
            selector = clf.named_steps["select"]
            selected = [(names[i], float(v)) for i, v in
                        enumerate(selector.estimator_.feature_importances_)
                        if selector.get_support()[i]]
            selected.sort(key=lambda t: -t[1])
            self._log(f"classifier trained: selected {len(selected)}/{len(names)} "
                "features: " + "  ".join(f"{n}={v:.3f}"
                                          for n, v in selected))
            acc = f", train acc {clf.score(X, y):.1%}"
            hint = " - selected: " + ", ".join(n for n, _v in selected[:3])
        elif hasattr(clf, "feature_importances_"):
            top = sorted(zip(names, clf.feature_importances_),
                         key=lambda t: -t[1])[:8]
            self._log("classifier trained: " + "  ".join(f"{n}={v:.3f}" for n, v in top))
            hint = " - top: " + ", ".join(n for n, _v in top[:3])
            acc = (f", OOB acc {clf.oob_score_:.1%}"
                   if getattr(clf, "oob_score", False) else "")
        else:                                # dense FC: no per-feature story
            acc = f", train acc {clf.score(X, y):.1%}"
            hint = ""
        self._refresh_model_strip()
        if ctx is not None and not ctx.empty():
            hint += " - " + ctx.brief()
        self.status_var.set(f"Trained {kind} on {len(y)} labeled regions in "
                            f"{dt_ms:.0f} ms{acc}{hint}{edge_note}")

    def _training_set(self, spec=None):
        """``(X, y, groups, names)`` over every labeled region of every primed
        slice -- `groups` is each row's slice index, for leave-slices-out CV --
        or None with the reason in the status bar. `names` carries the context
        columns of `spec` (the Model tab's when None) after the record's own."""
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.status_var.set("scikit-learn is not installed - "
                                "pip install scikit-learn to enable training")
            return None
        import numpy as np
        if spec is None:
            spec = self._context_spec_from_ui()
        slices = self._stream_stat_slices("Preparing training", spec, training=True)
        if slices is None:
            return None
        items = ((key, rec, table, self.catalogue.group_of(key), f"{si}:{li}")
                 for si, li, key, rec, table in slices)
        try:
            return self._training_builder.labeled_set(
                items, self.store, np,
                layer_of=lambda key, rec: self.regions.label_layer(key),
                gestures_of=self._gestures_for_key)
        except TrainingProblem as problem:
            self.status_var.set(str(problem))
            return None

    def _install_model(self, clf, names, kind, spec=None, preserve_view=False, edge=None,
                       ctx=None, latent=None):
        """Make `clf` the current model (Train and Optimize share this tail).
        `edge` is the edge model fit on top of it (None drops any old one: an
        edge model belongs to exactly one base net). `ctx` is the ContextSpec
        whose columns `names` carries (None = none): predictions and the
        compatibility gate rebuild exactly those columns. `latent` is the
        latent-ring head fit on top of `clf` (None drops any old one, for
        the same reason as the edge model)."""
        self._clf = clf
        self._clf_names = names
        self._clf_kind = kind
        self._clf_spec = spec
        # A model belongs to the regime it was fitted in, and is stamped here
        # rather than at save time: it must be refused the moment the profile
        # moves, not only after a round trip through a pickle.
        self._clf_scope = self._feature_scope()
        self._clf_context = ctx if ctx is not None else context.ContextSpec()
        self._context_model = latent
        self._edge_model = edge
        self._pred.clear()               # predictions belong to the old model
        if not preserve_view:
            self._cm_cell = None
            self._refresh_region_modes()
        self._refresh_confusion()
        self.classify_btn.config(state="normal")
        self._refresh_model_readout()
        self._refresh_edge_readout()

    @staticmethod
    def _has_embedding(clf):
        try:
            edge_model.hidden_widths(clf)
            return True
        except TypeError:
            return False

    def _edge_training_data(self, names, spec=None):
        """Every region of every primed slice (class 0 = unlabeled) with its
        slice index and extremum value, plus the edges of the region graph as
        global row pairs -- what the edge model is fit and evaluated on. Tk
        thread only (records, the store); returns copies the worker may keep.
        `spec` is the ContextSpec `names` was built under (the loaded model's
        when None). None (with the reason in the status bar) when a slice
        lacks stats."""
        import numpy as np
        if spec is None:
            spec = getattr(self, "_clf_context", None)
        slices = self._stream_stat_slices("Gathering edges", spec, training=True)
        if slices is None:
            return None
        items = ((key, rec, table, self.catalogue.group_of(key), f"{si}:{li}")
                 for si, li, key, rec, table in slices)
        # The pair model's `contact` feature (and a contact-weighted latent
        # ring) reads the arcs' shared boundary lengths: derived from the
        # label raster once per record and cached on the arcs dict, only when
        # the settings ask for it.
        want_contact = ("contact" in self._edge_spec_from_ui().features
                        or (spec is not None and spec.latent is not None
                            and spec.latent.weight == "contact"))

        def arcs_of(key, rec):
            arcs = self.regions.arcs(key, np)
            if want_contact and arcs is not None and rec.get("labels") is not None:
                context.ensure_contact(arcs, rec["labels"], np)
            return arcs

        try:
            return self._training_builder.edge_set(
                items, self.store, names, arcs_of, np,
                layer_of=lambda key, rec: self.regions.label_layer(key),
                gestures_of=self._gestures_for_key)
        except TrainingProblem as problem:
            self.status_var.set(str(problem))
            return None

    def _fit_edge_model_now(self, clf, names, spec=None):
        """Fit the pair model on top of `clf`: (EdgeModel | None, status note).
        `spec` is the ContextSpec `clf` was fit under."""
        if not self._has_embedding(clf):
            return None, " - edges skipped: the base has no hidden layer to embed with"
        data = self._edge_training_data(names, spec)
        if data is None:
            return None, " - edges skipped: no data"
        X_all, cls, _grp, ext, edges, _names = data
        spec = self._edge_spec_from_ui()
        self._compute_badge("Fitting edges")
        try:
            edge = edge_model.fit_edge_model(clf, X_all, cls, edges, ext, spec, names)
        except ValueError as exc:
            self._log(f"edge model not fit: {exc}")
            return None, f" - edges not fit: {exc}"
        finally:
            self._clear_compute_badge()
        self._log(f"edge model: {edge.describe()}; {int(edges['both'].sum())} labeled pairs of "
            f"{len(edges['a'])} edges, {edge.fit_s * 1e3:.0f} ms")
        return edge, (f" -> edges: {edge.n_edges:,} pairs, "
                      f"{edge.n_diff / max(1, edge.n_edges):.0%} boundaries, "
                      f"{'MSC saddles' if edge.used_saddle else 'pixel adjacency'}, "
                      f"{edge.fit_s * 1e3:.0f} ms")

    def _fit_latent_head_now(self, clf, names, ctx, spec=None):
        """Fit the latent-ring head on top of `clf` (the base net) under
        ContextSpec `ctx` (whose `latent` block says how):
        (LatentContextModel | None, status note). The head is the base's own
        architecture (the tuned spec when there is one, its feature mask
        widened by the latent columns) refit on the row plus the ring's
        embedding columns, with balanced weights like the base."""
        import numpy as np
        if ctx is None or ctx.latent is None:
            return None, ""
        if not self._has_embedding(clf):
            return None, " - latent head skipped: the base has no hidden layer to embed with"
        data = self._edge_training_data(names, ctx)
        if data is None:
            return None, " - latent head skipped: no data"
        X_all, cls, grp, _ext, edges, _names = data
        lspec = ctx.latent
        length = edges.get("length") if lspec.weight == "contact" else None
        g = context.directed_graph(edges["a"], edges["b"], edges["n_rows"], np, length)
        area = None
        af = self.FIELDS.area_field
        if lspec.weight == "area" and af in names:
            area = X_all[:, list(names).index(af)]
        # `spec` is the base being installed (None for the plain kinds: the
        # head is then the default dense net) -- never the PREVIOUS model's.
        base_spec = spec
        backend = self._search_settings()[4]

        def make_head(all_names):
            if base_spec is not None:
                feats = (None if base_spec.features is None
                         else list(base_spec.features) + list(all_names[len(names):]))
                hs = dataclasses.replace(base_spec, features=feats, max_iter=_MLP_MAX_ITER)
            else:
                hs = model_search.ModelSpec(max_iter=_MLP_MAX_ITER, backend=backend)
            return model_search.build_estimator(hs, all_names), model_search.fit_estimator

        self._compute_badge("Fitting latent head")
        try:
            model = context.fit_latent_head(clf, X_all, cls, g, lspec, names, make_head, np,
                                            area=area, row_slice=grp)
        except (ValueError, TypeError) as exc:
            self._log(f"latent head not fit: {exc}")
            return None, f" - latent head not fit: {exc}"
        finally:
            self._clear_compute_badge()
        self._log(f"latent head: {model.describe()}")
        return model, (f" -> {lspec.brief()}: +{len(context.latent_names(lspec, model.width))} "
                       f"columns, {model.fit_s * 1e3:.0f} ms")

    def _latent_for(self, clf, names, ctx, spec=None):
        """The latent head for an installed-elsewhere base (Optimize / sweep
        winners), or None -- the note goes to the log."""
        model, note = self._fit_latent_head_now(clf, names, ctx, spec)
        if note:
            self._log("latent head:" + note)
        return model

    def _latent_graph(self, key, table, labels, lspec, np):
        """``(graph, area)`` over the slice's rows for the latent head, from the
        provider's arcs (contact lengths derived when the weighting needs them)."""
        arcs = self.regions.arcs(key, np)
        ids = table.column(self.FIELDS.id_field)
        n = table.n_rows
        if arcs is None or ids is None or not len(arcs.get("a", ())):
            return context.directed_graph([], [], n, np), None
        ia, ib, keep = magic_fill.index_arcs(arcs, ids, np)
        length = None
        if lspec.weight == "contact":
            length = np.asarray(context.ensure_contact(arcs, labels, np), np.float64)[keep]
        area = table.column(self.FIELDS.area_field) if lspec.weight == "area" else None
        return context.directed_graph(ia, ib, n, np, length), area

    def _vote_entry(self, region_proba, aux, lam, rounds, np):
        """(final classes, flips) for one cached slice: neighbour voting in
        LABEL space over the classes that carry any probability mass; ids
        that are not living regions stay class 0."""
        raw = np.asarray(aux["raw"], np.uint8)
        cols = np.nonzero(region_proba.sum(0) > 0)[0]
        k = aux["keep"]
        if len(cols) < 2 or not k.any():
            return raw.copy(), 0
        P = region_proba[:, cols]
        voted = edge_model.vote(P, cols, np.asarray(aux["la"])[k], np.asarray(aux["lb"])[k],
                                np.asarray(aux["pdiff"])[k], lam, rounds)
        living = region_proba.sum(1) > 0
        final = raw.copy()
        final[living] = np.asarray(voted, np.uint8)[living]
        return final, int((final != raw).sum())

    def _revote_all(self):
        """Re-apply (or lift) neighbour voting on every cached slice from the
        cached per-arc p(diff): a kind flip or a lambda / rounds edit costs
        milliseconds, no forward pass. Entries classified before the edge
        model existed have no aux and force a Classify."""
        import numpy as np
        active = self._edge_model is not None and _is_edge_kind(self.model_kind_var.get())
        spec = self._edge_spec_from_ui()
        missing = False
        for key, entry in list(self._pred.items()):
            aux = self._pred_aux(entry)
            if aux is None:
                missing = missing or self._edge_model is not None
                continue
            if active:
                final, flips = self._vote_entry(entry[2], aux, spec.lam, spec.rounds, np)
            else:
                final, flips = np.asarray(aux["raw"], np.uint8).copy(), 0
            aux["lam"], aux["rounds"], aux["flips"], aux["active"] = spec.lam, spec.rounds, flips, active
            self._pred[key] = (entry[0], final, entry[2], aux)
        if missing and self._clf is not None:
            self._pred.clear()
            self._classify()
            return
        self._refresh_region_modes()
        self._refresh_confusion()
        self._refresh_render()
        self._refresh_edge_readout()

    @staticmethod
    def _make_model(kind, n_samples, n_features=None, names=None, spec=None):
        if kind == _CUSTOM_KIND:
            if not names or spec is None:
                raise ValueError("custom FC needs the feature names and a spec")
            return model_search.build_estimator(spec, list(names))
        if kind == _TUNED_KIND:
            # The search winner (or the baseline before any search), built by
            # model_search so the estimator and its description share a spec.
            if not names:
                raise ValueError("dense (tuned) needs the feature names")
            spec = spec or model_search.ModelSpec(max_iter=_MLP_MAX_ITER)
            return model_search.build_estimator(spec, list(names))
        if kind == "dense FC":
            # The scaler lives INSIDE the pipeline: an MLP needs standardized
            # inputs, and keeping it in the estimator means the pickle /
            # predict paths stay identical to the forest's.
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.neural_network import MLPClassifier
            return make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=_MLP_HIDDEN, max_iter=_MLP_MAX_ITER,
                              random_state=0))
        if kind in _DENSE_TOP_N:
            if n_features is None or n_features < 1:
                raise ValueError("dense-top-N requires at least one feature")
            # Keep selection inside the estimator: the model continues to
            # consume the full profile schema, while its fitted forest mask is
            # applied identically after pickle/load and during prediction.
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_selection import SelectFromModel
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.neural_network import MLPClassifier
            top_n = min(_DENSE_TOP_N[kind], n_features)
            forest = RandomForestClassifier(
                n_estimators=_FOREST_TREES, class_weight="balanced", n_jobs=-1,
                random_state=0)
            return Pipeline([
                ("select", SelectFromModel(forest, threshold=-np.inf,
                                           max_features=top_n)),
                ("scale", StandardScaler()),
                ("dense", MLPClassifier(hidden_layer_sizes=_MLP_HIDDEN,
                                        max_iter=_MLP_MAX_ITER, random_state=0)),
            ])
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=_FOREST_TREES, class_weight="balanced",
                                      oob_score=n_samples >= _OOB_MIN_SAMPLES, n_jobs=-1,
                                      random_state=0)

    def _train_and_classify(self, _e=None):
        """'R': one keystroke = retrain + reclassify."""
        if self._typing():
            return
        before = self._clf
        # Retrain + classify is one visual operation: keep the selected region
        # coloring and confusion cell through the transient empty prediction
        # cache. _classify refreshes them against the new predictions.
        self._train_classifier(preserve_view=True)
        if self._clf is not None and self._clf is not before:
            self._classify()
            if not self._pred:            # classification failed after training
                self._cm_cell = None
                self._refresh_region_modes()

    def _on_optimize_key(self, _e=None):
        """'O': search the dense network, install + classify the winner."""
        if self._typing():
            return
        self._optimize_network()

    # -- Optimize network ------------------------------------------------ #
    def _search_settings(self):
        """``(n_trials, timeout_s, seed, feature_search, backend)`` from the
        Model tab, each clamped to its range / choices (a malformed entry
        reads as the default)."""
        def _int(var, default, lo, hi):
            try:
                v = int(float(str(var.get()).strip()))
            except (ValueError, TypeError):
                v = default
            return max(lo, min(hi, v))
        backend = self.search_backend_var.get()
        if backend not in model_search.BACKENDS:
            backend = "auto"
        # The time limit is entered in minutes and kept in seconds.
        try:
            minutes = float(str(self.search_timeout_var.get()).strip())
            timeout_s = int(round(minutes * 60.0))
        except (ValueError, TypeError):
            timeout_s = _SEARCH_TIMEOUT_S
        lo, hi = _SEARCH_TIMEOUT_RANGE
        timeout_s = max(lo, min(hi, timeout_s))
        return (_int(self.search_trials_var, _SEARCH_TRIALS, *_SEARCH_TRIALS_RANGE),
                timeout_s,
                _int(self.search_seed_var, 0, 0, 2**31 - 1),
                bool(self.search_features_var.get()), backend)

    def _sweep_settings_trials(self):
        try:
            return self._sweep_settings()[1]
        except ValueError:
            return _SWEEP_TRIALS

    def _apply_search_view(self, d):
        """Restore the search settings from a session view dict, field by
        field; a missing or malformed value leaves that setting alone."""
        if not isinstance(d, dict):
            return
        for key, var in (("trials", self.search_trials_var),
                         ("seed", self.search_seed_var)):
            v = d.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                var.set(str(int(v)))
        v = d.get("timeout_s")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            self.search_timeout_var.set(f"{float(v) / 60.0:g}")     # entry in minutes
        if isinstance(d.get("feature_search"), bool):
            self.search_features_var.set(d["feature_search"])
        if d.get("backend") in model_search.BACKENDS:
            self.search_backend_var.set(d["backend"])
        sizes = d.get("sweep_sizes")
        if isinstance(sizes, str) and sizes.strip():
            try:
                model_search.parse_sizes(sizes)
                self.sweep_sizes_var.set(sizes)
            except ValueError:
                pass
        v = d.get("sweep_trials")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            self.sweep_trials_var.set(str(int(v)))

    def _optimize_network(self, sync=False):
        """Search the dense FC space by leave-slices-out cross-validation and
        install the winner as the "dense (tuned)" model, then classify.

        The labeled rows are gathered on the Tk thread (they touch the engine
        and the label store); the search itself runs on a worker thread and
        reports through a queue that `_search_pump` drains, so labeling stays
        live and Cancel keeps the best so far. `sync` runs the worker inline
        (the selftest)."""
        if self._search is not None:
            self.status_var.set("A search is already running - Cancel it first.")
            return
        # The training set reads the Model tab's context itself; the spec is
        # captured beside it so the winner is installed under the same one.
        ctx = self._context_spec_from_ui()
        got = self._training_set()
        if got is None:
            return
        X, y, groups, names = got
        n_trials, timeout_s, seed, feat, backend = self._search_settings()
        try:
            model_search.make_cv(y, groups, seed=seed)
        except ValueError as exc:
            self.status_var.set(f"Cannot optimize: {exc}.")
            return
        schema = self._feature_schema_with_context(ctx, names)
        searcher = "optuna" if model_search.have_optuna() else "random"
        q = queue.Queue()
        stop = threading.Event()
        space = model_search.SearchSpace(feature_search=feat)

        def progress(done, total, best):
            q.put(("progress", done, total, best))

        def work():
            try:
                res = model_search.run_search(
                    X, y, groups, names, schema, n_trials=n_trials,
                    timeout_s=timeout_s or None, seed=seed,
                    max_iter=model_search.SEARCH_MAX_ITER,
                    patience=model_search.SEARCH_PATIENCE,
                    refit_max_iter=_MLP_MAX_ITER,
                    space=space, progress_cb=progress, stop_event=stop,
                    searcher=searcher, backend=backend)
                q.put(("done", res))
            except Exception as exc:            # reported, never raised off-thread
                q.put(("error", f"{type(exc).__name__}: {exc}"))

        n_slices = len(set(groups.tolist()))
        self._search = {"queue": q, "stop": stop, "thread": None, "names": names,
                        "n": int(len(y)), "t0": time.perf_counter(), "context": ctx}
        self.model_kind_var.set(_TUNED_KIND)
        self._set_search_buttons(running=True)
        self.search_progress_var.set(
            f"{searcher}: scoring the baseline ({len(y)} labeled regions on "
            f"{n_slices} slice{'s' if n_slices != 1 else ''}, {len(names)} features)…")
        self._compute_badge("Optimizing")
        self._log(f"optimize: {searcher}, {model_search.backend_label(backend)}, {n_trials} "
            f"trials, {timeout_s or 'no'} s limit, seed {seed}, feature subset "
            f"{'on' if feat else 'off'}; {len(y)} rows, {n_slices} slices, "
            f"{len(names)} features; trial budget {model_search.SEARCH_MAX_ITER} epochs "
            f"/ patience {model_search.SEARCH_PATIENCE}, batches < "
            f"{model_search.TORCH_MIN_BATCH} on sklearn"
            + ("" if searcher == "optuna" else
               " (pip install optuna for the TPE searcher)"))
        if sync:
            work()
            self._search_pump()
            return
        t = threading.Thread(target=work, name=f"{self.LOG_PREFIX}-optimize", daemon=True)
        self._search["thread"] = t
        t.start()
        self.root.after(_SEARCH_PUMP_MS, self._search_pump)

    def _search_pump(self):
        st = self._search
        if st is None:
            return
        finished = False
        try:
            while not finished:
                ev = st["queue"].get_nowait()
                if ev[0] == "progress":
                    _, done, total, best = ev
                    text = f"trial {done}/{total}"
                    elapsed = time.perf_counter() - st["t0"]
                    if done > 0 and elapsed > 1.0:
                        text += (f" · {_hms(elapsed)} elapsed, ~{_hms(elapsed / done)}"
                                 f"/trial, ~{_hms(elapsed / done * max(0, total - done))} left")
                    if best is not None and best.cv_score is not None:
                        text += (f" · best: log-loss {best.cv_score:.3f}, balanced acc "
                                 f"{best.cv_bacc:.1%} · {best.layers_text()} · "
                                 f"{best.features_text(len(st['names']))}")
                    self.search_progress_var.set(text)
                    self._compute_badge(f"Optimizing {done}/{total}")
                elif ev[0] == "done":
                    self._finish_search(ev[1])
                    finished = True
                elif ev[0] == "sweep_progress":
                    self._on_sweep_progress(*ev[1:])
                elif ev[0] == "sweep_done":
                    self._finish_sweep(ev[1])
                    finished = True
                elif ev[0] == "edge_progress":
                    _, f, n = ev
                    self.edge_progress_var.set(
                        f"fold {f}/{n} · {_hms(time.perf_counter() - st['t0'])} elapsed")
                    self._compute_badge(f"Evaluating edges {f}/{n}")
                elif ev[0] == "edge_done":
                    self._finish_edge_eval(ev[1])
                    finished = True
                elif ev[0] == "seam_progress":
                    _, f, n = ev
                    self._compute_badge(f"Evaluating seams {f}/{n}")
                elif ev[0] == "seam_done":
                    self._finish_seam_eval(ev[1])
                    finished = True
                else:
                    self.search_progress_var.set(f"failed: {ev[1]}")
                    self.status_var.set(f"Optimize failed: {ev[1]}")
                    self._log(f"optimize failed: {ev[1]}")
                    finished = True
        except queue.Empty:
            pass
        thread = st.get("thread")
        if not finished and thread is not None and not thread.is_alive() \
                and st["queue"].empty():
            self.search_progress_var.set("failed: the search thread ended silently")
            self.status_var.set("Optimize failed: the search thread ended silently")
            finished = True
        if finished:
            self._search = None
            self._set_search_buttons(running=False)
            self._clear_compute_badge()
            self._refresh_sweep_buttons()
            return
        self.root.after(_SEARCH_PUMP_MS, self._search_pump)

    def _finish_search(self, res):
        st = self._search
        names = st["names"]
        spec = res.spec
        self._search_spec = spec
        ctx = st.get("context")
        latent = self._latent_for(res.estimator, names, ctx, spec)
        self._install_model(res.estimator, names, _TUNED_KIND, spec, preserve_view=True,
                            ctx=ctx, latent=latent)
        self._refresh_model_readout()
        gain = ""
        if spec.baseline_score is not None:
            gain = f" (baseline {spec.baseline_score:.3f} / {spec.baseline_bacc:.1%})"
        summary = (f"{spec.searcher}: {res.n_trials} trials in {res.elapsed_s:.0f} s"
                   f"{' (stopped early)' if res.stopped else ''} · best: log-loss "
                   f"{spec.cv_score:.3f}, balanced acc {spec.cv_bacc:.1%}{gain} · "
                   f"{spec.layers_text()} · {spec.features_text(len(names))}")
        self.search_progress_var.set(summary)
        self._log("optimize: " + summary + " -- " + spec.describe(len(names)))
        if res.importances:
            self._log("optimize: top features (permutation, log-loss): "
                + "  ".join(f"{n}={v:.3f}" for n, v in res.importances[:8]))
        self._refresh_model_strip()
        saved = self._autosave_tuned_model()
        if saved:
            self.search_progress_var.set(summary + f" · saved {saved}")
        # Classify first: it writes its own status line, and the summary
        # (with where the winner was saved) is the one that should stay.
        self._classify()
        self.status_var.set(f"Optimized network on {st['n']} labeled regions - {summary}"
                            + (f" - saved {saved}" if saved else ""))

    def _autosave_tuned_model(self):
        """Pickle the search winner under the session's models folder and
        record it, so the result of a run nobody watched survives: the
        session reloads its most recent recorded model on restore. Returns
        the path, or None (with a log line) when the write failed."""
        folder = self._models_dir
        if not folder or self._clf is None:
            return None
        try:
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, time.strftime("tuned_%Y%m%d_%H%M%S.pkl"))
            self._save_classifier_to(path)
        except Exception as exc:
            self._log(f"optimize: could not save the winner: {exc}")
            return None
        self._log(f"optimize: winner saved to {path}")
        return path

    # -- Size sweep ------------------------------------------------------ #
    def _sweep_network(self, sync=False):
        """Score a ladder of architectures, each at its own best settings
        (one fixed-size search per rung), report them, install the best rung
        and classify. Same worker/pump/Cancel as Optimize."""
        if self._search is not None:
            self.status_var.set("A search is already running - Cancel it first.")
            return
        try:
            sizes, per_size = self._sweep_settings()
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        # The training set reads the Model tab's context itself; the spec is
        # captured beside it so the winner is installed under the same one.
        ctx = self._context_spec_from_ui()
        got = self._training_set()
        if got is None:
            return
        X, y, groups, names = got
        _n_trials, timeout_s, seed, feat, backend = self._search_settings()
        try:
            model_search.make_cv(y, groups, seed=seed)
        except ValueError as exc:
            self.status_var.set(f"Cannot sweep: {exc}.")
            return
        schema = self._feature_schema_with_context(ctx, names)
        searcher = "optuna" if model_search.have_optuna() else "random"
        q = queue.Queue()
        stop = threading.Event()
        space = model_search.SearchSpace(feature_search=feat)

        def progress(k, n, hidden, done, total, row):
            q.put(("sweep_progress", k, n, hidden, done, total, row))

        def work():
            try:
                res = model_search.run_size_sweep(
                    X, y, groups, names, schema, sizes=sizes, trials_per_size=per_size,
                    timeout_s=timeout_s or None, seed=seed,
                    max_iter=model_search.SEARCH_MAX_ITER,
                    patience=model_search.SEARCH_PATIENCE, refit_max_iter=_MLP_MAX_ITER,
                    space=space, progress_cb=progress, stop_event=stop,
                    searcher=searcher, backend=backend)
                q.put(("sweep_done", res))
            except Exception as exc:            # reported, never raised off-thread
                q.put(("error", f"{type(exc).__name__}: {exc}"))

        self._sweep = None
        self._sweep_rows = []
        self._refresh_sweep_report([])
        self._search = {"queue": q, "stop": stop, "thread": None, "names": names,
                        "n": int(len(y)), "t0": time.perf_counter(), "kind": "sweep",
                        "sizes": sizes, "per_size": per_size, "context": ctx}
        self.model_kind_var.set(_TUNED_KIND)
        self._set_search_buttons(running=True)
        self.search_progress_var.set(
            f"size sweep: {len(sizes)} rung(s) x {per_size} trials on {len(y)} labeled "
            f"regions, {len(names)} features…")
        self._compute_badge("Sweeping sizes")
        self._log(f"size sweep: {[model_search.size_text(h) for h in sizes]}, {per_size} "
            f"trials/rung, {searcher}, {model_search.backend_label(backend)}, "
            f"{timeout_s or 'no'} s limit, seed {seed}; {len(y)} rows, {len(names)} features")
        if sync:
            work()
            self._search_pump()
            return
        t = threading.Thread(target=work, name=f"{self.LOG_PREFIX}-size-sweep", daemon=True)
        self._search["thread"] = t
        t.start()
        self.root.after(_SEARCH_PUMP_MS, self._search_pump)

    def _on_sweep_progress(self, k, n, hidden, done, total, row):
        st = self._search
        size = model_search.size_text(hidden)
        if row is not None:
            self._sweep_rows.append(row)
            self._sweep = model_search.SweepResult(
                rows=list(self._sweep_rows), names=list(st["names"]), n_classes=0,
                elapsed_s=time.perf_counter() - st["t0"])
            self._refresh_sweep_report(self._sweep_rows)
            self._log(f"size sweep: {size}: log-loss {row.cv_score:.3f}, bal. acc "
                f"{row.cv_bacc:.1%}, {row.n_params:,} params, {row.n_trials} trials, "
                f"{row.elapsed_s:.0f} s -- {row.spec.settings_text(len(st['names']))}")
        elapsed = time.perf_counter() - st["t0"]
        text = f"size {k + 1}/{n} ({size}): trial {done}/{total}"
        rungs_done = len(self._sweep_rows)
        if rungs_done and elapsed > 1.0:
            per_rung = elapsed / rungs_done
            text += (f" · {_hms(elapsed)} elapsed, ~{_hms(per_rung)}/rung, "
                     f"~{_hms(per_rung * max(0, n - rungs_done))} left")
        if self._sweep_rows:
            b = self._sweep_rows[min(range(len(self._sweep_rows)),
                                     key=lambda i: self._sweep_rows[i].cv_score)]
            text += f" · best so far {b.size} (log-loss {b.cv_score:.3f})"
        self.search_progress_var.set(text)
        self._compute_badge(f"Sweeping {k + 1}/{n} · {done}/{total}")

    def _finish_sweep(self, res):
        st = self._search
        self._sweep = res
        self._sweep_context = st.get("context")     # for a rung installed later
        self._sweep_rows = list(res.rows)
        self._refresh_sweep_report(res.rows, stopped=res.stopped)
        lines = res.report_lines()
        self._log("size sweep report:\n  " + "\n  ".join(lines))
        # Install the best rung (the Optimize convention); a smaller rung is
        # one click away in the table.
        best = res.rows[res.best_index()]
        self._install_sweep_row(best, announce=False)
        saved = self._autosave_tuned_model()
        report_path = self._save_sweep_report(res, saved)
        summary = (f"{len(res.rows)} rung(s) in {_hms(res.elapsed_s)}"
                   f"{' (stopped early)' if res.stopped else ''} · {res.summary()}")
        self.search_progress_var.set(summary + (f" · saved {saved}" if saved else ""))
        self._classify()
        self.status_var.set(f"Size sweep on {st['n']} labeled regions - {summary}"
                            + (f" - report {report_path}" if report_path else ""))

    def _install_sweep_row(self, row, announce=True):
        names = list(self._search["names"] if self._search is not None
                     else (self._sweep.names if self._sweep is not None else []))
        ctx = (self._search.get("context") if self._search is not None
               else getattr(self, "_sweep_context", None))
        self._search_spec = row.spec
        latent = self._latent_for(row.estimator, names, ctx, row.spec)
        self._install_model(row.estimator, names, _TUNED_KIND, row.spec, preserve_view=True,
                            ctx=ctx, latent=latent)
        self._refresh_model_readout()
        self._refresh_model_strip()
        if announce:
            self.status_var.set(f"Installed {row.size} ({row.n_params:,} params, CV log-loss "
                                f"{row.cv_score:.3f}, bal. acc {row.cv_bacc:.1%})")

    def _sweep_install_selected(self):
        """Make the selected rung the model, save it and classify."""
        if self._sweep is None or self._search is not None:
            return
        sel = self.sweep_tree.selection()
        if not sel:
            return
        try:
            row = self._sweep.rows[int(sel[0])]
        except (ValueError, IndexError):
            return
        self._install_sweep_row(row)
        saved = self._autosave_tuned_model()
        status = self.status_var.get()
        self._classify()
        self.status_var.set(status + (f" - saved {saved}" if saved else ""))

    def _save_sweep_report(self, res, model_path=None):
        """Write the sweep (rows, specs, the text table) as JSON beside the
        saved models, so the numbers outlive the session. Returns the path
        or None."""
        folder = self._models_dir
        if not folder:
            return None
        try:
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, time.strftime("sweep_%Y%m%d_%H%M%S.json"))
            doc = res.to_dict()
            doc["installed_model"] = model_path
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
        except Exception as exc:
            self._log(f"size sweep: could not save the report: {exc}")
            return None
        self._log(f"size sweep: report saved to {path}")
        return path

    # -- Evaluate edges -------------------------------------------------- #
    def _evaluate_edges(self, sync=False):
        """Leave-slices-out report for the edge model on top of the CURRENT
        base kind: per fold the base is refit on the training slices, the
        pair model on their labeled edges, both scored on the held-out slices,
        then voting before/after. Same worker / pump / Cancel as Optimize."""
        if self._search is not None:
            self.status_var.set("A search is already running - Cancel it first.")
            return
        if self._clf is None or not self._has_embedding(self._clf):
            self.status_var.set("Evaluate edges needs a trained dense base - Train a dense "
                                "or '-> edges' kind first.")
            return
        names = list(self._clf_names)
        data = self._edge_training_data(names)
        if data is None:
            return
        X_all, cls, grp, ext, edges, _n = data
        _trials, _timeout, seed, _feat, _backend = self._search_settings()
        try:
            model_search.make_cv(cls[cls > 0], grp[cls > 0], seed=seed)
        except ValueError as exc:
            self.status_var.set(f"Cannot evaluate edges: {exc}.")
            return
        if not (edges["both"] & edges["diff"]).any():
            self.status_var.set("Cannot evaluate edges: no labeled edge crosses classes - "
                                "label two touching regions of different classes.")
            return
        from sklearn.base import clone
        template = clone(self._clf)
        spec = self._edge_spec_from_ui()
        q = queue.Queue()
        stop = threading.Event()

        def work():
            try:
                rep = edge_model.evaluate_edges(
                    lambda: clone(template), X_all, cls, grp, edges, ext, spec, seed=seed,
                    progress_cb=lambda f, n: q.put(("edge_progress", f, n)), stop_event=stop)
                q.put(("edge_done", rep))
            except Exception as exc:            # reported, never raised off-thread
                q.put(("error", f"{type(exc).__name__}: {exc}"))

        self._search = {"queue": q, "stop": stop, "thread": None, "names": names,
                        "n": int((cls > 0).sum()), "t0": time.perf_counter(), "mode": "edges"}
        self._set_search_buttons(running=True)
        self.edge_progress_var.set(f"evaluating: {int(edges['both'].sum()):,} labeled pairs "
                                   f"({int(edges['diff'].sum()):,} boundaries) on "
                                   f"{len(set(grp[cls > 0].tolist()))} slice(s)…")
        self._compute_badge("Evaluating edges")
        self._log(f"evaluate edges: {spec.describe()}; {len(edges['a'])} edges, "
            f"{int(edges['both'].sum())} labeled pairs")
        if sync:
            work()
            self._search_pump()
            return
        t = threading.Thread(target=work, name=f"{self.LOG_PREFIX}-edge-eval", daemon=True)
        self._search["thread"] = t
        t.start()
        self.root.after(_SEARCH_PUMP_MS, self._search_pump)

    def _finish_edge_eval(self, report):
        if self._edge_model is not None:
            self._edge_model.report = report
        self._fill_edge_report(report)
        lines = edge_model.report_lines(report)
        self._log("evaluate edges:\n  " + "\n  ".join(lines))
        learned = report.get("edge", {}).get("learned")
        base = report.get("edge", {}).get("argmax differs", {})
        reg = report.get("region", {})
        if learned:
            summary = (f"learned {learned['diff_recall']:.0%}/{learned['diff_precision']:.0%} "
                       f"boundary recall/precision vs argmax {base.get('diff_recall', 0):.0%}/"
                       f"{base.get('diff_precision', 0):.0%}")
            if reg.get("before") and reg.get("after"):
                summary += (f" · held-out region errors {reg['before']['errors']} -> "
                            f"{reg['after']['errors']} with voting")
        else:
            summary = "no fold could be scored: " + "; ".join(report.get("skipped", [])[:2])
        summary += (f" · {report.get('n_folds', 0)}-fold {report.get('cv_kind', '')} CV in "
                    f"{_hms(report.get('elapsed_s', 0))}")
        if report.get("stopped"):
            summary += " (stopped early)"
        self.edge_progress_var.set(summary)
        self._refresh_model_readout()
        self._refresh_model_strip()
        self._refresh_edge_readout()
        self.status_var.set("Evaluated edges - " + summary)

    def _cancel_search(self):
        st = self._search
        if st is None:
            return
        st["stop"].set()
        self.search_progress_var.set("stopping after the current trial…")
        self.cancel_search_btn.config(state="disabled")

    def _set_search_buttons(self, running):
        for name, state in (("optimize_btn", "disabled" if running else "normal"),
                            ("sweep_btn", "disabled" if running else "normal"),
                            ("edge_eval_btn", "disabled" if running else "normal"),
                            ("seam_eval_btn", "disabled" if running else "normal"),
                            ("seam_train_btn", "disabled" if running else "normal"),
                            ("cancel_search_btn", "normal" if running else "disabled")):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.config(state=state)

    def _on_classify_key(self, _e=None):
        """'C': classify/reclassify with the current model."""
        if self._typing() or self._clf is None:
            return
        self._classify()

    # -- computing badge on the image plane (top-left canvas HUD) -------- #
    def _compute_badge(self, text):
        if self.viewer is not None:
            self.viewer.set_hud("busy", text)
            try:
                self.root.update_idletasks()   # paint before the blocking fit
            except tk.TclError:
                pass

    def _clear_compute_badge(self):
        self._update_busy()      # restores busy/stale/none per engine state

    def _expected_names_for(self, spec):
        """The feature names the ACTIVE profile produces under ContextSpec
        `spec` (the loaded model's when None): the app's own schema plus the
        context columns built over it. None when the app cannot say (no
        compiled extension), which skips the gate."""
        expected = self._expected_feature_names()
        if expected is None:
            return None
        if spec is None:
            spec = getattr(self, "_clf_context", None)
        expected = list(expected)
        if spec is not None and not spec.empty():
            expected += context.column_names(spec, expected, self.FIELDS)
        return expected

    def _feature_schema_with_context(self, spec, names):
        """The active profile's ``{name, channel, reduction}`` schema plus one
        entry per context column, so Optimize's feature mask files each
        context kind as its own group. None when neither exists."""
        schema = self._feature_schema_now()
        if spec is None or spec.empty():
            return schema
        return list(schema or []) + context.schema_entries(spec, list(names), self.FIELDS)

    def _check_model_compat(self, names, context, scope=None, ctx=None):
        """None when `names` matches the active profile's statistics schema AND
        the model's scope is the profile's; else a blocking message naming the
        exact mismatch. Names are compared as SETS: the feature matrix is
        assembled by name, so order never matters. `scope` defaults to the
        loaded model's (see `_feature_scope`); `ctx` is the ContextSpec whose
        columns `names` includes, defaulting to the loaded model's too."""
        expected = self._expected_names_for(ctx)
        if expected is None:
            self._log(f"model compatibility check skipped ({context}): "
                "compiled extension not available")
            return None
        prof = "?"
        if 0 <= self.active_profile_idx < len(self.profiles):
            prof = self.profiles[self.active_profile_idx]["name"]
        if scope is None:
            scope = getattr(self, "_clf_scope", None)
        return model_bundle.compat_message(names, expected, prof, context,
                                           scope, self._feature_scope())

    def _classify(self):
        """Predict a class for EVERY region of every computed slice and show
        the result as a translucent layer under the drawn labels. BLOCKS with
        a message when the model's features don't match the active profile
        (replacing the old silent per-slice skip)."""
        if self._clf is None:
            self.status_var.set("Train first.")
            return
        msg = self._check_model_compat(self._clf_names, "classify")
        if msg:
            messagebox.showerror(self.APP_TITLE, msg)
            self.status_var.set(msg)
            return
        import numpy as np
        slices = self._stream_stat_slices("Preparing classification")
        if slices is None:
            return
        count = 0
        t0 = time.perf_counter()
        try:
            for si, li, key, rec, table in slices:
                if self._predict_slice(si, li, rec, np) is None:
                    self._pred.clear()
                    self.status_var.set(
                        f"Classification stopped: incomplete statistics on "
                        f"slice {si}:{li}.")
                    return
                count += 1
        except TrainingProblem as problem:
            self._pred.clear()
            self.status_var.set(str(problem))
            return
        finally:
            self._clear_compute_badge()
        self._refresh_region_modes()
        self._refresh_confusion()
        self._refresh_render()
        self._refresh_edge_readout()
        self.status_var.set(f"Classified {count} slice(s) in "
                            f"{1e3 * (time.perf_counter() - t0):.0f} ms - "
                            "predictions shown under your labels")

    def _predict_slice(self, si, li, rec, np):
        """The slice's region->class predictions at rec's commit, computing
        and caching them when a model is loaded. Callers gate schema
        compatibility (this only checks that a model exists).

        Caches ``(commit, region_class, region_proba)``: the class
        probabilities come out of the same forward pass as the hard label
        (which is their argmax), and the regions coloring modes and any later
        confidence read need them per region, not per feature row."""
        ctx = getattr(self, "_clf_context", None)
        if (ctx is not None and ctx.labels is not None and self._pred
                and getattr(self, "_pred_store_rev", None) != self.store.rev):
            self._pred.clear()           # labels are context: the store moved
        pr = self._pred.get(self.catalogue.key_of(si, li))
        if pr is not None and pr[0] == rec.get("commit"):
            return pr[1]
        if self._clf is None:
            return None
        table = rec.get("stats")
        if getattr(table, "values", None) is None:
            return None
        # The model's OWN context columns (not the Model tab's, which may
        # already describe the next model): the loaded spec rebuilds them.
        try:
            table = self._context_table(self.catalogue.key_of(si, li), rec,
                                        getattr(self, "_clf_context", None), np)
        except ValueError as exc:
            self._log(f"context columns unavailable on slice {si}:{li}: {exc}")
            return None
        mat = self._feature_matrix(table, self._clf_names, np)
        fids = table.column("feature_id")
        if mat is None or fids is None:
            return None
        labels = rec["labels"]
        # The latent-ring head, when there is one: the base net embeds the
        # rows, the ring columns are built over the slice's graph, and the
        # head's probabilities are the prediction (the base's on any failure,
        # logged -- the edge model below still embeds with the base).
        proba = classes = None
        latent = getattr(self, "_context_model", None)
        if latent is not None:
            try:
                g, area = self._latent_graph(self.catalogue.key_of(si, li), table, labels,
                                             latent.spec, np)
                proba, classes = context.predict_latent(latent, self._clf, mat, g, np, area=area)
                proba = np.asarray(proba, np.float32)
                classes = np.asarray(classes, int)
            except (ValueError, TypeError, IndexError) as exc:
                self._log(f"latent head unusable on slice {si}:{li}: {exc}")
                proba = classes = None
        if proba is None:
            # Column order is the estimator's own classes_, NOT 1..N: both
            # kinds sit behind a Pipeline and neither promises contiguous ids.
            proba = np.asarray(self._clf.predict_proba(mat), np.float32)
            classes = np.asarray(self._clf.classes_, int)
        pred = classes[proba.argmax(1)].astype(np.uint8)
        K = max(int(labels.max()) + 1 if labels.size else 1, 1)
        region_class = np.zeros(K, np.uint8)
        # MAX_CLASSES-wide, not n_classes-wide: the class count can change
        # under a cached slice, and indexing by class id keeps the shape.
        region_proba = np.zeros((K, MAX_CLASSES), np.float32)
        fid = fids.astype(int)
        ok = (fid >= 0) & (fid < K)
        region_class[fid[ok]] = pred[ok]
        keep = (classes >= 0) & (classes < MAX_CLASSES)
        region_proba[fid[ok][:, None], classes[keep][None, :]] = \
            proba[ok][:, keep]
        # The edge model: p(diff) per arc of the RECORD's graph (so the magic
        # fill and the coloring modes index it without a row mapping), and
        # neighbour voting when an edge kind is selected. `raw` keeps the base
        # net's answer so a kind flip re-votes without a forward pass.
        aux = None
        final = region_class
        edge = self._edge_model
        if edge is not None:
            arcs = self.regions.arcs(self.catalogue.key_of(si, li), np)
            if arcs is not None and len(arcs.get("a", ())):
                try:
                    ia, ib, keep_e = magic_fill.index_arcs(arcs, fid, np)
                    names = list(self._clf_names)
                    ext = (mat[:, names.index("ext_filtered")]
                           if "ext_filtered" in names else None)
                    sad = arcs.get("saddle")
                    sad = None if sad is None else np.asarray(sad, np.float64)[keep_e]
                    contact = None
                    if "contact" in edge.spec.features:
                        contact = context.ensure_contact(arcs, labels, np)[keep_e]
                    pd_rows = edge_model.predict_pdiff(edge, self._clf, mat, ia, ib, sad, ext,
                                                       contact=contact)
                    pdiff = np.full(len(arcs["a"]), np.nan, np.float32)
                    pdiff[keep_e] = pd_rows
                    spec = self._edge_spec_from_ui()
                    aux = {"raw": region_class.copy(), "pdiff": pdiff, "keep": keep_e,
                           "la": np.asarray(arcs["a"]), "lb": np.asarray(arcs["b"]),
                           "lam": spec.lam, "rounds": spec.rounds, "flips": 0,
                           "active": False, "source": arcs.get("source")}
                    if _is_edge_kind(self.model_kind_var.get()):
                        final, flips = self._vote_entry(region_proba, aux, spec.lam,
                                                        spec.rounds, np)
                        aux["flips"], aux["active"] = flips, True
                except (ValueError, IndexError) as exc:
                    self._log(f"edge model unusable on slice {si}:{li}: {exc}")
                    aux = None
        entry = ((rec.get("commit"), final, region_proba) if aux is None
                 else (rec.get("commit"), final, region_proba, aux))
        self._pred[self.catalogue.key_of(si, li)] = entry
        self._pred_store_rev = self.store.rev
        return final

    # -- training-set export --------------------------------------------- #
    def _ensure_slice_record(self, si, li):
        """The slice's record at the current commit, computed synchronously when
        the lazy per-slice tier hasn't visited it yet (see
        RegionProvider.ensure_record)."""
        return self.regions.ensure_record(self.catalogue.key_of(si, li))

    # -- classifier persistence (pickle: the sklearn-native format) ------ #
    def _save_classifier(self):
        if self._clf is None:
            self.status_var.set("Train first.")
            return
        path = filedialog.asksaveasfilename(title="Save classifier",
                                            defaultextension=".pkl",
                                            initialfile="classifier.pkl",
                                            filetypes=[("Pickle", "*.pkl")])
        if not path:
            return
        self._save_classifier_to(path)
        self.status_var.set(f"Wrote {path}")

    def _save_classifier_to(self, path):
        """Pickle v4 (`msseg.labeler.bundle.ModelBundle`): model + feature names
        + kind + the statistics block it was trained under (the fingerprint the
        session records) + the tuned spec + the edge model and stack settings.
        v1 (no kind/statistics), v2 (no spec) and v3 (no edge/stack) pickles
        still load."""
        self._snapshot_active_profile()
        stats = dict(self.profiles[self.active_profile_idx].get("statistics") or {})
        spec = None if self._clf_spec is None else self._clf_spec.to_dict()
        # v4: the edge model on top (None for plain kinds) and the stack's
        # settings, so a loaded '-> edges' model can be retrained as is.
        edge = None if self._edge_model is None else self._edge_model.to_dict()
        stack = {"custom_hidden": self.custom_hidden_var.get(),
                 "edge_spec": self._edge_spec_from_ui().to_dict()}
        # The context columns the model was fit over, written only when there
        # are any so a context-free pickle stays the document it always was.
        ctx = getattr(self, "_clf_context", None)
        if ctx is not None and not ctx.empty():
            stack["context"] = ctx.to_dict()
        latent = getattr(self, "_context_model", None)
        if latent is not None:
            stack["latent"] = latent.to_dict()
        # The seam model rides along when one was fit (its own key, written
        # only then, so a seam-less pickle is unchanged).
        sm = getattr(self, "_seam_model", None)
        seam = None if sm is None else sm.to_dict()
        model_bundle.ModelBundle(model=self._clf, names=list(self._clf_names), kind=self._clf_kind,
                                 statistics=stats, spec=spec, edge=edge, stack=stack,
                                 seam=seam, app_tag=self.MODEL_APP_TAG).save(path)
        self._record_model(path, stats)

    def _record_model(self, path, statistics):
        """Register a saved/loaded model on the session (deduped by path)."""
        ctx = getattr(self, "_clf_context", None)
        entry = model_bundle.model_record_entry(
            path, self._clf_names, self._clf_kind, statistics,
            None if self._clf_spec is None else self._clf_spec.to_dict(),
            self._edge_model is not None, getattr(self, "_clf_scope", None),
            context=None if ctx is None or ctx.empty() else ctx.to_dict(),
            has_seam=getattr(self, "_seam_model", None) is not None)
        self.models = [m for m in self.models if m.get("path") != entry["path"]]
        self.models.append(entry)

    def _load_classifier(self):
        path = filedialog.askopenfilename(title="Load classifier",
                                          filetypes=[("Pickle", "*.pkl")])
        if not path:
            return
        try:
            self._load_classifier_from(path, interactive=True)
        except Exception as exc:     # missing sklearn, wrong file, incompat
            messagebox.showerror(self.APP_TITLE, str(exc))
            self.status_var.set(f"Could not load classifier: {exc}")
            return
        self.status_var.set(f"Loaded {self._clf_kind} from {path} "
                            f"({len(self._clf_names)} features)")

    def _load_classifier_from(self, path, interactive=False):
        """Install a pickled model. `interactive` is opt-in: session restore
        and the selftest call this headlessly, where a modal would hang."""
        doc = model_bundle.ModelBundle.load(path, self.MODEL_APP_TAG)
        # The context columns the pickle's names include: the gate must expect
        # them, and the Model tab follows them only once the gate has passed.
        ctx = context.ContextSpec.from_dict(doc.stack.get("context"))
        # The compatibility gate: a model trained under different statistics
        # is refused OUTRIGHT (per-feature values would silently mean the
        # wrong thing), before anything is installed. Interactively -- and only
        # for a v2 pickle, which carries the statistics it was trained under --
        # the user is first offered a profile built from those statistics.
        msg = self._check_model_compat(doc.names, "load", doc.scope, ctx)
        if msg and interactive and doc.statistics:
            if not messagebox.askyesno(
                    self.APP_TITLE,
                    msg + "\n\nCreate a profile from the model's own statistics "
                          "and switch to it?"):
                raise ValueError(msg)
            self._profile_from_model(path, doc.statistics)
            # The new profile can still miss: feature_fields may resolve
            # differently here than in the build that saved the pickle.
            msg = self._check_model_compat(doc.names, "load", doc.scope, ctx)
        if msg:
            raise ValueError(msg)
        self._pred.clear()               # predictions belong to the old model
        self._clf = doc.model
        self._clf_names = list(doc.names)
        self._clf_scope = doc.scope
        self._clf_context = ctx
        self._apply_context_spec(ctx)
        self._clf_kind = doc.kind
        self._clf_spec = (model_search.ModelSpec.from_dict(doc.spec)
                          if doc.spec is not None else None)
        if self._clf_spec is not None and _base_kind(self._clf_kind) == _TUNED_KIND:
            self._search_spec = self._clf_spec   # Train rebuilds what was loaded
        # v4: the edge model on top, only when it was fit over these features.
        self._edge_model = None
        if doc.edge is not None:
            try:
                em = edge_model.EdgeModel.from_dict(doc.edge)
                if em.names_hash == edge_model.names_hash(self._clf_names):
                    self._edge_model = em
                else:
                    self._log("edge model in the pickle was fit over other features - dropped")
            except Exception as exc:
                self._log(f"edge model in the pickle not restored: {exc}")
        # The seam model, only when its pair block did not embed with a base
        # net other than this one (a 'features' embedding is base-independent).
        self._seam_model = None
        self._seam_pred.clear()
        if doc.seam is not None:
            try:
                sm = seam_model.SeamModel.from_dict(doc.seam)
                if (sm.embed_used == "net" and sm.net_hash
                        and sm.net_hash != edge_model.net_hash(self._clf)):
                    self._log("seam model in the pickle was fit on another base net - dropped")
                else:
                    self._seam_model = sm
            except Exception as exc:
                self._log(f"seam model in the pickle not restored: {exc}")
        stack = doc.stack
        # The latent-ring head on top, only when it was fit over these
        # features and on this very base net.
        self._context_model = None
        if isinstance(stack.get("latent"), dict):
            try:
                lm = context.LatentContextModel.from_dict(stack["latent"])
                if (lm.names_hash == edge_model.names_hash(self._clf_names)
                        and (not lm.net_hash or lm.net_hash == edge_model.net_hash(self._clf))):
                    self._context_model = lm
                else:
                    self._log("latent head in the pickle was fit over another base - dropped")
            except Exception as exc:
                self._log(f"latent head in the pickle not restored: {exc}")
        if isinstance(stack.get("custom_hidden"), str):
            try:
                model_search.parse_sizes(stack["custom_hidden"])
                self.custom_hidden_var.set(stack["custom_hidden"])
            except ValueError:
                pass
        if isinstance(stack.get("edge_spec"), dict):
            self._apply_edge_spec(edge_model.EdgeSpec.from_dict(stack["edge_spec"]))
        if self._clf_kind in _MODEL_KINDS:
            self.model_kind_var.set(self._clf_kind)
        self._refresh_model_readout()
        self.classify_btn.config(state="normal")
        self._record_model(path, doc.statistics)
        self._refresh_model_strip()
        self._refresh_edge_readout()
