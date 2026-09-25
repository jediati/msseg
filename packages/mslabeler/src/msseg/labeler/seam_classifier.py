"""``SeamModelMixin``: the seam model's lifecycle in the labeler -- gather
every primed item's seam descriptors, fit (Train seams), score every seam
(the ``model`` toll and the boundaryness colouring), evaluate leave-items-out
on the Optimize worker/pump, export, and ride the classifier pickle
(``ModelBundle.seam``). Tk thread only, like ``ClassifierMixin``.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from tkinter import filedialog

from . import artifacts
from . import bundle as model_bundle
from . import edge_model, model_search, seam_export, seam_model
from .training import TrainingProblem
from .defaults import *  # noqa: F401,F403


class SeamModelMixin:
    # ------------------------------------------------------------------ #
    # Gathering
    # ------------------------------------------------------------------ #
    def _seam_spec(self):
        return seam_model.SeamSpec()

    # ------------------------------------------------------------------ #
    # Inputs (``artifacts``): the region work a polyline task reads
    # ------------------------------------------------------------------ #
    def _input_entry(self, slot):
        """The active polyline task's slot entry, or None."""
        if self._task_kind() != "polyline":
            return None
        return (self._task.inputs or {}).get(slot)

    def _input_resolved(self, slot):
        """``(provider, why)``: provider None when the slot is empty, why
        None when the provider can fill it under the active workflow."""
        ref = artifacts.ProviderRef.from_doc(self._input_entry(slot))
        if ref is None:
            return None, None
        prov = artifacts.resolve(ref, self.tasks, app_tag=self.MODEL_APP_TAG,
                                 consumer=self._task.uid)
        return prov, self._input_why_not(slot, prov)

    def _input_provider(self, slot):
        """The slot's provider when it is usable here, else None."""
        prov, why = self._input_resolved(slot)
        return prov if prov is not None and why is None else None

    def _input_why_not(self, slot, prov):
        """The slot's refusal, cached by what it depends on (the provider's
        model, the measurement, the scope and the workflow): the schema call
        is not free on the stage strip's poll."""
        sig = (prov.signature(slot), self._measure_key(), self._feature_scope(),
               self.active_profile_idx)
        cache = self.__dict__.setdefault("_input_why_cache", {})
        hit = cache.get(slot)
        if hit is not None and hit[0] == sig:
            return hit[1]
        why = prov.unavailable(slot)
        if why is None:
            try:
                have = self._expected_feature_names()
            except Exception:
                have = None
            prof = "?"
            if 0 <= self.active_profile_idx < len(self.profiles):
                prof = self.profiles[self.active_profile_idx]["name"]
            why = prov.requires(slot).why_not(have, self._feature_scope(), f"workflow {prof!r}")
        cache[slot] = (sig, why)
        return why

    def _input_boundary_class(self):
        """The class a derived boundary gets: the labels slot's setting,
        brought into the task's vocabulary (class 1 is "not a boundary")."""
        k = artifacts.boundary_class_of(self._input_entry("labels"))
        return max(2, min(k, self.store.n_classes - 1))

    def _input_labels_sig(self):
        prov = self._input_provider("labels")
        return None if prov is None else (prov.signature("labels"), self._input_boundary_class())

    def _input_signature(self):
        """What the polyline task's inputs are, as one value: each filled
        slot's provider signature (None when unusable) -- the seam model is
        stale when this moves under it. () in a region task."""
        if self._task_kind() != "polyline":
            return ()
        out = []
        for slot in artifacts.SLOTS:
            if self._input_entry(slot) is None:
                continue
            prov = self._input_provider(slot)
            out.append((slot, None if prov is None else prov.signature(slot)))
        if self._input_entry("labels") is not None:
            out.append(("boundary_class", self._input_boundary_class()))
        return tuple(out)

    def _input_region_labels(self, si, li, key, rec, np):
        """``(region_class, extents)`` over this item's regions from the
        labels slot's provider: its region gestures on the item's slide,
        resolved against THIS task's decomposition (gestures are geometry).
        ``(None, [])`` when the slot is empty or unusable."""
        prov = self._input_provider("labels")
        if prov is None or key is None:
            return None, []
        from . import derive
        from .labeling import resolve_sets, touched_sets
        slide, rect = self._binding_of(key)
        gestures = self._coarse_filter(prov.gestures_for(slide, rect), si, li)
        if not gestures:
            return None, []
        sets = touched_sets(gestures, rec["labels"], np, self.regions.label_layer(key))
        region_class = resolve_sets(sets, rec["labels"], np)
        return region_class, [(it, ids) for it, ids in sets if derive.is_extent(it)]

    def _input_pdiff(self, key, rec, np):
        """p(diff) per arc of this item's record from the pdiff slot's edge
        model, cached per (commit, provider); None when there is none."""
        prov = self._input_provider("pdiff")
        if prov is None or rec is None or rec.get("stats") is None:
            return None
        sig = prov.signature("pdiff")
        cache = self.__dict__.setdefault("_input_pdiff_cache", {})
        hit = cache.get(key)
        if hit is not None and hit[0] == rec.get("commit") and hit[1] == sig:
            return hit[2]
        try:
            pd = artifacts.pdiff_for(prov.stack(), rec["stats"], self.regions.arcs(key, np),
                                     rec["labels"], np)
        except (ValueError, IndexError, TypeError) as exc:
            self._log(f"p(diff) input unusable on {key}: {exc}")
            pd = None
        cache[key] = (rec.get("commit"), sig, pd)
        return pd

    def _set_input(self, slot, uid, boundary_class=None):
        """Fill (a task uid) or empty (None) one of the active polyline
        task's slots; headless-callable. False in a region task."""
        task = self._task
        if task.kind != "polyline" or slot not in artifacts.SLOTS:
            return False
        inputs = dict(task.inputs or {})
        old = inputs.get(slot) or {}
        if uid is None:
            inputs.pop(slot, None)
        else:
            entry = artifacts.ProviderRef.task(uid).to_doc()
            if slot == "labels":
                bc = boundary_class if boundary_class is not None else old.get("boundary_class")
                if bc is not None:
                    entry["boundary_class"] = artifacts.boundary_class_of({"boundary_class": bc})
            inputs[slot] = entry
        task.inputs = inputs or None
        self._input_changed(slot)
        return True

    def _set_input_boundary_class(self, k):
        entry = self._input_entry("labels")
        if entry is None:
            return False
        entry["boundary_class"] = artifacts.boundary_class_of({"boundary_class": k})
        self._input_changed("labels")
        return True

    def _input_changed(self, slot):
        self.__dict__.pop("_input_why_cache", None)
        self._clear_seam_caches()
        if slot != "labels":
            # The seam descriptor's values moved under the predictions.
            self._seam_pred.clear()
        refresh = getattr(self, "_refresh_inputs_panel", None)
        if refresh is not None:
            refresh()
        self._refresh_seam_panel()
        self._refresh_confusion()
        self._refresh_stages()
        self._refresh_render()

    def _seam_base(self):
        """``(pipeline, names)`` of the region net the pair block may embed
        with: in a polyline task, the embedding slot's provider; else the
        current classifier when it has hidden layers and its inputs are
        plain table columns (no context columns); else ``(None, None)``."""
        if self._task_kind() == "polyline":
            prov = self._input_provider("embedding")
            return prov.base() if prov is not None else (None, None)
        clf = getattr(self, "_clf", None)
        if clf is None or not seam_model.has_embedding(clf):
            return None, None
        ctx = getattr(self, "_clf_context", None)
        if ctx is not None and not ctx.empty():
            return None, None
        return clf, list(self._clf_names or [])

    def _seam_items(self, action="Gathering seams"):
        """``[(key, si, li, rec, graph, cls, F, feature_names, embed)]`` over
        every primed item, or None with the reason in the status bar."""
        import numpy as np
        slices = self._stream_stat_slices(action)
        if slices is None:
            return None
        base, base_names = self._seam_base()
        spec = self._seam_spec()
        out = []
        try:
            for si, li, key, rec, table in slices:
                graph = self.regions.seams(key, np)
                if graph is None:
                    raise TrainingProblem(f"{action} stopped: no seam graph for {key}.")
                cls = self._seam_cache_for(si, li, rec, graph, np)[2]
                names = base_names if base is not None else seam_model.region_feature_names(table, self.FIELDS)
                if base is not None and any(table.column(n) is None for n in names):
                    names = seam_model.region_feature_names(table, self.FIELDS)
                    use_base = None
                else:
                    use_base = base
                arcs = self.regions.arcs(key, np)
                pdiff = self._seam_pdiff(key, rec, np)
                F, fnames = seam_model.seam_features(graph, table, names, np, conv=self.FIELDS,
                                                     base_pipeline=use_base, arcs=arcs,
                                                     pdiff=pdiff, spec=spec)
                out.append((key, si, li, rec, graph, cls, F, fnames,
                            seam_model.embed_used(spec, use_base)))
        except TrainingProblem as problem:
            self.status_var.set(str(problem))
            return None
        except ValueError as exc:
            self.status_var.set(f"{action}: {exc}")
            return None
        if out and len({tuple(it[7]) for it in out}) > 1:
            self.status_var.set(f"{action}: the items' seam descriptors differ (different "
                                "statistics or an item without the base's columns).")
            return None
        return out

    def _seam_pdiff(self, key, rec, np):
        """p(diff) per arc for the seam descriptor and the edges toll: the
        pdiff input in a polyline task, the task's own edge model else."""
        if self._task_kind() == "polyline":
            return self._input_pdiff(key, rec, np)
        pr = self._pred.get(key)
        aux = self._pred_aux(pr) if pr is not None and pr[0] == rec.get("commit") else None
        return None if aux is None else aux.get("pdiff")

    def _seam_net_hash(self, embed):
        clf = self._seam_base()[0]
        if embed != "net" or clf is None:
            return ""
        try:
            return edge_model.net_hash(clf)
        except TypeError:
            return ""

    # ------------------------------------------------------------------ #
    # Train / score
    # ------------------------------------------------------------------ #
    def _train_seam_model(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.status_var.set("scikit-learn is not installed - pip install scikit-learn "
                                "to enable the seam model")
            return
        items = self._seam_items("Gathering seams")
        if not items:
            return
        data = seam_model.gather_seams(
            (key, F, cls, self.catalogue.group_of(key)) for key, _si, _li, _rec, _g, cls, F, _n, _e in items)
        fnames, embed = items[0][7], items[0][8]
        spec = self._seam_spec()
        t0 = time.perf_counter()
        self._compute_badge("Fitting seams", stage="model")
        try:
            model = seam_model.fit_seam_model(data["F"], data["y"], spec, fnames, embed=embed,
                                              net_hash=self._seam_net_hash(embed),
                                              n_classes=self.store.n_classes)
        except ValueError as exc:
            self.status_var.set(f"Seam model not fit: {exc}")
            return
        finally:
            self._clear_compute_badge()
        self._seam_model = model
        # What it was trained on, for the stage strip (as the region model).
        self._task.model.trained_rev = self.store.rev
        self._task.model.trained_measure = self._measure_key()
        self._task.model.trained_inputs = self._input_signature()
        self._seam_pred.clear()
        self._log(f"seam model: {model.describe()}; {len(fnames)} inputs, "
                  f"{1e3 * model.fit_s:.0f} ms")
        self._classify_seams(items)
        self._refresh_seam_panel()
        self._refresh_confusion()
        self._update_task_rows()
        self._refresh_render()
        self.status_var.set(f"Trained the seam model on {model.n_seams} labelled seams "
                            f"({model.n_boundary} boundary) in "
                            f"{1e3 * (time.perf_counter() - t0):.0f} ms - {model.brief()}")

    def _classify_seams(self, items=None):
        """Score every seam of every item with the seam model (boundaryness,
        cached per commit)."""
        model = self._seam_model
        if model is None:
            self.status_var.set("Train seams first.")
            return
        if items is None:
            items = self._seam_items("Scoring seams")
            if items is None:
                return
        for key, _si, _li, rec, _graph, _cls, F, fnames, embed in items:
            if list(fnames) != list(model.feature_names):
                self.status_var.set("The seam descriptor changed since the seam model was "
                                    "fit - Train seams again.")
                return
            if embed == "net" and model.net_hash and self._seam_net_hash(embed) != model.net_hash:
                self.status_var.set("The seam model was fit on another region net - "
                                    "Train seams again.")
                return
            proba = seam_model.predict_proba(model, F)
            self._seam_pred[key] = (rec.get("commit"), seam_model.boundaryness_of(proba), proba)
        self._refresh_confusion()
        self._refresh_stages()
        self._refresh_render()

    # ------------------------------------------------------------------ #
    # Evaluate (worker + the Optimize pump)
    # ------------------------------------------------------------------ #
    def _evaluate_seams(self, sync=False):
        if self._search is not None:
            self.status_var.set("A search is already running - Cancel it first.")
            return
        items = self._seam_items("Gathering seams")
        if not items:
            return
        data = seam_model.gather_seams(
            (key, F, cls, self.catalogue.group_of(key)) for key, _si, _li, _rec, _g, cls, F, _n, _e in items)
        mask, yb = seam_model.labeled_binary(data["y"])
        _trials, _timeout, seed, _feat, _backend = self._search_settings()
        try:
            model_search.make_cv(data["y"][mask], data["group"][mask], seed=seed)
        except ValueError as exc:
            self.status_var.set(f"Cannot evaluate seams: {exc}.")
            return
        spec = self._seam_spec()
        q = queue.Queue()
        stop = threading.Event()
        F, y, grp = data["F"], data["y"], data["group"]

        def work():
            try:
                rep = seam_model.evaluate_seams(
                    F, y, grp, spec, seed=seed,
                    progress_cb=lambda f, n: q.put(("seam_progress", f, n)), stop_event=stop)
                q.put(("seam_done", rep))
            except Exception as exc:            # reported, never raised off-thread
                q.put(("error", f"{type(exc).__name__}: {exc}"))

        self._search = {"queue": q, "stop": stop, "thread": None, "names": list(items[0][7]),
                        "n": int(mask.sum()), "t0": time.perf_counter(), "mode": "seams"}
        self._set_search_buttons(running=True)
        self._compute_badge("Evaluating seams")
        self._log(f"evaluate seams: {spec.describe()}; {int(mask.sum())} labelled seams "
                  f"({int(yb[mask].sum())} boundary) over {len(items)} item(s)")
        if sync:
            work()
            self._search_pump()
            return
        t = threading.Thread(target=work, name=f"{self.LOG_PREFIX}-seam-eval", daemon=True)
        self._search["thread"] = t
        t.start()
        self.root.after(_SEARCH_PUMP_MS, self._search_pump)

    def _finish_seam_eval(self, report):
        text = seam_model.summary(report)
        self._log("evaluate seams: " + text)
        if self._seam_model is not None:
            self._seam_model.report = report
        self._seam_report = report
        self._refresh_seam_panel()
        self.status_var.set("Evaluated seams - " + text)

    # ------------------------------------------------------------------ #
    # The seam model on disk (bundle.SeamBundle)
    # ------------------------------------------------------------------ #
    def _save_seam_model_to(self, path):
        model = self._seam_model
        if model is None:
            raise ValueError("no seam model - Train (R) first")
        self._snapshot_active_profile()
        stats = dict(self.profiles[self.active_profile_idx].get("statistics") or {})
        b = model_bundle.SeamBundle(seam=model.to_dict(), statistics=stats,
                                    classes={k: self._class_name(k)
                                             for k in range(1, self.store.n_classes)},
                                    scope=self._feature_scope(), app_tag=self.MODEL_APP_TAG)
        b.save(path)
        entry = b.record_entry(path)
        self.models = [m for m in self.models if m.get("path") != entry["path"]]
        self.models.append(entry)
        self._update_task_rows()
        return entry

    def _load_seam_model_from(self, path):
        """Install a saved seam model. Its descriptor is checked when it
        classifies (the features can only be built with a record in hand)."""
        b = model_bundle.SeamBundle.load(path, self.MODEL_APP_TAG)
        model = seam_model.SeamModel.from_dict(b.seam)
        self._seam_model = model
        self._seam_pred.clear()
        self._task.model.trained_rev = None      # a loaded model makes no claim
        self._task.model.trained_measure = None
        entry = b.record_entry(path)
        self.models = [m for m in self.models if m.get("path") != entry["path"]]
        self.models.append(entry)
        self._refresh_seam_panel()
        self._refresh_confusion()
        self._update_task_rows()
        return model

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def _export_seams(self):
        folder = filedialog.askdirectory(title="Export seams to folder")
        if not folder:
            return
        try:
            n_items, n_seams = self._export_seams_to(folder)
        except Exception as exc:
            self.status_var.set(f"Seam export failed: {exc}")
            return
        if n_items:
            self.status_var.set(f"Exported {n_seams} seams of {n_items} item(s) to {folder}")

    def _export_seams_to(self, folder):
        """Write ``seams_<item>.json`` per primed item and ``seams_summary.csv``;
        returns ``(n_items, n_seams)``."""
        import numpy as np
        slices = self._stream_stat_slices("Exporting seams")
        if slices is None:
            return 0, 0
        os.makedirs(folder, exist_ok=True)
        rows = []
        n_items = n_seams = 0
        try:
            for si, li, key, rec, _table in slices:
                graph = self.regions.seams(key, np)
                if graph is None:
                    continue
                cls = self._seam_cache_for(si, li, rec, graph, np)[2]
                entry = self._seam_pred.get(key)
                live = entry is not None and entry[0] == rec.get("commit")
                p = entry[1] if live else None
                pred = seam_model.predicted_class(entry[2]) if live and len(entry) > 2 else None
                names = {k: self._class_name(k) for k in range(1, self.store.n_classes)}
                path = os.path.join(folder, f"seams_{seam_export.item_stem(key)}.json")
                doc = seam_export.write_seams_json(path, key, graph, cls, p, np, names, pred)
                rows.extend(seam_export.summary_rows(key, doc))
                n_items += 1
                n_seams += graph.n_seams
        except TrainingProblem as problem:
            self.status_var.set(str(problem))
            return n_items, n_seams
        seam_export.write_summary_csv(os.path.join(folder, "seams_summary.csv"), rows)
        self._log(f"seams exported: {n_seams} seams of {n_items} item(s) to {folder}")
        return n_items, n_seams
