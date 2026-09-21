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

from . import edge_model, model_search, seam_export, seam_model
from .training import TrainingProblem
from .defaults import *  # noqa: F401,F403


class SeamModelMixin:
    # ------------------------------------------------------------------ #
    # Gathering
    # ------------------------------------------------------------------ #
    def _seam_spec(self):
        return seam_model.SeamSpec()

    def _seam_base(self):
        """``(pipeline, names)`` of the region net the pair block may embed
        with: the current classifier when it has hidden layers and its
        inputs are plain table columns (no context columns); else
        ``(None, None)``."""
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
                pr = self._pred.get(key)
                aux = self._pred_aux(pr) if pr is not None and pr[0] == rec.get("commit") else None
                pdiff = None if aux is None else aux.get("pdiff")
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

    def _seam_net_hash(self, embed):
        clf = getattr(self, "_clf", None)
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
        self._compute_badge("Fitting seams")
        try:
            model = seam_model.fit_seam_model(data["F"], data["y"], spec, fnames, embed=embed,
                                              net_hash=self._seam_net_hash(embed))
        except ValueError as exc:
            self.status_var.set(f"Seam model not fit: {exc}")
            return
        finally:
            self._clear_compute_badge()
        self._seam_model = model
        self._seam_pred.clear()
        self._log(f"seam model: {model.describe()}; {len(fnames)} inputs, "
                  f"{1e3 * model.fit_s:.0f} ms")
        self._classify_seams(items)
        self._refresh_seam_panel()
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
            self._seam_pred[key] = (rec.get("commit"), seam_model.predict_boundaryness(model, F))
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
            model_search.make_cv(yb[mask], data["group"][mask], seed=seed)
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
                p = entry[1] if entry is not None and entry[0] == rec.get("commit") else None
                path = os.path.join(folder, f"seams_{seam_export.item_stem(key)}.json")
                doc = seam_export.write_seams_json(path, key, graph, cls, p, np)
                rows.extend(seam_export.summary_rows(key, doc))
                n_items += 1
                n_seams += graph.n_seams
        except TrainingProblem as problem:
            self.status_var.set(str(problem))
            return n_items, n_seams
        seam_export.write_summary_csv(os.path.join(folder, "seams_summary.csv"), rows)
        self._log(f"seams exported: {n_seams} seams of {n_items} item(s) to {folder}")
        return n_items, n_seams
