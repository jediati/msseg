"""UI-free compute core shared by the mscoupon Tk apps (viewer + labeler).

``ComputeEngine`` owns the primed stacks, the per-slice result cache, the 3D
assemblies, and the worker threads that fill them. It never touches Tk: the
caller supplies parameter snapshots as plain dicts (``params_provider``, called
on the UI thread at launch time so a superseding request picks up the freshest
state), and drains completion events on its own thread via ``poll()``.

Threading model (unchanged from the original in-app implementation):

- ONE daemon priming worker walks slices serially (``start_run``); each slice's
  MSC can still be internally parallel inside MSCEER.
- ONE assembly worker at a time (single-flight): ``Msc2DPipeline`` is stateful
  (``select_persistence`` mutates it), so a newer request supersedes the pending
  one rather than running beside it.
- Workers post to an internal queue; ``poll()`` (UI thread) applies the engine
  bookkeeping and returns display events:
      ("progress", (done, total))     priming progress
      ("error", exc)                  priming failed
      ("primed",)                     priming finished; self.primed is set
      ("assembly_done", si, accepted) an assembly landed (accepted=False when a
                                      newer token superseded it)
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time

from .common import log, FeatureTable


class ComputeEngine:
    def __init__(self, params_provider):
        # params_provider(si, level, li) -> dict: the UI-state snapshot for one
        # assembly work item (pct/queries/pixels/connectivity/min_area/manifold/
        # json/reductions/extremum + "name" for logging). Called on the UI
        # thread when the work item actually LAUNCHES; the engine stamps
        # "commit"/"level"/"li" itself.
        self._params_provider = params_provider

        # primed[subseq_idx] = {"files":[...], "pipes":[pipe|None], "base":[arr],
        #                       "filtered":[arr]}  (populated by start_run)
        self.primed = []
        # subseq_idx -> 3D assembly result (cc/global label rasters + global table).
        # Only populated at the "global" level -- see the app's _needed_level().
        self.assembly = {}
        # (subseq_idx, local_idx) -> per-slice result at some commit:
        #   {commit, labels, stats, kept, cc (optional)}
        # The per-slice tiers write here; the global tier fills it for every slice
        # as a by-product, so navigating after a full assembly costs nothing.
        self.slices = {}
        self.work_q = queue.Queue()
        # Async assembly (off the UI thread): pipes are stateful, so only ONE
        # assembly worker runs at a time (single-flight); newer requests supersede.
        self.asm_token = 0
        self.asm_running = False
        self.asm_running_si = None               # subsequence the worker is assembling
        self.asm_pending = None                  # (token, si, level, li) latest requested
        self.commit_id = 0                       # committed parameter generation
        self.run_active = False                  # priming worker in flight

    # ------------------------------------------------------------------ #
    # Priming
    # ------------------------------------------------------------------ #
    def start_run(self, subseqs, params, run_info):
        """Prime `subseqs` (list of {"name", "files"} dicts) with `params`
        (the params_json string). `run_info` carries the UI's concurrency
        numbers for the log line so the worker never reads Tk state."""
        self.run_active = True
        t = threading.Thread(target=self._run_worker,
                             args=(subseqs, params, run_info), daemon=True)
        t.start()

    @staticmethod
    def _apply_base_chain(arr, base_filters, engine, log):
        """Run the base-channel chain, returning (raster, measured landmarks).

        A `normalize` stage is measured here rather than inside
        ``engine.filter_slice`` so the GUI can show the landmarks it resolved --
        it is the same C++ measure and the same affine map, so the exported
        config reproduces this exactly.
        """
        import numpy as np
        from msseg.mscoupon.normalize import measure_two_point

        cur = arr
        measured = []
        for i, f in enumerate(base_filters):
            if f.get("operation") == "normalize":
                params = dict(f.get("params", {}))
                tp = measure_two_point(cur, **params)
                cur = tp.apply(cur, clamp=bool(params.get("clamp", False)))
                measured.append(tp)
                log(f"  base[{i}] normalize({params.get('method', 'gmm')}) "
                    f"-> low={tp.low:.6g} high={tp.high:.6g} "
                    f"[{cur.min():.4g}, {cur.max():.4g}]")
            else:
                cur = engine.filter_slice(cur, json.dumps({"filter": f}))
                log(f"  base[{i}] {f['operation']}({f.get('params', {})}) "
                    f"-> min={cur.min():.4g} max={cur.max():.4g}")
        return np.ascontiguousarray(cur, dtype=np.float32), measured

    def _run_worker(self, subseqs, params, run_info):
        try:
            import numpy as np
            from msseg import mscoupon as engine
            from PIL import Image
            p = json.loads(params)
            filters = p.get("filters", [])
            base_filters = p.get("base_filters", [])
            msc = p.get("msc", {})
            # Serial-MSC variant of the params, used as a graceful fallback if the
            # linked MSCEER lacks the partitioned ComputeOptions surface.
            msc_serial = {k: v for k, v in msc.items()
                          if k not in ("compute_algorithm", "requested_parallelism")}
            params_serial = json.dumps({"filters": filters, "msc": msc_serial})
            use_serial = "compute_algorithm" not in msc
            total = sum(len(s["files"]) for s in subseqs)
            log("=" * 60)
            log(f"RUN: {len(subseqs)} subsequence(s), {total} slices")
            log(f"  filters: {[f['operation'] for f in filters] or ['(none)']}")
            log(f"  base_filters: {[f['operation'] for f in base_filters] or ['(none)']}")
            log(f"  msc: manifold={msc.get('manifold')} "
                f"persistence_percent={msc.get('persistence_percent')} "
                f"accurate={msc.get('accurate_ascending')} "
                f"algorithm={msc.get('compute_algorithm', 'serial')} "
                f"requested_parallelism={msc.get('requested_parallelism', 0)}")
            # Concurrency picture: priming still walks slices serially in ONE daemon
            # worker thread, but each slice's MSC (discrete gradient / partitioned
            # construction / manifold labeling) runs cores/slice-way parallel inside
            # MSCEER when compute_algorithm=partitioned. "Concurrent slices" (running
            # whole slices at once) is honored by the exported CLI config, which has
            # the full lane pipeline; the live GUI would additionally need the pybind
            # bindings to release the GIL to overlap slices.
            log(f"  concurrency: worker_thread={threading.current_thread().name!r} "
                f"os.cpu_count={os.cpu_count()} "
                f"cores/slice(MSC)={run_info.get('cores_per_slice')} "
                f"concurrent_slices(export)={run_info.get('concurrent_slices')} "
                f"scheduling=serial(1 slice at a time)")
            done = 0
            primed = []
            for s in subseqs:
                log(f"subsequence: {os.path.basename(s['files'][0])} .. "
                    f"({len(s['files'])} slices)")
                base_slices, filt_slices, pipes, norms = [], [], [], []
                for path in s["files"]:
                    t_slice = time.perf_counter()
                    arr = np.asarray(Image.open(path), dtype=np.float32)
                    if arr.ndim == 3:
                        arr = arr.mean(axis=2).astype(np.float32)
                    arr = np.ascontiguousarray(arr)
                    t_load = time.perf_counter()
                    log(f"slice {done + 1}/{total} {os.path.basename(path)}: "
                        f"shape={arr.shape} min={arr.min():.4g} max={arr.max():.4g} "
                        f"mean={arr.mean():.4g}")
                    # Apply the filter chain step by step so each stage's params +
                    # output range are logged (functionally == filter_chain).
                    cur = arr
                    for i, f in enumerate(filters):
                        cur = engine.filter_slice(cur, json.dumps({"filter": f}))
                        log(f"  filter[{i}] {f['operation']}({f.get('params', {})}) "
                            f"-> min={cur.min():.4g} max={cur.max():.4g}")
                    filt = np.ascontiguousarray(cur, dtype=np.float32)
                    # Base channel: the raster statistics and pixel thresholds are
                    # read from. Derived from the raw slice like `filters`, not
                    # chained onto it, matching the C++ pipeline.
                    base, slice_norms = self._apply_base_chain(arr, base_filters, engine, log)
                    t_filter = time.perf_counter()
                    if use_serial:
                        pipe = engine.prime_slice(base, filt, params_serial)
                    else:
                        try:
                            pipe = engine.prime_slice(base, filt, params)
                        except RuntimeError as pe:
                            msg = str(pe)
                            if any(t in msg for t in ("BuilderMode", "ComputeOptions", "partitioned")):
                                log(f"  WARN: partitioned MSC unavailable ({msg}); "
                                    "falling back to serial MSC for remaining slices")
                                use_serial = True
                                pipe = engine.prime_slice(base, filt, params_serial)
                            else:
                                raise
                    t_prime = time.perf_counter()
                    n_at_build = len(pipe.feature_stats())
                    log(f"  MSC primed: value_range={pipe.value_range():.4g} "
                        f"relevance_range=[{pipe.base_relevance_floor():.4g},"
                        f"{pipe.base_relevance_ceiling():.4g}] "
                        f"regions@{msc.get('persistence_percent')}%={n_at_build}")
                    log(f"  slice timings [thread={threading.current_thread().name!r}]: "
                        f"load={1e3 * (t_load - t_slice):.0f}ms "
                        f"filters={1e3 * (t_filter - t_load):.0f}ms "
                        f"prime={1e3 * (t_prime - t_filter):.0f}ms "
                        f"total={1e3 * (t_prime - t_slice):.0f}ms")
                    base_slices.append(base); filt_slices.append(filt); pipes.append(pipe)
                    norms.append(slice_norms)
                    done += 1
                    self.work_q.put(("progress", (done, total)))
                primed.append({"files": s["files"], "base": base_slices,
                               "filtered": filt_slices, "pipes": pipes,
                               "normalizers": norms})
            log(f"RUN complete: primed {total} slices")
            self.work_q.put(("done", primed))
        except Exception as exc:  # surfaced on the UI thread
            log(f"ERROR: {exc}")
            self.work_q.put(("error", exc))

    # ------------------------------------------------------------------ #
    # Event pump (called on the UI thread)
    # ------------------------------------------------------------------ #
    def pending_work(self):
        """True while any async work (priming or assembly) is outstanding."""
        return self.run_active or self.asm_running or self.asm_pending is not None

    def poll(self):
        """Drain the work queue, apply the engine-side bookkeeping, and return
        the display events for the caller (see the module docstring)."""
        events = []
        try:
            while True:
                kind, payload = self.work_q.get_nowait()
                events.append(self._apply(kind, payload))
        except queue.Empty:
            pass
        return [e for e in events if e is not None]

    def _apply(self, kind, payload):
        if kind == "progress":
            return ("progress", payload)
        if kind == "error":
            self.run_active = False
            return ("error", payload)
        if kind == "done":
            self.run_active = False
            self.primed = payload
            self.assembly.clear()
            self.slices.clear()
            return ("primed",)
        if kind == "assembly":
            token, si, out = payload
            self.asm_running = False
            self.asm_running_si = None
            accepted = out is not None and token == self.asm_token
            if accepted:
                if out.get("_level") == "global":
                    self.assembly[si] = out
                    # The global tier also produced every slice's record; cache
                    # them so navigating the stack afterwards needs no further
                    # work.
                    for li, rec in enumerate(out.pop("_slices", [])):
                        self.slices[(si, li)] = rec
                else:
                    self.slices[(si, out["_li"])] = out["_slice"]
            return ("assembly_done", si, accepted)
        return None

    # ------------------------------------------------------------------ #
    # Cache queries
    # ------------------------------------------------------------------ #
    def slice_ready(self, si, li, level):
        """True iff slice `li` is cached at `level` for the current commit."""
        rec = self.slices.get((si, li))
        if rec is None or rec.get("commit") != self.commit_id:
            return False
        if level == "cc" and rec.get("cc") is None:
            return False
        if level == "global":
            data = self.assembly.get(si)
            return data is not None and data.get("_commit") == self.commit_id
        return True

    def record(self, si, li):
        """The per-slice record at the current commit, or None."""
        rec = self.slices.get((si, li))
        if rec is not None and rec.get("commit") != self.commit_id:
            return None
        return rec

    def assembly_for(self, si):
        """The 3D assembly at the current commit, or None."""
        data = self.assembly.get(si)
        if data is not None and data.get("_commit") != self.commit_id:
            return None
        return data

    # ------------------------------------------------------------------ #
    # Commit / reset
    # ------------------------------------------------------------------ #
    def commit_selection(self):
        """Commit the current selection parameters: bump the generation and
        drop the now-stale per-slice records (they are keyed by commit, so they
        fall out of date automatically; pruning keeps memory from growing one
        stack per Rerun)."""
        self.commit_id += 1
        self.slices = {k: v for k, v in self.slices.items()
                       if v.get("commit") == self.commit_id}

    def reset(self):
        """Drop every primed stack, cached slice and assembly (a config load is
        replacing the parameters that produced them).

        A running assembly worker is deliberately left running. Bumping the
        token makes its result land as not-accepted, and the worker's own
        except-clause posts back when it trips over the empty primed list.
        Clearing asm_running here would let a second worker start beside the
        first, and the pipes are not re-entrant."""
        self.primed = []
        self.assembly.clear()
        self.slices.clear()
        self.commit_id += 1        # stale slice records now fail slice_ready
        self.asm_token += 1
        self.asm_pending = None

    # ------------------------------------------------------------------ #
    # Assembly (off the UI thread; single-flight over the stateful pipes)
    # ------------------------------------------------------------------ #
    def request_assembly(self, si, li, level):
        """Queue off-thread work for subsequence `si` at the current parameters.
        Pipes are stateful (select_persistence mutates them), so only ONE worker
        runs at a time; a newer request supersedes an in-flight one.

        Returns the (si, level, li) actually launched now, or None (nothing to
        do, or a worker is running and the request was left pending)."""
        if not self.primed or si is None:
            return None
        if self.slice_ready(si, li, level):
            return None                # already have it at this tier
        self.asm_token += 1
        self.asm_pending = (self.asm_token, si, level, li)
        return self.launch_pending()

    def launch_pending(self):
        """Start the pending work item if no worker is running. Returns the
        (si, level, li) launched, or None."""
        if self.asm_running or self.asm_pending is None:
            return None
        token, si, level, li = self.asm_pending
        self.asm_pending = None
        self.asm_running = True
        self.asm_running_si = si
        # The parameter snapshot is taken NOW (launch time), on the UI thread,
        # so a superseding request measures against the state the user had when
        # it actually runs -- not whatever the panel said when it was queued.
        params = dict(self._params_provider(si, level, li))
        params["commit"] = self.commit_id
        params["level"] = level
        params["li"] = li
        threading.Thread(target=self._assemble_worker, args=(token, si, params),
                         daemon=True).start()
        return (si, level, li)

    def _slice_result(self, si, li, params, engine, np, tm):
        """Per-slice work for one slice: re-threshold, labels, stats, selection.
        Returns the record cached in self.slices."""
        p = self.primed[si]
        pipe = p["pipes"][li]
        pct, queries, min_area = params["pct"], params["queries"], params["min_area"]
        qjson = json.dumps(queries)

        # Re-thresholding is the dominant per-slice cost (MSCEER cancellation), and
        # a Rerun triggered by a filter edit does not move persistence at all.
        # Track what each pipe is already at and skip the no-op -- comparing the
        # requested percentage rather than current_persistence(), because
        # select_persistence clamps to the build-time cap, so a request above the
        # cap would never compare equal.
        applied = p.setdefault("_applied_pct", [None] * len(p["pipes"]))
        t = time.perf_counter()
        if applied[li] != pct:
            pipe.select_persistence(pipe.value_range() * pct / 100.0)
            applied[li] = pct
        tm["persist"] += time.perf_counter() - t

        t = time.perf_counter()
        lab = np.asarray(pipe.labels())
        tm["labels"] += time.perf_counter() - t

        # Columnar: the field names once, then an (n, f) float64 block. A dict
        # per feature cost a Python string and a dict entry per FIELD per
        # feature on every persistence commit -- with a twelve-channel stack
        # that is ~60 fields, so it dominated the tick.
        t = time.perf_counter()
        names, values = pipe.feature_table()
        table = FeatureTable(list(names), values)
        tm["stats"] += time.perf_counter() - t

        t = time.perf_counter()
        n_feat = table.n_rows
        if queries and hasattr(engine, "evaluate_queries_table"):
            flags = engine.evaluate_queries_table(table.names, table.values, qjson)
        elif queries:      # extension too old to know the columnar call
            flags = engine.evaluate_queries(table.rows(), qjson)
        else:
            flags = [True] * n_feat
        ids = table.column("feature_id")
        areas = table.column("area")
        kept = set()
        for r, ok in enumerate(flags):
            if not ok:
                continue
            if min_area is not None and areas is not None and areas[r] < min_area:
                continue
            kept.add(int(ids[r]))
        tm["query"] += time.perf_counter() - t

        return {"commit": params.get("commit", 0), "labels": lab, "stats": table,
                "kept": kept, "cc": None, "n_feat": n_feat}

    def _assemble_worker(self, token, si, params):
        """Worker, tiered by params["level"] -- see the app's _needed_level().

        "slice"/"cc" touch only the visible slice; "global" does every slice and
        the cross-slice assembly. Posts the result to the UI thread."""
        try:
            import numpy as np
            from msseg import mscoupon as engine
            from . import assembly as asm_mod
            p = self.primed[si]
            pct, queries = params["pct"], params["queries"]
            pixels, conn = params["pixels"], params["connectivity"]
            level, li0 = params["level"], params["li"]
            ascending = params.get("manifold", "ascending") != "descending"
            t0 = time.perf_counter()
            # Per-stage wall clock. When a re-run feels slow the log has to say
            # which stage owns it, not just the total.
            tm = {"persist": 0.0, "labels": 0.0, "stats": 0.0, "query": 0.0, "rasters": 0.0}
            name = params.get("name", str(si))

            if level != "global":
                # --- cheap tiers: the visible slice only --------------------- #
                rec = self._slice_result(si, li0, params, engine, np, tm)
                if level == "cc":
                    t = time.perf_counter()
                    base = np.asarray(p["base"][li0], dtype=np.float32)
                    filt = np.asarray(p["filtered"][li0], dtype=np.float32)
                    tm["rasters"] += time.perf_counter() - t
                    t = time.perf_counter()
                    mask = asm_mod.selection_mask(rec["labels"], rec["kept"])
                    mask = asm_mod.apply_pixel_filters(mask, base, filt, pixels)
                    lbl, n = asm_mod.per_slice_cc(mask, conn)
                    rec["cc"] = np.where(lbl > 0, lbl - 1, -1)
                    tm["cc"] = time.perf_counter() - t
                total = time.perf_counter() - t0
                log(f"slice '{name}' [{li0}] level={level}: persistence={pct:.2f}% "
                    f"selection={len(queries)} -> {len(rec['kept'])}/{rec['n_feat']} kept "
                    f"({1e3 * total:.0f}ms)")
                log("  stages: " + "  ".join(f"{k}={1e3 * v:.0f}ms"
                                             for k, v in tm.items() if v > 0.0))
                self.work_q.put(("assembly", (token, si, {"_level": level, "_li": li0,
                                                          "_commit": params.get("commit", 0),
                                                          "_slice": rec})))
                return

            # --- global: every slice + the cross-slice 3D assembly ----------- #
            merged_labels, merged_stats, base_list, filt_list, kept_list = [], [], [], [], []
            recs, n_feat = [], 0
            for li in range(len(p["pipes"])):
                rec = self._slice_result(si, li, params, engine, np, tm)
                recs.append(rec)
                n_feat += rec["n_feat"]
                merged_labels.append(rec["labels"])
                merged_stats.append(rec["stats"])
                kept_list.append(rec["kept"])
                t = time.perf_counter()
                base_list.append(np.asarray(p["base"][li], dtype=np.float32))
                filt_list.append(np.asarray(p["filtered"][li], dtype=np.float32))
                tm["rasters"] += time.perf_counter() - t

            # The measurement channels for every slice, so the GUI's 3D assembly
            # reports the same fields the CLI's matcher does. Computed here and
            # dropped with the assembly rather than cached per primed slice.
            t = time.perf_counter()
            channels_list = None
            if hasattr(engine, "stat_channel_images"):
                try:
                    channels_list = []
                    for li in range(len(base_list)):
                        names, imgs = engine.stat_channel_images(
                            base_list[li], filt_list[li], params["json"])
                        channels_list.append(list(zip(list(names), list(imgs))))
                except Exception as exc:
                    log(f"measurement channels unavailable, falling back to base: {exc}")
                    channels_list = None
            tm["rasters"] += time.perf_counter() - t

            t_sel = time.perf_counter()
            asm_timing = {}
            out = asm_mod.assemble_cc(merged_labels, kept_list, base_list, filt_list,
                                      pixel_rules=pixels, connectivity=conn,
                                      ascending=ascending,
                                      channels_list=channels_list,
                                      reductions=params["reductions"],
                                      extremum=params["extremum"],
                                      timing=asm_timing)
            tm["assemble"] = time.perf_counter() - t_sel
            out["merged_labels"] = merged_labels
            out["merged_stats"] = merged_stats
            out["kept_list"] = kept_list
            out["_commit"] = params.get("commit", 0)
            out["_level"] = "global"
            # Hand back the per-slice records too, so navigating after a full
            # assembly is free rather than re-deriving each slice on arrival.
            out["_slices"] = recs
            for li, rec in enumerate(recs):
                rec["cc"] = out["cc_labels"][li]
            total = time.perf_counter() - t0
            log(f"assemble '{name}': persistence={pct:.2f}% "
                f"conn={conn} selection={len(queries)} pixel_rules={len(pixels)} "
                f"-> {out['n_global']} global features "
                f"({1e3 * total:.0f}ms)")
            log("  stages: " + "  ".join(f"{k}={1e3 * v:.0f}ms" for k, v in tm.items())
                + f"   [{len(p['pipes'])} slices, {n_feat} features]")
            if asm_timing:
                log("  assemble: " + "  ".join(f"{k}={1e3 * v:.0f}ms"
                                               for k, v in asm_timing.items()))
            self.work_q.put(("assembly", (token, si, out)))
        except Exception as exc:
            log(f"ASSEMBLY ERROR: {exc}")
            self.work_q.put(("assembly", (token, si, None)))
