"""Priming one place on a slide, and holding as few of them as possible.

``SlideEngine`` is to mspath what ``ComputeEngine`` is to mscoupon, with two
differences that follow from the data rather than from taste:

* **The unit of work is an item, not a sequence.** A coupon run primes every
  slice of a subsequence up front because the whole stack is the subject. A
  slide is 4 gigapixels; the subject is wherever the user is looking. So work
  is per item, requested one at a time.
* **A primed item is expensive enough to evict.** Each keeps a live
  ``Msc2DPipeline`` -- the MSCEER engine, the base labelling, the per-manifold
  statistics -- because that is what makes ``select_persistence`` a few
  milliseconds instead of a re-prime. Measured on the ARPA-H slide that is
  ~1 GB for a level-4 overview and ~0.3-1 GB for a 2048-4096-square ROI (see
  `experiments/roi_bench.py`), so the pipelines live in an LRU
  (``MSPATH_LIVE_ITEMS``, default 3) and an evicted item keeps its record but
  loses interactive persistence until it is primed again.

Two things every item must share, or its statistics are not comparable with
any other item's:

* **the persistence threshold, in the field's own units.** ``persistence_percent``
  resolves against the item's OWN value range, which differs between a coarse
  overview and a level-0 ROI, and even between two reads of the same rect with
  different padding -- measured, in the halo experiment, as a labelling that
  changed for no other reason. The engine therefore resolves a percentage ONCE,
  against the first item primed in a run (normally the overview), and pins the
  absolute for the rest.
* **the positional columns, in slide coordinates.** ``Msc2DPipeline`` knows
  nothing of an origin: every coordinate it reports is 0-based in the raster it
  was handed. The six positional columns -- and only those six, which
  ``FieldConventions.positional`` already names and excludes from every feature
  vector -- are mapped into slide pixels here, so a region's position means the
  same thing whatever item found it.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time

from msseg.labeler.pyramid import PyramidImageSource
from msseg.labeler.table import FeatureTable

from .common import log
from .items import Item, parse_key

DEFAULT_LIVE_ITEMS = 3

# The six columns that say WHERE a region is. Everything else is a measurement
# and is left alone. Kept as (name, axis) so the mapping is one loop.
POSITIONAL = (("min_x", 0), ("max_x", 0), ("ext_x", 0),
              ("min_y", 1), ("max_y", 1), ("ext_y", 1))


def live_budget():
    try:
        return max(1, int(os.environ.get("MSPATH_LIVE_ITEMS", "") or DEFAULT_LIVE_ITEMS))
    except ValueError:
        return DEFAULT_LIVE_ITEMS


class Primed:
    """One item's live compute state."""

    __slots__ = ("item", "pipe", "base", "filtered", "origin", "scale", "level",
                 "shape", "value_range", "halo", "channel_sources")

    def __init__(self, item, pipe, base, filtered, origin, scale, level, shape, halo=0):
        self.item = item
        self.halo = int(halo)
        self.channel_sources = {}         # channel -> PlacedImageSource (display cache)
        self.pipe = pipe
        self.base = base
        self.filtered = filtered
        self.origin = origin              # (x, y) in slide pixels
        self.scale = float(scale)         # slide pixels per item pixel
        self.level = int(level)
        self.shape = shape                # (h, w) of the item's raster
        self.value_range = float(pipe.value_range()) if pipe is not None else 0.0

    @property
    def live(self):
        return self.pipe is not None

    def release(self):
        pipe, self.pipe = self.pipe, None
        if pipe is not None:
            try:
                pipe.release_gpu()
            except Exception:
                pass
        del pipe


class SlideEngine:
    """Primes items off a slide pyramid; hands back records the framework's
    ``RegionProvider`` serves.

    Events on ``poll()`` mirror the coupon engine's so ``ViewerShell``'s pump
    needs no special case: ``("progress", (done, total))``, ``("primed",)``,
    ``("item_done", key)``, ``("error", exc)``.
    """

    def __init__(self):
        self.sources = {}          # slide id -> PyramidImageSource
        self.paths = {}            # slide id -> filesystem path
        self.primed = {}           # ItemKey -> Primed
        self.slices = {}           # ItemKey -> record
        self.commit_id = 1
        # level -> the REFERENCE value range the percentage resolves against,
        # taken from the first item primed at that level. What is pinned is the
        # range, not the threshold: the slider must stay live, and every item
        # at a level must read the same percentage as the same threshold.
        self.level_range = {}
        self.persistence_abs = {}        # level -> the threshold last resolved (readout)
        self.work_q = queue.Queue()
        self._order = []                 # LRU of live keys, oldest first
        self._worker = None
        self._busy = False
        self.running_keys = ()           # the keys the worker is priming right now

    # ------------------------------------------------------------------ #
    # slides
    # ------------------------------------------------------------------ #
    def register(self, slide, path):
        self.paths[str(slide)] = str(path)

    def source(self, slide):
        """The slide's pyramid, opened once and kept (a reopen per read is what
        made the old canvas path slow)."""
        slide = str(slide)
        src = self.sources.get(slide)
        if src is None:
            path = self.paths.get(slide)
            if not path:
                raise KeyError(f"no path registered for slide {slide!r}")
            src = PyramidImageSource(path)
            self.sources[slide] = src
        return src

    def close_sources(self):
        for src in self.sources.values():
            try:
                src.close()
            except Exception:
                pass
        self.sources.clear()

    def item_geometry(self, item: Item):
        """(level, x, y, w, h) in the item's own level pixels, plus the origin
        in slide pixels and the slide-pixels-per-item-pixel scale."""
        src = self.source(item.slide)
        level = max(0, min(int(item.level), src.levels - 1))
        scale = src.level_scale(level)
        if item.rect is None:
            h, w = src.level_shape(level)
            return level, 0, 0, w, h, (0, 0), scale
        lx, ly, lw, lh = item.level_rect(scale)
        return level, lx, ly, lw, lh, (item.rect[0], item.rect[1]), scale

    # ------------------------------------------------------------------ #
    # priming
    # ------------------------------------------------------------------ #
    def prime_item(self, item: Item, profile, halo=0, quiet=True):
        """Read the item, run both chains, prime it. Synchronous; the caller
        decides which thread it is on."""
        from msseg.mscoupon import mscoupon_py as ext
        from msseg.mscoupon.engine import ComputeEngine, prime
        import numpy as np

        say = (lambda _m: None) if quiet else log
        src = self.source(item.slide)
        level, lx, ly, lw, lh, origin, scale = self.item_geometry(item)
        halo = int(halo)

        t0 = time.perf_counter()
        tile = src.read_region(level, lx - halo, ly - halo, lw + 2 * halo, lh + 2 * halo)
        t_read = time.perf_counter()
        arr = (np.ascontiguousarray(np.transpose(tile, (2, 0, 1)), dtype=np.float32)
               if tile.ndim == 3 else np.ascontiguousarray(tile, dtype=np.float32))

        cur, rest = ComputeEngine._leading_color(arr, profile.get("filters") or [], ext, say,
                                                 profile.get("color_method", "luminance"))
        for f in rest:
            cur = ext.filter_slice(cur, json.dumps({"filter": f}))
        filtered = np.ascontiguousarray(cur, dtype=np.float32)
        base, _norms = ComputeEngine._apply_base_chain(
            arr, profile.get("base_filters") or [], ext, say,
            profile.get("color_method", "luminance"))
        t_filter = time.perf_counter()

        planes = arr if arr.ndim == 3 else None
        pipe = prime(ext, base, filtered, json.dumps(profile), planes)
        t_prime = time.perf_counter()

        # The halo is compute-only: everything downstream sees the item itself.
        if halo:
            base = np.ascontiguousarray(base[halo:halo + lh, halo:halo + lw])
            filtered = np.ascontiguousarray(filtered[halo:halo + lh, halo:halo + lw])
        p = Primed(item, pipe, base, filtered, origin, scale, level, (lh, lw), halo)
        log(f"primed {item.key}: {lw}x{lh} @L{level} (halo {halo}) "
            f"read={1e3 * (t_read - t0):.0f}ms filters={1e3 * (t_filter - t_read):.0f}ms "
            f"prime={1e3 * (t_prime - t_filter):.0f}ms range={p.value_range:.4g}")
        self._install(item.key, p)
        return p

    def _install(self, key, p):
        old = self.primed.get(key)
        if old is not None:
            old.release()
        self.primed[key] = p
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)
        self._evict()

    def touch(self, key):
        if key in self._order:
            self._order.remove(key)
            self._order.append(key)

    def _evict(self):
        budget = live_budget()
        live = [k for k in self._order if self.primed.get(k) is not None
                and self.primed[k].live]
        while len(live) > budget:
            key = live.pop(0)
            p = self.primed.get(key)
            if p is not None:
                log(f"releasing pipeline for {key} (over the {budget}-item budget)")
                p.release()

    # ------------------------------------------------------------------ #
    # selection -> records
    # ------------------------------------------------------------------ #
    def resolve_persistence(self, p: Primed, profile):
        """The threshold in field units, pinned PER LEVEL for a run.

        A percentage resolves against the item's own value range, so the same
        number is a different threshold on every item -- the drift the halo
        experiment tripped over. Resolving it once and holding it fixes that
        for items that are comparable, but only those: the topology field at
        level 4 is a different field from the one at level 0 (a sigma is in
        pixels, and a pixel is 16x bigger), so a threshold pinned on an
        overview collapses a level-0 ROI to nothing and vice versa. Hence one
        pin per level, taken from the first item primed at that level, and
        shared across slides -- which is deliberate: one threshold per level
        for the whole study is what makes two slides' statistics comparable.

        An explicit ``persistence_absolute`` in the profile wins outright.
        """
        msc = profile.get("msc") or {}
        absolute = msc.get("persistence_absolute")
        if absolute is not None:
            return float(absolute)
        level = int(p.level)
        ref = self.level_range.get(level)
        if ref is None:
            ref = self.level_range[level] = float(p.value_range)
            log(f"level {level}: persistence % now resolves against "
                f"{p.item.key}'s range {ref:.6g} -- every item at this level shares it")
        # Derived from the CURRENT percentage every time. The first version of
        # this cached the threshold itself, which pinned the level correctly
        # and also made the persistence slider do nothing at all.
        pct = float(msc.get("persistence_percent", 10.0) or 0.0)
        hit = ref * pct / 100.0
        self.persistence_abs[level] = hit
        return hit

    def record(self, key):
        """The item's record at the current commit, or None. Never computes."""
        rec = self.slices.get(key)
        return None if rec is None or rec.get("commit") != self.commit_id else rec

    def ensure_record(self, key, profile):
        """The item's record, computing it synchronously if needed. None when
        the item is not primed (a released pipeline must be re-primed first)."""
        import numpy as np
        rec = self.record(key)
        if rec is not None:
            return rec
        p = self.primed.get(key)
        if p is None or not p.live:
            return None
        self.touch(key)
        t0 = time.perf_counter()
        p.pipe.select_persistence(self.resolve_persistence(p, profile))
        labels = p.pipe.labels()
        names, values = p.pipe.feature_table()

        # The pipeline saw the PADDED raster, so its coordinates are relative to
        # the halo's top-left, not the item's.
        pad_origin = (p.origin[0] - p.halo * p.scale, p.origin[1] - p.halo * p.scale)
        values = self._to_slide_coords(values, list(names), pad_origin, p.scale, np)

        if p.halo:
            h, (lh, lw) = p.halo, p.shape
            labels = np.ascontiguousarray(labels[h:h + lh, h:h + lw])
            # Rows for regions that live entirely in the halo describe tissue
            # outside the item: nothing can select or annotate them, and they
            # would train a model on pixels the user never saw. Drop them.
            # A region STRADDLING the edge is kept, and its statistics still
            # cover its halo part -- which is the point of the halo: a boundary
            # region measured only inside the cut is measured wrong.
            keep_ids = np.unique(labels)
            ids = values[:, list(names).index("feature_id")].astype(np.int64)
            values = values[np.isin(ids, keep_ids)]
        table = FeatureTable(list(names), values)
        try:
            a, b, saddle = p.pipe.region_arcs()
            arcs = ({"a": a, "b": b, "saddle": saddle, "source": "msc"}
                    if a is not None and len(a) else None)
        except Exception:
            arcs = None
        rec = {"commit": self.commit_id, "labels": labels, "stats": table, "arcs": arcs,
               "kept": None, "cc": None, "origin": p.origin, "scale": p.scale,
               "level": p.level, "n_ids": int(labels.max()) + 1 if labels.size else 1}
        self.slices[key] = rec
        log(f"{key}: {table.n_rows} regions at persistence "
            f"{self.resolve_persistence(p, profile):.6g} "
            f"({1e3 * (time.perf_counter() - t0):.0f}ms)")
        return rec

    @staticmethod
    def _to_slide_coords(values, names, origin, scale, np):
        """Map the six positional columns from item pixels into slide pixels.

        Only these six move. Everything else -- ``area``, ``bbox_w/h`` and every
        measurement -- stays in the item's own level pixels, which is why a
        model is valid at one level only: a region 40 px across at level 4 is
        640 slide pixels, and calling both "40" would be a lie in the other
        direction.
        """
        out = np.array(values, dtype=np.float64, copy=True)
        idx = {n: i for i, n in enumerate(names)}
        for name, axis in POSITIONAL:
            i = idx.get(name)
            if i is not None:
                out[:, i] = out[:, i] * float(scale) + float(origin[axis])
        return out

    def label_layer(self, key):
        from .sources import RoiLabelLayer
        rec = self.record(key)
        if rec is None or rec.get("labels") is None:
            return None
        item = parse_key(key)
        slide_shape = None
        try:
            slide_shape = self.source(item.slide).level_shape(0) if item else None
        except Exception:
            pass
        return RoiLabelLayer(rec["labels"], origin=rec["origin"], scale=rec["scale"],
                             slide_shape=slide_shape, rev=int(rec["commit"]),
                             n_ids=int(rec["n_ids"]))

    # ------------------------------------------------------------------ #
    # generations
    # ------------------------------------------------------------------ #
    def commit_selection(self):
        """A new parameter generation: records fall stale by commit, and the
        stale ones are dropped so a Rerun does not stack a second copy."""
        self.commit_id += 1
        self.slices = {k: v for k, v in self.slices.items()
                       if v.get("commit") == self.commit_id}

    def reset(self):
        """Drop everything primed (the parameters that produced it are gone)."""
        for p in self.primed.values():
            p.release()
        self.primed.clear()
        self.slices.clear()
        self._order = []
        self.level_range = {}
        self.persistence_abs = {}
        self.commit_id += 1

    # ------------------------------------------------------------------ #
    # off-thread work
    # ------------------------------------------------------------------ #
    def pending_work(self):
        return self._busy

    def poll(self):
        """Drain the worker's events. The engine's own bookkeeping happens
        here, on the UI thread, so the shell only has to display."""
        out = []
        while True:
            try:
                ev = self.work_q.get_nowait()
            except queue.Empty:
                break
            if ev[0] in ("done", "error"):
                self._busy = False
                self.running_keys = ()
                if ev[0] == "error":
                    out.append(ev)
                    continue
                out.append(("primed",))
                continue
            out.append(ev)
        return out

    def start_run(self, items, profile, halo=0, reset_pins=True):
        """Prime `items` in order on one worker thread. Serial on purpose: the
        pipelines are stateful and each one is most of a gigabyte.

        `reset_pins` False keeps the per-level thresholds already resolved --
        which is what priming ONE more ROI into an existing session must do.
        Re-resolving them would silently re-threshold every item primed before
        it, against whatever range this one happened to have.
        """
        if self._busy:
            return False
        self._busy = True
        if reset_pins:
            self.level_range = {}
            self.persistence_abs = {}
        self.running_keys = tuple(it.key for it in items)
        self._worker = threading.Thread(target=self._run_worker, name="mspath-prime",
                                        args=(list(items), dict(profile), int(halo)),
                                        daemon=True)
        self._worker.start()
        return True

    def _run_worker(self, items, profile, halo):
        total = len(items)
        failed = []
        for n, item in enumerate(items, start=1):
            try:
                # The overview is a whole level: there is nothing beyond its
                # edges to borrow, so it takes no halo whatever the profile says.
                self.prime_item(item, profile, halo=0 if item.is_overview else halo)
                self.work_q.put(("item_done", item.key))
            except Exception as exc:
                # One unreadable slide must not take the rest of the run with
                # it: report it, and go on. It is still an error the user
                # sees -- as a status line and a log entry, not a modal that
                # also throws away the items that did prime.
                msg = f"{type(exc).__name__}: {exc}"
                log(f"{item.key}: prime FAILED -- {msg}")
                failed.append((item.key, msg))
                self.work_q.put(("item_error", item.key, msg))
            self.work_q.put(("progress", (n, total)))
        if failed and len(failed) == total:
            self.work_q.put(("error", RuntimeError(
                "no item could be primed:\n" + "\n".join(f"{k}: {m}" for k, m in failed))))
        else:
            self.work_q.put(("done",))
