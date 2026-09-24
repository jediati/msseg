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
from msseg.labeler.record_keys import Interner, RecordCache
from msseg.labeler.table import FeatureTable

from msseg.mscoupon.fingerprints import field_fingerprint_of, measure_fingerprint_of

from .common import log
from .items import Item, parse_key

DEFAULT_LIVE_ITEMS = 3

# The six columns that say WHERE a region is. Everything else is a measurement
# and is left alone. Kept as (name, axis) so the mapping is one loop.
POSITIONAL = (("min_x", 0), ("max_x", 0), ("ext_x", 0),
              ("min_y", 1), ("max_y", 1), ("ext_y", 1))


def default_color_method(profile):
    """The colour->scalar method a chain without a leading `color` stage gets.

    It lives at ``input.color.default_method`` -- the one place the coupon
    engine, ``profile_params_json`` and the C++ config parser all read it from.
    mspath used to spell this ``profile.get("color_method")``, a key nothing has
    ever written, so the declared method was silently ignored and every such
    chain converted by luminance. Nothing noticed because mspath always puts an
    explicit `color` card at index 0; delete that card and the bug surfaced.
    """
    from msseg.mscoupon.session import color_input_from_json
    return color_input_from_json((profile or {}).get("input"))["default_method"]


def live_budget():
    try:
        return max(1, int(os.environ.get("MSPATH_LIVE_ITEMS", "") or DEFAULT_LIVE_ITEMS))
    except ValueError:
        return DEFAULT_LIVE_ITEMS


class Primed:
    """One item's live compute state."""

    __slots__ = ("item", "pipe", "base", "filtered", "origin", "scale", "level",
                 "shape", "value_range", "halo", "channel_sources", "field", "measured", "chains",
                 "pipe_id", "slot")

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
        # What the pipe was built and measured under (fingerprints.py): the
        # field decides whether the pipe can be kept at all, the measurement
        # whether its rows are current or need a re-measure.
        self.field = None
        self.measured = None
        # The chains the pipe was primed with (filters, base_filters, input):
        # a re-measure must read the rasters the MSC saw, not whatever the
        # panel says now -- an edited, un-Run chain is a preview, not a prime.
        self.chains = None
        # Which prime this is (record ids name it: region ids are not stable
        # from one prime to the next) and which field slot holds it.
        self.pipe_id = 0
        self.slot = None

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
        # Field slots: field fingerprint -> {"primed": {ItemKey: Primed},
        # "level_range": {level: ref}, "persistence_abs": {level: abs}}. The
        # ACTIVE slot is what `primed` / `level_range` / `persistence_abs`
        # read and write, so every caller sees one field at a time while a
        # task on another field keeps its pipes and its pins.
        #   level_range: level -> the REFERENCE value range the percentage
        #   resolves against, taken from the first item primed at that level
        #   in that field. What is pinned is the range, not the threshold: the
        #   slider must stay live, and every item at a level must read the
        #   same percentage as the same threshold.
        self._slots = {}
        self._active = None
        # Records by identity (record_keys.py): a few per item, each under the
        # interned id of (item, pipe, measurement, persistence).
        self.records = RecordCache()
        self._ids = Interner(start=2)
        self._pipe_seq = 0
        self._params = None              # {"meas": fp, "msc": {...}} -- see set_params
        self.commit_id = 1               # a generation, for RegionProvider.commit only
        self.work_q = queue.Queue()
        self._order = []                 # LRU of live (slot, key), oldest first, across fields
        self._worker = None
        self._busy = False
        self.running_keys = ()           # the keys the worker is priming right now
        # "prime" or "measure": what the worker is doing to running_keys, for
        # the badge. A run that primes anything says "prime".
        self.running_kind = "prime"
        self._remeasure_only = False

    # ------------------------------------------------------------------ #
    # field slots
    # ------------------------------------------------------------------ #
    def _slot_dict(self, slot=None):
        s = self._slots.get(slot)
        if s is None:
            s = self._slots[slot] = {"primed": {}, "level_range": {}, "persistence_abs": {}}
        return s

    @property
    def primed(self):
        """The ACTIVE field's items -> Primed."""
        return self._slot_dict(self._active)["primed"]

    @primed.setter
    def primed(self, value):
        self._slot_dict(self._active)["primed"] = value

    @property
    def level_range(self):
        return self._slot_dict(self._active)["level_range"]

    @level_range.setter
    def level_range(self, value):
        self._slot_dict(self._active)["level_range"] = value

    @property
    def persistence_abs(self):
        return self._slot_dict(self._active)["persistence_abs"]

    @persistence_abs.setter
    def persistence_abs(self, value):
        self._slot_dict(self._active)["persistence_abs"] = value

    @property
    def slices(self):
        """The records (kept under this name for the callers that ask
        whether anything has been computed)."""
        return self.records

    @property
    def active_field(self):
        return self._active

    def slot_of(self, profile):
        return field_fingerprint_of(profile)

    def use_field(self, profile, halo=0):
        """Make `profile`'s field the active slot -- dropping nothing in any
        other slot -- and, in it, release the pipes the profile cannot serve
        (another field primed on demand under an edited chain, or an ROI
        read with another halo). Returns how many live items it keeps. A
        switch to a field with nothing primed simply shows nothing primed:
        its items are primed when a Run (or a navigation to an ROI) asks."""
        field = field_fingerprint_of(profile)
        self._adopt(field)
        self._active = field
        slot = self._slot_dict(field)
        kept = 0
        for key in list(slot["primed"]):
            p = slot["primed"][key]
            ok = ((p.field is None or p.field == field)
                  and (p.item.is_overview or p.halo == int(halo)))
            if ok:
                kept += int(p.live)
                continue
            p.release()
            del slot["primed"][key]
            self._order = [e for e in self._order if e != (field, key)]
        self.commit_id += 1
        return kept

    def keep_field(self, profile, halo=0):
        """Old name of ``use_field`` (it used to drop every other field)."""
        return self.use_field(profile, halo)

    def _adopt(self, field):
        """Before any field is named, everything lives in the unnamed slot
        (pins set by a caller, items installed by hand). The first field
        named takes it over, so nothing primed or pinned so far is lost."""
        if self._active is not None or None not in self._slots:
            return
        loose = self._slots.pop(None)
        mine = self._slots.get(field)
        if mine is None:
            self._slots[field] = loose
        else:
            for k in ("primed", "level_range", "persistence_abs"):
                for kk, vv in loose[k].items():
                    mine[k].setdefault(kk, vv)
        for p in self._slots[field]["primed"].values():
            if p.slot is None:
                p.slot = field
        self._order = [(field if s is None else s, k) for s, k in self._order]

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
    def read_item(self, item: Item, halo=0):
        """The pixels a prime will run its chains over: the item's rect at its
        own level, padded by the halo, as a (C,H,W) or (H,W) float32 array,
        plus its geometry.

        Split out of prime_item so a PREVIEW of an edited chain computes on
        exactly the array a prime would -- and so it can be cached per item
        and re-used across edits, since the read is seconds at a deep level."""
        import numpy as np

        src = self.source(item.slide)
        geom = self.item_geometry(item)
        level, lx, ly, lw, lh = geom[:5]
        halo = int(halo)
        t0 = time.perf_counter()
        tile = src.read_region(level, lx - halo, ly - halo, lw + 2 * halo, lh + 2 * halo)
        arr = (np.ascontiguousarray(np.transpose(tile, (2, 0, 1)), dtype=np.float32)
               if tile.ndim == 3 else np.ascontiguousarray(tile, dtype=np.float32))
        return arr, geom, time.perf_counter() - t0

    def prime_item(self, item: Item, profile, halo=0, quiet=True, slot=False):
        """Read the item, run both chains, prime it. Synchronous; the caller
        decides which thread it is on. It lands in the ACTIVE field slot
        unless `slot` names another (``Run all tasks`` primes a task on
        another workflow into that workflow's field)."""
        from msseg.mscoupon import mscoupon_py as ext
        from msseg.mscoupon.engine import ComputeEngine, build_timings_brief, prime
        import numpy as np

        say = (lambda _m: None) if quiet else log
        halo = int(halo)

        t0 = time.perf_counter()
        arr, geom, _dt = self.read_item(item, halo)
        level, lx, ly, lw, lh, origin, scale = geom
        t_read = time.perf_counter()

        method = default_color_method(profile)
        cur, rest = ComputeEngine._leading_color(arr, profile.get("filters") or [], ext, say,
                                                 method)
        # One call for the remainder: core plans what it is handed, and a stage
        # passed alone is a chain of one, which can plan differently from the
        # same stage inside its chain.
        if rest:
            cur = ext.filter_chain(cur, json.dumps({"filters": rest}), method)
        filtered = np.ascontiguousarray(cur, dtype=np.float32)
        base, _norms = ComputeEngine._apply_base_chain(
            arr, profile.get("base_filters") or [], ext, say, method)
        t_filter = time.perf_counter()

        planes = arr if arr.ndim == 3 else None
        pipe = prime(ext, base, filtered, json.dumps(profile), planes)
        t_prime = time.perf_counter()

        # The halo is compute-only: everything downstream sees the item itself.
        if halo:
            base = np.ascontiguousarray(base[halo:halo + lh, halo:halo + lw])
            filtered = np.ascontiguousarray(filtered[halo:halo + lh, halo:halo + lw])
        p = Primed(item, pipe, base, filtered, origin, scale, level, (lh, lw), halo)
        p.field = field_fingerprint_of(profile)
        p.measured = measure_fingerprint_of(profile)
        p.chains = {k: profile.get(k) for k in ("filters", "base_filters", "input")}
        self._pipe_seq += 1
        p.pipe_id = self._pipe_seq
        if slot is False and self._active is None:
            # The first prime names the active field.
            self._adopt(p.field)
            self._active = p.field
        p.slot = self._active if slot is False else slot
        log(f"primed {item.key}: {lw}x{lh} @L{level} (halo {halo}) "
            f"read={1e3 * (t_read - t0):.0f}ms filters={1e3 * (t_filter - t_read):.0f}ms "
            f"prime={1e3 * (t_prime - t_filter):.0f}ms{build_timings_brief(pipe)} "
            f"range={p.value_range:.4g}")
        self._install(item.key, p)
        return p

    # ------------------------------------------------------------------ #
    # the measurement, separately from the field
    # ------------------------------------------------------------------ #
    def _get(self, key, slot=False):
        return (self.primed if slot is False else self._slot_dict(slot)["primed"]).get(key)

    def can_remeasure(self, key, slot=False):
        p = self._get(key, slot)
        return p is not None and p.live and hasattr(p.pipe, "remeasure")

    def measure_stale(self, key, profile, slot=False):
        """True iff the item is live and its rows were measured under other
        statistics than `profile` asks for."""
        p = self._get(key, slot)
        if p is None or not p.live or p.measured is None:
            return False
        return p.measured != measure_fingerprint_of(profile)

    def needs_prime(self, key, profile, halo=None, slot=False):
        """True iff the item has no pipe that can serve `profile`: not primed,
        released, built under another field (or halo), or measured under other
        statistics on an extension that cannot re-measure."""
        p = self._get(key, slot)
        if p is None or not p.live:
            return True
        if p.field is not None and p.field != field_fingerprint_of(profile):
            return True
        if halo is not None and not p.item.is_overview and p.halo != int(halo):
            return True
        return self.measure_stale(key, profile, slot) and not hasattr(p.pipe, "remeasure")

    def remeasure_item(self, key, profile, quiet=True, slot=False):
        """Rebuild the item's statistics under `profile` on its live pipe: the
        MSC, labels and arcs stay. The rasters are re-read and re-filtered
        (the pipe saw the padded raster and the colour planes, neither of
        which is kept) -- the read is most of the cost, and still well under
        a prime. Synchronous; the caller picks the thread."""
        from msseg.mscoupon import mscoupon_py as ext
        from msseg.mscoupon.engine import ComputeEngine, build_timings_brief
        import numpy as np

        p = self._get(key, slot)
        if p is None or not p.live:
            raise RuntimeError(f"{key}: no live pipeline to re-measure")
        say = (lambda _m: None) if quiet else log
        chains = p.chains or profile
        t0 = time.perf_counter()
        arr, _geom, _dt = self.read_item(p.item, p.halo)
        method = default_color_method(chains)
        cur, rest = ComputeEngine._leading_color(arr, chains.get("filters") or [], ext, say,
                                                 method)
        if rest:
            cur = ext.filter_chain(cur, json.dumps({"filters": rest}), method)
        filtered = np.ascontiguousarray(cur, dtype=np.float32)
        # The base chain is a measurement: the NEW profile's, not the prime's.
        base, _norms = ComputeEngine._apply_base_chain(
            arr, profile.get("base_filters") or [], ext, say, method)
        t_read = time.perf_counter()
        planes = arr if arr.ndim == 3 else None
        if planes is None:
            p.pipe.remeasure(json.dumps(profile), base, filtered)
        else:
            p.pipe.remeasure(json.dumps(profile), base, filtered, planes)
        p.measured = measure_fingerprint_of(profile)
        p.channel_sources = {}
        if p.halo:
            h, (lh, lw) = p.halo, p.shape
            p.base = np.ascontiguousarray(base[h:h + lh, h:h + lw])
        else:
            p.base = base
        self.touch(key, p.slot)
        log(f"re-measured {key}: read+filters={1e3 * (t_read - t0):.0f}ms "
            f"measure={1e3 * (time.perf_counter() - t_read):.0f}ms{build_timings_brief(p.pipe)}")
        return p

    def _install(self, key, p):
        slot = self._slot_dict(p.slot)["primed"]
        old = slot.get(key)
        if old is not None:
            old.release()
        slot[key] = p
        entry = (p.slot, key)
        if entry in self._order:
            self._order.remove(entry)
        self._order.append(entry)
        self._evict()

    def touch(self, key, slot=False):
        entry = (self._active if slot is False else slot, key)
        if entry in self._order:
            self._order.remove(entry)
            self._order.append(entry)

    def _evict(self):
        """One live budget across every field: a pipe is a pipe, whichever
        task's workflow primed it. A released item keeps its Primed (and so
        its records); only interactive persistence needs a re-prime."""
        budget = live_budget()

        def live(entry):
            p = self._slot_dict(entry[0])["primed"].get(entry[1])
            return p is not None and p.live
        self._order = [e for e in self._order if live(e)]
        while len(self._order) > budget:
            slot, key = self._order.pop(0)
            p = self._slot_dict(slot)["primed"].get(key)
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
        pins = self._slot_dict(p.slot)
        level = int(p.level)
        ref = pins["level_range"].get(level)
        if ref is None:
            ref = pins["level_range"][level] = float(p.value_range)
            log(f"level {level}: persistence % now resolves against "
                f"{p.item.key}'s range {ref:.6g} -- every item at this level shares it")
        # Derived from the CURRENT percentage every time. The first version of
        # this cached the threshold itself, which pinned the level correctly
        # and also made the persistence slider do nothing at all.
        pct = float(msc.get("persistence_percent", 10.0) or 0.0)
        hit = ref * pct / 100.0
        pins["persistence_abs"][level] = hit
        return hit

    def _peek_persistence(self, p, msc):
        """``resolve_persistence`` without writing a pin (a lookup must not
        pin a level on an item nobody computed)."""
        absolute = msc.get("persistence_absolute")
        if absolute is not None:
            return float(absolute)
        ref = self._slot_dict(p.slot)["level_range"].get(int(p.level), float(p.value_range))
        return float(ref) * float(msc.get("persistence_percent", 10.0) or 0.0) / 100.0

    # ------------------------------------------------------------------ #
    # records by identity
    # ------------------------------------------------------------------ #
    def set_params(self, profile):
        """The parameters `record()` answers for: the measurement the rows
        must be under and the persistence the labels are at. Cheap; the
        callers that change either (a threshold, a statistics edit, a switch,
        a Run) call it, and ``ensure_record`` does too."""
        self._params = {"meas": measure_fingerprint_of(profile),
                        "msc": dict((profile or {}).get("msc") or {})}

    @property
    def has_params(self):
        return self._params is not None

    @property
    def current_measure(self):
        return self._params["meas"] if self._params else None

    def record_id(self, key, p, persistence):
        """The interned id of (item, prime, measurement, persistence)."""
        meas = self._params["meas"] if self._params else None
        return self._ids.id_of((key, int(p.pipe_id), meas, float(persistence)))

    def record(self, key):
        """The item's record for the current parameters, or None. Never
        computes. A released pipe still answers: its records outlive it."""
        if self._params is None:
            return None
        p = self.primed.get(key)
        if p is None:
            return None
        rid = self.record_id(key, p, self._peek_persistence(p, self._params["msc"]))
        return self.records.get(key, rid)

    def ensure_record(self, key, profile):
        """The item's record, computing it synchronously if needed. None when
        the item is not primed (a released pipeline must be re-primed first)."""
        import numpy as np
        self.set_params(profile)
        p = self.primed.get(key)
        if p is None:
            return None
        pers = self.resolve_persistence(p, profile)
        rid = self.record_id(key, p, pers)
        rec = self.records.get(key, rid)
        if rec is not None:
            return rec
        if not p.live:
            return None
        if self.measure_stale(key, profile):
            if not hasattr(p.pipe, "remeasure"):
                return None            # the caller re-primes under the new spec
            self.remeasure_item(key, profile)
        self.touch(key, p.slot)
        t0 = time.perf_counter()
        p.pipe.select_persistence(pers)
        shared = self.records.shared(key, (p.pipe_id, pers))
        labels = None if shared is not None else p.pipe.labels()
        names, values = p.pipe.feature_table()

        # The pipeline saw the PADDED raster, so its coordinates are relative to
        # the halo's top-left, not the item's.
        pad_origin = (p.origin[0] - p.halo * p.scale, p.origin[1] - p.halo * p.scale)
        values = self._to_slide_coords(values, list(names), pad_origin, p.scale, np)

        if shared is not None:
            labels = shared[0]
        elif p.halo:
            h, (lh, lw) = p.halo, p.shape
            labels = np.ascontiguousarray(labels[h:h + lh, h:h + lw])
        if p.halo:
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
        if shared is not None:
            arcs = shared[1]
        else:
            try:
                a, b, saddle = p.pipe.region_arcs()
                arcs = ({"a": a, "b": b, "saddle": saddle, "source": "msc"}
                        if a is not None and len(a) else None)
            except Exception:
                arcs = None
            # One decomposition, whatever it is measured by: a second
            # statistics spec reuses these rather than holding its own copy.
            self.records.share(key, (p.pipe_id, pers), (labels, arcs))
        rec = {"commit": rid, "labels": labels, "stats": table, "arcs": arcs,
               "kept": None, "cc": None, "origin": p.origin, "scale": p.scale,
               "level": p.level, "n_ids": int(labels.max()) + 1 if labels.size else 1}
        self.records.put(key, rid, rec)
        log(f"{key}: {table.n_rows} regions at persistence {pers:.6g} "
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
    # forgetting
    # ------------------------------------------------------------------ #
    def forget(self, key):
        """Drop one item's primed state and record (it left the session).
        Not while the worker runs: it may be priming that very key."""
        if self._busy:
            raise RuntimeError("cannot forget an item while a prime is running")
        found = False
        for slot, s in self._slots.items():
            p = s["primed"].pop(key, None)
            if p is not None:
                p.release()
                found = True
        self._order = [e for e in self._order if e[1] != key]
        self.records.drop(key)
        return found

    def forget_slide(self, slide):
        """Drop every item of a slide, then the slide itself: its open
        pyramid is closed and its path forgotten. Returns the item count."""
        slide = str(slide)
        every = set(self.records.items())
        for s in self._slots.values():
            every |= set(s["primed"])
        keys = [k for k in every if (parse_key(k) or Item("", 0)).slide == slide]
        for k in keys:
            self.forget(k)
        src = self.sources.pop(slide, None)
        if src is not None:
            try:
                src.close()
            except Exception:
                pass
        self.paths.pop(slide, None)
        return len(keys)

    # ------------------------------------------------------------------ #
    # generations
    # ------------------------------------------------------------------ #
    def commit_selection(self, profile=None):
        """A new parameter generation. Nothing is dropped any more: a record
        is found by the id of what it is a function of, so the parameters
        (``set_params``, done here when `profile` is given) decide which
        record is current, and going back to earlier ones finds theirs."""
        self.commit_id += 1
        if profile is not None:
            self.set_params(profile)

    def reset(self):
        """Drop everything primed, in every field (the session is being
        replaced, or the level changed so every key is different)."""
        for s in self._slots.values():
            for p in s["primed"].values():
                p.release()
        self._slots = {}
        self.records.clear()
        self._order = []
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
                out.append(("item_primed",) if getattr(self, "_incremental", False)
                           else ("primed",))
                continue
            out.append(ev)
        return out

    def start_run(self, items, profile, halo=0, reset_pins=True, incremental=False,
                  remeasure_only=False):
        """Prime `items` in order on one worker thread. Serial on purpose: the
        pipelines are stateful and each one is most of a gigabyte.

        `reset_pins` False keeps the per-level thresholds already resolved --
        which is what priming ONE more ROI into an existing session must do.
        Re-resolving them would silently re-threshold every item primed before
        it, against whatever range this one happened to have.

        `incremental` says the same thing to the UI: the run ends with
        ("item_primed",) rather than ("primed",). The labeler drops every
        prediction and class LUT on "primed", which is right when a Run has
        recomputed everything and wrong when one ROI was added to a session
        whose other items are exactly as they were.

        A live pipe the profile's field can serve is never primed again: it is
        re-measured when the statistics moved and otherwise left alone.
        `remeasure_only` goes further and never primes at all -- a live pipe
        is re-measured whatever its field, a dead one skipped -- which is what
        navigation wants while the chains are being edited (a preview is not
        a Run).

        An entry of `items` may also be an ``(item, profile)`` pair: that item
        is primed under ITS profile, into that profile's field slot, with that
        field's pins -- ``Run all tasks`` over tasks on other workflows. Only
        the active field's pins are reset by `reset_pins`.
        """
        if self._busy:
            return False
        self._busy = True
        if reset_pins:
            self.level_range = {}
            self.persistence_abs = {}
        jobs = self._jobs(items, profile)
        self.running_keys = tuple(j[0].key for j in jobs)
        self._incremental = bool(incremental)
        self._remeasure_only = bool(remeasure_only)
        self.running_kind = ("measure" if remeasure_only or (jobs and all(
            not self.needs_prime(it.key, prof, halo, slot) for it, prof, slot in jobs))
            else "prime")
        self._worker = threading.Thread(target=self._run_worker, name="mspath-prime",
                                        args=(jobs, dict(profile), int(halo)),
                                        daemon=True)
        self._worker.start()
        return True

    def _jobs(self, items, profile):
        """(item, profile, slot) per entry; `slot` False = the active field."""
        out = []
        for entry in items:
            if isinstance(entry, Item):             # (Item is a namedtuple itself)
                out.append((entry, profile, False))
            elif len(entry) == 3:
                out.append(entry)                   # already a job
            elif len(entry) == 2:
                item, prof = entry
                slot = field_fingerprint_of(prof)
                out.append((item, dict(prof), False if slot == self._active else slot))
        return out

    def _run_worker(self, items, profile, halo):
        jobs = self._jobs(items, profile)
        total = len(jobs)
        failed = []
        for n, (item, prof, slot) in enumerate(jobs, start=1):
            kw = {} if slot is False else {"slot": slot}
            try:
                # A live pipe built under this field is kept: its rows are
                # re-measured if the statistics moved, else it is left alone.
                # Only what cannot be served is primed.
                if ((self._remeasure_only and slot is False)
                        or not self.needs_prime(item.key, prof, halo, **kw)):
                    if (self.measure_stale(item.key, prof, **kw)
                            and self.can_remeasure(item.key, **kw)):
                        self.remeasure_item(item.key, prof, **kw)
                    self.work_q.put(("item_done", item.key))
                    self.work_q.put(("progress", (n, total)))
                    continue
                # The overview is a whole level: there is nothing beyond its
                # edges to borrow, so it takes no halo whatever the profile says.
                self.prime_item(item, prof, halo=0 if item.is_overview else halo, **kw)
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
