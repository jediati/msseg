"""mspath's implementations of the labeler framework's seams.

``SlideCatalogue`` is the ``ItemCatalogue`` over slides and the places on them:
an item is an overview or an ROI, its key the self-describing string
``annotations.json`` stores (see ``items.py``), its address the ``(si, li)``
pair the shell navigates by -- slide index, item index within the slide.
``group_of`` returns the SLIDE, so cross-validation leaves whole slides out;
holding out one ROI while training on its neighbour would score a model on
tissue it has already seen.

``SlideRegionProvider`` is the ``RegionProvider`` over ``SlideEngine``: cached
per-item records, priming on request, and a ``LabelLayer`` that places the
item's ids in slide coordinates.
"""
from __future__ import annotations

import time

from msseg.labeler import magic_fill

from .common import log
from .items import parse_key


class SlideCatalogue:
    def __init__(self, app):
        self.app = app
        self._keys = []
        self._index = {}
        self._pos = {}

    def refresh(self):
        """Re-derive the key list from the app's flat item order (call after
        `_rebuild_flat_slices`)."""
        self._keys, self._index, self._pos = [], {}, {}
        for pos, (si, li) in enumerate(self.app.flat_slices):
            key = self.key_of(si, li)
            if key is not None and key not in self._index:
                self._keys.append(key)
                self._index[key] = (si, li)
                self._pos[key] = pos

    def keys(self):
        return list(self._keys)

    def item_of(self, si, li):
        """The ``Item`` at an address, or None. The app owns the mapping: a
        sequence is a slide, and what of it is annotatable depends on the
        profile (the overview's level) as much as on the session."""
        return self.app._item_at(si, li)

    def key_of(self, si, li):
        item = self.item_of(si, li)
        return None if item is None else item.key

    def index_of(self, key):
        hit = self._index.get(key)
        if hit is not None or key is None:
            return hit
        for si, li in self.app._enumerate_items():        # not (yet) in the flat order
            item = self.item_of(si, li)
            if item is not None and item.key == key:
                return (si, li)
        # A slide key (what a gesture is bound to) resolves to the slide's
        # first row -- its overview -- so `bound` means "the slide is here".
        if parse_key(key) is None:
            for si in range(len(self.app.subsequences)):
                sid, _path = self.app._slide_of(si)
                if sid == key:
                    return (si, 0)
        return None

    def label(self, key):
        item = parse_key(key)
        return item.label() if item is not None else str(key)

    def group_of(self, key):
        """Leave-SLIDES-out grouping. Two ROIs on one slide share tissue,
        staining and scanner, so holding one out while training on the other
        measures memorisation, not generalisation."""
        item = parse_key(key)
        return item.slide if item is not None else key

    def binding_of(self, key):
        """Gestures are keyed by the SLIDE (``items.slide_id``), and an item
        sees those meeting its place: an ROI's rect in slide pixels, the
        whole slide for the overview. A bare slide key binds to itself."""
        item = parse_key(key)
        if item is None:
            return key, None
        return item.slide, item.rect

    def rebase(self, key):
        """An item key from a store written before gestures were slide-bound
        -> the slide, and the item's level and slide-px-per-raster-px as the
        gesture's scale of intent. A slide key is not rebased."""
        item = parse_key(key)
        if item is None:
            return None
        scale = None
        try:
            src = self.app.engine.source(item.slide)
            scale = float(src.level_scale(item.level)) if src is not None else None
        except Exception:
            scale = None
        if scale is None:
            scale = float(2 ** int(item.level))
        return item.slide, int(item.level), scale

    def tree(self):
        out = []
        for si, s in enumerate(self.app.subsequences):
            kids = []
            for li in range(len(s.get("files") or [])):
                item = self.item_of(si, li)
                if item is not None:
                    kids.append({"key": item.key, "label": _short(item), "children": []})
            out.append({"key": None, "label": str(s.get("name") or f"slide {si}"),
                        "children": kids})
        return out


def _short(item):
    if item.rect is None:
        return f"overview L{item.level}"
    x, y, w, h = item.rect
    return f"L{item.level} {w}x{h} @({x},{y})"


class SlideRegionProvider:
    def __init__(self, app):
        self.app = app

    @property
    def engine(self):
        return self.app.engine

    @property
    def catalogue(self):
        return self.app.catalogue

    @property
    def commit(self):
        return self.engine.commit_id

    def keys(self):
        return self.catalogue.keys()

    def record(self, key):
        return self.engine.record(key)

    def ensure_record(self, key):
        """The item's record, computing it synchronously when it is primed but
        not yet selected. An item whose pipeline the LRU released is primed
        again first -- a stack-wide operation must not silently skip it, which
        is the failure mode ``ClassifierMixin`` exists to avoid.

        Caller must ensure no prime worker is running: the pipelines are
        stateful, exactly as in the coupon engine.
        """
        rec = self.engine.record(key)
        if rec is not None:
            return rec
        item = parse_key(key)
        if item is None:
            return None
        profile = self.app._profile_for_compute()
        p = self.engine.primed.get(key)
        if p is None or not p.live:
            try:
                self.engine.prime_item(item, profile, halo=self.app._halo())
            except Exception as exc:
                log(f"{key}: prime failed: {type(exc).__name__}: {exc}")
                return None
        try:
            return self.engine.ensure_record(key, profile)
        except Exception as exc:
            log(f"{key}: record failed: {type(exc).__name__}: {exc}")
            return None

    def request(self, key):
        self.app._request_item(key)

    def pending(self):
        return self.engine.pending_work()

    def poll(self):
        return self.engine.poll()

    def arcs(self, key, np):
        rec = self.record(key)
        return None if rec is None else self.arcs_for_record(rec, np, key)

    def label_layer(self, key):
        return self.engine.label_layer(key)

    def seams(self, key, np):
        """The item's seam graph, PLACED on the slide (corner points convert
        through the record's origin/scale, so a trace stores slide
        coordinates like every other gesture)."""
        rec = self.record(key)
        if rec is None:
            return None
        from msseg.labeler.labeling import Placement
        return self.seams_for_record(rec, np, key,
                                     placement=Placement(rec["origin"], rec["scale"]))

    @staticmethod
    def seams_for_record(rec, np, key=None, placement=None):
        """The record's seam graph (seams.SeamGraph over its label raster),
        derived once and cached on the record (commit-keyed, so a Rerun
        recomputes it). The compiled ``seam_graph`` when the extension has it,
        else the numpy reference."""
        graph = rec.get("_seams")
        if graph is None and rec.get("labels") is not None:
            from msseg.labeler.seams import SeamGraph
            try:
                from msseg import mscoupon as _m
                ext = getattr(_m, "_ext", None)
            except ImportError:
                ext = None
            t0 = time.perf_counter()
            graph = SeamGraph.from_labels(rec["labels"], np, ext=ext, placement=placement,
                                          rev=int(rec.get("commit") or 0))
            rec["_seams"] = graph
            log(f"seams: {graph.n_seams} seams / {graph.n_junctions} junctions for "
                f"{key or 'slice'} ({1e3 * (time.perf_counter() - t0):.0f}ms, "
                f"{'c++' if getattr(ext, 'seam_graph', None) else 'numpy'})")
        return graph

    @staticmethod
    def arcs_for_record(rec, np, key=None):
        """The record's living-region arcs (MSC saddles), or pixel adjacency
        derived once and cached on the record (commit-keyed, so a Rerun
        recomputes it)."""
        arcs = rec.get("arcs")
        if arcs is None and rec.get("labels") is not None:
            t0 = time.perf_counter()
            arcs = magic_fill.arcs_from_labels(rec["labels"], np)
            rec["arcs"] = arcs
            log(f"magic fill: pixel adjacency for {key or 'item'}: "
                f"{len(arcs['a'])} pairs ({1e3 * (time.perf_counter() - t0):.0f}ms)")
        return arcs
