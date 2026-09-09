"""The coupon viewer's implementations of the labeler framework's seams.

``SequenceCatalogue`` is the ``ItemCatalogue`` over the viewer's subsequences
of TIFF slices: an item is one slice, its key the folder-qualified
``"folder/basename"`` the annotations have always been stored under, its
address the ``(si, li)`` pair the viewer navigates by. ``EngineRegionProvider``
is the ``RegionProvider`` over ``ComputeEngine``: the per-slice records the
lazy tiers cache, computed synchronously on demand when a stack-wide
operation needs every slice.
"""
from __future__ import annotations

import os
import time

from msseg.labeler import magic_fill

from .common import log


class SequenceCatalogue:
    def __init__(self, app):
        self.app = app
        self._keys = []
        self._index = {}
        self._pos = {}

    def refresh(self):
        """Re-derive the key list from the viewer's flat slice order (call after
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

    def key_of(self, si, li):
        """Folder-qualified slice identity ("folder/basename") -- basenames
        collide across a session's folders, so the folder is part of the key."""
        try:
            s = self.app.subsequences[si]
            return f"{s.get('folder', '')}/{os.path.basename(s['files'][li])}"
        except (IndexError, KeyError, TypeError):
            return None

    def index_of(self, key):
        hit = self._index.get(key)
        if hit is not None or key is None:
            return hit
        for si, s in enumerate(self.app.subsequences):      # not (yet) primed
            for li in range(len(s.get("files") or [])):
                if self.key_of(si, li) == key:
                    return (si, li)
        return None

    def label(self, key):
        return key

    def group_of(self, key):
        """Leave-slices-out grouping: one group per slice."""
        pos = self._pos.get(key)
        return pos if pos is not None else key

    def tree(self):
        out = []
        for si, s in enumerate(self.app.subsequences):
            out.append({"key": None, "label": str(s.get("name") or f"sequence {si}"),
                        "children": [{"key": self.key_of(si, li),
                                      "label": os.path.basename(p), "children": []}
                                     for li, p in enumerate(s.get("files") or [])]})
        return out


class EngineRegionProvider:
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
        idx = self.catalogue.index_of(key)
        return None if idx is None else self.engine.record(*idx)

    def ensure_record(self, key):
        """The slice's record at the current commit, computing it SYNCHRONOUSLY
        when the lazy per-slice tier hasn't visited it yet (the training-set
        export needs every slice, not just the browsed ones). Caller must
        ensure no assembly worker is running (the pipes are stateful)."""
        idx = self.catalogue.index_of(key)
        if idx is None:
            return None
        rec = self.engine.record(*idx)
        if rec is not None and rec.get("labels") is not None:
            return rec
        try:
            import numpy as np
            from msseg import mscoupon as ext
        except ImportError:
            return None
        si, li = idx
        try:
            return self.engine.ensure_slice(si, li, self.app._assembly_params(si, "slice", li),
                                            ext, np)
        except Exception as exc:
            log(f"slice ({si},{li}) record failed: {exc}")
            return None

    def request(self, key):
        idx = self.catalogue.index_of(key)
        if idx is not None:
            self.app._request_assembly(idx[0])

    def pending(self):
        return self.engine.pending_work()

    def poll(self):
        return self.engine.poll()

    def arcs(self, key, np):
        rec = self.record(key)
        return None if rec is None else self.arcs_for_record(rec, np, key)

    def label_layer(self, key):
        rec = self.record(key)
        if rec is None or rec.get("labels") is None:
            return None
        from msseg.labeler.sources import ArrayLabelLayer
        return ArrayLabelLayer(rec["labels"], rev=int(rec.get("commit") or 0))

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
            log(f"magic fill: pixel adjacency for {key or 'slice'}: "
                f"{len(arcs['a'])} pairs ({1e3 * (time.perf_counter() - t0):.0f}ms)")
        return arcs
