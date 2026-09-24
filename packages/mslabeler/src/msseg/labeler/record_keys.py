"""A record's ``commit`` is its identity, not a generation.

Every cache in the framework keys on ``rec["commit"]`` and asks one question:
is this entry still about that record? (``entry[0] == rec["commit"]``, a
LabelLayer's ``rev``.) A global generation counter answers it too
conservatively -- every persistence tick, statistics edit or task switch bumps
it, so switching from task A to task B and back throws away A's predictions
although A's record never changed.

So a record's ``commit`` is the interned id of its CONTENT key: what the
record is a function of. Equal ids mean identical labels and rows; returning
to an earlier parameter set returns the earlier id, and every cache keyed on
it is valid again with no consumer changed. The key must include the identity
of the pipeline that produced the labels (a prime counter, never a hash of the
parameters): region ids are not stable from one prime to the next, so a
re-prime must never alias an older record.

``RecordCache`` is the bounded per-item store both engines hold: a few
records per item (``MSSEG_RECORDS_PER_ITEM``, default 4), oldest out first,
plus the rasters shared between records that differ only in their
measurement (the labels and arcs of one pipe at one persistence), so a second
statistics spec costs its rows rather than a second label raster.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Dict, Hashable, Optional

DEFAULT_RECORDS_PER_ITEM = 4


def records_per_item() -> int:
    try:
        return max(1, int(os.environ.get("MSSEG_RECORDS_PER_ITEM", "")
                          or DEFAULT_RECORDS_PER_ITEM))
    except ValueError:
        return DEFAULT_RECORDS_PER_ITEM


class Interner:
    """Hashable key -> a stable positive int, first come first numbered.

    ``start`` lets an engine keep ids above ones it handed out before (the
    coupon's generation started at 0, mspath's at 1). Nothing is ever
    forgotten: a key is small and a session sees thousands at most."""

    def __init__(self, start: int = 1):
        self._ids: Dict[Hashable, int] = {}
        self._next = int(start)

    def id_of(self, key: Hashable) -> int:
        got = self._ids.get(key)
        if got is None:
            got = self._ids[key] = self._next
            self._next += 1
        return got

    def fresh(self) -> int:
        """An id no key maps to (a generation that must match nothing)."""
        got = self._next
        self._next += 1
        return got

    def __len__(self):
        return len(self._ids)


class RecordCache:
    """Per item: record id -> record, least recently used first out."""

    def __init__(self, cap: Optional[int] = None):
        self.cap = int(cap) if cap else records_per_item()
        self._items: Dict[Hashable, "OrderedDict[int, Dict[str, Any]]"] = {}
        # (item, share key) -> (labels, arcs): the decomposition part of a
        # record, shared by records that differ only in their measurement.
        self._shared: Dict[Hashable, "OrderedDict[Hashable, Any]"] = {}

    def get(self, item, rid) -> Optional[Dict[str, Any]]:
        recs = self._items.get(item)
        if not recs:
            return None
        rec = recs.get(rid)
        if rec is not None:
            recs.move_to_end(rid)
        return rec

    def put(self, item, rid, rec):
        recs = self._items.setdefault(item, OrderedDict())
        recs[rid] = rec
        recs.move_to_end(rid)
        while len(recs) > self.cap:
            recs.popitem(last=False)
        return rec

    def shared(self, item, share_key):
        got = self._shared.get(item)
        if not got:
            return None
        val = got.get(share_key)
        if val is not None:
            got.move_to_end(share_key)
        return val

    def share(self, item, share_key, value):
        got = self._shared.setdefault(item, OrderedDict())
        got[share_key] = value
        got.move_to_end(share_key)
        while len(got) > self.cap:
            got.popitem(last=False)
        return value

    def latest(self, item) -> Optional[Dict[str, Any]]:
        recs = self._items.get(item)
        if not recs:
            return None
        return next(reversed(recs.values()))

    def drop(self, item):
        self._items.pop(item, None)
        self._shared.pop(item, None)

    def clear(self):
        self._items.clear()
        self._shared.clear()

    def items(self):
        return list(self._items)

    def count(self, item) -> int:
        return len(self._items.get(item) or ())

    def __len__(self):
        return sum(1 for recs in self._items.values() if recs)
