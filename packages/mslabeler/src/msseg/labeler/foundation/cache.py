"""``TokenCache``: token grids by (item, level, halo, extract key).

A grid is a function of the item's PIXELS and the extract settings only --
not of persistence, the statistics spec or the projection -- so it survives
every re-prime, re-measure and refit. It is also the expensive part (a GPU
pass) and fp16 attention is not bit-reproducible, so a grid is computed once
and kept: in memory (LRU, ``max_items``) and, with a ``root``, on disk as
``<root>/<sha1 of the key>.npz`` (the grid stored float16, plus its stride).
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from typing import Any, Callable, Optional, Tuple

import numpy as np

GridEntry = Tuple[np.ndarray, float]          # (grid, stride)


def cache_key(item_key: Any, level: Any, halo: int, extract_key: str) -> str:
    return json.dumps([str(item_key), str(level), int(halo), str(extract_key)])


class TokenCache:
    def __init__(self, root: Optional[str] = None, max_items: int = 16):
        self.root = root
        self.max_items = int(max_items)
        self._mem: "OrderedDict[str, GridEntry]" = OrderedDict()
        self.hits = self.misses = self.disk_hits = 0
        if root:
            os.makedirs(root, exist_ok=True)

    def _path(self, key: str) -> Optional[str]:
        if not self.root:
            return None
        return os.path.join(self.root, hashlib.sha1(key.encode("utf-8")).hexdigest() + ".npz")

    def get(self, key: str) -> Optional[GridEntry]:
        e = self._mem.get(key)
        if e is not None:
            self._mem.move_to_end(key)
            self.hits += 1
            return e
        path = self._path(key)
        if path and os.path.exists(path):
            with np.load(path, allow_pickle=False) as z:
                if str(z["key"]) == key:
                    e = (z["grid"], float(z["stride"]))
            if e is not None:
                self.disk_hits += 1
                self._remember(key, e)
                return e
        self.misses += 1
        return None

    def put(self, key: str, grid: np.ndarray, stride: float) -> GridEntry:
        e = (np.asarray(grid, np.float16), float(stride))
        self._remember(key, e)
        path = self._path(key)
        if path:
            tmp = path + ".tmp.npz"
            np.savez(tmp, grid=e[0], stride=np.float64(e[1]), key=np.str_(key))
            os.replace(tmp, path)
        return e

    def get_or_compute(self, key: str, compute: Callable[[], GridEntry]) -> GridEntry:
        e = self.get(key)
        if e is None:
            grid, stride = compute()
            e = self.put(key, grid, stride)
        return e

    def _remember(self, key: str, e: GridEntry) -> None:
        self._mem[key] = e
        self._mem.move_to_end(key)
        while len(self._mem) > self.max_items:
            self._mem.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        if key in self._mem:
            return True
        path = self._path(key)
        return bool(path and os.path.exists(path))
