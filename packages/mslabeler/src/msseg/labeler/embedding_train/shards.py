"""The harvest on disk.

    <out>/harvest.json          what was harvested, and how (see HarvestWriter)
    <out>/shards/<name>.npz     one per (item, persistence): rows + arcs

A shard holds the statistics table of one item at one persistence
(``rows`` float32 ``[n, n_fields]`` in the harvest's column order, positional
columns included -- the trainer drops them), the region arcs as ROW indices
into that table (``ia``, ``ib`` int32, ``saddle`` float32), and a JSON
``meta`` string (item key, slide, level, the absolute persistence and the
factor it came from). Rows never reference another shard, so a random walk
stays inside its item by construction, and a shard is the unit of resume: a
tile whose shards all exist is skipped by the harvest.

``harvest.json`` is rewritten after every shard, so a harvest interrupted
half-way is a valid, smaller harvest.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

HARVEST_FILE = "harvest.json"
SHARD_DIR = "shards"
HARVEST_VERSION = 1
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def shard_name(slide_index: int, tile_index: int, factor: float) -> str:
    return f"s{int(slide_index):03d}_t{int(tile_index):05d}_p{float(factor):g}.npz"


def _safe(text: str) -> str:
    return _SAFE.sub("_", str(text))


class HarvestWriter:
    """Creates or resumes a harvest directory.

    `names` are the table's column names; a resumed harvest must carry the
    same ones (a different profile is a different harvest). `doc` seeds
    ``harvest.json`` on creation and is ignored on resume except for keys
    that were absent -- the harvest's own record wins, so a resume under
    changed flags is visible in the file rather than silently mixed.
    """

    def __init__(self, out_dir: str, names: Sequence[str], doc: Optional[Dict[str, Any]] = None):
        self.dir = str(out_dir)
        self.shard_dir = os.path.join(self.dir, SHARD_DIR)
        os.makedirs(self.shard_dir, exist_ok=True)
        self.path = os.path.join(self.dir, HARVEST_FILE)
        names = [str(n) for n in names]
        self.resumed = os.path.isfile(self.path)
        if self.resumed:
            with open(self.path, "r", encoding="utf-8") as f:
                self.doc = json.load(f)
            have = [str(n) for n in self.doc.get("names") or []]
            if have != names:
                raise ValueError(f"{self.path}: the harvest's columns differ from this "
                                 f"profile's ({len(have)} vs {len(names)}); use a new directory")
            for k, v in (doc or {}).items():
                self.doc.setdefault(k, v)
        else:
            self.doc = {"version": HARVEST_VERSION, "names": names, "shards": []}
            self.doc.update(doc or {})
            self.save()
        self.doc.setdefault("shards", [])
        self._known = {s["file"] for s in self.doc["shards"]}

    @property
    def names(self) -> List[str]:
        return list(self.doc["names"])

    def has(self, slide_index: int, tile_index: int, factor: float) -> bool:
        name = shard_name(slide_index, tile_index, factor)
        return name in self._known and os.path.isfile(os.path.join(self.shard_dir, name))

    def write(self, slide_index: int, tile_index: int, factor: float, rows, ia, ib, saddle,
              meta: Dict[str, Any]) -> str:
        rows = np.ascontiguousarray(rows, np.float32)
        if rows.ndim != 2 or rows.shape[1] != len(self.doc["names"]):
            raise ValueError("shard rows do not match the harvest's columns")
        ia = np.asarray(ia, np.int32).ravel()
        ib = np.asarray(ib, np.int32).ravel()
        n = rows.shape[0]
        if len(ia) != len(ib) or (len(ia) and (ia.min() < 0 or ib.min() < 0
                                                or ia.max() >= n or ib.max() >= n)):
            raise ValueError("shard arcs must be row indices into the shard's rows")
        saddle = (np.full(len(ia), np.nan, np.float32) if saddle is None
                  else np.asarray(saddle, np.float32).ravel())
        name = shard_name(slide_index, tile_index, factor)
        path = os.path.join(self.shard_dir, name)
        meta = dict(meta)
        meta.update({"slide_index": int(slide_index), "tile_index": int(tile_index),
                     "factor": float(factor), "n_rows": int(n), "n_arcs": int(len(ia))})
        tmp = path + ".part"
        with open(tmp, "wb") as f:              # a handle: savez appends no suffix
            np.savez(f, rows=rows, ia=ia, ib=ib, saddle=saddle,
                     meta=np.array(json.dumps(meta, sort_keys=True)))
        os.replace(tmp, path)
        entry = dict(meta)
        entry["file"] = name
        self.doc["shards"] = [s for s in self.doc["shards"] if s.get("file") != name] + [entry]
        self._known.add(name)
        self.save()
        return path

    def update(self, **fields) -> None:
        self.doc.update(fields)
        self.save()

    def save(self) -> None:
        tmp = self.path + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.doc, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)


@dataclass
class Harvest:
    """Every loaded shard concatenated: rows in one block, arcs as GLOBAL row
    indices, and per-row provenance."""
    names: List[str]
    rows: np.ndarray                   # float32 [n, n_fields]
    ia: np.ndarray                     # int64 [m]
    ib: np.ndarray                     # int64 [m]
    saddle: np.ndarray                 # float32 [m]
    row_shard: np.ndarray              # int32 [n]: index into `shards`
    shards: List[Dict[str, Any]]       # the loaded shards' meta, in row order
    doc: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return int(self.rows.shape[0])

    @property
    def n_arcs(self) -> int:
        return int(len(self.ia))

    def column(self, name: str) -> Optional[np.ndarray]:
        try:
            return self.rows[:, self.names.index(name)]
        except ValueError:
            return None


class HarvestReader:
    def __init__(self, harvest_dir: str):
        self.dir = str(harvest_dir)
        self.path = os.path.join(self.dir, HARVEST_FILE)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"{self.path}: not a harvest directory")
        with open(self.path, "r", encoding="utf-8") as f:
            self.doc = json.load(f)
        self.names = [str(n) for n in self.doc.get("names") or []]
        self.shard_dir = os.path.join(self.dir, SHARD_DIR)

    def shard_entries(self) -> List[Dict[str, Any]]:
        out = []
        for s in self.doc.get("shards") or []:
            p = os.path.join(self.shard_dir, s.get("file", ""))
            if os.path.isfile(p):
                out.append(dict(s))
        return out

    def load(self, max_rows: Optional[int] = None, seed: int = 0,
             factors: Optional[Iterable[float]] = None) -> Harvest:
        """Concatenate the shards. `max_rows` bounds the total by taking a
        seeded random subset of WHOLE shards (a walk needs its whole item);
        `factors` restricts to the listed persistence factors."""
        entries = self.shard_entries()
        if factors is not None:
            want = {float(f) for f in factors}
            entries = [e for e in entries if float(e.get("factor", 1.0)) in want]
        if max_rows is not None:
            rng = np.random.default_rng(int(seed))
            order = rng.permutation(len(entries))
            picked, total = [], 0
            for i in order:
                if total >= int(max_rows):
                    break
                picked.append(entries[i])
                total += int(entries[i].get("n_rows", 0))
            entries = picked
        rows, ia, ib, saddle, row_shard, metas = [], [], [], [], [], []
        offset = 0
        for k, e in enumerate(entries):
            with np.load(os.path.join(self.shard_dir, e["file"])) as z:
                r = np.asarray(z["rows"], np.float32)
                a = np.asarray(z["ia"], np.int64) + offset
                b = np.asarray(z["ib"], np.int64) + offset
                s = np.asarray(z["saddle"], np.float32)
                meta = json.loads(str(z["meta"]))
            if r.shape[1] != len(self.names):
                raise ValueError(f"{e['file']}: {r.shape[1]} columns, harvest has {len(self.names)}")
            rows.append(r); ia.append(a); ib.append(b); saddle.append(s)
            row_shard.append(np.full(r.shape[0], k, np.int32))
            metas.append(meta)
            offset += r.shape[0]
        if not rows:
            raise ValueError(f"{self.dir}: no shards to load")
        return Harvest(names=list(self.names),
                       rows=np.concatenate(rows), ia=np.concatenate(ia), ib=np.concatenate(ib),
                       saddle=np.concatenate(saddle), row_shard=np.concatenate(row_shard),
                       shards=metas, doc=dict(self.doc))
