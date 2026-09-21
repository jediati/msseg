"""``mspath-embed`` -- harvest region rows off slides, train the task-free
region encoder (``docs/design_region_autoencoder.md``).

    mspath-embed harvest --tiff-folder DIR --process-profile P.json --out harvest_L4/ [--level 4]
    mspath-embed train   harvest_L4/ --out encoders/wsi_L4.msenc [--dim 8] [--arch mlp|pca]

**Harvest** runs the labeler's own prime -- ``SlideEngine.prime_item`` over
the same ``Item`` rects the GUI would make -- so a harvested row is exactly a
row the labeler shows, positional columns in slide coordinates included. Per
slide: the requested level is cut into ``--roi`` square tiles (or taken whole
when it fits), the tiles are ranked by tissue content off a coarse level and
those under ``--min-tissue`` are never primed (most of a slide is glass), and
each primed tile is recorded at the profile's persistence and at every other
``--factor`` of it before its pipeline is released. One ``.npz`` shard per
(tile, persistence) makes the harvest resumable; the per-level persistence
pin is written to ``harvest.json`` after the first tile and restored on
resume, so a resumed harvest thresholds like the first run did rather than
against whatever tile happens to come first.

**Train** is the framework's (``msseg.labeler.embedding_train.cli``): the same
sub-command a coupon driver would expose.
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from msseg.labeler.embedding_train.cli import add_train_arguments, run_train
from msseg.labeler.embedding_train.shards import HarvestWriter
from msseg.labeler.fields import DEFAULT as FIELDS
from msseg.labeler.magic_fill import index_arcs

from .common import log
from .items import Item, overview, roi, slide_id

DEFAULT_ROI = 4096
DEFAULT_FACTORS = (1.0, 0.5, 2.0)
DEFAULT_MIN_TISSUE = 0.05
# A brightfield background is near the top of the range; a pixel whose darkest
# plane is below this fraction of the range counts as tissue.
DEFAULT_TISSUE_FRACTION = 0.86
MAP_PIXELS = 4_000_000            # the tissue map is read at the coarsest level with at most this
SLIDE_GLOBS = ("*.tif", "*.tiff", "*.svs", "*.ndpi")
_RULE = re.compile(r"^\s*([A-Za-z_][\w.\-\[\]]*)\s*(<=|>=|==|!=|<|>)\s*([-+]?[\d.eE+-]+)\s*$")


# --------------------------------------------------------------------------- #
# pure helpers (tested without a slide)
# --------------------------------------------------------------------------- #
def tile_grid(level_shape: Tuple[int, int], roi: int) -> List[Optional[Tuple[int, int, int, int]]]:
    """Tiles ``(lx, ly, lw, lh)`` in LEVEL pixels covering `level_shape`;
    ``[None]`` when the whole level fits in one `roi` square (the overview
    case: one item, no rect)."""
    h, w = int(level_shape[0]), int(level_shape[1])
    roi = max(1, int(roi))
    if w <= roi and h <= roi:
        return [None]
    out = []
    for y in range(0, h, roi):
        for x in range(0, w, roi):
            out.append((x, y, min(roi, w - x), min(roi, h - y)))
    return out


def parse_rule(text: Optional[str]) -> Optional[Tuple[str, str, float]]:
    """``"mean_base>200"`` -> ``("mean_base", ">", 200.0)``; None for None."""
    if text is None or not str(text).strip():
        return None
    m = _RULE.match(str(text))
    if m is None:
        raise ValueError(f"bad rule {text!r}: expected FIELD OP NUMBER, e.g. mean_base>200")
    return m.group(1), m.group(2), float(m.group(3))


def apply_rule(rule, table, np_=np) -> np.ndarray:
    """Boolean mask of the rows the rule MATCHES (the harvest drops them)."""
    field, op, value = rule
    col = table.column(field)
    if col is None:
        raise ValueError(f"rule names {field!r}, which the statistics table has no column for "
                         f"(have: {', '.join(list(table.names)[:8])}…)")
    c = np_.asarray(col, np_.float64)
    if op == "<":
        return c < value
    if op == "<=":
        return c <= value
    if op == ">":
        return c > value
    if op == ">=":
        return c >= value
    if op == "==":
        return c == value
    return c != value


def scaled_profile(params: Dict[str, Any], factor: float) -> Dict[str, Any]:
    """The params with the persistence (percent, or an explicit absolute)
    multiplied by `factor`."""
    out = copy.deepcopy(params)
    msc = out.setdefault("msc", {})
    if msc.get("persistence_absolute") is not None:
        msc["persistence_absolute"] = float(msc["persistence_absolute"]) * float(factor)
    else:
        msc["persistence_percent"] = float(msc.get("persistence_percent", 10.0) or 0.0) * float(factor)
    return out


def tissue_fractions(tissue: np.ndarray, map_scale: float, tiles, level_scale: float,
                     level_shape: Tuple[int, int]) -> List[float]:
    """Per tile, the fraction of tissue pixels under it on the boolean map
    `tissue` (a coarser level with `map_scale` slide px per pixel)."""
    r = float(level_scale) / float(map_scale)          # map px per level px
    mh, mw = tissue.shape
    out = []
    for t in tiles:
        lx, ly, lw, lh = (0, 0, level_shape[1], level_shape[0]) if t is None else t
        x0, y0 = int(np.floor(lx * r)), int(np.floor(ly * r))
        x1, y1 = int(np.ceil((lx + lw) * r)), int(np.ceil((ly + lh) * r))
        x0, y0 = max(0, min(mw, x0)), max(0, min(mh, y0))
        x1, y1 = max(x0 + 1, min(mw, x1)), max(y0 + 1, min(mh, y1))
        block = tissue[y0:y1, x0:x1]
        out.append(float(block.mean()) if block.size else 0.0)
    return out


def drop_rows(keep: np.ndarray, ia: np.ndarray, ib: np.ndarray, saddle):
    """Arcs re-indexed to the kept rows: ``(ia, ib, saddle)`` over arcs whose
    both endpoints survive."""
    keep = np.asarray(keep, bool)
    new = np.cumsum(keep) - 1
    ia = np.asarray(ia, np.int64)
    ib = np.asarray(ib, np.int64)
    ok = keep[ia] & keep[ib]
    s = None if saddle is None else np.asarray(saddle, np.float32)[ok]
    return new[ia[ok]], new[ib[ok]], s


def find_slides(folder: Optional[str], explicit: Sequence[str], pattern: Optional[str]) -> List[str]:
    paths = [os.path.abspath(p) for p in explicit or ()]
    if folder:
        pats = [pattern] if pattern else list(SLIDE_GLOBS)
        for pat in pats:
            paths += sorted(os.path.abspath(p) for p in glob.glob(os.path.join(folder, pat)))
    seen, out = set(), []
    for p in paths:
        k = os.path.normcase(p)
        if k not in seen and os.path.isfile(p):
            seen.add(k)
            out.append(p)
    return out


# --------------------------------------------------------------------------- #
# the harvest
# --------------------------------------------------------------------------- #
class Harvester:
    """One harvest run. Split from ``run_harvest`` so a test can drive it on a
    synthetic slide and count primes."""

    def __init__(self, ext, params: Dict[str, Any], out_dir: str, level: Optional[int],
                 roi: int = DEFAULT_ROI, halo: int = 0, factors=DEFAULT_FACTORS,
                 min_tissue: float = DEFAULT_MIN_TISSUE, tissue_max: Optional[float] = None,
                 blank=None, max_tiles: Optional[int] = None, say=log, profile_doc=None,
                 profile_path: Optional[str] = None):
        from .engine import SlideEngine
        self.ext = ext
        self.params = params
        self.params_json = json.dumps(params, sort_keys=True)
        self.profile_hash = hashlib.sha1(self.params_json.encode("utf-8")).hexdigest()
        self.level = level
        self.roi = int(roi)
        self.halo = int(halo)
        self.factors = tuple(float(f) for f in factors) or (1.0,)
        self.min_tissue = float(min_tissue)
        self.tissue_max = tissue_max
        self.blank = blank
        self.max_tiles = max_tiles
        self.say = say
        self.engine = SlideEngine()
        self.names = list(ext.feature_fields(self.params_json))
        try:
            schema = list(ext.feature_schema(self.params_json))
        except Exception:
            schema = None
        doc = {"profile": profile_doc, "profile_path": profile_path, "params": params,
               "profile_hash": self.profile_hash, "level": level, "roi": self.roi,
               "halo": self.halo, "factors": list(self.factors), "schema": schema,
               "positional": sorted(FIELDS.positional), "min_tissue": self.min_tissue,
               "tissue": None, "blank": (list(blank) if blank else None),
               "scope": (f"L{int(level)}" if level is not None else None),
               "slides": [], "pins": {}, "tiles": {}, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.writer = HarvestWriter(out_dir, self.names, doc)
        if self.writer.resumed:
            say(f"resuming {self.writer.path} ({len(self.writer.doc['shards'])} shards present)")
            lv = self.writer.doc.get("level")
            if lv is not None and level is not None and int(lv) != int(level):
                raise ValueError(f"the harvest is at level {lv}; asked for {level}")
            if self.writer.doc.get("profile_hash") not in (None, self.profile_hash):
                raise ValueError("the harvest was made with another profile; use a new directory")
        self.primes = 0

    # -- tissue ------------------------------------------------------------ #
    def tissue_map(self, src, level: int):
        """``(mask, map_scale)``: a boolean tissue map on the coarsest level
        with at most MAP_PIXELS pixels (never coarser than `level` itself)."""
        map_level = int(level)
        for lv in range(src.levels - 1, int(level) - 1, -1):
            h, w = src.level_shape(lv)
            if h * w <= MAP_PIXELS:
                map_level = lv
                break
        h, w = src.level_shape(map_level)
        a = src.read_region(map_level, 0, 0, w, h)
        lo, hi = src.value_range()
        top = self.tissue_max if self.tissue_max is not None else lo + DEFAULT_TISSUE_FRACTION * (hi - lo)
        dark = a.min(axis=2) if a.ndim == 3 else a
        mask = np.asarray(dark, np.float64) < float(top)
        return mask, src.level_scale(map_level), float(top), map_level

    # -- one slide ----------------------------------------------------------- #
    def harvest_slide(self, slide_index: int, path: str, folder: Optional[str] = None) -> int:
        sid = slide_id(folder or os.path.dirname(path), path)
        self.engine.register(sid, path)
        src = self.engine.source(sid)
        if self.level is None:
            self.level = src.levels - 1
            self.writer.update(level=self.level, scope=f"L{self.level}")
        level = int(self.level)
        if not 0 <= level < src.levels:
            # Never clamp: a level is the scope every row is measured under,
            # and a slide read at another one would poison the harvest.
            raise ValueError(f"{sid}: level {level} requested but the reader ({src.be.name}) "
                             f"sees {src.levels} level(s)")
        shape = src.level_shape(level)
        tiles = tile_grid(shape, self.roi)
        mask, map_scale, top, map_level = self.tissue_map(src, level)
        fracs = tissue_fractions(mask, map_scale, tiles, src.level_scale(level), shape)
        order = sorted(range(len(tiles)), key=lambda i: -fracs[i])
        wanted = [i for i in order if fracs[i] >= self.min_tissue]
        if self.max_tiles is not None:
            wanted = wanted[:int(self.max_tiles)]
        self.say(f"{sid}: level {level} {shape[1]}x{shape[0]}, {len(tiles)} tile(s), "
                 f"{len(wanted)} with tissue >= {self.min_tissue:g} "
                 f"(tissue = darkest plane < {top:g} on level {map_level})")
        slides = [s for s in self.writer.doc.get("slides", []) if s.get("index") != slide_index]
        slides.append({"index": int(slide_index), "id": sid, "path": os.path.abspath(path),
                       "level": int(level), "shape": [int(shape[0]), int(shape[1])],
                       "scale": float(src.level_scale(level)), "tiles": len(tiles),
                       "tissue_tiles": len(wanted)})
        self.writer.update(slides=slides, tissue={"max": top, "map_level": int(map_level)})
        # The pins: what a resume must restore before its first prime.
        for k, v in (self.writer.doc.get("pins") or {}).items():
            self.engine.level_range.setdefault(int(k), float(v))

        written = 0
        for n, ti in enumerate(wanted, start=1):
            if all(self.writer.has(slide_index, ti, f) for f in self.factors):
                continue
            t = tiles[ti]
            if t is None:
                item = overview(sid, level)
                halo = 0
            else:
                s = src.level_scale(level)
                lx, ly, lw, lh = t
                item = roi(sid, level, round(lx * s), round(ly * s), round(lw * s), round(lh * s))
                halo = self.halo
            t0 = time.perf_counter()
            self.engine.prime_item(item, self.params, halo=halo, quiet=True)
            self.primes += 1
            rows_written = []
            for f in self.factors:
                self.engine.commit_selection()
                rec = self.engine.ensure_record(item.key, scaled_profile(self.params, f))
                if rec is None:
                    raise RuntimeError(f"{item.key}: no record after priming")
                rows_written.append(self._write_record(slide_index, ti, f, item, rec))
            self.engine.forget(item.key)
            # The pin resolves inside the first record, so it is written after.
            self.writer.update(pins={str(k): float(v) for k, v in self.engine.level_range.items()})
            written += 1
            self.say(f"  tile {n}/{len(wanted)} {item.key}: rows {'/'.join(map(str, rows_written))} "
                     f"at factors {'/'.join(f'{f:g}' for f in self.factors)} "
                     f"({time.perf_counter() - t0:.1f}s)")
        return written

    def _write_record(self, slide_index, tile_index, factor, item: Item, rec) -> int:
        table = rec["stats"]
        if list(table.names) != self.names:
            raise RuntimeError("the record's columns differ from the profile's schema")
        values = np.asarray(table.values, np.float64)
        n = values.shape[0]
        ids = table.column(FIELDS.id_field)
        arcs = rec.get("arcs")
        if arcs is not None and ids is not None and len(arcs.get("a", ())):
            ia, ib, keep_arcs = index_arcs(arcs, ids, np)
            saddle = arcs.get("saddle")
            saddle = None if saddle is None else np.asarray(saddle, np.float32)[keep_arcs]
        else:
            ia = ib = np.zeros(0, np.int64)
            saddle = None
        keep = np.ones(n, bool)
        if self.blank is not None:
            keep &= ~apply_rule(self.blank, table, np)
        if not keep.all():
            values = values[keep]
            ia, ib, saddle = drop_rows(keep, ia, ib, saddle)
        level = int(rec.get("level", item.level))
        meta = {"key": item.key, "slide": item.slide, "level": level,
                "persistence": float(self.engine.persistence_abs.get(level, float("nan"))),
                "origin": list(rec.get("origin") or (0, 0)), "scale": float(rec.get("scale") or 1.0),
                "n_dropped": int(n - int(keep.sum()))}
        self.writer.write(slide_index, tile_index, factor, values, ia, ib, saddle, meta)
        return int(values.shape[0])

    def close(self):
        self.engine.close_sources()


def compose_params(profile_doc: Dict[str, Any], channels: int) -> Dict[str, Any]:
    """The priming params the GUI would compose from this profile (the coupon
    composer, so a slide is primed by exactly one pipeline description)."""
    from msseg.mscoupon import session as coupon_session
    return json.loads(coupon_session.profile_params_json(profile_doc, 1, channels))


def run_harvest(args: argparse.Namespace) -> int:
    say = (lambda _m: None) if args.quiet else log
    try:
        from msseg.mscoupon import mscoupon_py as ext
    except Exception as exc:
        print(f"the compiled mscoupon extension is needed to harvest: {exc}", file=sys.stderr)
        return 2
    slides = find_slides(args.tiff_folder, args.slide, args.glob)
    if not slides:
        print("no slides found", file=sys.stderr)
        return 2
    with open(args.process_profile, "r", encoding="utf-8") as f:
        profile_doc = json.load(f)
    slide_block = profile_doc.get("slide") or {}
    level = args.level if args.level is not None else slide_block.get("overview_level")
    halo = args.halo if args.halo is not None else int(slide_block.get("halo", 0) or 0)
    # The plane count is a declared fact of the params; take it from the first slide.
    from msseg.labeler.pyramid import PyramidImageSource
    first = PyramidImageSource(slides[0])
    try:
        channels = int(first.channels)
    finally:
        first.close()
    params = compose_params(profile_doc, channels)
    if args.persistence_absolute is not None:
        params.setdefault("msc", {})["persistence_absolute"] = float(args.persistence_absolute)
    factors = tuple(float(v) for v in str(args.factors).split(",") if v.strip()) or (1.0,)
    blank = parse_rule(args.blank)
    h = Harvester(ext, params, args.out, None if level is None else int(level), roi=args.roi,
                  halo=halo, factors=factors, min_tissue=args.min_tissue,
                  tissue_max=args.tissue_max, blank=blank, max_tiles=args.max_tiles, say=say,
                  profile_doc=profile_doc, profile_path=os.path.abspath(args.process_profile))
    t0 = time.perf_counter()
    total = 0
    try:
        for i, path in enumerate(slides):
            total += h.harvest_slide(i, path, args.tiff_folder)
    except (ValueError, RuntimeError) as exc:
        print(f"harvest stopped: {exc}", file=sys.stderr)
        return 2
    finally:
        h.close()
    n_shards = len(h.writer.doc.get("shards", []))
    say(f"harvest {args.out}: {len(slides)} slide(s), {total} tile(s) primed this run, "
        f"{n_shards} shard(s) on disk ({time.perf_counter() - t0:.0f}s)")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mspath-embed", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    hv = sub.add_parser("harvest", help="prime slides tile by tile and write region rows + arcs")
    hv.add_argument("--tiff-folder", default=None, help="folder of slides (*.tif, *.tiff, *.svs, *.ndpi)")
    hv.add_argument("--slide", action="append", default=[], help="a slide path (repeatable)")
    hv.add_argument("--glob", default=None, help="pattern inside --tiff-folder (default: the slide extensions)")
    hv.add_argument("--process-profile", required=True, help="a labeler profile JSON")
    hv.add_argument("--out", required=True, help="harvest directory (created; resumed if present)")
    hv.add_argument("--level", type=int, default=None,
                    help="pyramid level to harvest (default: the profile's overview level)")
    hv.add_argument("--roi", type=int, default=DEFAULT_ROI, help="tile side in level pixels (default 4096)")
    hv.add_argument("--halo", type=int, default=None, help="compute halo in level pixels (default: the profile's)")
    hv.add_argument("--factors", default=",".join(f"{f:g}" for f in DEFAULT_FACTORS),
                    help="persistence multipliers to record each tile at (default 1,0.5,2)")
    hv.add_argument("--persistence-absolute", type=float, default=None,
                    help="pin the threshold in field units instead of resolving the profile's percent")
    hv.add_argument("--min-tissue", type=float, default=DEFAULT_MIN_TISSUE,
                    help="skip tiles whose tissue fraction is below this (default 0.05; 0 = prime every tile)")
    hv.add_argument("--tissue-max", type=float, default=None,
                    help="a pixel is tissue when its darkest plane is below this value "
                         "(default 0.86 of the data range: brightfield background is bright)")
    hv.add_argument("--blank", default=None,
                    help="drop rows matching FIELD OP NUMBER, e.g. \"mean_base>230\"")
    hv.add_argument("--max-tiles", type=int, default=None, help="prime at most this many tiles per slide")
    hv.add_argument("--quiet", action="store_true")
    hv.set_defaults(func=run_harvest)

    tr = sub.add_parser("train", help="train the encoder on a harvest")
    add_train_arguments(tr)
    tr.set_defaults(func=run_train)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
