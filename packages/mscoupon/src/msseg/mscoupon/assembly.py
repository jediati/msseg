"""Per-slice trim + connected-components 3D assembly for the mscoupon viewer.

Pipeline (mirrors the C++ CLI so the GUI and an exported config agree):
  MSC per slice -> PER-SLICE SELECTION (kept 2D merged regions) -> PIXEL TRIM
  (keep/omit pixels by base/filtered value) -> PER-SLICE connected components
  (in-plane) -> STREAMING 6-neighbor GLOBAL connected components.

Nodes are the per-slice connected components of the selected+trimmed raster (NOT
raw MSC ids -- trimming can split a region). Cross-slice identity is resolved with
``scipy.sparse.csgraph.connected_components`` over the accumulated edge graph, then
global ids are renumbered in FIRST-SEEN (appearance) order so the numbering matches
the streaming C++ ``SliceMatcher``. Statistics are a node-level reduction over the
final components (no pixel revisit), computed from the actual (trimmed) pixel sets.

Pure Python + numpy + scipy (all mscoupon deps); importable/testable headlessly.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


# Cross-slice in-plane offsets for each 3D connectivity (the z+1 neighborhood).
# 6 = face only (same x,y); 18 = + edges; 26 = + corners. This ALSO selects the
# in-plane 2D structuring element for the per-slice CC (see `cc_structure`).
_OFFSETS = {
    6: [(0, 0)],
    18: [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
    26: [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
}


def _shift(b: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Return `bs` with bs[y,x] = b[y+dy, x+dx], out-of-range -> 0 (non-wrapping)."""
    out = np.zeros_like(b)
    h, w = b.shape
    y_dst = slice(max(0, -dy), h - max(0, dy))
    x_dst = slice(max(0, -dx), w - max(0, dx))
    y_src = slice(max(0, dy), h - max(0, -dy))
    x_src = slice(max(0, dx), w - max(0, -dx))
    out[y_dst, x_dst] = b[y_src, x_src]
    return out


# --------------------------------------------------------------------------- #
# Stage helpers (each operates on one slice's rasters)
# --------------------------------------------------------------------------- #
def selection_mask(labels: np.ndarray, kept: Optional[Set[int]]) -> np.ndarray:
    """Boolean mask of pixels whose MSC region id is in `kept` (all if None)."""
    fg = labels >= 0
    if kept is None:
        return fg
    K = int(labels.max()) + 1 if labels.size else 1
    lut = np.zeros(K, dtype=bool)
    if kept:
        ids = np.fromiter((int(i) for i in kept), dtype=np.int64)
        ids = ids[(ids >= 0) & (ids < K)]
        lut[ids] = True
    return fg & lut[np.where(fg, labels, 0)]


def apply_pixel_filters(mask: np.ndarray, base: np.ndarray, filt: np.ndarray,
                        rules: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Refine `mask` by the pixel intensity trim chain. Each rule:
    {channel: base|filtered, mode: keep|omit, op: lt|le|gt|ge, value: float}.
    keep -> drop pixels that FAIL the predicate; omit -> drop pixels that PASS."""
    out = mask.copy()
    for r in rules:
        chan = base if r.get("channel") == "base" else filt
        op, val = r.get("op", "gt"), float(r.get("value", 0.0))
        if op == "lt":
            passed = chan < val
        elif op == "le":
            passed = chan <= val
        elif op == "gt":
            passed = chan > val
        elif op == "ge":
            passed = chan >= val
        else:
            continue
        out &= passed if r.get("mode", "keep") == "keep" else ~passed
    return out


def cc_structure(connectivity: int):
    """In-plane structuring element: 6 -> 4-connectivity, 18/26 -> 8-connectivity."""
    from scipy import ndimage
    return ndimage.generate_binary_structure(2, 1 if connectivity == 6 else 2)


def per_slice_cc(mask: np.ndarray, connectivity: int = 6):
    """In-plane connected components of `mask`. Returns (lbl, n) where lbl is the
    scipy labeling (0 = background, 1..n = components)."""
    from scipy import ndimage
    lbl, n = ndimage.label(mask, structure=cc_structure(connectivity))
    return lbl.astype(np.int64), int(n)


def chan_col(name: str, part: str) -> str:
    """Column key for one measurement channel's accumulator.

    Namespaced so a channel called e.g. "min_x" could never collide with the
    geometry columns; the C++ side gets this for free by indexing slots.
    """
    return f"ch:{name}:{part}"


_CHANNEL_PARTS = ("sum", "sumsq", "min", "max", "ext")


def node_stats(lbl: np.ndarray, n: int, base: np.ndarray, filt: np.ndarray,
               ascending: bool = True, relevance_floor: float = 0.0,
               relevance_ceiling: float = 0.0,
               channels: Optional[Sequence[Tuple[str, np.ndarray]]] = None,
               ) -> Dict[str, np.ndarray]:
    """Per-component (1..n) statistics from the actual pixel sets. Structure-of-
    arrays indexed 0..n-1. Sums use bincount; min/max/bbox use ufunc.at.

    `ascending` picks the side the seeding extremum comes from, mirroring
    ``mscoupon::label_selected_components``: the pixel attaining the component's
    filtered minimum (ascending) or maximum (descending). It is taken from the
    component's own pixels rather than inherited from the MSC feature, because
    the pixel trim runs first and can remove the feature's true extremum.

    `channels` is the measurement set: [(name, raster)], mirroring the resolved
    slot list on the C++ side. Each contributes sum/sumsq/min/max plus its value
    at the seeding extremum. `filt` is still read separately and unconditionally,
    because its extent is what LOCATES that extremum -- exactly as `filt_min`/
    `filt_max` survive on Msc2DFeatureStat whether or not `filtered` is measured.
    Defaults to the base channel alone, which is the default StatsSpec.
    """
    h, w = lbl.shape
    if channels is None:
        channels = [("base", base)]
    out: Dict[str, np.ndarray] = {}
    if n == 0:
        for k in ("area", "filt_min", "filt_max",
                  "base_relevance_floor", "base_relevance_ceiling",
                  "min_x", "max_x", "min_y", "max_y",
                  "ext_x", "ext_y", "ext_filtered"):
            out[k] = np.zeros(0, dtype=np.float64)
        for name, _ in channels:
            for part in _CHANNEL_PARTS:
                out[chan_col(name, part)] = np.zeros(0, dtype=np.float64)
        return out
    # Everything below works on the compressed foreground vectors, not the full
    # raster. `ndimage.minimum`/`maximum`/`minimum_position` with an index array
    # are ~900-1000 ms per call on a 3232^2 slice -- five calls made this function
    # ~95% of a Rerun-selection. The ufunc.at equivalents are ~5 ms, so the same
    # numbers come out ~175x faster.
    flat = lbl.ravel()
    fg = flat > 0
    idx = flat[fg] - 1                                   # 0..n-1
    f = filt.ravel()[fg].astype(np.float64)
    out["area"] = np.bincount(idx, minlength=n).astype(np.float64)

    def _reduce(op, values, init):
        acc = np.full(n, init, dtype=np.float64)
        op.at(acc, idx, values)
        # A component with no pixels keeps the sentinel; report 0 rather than inf.
        return np.where(np.isfinite(acc), acc, 0.0)

    # One compressed vector per measurement channel, kept for the extremum pass.
    chan_values: List[Tuple[str, np.ndarray]] = []
    for name, raster in channels:
        v = np.asarray(raster).ravel()[fg].astype(np.float64)
        chan_values.append((name, v))
        out[chan_col(name, "sum")] = np.bincount(idx, weights=v, minlength=n)
        out[chan_col(name, "sumsq")] = np.bincount(idx, weights=v * v, minlength=n)
        out[chan_col(name, "min")] = _reduce(np.minimum, v, np.inf)
        out[chan_col(name, "max")] = _reduce(np.maximum, v, -np.inf)

    out["base_relevance_floor"] = np.full(n, relevance_floor, dtype=np.float64)
    out["base_relevance_ceiling"] = np.full(n, relevance_ceiling, dtype=np.float64)
    out["filt_min"] = _reduce(np.minimum, f, np.inf)
    out["filt_max"] = _reduce(np.maximum, f, -np.inf)

    # Bounding box, same treatment (find_objects also walks per component).
    flat_pos = np.nonzero(fg)[0]
    py = (flat_pos // w).astype(np.float64)
    px = (flat_pos % w).astype(np.float64)
    out["min_x"], out["max_x"] = _reduce(np.minimum, px, np.inf), _reduce(np.maximum, px, -np.inf)
    out["min_y"], out["max_y"] = _reduce(np.minimum, py, np.inf), _reduce(np.maximum, py, -np.inf)

    # Seeding extremum per component: the pixel attaining the component's filtered
    # min (ascending) or max (descending). Found by locating the pixels that equal
    # their own component's extreme value, then keeping the first of them per
    # component -- assigning in reverse so the earliest index wins, matching the
    # C++, which keeps the first strictly-better pixel in scan order.
    ext_val = out["filt_min"] if ascending else out["filt_max"]
    ext_x = np.full(n, -1.0); ext_y = np.full(n, -1.0)
    ext_filt = np.zeros(n)
    ext_chan = {name: np.zeros(n) for name, _ in chan_values}
    hit = np.nonzero(f == ext_val[idx])[0]
    if hit.size:
        pos = np.full(n, -1, dtype=np.int64)
        pos[idx[hit[::-1]]] = hit[::-1]
        have = pos >= 0
        sel = pos[have]
        ext_x[have] = px[sel]
        ext_y[have] = py[sel]
        ext_filt[have] = f[sel]
        # Every channel is sampled at the SAME pixel, so a scale-space stack
        # reports what the seed looks like at each scale.
        for name, v in chan_values:
            ext_chan[name][have] = v[sel]
    out["ext_x"], out["ext_y"] = ext_x, ext_y
    out["ext_filtered"] = ext_filt
    for name, _ in chan_values:
        out[chan_col(name, "ext")] = ext_chan[name]
    return out


def _std(sum_: np.ndarray, sumsq: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Vectorized population std from sum / sum-of-squares (0 where area==0)."""
    area = np.where(area > 0, area, 1.0)
    mean = sum_ / area
    var = sumsq / area - mean * mean
    return np.sqrt(np.clip(var, 0.0, None))


def _relevance_base(f_m: float, f_s: float, floor: float, ceiling: float) -> float:
    numerator = float(f_s) - float(f_m)
    denominator = float(f_m) + (float(ceiling) - float(floor))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


# --------------------------------------------------------------------------- #
# Streaming assembly
# --------------------------------------------------------------------------- #
def assemble_cc(
    labels_list: Sequence[np.ndarray],
    kept_list: Sequence[Optional[Set[int]]],
    base_list: Sequence[np.ndarray],
    filt_list: Sequence[np.ndarray],
    pixel_rules: Sequence[Dict[str, Any]] = (),
    connectivity: int = 6,
    ascending: bool = True,
    relevance_enabled: bool = True,
    relevance_low_percentile: float = 0.0,
    relevance_high_percentile: float = 100.0,
    channels_list: Optional[Sequence[Sequence[Tuple[str, np.ndarray]]]] = None,
    reductions: Sequence[str] = ("mean", "min", "max", "std"),
    extremum: bool = True,
    timing: Optional[Dict[str, float]] = None,
):
    """Assemble a stack into 3D connected components.

    Args:
        labels_list: [Z] MSC merged feature-id rasters (-1 background).
        kept_list:   [Z] sets of selected MSC feature ids (None = keep all).
        base_list, filt_list: [Z] the base (original) and filtered rasters.
        pixel_rules: pixel intensity trim chain (see apply_pixel_filters).
        connectivity: 6/18/26 (drives in-plane CC + cross-slice stencil).
        channels_list: [Z] measurement channels per slice, [(name, raster)], in
                       the run's resolved slot order. None means the base channel
                       alone, which is the default StatsSpec. This is where the
                       scale-space stack enters: the GUI materializes it per
                       slice via ``mscoupon.stat_channel_images``.
        reductions:    which aggregates to emit per channel, mirroring
                       ``statistics.reductions``.
        extremum:      emit the seeding-extremum block.

    Returns dict:
        cc_labels:     [Z] per-slice CC id raster (-1 bg, 0..n_z-1).
        global_labels: [Z] per-slice GLOBAL id raster (-1 bg, 0..G-1).
        global_table:  [G] stat dicts: identity + extent, then one field per
                       (reduction, channel) named exactly as the 2D schema and
                       the CLI's global_segments.csv name them, then the extremum
                       block, then the per-slice reductions.
        n_global:      G.

    Reductions over a channel are VOXEL-pooled, matching ``SliceMatcher``: a mean
    of per-slice means would weight by slice count rather than by area. The
    per-slice (area/bbox) reductions are the ones that run across slices.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if timing is None:
        timing = {}
    Z = len(labels_list)
    if not (0.0 <= relevance_low_percentile <= relevance_high_percentile <= 100.0):
        raise ValueError("relevance percentiles must satisfy 0 <= low <= high <= 100")
    offsets = _OFFSETS.get(connectivity, _OFFSETS[6])

    # --- per-slice: select -> trim -> CC -> node stats (streaming) --------- #
    lbls: List[np.ndarray] = []               # scipy labelings (0 bg, 1..n)
    slice_offset = [0]
    node_cols: Dict[str, List[np.ndarray]] = {}
    node_z: List[np.ndarray] = []
    import time as _time
    _t = _time.perf_counter
    for k in ("mask", "cc", "node_stats"):
        timing.setdefault(k, 0.0)
    for z in range(Z):
        t = _t()
        mask = selection_mask(labels_list[z], kept_list[z] if z < len(kept_list) else None)
        mask = apply_pixel_filters(mask, base_list[z], filt_list[z], pixel_rules)
        timing["mask"] += _t() - t

        t = _t()
        lbl, n = per_slice_cc(mask, connectivity)
        timing["cc"] += _t() - t
        lbls.append(lbl)

        t = _t()
        finite_base = np.asarray(base_list[z])[np.isfinite(base_list[z])]
        if relevance_enabled and finite_base.size:
            floor = (finite_base.min() if relevance_low_percentile == 0.0 else
                     np.percentile(finite_base, relevance_low_percentile))
            ceiling = (finite_base.max() if relevance_high_percentile == 100.0 else
                       np.percentile(finite_base, relevance_high_percentile))
        else:
            floor = ceiling = 0.0
        slice_channels = (list(channels_list[z]) if channels_list is not None
                          else [("base", base_list[z])])
        st = node_stats(lbl, n, base_list[z], filt_list[z], ascending,
                        float(floor), float(ceiling), slice_channels)
        timing["node_stats"] += _t() - t
        for k, v in st.items():
            node_cols.setdefault(k, []).append(v)
        node_z.append(np.full(n, z, dtype=np.int64))
        slice_offset.append(slice_offset[-1] + n)

    N = slice_offset[-1]
    cols = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in node_cols.items()}
    z_of = np.concatenate(node_z) if node_z else np.zeros(0, dtype=np.int64)

    # --- cross-slice edges (vectorized 6/18/26 stencil, +z only) ----------- #
    _t_edges = _t()
    pa_parts: List[np.ndarray] = []
    pb_parts: List[np.ndarray] = []
    for z in range(Z - 1):
        a, b = lbls[z], lbls[z + 1]
        if a.max(initial=0) == 0 or b.max(initial=0) == 0:
            continue
        off_a, off_b = slice_offset[z], slice_offset[z + 1]
        for dy, dx in offsets:
            bs = b if (dy == 0 and dx == 0) else _shift(b, dy, dx)
            m = (a > 0) & (bs > 0)
            if not m.any():
                continue
            ga = off_a + (a[m] - 1).astype(np.int64)
            gb = off_b + (bs[m] - 1).astype(np.int64)
            key = np.unique(ga * np.int64(N) + gb)
            pa_parts.append(key // N)
            pb_parts.append(key % N)

    timing["edges"] = _t() - _t_edges

    # --- resolve identity + first-seen (appearance-order) global ids ------- #
    _t_cc3d = _t()
    if N == 0:
        comp = np.zeros(0, dtype=np.int64)
        n_global = 0
    else:
        if pa_parts:
            pa = np.concatenate(pa_parts); pb = np.concatenate(pb_parts)
            g = coo_matrix((np.ones(len(pa), dtype=np.int8), (pa, pb)), shape=(N, N))
        else:
            g = coo_matrix((N, N), dtype=np.int8)
        n_comp, raw = connected_components(g, directed=False, connection="weak")
        # Renumber components by the smallest node id they contain (= earliest
        # slice / lowest local id) so global ids come out in first-seen order.
        min_node = np.full(n_comp, N, dtype=np.int64)
        np.minimum.at(min_node, raw, np.arange(N, dtype=np.int64))
        order = np.argsort(np.argsort(min_node, kind="stable"), kind="stable")
        comp = order[raw].astype(np.int64)
        n_global = int(n_comp)

    timing["cc3d"] = _t() - _t_cc3d

    # --- node-level stats reduction by global id --------------------------- #
    _t_reduce = _t()

    def reduce_sum(col):
        return np.bincount(comp, weights=cols[col], minlength=n_global) if N else np.zeros(n_global)

    def reduce_min(col):
        acc = np.full(n_global, np.inf)
        if N:
            np.minimum.at(acc, comp, cols[col])
        return acc

    def reduce_max(col):
        acc = np.full(n_global, -np.inf)
        if N:
            np.maximum.at(acc, comp, cols[col])
        return acc

    area_g = reduce_sum("area")
    # Channel names in slot order, taken from the first slice that had any.
    channel_names: List[str] = []
    if channels_list is not None:
        seen = set()
        for per_slice_channels in channels_list:
            for name, _ in per_slice_channels:
                if name not in seen:
                    seen.add(name)
                    channel_names.append(name)
    else:
        channel_names = ["base"]
    chan_g = {}
    for name in channel_names:
        chan_g[name] = {
            "sum": reduce_sum(chan_col(name, "sum")),
            "sumsq": reduce_sum(chan_col(name, "sumsq")),
            "min": reduce_min(chan_col(name, "min")),
            "max": reduce_max(chan_col(name, "max")),
        }
    relevance_floor_g = reduce_min("base_relevance_floor")
    relevance_ceiling_g = reduce_max("base_relevance_ceiling")
    filt_min_g, filt_max_g = reduce_min("filt_min"), reduce_max("filt_max")
    min_x_g, max_x_g = reduce_min("min_x"), reduce_max("max_x")
    min_y_g, max_y_g = reduce_min("min_y"), reduce_max("max_y")
    min_z_g = np.full(n_global, np.inf); max_z_g = np.full(n_global, -np.inf)
    if N:
        np.minimum.at(min_z_g, comp, z_of.astype(np.float64))
        np.maximum.at(max_z_g, comp, z_of.astype(np.float64))
    # distinct slices per global id
    num_slices_g = np.zeros(n_global, dtype=np.int64)
    if N:
        gz = np.unique(comp * np.int64(Z) + z_of)
        num_slices_g = np.bincount((gz // Z).astype(np.int64), minlength=n_global)

    area_safe = np.where(area_g > 0, area_g, 1.0)

    # Seeding extremum: pick the constituent NODE whose ext_filtered is most
    # extreme, then take its whole tuple. Reducing each ext_* column separately
    # would pair a position from one slice with a value from another.
    ext_x_g = np.full(n_global, -1.0); ext_y_g = np.full(n_global, -1.0)
    ext_z_g = np.full(n_global, -1, dtype=np.int64)
    ext_filt_g = np.zeros(n_global)
    ext_chan_g = {name: np.zeros(n_global) for name in channel_names}
    if N and "ext_filtered" in cols:
        key = cols["ext_filtered"] if ascending else -cols["ext_filtered"]
        best = np.full(n_global, np.inf)
        np.minimum.at(best, comp, key)
        # First node attaining its component's best key wins ties, matching the
        # C++ (which keeps the first strictly-better node in slice-major order).
        winner = np.full(n_global, -1, dtype=np.int64)
        is_best = key <= best[comp]
        for node in np.nonzero(is_best)[0][::-1]:
            winner[comp[node]] = node
        take = winner >= 0
        w = winner[take]
        ext_x_g[take] = cols["ext_x"][w]
        ext_y_g[take] = cols["ext_y"][w]
        ext_z_g[take] = z_of[w]
        ext_filt_g[take] = cols["ext_filtered"][w]
        for name in channel_names:
            key_col = chan_col(name, "ext")
            if key_col in cols:
                ext_chan_g[name][take] = cols[key_col][w]

    # Per-slice reductions: how the footprint varies across the slices a feature
    # spans. Unlike the field statistics, which pool voxels.
    per_slice_cols = {"area": cols["area"] if N else np.zeros(0),
                      "bbox_w": (cols["max_x"] - cols["min_x"] + 1) if N else np.zeros(0),
                      "bbox_h": (cols["max_y"] - cols["min_y"] + 1) if N else np.zeros(0)}
    ps: Dict[str, np.ndarray] = {}
    node_count = np.bincount(comp, minlength=n_global) if N else np.zeros(n_global)
    count_safe = np.where(node_count > 0, node_count, 1.0)
    for name, vals in per_slice_cols.items():
        if not N:
            for r in ("mean", "min", "max", "std"):
                ps[f"{name}_{r}"] = np.zeros(n_global)
            continue
        s1 = np.bincount(comp, weights=vals, minlength=n_global)
        s2 = np.bincount(comp, weights=vals * vals, minlength=n_global)
        lo = np.full(n_global, np.inf); np.minimum.at(lo, comp, vals)
        hi = np.full(n_global, -np.inf); np.maximum.at(hi, comp, vals)
        mean = s1 / count_safe
        ps[f"{name}_mean"] = mean
        ps[f"{name}_min"] = np.where(np.isfinite(lo), lo, 0.0)
        ps[f"{name}_max"] = np.where(np.isfinite(hi), hi, 0.0)
        ps[f"{name}_std"] = np.sqrt(np.clip(s2 / count_safe - mean * mean, 0.0, None))
    # Precompute each channel's derived reductions once for the whole table
    # rather than per row.
    want = set(reductions)
    chan_fields: Dict[str, np.ndarray] = {}
    for name in channel_names:
        acc = chan_g[name]
        if "mean" in want:
            chan_fields[f"mean_{name}"] = acc["sum"] / area_safe
        if "min" in want:
            chan_fields[f"min_{name}"] = acc["min"]
        if "max" in want:
            chan_fields[f"max_{name}"] = acc["max"]
        if "std" in want:
            chan_fields[f"std_{name}"] = _std(acc["sum"], acc["sumsq"], area_g)

    base_acc = chan_g.get("base")
    global_table: List[dict] = []
    for gid in range(n_global):
        a = float(area_g[gid])
        row = {
            "global_id": gid,
            "area": a, "voxel_count": a,
            "num_slices": int(num_slices_g[gid]),
            "bbox_w": int(max_x_g[gid] - min_x_g[gid] + 1) if a else 0,
            "bbox_h": int(max_y_g[gid] - min_y_g[gid] + 1) if a else 0,
            "bbox_d": int(max_z_g[gid] - min_z_g[gid] + 1) if a else 0,
        }
        for field, values in chan_fields.items():
            row[field] = float(values[gid])
        if relevance_enabled and base_acc is not None:
            row["relevance_base"] = _relevance_base(
                base_acc["min"][gid], base_acc["max"][gid],
                relevance_floor_g[gid], relevance_ceiling_g[gid])
        if extremum:
            row["ext_x"] = float(ext_x_g[gid])
            row["ext_y"] = float(ext_y_g[gid])
            row["ext_z"] = int(ext_z_g[gid])
            for name in channel_names:
                # `filtered` is reported once, as ext_filtered below; emitting it
                # here too would produce the same key twice. Keeping that key last
                # matches the C++ schema's column order.
                if name != "filtered":
                    row[f"ext_{name}"] = float(ext_chan_g[name][gid])
            row["ext_filtered"] = float(ext_filt_g[gid])
        row.update({k: float(v[gid]) for k, v in ps.items()})
        global_table.append(row)

    timing["reduce"] = _t() - _t_reduce

    # --- output rasters: per-slice CC (-1 bg) and global id (-1 bg) --------- #
    _t_out = _t()
    cc_labels: List[np.ndarray] = []
    global_labels: List[np.ndarray] = []
    for z in range(Z):
        lbl = lbls[z]
        cc_labels.append(np.where(lbl > 0, lbl - 1, -1))
        if lbl.max(initial=0) == 0:
            global_labels.append(np.full(lbl.shape, -1, dtype=np.int64))
            continue
        node_gid = comp[slice_offset[z]:slice_offset[z + 1]]   # local cc id -> gid
        gmap = np.concatenate([node_gid, np.array([-1])])       # -1 sentinel at index n
        global_labels.append(gmap[np.where(lbl > 0, lbl - 1, len(node_gid))])

    timing["rasters"] = _t() - _t_out

    return {"cc_labels": cc_labels, "global_labels": global_labels,
            "global_table": global_table, "n_global": n_global}


def _selftest():
    # Feature 1 (value 1, cols 0-1) spans all 3 slices; feature 2 (value 2, col 3)
    # only slice 0. A background gap (col 2) keeps them as DISTINCT in-plane CCs
    # (binary-foreground CC merges only touching pixels). base == filt here.
    z0 = np.array([[1, 1, -1, 2], [1, 1, -1, 2]], dtype=np.int32)
    z1 = np.array([[1, 1, -1, -1], [1, 1, -1, -1]], dtype=np.int32)
    z2 = np.array([[1, 1, -1, -1], [1, 1, -1, -1]], dtype=np.int32)
    labels = [z0, z1, z2]
    base = [z.astype(np.float32) for z in labels]
    kept = [None, None, None]
    out = assemble_cc(labels, kept, base, base, pixel_rules=(), connectivity=6)
    g0 = out["global_labels"][0]
    assert g0[0, 0] == g0[0, 1] and g0[0, 0] != g0[0, 3], "distinct components on slice 0"
    assert out["global_labels"][1][0, 0] == g0[0, 0], "feature 1 links across slices"
    assert out["n_global"] == 2, out["n_global"]
    big = out["global_table"][int(g0[0, 0])]
    assert big["voxel_count"] == 12 and big["num_slices"] == 3 and big["bbox_d"] == 3, big
    assert big["mean_base"] == 1.0 and big["bbox_w"] == 2, big
    # First-seen numbering: feature 1 (min node id 0) is global id 0.
    assert g0[0, 0] == 0, "first-seen component is global id 0"

    # Pixel trim: keep base >= 2 drops feature-1 pixels (value 1) -> only feature 2.
    rules = [{"channel": "base", "mode": "keep", "op": "ge", "value": 2.0}]
    out2 = assemble_cc(labels, kept, base, base, pixel_rules=rules, connectivity=6)
    assert out2["n_global"] == 1, out2["n_global"]      # only feature-2 blob survives
    assert (out2["global_labels"][0] >= 0).sum() == 2, "trim kept only value>=2 pixels"

    # Per-slice selection: keep only feature 2 (id 2) on slice 0.
    out3 = assemble_cc(labels, [{2}, set(), set()], base, base, connectivity=6)
    assert out3["n_global"] == 1 and (out3["global_labels"][0] >= 0).sum() == 2

    print("assembly selftest OK:", out["n_global"], "components; trim + selection verified")


if __name__ == "__main__":
    _selftest()
