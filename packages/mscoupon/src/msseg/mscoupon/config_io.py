"""Config model + JSON (de)serialization for the mscoupon viewer.

The viewer's workflow (filter chain + MSC params + per-slice size gate +
feature-query chain + 3D assembly) is serialized to the exact ``AppConfig`` JSON
schema the C++ CLI parses (see ``packages/mscoupon/lib/mscoupon/config.{hpp,cpp}``),
so ``mscoupon --config exported.json`` reproduces the viewer's output. Each
subsequence is exported as its own config (its own 3D stack) via an explicit
``input.files`` list.

This module is pure Python (no compiled/large deps) so it is unit-testable and
importable in headless environments.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence


# --------------------------------------------------------------------------- #
# Filter chain schema: operation -> ordered [(param, kind, default), ...].
# `kind` is one of "float" | "int" | "bool" | "choice:<a>,<b>,..." and drives
# both widget creation (app.py) and JSON typing here.
# --------------------------------------------------------------------------- #
FILTER_SCHEMA: Dict[str, List[tuple]] = {
    "none": [],
    "blur": [("sigma", "float", 1.0)],
    "derivative": [("sigma", "float", 1.0), ("order_x", "int", 1), ("order_y", "int", 0)],
    "laplacian": [("sigma", "float", 1.0)],
    "zero_crossings": [("sigma", "float", 1.0)],
    "hessian_eigenvalues": [("sigma", "float", 1.0),
                            ("component", "choice:largest,middle,smallest", "largest")],
    "structure_eigenvalues": [("smoothing_sigma", "float", 1.0),
                              ("integration_sigma", "float", 2.0),
                              ("component", "choice:largest,middle,smallest", "largest")],
    "edges": [("sigma", "float", 1.0), ("suppress_nonmax", "bool", False),
              ("low_threshold", "float", 0.0), ("high_threshold", "float", 0.0),
              ("output", "choice:magnitude,mask", "magnitude")],
    "erode": [("radius", "int", 1)],
    "dilate": [("radius", "int", 1)],
    "open": [("radius", "int", 1)],
    "close": [("radius", "int", 1)],
    "label_components": [("threshold", "float", 0.0)],
    # Two-point normalization. Rewrites the channel onto a [0,1] scale set by
    # two measured landmarks, so a threshold can be written once as a normalized
    # number ("0.7" == 0.3*low + 0.7*high) and stay meaningful across a stack
    # whose absolute intensity drifts. Implemented in mscoupon (not core) since
    # it needs this package's intensity measures -- see lib/mscoupon/normalize.*.
    # low_from/high_from blank = the method's default landmark pair; low/high
    # blank = no manual pair. Blanks are dropped on export (see filters_to_json)
    # rather than exported as "" or 0.0, which the CLI would read as real values.
    # omit_value is the stack's no-data sentinel -- usually 0, but a stack may pad
    # with any constant, and dropping the wrong value leaves that plateau in the
    # fit as a spurious population. It is "nullfloat" rather than "optfloat"
    # because blank has to reach the CLI as an explicit null ("keep every pixel"),
    # not be dropped and fall back to the default of 0.
    "normalize": [("method", "choice:gmm,histogram,regions,manual", "gmm"),
                  ("low_from", "str", ""), ("high_from", "str", ""),
                  ("low", "optfloat", ""), ("high", "optfloat", ""),
                  ("omit_value", "nullfloat", 0.0),
                  ("downsample_factor", "int", 1),
                  ("clamp", "bool", False)],
}

FILTER_OPERATIONS = list(FILTER_SCHEMA.keys())

# Which landmark names each method can pick from, for the GUI's pickers. The
# empty default means "use the method's own default pair" (mu_1/mu_2,
# peak_low/peak_high, air_median/metal_median).
NORMALIZE_LANDMARKS = {
    "gmm": ["mu_1", "mu_2", "hard_mean_1", "hard_mean_2", "median_1", "median_2",
            "mode_1", "mode_2"],
    "histogram": ["peak_low", "peak_high", "hist_lo", "hist_hi", "min", "max",
                  "p1_0", "p5_0", "p50_0", "p95_0", "p99_0"],
    "regions": ["air_median", "metal_median", "air_mean", "metal_mean",
                "air_min", "air_max", "metal_min", "metal_max"],
    "manual": [],
}

NORMALIZE_METHODS = list(NORMALIZE_LANDMARKS.keys())

# Per-slice selection fields. These gate 2D per-slice MSC merged regions (NOT 3D
# features).
#
# The authoritative list is `mscoupon::feature_row` in query.cpp, and it now
# depends on the statistics spec, so ask the extension rather than mirroring it:
# a hand-kept copy drifts silently, and the failure mode is the GUI offering a
# field the CLI then rejects. The literal below is only the fallback for a
# headless environment with no compiled extension (the GUI --selftest), and lists
# the default spec's fields.
_DEFAULT_QUERY_FIELDS = [
    "area", "bbox_h", "bbox_w",
    # The region's seeding critical point (minimum for ascending manifolds,
    # maximum for descending): its position and the two channels sampled there.
    # ext_base answers "is the well bottom actually dark?", which mean_base
    # cannot -- a shallow dip inside metal can share a void's mean.
    "ext_base", "ext_filtered", "ext_x", "ext_y",
    "max_base", "max_x", "max_y", "mean_base", "min_base", "min_x", "min_y",
    "std_base",
]


def query_fields(params_json: str = "") -> List[str]:
    """Field names `feature_filters` may name under `params_json`'s statistics
    block, straight from the C++ schema. Falls back to the default spec's list
    when the extension is not built.

    `feature_id` is dropped: the row carries it so a caller can identify the
    feature, but selecting on an id is not a statistic and offering it in the
    dropdown only invites configs that break the moment persistence changes."""
    try:
        from msseg import mscoupon as engine
        names = list(engine.feature_fields(params_json))
    except Exception:      # extension absent, or too old to know the call
        names = list(_DEFAULT_QUERY_FIELDS)
    return [n for n in names if n != "feature_id"]


# Back-compat alias for the default spec. Prefer query_fields() so a non-default
# `statistics` block is reflected.
QUERY_FIELDS = _DEFAULT_QUERY_FIELDS
QUERY_OPS = ["lt", "le", "gt", "ge", "eq", "between"]

# Pixel intensity filter (trim): per-pixel keep/omit by a channel value threshold,
# applied to the selected raster before per-slice connected components. Must match
# the C++ mscoupon::PixelFilter / apply_pixel_filters.
PIXEL_CHANNELS = ["base", "filtered"]
PIXEL_MODES = ["keep", "omit"]     # keep = drop pixels that FAIL; omit = drop pixels that PASS
PIXEL_OPS = ["lt", "le", "gt", "ge"]


def _is_nullfloat(operation: str, param: str) -> bool:
    """True for params whose blank must export as JSON null, not be dropped."""
    for name, kind, _default in FILTER_SCHEMA.get(operation, ()):
        if name == param:
            return kind == "nullfloat"
    return False


def filters_to_json(filters: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a list of {"operation":..., "params":{...}} filter cards,
    dropping "none" stages (they are no-ops / the trailing add-row)."""
    out: List[Dict[str, Any]] = []
    for f in filters:
        op = f.get("operation", "none")
        if op == "none":
            continue
        # Blank string params mean "unset" (e.g. normalize's low_from/high_from,
        # where the empty value asks for the method's default landmark pair).
        # Emitting "" would override that default with an unknown name.
        raw = dict(f.get("params", {}))
        params = {}
        for k, v in raw.items():
            if _is_nullfloat(op, k):
                # Blank here means "keep every pixel", which is a real setting and
                # must survive as null -- dropping it would silently restore the
                # default sentinel instead.
                params[k] = None if v == "" else float(v)
            elif v != "":
                params[k] = v
        out.append({"operation": op, "params": params})
    return out


def queries_to_json(queries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a list of {"field","op","value"[,"value2"]} query cards,
    dropping incomplete rows (empty field)."""
    out: List[Dict[str, Any]] = []
    for q in queries:
        field = q.get("field", "")
        if not field:
            continue
        row = {"field": field, "op": q.get("op", "gt"), "value": float(q.get("value", 0.0))}
        if q.get("op") == "between":
            row["value2"] = float(q.get("value2", 0.0))
        out.append(row)
    return out


def pixel_filters_to_json(rules: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a list of pixel-filter cards
    {"channel","mode","op","value"}, dropping incomplete rows (empty channel)."""
    out: List[Dict[str, Any]] = []
    for r in rules:
        channel = r.get("channel", "")
        if not channel:
            continue
        out.append({
            "channel": channel,
            "mode": r.get("mode", "keep"),
            "op": r.get("op", "gt"),
            "value": float(r.get("value", 0.0)),
        })
    return out


def build_config(
    *,
    files: Sequence[str],
    output_folder: str,
    filters: Sequence[Dict[str, Any]],
    base_filters: Sequence[Dict[str, Any]] = (),
    persistence_percent: Optional[float] = 10.0,
    persistence_absolute: Optional[float] = None,
    manifold: str = "ascending",
    accurate: bool = False,
    extremum_sample_radius: int = 0,
    min_area: Optional[int] = None,
    feature_filters: Sequence[Dict[str, Any]] = (),
    pixel_filters: Sequence[Dict[str, Any]] = (),
    connectivity: int = 6,
    matching_enabled: bool = True,
    cores_per_slice: int = 1,
    concurrent_slices: int = 1,
    folder: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the AppConfig-shaped dict for one subsequence (an explicit file list).

    `folder` defaults to the common parent directory of `files`; `input.files`
    carries the ordered, absolute selection so the CLI reproduces this stack.

    `filters` builds the topology field the MSC runs on; `base_filters` is the
    optional chain applied to the base channel -- the raster statistics and
    pixel filters are measured against. Both derive from the raw slice, so an
    empty `base_filters` reproduces the pre-normalization behaviour exactly.

    `cores_per_slice` / `concurrent_slices` map onto the CLI's parallelism: the
    former sets `execution.threads_per_slice` (OpenMP threads for filtering etc.)
    AND, when > 1, `msc.compute_algorithm='partitioned'` + `msc.requested_parallelism`
    (MSCEER's discrete gradient / partitioned MSC / manifold labeling); the latter
    sets `execution.concurrent_slices` (compute lanes = whole slices at once).
    """
    files = list(files)
    if folder is None:
        folder = os.path.dirname(files[0]) if files else output_folder

    msc: Dict[str, Any] = {
        "manifold": manifold,
        "accurate_ascending": bool(accurate),
        "accurate_descending": bool(accurate),
    }
    if persistence_absolute is not None:
        msc["persistence_absolute"] = float(persistence_absolute)
    else:
        msc["persistence_percent"] = float(persistence_percent if persistence_percent is not None else 10.0)
    # Only emitted when non-default, so existing exported configs are unchanged.
    if int(extremum_sample_radius) > 0:
        msc["extremum_sample_radius"] = int(extremum_sample_radius)
    cores = max(1, int(cores_per_slice))
    if cores > 1:
        msc["compute_algorithm"] = "partitioned"
        msc["requested_parallelism"] = cores

    cfg: Dict[str, Any] = {
        "input": {"folder": folder, "files": files},
        "output": {"folder": output_folder},
        "filters": filters_to_json(filters),
        "msc": msc,
        # per-slice selection (2D merged-region gate) + pixel intensity trim
        "feature_filters": queries_to_json(feature_filters),
        "pixel_filters": pixel_filters_to_json(pixel_filters),
        # connectivity drives both the in-plane 2D CC and the cross-slice stencil
        "assembly": {"connectivity": int(connectivity)},
        "matching": {"enabled": bool(matching_enabled),
                     "write_cc_labels": True, "write_global_labels": True},
        "execution": {"threads_per_slice": cores,
                      "concurrent_slices": max(1, int(concurrent_slices))},
    }
    # Base channel chain (typically a `normalize` stage). Emitted only when
    # non-empty so an unnormalized workflow exports the same config as before.
    base_json = filters_to_json(base_filters)
    if base_json:
        cfg["base_filters"] = base_json
    if min_area is not None:
        cfg["segments"] = {"min_area": int(min_area)}
    return cfg


def dump_config(cfg: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
