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
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple


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
    # The thresholds are optfloat: BLANK means "no threshold" (diffg's
    # EdgeOptions are std::optional -- absent skips the hysteresis mask). They
    # were plain floats defaulting to 0.0 once, which made "unset"
    # inexpressible: an explicit 0.0 threshold on output=mask passes every
    # pixel and yields a CONSTANT field (a one-region MSC).
    "edges": [("sigma", "float", 1.0), ("suppress_nonmax", "bool", False),
              ("low_threshold", "optfloat", ""), ("high_threshold", "optfloat", ""),
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
    "relevance_base", "std_base",
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


# The pseudo-channel the geometry columns (area, bbox_*, min_x, ...) live under,
# so one [channel][reduction] pair of dropdowns covers every field.
GEOMETRY_CHANNEL = "geometry"


def feature_schema(params_json: str = "") -> List[Dict[str, str]]:
    """Per-column schema as {name, channel, reduction}, in table order.

    This is what the GUI's two-level [channel][reduction] pickers are built from.
    Parsing a field name would be wrong as well as fragile: "min_x" is a geometry
    column, not the `min` reduction of a channel called `x`.

    Falls back to deriving a flat geometry-only view from query_fields() when the
    extension is not built (the headless --selftest path)."""
    try:
        from msseg import mscoupon as engine
        return [dict(e) for e in engine.feature_schema(params_json)]
    except Exception:
        return [{"name": n, "channel": "", "reduction": ""}
                for n in query_fields(params_json)]


def stat_channels(params_json: str = "") -> List[Dict[str, Any]]:
    """The measurement channels `params_json` resolves to, in slot order:
    {name, kind, sigma}. `base`/`filtered` are the two rasters the pipeline
    already builds; everything else is a derived scale-space response measured on
    the base channel."""
    try:
        from msseg import mscoupon as engine
        return [dict(c) for c in engine.stat_channels(params_json)]
    except Exception:
        return [{"name": "base", "kind": "base", "sigma": 0.0}]


def channels_and_reductions(params_json: str = "") -> Tuple[List[str], Dict[str, List[str]]]:
    """(channel order, reductions available per channel) for the GUI pickers.

    Geometry columns are grouped under the pseudo-channel `GEOMETRY_CHANNEL` so a
    single pair of dropdowns can reach every selectable field."""
    order: List[str] = []
    by_channel: Dict[str, List[str]] = {}
    for entry in feature_schema(params_json):
        name = entry.get("name", "")
        if name == "feature_id":
            continue
        channel = entry.get("channel") or GEOMETRY_CHANNEL
        reduction = entry.get("reduction") or name
        if channel not in by_channel:
            by_channel[channel] = []
            order.append(channel)
        if reduction not in by_channel[channel]:
            by_channel[channel].append(reduction)
    # Geometry reads better last: a user picks a channel far more often.
    if GEOMETRY_CHANNEL in order:
        order = [c for c in order if c != GEOMETRY_CHANNEL] + [GEOMETRY_CHANNEL]
    return order, by_channel


def compose_field(channel: str, reduction: str) -> str:
    """The field name a (channel, reduction) pair selects."""
    if channel == GEOMETRY_CHANNEL or not channel:
        return reduction
    return f"{reduction}_{channel}"


def split_field(name: str, params_json: str = "") -> Tuple[str, str]:
    """Inverse of compose_field, resolved against the schema rather than parsed."""
    for entry in feature_schema(params_json):
        if entry.get("name") == name:
            return (entry.get("channel") or GEOMETRY_CHANNEL,
                    entry.get("reduction") or name)
    return (GEOMETRY_CHANNEL, name)


# Back-compat alias for the default spec. Prefer query_fields() so a non-default
# `statistics` block is reflected.
QUERY_FIELDS = _DEFAULT_QUERY_FIELDS
QUERY_OPS = ["lt", "le", "gt", "ge", "eq", "between"]

# Derived measurement-channel kinds a `statistics.channels[]` entry may name.
# `base`/`filtered` are the two rasters the pipeline already builds and are
# spelled as bare strings instead. Must match msseg::derived_channel_kinds().
DERIVED_CHANNEL_KINDS = ["blur", "edges", "gradmag", "laplacian", "hessian"]
STAT_REDUCTIONS = ["mean", "min", "max", "std"]

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
                # default sentinel instead. None passes through so an
                # already-normalized list is a fixed point (profiles store the
                # normalized form and re-normalize on every snapshot).
                params[k] = None if v in ("", None) else float(v)
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


def statistics_to_json(channels: Sequence[Dict[str, Any]],
                       reductions: Sequence[str],
                       extremum: bool = True,
                       extremum_sample_radius: int = 0,
                       relevance: bool = True) -> Dict[str, Any]:
    """The `statistics` block for a config.

    `channels` mirrors what the CLI parses: a bare string for `base`/`filtered`,
    or a dict {kind, sigmas, [sort_by_absolute_value], [name]} for a derived
    scale-space channel. Emitted only in the shape the parser accepts, so a GUI
    export and a hand-written config are the same document.
    """
    out: List[Any] = []
    for entry in channels:
        kind = str(entry.get("kind") or "")
        if kind in ("base", "filtered"):
            out.append(kind)
            continue
        if kind not in DERIVED_CHANNEL_KINDS:
            continue
        sigmas = [float(v) for v in entry.get("sigmas") or [] if _is_number(v)]
        if not sigmas:
            continue
        item: Dict[str, Any] = {"kind": kind, "sigmas": sigmas}
        if kind == "hessian" and not entry.get("sort_by_absolute_value", True):
            item["sort_by_absolute_value"] = False
        if entry.get("name"):
            item["name"] = str(entry["name"])
        out.append(item)

    block: Dict[str, Any] = {
        "channels": out,
        "reductions": [r for r in reductions if r in STAT_REDUCTIONS],
        "extremum": bool(extremum),
    }
    if extremum_sample_radius > 0:
        block["extremum_sample_radius"] = int(extremum_sample_radius)
    # True is the core/CLI default. Emit only the labeler's lean opt-out so
    # existing viewer profiles and exported configs retain their old shape.
    if not relevance:
        block["relevance"] = False
    return block


def statistics_from_json(doc: Any, notes: Optional[List[str]] = None) -> Dict[str, Any]:
    """Inverse of statistics_to_json, total (never raises) like the rest of the
    read side. Returns {channels, reductions, extremum, extremum_sample_radius,
    relevance} with `channels` always in the dict form the GUI edits."""
    block = _as_dict(doc)
    raw = block.get("channels")
    channels: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                if item in ("base", "filtered"):
                    channels.append({"kind": item})
                else:
                    _note(notes, f"statistics.channels: unknown channel {item!r} - dropped")
                continue
            entry = _as_dict(item)
            kind = str(entry.get("kind") or "")
            if kind in ("base", "filtered"):
                channels.append({"kind": kind})
                continue
            if kind not in DERIVED_CHANNEL_KINDS:
                _note(notes, f"statistics.channels: unknown kind {kind!r} - dropped")
                continue
            sigmas_raw = entry.get("sigmas")
            if sigmas_raw is None and _is_number(entry.get("sigma")):
                sigmas_raw = [entry.get("sigma")]
            sigmas = [float(v) for v in (sigmas_raw or []) if _is_number(v) and float(v) > 0.0]
            if not sigmas:
                _note(notes, f"statistics.channels: {kind!r} has no positive sigma - dropped")
                continue
            out: Dict[str, Any] = {"kind": kind, "sigmas": sigmas}
            if kind == "hessian":
                out["sort_by_absolute_value"] = bool(
                    entry.get("sort_by_absolute_value", True))
            if entry.get("name"):
                out["name"] = str(entry["name"])
            channels.append(out)
    else:
        channels = [{"kind": "base"}]

    reductions_raw = block.get("reductions")
    if isinstance(reductions_raw, list):
        reductions = [str(r) for r in reductions_raw if str(r) in STAT_REDUCTIONS]
    else:
        reductions = list(STAT_REDUCTIONS)

    raw_relevance = block.get("relevance", True)
    relevance = (bool(raw_relevance.get("enabled", True))
                 if isinstance(raw_relevance, dict) else bool(raw_relevance))
    return {
        "channels": channels,
        "reductions": reductions,
        "extremum": bool(block.get("extremum", True)),
        "extremum_sample_radius": _as_int(block.get("extremum_sample_radius"), 0),
        "relevance": relevance,
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
    stat_channels: Optional[Sequence[Dict[str, Any]]] = None,
    stat_reductions: Optional[Sequence[str]] = None,
    stat_extremum: bool = True,
    stat_relevance: bool = True,
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
    # Measurement channels + reductions. Emitted only when they differ from the
    # default spec (base channel, all reductions, extremum on), so a workflow
    # that never touched the statistics panel exports exactly what it did before.
    if stat_channels is not None or stat_reductions is not None or not stat_relevance:
        channels = list(stat_channels) if stat_channels is not None else [{"kind": "base"}]
        reductions = (list(stat_reductions) if stat_reductions is not None
                      else list(STAT_REDUCTIONS))
        block = statistics_to_json(channels, reductions, stat_extremum,
                                   int(extremum_sample_radius), stat_relevance)
        default = statistics_to_json([{"kind": "base"}], STAT_REDUCTIONS, True, 0)
        if block != default:
            cfg["statistics"] = block
    if min_area is not None:
        cfg["segments"] = {"min_area": int(min_area)}
    return cfg


def dump_config(cfg: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)


# --------------------------------------------------------------------------- #
# Read side: JSON -> GUI state.
#
# The inverse of the writers above, so the viewer can reload a config instead of
# re-entering every parameter. It is deliberately total: a config may be
# hand-edited, may predate a schema change, or may name a statistic the current
# `statistics` spec no longer offers, and the viewer must survive all of it. Any
# value it cannot use is replaced by the schema default (or its row dropped) and
# described in `notes`, which the caller shows in the status bar -- never an
# exception, which in a Tk callback surfaces as a traceback and a dead button.
# --------------------------------------------------------------------------- #
_UNSET = object()


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _note(notes: Optional[List[str]], msg: str) -> None:
    if notes is not None:
        notes.append(msg)


def _as_float(value: Any, default: Any) -> Any:
    # bool is an int subclass, but a JSON `true` where a number belongs is a
    # mistake, not the number 1.
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: Any) -> Any:
    if isinstance(value, bool):
        return default
    try:
        return int(float(value))       # tolerate "3" and 3.0 alike
    except (TypeError, ValueError):
        return default


def _opt_float(value: Any) -> Optional[float]:
    """A number, or None when the key was absent/null/unusable."""
    return None if value is None else _as_float(value, None)


def filter_params_from_json(operation: str, params: Any,
                            notes: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """One GUI filter card from a JSON stage, or None for an unknown operation.

    Params are merged over FILTER_SCHEMA's defaults so every row the operation
    declares renders with a value -- `_build_param_row` would otherwise fill them
    in silently and the card would disagree with the file it came from.
    """
    schema = FILTER_SCHEMA.get(operation)
    if schema is None:
        _note(notes, f"unknown filter operation {operation!r} - stage skipped")
        return None
    raw = _as_dict(params)
    known = {name for name, _kind, _default in schema}
    for key in raw:
        if key not in known:
            _note(notes, f"{operation}: ignoring unknown param {key!r}")

    out: Dict[str, Any] = {}
    for name, kind, default in schema:
        if name not in raw:
            out[name] = default
            continue
        value = raw[name]
        if kind == "bool":
            out[name] = bool(value)
        elif kind.startswith("choice:"):
            choices = kind.split(":", 1)[1].split(",")
            text = str(value)
            if text in choices:
                out[name] = text
            else:
                out[name] = default
                _note(notes, f"{operation}.{name}: {text!r} is not one of "
                             f"{', '.join(choices)} - using {default!r}")
        elif kind == "str":
            out[name] = "" if value is None else str(value)
        elif kind in ("optfloat", "nullfloat"):
            # Blank is a real state in the GUI, and the two kinds mean different
            # things by it: optfloat blank is "unset" (dropped on export),
            # nullfloat blank is "keep every pixel" (exported as an explicit
            # null). So an explicit null here must come back BLANK, while an
            # ABSENT key falls through to the schema default above -- collapsing
            # the two would silently turn "keep every pixel" into the default
            # sentinel of 0 the next time the config is written.
            if value is None or value == "":
                out[name] = ""
            else:
                num = _as_float(value, _UNSET)
                if num is _UNSET:
                    out[name] = default
                    _note(notes, f"{operation}.{name}: {value!r} is not a number "
                                 f"- using the default")
                else:
                    out[name] = num
        else:                                   # float | int
            num = (_as_int if kind == "int" else _as_float)(value, _UNSET)
            if num is _UNSET:
                out[name] = default
                _note(notes, f"{operation}.{name}: {value!r} is not a number "
                             f"- using {default!r}")
            else:
                out[name] = num
    return {"operation": operation, "params": out}


def filters_from_json(items: Any,
                      notes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Inverse of `filters_to_json`: GUI filter cards, WITHOUT the trailing blank
    "none" card (the caller appends it -- only the GUI needs an add-row)."""
    out: List[Dict[str, Any]] = []
    for item in _as_list(items):
        if not isinstance(item, dict):
            _note(notes, "filter stage is not an object - skipped")
            continue
        op = item.get("operation", "none")
        op = "none" if op is None else str(op)
        if op == "none":
            continue
        card = filter_params_from_json(op, item.get("params"), notes)
        if card is not None:
            if op == "edges":
                # Legacy migration: the GUI could not express "no threshold"
                # and always wrote 0.0. As an explicit mask threshold, 0.0
                # passes every pixel (constant field), which is never what a
                # saved 0.0 meant -- read it back as unset.
                for k in ("low_threshold", "high_threshold"):
                    if card["params"].get(k) == 0.0:
                        card["params"][k] = ""
            out.append(card)
    return out


def queries_from_json(items: Any, fields: Optional[Sequence[str]] = None,
                      notes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Inverse of `queries_to_json`.

    A row naming a field outside `fields` (when given) is DROPPED rather than
    loaded: the CLI rejects an unknown `feature_filters[].field` outright, so
    keeping it only defers the failure to the next run -- with the field no
    longer in the dropdown that would explain it.
    """
    allowed = set(fields) if fields is not None else None
    out: List[Dict[str, Any]] = []
    for item in _as_list(items):
        if not isinstance(item, dict):
            _note(notes, "feature filter is not an object - skipped")
            continue
        field = str(item.get("field") or "")
        if not field:
            continue
        if allowed is not None and field not in allowed:
            _note(notes, f"feature field {field!r} is not an available statistic "
                         f"- row dropped")
            continue
        op = str(item.get("op") or "gt")
        if op not in QUERY_OPS:
            _note(notes, f"feature filter {field}: unknown op {op!r} - row dropped")
            continue
        # value2 is always materialized: the GUI card carries it whether or not
        # the op is "between", and `_build_query_card` reads it unconditionally.
        out.append({"field": field, "op": op,
                    "value": _as_float(item.get("value"), 0.0),
                    "value2": _as_float(item.get("value2"), 0.0)})
    return out


def pixel_filters_from_json(items: Any,
                            notes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Inverse of `pixel_filters_to_json`. An unusable channel drops the row (it
    names which raster to threshold, so there is no safe substitute); an unusable
    mode/op falls back to the same defaults the writer assumes."""
    out: List[Dict[str, Any]] = []
    for item in _as_list(items):
        if not isinstance(item, dict):
            _note(notes, "pixel filter is not an object - skipped")
            continue
        channel = str(item.get("channel") or "")
        if not channel:
            continue
        if channel not in PIXEL_CHANNELS:
            _note(notes, f"pixel filter: unknown channel {channel!r} - row dropped")
            continue
        mode = str(item.get("mode") or "keep")
        if mode not in PIXEL_MODES:
            _note(notes, f"pixel filter: unknown mode {mode!r} - using 'keep'")
            mode = "keep"
        op = str(item.get("op") or "gt")
        if op not in PIXEL_OPS:
            _note(notes, f"pixel filter: unknown op {op!r} - using 'gt'")
            op = "gt"
        out.append({"channel": channel, "mode": mode, "op": op,
                    "value": _as_float(item.get("value"), 0.0)})
    return out


def config_to_state(cfg: Any, fields: Optional[Sequence[str]] = None,
                    notes: Optional[List[str]] = None) -> Dict[str, Any]:
    """A flat, GUI-agnostic state dict from an AppConfig-shaped document.

    Every key is always present, so a caller never has to guess whether a block
    was missing. `build_config` folds several GUI controls into more than one
    JSON key (and omits keys at their defaults); this undoes each of those:

      * `filters` array, or the legacy singular `filter` object;
      * one `accurate` checkbox <- `accurate_ascending` OR `accurate_descending`;
      * one cores control <- `msc.requested_parallelism`, else
        `execution.threads_per_slice`;
      * `extremum_sample_radius` from `statistics` when non-zero, else `msc`
        (the precedence `config.cpp` applies);
      * `input.files` resolved against `input.folder`, since the C++ side accepts
        relative names and the viewer needs absolute ones.

    Persistence is reported as BOTH numbers rather than reconciled: the viewer
    only has a percent entry, and a percent cannot be derived from an absolute
    threshold without the slice value range, which is unknown until Run. A
    percent-less config leaves the entry alone and says so in `notes`.
    """
    root = _as_dict(cfg)
    inp = _as_dict(root.get("input"))
    msc = _as_dict(root.get("msc"))
    stats = _as_dict(root.get("statistics"))
    execution = _as_dict(root.get("execution"))
    segments = _as_dict(root.get("segments"))
    assembly = _as_dict(root.get("assembly"))

    folder = str(inp.get("folder") or "")
    files: List[str] = []
    for path in _as_list(inp.get("files")):
        if not isinstance(path, str) or not path:
            continue
        files.append(path if os.path.isabs(path) else os.path.join(folder, path))

    if isinstance(root.get("filters"), list):
        raw_filters = root["filters"]
    elif isinstance(root.get("filter"), dict):     # legacy singular form
        raw_filters = [root["filter"]]
    else:
        raw_filters = []

    percent = _opt_float(msc.get("persistence_percent"))
    absolute = _opt_float(msc.get("persistence_absolute"))
    if absolute is None:
        absolute = _opt_float(msc.get("persistence"))          # legacy alias
    if percent is None and absolute is not None:
        _note(notes, f"config sets persistence_absolute={absolute:g} and no percent; "
                     f"the Max persistence % entry was left unchanged")

    manifold: Optional[str] = str(msc.get("manifold") or "")
    if manifold not in ("ascending", "descending"):
        if manifold:
            _note(notes, f"unknown manifold {manifold!r} - left unchanged")
        manifold = None

    connectivity: Optional[int] = None
    if assembly.get("connectivity") is not None:
        value = _as_int(assembly.get("connectivity"), None)
        if value in (6, 18, 26):
            connectivity = value
        else:
            _note(notes, f"assembly.connectivity={assembly.get('connectivity')!r} is not "
                         f"6/18/26 - left unchanged")

    min_area: Optional[int] = None
    if segments.get("min_area") is not None:
        min_area = max(0, _as_int(segments.get("min_area"), 0))

    cores = (_as_int(msc.get("requested_parallelism"), 0)
             or _as_int(execution.get("threads_per_slice"), 0))
    concurrent = _as_int(execution.get("concurrent_slices"), 0)
    stat_state = statistics_from_json(root.get("statistics"), notes)
    # A query is validated against the schema THIS document's statistics block
    # produces. Using the caller's current field list instead would drop every
    # row naming a derived channel the moment such a config is loaded.
    if root.get("statistics") is not None:
        doc_fields = query_fields(json.dumps({"statistics": root.get("statistics")}))
    else:
        doc_fields = fields

    return {
        "folder": folder,
        "files": files,
        "output_folder": str(_as_dict(root.get("output")).get("folder") or ""),
        "filters": filters_from_json(raw_filters, notes),
        "base_filters": filters_from_json(root.get("base_filters"), notes),
        "feature_filters": queries_from_json(root.get("feature_filters"), doc_fields, notes),
        "pixel_filters": pixel_filters_from_json(root.get("pixel_filters"), notes),
        "persistence_percent": percent,
        "persistence_absolute": absolute,
        "manifold": manifold,
        "accurate": bool(msc.get("accurate_ascending")) or bool(msc.get("accurate_descending")),
        "extremum_sample_radius": (_as_int(stats.get("extremum_sample_radius"), 0)
                                   or _as_int(msc.get("extremum_sample_radius"), 0)),
        "min_area": min_area,
        "connectivity": connectivity,
        "cores_per_slice": cores if cores > 0 else None,
        "concurrent_slices": concurrent if concurrent > 0 else None,
        # The measurement spec. Previously the whole `statistics` block was
        # dropped on load except extremum_sample_radius, so a config naming a
        # derived channel could not survive a round trip through the viewer.
        "stat_channels": stat_state["channels"],
        "stat_reductions": stat_state["reductions"],
        "stat_extremum": stat_state["extremum"],
        "stat_relevance": stat_state["relevance"],
    }


# --------------------------------------------------------------------------- #
# Session file: the viewer's auto-saved "last used configuration".
#
# It IS a config -- a valid AppConfig for the first subsequence -- plus one extra
# top-level key carrying what AppConfig cannot express (every subsequence with
# its name, the browsed folder, the view state). The C++ parser reads named keys
# only, so the extra one is ignored and `mscoupon --config last_session.json`
# still runs. Exported configs never carry it.
# --------------------------------------------------------------------------- #
SESSION_KEY = "_gui"
SESSION_VERSION = 1


def build_session(cfg: Dict[str, Any], gui: Dict[str, Any]) -> Dict[str, Any]:
    doc = dict(cfg)
    doc[SESSION_KEY] = dict(gui)
    return doc


def split_session(doc: Any) -> tuple:
    """(config, gui) for a session file; (config, {}) for a bare AppConfig.

    This is what lets loading a hand-written config, an exported config and a
    saved session share one code path."""
    root = _as_dict(doc)
    gui = _as_dict(root.get(SESSION_KEY))
    return {k: v for k, v in root.items() if k != SESSION_KEY}, gui


def app_data_dir(app: str = "mscoupon") -> str:
    """Per-user config directory. Stdlib only -- one auto-saved file does not
    justify a dependency. Never raises, never creates."""
    base = os.environ.get("APPDATA") if os.name == "nt" else None
    if not base:
        base = os.environ.get("XDG_CONFIG_HOME")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, app)


def session_path(app: str = "mscoupon", name: str = "last_session.json") -> str:
    return os.path.join(app_data_dir(app), name)


def read_json_file(path: str) -> Optional[Dict[str, Any]]:
    """The parsed top-level object, or None when the file is missing,
    unreadable, malformed, or not an object. Never raises."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def serialize_session(doc: Dict[str, Any]) -> str:
    """`sort_keys` so key ordering can never make an unchanged session look
    changed -- the auto-save compares this text against the last one written."""
    return json.dumps(doc, indent=2, sort_keys=True)


def rotate_session_backups(path: str, keep: int = 3) -> None:
    """Roll `path` back one generation: .1 -> .2 -> ... -> .<keep>, then copy
    the live file to .1. Never raises, and never removes the live file.

    Auto-save is the only writer that runs without the user asking for it, so
    it is the one that needs an undo: a bad automatic write then costs the last
    `keep` states rather than everything. The copy (rather than a move) for the
    newest generation keeps `path` readable even if the write that follows
    fails."""
    if keep < 1 or not os.path.isfile(path):
        return
    root, ext = os.path.splitext(path)
    try:
        for i in range(keep, 1, -1):
            src = f"{root}.{i - 1}{ext}"
            if os.path.isfile(src):
                os.replace(src, f"{root}.{i}{ext}")
        shutil.copy2(path, f"{root}.1{ext}")
    except OSError:
        return          # a full or read-only directory must not stop the save


def write_session_text(text: str, path: str) -> bool:
    """Atomically replace `path` with `text`. False (never an exception) when the
    directory is unwritable, read-only, or full."""
    tmp = path + ".tmp"
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def write_session(doc: Dict[str, Any], path: str) -> bool:
    return write_session_text(serialize_session(doc), path)
