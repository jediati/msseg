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
}

FILTER_OPERATIONS = list(FILTER_SCHEMA.keys())

# Feature-query fields (must match mscoupon::feature_row in query.cpp) and ops.
QUERY_FIELDS = [
    "area", "mean_base", "mean_filtered", "min_base", "max_base", "std_base",
    "min_filtered", "max_filtered", "std_filtered", "bbox_w", "bbox_h",
]
QUERY_OPS = ["lt", "le", "gt", "ge", "eq", "between"]


def filters_to_json(filters: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a list of {"operation":..., "params":{...}} filter cards,
    dropping "none" stages (they are no-ops / the trailing add-row)."""
    out: List[Dict[str, Any]] = []
    for f in filters:
        op = f.get("operation", "none")
        if op == "none":
            continue
        out.append({"operation": op, "params": dict(f.get("params", {}))})
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


def build_config(
    *,
    files: Sequence[str],
    output_folder: str,
    filters: Sequence[Dict[str, Any]],
    persistence_percent: Optional[float] = 10.0,
    persistence_absolute: Optional[float] = None,
    manifold: str = "ascending",
    accurate: bool = False,
    min_area: Optional[int] = None,
    feature_filters: Sequence[Dict[str, Any]] = (),
    connectivity: int = 26,
    matching_enabled: bool = True,
    folder: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the AppConfig-shaped dict for one subsequence (an explicit file list).

    `folder` defaults to the common parent directory of `files`; `input.files`
    carries the ordered, absolute selection so the CLI reproduces this stack.
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

    cfg: Dict[str, Any] = {
        "input": {"folder": folder, "files": files},
        "output": {"folder": output_folder},
        "filters": filters_to_json(filters),
        "msc": msc,
        "feature_filters": queries_to_json(feature_filters),
        "assembly": {"connectivity": int(connectivity)},
        "matching": {"enabled": bool(matching_enabled)},
    }
    if min_area is not None:
        cfg["segments"] = {"min_area": int(min_area)}
    return cfg


def dump_config(cfg: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
