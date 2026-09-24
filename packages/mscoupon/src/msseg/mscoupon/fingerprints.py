"""What a primed slice depends on, split in two.

A prime is two computations with different inputs:

* the **field** -- the chains, the colour input, the MSC and its cancellation
  hierarchy, the base labelling. Everything ``Msc2DPipeline::build`` does
  before it measures anything. Changing any of these means a new prime.
* the **measurement** -- the ``statistics`` block, the extremum sample radius
  and the declared plane count. Everything the per-region rows are made of.
  Changing only these means ``pipe.remeasure``: the MSC, the labels and the
  arcs stay, and only the rows are rebuilt (~0.2-2 s against ~4 s).

Both fingerprints are over the PARAMS document ``profile_params_json`` composes
(a JSON string or the dict it decodes to), because that is what both engines
prime with. Keys that do not change a result are in neither: the core count
and builder choice (the partitioned builder is bit-identical), the GPU
gradient and GPU statistics flags (the gradient is bit-identical; the device
statistics differ from the host loop only in summation order).

The persistence percentage is on the field side only through the one thing
the build reads it for: the cancellation cap, ``max(10 %, pct)``. Moving it
anywhere under 10 % is therefore not a new field; above 10 % the cap follows
the percentage, and a pipe built to a lower cap would clamp, so it is.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, Union

Params = Union[str, Dict[str, Any], None]

# msc keys that never change a result.
_RESULT_FREE = ("use_gpu_gradient", "use_gpu_stats", "compute_algorithm",
                "requested_parallelism")
# msc keys that belong to the measurement, not the field.
_MEASURE_MSC = ("extremum_sample_radius",)
# MSCEER's own floor on the cancellation cap, as a percentage of the range.
_CAP_FLOOR_PCT = 10.0


def _doc(params: Params) -> Dict[str, Any]:
    if params is None:
        return {}
    if isinstance(params, str):
        try:
            doc = json.loads(params) if params.strip() else {}
        except ValueError:
            return {}
        return doc if isinstance(doc, dict) else {}
    return params if isinstance(params, dict) else {}


def _dumps(doc: Any) -> str:
    return json.dumps(doc, sort_keys=True, default=str)


def field_doc(params: Params) -> Dict[str, Any]:
    """The part of a params document the field (MSC + base labelling) reads."""
    doc = copy.deepcopy(_doc(params))
    doc.pop("statistics", None)
    msc = doc.get("msc")
    if isinstance(msc, dict):
        for k in _RESULT_FREE + _MEASURE_MSC:
            msc.pop(k, None)
        # The build reads the percentage only as the cancellation cap.
        if msc.get("persistence_absolute") is None and "persistence_percent" in msc:
            try:
                pct = float(msc.pop("persistence_percent"))
            except (TypeError, ValueError):
                pct = _CAP_FLOOR_PCT
            msc["persistence_cap_percent"] = max(_CAP_FLOOR_PCT, pct)
    col = (doc.get("input") or {}).get("color") if isinstance(doc.get("input"), dict) else None
    if isinstance(col, dict):
        col.pop("channels", None)
        if not col:
            doc["input"].pop("color", None)
        if not doc["input"]:
            doc.pop("input", None)
    return doc


def measure_doc(params: Params) -> Dict[str, Any]:
    """The part of a params document the per-region rows are made of."""
    doc = _doc(params)
    msc = doc.get("msc") if isinstance(doc.get("msc"), dict) else {}
    inp = doc.get("input") if isinstance(doc.get("input"), dict) else {}
    col = inp.get("color") if isinstance(inp.get("color"), dict) else {}
    out = {"statistics": doc.get("statistics")}
    radius = msc.get("extremum_sample_radius")
    if radius:
        out["extremum_sample_radius"] = radius
    if col.get("channels"):
        out["channels"] = col.get("channels")
    return out


def field_fingerprint_of(params: Params) -> str:
    return _dumps(field_doc(params))


def measure_fingerprint_of(params: Params) -> str:
    return _dumps(measure_doc(params))
