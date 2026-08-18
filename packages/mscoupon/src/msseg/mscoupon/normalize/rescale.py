"""Two-point normalization: the value mapping itself.

Mirrors ``TwoPoint`` in ``lib/mscoupon/normalize.hpp``. Given two measured
landmarks (low = air/void, high = metal/solid), a threshold can be written as a
single normalized number that transfers across a whole stack: ``0.7`` means
``0.3*low + 0.7*high``, and per-slice landmarks absorb slice-to-slice drift.

Normalization is applied as a *filter* on a channel rather than as a transform
on each threshold. That keeps the arithmetic honest for free: statistics
computed on a normalized channel are already normalized, means map affinely,
standard deviations map by scale alone (the offset cancels), and sums merge
correctly across slices that have different landmarks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class TwoPoint:
    """A pair of intensity landmarks in one channel's own units."""

    low: float
    high: float

    @property
    def valid(self) -> bool:
        return self.high > self.low

    @property
    def scale(self) -> float:
        return self.high - self.low

    def to_raw(self, t):
        """Normalized -> raw. ``t=0.7`` gives ``0.3*low + 0.7*high``."""
        return self.low + np.asarray(t) * self.scale if np.ndim(t) else self.low + t * self.scale

    def to_norm(self, v):
        """Raw -> normalized."""
        return (np.asarray(v) - self.low) / self.scale if np.ndim(v) else (v - self.low) / self.scale

    def apply(self, array: np.ndarray, clamp: bool = False, out: Optional[np.ndarray] = None):
        """Map a whole raster to normalized units.

        Writes into `out` (or a new array); pass ``out=array`` to normalize in
        place. A degenerate pair is a no-op rather than a divide by zero.
        """
        array = np.asarray(array)
        if not self.valid:
            return array if out is None else out

        result = np.subtract(array, self.low, out=out, dtype=np.float32) \
            if out is not None else (array - self.low).astype(np.float32)
        result /= np.float32(self.scale)
        if clamp:
            np.clip(result, 0.0, 1.0, out=result)
        return result

    def as_dict(self) -> Dict[str, float]:
        return {"low": float(self.low), "high": float(self.high)}


# Default landmark pair per method, matching parse_normalize_config() in
# lib/mscoupon/normalize.cpp so the GUI and the CLI resolve the same pair.
DEFAULT_LANDMARKS = {
    "gmm": ("mu_1", "mu_2"),
    "histogram": ("peak_low", "peak_high"),
    "regions": ("air_median", "metal_median"),
}


def measure_two_point(image, method: str = "gmm", low_from: Optional[str] = None,
                      high_from: Optional[str] = None, low: Optional[float] = None,
                      high: Optional[float] = None, **params: Any) -> TwoPoint:
    """Measure a slice's two landmarks -- the Python mirror of the C++
    ``measure_two_point`` in ``lib/mscoupon/normalize.cpp``.

    `method` is gmm | histogram | regions | manual. `low`/`high` give the manual
    pair and double as the fallback when a measure fails or returns a degenerate
    pair, so one bad slice does not abort a stack.
    """
    manual = TwoPoint(float(low), float(high)) if low is not None and high is not None else None

    if method == "manual":
        if manual is None:
            raise ValueError("normalize: method 'manual' requires both 'low' and 'high'")
        return manual

    if method not in DEFAULT_LANDMARKS:
        raise ValueError("normalize: method must be 'gmm', 'histogram', 'regions' or 'manual'")

    default_low, default_high = DEFAULT_LANDMARKS[method]
    low_from = low_from or default_low
    high_from = high_from or default_high

    try:
        if method == "gmm":
            from .gmm import fit_two_gaussians
            result = fit_two_gaussians(image, **params)
        elif method == "histogram":
            from .histogram import measure_histogram
            result = measure_histogram(image, **params)
        else:
            from .regions import measure_regions
            rects = params.pop("rects", None) or {}
            flat = {}
            for name, stats in measure_regions(image, rects, **params).items():
                flat.update({f"{name}_{k}": v for k, v in stats.items()})
                if "p50_0" in stats:
                    flat[f"{name}_median"] = stats["p50_0"]
            result = flat
        tp = two_point_from_result(result, low_from, high_from)
    except Exception:
        if manual is not None and manual.valid:
            return manual
        raise

    if not tp.valid:
        if manual is not None and manual.valid:
            return manual
        raise ValueError(f"normalize: measured landmarks are degenerate "
                         f"(low={tp.low:.6g} high={tp.high:.6g})")
    return tp


def two_point_from_result(result: Mapping[str, Any], low_from: str, high_from: str) -> TwoPoint:
    """Build a :class:`TwoPoint` by naming two outputs of a measure.

    Works with any of the measure result dicts -- ``("mu_1", "mu_2")`` for the
    GMM, ``("peak_low", "peak_high")`` for the histogram, ``("air_median",
    "metal_median")`` for regions.
    """
    def get(key: str) -> float:
        if key not in result:
            raise KeyError(f"measure result has no landmark {key!r}; "
                           f"available: {sorted(result)}")
        return float(result[key])

    return TwoPoint(get(low_from), get(high_from))
