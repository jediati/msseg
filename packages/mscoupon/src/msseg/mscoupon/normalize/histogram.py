"""Histogram peak-finding measurement.

Wraps the compiled ``measure_histogram`` (``lib/mscoupon/histogram_peaks.cpp``),
the port of ``measure_im.py``: mask, subsample, build a histogram between two
percentiles, smooth it, then take the two strongest *separated* local maxima and
refine each to sub-bin precision.

Peaks are returned ordered by intensity, not by height.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from . import _engine
from .pixels import (_UNSET, PERCENTILE_NAMES, PERCENTILES, count_zeros, get_valid_pixels,
                     parabolic_offset, random_downsample, resolve_omit_value)


def measure_histogram(
    image,
    downsample_factor: int = 1,
    omit_value=_UNSET,
    bins: int = 1024,
    smooth_width: int = 11,
    peak_window: int = 32,
    min_peak_distance: int = 64,
    hist_lo_percentile: float = 1.0,
    hist_hi_percentile: float = 99.0,
    seed: int = 0,
    omit_zeros: Optional[bool] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Locate the two intensity populations by histogram peak finding.

    Returns ``peak_low``/``peak_high`` (plus ``peak_*_bin``/``peak_*_height``),
    the histogram support ``hist_lo``/``hist_hi``, ``min``/``max``, the
    ``p0_01..p99_99`` ladder and the ``n_*_pixels`` counts.
    """
    params = {
        "downsample_factor": int(downsample_factor),
        "omit_value": resolve_omit_value(omit_value, omit_zeros),
        "bins": int(bins),
        "smooth_width": int(smooth_width),
        "peak_window": int(peak_window),
        "min_peak_distance": int(min_peak_distance),
        "hist_lo_percentile": float(hist_lo_percentile),
        "hist_hi_percentile": float(hist_hi_percentile),
        "seed": int(seed),
    }
    params.update(overrides)

    result = _engine.call("measure_histogram", np.asarray(image), {"histogram": params})
    if result is None:
        result = _measure_with_numpy(image, params)
    return dict(result)


def moving_average(counts, width: int) -> np.ndarray:
    """Centred moving average with edge padding."""
    counts = np.asarray(counts, dtype=np.float64)
    width = int(width)
    if width <= 1:
        return counts
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width, dtype=np.float64) / width
    padded = np.pad(counts, width // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def local_maximum_indices(y, radius: int) -> List[int]:
    """Indices that dominate a +/- `radius` window.

    Stricter than "higher than both neighbours" so the shoulders of a broad peak
    are not reported as separate maxima; a plateau is recorded once, at its top.

    Known limitation, inherited from measure_im.py and kept for parity: a
    candidate needs `radius` bins on each side, so a population within `radius`
    bins of an edge can never be selected. That bites when one population is so
    dominant that hist_lo_percentile lands inside it; the symptom is two peaks
    reported in the flat region between the real populations.
    """
    y = np.asarray(y)
    radius = max(1, int(radius))
    out: List[int] = []
    for i in range(radius, len(y) - radius):
        if y[i] < np.max(y[i - radius:i + radius + 1]):
            continue
        if out and i - out[-1] <= radius:
            if y[i] > y[out[-1]]:
                out[-1] = i
        else:
            out.append(i)
    return out


def find_two_peaks(counts, centers, smooth_width: int = 11, peak_window: int = 32,
                   min_peak_distance: int = 64) -> Dict[str, Any]:
    """Two strongest separated maxima of a histogram, ordered by intensity."""
    smooth = moving_average(counts, smooth_width)
    centers = np.asarray(centers, dtype=np.float64)

    candidates = sorted(local_maximum_indices(smooth, peak_window),
                        key=lambda i: smooth[i], reverse=True)

    selected: List[int] = []
    for i in candidates:
        if all(abs(i - j) >= min_peak_distance for j in selected):
            selected.append(i)
        if len(selected) == 2:
            break

    if len(selected) < 2:
        # Fall back to the global maximum plus the best point outside its
        # neighbourhood, so a single broad peak still yields two landmarks.
        first = int(np.argmax(smooth))
        suppressed = smooth.copy()
        lo = max(0, first - min_peak_distance)
        hi = min(len(suppressed), first + min_peak_distance + 1)
        suppressed[lo:hi] = -np.inf
        selected = [first, int(np.argmax(suppressed))]

    i1, i2 = sorted(selected[:2])
    bin_width = centers[1] - centers[0]

    def refine(i: int) -> float:
        if i <= 0 or i >= len(smooth) - 1:
            return float(centers[i])
        return float(centers[i] + parabolic_offset(smooth[i - 1], smooth[i], smooth[i + 1]) * bin_width)

    return {
        "peak_low": refine(i1),
        "peak_high": refine(i2),
        "peak_low_bin": int(i1),
        "peak_high_bin": int(i2),
        "peak_low_height": float(smooth[i1]),
        "peak_high_height": float(smooth[i2]),
    }


def _measure_with_numpy(image, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback used only when the compiled extension is unavailable."""
    arr = np.asarray(image)
    n_zero = count_zeros(arr)
    x = get_valid_pixels(arr, omit_value=params.get("omit_value", 0.0))
    n_valid = x.size
    if n_valid < 10:
        raise ValueError(f"Only {n_valid} valid nonzero pixels")

    rng = np.random.default_rng(params.get("seed", 0))
    x = random_downsample(x, params.get("downsample_factor", 1), rng)

    out: Dict[str, Any] = {
        "n_total_pixels": int(arr.size),
        "n_zero_pixels": n_zero,
        "n_valid_pixels": int(n_valid),
        "n_sampled_pixels": int(x.size),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }
    for name, value in zip(PERCENTILE_NAMES, np.percentile(x, PERCENTILES)):
        out[name] = float(value)

    lo = float(np.percentile(x, params.get("hist_lo_percentile", 1.0)))
    hi = float(np.percentile(x, params.get("hist_hi_percentile", 99.0)))
    if hi <= lo:
        raise ValueError("degenerate percentile range; cannot construct a histogram")

    counts, edges = np.histogram(x, bins=params.get("bins", 1024), range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    out.update(find_two_peaks(counts, centers,
                              smooth_width=params.get("smooth_width", 11),
                              peak_window=params.get("peak_window", 32),
                              min_peak_distance=params.get("min_peak_distance", 64)))
    out["hist_lo"] = lo
    out["hist_hi"] = hi
    return out
