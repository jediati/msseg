"""Pixel masking, subsampling and the small numeric helpers the measures share.

``get_valid_pixels`` is the single definition of "which pixels count" -- the
no-data / NaN / Inf mask that used to be copy-pasted into six scripts, each with
zero-dropping hardcoded. Here it is the ``omit_value`` parameter, and it mirrors
``collect_valid_pixels`` in ``lib/mscoupon/measure_util.hpp`` exactly, so the
Python and C++ measures agree on their input.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# The percentile ladder reported by the measure CSVs, and its column spelling.
# Kept in lockstep with default_percentiles() / default_percentile_names() in
# lib/mscoupon/measure_util.hpp.
PERCENTILES: List[float] = [0.01, 0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9, 99.99]
PERCENTILE_NAMES: List[str] = ["p0_01", "p0_1", "p1_0", "p5_0", "p25_0", "p50_0",
                               "p75_0", "p95_0", "p99_0", "p99_9", "p99_99"]


_UNSET = object()


def resolve_omit_value(omit_value=_UNSET, omit_zeros=None, default: Optional[float] = 0.0):
    """Reconcile the no-data policy from the new and old spellings.

    ``omit_value`` is the sentinel to drop (None keeps every pixel); the older
    boolean ``omit_zeros`` maps True -> 0.0 and False -> None. An explicit
    ``omit_value`` wins. Shared so every entry point agrees, and mirrors
    parse_omit_policy() in lib/mscoupon/measure_config.cpp.
    """
    if omit_value is not _UNSET:
        return omit_value
    if omit_zeros is not None:
        return 0.0 if omit_zeros else None
    return default


def get_valid_pixels(image, omit_value=_UNSET, omit_nonfinite: bool = True,
                     omit_zeros: Optional[bool] = None) -> np.ndarray:
    """Flatten `image` and drop the pixels that should not be measured.

    Args:
        omit_value: the stack's no-data sentinel, dropped before measuring.
            Defaults to 0.0 -- reconstructed slices usually carry a large no-data
            background at exactly 0, which would otherwise dominate any fit --
            but a stack may pad with any constant (43, say), and dropping the
            wrong value leaves that plateau in as a spurious population. None
            keeps every pixel; the region measure passes None because its
            rectangles are hand-picked physical regions.
        omit_nonfinite: drop NaN/Inf. Only meaningful for floating-point input;
            integer images skip the test, mirroring the
            ``np.issubdtype(x.dtype, np.floating)`` branch this replaces.
        omit_zeros: deprecated boolean spelling of `omit_value`; True means the
            sentinel is 0, False means keep everything.

    Returns a 1-D float64 array.
    """
    omit = resolve_omit_value(omit_value, omit_zeros)
    x = np.asarray(image).ravel()

    keep = np.ones(x.shape, dtype=bool)
    if omit_nonfinite and np.issubdtype(x.dtype, np.floating):
        keep &= np.isfinite(x)
    if omit is not None:
        # Compare in the image's own dtype so a float32 raster is matched against
        # the sentinel rounded the way it was stored, not a double that never
        # compares equal.
        keep &= x != np.asarray(omit).astype(x.dtype, copy=False)

    return np.asarray(x[keep], dtype=np.float64)


def count_omitted(image, omit_value: Optional[float] = 0.0) -> int:
    """Number of pixels equal to the no-data sentinel, counted before masking."""
    if omit_value is None:
        return 0
    x = np.asarray(image)
    return int(np.count_nonzero(x == np.asarray(omit_value).astype(x.dtype, copy=False)))


def count_zeros(image) -> int:
    """Number of exact-zero pixels, counted before masking."""
    return count_omitted(image, 0.0)


def random_downsample(x: np.ndarray, factor: int, rng=None) -> np.ndarray:
    """Keep a random ~1/factor of `x`, without replacement (at least 10 values)."""
    if factor is None or factor <= 1:
        return np.asarray(x)

    x = np.asarray(x)
    n = x.size
    n_sample = min(max(10, n // int(factor)), n)
    if n_sample >= n:
        return x

    if rng is None:
        rng = np.random.default_rng()
    return x[rng.choice(n, size=n_sample, replace=False)]


def trim_pixels(x: np.ndarray, trim_percent: float) -> Tuple[np.ndarray, float, float]:
    """Symmetrically trim the intensity tails.

    ``trim_percent=0.5`` keeps the 0.5th..99.5th percentiles. Returns
    ``(trimmed, lo, hi)``; with no trimming the cut points are min/max.
    """
    x = np.asarray(x)
    if not trim_percent or trim_percent <= 0:
        return x, float(np.min(x)), float(np.max(x))

    lo, hi = np.percentile(x, [trim_percent, 100.0 - trim_percent])
    return x[(x >= lo) & (x <= hi)], float(lo), float(hi)


def percentile_stats(x: np.ndarray, percentiles: Optional[Sequence[float]] = None) -> Dict:
    """Basic statistics plus the percentile ladder, keyed as in the CSVs."""
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("Region contains no usable pixels")

    ladder = list(PERCENTILES if percentiles is None else percentiles)
    names = (PERCENTILE_NAMES if percentiles is None
             else [f"p{str(p).replace('.', '_')}" for p in ladder])

    out = {
        "n_pixels": int(x.size),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }
    for name, value in zip(names, np.percentile(x, ladder)):
        out[name] = float(value)
    return out


def parabolic_offset(y1: float, y2: float, y3: float) -> float:
    """Sub-bin peak offset, in bins, from a parabola through three samples.

    Returns 0 for a degenerate triple; clamped to [-1, 1] so noise cannot throw
    the peak into a neighbouring bin.
    """
    denom = float(y1) - 2.0 * float(y2) + float(y3)
    if denom == 0.0:
        return 0.0
    return float(np.clip(0.5 * (float(y1) - float(y3)) / denom, -1.0, 1.0))


def estimate_mode(x: np.ndarray, n_bins: int = 512) -> float:
    """Empirical mode via a histogram plus parabolic interpolation of the peak.

    This is the mode of the *samples*, not of a fitted Gaussian (whose mode is
    its mean).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 10:
        return float("nan")

    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmin == xmax:
        return xmin

    counts, edges = np.histogram(x, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    i = int(np.argmax(counts))
    if i == 0 or i == len(counts) - 1:
        return float(centers[i])

    offset = parabolic_offset(counts[i - 1], counts[i], counts[i + 1])
    return float(centers[i] + offset * (centers[1] - centers[0]))
