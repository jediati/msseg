"""Brightness windows for the image channels: a first guess from the data.

A channel's window is a pair of FRACTIONS of the shown source's value range
(what ``SliceCanvas.set_window`` takes), and each channel keeps its own so a
flip between the original image and a derived field lands on a sensible
brightness for both. The first time a channel is shown its window is taken
from the raster itself -- the 1st and 99th percentiles, so a few hot pixels
do not own the whole range -- and from then on it is whatever the user set.

Pure numpy; the shell calls these with the ``ImageSource`` about to be drawn.
"""
from __future__ import annotations

import math

DEFAULT_LO_PERCENTILE = 1.0
DEFAULT_HI_PERCENTILE = 99.0
DEFAULT_BUDGET = 1_000_000          # samples -- a strided view of anything bigger


def window_sample(source, np, budget=DEFAULT_BUDGET):
    """A flat float32 sample of the source's finite values, at most about
    ``budget`` of them.

    In-memory sources are read through their array (``.array`` for the
    canvas's ``ArrayImageSource``, ``.raster`` for a placed item raster) --
    the placed one on purpose: its ``read_region`` fills everything outside
    the item with the raster's minimum, which would drag the low percentile
    down to it. A pyramid is sampled at its coarsest level. Colour planes are
    pooled, since the canvas windows them with one shared range.
    """
    arr = getattr(source, "array", None)
    if arr is None:
        arr = getattr(source, "raster", None)
    if arr is None:
        levels = int(getattr(source, "levels", 1) or 1)
        top = max(0, levels - 1)
        h, w = (int(v) for v in source.level_shape(top))
        arr = source.read_region(top, 0, 0, w, h)
    arr = np.asarray(arr)
    if arr.ndim >= 2:
        n = int(arr.shape[0]) * int(arr.shape[1])
        if n > budget:
            step = int(math.ceil(math.sqrt(n / float(budget))))
            arr = arr[::step, ::step]
    flat = np.asarray(arr, dtype=np.float32).ravel()
    return flat[np.isfinite(flat)]


def percentile_window(sample, value_range, lo=DEFAULT_LO_PERCENTILE,
                      hi=DEFAULT_HI_PERCENTILE):
    """``(vmin, vmax)`` window fractions of ``value_range`` at the sample's
    ``lo``/``hi`` percentiles; the full range when the sample or the range
    cannot say (empty, constant, NaN).
    """
    import numpy as np
    sample = np.asarray(sample, dtype=np.float32).ravel()
    try:
        rmin, rmax = float(value_range[0]), float(value_range[1])
    except (TypeError, ValueError, IndexError):
        return (0.0, 1.0)
    span = rmax - rmin
    if sample.size == 0 or not (span > 0.0) or not math.isfinite(span):
        return (0.0, 1.0)
    p_lo, p_hi = np.percentile(sample, [float(lo), float(hi)])
    f_lo = min(max((float(p_lo) - rmin) / span, 0.0), 1.0)
    f_hi = min(max((float(p_hi) - rmin) / span, 0.0), 1.0)
    if not (f_hi > f_lo):
        return (0.0, 1.0)
    return (f_lo, f_hi)


def source_window(source, np, lo=DEFAULT_LO_PERCENTILE, hi=DEFAULT_HI_PERCENTILE,
                  budget=DEFAULT_BUDGET):
    """The percentile window of an ``ImageSource``: sample + its value range."""
    return percentile_window(window_sample(source, np, budget=budget),
                             source.value_range(), lo=lo, hi=hi)
