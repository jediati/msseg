"""Fixed-rectangle intensity measurement.

Wraps the compiled ``measure_regions`` (``lib/mscoupon/region_measure.cpp``),
the port of ``measure_2_regions.py``. The manual counterpart to the GMM and
histogram measures: instead of inferring the two populations, the caller names
two rectangles known to sit in air and in metal.

NOTE the zero policy differs from the other two measures on purpose. These are
explicitly chosen physical regions, so exact zeros are RETAINED by default --
``omit_value`` is None here and 0 everywhere else.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import _engine
from .io import parse_range
from .pixels import _UNSET, get_valid_pixels, percentile_stats, resolve_omit_value


def parse_rect(rows: str, cols: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Parse ``ROW_START:ROW_END``, ``COL_START:COL_END``.

    Rows are Y and columns are X, matching ``image[r0:r1, c0:c1]``.
    """
    return parse_range(rows), parse_range(cols)


def extract_region(image, row_range, col_range) -> np.ndarray:
    """Slice ``image[y0:y1, x0:x1]``, bounds-checked."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape {image.shape}")

    (y0, y1), (x0, x1) = row_range, col_range
    height, width = image.shape
    if y1 > height:
        raise ValueError(f"Row range {y0}:{y1} exceeds image height {height}")
    if x1 > width:
        raise ValueError(f"Column range {x0}:{x1} exceeds image width {width}")
    return image[y0:y1, x0:x1]


def measure_region(image, row_range, col_range, omit_value=_UNSET,
                   omit_zeros: Optional[bool] = None) -> Dict[str, Any]:
    """Statistics for one rectangle: n_pixels, min, max, mean, std, percentiles."""
    values = get_valid_pixels(extract_region(image, row_range, col_range),
                              omit_value=resolve_omit_value(omit_value, omit_zeros, None))
    return percentile_stats(values)


def measure_regions(image, rects: Dict[str, Tuple], omit_value=_UNSET,
                    omit_zeros: Optional[bool] = None) -> Dict[str, Dict]:
    """Measure several named rectangles at once.

    `rects` maps a name to ``((row0, row1), (col0, col1))``. Returns a dict of
    name -> stats.
    """
    arr = np.asarray(image)

    spec = {name: {"rows": f"{r[0]}:{r[1]}", "cols": f"{c[0]}:{c[1]}"}
            for name, (r, c) in rects.items()}
    result = _engine.call("measure_regions", arr,
                          {"regions": spec,
                           "omit_value": resolve_omit_value(omit_value, omit_zeros, None)})
    if result is not None:
        return {name: dict(stats) for name, stats in result.items()}

    return {name: measure_region(arr, r, c, omit_value=omit_value, omit_zeros=omit_zeros)
            for name, (r, c) in rects.items()}


def measure_two_regions(image, air_rows, air_cols, metal_rows, metal_cols,
                        omit_value=_UNSET, omit_zeros: Optional[bool] = None) -> Dict[str, Dict]:
    """The air/metal pair the coupon workflow uses."""
    return measure_regions(image,
                           {"air": (air_rows, air_cols), "metal": (metal_rows, metal_cols)},
                           omit_value=omit_value, omit_zeros=omit_zeros)


def prefixed_stats(prefix: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one region's stats into ``<prefix>_<key>`` CSV columns."""
    return {f"{prefix}_{key}": value for key, value in stats.items()}
