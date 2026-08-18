"""Two-point normalization and the intensity measures behind it.

A coupon stack is thresholded by intensity, but the absolute intensity drifts
from slice to slice, so a raw threshold that is right at the start of a scan is
wrong by the end. This subpackage measures two landmarks per slice -- low
(air/void) and high (metal/solid) -- so a threshold can be written once as a
normalized number: ``0.7`` means ``0.3*low + 0.7*high``, resolved per slice.

Three measures produce the landmarks:

* :mod:`.gmm`        -- fit a two-component Gaussian mixture
* :mod:`.histogram`  -- smooth the histogram and take its two strongest peaks
* :mod:`.regions`    -- average two hand-picked rectangles

All three are implemented in C++ (``lib/mscoupon/``) and wrapped here, so the
batch CLI and the GUI cannot disagree about what a landmark is; pure-numpy
fallbacks keep the analysis scripts runnable without the compiled extension.
scikit-learn is only a test dependency now (``tests/gmm_parity.py``).

Normalization is applied as a *filter stage on a channel*
(``base_filters: [{"operation": "normalize", ...}]``) rather than as a transform
on each threshold. Pixel values are rewritten once, so every downstream
statistic and threshold is already in normalized units -- which also means
standard deviations scale correctly and per-slice sums merge correctly into 3D
features. See :mod:`.rescale` for the mapping itself.
"""
from __future__ import annotations

from ._engine import have_extension
from .gmm import (PARAMETERS, PRESET_MEASURE, PRESET_TWO_GAUSSIAN, fit_two_gaussians,
                  load_reference_csv, void_probability_image)
from .histogram import find_two_peaks, measure_histogram
from .io import (CsvWriter, find_tiff_files, format_float, iter_images, natural_sort_key,
                 parse_range, parse_samples, read_tiff, select_files)
from .pixels import (PERCENTILE_NAMES, PERCENTILES, count_zeros, estimate_mode,
                     get_valid_pixels, parabolic_offset, percentile_stats,
                     random_downsample, trim_pixels)
from .regions import (extract_region, measure_region, measure_regions, measure_two_regions,
                      prefixed_stats)
from .rescale import DEFAULT_LANDMARKS, TwoPoint, measure_two_point, two_point_from_result

__all__ = [
    # value mapping
    "TwoPoint",
    "two_point_from_result",
    "measure_two_point",
    "DEFAULT_LANDMARKS",
    # measures
    "fit_two_gaussians",
    "measure_histogram",
    "measure_regions",
    "measure_two_regions",
    "measure_region",
    "find_two_peaks",
    "void_probability_image",
    "PRESET_MEASURE",
    "PRESET_TWO_GAUSSIAN",
    "PARAMETERS",
    # pixels
    "get_valid_pixels",
    "count_zeros",
    "random_downsample",
    "trim_pixels",
    "percentile_stats",
    "parabolic_offset",
    "estimate_mode",
    "extract_region",
    "prefixed_stats",
    "PERCENTILES",
    "PERCENTILE_NAMES",
    # io
    "natural_sort_key",
    "find_tiff_files",
    "parse_samples",
    "parse_range",
    "read_tiff",
    "select_files",
    "iter_images",
    "CsvWriter",
    "format_float",
    "load_reference_csv",
    "have_extension",
]
