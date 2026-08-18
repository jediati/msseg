#!/usr/bin/env python3

"""
compare_gaussians.py

Compatibility shim. This module used to hold the shared GMM utilities; they now
live in msseg.mscoupon.normalize, which wraps the C++ implementation instead of
re-fitting with scikit-learn.

Keeping the old names importable means gaussian_downsample_error.py (and any
notebook that did `from compare_gaussians import ...`) still works. New code
should import from msseg.mscoupon.normalize directly.
"""

from msseg.mscoupon.normalize import (PARAMETERS, find_tiff_files, get_valid_pixels,
                                      load_reference_csv, natural_sort_key, read_tiff)
from msseg.mscoupon.normalize import fit_two_gaussians as _fit_two_gaussians

__all__ = [
    "PARAMETERS",
    "natural_sort_key",
    "find_tiff_files",
    "get_valid_pixels",
    "fit_two_gaussians_from_pixels",
    "fit_two_gaussians",
    "fit_tiff",
    "load_reference_csv",
]


def fit_two_gaussians_from_pixels(pixels, downsample_factor=1, rng=None, n_init=3,
                                  max_iter=300, tol=1e-6):
    """Fit a mixture to already-masked pixels.

    `rng` is accepted for signature compatibility but ignored -- the seed is now
    an explicit option on the underlying fit. Pass `seed=` to the library call
    if you need to vary the subsample.
    """
    # The values are already masked, so nothing further should be dropped.
    return _fit_two_gaussians(pixels, downsample_factor=downsample_factor,
                              omit_zeros=False, n_init=n_init, max_iter=max_iter, tol=tol)


def fit_two_gaussians(image, downsample_factor=1, rng=None, n_init=3, max_iter=300, tol=1e-6):
    """Convenience wrapper operating directly on an image (zeros omitted)."""
    return _fit_two_gaussians(image, downsample_factor=downsample_factor, omit_zeros=True,
                              n_init=n_init, max_iter=max_iter, tol=tol)


def fit_tiff(path, downsample_factor=1, rng=None, n_init=3):
    """Load and fit one TIFF."""
    from pathlib import Path

    path = Path(path)
    result = fit_two_gaussians(read_tiff(path), downsample_factor=downsample_factor,
                               n_init=n_init)
    result["filename"] = path.name
    return result
