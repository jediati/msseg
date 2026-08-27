"""MSSeg mscoupon: 2D Morse-Smale slice segmentation.

Part of the ``msseg`` namespace package. The compiled ``mscoupon_py`` extension
(installed alongside this file by CMake) provides the C++ entry points; they are
re-exported here so ``from msseg import mscoupon; mscoupon.segment_slice(...)``
works, matching the original notebook/example usage.
"""
# The compiled extension provides the C++ entry points. Guard the import so the
# pure-Python submodules (app, config_io) remain importable in environments where
# the extension has not been built (e.g. the headless GUI --selftest); the GUI
# imports these symbols lazily only when it actually primes/segments.
#
# Re-export by iterating the module rather than naming each symbol: a `from ... import
# a, b, c` binds names one at a time and stops at the first missing one, so an
# extension older than this file left the package half-populated -- the failure
# mode behind fields showing as "n/a" instead of erroring.
_EXPORTS = (
    "version",
    "filter_slice",
    "filter_chain",
    "segment_slice",
    "prime_slice",
    "evaluate_queries",
    "evaluate_queries_table",
    "feature_fields",
    # The measurement-channel schema: what channels a spec resolves to, the
    # per-column {name, channel, reduction} breakdown the GUI's pickers are built
    # from, and the channel rasters its 3D assembly measures on.
    "feature_schema",
    "stat_channels",
    "stat_channel_images",
    "fit_gmm",
    "measure_histogram",
    "measure_regions",
    "Msc2DPipeline",
)

try:
    from . import mscoupon_py as _ext
except ImportError:  # pragma: no cover - only when the extension isn't built
    _ext = None
    _HAVE_EXTENSION = False
else:
    _missing = [n for n in _EXPORTS if not hasattr(_ext, n)]
    for _n in _EXPORTS:
        if hasattr(_ext, _n):
            globals()[_n] = getattr(_ext, _n)
    # A partial extension is a stale build, not a working install. Say so plainly
    # rather than letting a downstream lookup fail somewhere unrelated.
    _HAVE_EXTENSION = not _missing
    if _missing:  # pragma: no cover - only with a stale .pyd
        import warnings
        warnings.warn(
            "msseg.mscoupon: the compiled extension is missing "
            + ", ".join(_missing)
            + " -- it predates this source tree. Rebuild and reinstall the package.",
            RuntimeWarning,
            stacklevel=2,
        )

__all__ = list(_EXPORTS)
