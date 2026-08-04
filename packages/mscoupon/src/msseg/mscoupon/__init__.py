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
try:
    from .mscoupon_py import (  # noqa: F401
        version,
        filter_slice,
        filter_chain,
        segment_slice,
        prime_slice,
        evaluate_queries,
        Msc2DPipeline,
    )
    _HAVE_EXTENSION = True
except ImportError:  # pragma: no cover - only when the extension isn't built
    _HAVE_EXTENSION = False

__all__ = [
    "version",
    "filter_slice",
    "filter_chain",
    "segment_slice",
    "prime_slice",
    "evaluate_queries",
    "Msc2DPipeline",
]
