"""MSSeg mscoupon: 2D Morse-Smale slice segmentation.

Part of the ``msseg`` namespace package. The compiled ``mscoupon_py`` extension
(installed alongside this file by CMake) provides the C++ entry points; they are
re-exported here so ``from msseg import mscoupon; mscoupon.segment_slice(...)``
works, matching the original notebook/example usage.
"""
from .mscoupon_py import version, filter_slice, segment_slice  # noqa: F401

__all__ = ["version", "filter_slice", "segment_slice"]
