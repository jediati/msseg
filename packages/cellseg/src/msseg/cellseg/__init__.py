"""MSSeg cellseg: 3D fluorescent-membrane cell segmentation.

Part of the ``msseg`` namespace package. Re-exports the compiled ``cellseg_py``
extension (installed alongside by CMake) so ``from msseg import cellseg;
cellseg.heavy_lift(...)`` works, matching the original example/smoke-test usage.
"""
from . import cellseg_py  # noqa: F401
from .cellseg_py import *  # noqa: F401,F403
