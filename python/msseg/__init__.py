"""MSSeg: Morse-Smale segmentation toolkit.

The compiled per-frontend extension modules are installed as submodules of this
package by scikit-build-core (see cmake/AddInstance.cmake and the generic
runner). They are imported lazily so that a partial build (e.g. only the core
CLI, no python) still yields an importable package.
"""

__all__ = ["mscoupon", "cellseg", "workflow"]
__version__ = "0.1.0"

try:  # 2D slice-segmentation instance
    from .mscoupon import mscoupon_py as mscoupon  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - optional component
    mscoupon = None  # type: ignore

try:  # 3D fluorescent-membrane cell-segmentation instance
    from .cellseg import cellseg_py as cellseg  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - optional component
    cellseg = None  # type: ignore

try:  # generic JSON workflow runner
    from .workflow import msworkflow_py as workflow  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - optional component
    workflow = None  # type: ignore
