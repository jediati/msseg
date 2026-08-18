"""Bridge to the compiled measures in ``mscoupon_py``.

The three measures are implemented in C++ (``lib/mscoupon/gmm.cpp``,
``histogram_peaks.cpp``, ``region_measure.cpp``) so that the batch CLI and the
GUI cannot disagree about what a landmark is. The Python wrappers in this
subpackage call those, rather than re-fitting with scikit-learn.

The extension is optional: when it is not built, callers fall back to the pure
numpy implementations, which is enough to keep the analysis scripts runnable in
a bare checkout. Guarded the same way as ``msseg/mscoupon/__init__.py``.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional


def _load():
    try:
        from msseg import mscoupon as engine
        if getattr(engine, "_HAVE_EXTENSION", False):
            return engine
    except ImportError:
        pass
    try:  # a raw build tree, without a pip install
        import mscoupon_py as engine  # type: ignore
        return engine
    except ImportError:
        return None


_ENGINE = _load()


def have_extension() -> bool:
    return _ENGINE is not None


def call(name: str, image, params: Optional[Dict[str, Any]] = None):
    """Invoke a compiled measure, or return None when the extension is absent."""
    if _ENGINE is None:
        return None
    fn = getattr(_ENGINE, name, None)
    if fn is None:
        return None
    return fn(image, json.dumps(params or {}))
