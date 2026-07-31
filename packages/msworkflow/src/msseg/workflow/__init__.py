"""MSSeg generic JSON-workflow runner (``msseg.workflow``).

Part of the ``msseg`` namespace package. Re-exports the compiled ``msworkflow_py``
extension: ``run(volume, workflow_json) -> labels`` and ``version()``.
"""
from . import msworkflow_py  # noqa: F401
from .msworkflow_py import run, version  # noqa: F401

__all__ = ["run", "version"]
