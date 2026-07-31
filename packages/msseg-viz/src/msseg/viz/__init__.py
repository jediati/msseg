"""MSSeg shared visualization utilities (``msseg.viz``).

Instance-agnostic building blocks that individual segmentation frontends build
their viewers on: a shared per-minimum color palette and the merge-tree icicle
plot. Part of the ``msseg`` namespace package; depends only on numpy + matplotlib
(no compiled extension, no instance package), so it installs as a universal wheel.
"""
from .icicle import draw_icicle
from .palette import min_color, _MIN_LUT, _MIN_K

__all__ = ["draw_icicle", "min_color", "_MIN_LUT", "_MIN_K"]
