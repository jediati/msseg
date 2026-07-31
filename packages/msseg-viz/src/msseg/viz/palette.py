"""Shared color palettes for MSSeg viewers.

The per-minimum palette assigns each Morse minimum a deterministic, well-spread
color via a golden-ratio hue walk. It is shared by the merge-tree icicle and the
segmentation overlays so a given minimum has the same color everywhere.
"""
import colorsys

import numpy as np

_MIN_K = 4096
_MIN_LUT = np.array(
    [colorsys.hsv_to_rgb((i * 0.6180339887) % 1.0, 0.55, 0.92) for i in range(_MIN_K)],
    dtype=np.float32,
)


def min_color(node_id):
    """Deterministic RGB for a minimum NodeId (shared tree/slice palette)."""
    return tuple(_MIN_LUT[int(node_id) % _MIN_K])
