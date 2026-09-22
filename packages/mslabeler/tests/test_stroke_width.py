"""A stroke keeps the width the user saw: ``meta["px"]`` (slide px per screen
px at draw time) over the raster's scale is the width in raster px, applied
when it exceeds a hairline. Gestures without it are the hairlines they were."""
import numpy as np

from msseg.labeler.labeling import (Interaction, gesture_bbox, resolve_opts, stroke_mask,
                                     touched_ids, touched_ids_over)
from msseg.labeler.sources import ArrayLabelLayer


def rows_raster():
    """20x20: rows 0-9 are region 1, rows 10-19 region 2, background border."""
    lab = np.full((20, 20), -1, np.int32)
    lab[1:10, 1:19] = 1
    lab[10:19, 1:19] = 2
    return lab


def _sq(pts, meta=None):
    return Interaction(1, "k", 0, 0, "squiggle", pts, 1, meta=meta)


def test_a_wide_stroke_reaches_what_a_hairline_misses():
    lab = rows_raster()
    stroke = [(3.0, 8.0), (16.0, 8.0)]            # along the last row of region 1
    assert touched_ids(_sq(stroke), lab, np) == {1}
    assert touched_ids(_sq(stroke), lab, np, width=5.0) == {1, 2}
    assert touched_ids(_sq(stroke), lab, np, width=1.4) == {1}, "up to 1.5 px is the hairline"
    # A click with a width is a disk.
    assert touched_ids(_sq([(9.0, 9.0)]), lab, np, width=4.0) == {1, 2}
    assert touched_ids(_sq([(9.0, 9.0)]), lab, np) == {1}
    # Outside the raster: nothing, never an error.
    assert touched_ids(_sq([(-30.0, -30.0), (-20.0, -30.0)]), lab, np, width=6.0) == set()


def test_stroke_mask_shape_and_bounds():
    m = stroke_mask([(3.0, 8.0), (16.0, 8.0)], 5.0, 20, 20, np)
    assert m is not None
    mask, ya, xa = m
    assert mask.dtype == bool and mask.any()
    assert ya >= 0 and xa >= 0 and ya + mask.shape[0] <= 20 and xa + mask.shape[1] <= 20
    assert stroke_mask([(100.0, 100.0)], 3.0, 20, 20, np) is None


def test_resolve_opts_from_meta():
    assert resolve_opts(_sq([(0, 0)]), 1.0) == {"width": None, "off_level": False}
    o = resolve_opts(_sq([(0, 0)], meta={"px": 8.0, "scale": 16.0}), 16.0)
    assert o == {"width": 0.5, "off_level": False}, "one screen px at 1/16 is half a raster px"
    o = resolve_opts(_sq([(0, 0)], meta={"px": 8.0, "scale": 16.0}), 1.0)
    assert o == {"width": 8.0, "off_level": True}, "the same stroke on a level-0 raster"
    assert resolve_opts(_sq([(0, 0)], meta={"px": True}), 1.0)["width"] is None


def test_width_flows_through_the_layer_path_and_the_bbox():
    lab = rows_raster()
    layer = ArrayLabelLayer(lab)
    stroke = [(3.0, 8.0), (16.0, 8.0)]
    assert touched_ids_over(_sq(stroke), layer, np) == {1}
    assert touched_ids_over(_sq(stroke, meta={"px": 5.0}), layer, np) == {1, 2}
    # The crop the gesture asks for grows by its radius only.
    assert gesture_bbox(_sq(stroke), np) == (2, 7, 16, 3)
    assert gesture_bbox(_sq(stroke, meta={"px": 5.0}), np) == (-1, 4, 22, 9), "pad = 1 + ceil(px/2)"


class _Placed:
    """A layer whose raster sits at an origin and a scale (the mspath shape
    of things), served by crop; no full raster."""
    def __init__(self, raster, origin, scale):
        self._r, self._o, self._s = raster, origin, float(scale)
        self.shape = (int(raster.shape[0] * scale), int(raster.shape[1] * scale))
        self.native_level = 0

    def full(self):
        return None

    def placement(self):
        return self._o[0], self._o[1], self._s

    @property
    def raster_shape(self):
        return self._r.shape

    def crop_raster(self, x, y, w, h):
        return self._r[y:y + h, x:x + w]


def test_width_on_a_placed_coarse_raster():
    lab = rows_raster()
    layer = _Placed(lab, (1000, 2000), 16.0)
    # Drawn on this very raster at 4 slide px per screen px: a quarter of a
    # raster pixel wide -> the hairline it always was.
    stroke = [(1000 + 3 * 16.0, 2000 + 8 * 16.0), (1000 + 16 * 16.0, 2000 + 8 * 16.0)]
    on = _sq(stroke, meta={"level": 4, "scale": 16.0, "px": 4.0})
    assert touched_ids_over(on, layer, np) == {1}
    # The same stroke drawn zoomed out (64 slide px per screen px) covers four
    # raster px -- and, drawn at level 0 and applied here, 4 px too.
    assert touched_ids_over(_sq(stroke, meta={"scale": 16.0, "px": 64.0}), layer, np) == {1, 2}
    assert touched_ids_over(_sq(stroke, meta={"scale": 1.0, "px": 64.0}), layer, np) == {1, 2}
