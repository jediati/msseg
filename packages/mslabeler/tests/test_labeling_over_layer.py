"""``touched_ids_over``: the same answer as ``touched_ids``, off a LabelLayer.

A gesture is rasterized against the item's region ids to decide what it paints.
That has always needed the whole raster, which is fine when the item IS the
image and impossible when it is a rect on a 4-gigapixel slide. The bbox path
must therefore agree with the full-raster path *exactly* -- an annotation that
resolves differently depending on how the ids were fetched is a silent
corruption of somebody's labels -- so every case here is checked against
``touched_ids`` on the same data.
"""
import numpy as np
import pytest

from msseg.labeler.labeling import (Interaction, touched_ids, touched_ids_over,
                                    gesture_bbox, shifted, resolve_slice)
from msseg.labeler.sources import ArrayLabelLayer


def lab(h=20, w=20):
    """4x4 blocks of 5x5, so ids and positions are easy to reason about."""
    out = np.zeros((h, w), np.int32)
    for j in range(4):
        for i in range(4):
            out[j * 5:(j + 1) * 5, i * 5:(i + 1) * 5] = j * 4 + i
    out[0, 0] = -1                       # a background pixel, never selectable
    return out


class CropOnlyLayer:
    """A LabelLayer that refuses to hand over a full raster -- the whole-slide
    case, without needing a slide."""

    def __init__(self, labels, offset=(0, 0), slide_shape=None):
        self._inner = ArrayLabelLayer(labels)
        self.ox, self.oy = offset
        self._shape = slide_shape or (labels.shape[0] + self.oy,
                                      labels.shape[1] + self.ox)
        self.crops = []

    @property
    def shape(self):
        return self._shape

    @property
    def n_ids(self):
        return self._inner.n_ids

    @property
    def rev(self):
        return 0

    def crop(self, level, x, y, w, h):
        self.crops.append((x, y, w, h))
        out = np.full((h, w), -1, np.int32)
        src = self._inner.labels
        sx0, sy0 = max(0, x - self.ox), max(0, y - self.oy)
        sx1 = min(src.shape[1], x - self.ox + w)
        sy1 = min(src.shape[0], y - self.oy + h)
        if sx1 > sx0 and sy1 > sy0:
            out[sy0 - (y - self.oy):sy1 - (y - self.oy),
                sx0 - (x - self.ox):sx1 - (x - self.ox)] = src[sy0:sy1, sx0:sx1]
        return out

    def id_at(self, x, y):
        return self._inner.id_at(x - self.ox, y - self.oy)

    def full(self):
        return None


def make(tool, pts, uid=1):
    return Interaction(uid, "k", 0, 0, tool, pts, 1)


GESTURES = [
    make("taps", [(2.0, 2.0), (7.0, 12.0)]),
    make("taps", [(0.0, 0.0)]),                         # on the background pixel
    make("squiggle", [(1.0, 1.0), (18.0, 1.0)]),
    make("squiggle", [(3.0, 3.0), (16.0, 16.0)]),
    make("squiggle", [(12.0, 12.0)]),                   # a click
    make("box", [(6.0, 6.0), (13.0, 13.0)]),
    make("box", [(13.0, 13.0), (6.0, 6.0)]),            # drawn backwards
    make("box", [(2.0, 2.0), (2.0, 2.0)]),              # zero area
    make("polygon", [(2.0, 2.0), (17.0, 3.0), (9.0, 18.0)]),
    make("polygon", [(2.0, 2.0), (17.0, 3.0)]),         # degenerate
]


@pytest.mark.parametrize("it", GESTURES, ids=lambda i: f"{i.tool}-{len(i.points)}")
def test_layer_path_matches_the_full_raster_path(it):
    labels = lab()
    want = touched_ids(it, labels, np)
    assert touched_ids_over(it, CropOnlyLayer(labels), np) == want
    # a layer that CAN hand over its raster takes the old path, same answer
    assert touched_ids_over(it, ArrayLabelLayer(labels), np) == want


@pytest.mark.parametrize("it", GESTURES, ids=lambda i: f"{i.tool}-{len(i.points)}")
def test_the_item_may_sit_anywhere_on_a_much_larger_canvas(it):
    """The gesture arrives in image coordinates; the ids live at an offset on a
    canvas orders of magnitude bigger. Shifting both must not change what is
    painted."""
    labels = lab()
    want = touched_ids(it, labels, np)
    off = (30000, 40000)
    layer = CropOnlyLayer(labels, offset=off, slide_shape=(90000, 47040))
    moved = shifted(it, off[0], off[1])
    assert touched_ids_over(moved, layer, np) == want


def test_only_the_gestures_own_box_is_fetched():
    """The point of the exercise: a small gesture must not read a slide-sized
    rect."""
    labels = lab()
    layer = CropOnlyLayer(labels, offset=(30000, 40000), slide_shape=(90000, 47040))
    it = shifted(make("box", [(6.0, 6.0), (9.0, 9.0)]), 30000, 40000)
    touched_ids_over(it, layer, np)
    assert len(layer.crops) == 1
    _x, _y, w, h = layer.crops[0]
    assert w <= 8 and h <= 8, layer.crops


def test_a_gesture_entirely_outside_the_item_paints_nothing():
    labels = lab()
    layer = CropOnlyLayer(labels, offset=(1000, 1000), slide_shape=(5000, 5000))
    assert touched_ids_over(make("box", [(10.0, 10.0), (20.0, 20.0)]), layer, np) == set()
    assert touched_ids_over(make("taps", [(4000.0, 4000.0)]), layer, np) == set()


def test_a_gesture_straddling_the_items_edge_keeps_what_is_inside():
    labels = lab()
    layer = CropOnlyLayer(labels, offset=(100, 100), slide_shape=(500, 500))
    it = make("box", [(90.0, 90.0), (107.0, 107.0)])      # half off the item
    got = touched_ids_over(it, layer, np)
    assert got == touched_ids(make("box", [(0.0, 0.0), (7.0, 7.0)]), labels, np)
    assert -1 not in got and got, got


def test_gesture_bbox_always_has_area():
    assert gesture_bbox(make("taps", [(5.0, 5.0)]), np) == (4, 4, 3, 3)
    x, y, w, h = gesture_bbox(make("squiggle", [(2.0, 7.0), (9.0, 7.0)]), np)
    assert (w > 0 and h > 0) and y <= 7 <= y + h
    assert gesture_bbox(make("taps", []), np) is None


def test_resolve_slice_agrees_through_either_path():
    labels = lab()
    its = [make("box", [(1.0, 1.0), (8.0, 8.0)], uid=1),
           make("squiggle", [(11.0, 2.0), (18.0, 2.0)], uid=2),
           make("taps", [(12.0, 12.0)], uid=3)]
    want = resolve_slice(its, labels, np)
    got = resolve_slice(its, labels, np, layer=CropOnlyLayer(labels))
    assert np.array_equal(got, want)


def test_shifted_keeps_everything_but_the_points():
    it = Interaction(7, "key", 3, 4, "box", [(1.0, 2.0)], 5, {"tool": "magic"})
    out = shifted(it, 10, 20)
    assert out.points == [(11.0, 22.0)]
    assert (out.uid, out.slice_key, out.si, out.li, out.tool, out.class_id) == \
        (7, "key", 3, 4, "box", 5)
    assert out.meta == {"tool": "magic"}
