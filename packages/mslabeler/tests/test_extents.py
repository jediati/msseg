"""Extents: a region set's outline as closed loops, the even-odd fill, and a
taps gesture resolving by its outline off the level it was drawn on."""
import numpy as np

from msseg.labeler.extents import extent_mask, outline_loops, outline_of_ids
from msseg.labeler.labeling import Interaction, Placement, touched_ids, touched_ids_over


def _painted(loops, w, h):
    em = extent_mask(loops, w, h, np)
    if em is None:
        return np.zeros((h, w), bool)
    mask, ya, xa = em
    full = np.zeros((h, w), bool)
    full[ya:ya + mask.shape[0], xa:xa + mask.shape[1]] = mask
    return full


def test_one_block_is_one_loop_and_fills_itself():
    m = np.zeros((6, 6), bool)
    m[1:3, 1:3] = True                          # a 2x2 block
    loops = outline_loops(m, np)
    assert len(loops) == 1
    assert sorted(loops[0]) == [(1, 1), (1, 3), (3, 1), (3, 3)], "four corners, collinear runs gone"
    assert (_painted(loops, 6, 6) == m).all(), "a loop around a pixel fills that pixel"


def test_two_adjacent_regions_share_one_outline():
    lab = np.full((8, 8), -1, np.int32)
    lab[2:6, 1:4] = 3
    lab[2:6, 4:7] = 7
    loops = outline_of_ids(lab, [3, 7], np)
    assert len(loops) == 1
    xs = [p[0] for p in loops[0]]; ys = [p[1] for p in loops[0]]
    assert (min(xs), min(ys), max(xs), max(ys)) == (1, 2, 7, 6)
    # Only one of them: the shared edge is now the boundary.
    one = outline_of_ids(lab, [3], np)
    assert len(one) == 1 and max(p[0] for p in one[0]) == 4


def test_a_hole_is_a_second_loop_left_unpainted():
    m = np.zeros((10, 10), bool)
    m[1:8, 1:8] = True
    m[3:5, 3:5] = False                         # a 2x2 hole
    loops = outline_loops(m, np)
    assert len(loops) == 2
    painted = _painted(loops, 10, 10)
    assert (painted == m).all(), "even-odd: the hole stays a hole"
    assert painted.sum() == 49 - 4


def test_a_checkerboard_touch_still_closes():
    m = np.zeros((5, 5), bool)
    m[1, 1] = m[2, 2] = True                     # two pixels meeting at a corner
    loops = outline_loops(m, np)
    assert loops and all(len(l) >= 4 for l in loops)
    painted = _painted(loops, 5, 5)
    assert (painted == m).all()


def test_outline_lifts_through_a_placement():
    lab = np.full((4, 4), -1, np.int32)
    lab[1:3, 1:3] = 0
    loops = outline_of_ids(lab, [0], np, Placement((100, 200), 4.0))
    assert len(loops) == 1
    assert sorted(tuple(p) for p in loops[0]) == [(104.0, 204.0), (104.0, 212.0),
                                                  (112.0, 204.0), (112.0, 212.0)]
    assert outline_of_ids(lab, [], np) == [] and outline_of_ids(lab, [99], np) == []


def test_off_level_taps_resolve_by_outline():
    # The drawing raster: one 4x4 region (id 1) in a 6x6 field.
    coarse = np.full((6, 6), -1, np.int32)
    coarse[1:5, 1:5] = 1
    outline = outline_of_ids(coarse, [1], np)
    seed = [(2.0, 2.0)]
    # The target: the same field twice as fine, the block split into four.
    fine = np.full((12, 12), -1, np.int32)
    fine[2:6, 2:6], fine[2:6, 6:10], fine[6:10, 2:6], fine[6:10, 6:10] = 10, 11, 12, 13
    fine_outline = [[[x * 2, y * 2] for x, y in loop] for loop in outline]
    it = Interaction(1, "k", 0, 0, "taps", [(4.0, 4.0)], 1, meta={"outline": fine_outline})
    assert touched_ids(it, fine, np) == {10}, "the seed alone lands in one quarter"
    assert touched_ids(it, fine, np, off_level=True) == {10, 11, 12, 13}, "the outline gets them all"
    # On its own raster the outline is not consulted: byte-identical to the seed.
    own = Interaction(1, "k", 0, 0, "taps", seed, 1, meta={"outline": outline})
    assert touched_ids(own, coarse, np, off_level=True) == {1} == touched_ids(own, coarse, np)


class _Placed:
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


def test_outline_transforms_through_the_layer_path():
    # Drawn on a level-2 raster (scale 4): a 4x4 region whose outline, in slide
    # px, is the square (4,4)-(20,20). Applied to a level-0 raster (scale 1)
    # where that square holds four regions.
    fine = np.full((24, 24), -1, np.int32)
    fine[4:12, 4:12], fine[4:12, 12:20], fine[12:20, 4:12], fine[12:20, 12:20] = 1, 2, 3, 4
    layer = _Placed(fine, (0, 0), 1.0)
    outline = [[[4, 4], [20, 4], [20, 20], [4, 20]]]
    it = Interaction(1, "k", 0, 0, "taps", [(6.0, 6.0)], 1,
                     meta={"level": 2, "scale": 4.0, "outline": outline})
    assert touched_ids_over(it, layer, np) == {1, 2, 3, 4}
    same_level = Interaction(1, "k", 0, 0, "taps", [(6.0, 6.0)], 1,
                             meta={"level": 0, "scale": 1.0, "outline": outline})
    assert touched_ids_over(same_level, layer, np) == {1}, "on its own level the seed decides"
