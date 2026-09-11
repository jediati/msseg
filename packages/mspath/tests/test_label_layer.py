"""``RoiLabelLayer``: an item's region ids, addressed in slide coordinates.

The canvas draws in slide pixels whatever item is on screen, so this layer is
what makes a 2048-square ROI at level 0 and a whole level-4 overview composite
in the same space. Everything here is about that mapping being exact and about
the outside of an item reading as background rather than as region 0.
"""
import numpy as np
import pytest

from msseg.mspath.sources import RoiLabelLayer


def labels(h=8, w=8):
    """Ids that encode their own position, so a wrong offset is visible."""
    return (np.arange(h * w, dtype=np.int32).reshape(h, w))


def test_native_level_is_the_ladder_rung_nearest_the_raster():
    assert RoiLabelLayer(labels(), scale=1.0).native_level == 0
    assert RoiLabelLayer(labels(), scale=16.0).native_level == 4
    assert RoiLabelLayer(labels(), scale=128.099).native_level == 7   # floors, never overshoots
    assert RoiLabelLayer(labels(), scale=0.5).native_level == 0


def test_shape_is_the_slide_not_the_item():
    lay = RoiLabelLayer(labels(), origin=(100, 200), scale=1.0, slide_shape=(1000, 900))
    assert lay.shape == (1000, 900)
    assert lay.slide_rect() == (100, 200, 8, 8)


def test_shape_is_derived_when_the_slide_is_not_given():
    lay = RoiLabelLayer(labels(4, 4), origin=(10, 20), scale=16.0)
    assert lay.shape == (20 + 4 * 16, 10 + 4 * 16)


def test_n_ids_is_declared_not_measured():
    """``labels.max()`` over a 16-megapixel overview on every frame is exactly
    the image-sized work this layer exists to avoid."""
    lay = RoiLabelLayer(labels(), n_ids=9999)
    assert lay.n_ids == 9999
    assert RoiLabelLayer(labels(4, 4)).n_ids == 16      # measured when not declared


def test_id_at_is_in_slide_coordinates():
    lay = RoiLabelLayer(labels(), origin=(100, 200), scale=1.0, slide_shape=(1000, 900))
    assert lay.id_at(100, 200) == 0
    assert lay.id_at(103, 201) == 8 + 3
    assert lay.id_at(99, 200) == -1 and lay.id_at(100, 199) == -1
    assert lay.id_at(108, 200) == -1                    # just past the item
    assert lay.id_at(-5, -5) == -1


def test_id_at_under_a_coarse_scale():
    lay = RoiLabelLayer(labels(4, 4), origin=(0, 0), scale=16.0)
    assert lay.id_at(0, 0) == 0
    assert lay.id_at(15, 15) == 0                       # still inside label pixel (0,0)
    assert lay.id_at(16, 0) == 1
    assert lay.id_at(0, 16) == 4


def test_crop_at_level_zero_is_the_raster_at_the_origin():
    lab = labels()
    lay = RoiLabelLayer(lab, origin=(100, 200), scale=1.0, slide_shape=(1000, 900))
    assert np.array_equal(lay.crop(0, 100, 200, 8, 8), lab)


def test_crop_across_the_edge_is_background_outside():
    lab = labels()
    lay = RoiLabelLayer(lab, origin=(100, 200), scale=1.0, slide_shape=(1000, 900))
    got = lay.crop(0, 96, 196, 16, 16)
    assert got.shape == (16, 16)
    assert (got[:4] == -1).all() and (got[:, :4] == -1).all()
    assert (got[12:] == -1).all() and (got[:, 12:] == -1).all()
    assert np.array_equal(got[4:12, 4:12], lab)


def test_crop_entirely_outside_is_all_background():
    lay = RoiLabelLayer(labels(), origin=(100, 200), scale=1.0, slide_shape=(1000, 900))
    assert (lay.crop(0, 0, 0, 16, 16) == -1).all()
    assert (lay.crop(0, 500, 500, 16, 16) == -1).all()


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_crop_samples_the_pixel_under_each_output_centre(level):
    """The canvas's overlay ladder is powers of two of the slide; each output
    pixel must sample the label pixel under its own centre, the same rule
    ``ArrayLabelLayer`` follows for in-memory rasters."""
    lab = labels(16, 16)
    lay = RoiLabelLayer(lab, origin=(32, 64), scale=2.0, slide_shape=(4096, 4096))
    w = h = 6
    x, y = 10, 20
    got = lay.crop(level, x, y, w, h)
    s = float(2 ** level)
    for j in range(h):
        for i in range(w):
            sx = (x + i + 0.5) * s
            sy = (y + j + 0.5) * s
            cx = int(np.floor((sx - 32) / 2.0))
            cy = int(np.floor((sy - 64) / 2.0))
            want = lab[cy, cx] if (0 <= cx < 16 and 0 <= cy < 16) else -1
            assert got[j, i] == want, (level, i, j, cx, cy)


def test_full_is_offered_only_when_the_layer_really_is_the_slide():
    lab = labels(300, 400)
    whole = RoiLabelLayer(lab, origin=(0, 0), scale=1.0, slide_shape=(300, 400))
    assert whole.full() is lab
    # a coarse level, or an offset item, must NOT claim to be the whole slide:
    # the canvas would gather over a raster it would have to invent
    assert RoiLabelLayer(lab, origin=(0, 0), scale=16.0).full() is None
    assert RoiLabelLayer(lab, origin=(5, 0), scale=1.0, slide_shape=(300, 400)).full() is None
    assert RoiLabelLayer(lab, origin=(0, 0), scale=1.0, slide_shape=(9000, 9000)).full() is None


def test_background_ids_survive_the_crop():
    lab = labels(8, 8).copy()
    lab[2:5, 2:5] = -1
    lay = RoiLabelLayer(lab, origin=(0, 0), scale=1.0, slide_shape=(8, 8))
    got = lay.crop(0, 0, 0, 8, 8)
    assert (got[2:5, 2:5] == -1).all()
    assert got.dtype == np.int32


# --------------------------------------------------------------------------- #
# PlacedImageSource: a scalar channel of one item, drawn on the whole slide
# --------------------------------------------------------------------------- #
from msseg.mspath.sources import PlacedImageSource  # noqa: E402


def raster(h=8, w=8):
    return np.arange(h * w, dtype=np.float32).reshape(h, w)


def test_placed_source_spans_the_slide_not_the_item():
    src = PlacedImageSource(raster(), origin=(100, 200), scale=16.0, slide_shape=(4000, 3000))
    assert src.native and src.channels == 1
    assert src.level_shape(0) == (4000, 3000)
    assert src.value_range() == (0.0, 63.0)
    # a ladder deep enough that the top level is a thumbnail, never one level
    assert src.levels > 1 and max(src.level_shape(src.levels - 1)) <= 256 * 2


def test_placed_source_samples_under_each_level_pixel():
    r = raster()
    src = PlacedImageSource(r, origin=(0, 0), scale=2.0, slide_shape=(16, 16))
    got = src.read_region(0, 0, 0, 16, 16)
    assert got.shape == (16, 16)
    # slide pixel (3, 5) sits over raster pixel (1, 2)
    assert got[5, 3] == r[2, 1]
    # at level 1 a level pixel is 2 slide pixels = 1 raster pixel
    assert np.array_equal(src.read_region(1, 0, 0, 8, 8), r)


def test_outside_the_item_is_the_rasters_minimum_not_zero():
    r = raster() + 10.0
    src = PlacedImageSource(r, origin=(100, 100), scale=1.0, slide_shape=(500, 500))
    got = src.read_region(0, 90, 90, 20, 20)
    assert (got[:10, :] == 10.0).all() and (got[:, :10] == 10.0).all()
    assert np.array_equal(got[10:18, 10:18], r)
    assert (src.read_region(0, 0, 0, 4, 4) == 10.0).all()


def test_value_at_is_in_slide_coordinates():
    r = raster()
    src = PlacedImageSource(r, origin=(100, 200), scale=4.0, slide_shape=(1000, 1000))
    assert src.value_at(100, 200) == r[0, 0]
    assert src.value_at(107, 203) == r[0, 1]
    assert src.value_at(99, 200) is None and src.value_at(5000, 5000) is None


def test_a_blank_raster_still_has_a_usable_range():
    src = PlacedImageSource(np.full((4, 4), np.nan, np.float32), slide_shape=(8, 8))
    assert src.value_range() == (0.0, 1.0)
    assert np.isfinite(src.read_region(0, 0, 0, 8, 8)).all()


def test_best_level_never_coarser_than_the_screen():
    src = PlacedImageSource(raster(), scale=16.0, slide_shape=(90000, 47040))
    assert src.best_level(1.0) == 0
    assert src.level_scale(src.best_level(16.0)) <= 16.0
    assert src.level_scale(src.best_level(1e9)) == src.level_scale(src.levels - 1)



# --------------------------------------------------------------------------- #
# gesture resolution on a placed layer is EXACT, whatever the scale
# --------------------------------------------------------------------------- #
from msseg.labeler.labeling import Interaction, touched_ids, touched_ids_over  # noqa: E402


def _it(tool, pts, uid=1):
    return Interaction(uid, "k", 0, 0, tool, pts, 1)


def test_crop_raster_is_a_slice_with_background_outside():
    lab = labels(8, 8)
    lay = RoiLabelLayer(lab, origin=(1000, 2000), scale=128.099, slide_shape=(90000, 47040))
    assert lay.placement() == (1000.0, 2000.0, 128.099)
    assert np.array_equal(lay.crop_raster(2, 3, 4, 2), lab[3:5, 2:6])
    got = lay.crop_raster(-2, 6, 6, 6)
    assert got.shape == (6, 6) and (got[:, :2] == -1).all() and (got[2:, :] == -1).all()
    assert np.array_equal(got[:2, 2:], lab[6:8, 0:4])
    assert lay.crop_raster(50, 50, 4, 4).shape == (4, 4) and (lay.crop_raster(50, 50, 4, 4) == -1).all()


@pytest.mark.parametrize("scale", [1.0, 16.0, 128.099, 515.604])
def test_a_tap_on_every_regions_extremum_hits_it(scale):
    """The overview's extrema are mapped into slide coordinates by
    `x * scale + origin`; resolving a tap there through the canvas ladder's
    nearest power of two drifts half a pixel over a few hundred and missed.
    Through the exact placement it cannot."""
    lab = np.arange(700 * 400, dtype=np.int32).reshape(700, 400)   # one id per pixel
    origin = (12000, 4000)
    lay = RoiLabelLayer(lab, origin=origin, scale=scale, slide_shape=(400000, 400000))
    rng = np.random.default_rng(3)
    for _ in range(200):
        ry, rx = int(rng.integers(700)), int(rng.integers(400))
        px, py = rx * scale + origin[0], ry * scale + origin[1]       # the engine's mapping
        got = touched_ids_over(_it("taps", [(float(px), float(py))]), lay, np)
        assert got == {int(lab[ry, rx])}, (scale, rx, ry, got)


def test_every_gesture_matches_the_raster_at_an_odd_scale():
    lab = np.zeros((20, 20), np.int32)
    for j in range(4):
        for i in range(4):
            lab[j * 5:(j + 1) * 5, i * 5:(i + 1) * 5] = j * 4 + i
    scale, origin = 128.099, (777, 333)
    lay = RoiLabelLayer(lab, origin=origin, scale=scale, slide_shape=(20000, 20000))
    gestures = [("taps", [(2.0, 2.0), (7.0, 12.0)]), ("squiggle", [(1.0, 1.0), (18.0, 1.0)]),
                ("squiggle", [(3.0, 3.0), (16.0, 16.0)]), ("box", [(6.0, 6.0), (13.0, 13.0)]),
                ("box", [(13.0, 13.0), (6.0, 6.0)]), ("polygon", [(2.0, 2.0), (17.0, 3.0), (9.0, 18.0)])]
    for tool, pts in gestures:
        want = touched_ids(_it(tool, pts), lab, np)
        slide_pts = [(x * scale + origin[0], y * scale + origin[1]) for x, y in pts]
        assert touched_ids_over(_it(tool, slide_pts), lay, np) == want, tool


def test_a_box_over_the_whole_item_reads_only_the_raster():
    """Seventeen gigabytes on the real slide, before: the bbox at level 0."""
    lab = labels(20, 20)
    lay = RoiLabelLayer(lab, origin=(0, 0), scale=128.099, slide_shape=(90000, 47040))
    calls = []
    real = lay.crop_raster
    lay.crop_raster = lambda *a: (calls.append(a), real(*a))[1]
    got = touched_ids_over(_it("box", [(0.0, 0.0), (89999.0, 47039.0)]), lay, np)
    assert got == set(int(v) for v in np.unique(lab) if v >= 0)
    assert calls and all(a[2] * a[3] <= 30 * 30 for a in calls), calls
