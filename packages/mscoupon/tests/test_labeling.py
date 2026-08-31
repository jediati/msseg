"""Pure-function tests for the labeler's interaction model (msseg.mscoupon.labeling).

Runs headless with numpy + PIL only (no Tk, no compiled extension):

    pytest packages/mscoupon/tests/test_labeling.py
"""
import numpy as np
import pytest

from msseg.mscoupon import labeling
from msseg.mscoupon.labeling import (LabelStore, Interaction, touched_ids,
                                     resolve_slice, class_lut, line_pixels,
                                     scalar_lut, CLASS_COLORS)


def blocks_raster():
    """20x20 raster: four 10x10 blocks carrying SPARSE living ids {0, 2, 5, 9}
    (the ids are compact-base ids of surviving extrema, so never assume dense),
    with a -1 background border two pixels wide."""
    lab = np.full((20, 20), -1, np.int32)
    lab[2:10, 2:10] = 0      # top-left
    lab[2:10, 10:18] = 2     # top-right
    lab[10:18, 2:10] = 5     # bottom-left
    lab[10:18, 10:18] = 9    # bottom-right
    return lab


def make(tool, points, class_id=1, uid=1, key="s0.tiff"):
    return Interaction(uid, key, 0, 0, tool, points, class_id)


# --------------------------------------------------------------------------- #
# touched_ids
# --------------------------------------------------------------------------- #
def test_squiggle_touches_crossed_blocks_only():
    lab = blocks_raster()
    # Horizontal stroke through the two top blocks.
    ids = touched_ids(make("squiggle", [(3.0, 5.0), (15.0, 5.0)]), lab, np)
    assert ids == {0, 2}


def test_squiggle_ignores_background_and_out_of_bounds():
    lab = blocks_raster()
    # Runs from outside the image through the border into the top-left block.
    ids = touched_ids(make("squiggle", [(-5.0, 5.0), (5.0, 5.0)]), lab, np)
    assert ids == {0}


def test_squiggle_diagonal_covers_every_unit_step():
    lab = blocks_raster()
    ids = touched_ids(make("squiggle", [(3.0, 3.0), (16.0, 16.0)]), lab, np)
    assert ids == {0, 9}    # diagonal through TL and BR, skirting TR/BL corners


def test_single_click_squiggle_samples_the_point():
    lab = blocks_raster()
    assert touched_ids(make("squiggle", [(12.0, 12.0)]), lab, np) == {9}


def test_box_picks_intersecting_regions():
    lab = blocks_raster()
    # Box straddling all four blocks.
    ids = touched_ids(make("box", [(8.0, 8.0), (12.0, 12.0)]), lab, np)
    assert ids == {0, 2, 5, 9}
    # Box wholly inside one block.
    assert touched_ids(make("box", [(3.0, 3.0), (7.0, 7.0)]), lab, np) == {0}
    # Reversed corners span the same rectangle.
    assert touched_ids(make("box", [(12.0, 12.0), (8.0, 8.0)]), lab, np) == {0, 2, 5, 9}


def test_taps_sample_points_independently():
    lab = blocks_raster()
    # Three taps in three blocks: no connecting segments, so the block the
    # (5,15)-(15,5) diagonal would cross is NOT picked up...
    pts = [(5.0, 5.0), (5.0, 15.0), (15.0, 5.0)]
    assert touched_ids(make("taps", pts), lab, np) == {0, 5, 2}
    # ...whereas a squiggle through the same points is.
    assert 9 in touched_ids(make("squiggle", pts), lab, np)
    assert touched_ids(make("taps", [(-3.0, -3.0)]), lab, np) == set()


def test_polygon_picks_enclosed_regions_only():
    lab = blocks_raster()
    # Triangle over the top-left block only.
    ids = touched_ids(make("polygon", [(3.0, 3.0), (8.0, 3.0), (3.0, 8.0)]), lab, np)
    assert ids == {0}
    # Fewer than 3 points is not a polygon.
    assert touched_ids(make("polygon", [(3.0, 3.0), (8.0, 3.0)]), lab, np) == set()


def test_line_pixels_clips_to_bounds():
    ys, xs = line_pixels(-3.0, -3.0, 4.0, 4.0, 20, 20, np)
    assert (xs >= 0).all() and (ys >= 0).all()
    assert (xs == ys).all()          # the diagonal
    assert xs.max() == 4


# --------------------------------------------------------------------------- #
# resolve_slice: paint-over ordering
# --------------------------------------------------------------------------- #
def test_later_interaction_paints_over_earlier():
    lab = blocks_raster()
    a = make("squiggle", [(3.0, 5.0), (15.0, 5.0)], class_id=1, uid=1)   # {0, 2} -> 1
    b = make("squiggle", [(12.0, 4.0), (12.0, 6.0)], class_id=2, uid=2)  # {2} -> 2
    rc = resolve_slice([a, b], lab, np)
    assert rc[0] == 1        # only the first touched it
    assert rc[2] == 2        # the later interaction wins
    assert rc[5] == 0 and rc[9] == 0
    # Order comes from uid, not list position.
    rc = resolve_slice([b, a], lab, np)
    assert rc[0] == 1 and rc[2] == 2


def test_resolve_sized_by_max_id():
    lab = blocks_raster()
    rc = resolve_slice([], lab, np)
    assert rc.shape == (10,) and (rc == 0).all()    # ids go up to 9 -> K = 10


# --------------------------------------------------------------------------- #
# class_lut
# --------------------------------------------------------------------------- #
def test_class_lut_colors_and_alpha():
    rc = np.array([0, 1, 0, 2], np.uint8)
    lut = class_lut(rc, np)
    assert lut.shape == (4, 4) and lut.dtype == np.uint8
    assert tuple(lut[1]) == CLASS_COLORS[1]
    assert tuple(lut[3]) == CLASS_COLORS[2]
    assert lut[0, 3] == 0 and lut[2, 3] == 0        # unlabeled -> transparent


# --------------------------------------------------------------------------- #
# LabelStore
# --------------------------------------------------------------------------- #
def test_store_add_orders_and_bumps_rev():
    store = LabelStore(n_classes=3)
    r0 = store.rev
    a = store.add("squiggle", [(0, 0), (1, 1)], 1, "s0.tiff", 0, 0)
    b = store.add("box", [(0, 0), (5, 5)], 2, "s1.tiff", 0, 1)
    assert (a.uid, b.uid) == (1, 2)
    assert store.rev > r0
    assert [it.uid for it in store.for_slice("s0.tiff")] == [1]
    with pytest.raises(ValueError):
        store.add("squiggle", [(0, 0)], 3, "s0.tiff")   # class 3 doesn't exist
    with pytest.raises(ValueError):
        store.add("scribble", [(0, 0)], 1, "s0.tiff")   # unknown tool


def test_store_remove_and_set_class():
    store = LabelStore(n_classes=4)
    a = store.add("squiggle", [(0, 0), (1, 1)], 1, "s0.tiff")
    r = store.rev
    store.set_class(a.uid, 3)
    assert store.get(a.uid).class_id == 3 and store.rev > r
    r = store.rev
    store.set_class(a.uid, 3)            # no-op: same class
    assert store.rev == r
    store.remove(a.uid)
    assert store.get(a.uid) is None
    r = store.rev
    store.remove(999)                    # no-op: unknown uid
    assert store.rev == r


def test_set_n_classes_clamps_orphans():
    store = LabelStore(n_classes=5)
    a = store.add("squiggle", [(0, 0), (1, 1)], 4, "s0.tiff")
    b = store.add("squiggle", [(0, 0), (1, 1)], 1, "s0.tiff")
    changed = store.set_n_classes(2)
    assert changed == [a.uid]
    assert store.get(a.uid).class_id == 1 and store.get(b.uid).class_id == 1
    with pytest.raises(ValueError):
        store.set_n_classes(1)
    with pytest.raises(ValueError):
        store.set_n_classes(labeling.MAX_CLASSES + 1)


def test_custom_class_colors():
    store = LabelStore(n_classes=3)
    store.set_color(1, "#00ffee")
    assert store.color(1) == "#00ffee"
    assert store.rgba(1) == (0, 255, 238, 255)
    assert store.color(2) == labeling.class_color_hex(2), "others keep defaults"
    r0 = store.rev
    store.set_color(1, "#00ffee")            # no-op: same color
    assert store.rev == r0
    doc = store.to_json()
    assert {"id": 1, "color": "#00ffee"} in doc["classes"]
    back = LabelStore.from_json(doc)
    assert back.color(1) == "#00ffee"
    colors = np.asarray([back.rgba(k) for k in range(labeling.MAX_CLASSES)],
                        np.uint8)
    lut = class_lut(np.array([0, 1, 2], np.uint8), np, colors)
    assert tuple(lut[1]) == (0, 255, 238, 255)
    assert tuple(lut[2]) == CLASS_COLORS[2]
    assert store.rgba(1) != CLASS_COLORS[1]
    store.set_color(1, "junk")               # unusable hex -> default at render
    assert store.rgba(1) == CLASS_COLORS[1]


def test_json_round_trip():
    store = LabelStore(n_classes=4)
    store.add("squiggle", [(1.5, 2.25), (3.0, 4.0)], 1, "s0.tiff", 0, 0)
    store.add("box", [(0.0, 0.0), (5.0, 5.0)], 3, "s1.tiff", 0, 1)
    doc = store.to_json()
    assert doc["version"] == 2 and doc["n_classes"] == 4
    back = LabelStore.from_json(doc)
    assert back.n_classes == 4
    assert [it.uid for it in back.interactions] == [1, 2]
    assert back.interactions[0].points == [(1.5, 2.25), (3.0, 4.0)]
    assert back.interactions[1].class_id == 3
    # New interactions continue the uid sequence.
    c = back.add("polygon", [(0, 0), (1, 0), (0, 1)], 2, "s0.tiff")
    assert c.uid == 3


def test_rebind_qualified_keys():
    store = LabelStore(n_classes=3)
    store.add("squiggle", [(0, 0), (1, 1)], 1, "spears/b.tiff", 5, 5)  # stale hints
    store.add("squiggle", [(0, 0), (1, 1)], 2, "spears/gone.tiff", 0, 0)
    subseqs = [{"name": "s", "folder": "spears",
                "files": [r"C:\x\a.tiff", r"C:\x\b.tiff"]}]
    unbound = store.rebind(subseqs)
    assert unbound == 1
    it = store.for_slice("spears/b.tiff")[0]
    assert (it.si, it.li) == (0, 1) and it.bound
    assert not store.for_slice("spears/gone.tiff")[0].bound


def test_rebind_migrates_unambiguous_legacy_keys():
    store = LabelStore(n_classes=3)
    store.add("squiggle", [(0, 0), (1, 1)], 1, "b.tiff")     # v1 bare basename
    subseqs = [{"name": "s", "folder": "spears",
                "files": [r"C:\x\a.tiff", r"C:\x\b.tiff"]}]
    assert store.rebind(subseqs) == 0
    it = store.interactions[0]
    assert it.slice_key == "spears/b.tiff", "legacy key upgraded in place"
    assert (it.si, it.li) == (0, 1)


def test_rebind_leaves_ambiguous_legacy_keys_unbound():
    store = LabelStore(n_classes=3)
    store.add("squiggle", [(0, 0), (1, 1)], 1, "a.tiff")     # v1 bare basename
    subseqs = [{"name": "s1", "folder": "spears", "files": [r"C:\x\a.tiff"]},
               {"name": "s2", "folder": "tomo", "files": [r"C:\y\a.tiff"]}]
    assert store.rebind(subseqs) == 1
    it = store.interactions[0]
    assert not it.bound and it.slice_key == "a.tiff", "ambiguous: kept, not guessed"


def test_rebind_qualified_beats_basename_fallback():
    store = LabelStore(n_classes=3)
    store.add("squiggle", [(0, 0), (1, 1)], 1, "tomo/a.tiff")
    subseqs = [{"name": "s1", "folder": "spears", "files": [r"C:\x\a.tiff"]},
               {"name": "s2", "folder": "tomo", "files": [r"C:\y\a.tiff"]}]
    assert store.rebind(subseqs) == 0
    assert (store.interactions[0].si, store.interactions[0].li) == (1, 0)


# --------------------------------------------------------------------------- #
# scalar_lut: the continuous regions coloring modes (class probability,
# prediction uncertainty) share one ramp with the id LUT's canvas contract.
# --------------------------------------------------------------------------- #
def test_scalar_lut_shape_and_endpoints():
    lut = scalar_lut([0.0, 0.5, 1.0], np)
    assert lut.shape == (3, 4) and lut.dtype == np.uint8
    assert tuple(lut[0][:3]) == (49, 54, 149), "ramp floor"
    assert tuple(lut[2][:3]) == (165, 15, 21), "ramp ceiling"
    assert list(lut[:, 3]) == [255, 255, 255], "opaque unless masked"


def test_scalar_lut_is_monotone_and_clamps():
    v = np.linspace(0.0, 1.0, 32)
    lut = scalar_lut(v, np)
    # Red rises and blue falls across the ramp: neither is flat, so distinct
    # probabilities never collide on one color.
    assert lut[-1, 0] > lut[0, 0] and lut[-1, 2] < lut[0, 2]
    assert (scalar_lut([-5.0], np)[0] == scalar_lut([0.0], np)[0]).all()
    assert (scalar_lut([9.0], np)[0] == scalar_lut([1.0], np)[0]).all()


def test_scalar_lut_alpha_and_mask():
    lut = scalar_lut([0.2, 0.8, 0.5], np, alpha=150,
                     mask=[True, False, True])
    assert list(lut[:, 3]) == [150, 0, 150], "unscored regions stay invisible"
    assert list(scalar_lut([0.5], np, alpha=999)[:, 3]) == [255], "alpha clamps"
