"""The seam graph reference (msseg.labeler.seams): crack ids, junction rule,
canonical form, snapping, the pixel raster and the LUTs. Headless, numpy only:

    pytest packages/mslabeler/tests/test_seams.py
"""
import numpy as np
import pytest

from msseg.labeler import seams
from msseg.labeler.labeling import Placement
from msseg.labeler.seams import (SeamGraph, crack_id, cracks_of_polyline,
                                 nearest_seam_point, seam_class_lut,
                                 seam_pixel_raster, seams_from_labels,
                                 SEAM_BOUNDARY, SEAM_INTERIOR)


def blocks(n=12, half=6):
    lab = np.zeros((n, n), np.int32)
    lab[:half, half:] = 1
    lab[half:, :half] = 2
    lab[half:, half:] = 3
    return lab


def check_invariants(g, lab):
    # Every crack of the raster is on exactly one seam.
    p = np.concatenate([lab[:, :-1].ravel(), lab[:-1, :].ravel()])
    q = np.concatenate([lab[:, 1:].ravel(), lab[1:, :].ravel()])
    n_cracks = int(((p != q) & (p >= 0) & (q >= 0)).sum())
    assert int(g.lengths(np).sum()) == n_cracks
    ids, _ = g.all_cracks(np)
    assert len(np.unique(ids)) == n_cracks
    # Unit axis-aligned steps, flanks a < b, sorted.
    for i in range(g.n_seams):
        pts = g.seam_points(i)
        d = np.abs(np.diff(pts, axis=0)).sum(axis=1)
        assert (d == 1).all()
        assert g.a[i] < g.b[i]
        if g.j0[i] < 0:
            assert g.j1[i] < 0 and (pts[0] == pts[-1]).all()
            assert pts[1, 0] == pts[0, 0] + 1 and pts[1, 1] == pts[0, 1]
            assert tuple(pts[0][::-1]) == min(tuple(p[::-1]) for p in pts[:-1])
        else:
            assert g.j0[i] <= g.j1[i]
            assert (g.junction_xy[g.j0[i]] == pts[0]).all()
            assert (g.junction_xy[g.j1[i]] == pts[-1]).all()
    keys = [(int(g.a[i]), int(g.b[i]), int(g.j0[i]), int(g.j1[i]),
             int(g.seam_points(i)[0, 1]), int(g.seam_points(i)[0, 0]))
            for i in range(g.n_seams)]
    assert keys == sorted(keys)
    jy = g.junction_xy[:, 1] * (g.width + 1) + g.junction_xy[:, 0]
    assert (np.diff(jy) > 0).all()


# --------------------------------------------------------------------------- #
# crack ids
# --------------------------------------------------------------------------- #
def test_crack_ids_are_label_independent_and_unique():
    w = 5
    assert crack_id(2, 3, 3, 3, w) == 2 * (3 * 6 + 2)
    assert crack_id(3, 3, 2, 3, w) == crack_id(2, 3, 3, 3, w)
    assert crack_id(2, 3, 2, 4, w) == 2 * (3 * 6 + 2) + 1
    with pytest.raises(ValueError):
        crack_id(0, 0, 1, 1, w)
    ids = cracks_of_polyline([(0, 2), (4, 2), (4, 5)], w, 6, np)
    assert ids.tolist() == [crack_id(x, 2, x + 1, 2, w) for x in range(4)] + \
        [crack_id(4, y, 4, y + 1, w) for y in range(2, 5)]
    # A diagonal step walks x first, then y; steps off the lattice are dropped.
    assert cracks_of_polyline([(0, 0), (1, 1)], w, 6, np).tolist() == \
        [crack_id(0, 0, 1, 0, w), crack_id(1, 0, 1, 1, w)]
    assert len(cracks_of_polyline([(-3, 0), (0, 0)], w, 6, np)) == 0
    assert len(cracks_of_polyline([(1, 1)], w, 6, np)) == 0


# --------------------------------------------------------------------------- #
# the junction rule and the canonical form
# --------------------------------------------------------------------------- #
def test_four_blocks():
    lab = blocks()
    g = SeamGraph.from_labels(lab, np)
    check_invariants(g, lab)
    assert g.n_seams == 4 and g.n_junctions == 5
    assert [(int(a), int(b)) for a, b in zip(g.a, g.b)] == [(0, 1), (0, 2), (1, 3), (2, 3)]
    assert g.lengths(np).tolist() == [6, 6, 6, 6]
    # The centre junction (6, 6) is on every seam; edge junctions are degree 1.
    assert (6, 6) in {tuple(p) for p in g.junction_xy}
    assert set(map(tuple, g.junction_xy)) == {(6, 0), (0, 6), (6, 6), (12, 6), (6, 12)}


def test_island_is_one_loop():
    lab = np.zeros((10, 10), np.int32)
    lab[3:7, 2:5] = 1
    g = SeamGraph.from_labels(lab, np)
    check_invariants(g, lab)
    assert g.n_seams == 1 and g.n_junctions == 0
    assert g.j0[0] == -1 and g.j1[0] == -1
    pts = g.seam_points(0)
    assert len(pts) == 15 and (pts[0] == pts[-1]).all()
    assert pts[0].tolist() == [2, 3] and pts[1].tolist() == [3, 3]


def test_region_against_background_ends_at_degree_one():
    lab = np.full((8, 8), -1, np.int32)
    lab[2:6, 1:4] = 3
    lab[2:6, 4:7] = 7
    g = SeamGraph.from_labels(lab, np)
    check_invariants(g, lab)
    assert g.n_seams == 1 and g.n_junctions == 2
    assert (g.a[0], g.b[0]) == (3, 7) and g.lengths(np)[0] == 4
    assert g.seam_points(0).tolist() == [[4, 2], [4, 3], [4, 4], [4, 5], [4, 6]]


def test_degree_two_corner_with_differing_pairs_is_a_junction():
    # TL=0 TR=1 / BL=-1 BR=2 around corner (1, 1): cracks 0|1 (up) and 1|2 (right).
    lab = np.array([[0, 1],
                    [-1, 2]], np.int32)
    g = SeamGraph.from_labels(lab, np)
    check_invariants(g, lab)
    assert g.n_seams == 2
    assert (1, 1) in {tuple(p) for p in g.junction_xy}
    # Whereas a plain turn of the same pair is interior to one seam.
    lab2 = np.array([[0, 1],
                     [1, 1]], np.int32)
    g2 = SeamGraph.from_labels(lab2, np)
    check_invariants(g2, lab2)
    assert g2.n_seams == 1 and g2.lengths(np)[0] == 2


def test_self_seam_from_a_junction_back_to_itself():
    # Region 1 hangs off region 0's boundary with 2 at one corner: the 0|1
    # seam leaves junction and returns to it.
    lab = np.zeros((6, 6), np.int32)
    lab[1:4, 1:4] = 1
    lab[0, 0] = 2
    lab[0, 1] = 2
    lab[1, 0] = 2
    g = SeamGraph.from_labels(lab, np)
    check_invariants(g, lab)
    for i in range(g.n_seams):
        if g.j0[i] == g.j1[i] and g.j0[i] >= 0:
            pts = g.seam_points(i)
            assert tuple(pts[1][::-1]) < tuple(pts[-2][::-1])


def test_random_rasters_hold_the_invariants():
    rng = np.random.default_rng(7)
    for _ in range(25):
        h, w = rng.integers(3, 30, size=2)
        k = int(rng.integers(1, 8))
        seeds = rng.integers(0, [w, h], size=(k, 2))
        ys, xs = np.mgrid[0:h, 0:w]
        d = (xs[..., None] - seeds[:, 0]) ** 2 + (ys[..., None] - seeds[:, 1]) ** 2
        lab = np.argmin(d, axis=-1).astype(np.int32) * 3   # sparse ids
        lab[rng.random((h, w)) < 0.08] = -1
        if rng.random() < 0.5:
            lab[0, :] = -1
        if rng.random() < 0.3:
            lab[h // 2, w // 2] = 999               # a single-pixel region
        g = SeamGraph.from_labels(lab, np)
        check_invariants(g, lab)


def test_empty_and_uniform_rasters():
    g = SeamGraph.from_labels(np.zeros((0, 0), np.int32), np)
    assert g.n_seams == 0 and g.offsets.tolist() == [0]
    g = SeamGraph.from_labels(np.zeros((5, 5), np.int32), np)
    assert g.n_seams == 0 and g.n_junctions == 0
    assert seam_pixel_raster(g, np).shape == (5, 5)
    assert seam_class_lut(np.zeros(0, np.uint8), np).shape == (0, 4)


# --------------------------------------------------------------------------- #
# lookups
# --------------------------------------------------------------------------- #
def test_crack_index_and_seam_cracks_agree():
    lab = blocks()
    g = SeamGraph.from_labels(lab, np)
    for i in range(g.n_seams):
        ids = g.seam_cracks(i, np)
        assert (g.seam_of_cracks(ids, np) == i).all()
    assert g.seam_of_cracks(np.asarray([crack_id(0, 1, 1, 1, 12)]), np).tolist() == [-1]
    bb = g.bboxes(np)
    assert bb.shape == (4, 4) and bb[0].tolist() == [6, 0, 6, 6]
    nbr, sid, ptr = g.junction_csr(np)
    centre = [i for i, p in enumerate(g.junction_xy.tolist()) if p == [6, 6]][0]
    assert ptr[centre + 1] - ptr[centre] == 4
    assert sorted(sid[ptr[centre]:ptr[centre + 1]]) == [0, 1, 2, 3]


def test_nearest_seam_point_snaps_to_the_nearest_crack():
    lab = blocks()
    g = SeamGraph.from_labels(lab, np)
    hit = nearest_seam_point(g, lab, 6.3, 2.4, np)
    assert hit is not None
    s, k = hit
    assert (g.a[s], g.b[s]) == (0, 1)
    assert g.seam_points(s)[k].tolist() == [6, 2]
    # Far from every crack within the radius: None.
    assert nearest_seam_point(g, lab, 1.0, 1.0, np, radius=2) is None
    # Inside the radius from a corner region.
    assert nearest_seam_point(g, lab, 1.0, 1.0, np, radius=6) is not None


def test_pixel_raster_paints_both_flanks():
    lab = blocks()
    g = SeamGraph.from_labels(lab, np)
    r = seam_pixel_raster(g, np)
    painted = r >= 0
    # Both pixels beside the 0|1 seam (x = 6, rows 0..5) carry that seam.
    s01 = [i for i in range(g.n_seams) if (g.a[i], g.b[i]) == (0, 1)][0]
    assert (r[0:5, 5] == s01).all() and (r[0:5, 6] == s01).all()
    # Nothing away from a crack.
    assert not painted[0, 0] and not painted[11, 11] and not painted[2, 2]
    # Every painted pixel touches a crack.
    lab_p = np.pad(lab, 1, constant_values=-1)
    touch = np.zeros_like(painted)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb = lab_p[1 + dy:1 + dy + 12, 1 + dx:1 + dx + 12]
        touch |= (nb != lab) & (nb >= 0)
    assert (painted <= touch).all()


def test_luts():
    cls = np.array([0, SEAM_INTERIOR, SEAM_BOUNDARY, 9], np.uint8)
    lut = seam_class_lut(cls, np, alpha=100)
    assert lut.shape == (4, 4)
    assert lut[0, 3] == 0 and lut[1, 3] == 100 and lut[2, 3] == 100
    assert lut[1, :3].tolist() == list(seams.SEAM_COLORS[1][:3])
    assert lut[3, :3].tolist() == list(seams.SEAM_COLORS[-1][:3])   # clamped
    sl = seams.seam_scalar_lut(np.array([0.0, 1.0]), np, mask=np.array([True, False]))
    assert sl[0, 3] == 255 and sl[1, 3] == 0


def test_placement_round_trip():
    lab = blocks()
    g = SeamGraph.from_labels(lab, np, placement=Placement((100, 200), 4.0))
    pts = g.seam_points(0)
    img = g.to_image(pts)
    assert img[0] == (100 + 4 * pts[0, 0], 200 + 4 * pts[0, 1])
    back = g.to_raster(img, np)
    assert (back == pts).all()
