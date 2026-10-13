"""Region pooling of a token grid (msseg.labeler.foundation.pool), the spec's
naming and the token cache -- numpy only.

    pytest packages/mslabeler/tests/test_foundation_pool.py
"""
import numpy as np
import pytest

from msseg.labeler.foundation import pool as fp
from msseg.labeler.foundation.cache import TokenCache, cache_key
from msseg.labeler.foundation.project import fit_basis, project
from msseg.labeler.foundation.spec import (Basis, FoundationSpec, column_names,
                                           is_foundation_column, schema_entries)
from msseg.labeler.table import FeatureTable

ALL = ("mean", "purity", "interior", "boundary")


def brute(labels, grid, stride, ids, offset):
    """Per region, per pixel: the obvious loop the vectorised pooler must equal."""
    Gh, Gw, k = grid.shape
    h, w = labels.shape
    cell = np.zeros((h, w), np.int64)
    for y in range(h):
        for x in range(w):
            ty = min(max(int(np.floor((y + offset[0]) / stride)), 0), Gh - 1)
            tx = min(max(int(np.floor((x + offset[1]) / stride)), 0), Gw - 1)
            cell[y, x] = ty * Gw + tx
    cap = np.bincount(cell.ravel(), minlength=Gh * Gw)
    flat = grid.reshape(-1, k).astype(np.float64)
    rows = []
    for rid in ids:
        m = labels == rid
        if not m.any():
            rows.append(np.zeros(4 * k + 2))
            continue
        cells, cnt = np.unique(cell[m], return_counts=True)
        frac = cnt / cap[cells]
        full = cnt >= cap[cells]
        mean = (flat[cells] * cnt[:, None]).sum(0) / cnt.sum()

        def wm(wt):
            return mean if wt.sum() == 0 else (flat[cells] * wt[:, None]).sum(0) / wt.sum()
        rows.append(np.concatenate([mean, wm(cnt * frac), wm(cnt * full), wm(cnt * ~full),
                                    [frac.sum(), (cnt * frac).sum() / cnt.sum()]]))
    return np.array(rows)


@pytest.mark.parametrize("stride,offset", [(4, (0, 0)), (4, (3, 5)), (3.5, (2, 0)), (7, (0, 0))])
def test_pool_matches_brute_force(stride, offset):
    rng = np.random.default_rng(0)
    h, w = 23, 31
    labels = rng.integers(0, 5, size=(h, w))
    labels[:4, :6] = 9                      # one compact region
    labels[10, 10] = 11                     # a one-pixel region (sub-token)
    labels[0, :] = -1                       # background row
    Gh = int(np.ceil((h + offset[0] + 2) / stride))
    Gw = int(np.ceil((w + offset[1] + 2) / stride))
    grid = rng.normal(size=(Gh, Gw, 3)).astype(np.float32)
    ids = [0, 1, 2, 3, 4, 9, 11, 42]        # 42 has no pixel
    spec = FoundationSpec(dims=3, pools=ALL)
    got = fp.pool_regions(labels, grid, stride, ids, spec, offset)
    np.testing.assert_allclose(got, brute(labels, grid, stride, ids, offset), rtol=1e-9, atol=1e-9)
    assert not got[-1].any()                # absent region -> zeros


def test_sub_token_region_falls_back_to_mean():
    labels = np.zeros((8, 8), np.int64)
    labels[1, 1] = 1                        # 1 pixel of a 4x4 cell
    grid = np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2)
    spec = FoundationSpec(dims=2, pools=ALL)
    row = fp.pool_regions(labels, grid, 4, [1], spec)[0]
    mean, purity, interior, boundary = row[0:2], row[2:4], row[4:6], row[6:8]
    np.testing.assert_allclose(mean, grid[0, 0])
    np.testing.assert_allclose(interior, mean)          # no whole cell -> mean
    np.testing.assert_allclose(boundary, mean)
    np.testing.assert_allclose(purity, mean)
    assert row[-2] == pytest.approx(1 / 16)             # a sixteenth of a token
    assert row[-1] == pytest.approx(1 / 16)


def test_interior_ignores_straddling_cells():
    labels = np.zeros((8, 8), np.int64)
    labels[:, 4:] = 1
    labels[0, 4] = 0                        # region 0 nibbles one of region 1's cells
    grid = np.zeros((2, 2, 1), np.float32)
    grid[0, 1, 0], grid[1, 1, 0] = 10.0, 20.0
    spec = FoundationSpec(dims=1, pools=("mean", "interior", "boundary"))
    r1 = fp.pool_regions(labels, grid, 4, [1], spec)[0]
    assert r1[1] == pytest.approx(20.0)     # the whole cell only
    assert r1[2] == pytest.approx(10.0)     # the shared one only
    assert r1[0] == pytest.approx((15 * 10 + 16 * 20) / 31)


def test_columns_appends_by_id():
    labels = np.array([[3, 3, 5, 5]] * 4)
    grid = np.array([[[1.0], [2.0]]], np.float32)       # 1x2 cells of 2 px
    spec = FoundationSpec(dims=1, pools=("mean",))
    table = FeatureTable(["feature_id", "area"], np.array([[5.0, 8], [3.0, 8]]))
    out = fp.columns(table, labels[:, :4], grid, 2, spec)
    assert out.names[:2] == ["feature_id", "area"]
    assert out.names[2:] == column_names(spec)
    np.testing.assert_allclose(out.column(column_names(spec)[0]), [2.0, 1.0])
    assert table.values.shape == (2, 2)                 # input untouched
    with pytest.raises(ValueError):
        fp.columns(table, labels, np.zeros((1, 2, 4), np.float32), 2, spec)


def test_spec_round_trip_and_hash():
    spec = FoundationSpec(layers=(-4, -1), pools=("interior", "mean", "bogus", "mean"), dims=8)
    d = spec.to_dict()
    back = FoundationSpec.from_dict(d)
    assert back.pools == ("interior", "mean")
    assert back.extract_key() == spec.extract_key() and back.hash8 == spec.hash8
    assert FoundationSpec.from_dict({"tile": "x", "layers": ["a"]}).tile == FoundationSpec().tile
    rng = np.random.default_rng(1)
    basis = fit_basis([rng.normal(size=(6, 7, 16))], dims=4)
    fitted = back.with_basis(basis)
    assert fitted.dims == 4 and fitted.extract_key() == spec.extract_key()
    assert fitted.hash8 != spec.hash8
    again = FoundationSpec.from_dict(fitted.to_dict())
    assert again.hash8 == fitted.hash8
    assert "basis" not in fitted.to_dict(include_basis=False)
    assert FoundationSpec(model="dinov2_vitl14_reg").extract_key() != spec.extract_key()
    names = column_names(fitted)
    assert all(is_foundation_column(n) for n in names) and len(names) == 2 * 4 + 2
    assert not is_foundation_column("mean_base") and not is_foundation_column("ring_mean__mean_base")
    groups = {e["channel"] for e in schema_entries(fitted)}
    assert len(groups) == 3                 # two pools + reliability


def test_projection_whitens():
    rng = np.random.default_rng(2)
    g = rng.normal(size=(20, 30, 12)) * np.linspace(1, 5, 12)
    b = fit_basis([g], dims=5, max_tokens=10_000)
    z = project(g, b).reshape(-1, 5)
    np.testing.assert_allclose(z.std(0, ddof=1), 1.0, rtol=1e-4)
    np.testing.assert_allclose(z.mean(0), 0.0, atol=1e-4)
    assert isinstance(Basis.from_dict(b.to_dict()).components, np.ndarray)


def test_token_cache_disk_round_trip(tmp_path):
    c = TokenCache(str(tmp_path), max_items=1)
    k1 = cache_key("slide@L2#0,0,10,10", 2, 16, "abc")
    g = np.random.default_rng(3).normal(size=(3, 4, 5)).astype(np.float32)
    calls = []
    e = c.get_or_compute(k1, lambda: (calls.append(1), (g, 14.0))[1])
    assert e[0].dtype == np.float16 and e[1] == 14.0
    c.get_or_compute(k1, lambda: (calls.append(1), (g, 14.0))[1])
    assert calls == [1] and c.hits == 1
    c.put(cache_key("other", 2, 16, "abc"), g, 14.0)     # evicts k1 from memory
    fresh = TokenCache(str(tmp_path))
    e2 = fresh.get(k1)
    assert e2 is not None and fresh.disk_hits == 1
    np.testing.assert_array_equal(e2[0], e[0])
    assert k1 in fresh and cache_key("nope", 0, 0, "x") not in fresh


def test_seam_on_a_cell_edge_reads_both_cells():
    from msseg.labeler.seams import SeamGraph
    lab = np.zeros((8, 8), np.int64)
    lab[:, 4:] = 1                          # one vertical seam on the x=4 corner column
    g = SeamGraph.from_labels(lab, np)
    grid = np.zeros((2, 2, 1), np.float32)
    grid[:, 0, 0], grid[:, 1, 0] = 1.0, 3.0
    out = fp.pool_seams(g, grid, 4)
    assert out.shape == (1, 2)
    assert out[0, 0] == pytest.approx(2.0)  # half of every crack to each flank's cell
    assert out[0, 1] == pytest.approx(8 / 4)


def test_seam_inside_a_cell_column_and_halo_offset():
    from msseg.labeler.seams import SeamGraph
    lab = np.zeros((8, 8), np.int64)
    lab[:, 2:] = 1                          # seam at x=2: both flanks in cell column 0 ...
    g = SeamGraph.from_labels(lab, np)
    grid = np.zeros((3, 3, 1), np.float32)
    grid[:, 0, 0], grid[:, 1, 0] = 5.0, 7.0
    assert fp.pool_seams(g, grid, 4)[0, 0] == pytest.approx(5.0)
    # ... unless the raster sits 2 px into the grid's image: x=1 -> col 0, x=2 -> col 1
    assert fp.pool_seams(g, grid, 4, offset=(0, 2))[0, 0] == pytest.approx(6.0)


def test_seam_pool_matches_brute_force():
    from msseg.labeler.seams import SeamGraph
    rng = np.random.default_rng(4)
    lab = rng.integers(0, 4, size=(13, 17))
    g = SeamGraph.from_labels(lab, np)
    grid = rng.normal(size=(5, 6, 3)).astype(np.float32)
    stride, off = 3.0, (1, 2)
    got = fp.pool_seams(g, grid, stride, off)
    ids, seam_of = g.all_cracks(np)
    x0, y0, x1, y1 = fp.crack_flank_pixels(ids, g.width)
    h, w = lab.shape
    for s in range(g.n_seams):
        acc, tot = np.zeros(3), 0.0
        for c in np.flatnonzero(seam_of == s):
            pts = [(x0[c], y0[c]), (x1[c], y1[c])]
            pts = [(x, y) for x, y in pts if 0 <= x < w and 0 <= y < h]
            for x, y in pts:
                ty = min(int((y + off[0]) // stride), 4)
                tx = min(int((x + off[1]) // stride), 5)
                acc += grid[ty, tx] / len(pts)
                tot += 1.0 / len(pts)
        np.testing.assert_allclose(got[s, :3], acc / tot, rtol=1e-6)
        assert got[s, 3] == pytest.approx(tot / stride)
    # every crack really separates two different labels
    a = lab[y0.clip(0, h - 1), x0.clip(0, w - 1)]
    b = lab[y1.clip(0, h - 1), x1.clip(0, w - 1)]
    inside = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    assert (a[inside] != b[inside]).all()
