"""Regions gather tokens: one fixed-length vector per region, whatever its size.

Every label pixel falls in exactly one grid cell; a region's pixels in a cell
are that cell's WEIGHT for the region. So a region of 3 pixels and one of
30 000 both reduce to a weighted mean over the cells they touch -- no padding,
no bags. ``A[r, t]`` (pixels of region r in cell t) is built sparse, from one
``np.unique`` over ``region * T + cell``, and every pool is a reweighting of
it:

* ``mean``      -- weight = pixels: the region's area-weighted mean token.
* ``purity``    -- weight = pixels x (the fraction of the cell the region
                   covers): a cell straddling a boundary describes both
                   flanks, so it counts less.
* ``interior``  -- only cells the region covers WHOLLY;
* ``boundary``  -- only the cells it shares. Either falls back to ``mean``
                   for a region that has none (every region smaller than a
                   cell has no interior).

Two reliability columns ride along so the head can learn when to trust the
rest: ``tokens`` (the region's area in cell units -- below 1 it is a fraction
of one token) and ``purity_frac`` (the pixel-weighted mean coverage of the
cells it touches -- 1 means every cell is all this region).

A cell's capacity is counted from the raster itself, so a fractional stride
(``scale`` 2 on a 14-px patch) and the cells overhanging the item edge are
exact rather than assumed ``stride**2``.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from ..table import FeatureTable
from .spec import FoundationSpec, column_names


def cell_index(shape: Tuple[int, int], grid_shape: Tuple[int, int], stride: float,
               offset: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """``int64 (h, w)``: the flat grid cell of every pixel of a ``shape``
    raster whose pixel (0, 0) sits at ``offset`` (row, col) pixels into the
    image the grid was computed on (the halo)."""
    h, w = int(shape[0]), int(shape[1])
    Gh, Gw = int(grid_shape[0]), int(grid_shape[1])
    ty = np.floor((np.arange(h) + offset[0]) / float(stride)).astype(np.int64)
    tx = np.floor((np.arange(w) + offset[1]) / float(stride)).astype(np.int64)
    ty = np.clip(ty, 0, Gh - 1)
    tx = np.clip(tx, 0, Gw - 1)
    return ty[:, None] * Gw + tx[None, :]


def pool_regions(labels: np.ndarray, grid: np.ndarray, stride: float, ids: Sequence[int],
                 spec: FoundationSpec, offset: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """``float64 (len(ids), len(column_names(spec)))``: the pooled columns
    for the regions `ids` (label values; -1 is background), in that order. A
    region with no pixel in `labels` gets a row of zeros."""
    lab = np.asarray(labels)
    G = np.asarray(grid, np.float32)
    Gh, Gw, k = G.shape
    T = Gh * Gw
    ids = np.asarray(ids, np.int64)
    n = len(ids)
    pools = tuple(spec.pools)
    out = np.zeros((n, len(pools) * k + 2), np.float64)
    if n == 0 or lab.size == 0:
        return out
    tok = cell_index(lab.shape, (Gh, Gw), stride, offset)
    cap = np.bincount(tok.ravel(), minlength=T).astype(np.float64)

    # region id -> row; a label value with no row (or background) drops out
    top = int(max(int(ids.max()), int(lab.max()), 0))
    lut = np.full(top + 2, -1, np.int64)
    ok_ids = ids >= 0
    lut[ids[ok_ids]] = np.flatnonzero(ok_ids)
    flat = lab.ravel().astype(np.int64)
    sel = flat >= 0
    r = lut[flat[sel]]
    t = tok.ravel()[sel]
    keep = r >= 0
    pair = r[keep] * T + t[keep]
    if pair.size == 0:
        return out
    u, c = np.unique(pair, return_counts=True)        # sorted: region-major
    ru, tu = u // T, u % T
    c = c.astype(np.float64)
    frac = c / cap[tu]
    full = c >= cap[tu]
    starts = np.flatnonzero(np.r_[True, ru[1:] != ru[:-1]])
    rows = ru[starts]
    Gf = G.reshape(T, k)[tu].astype(np.float64)

    def wmean(w):
        num = np.add.reduceat(Gf * w[:, None], starts, axis=0)
        den = np.add.reduceat(w, starts)
        return num, den

    mean_num, area = wmean(c)
    mean = mean_num / area[:, None]
    for j, pool in enumerate(pools):
        if pool == "mean":
            v = mean
        else:
            w = {"purity": c * frac, "interior": c * full,
                 "boundary": c * (~full)}[pool]
            num, den = wmean(w)
            v = np.where(den[:, None] > 0, num / np.maximum(den, 1e-300)[:, None], mean)
        out[rows, j * k:(j + 1) * k] = v
    out[rows, -2] = np.add.reduceat(frac, starts)                 # area in cell units
    out[rows, -1] = np.add.reduceat(c * frac, starts) / area      # mean coverage
    return out


def columns(table: FeatureTable, labels: np.ndarray, grid: np.ndarray, stride: float,
            spec: FoundationSpec, offset: Tuple[int, int] = (0, 0),
            id_field: str = "feature_id") -> FeatureTable:
    """A NEW table: `table`'s columns plus the spec's ``fm<hash8>__..``
    columns for its rows (matched by `id_field`). `table` is untouched."""
    ids = np.asarray(table.column(id_field)).astype(np.int64)
    block = pool_regions(labels, grid, stride, ids, spec, offset)
    if block.shape[1] != len(column_names(spec)):
        raise ValueError(f"grid width {np.asarray(grid).shape[-1]} does not match "
                         f"spec dims {spec.dims} (fit or drop the projection)")
    values = np.concatenate([np.asarray(table.values, np.float64), block], axis=1)
    return FeatureTable(list(table.names) + column_names(spec), values)


# --------------------------------------------------------------------------- #
# Seams gather the cells they run through (polyline tasks)
# --------------------------------------------------------------------------- #
def seam_column_names(spec: FoundationSpec) -> list:
    """The seam block: ``seam_<kk>`` (the crack-weighted mean token) and
    ``seam_cells`` (the seam's length in cell units)."""
    pre = f"fm{spec.hash8}__"
    w = max(2, len(str(max(int(spec.dims) - 1, 0))))
    return [f"{pre}seam_{k:0{w}d}" for k in range(int(spec.dims))] + [f"{pre}seam_cells"]


def crack_flank_pixels(crack_ids: np.ndarray, width: int) -> Tuple[np.ndarray, ...]:
    """``(x0, y0, x1, y1)``: the two pixels a crack separates. A horizontal
    crack (even id) at corner row y between corners x and x+1 lies between
    pixels (x, y-1) and (x, y); a vertical one (odd id) at corner column x
    between rows y and y+1 lies between (x-1, y) and (x, y). The lattice is
    ``seams.crack_id``'s, ``cw = width + 1`` corners per row."""
    c = np.asarray(crack_ids, np.int64)
    cw = int(width) + 1
    vertical = (c & 1).astype(bool)
    q = c >> 1
    y, x = q // cw, q % cw
    x0 = np.where(vertical, x - 1, x)
    y0 = np.where(vertical, y, y - 1)
    return x0, y0, x, y


def pool_seams(graph, grid: np.ndarray, stride: float,
               offset: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """``float64 (S, k + 1)``: per seam of `graph` (a ``seams.SeamGraph`` on
    the record's label raster), the mean over the cells it runs through,
    each cell weighted by the crack length inside it -- a crack counts half
    to each flank pixel's cell, so a seam on a cell edge reads both cells
    equally -- then its length in cell units. A seam with no crack gets zeros."""
    G = np.asarray(grid, np.float32)
    Gh, Gw, k = G.shape
    T = Gh * Gw
    S = int(graph.n_seams)
    out = np.zeros((S, k + 1), np.float64)
    ids, seam_of = graph.all_cracks(np)
    if S == 0 or len(ids) == 0:
        return out
    h, w = graph.shape
    x0, y0, x1, y1 = crack_flank_pixels(ids, w)
    seams, cells, wts = [], [], []
    for px, py in ((x0, y0), (x1, y1)):
        ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        ty = np.clip(np.floor((py[ok] + offset[0]) / float(stride)).astype(np.int64), 0, Gh - 1)
        tx = np.clip(np.floor((px[ok] + offset[1]) / float(stride)).astype(np.int64), 0, Gw - 1)
        seams.append(np.asarray(seam_of, np.int64)[ok])
        cells.append(ty * Gw + tx)
        wts.append(np.where(((x0 >= 0) & (x0 < w) & (y0 >= 0) & (y0 < h)) &
                            ((x1 >= 0) & (x1 < w) & (y1 >= 0) & (y1 < h)), 0.5, 1.0)[ok])
    pair = np.concatenate(seams) * T + np.concatenate(cells)
    u, inv = np.unique(pair, return_inverse=True)
    wsum = np.bincount(inv.ravel(), weights=np.concatenate(wts))
    su, tu = u // T, u % T
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    rows = su[starts]
    num = np.add.reduceat(G.reshape(T, k)[tu].astype(np.float64) * wsum[:, None], starts, axis=0)
    den = np.add.reduceat(wsum, starts)
    out[rows, :k] = num / den[:, None]
    out[rows, k] = den / float(stride)
    return out
