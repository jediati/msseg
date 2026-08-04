"""On-the-fly 3D assembly of per-slice 2D features into 3D features.

Mirrors the C++ ``SliceMatcher`` (union-find over ``(slice, feature_id)`` nodes,
linked by an N-neighbor stencil between consecutive slices) so the viewer and the
CLI produce the same 3D grouping. Per-2D-feature statistics are mergeable, so 3D
feature stats are combined cheaply as features are unioned -- no re-scan of voxels
as the persistence/query controls change.

Pure Python + numpy (scipy optional); importable and testable headlessly.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


# Cross-slice in-plane offsets for each 3D connectivity (the z+1 neighborhood).
_OFFSETS = {
    6: [(0, 0)],
    18: [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
    26: [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
}


def _shift(b: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Return an array `bs` with bs[y,x] = b[y+dy, x+dx], out-of-range -> 0
    (non-wrapping, unlike np.roll, so features can't link across image edges)."""
    out = np.zeros_like(b)
    h, w = b.shape
    y_dst = slice(max(0, -dy), h - max(0, dy))
    x_dst = slice(max(0, -dx), w - max(0, dx))
    y_src = slice(max(0, dy), h - max(0, -dy))
    x_src = slice(max(0, dx), w - max(0, -dx))
    out[y_dst, x_dst] = b[y_src, x_src]
    return out


class _UnionFind:
    def __init__(self):
        self.parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def add(self, x):
        self.parent.setdefault(x, x)

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _combine(dst: dict, src: dict, slice_index: int):
    """Merge one 2D feature stat (`src`) into a 3D accumulator (`dst`)."""
    area = float(src.get("area", 0.0))
    base_sum = float(src.get("mean_base", 0.0)) * area
    if dst.get("area", 0.0) == 0.0:
        dst.update(
            area=area, base_sum=base_sum,
            min_base=float(src.get("min_base", 0.0)), max_base=float(src.get("max_base", 0.0)),
            min_x=int(src.get("min_x", 0)), max_x=int(src.get("max_x", 0)),
            min_y=int(src.get("min_y", 0)), max_y=int(src.get("max_y", 0)),
            min_z=slice_index, max_z=slice_index, slices={slice_index},
        )
        return
    dst["area"] += area
    dst["base_sum"] += base_sum
    dst["min_base"] = min(dst["min_base"], float(src.get("min_base", 0.0)))
    dst["max_base"] = max(dst["max_base"], float(src.get("max_base", 0.0)))
    dst["min_x"] = min(dst["min_x"], int(src.get("_min_x", 0)))
    dst["max_x"] = max(dst["max_x"], int(src.get("_max_x", 0)))
    dst["min_y"] = min(dst["min_y"], int(src.get("_min_y", 0)))
    dst["max_y"] = max(dst["max_y"], int(src.get("_max_y", 0)))
    dst["min_z"] = min(dst["min_z"], slice_index)
    dst["max_z"] = max(dst["max_z"], slice_index)
    dst["slices"].add(slice_index)


def assemble(
    labels_per_slice: Sequence[np.ndarray],
    kept_per_slice: Sequence[Set[int]],
    stats_per_slice: Sequence[Dict[int, dict]],
    connectivity: int = 26,
) -> Tuple[Dict[Tuple[int, int], int], List[dict]]:
    """Link kept per-slice features into 3D features.

    Args:
        labels_per_slice: [Z] int arrays (H,W) of feature id per pixel.
        kept_per_slice:   [Z] sets of feature ids that participate (size-gated).
        stats_per_slice:  [Z] dict feature_id -> stat dict (needs area, mean_base,
                          min_base, max_base, and bbox fields _min_x/_max_x/_min_y/_max_y).
        connectivity:     6 / 18 / 26 (cross-slice in-plane stencil).

    Returns:
        (global_of, features_3d) where global_of maps (slice, feature_id) -> global
        3D id (0..K-1), and features_3d[i] is a stat dict for global id i.
    """
    offsets = _OFFSETS.get(connectivity, _OFFSETS[26])
    uf = _UnionFind()
    for z, kept in enumerate(kept_per_slice):
        for fid in kept:
            uf.add((z, int(fid)))

    # Cross-slice linking between consecutive slices.
    for z in range(len(labels_per_slice) - 1):
        a, b = labels_per_slice[z], labels_per_slice[z + 1]
        keep_a, keep_b = kept_per_slice[z], kept_per_slice[z + 1]
        if not keep_a or not keep_b:
            continue
        amask = np.isin(a, list(keep_a))
        for dy, dx in offsets:
            bs = _shift(b, dy, dx)
            both = amask & np.isin(bs, list(keep_b))
            if not both.any():
                continue
            fa = a[both].astype(np.int64)
            fb = bs[both].astype(np.int64)
            pairs = np.unique(np.stack([fa, fb], axis=1), axis=0)
            for pa, pb in pairs:
                uf.union((z, int(pa)), (z + 1, int(pb)))

    # Number the roots into contiguous global ids (deterministic: sorted).
    roots = sorted({uf.find(n) for n in uf.parent})
    root_to_gid = {r: i for i, r in enumerate(roots)}
    global_of: Dict[Tuple[int, int], int] = {}
    accum: List[dict] = [dict() for _ in roots]
    for (z, fid) in uf.parent:
        gid = root_to_gid[uf.find((z, fid))]
        global_of[(z, fid)] = gid
        st = stats_per_slice[z].get(fid)
        if st is not None:
            _combine(accum[gid], st, z)

    features_3d: List[dict] = []
    for gid, a in enumerate(accum):
        area = a.get("area", 0.0)
        features_3d.append({
            "global_id": gid,
            "area": area,
            "voxel_count": area,
            "mean_base": (a.get("base_sum", 0.0) / area) if area else 0.0,
            "min_base": a.get("min_base", 0.0),
            "max_base": a.get("max_base", 0.0),
            "num_slices": len(a.get("slices", ())),
            "bbox_w": (a.get("max_x", 0) - a.get("min_x", 0) + 1) if area else 0,
            "bbox_h": (a.get("max_y", 0) - a.get("min_y", 0) + 1) if area else 0,
            "bbox_d": (a.get("max_z", 0) - a.get("min_z", 0) + 1) if area else 0,
        })
    return global_of, features_3d


def _selftest():
    # Two features per slice across 3 slices; feature 1 overlaps across all
    # slices (one 3D feature), feature 2 only in slice 0 (its own 3D feature).
    z0 = np.array([[1, 1, 2], [1, 1, 2]], dtype=np.int32)
    z1 = np.array([[1, 1, 0], [1, 1, 0]], dtype=np.int32)
    z2 = np.array([[1, 1, 0], [1, 1, 0]], dtype=np.int32)
    labels = [z0, z1, z2]
    kept = [{1, 2}, {1}, {1}]
    stat = lambda a, mb: {"area": a, "mean_base": mb, "min_base": 0.0, "max_base": 1.0,
                          "min_x": 0, "max_x": 1, "min_y": 0, "max_y": 1}
    stats = [{1: stat(4, 2.0), 2: stat(2, 5.0)}, {1: stat(4, 2.0)}, {1: stat(4, 2.0)}]
    # 6-connectivity: feature 2 (only in slice 0, no direct z-overlap) stays
    # distinct; feature 1 unifies across the 3 slices by direct overlap.
    global_of, feats = assemble(labels, kept, stats, connectivity=6)
    assert global_of[(0, 1)] == global_of[(1, 1)] == global_of[(2, 1)]
    assert global_of[(0, 2)] != global_of[(0, 1)]
    assert len(feats) == 2
    big = feats[global_of[(0, 1)]]
    assert big["voxel_count"] == 12 and big["num_slices"] == 3 and big["bbox_d"] == 3
    # 26-connectivity: feature 2's corner voxel is diagonally adjacent to feature
    # 1 in slice 1, so they unify (matches the C++ matcher's 26-neighbor rule).
    g26, f26 = assemble(labels, kept, stats, connectivity=26)
    assert g26[(0, 2)] == g26[(0, 1)] and len(f26) == 1
    # non-wrapping shift: an edge-only feature must NOT link across the image edge.
    edge = [np.array([[3, 0, 0]], dtype=np.int32), np.array([[0, 0, 4]], dtype=np.int32)]
    ge, _ = assemble(edge, [{3}, {4}], [{3: stat(1, 1.0)}, {4: stat(1, 1.0)}], connectivity=26)
    assert ge[(0, 3)] != ge[(1, 4)], "features must not wrap across image edges"
    print("assembly selftest OK:", len(feats), "(6-conn) /", len(f26), "(26-conn) 3D features")


if __name__ == "__main__":
    _selftest()
