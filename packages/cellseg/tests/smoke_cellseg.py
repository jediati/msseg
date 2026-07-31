"""Smoke test for the cellseg Python interface (two-phase 3D cell segmentation).

Synthesizes a few hollow fluorescent shells in memory, drives the full cellseg
Python API (heavy_lift -> set_persistence -> merge_tree_json -> segment), and
checks structural invariants of the merge tree and the seg8 / ids volumes.

No external data required (numpy only -- no tifffile/matplotlib). It mirrors the
known-good synthetic volume and thresholds from the C++ test
``tests/cellseg_tests.cpp`` (48^3, three shells, blur_sigma=1.5,
persistence_percent=5.0, segment(cut_threshold=0.0, background_threshold=0.4)).

Run standalone:

    python python/tests/smoke_cellseg.py

or under pytest (``test_cellseg_smoke`` is collected automatically).
"""

import json
import sys

import numpy as np

try:
    from msseg import cellseg
except Exception as exc:  # pragma: no cover - import-time failure path
    cellseg = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def make_shells(n: int = 48) -> np.ndarray:
    """Three bright hollow shells (fluorescent membranes) on a dark background.

    Returns a C-contiguous float32 (depth, height, width) volume, matching the
    ``make_shells`` used by the C++ smoke test.
    """
    centers = [(14.0, 14.0, 14.0), (34.0, 14.0, 20.0), (24.0, 34.0, 28.0)]  # (cx, cy, cz)
    radius, thick = 8.0, 1.5

    # mgrid axes are (z, y, x); replicate r = sqrt((x-cx)^2 + (y-cy)^2 + (z-cz)^2).
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(np.float32)
    vol = np.zeros((n, n, n), dtype=np.float32)
    for cx, cy, cz in centers:
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
        vol += np.exp(-((r - radius) ** 2) / (2.0 * thick * thick))
    return np.ascontiguousarray(vol, dtype=np.float32)


def _count_leaves(tree: dict) -> int:
    """Count leaf nodes (minima) in the flat {nodes, roots} merge-tree JSON.

    ``roots`` and each node's ``children`` are integer indices into ``nodes``; a
    leaf (minimum) is a node with no children.
    """
    nodes = tree["nodes"]
    total = 0
    stack = list(tree.get("roots", []))
    while stack:
        idx = stack.pop()
        children = nodes[idx]["children"]
        if not children:
            total += 1
        else:
            stack.extend(children)
    return total


def run() -> None:
    """Exercise the cellseg pipeline end-to-end, asserting invariants."""
    assert cellseg is not None, (
        "the cellseg extension is not built into the msseg package "
        f"(import error: {_IMPORT_ERROR!r}). Re-run `pip install -v .`."
    )
    assert cellseg.version() == "0.1.0", f"unexpected version {cellseg.version()!r}"

    vol = make_shells(48)
    assert vol.dtype == np.float32 and vol.ndim == 3

    # --- Phase A: heavy lift (once) -----------------------------------------
    pipe = cellseg.heavy_lift(
        vol, json.dumps({"blur_sigma": 1.5, "persistence_percent": 5.0})
    )
    value_range = pipe.value_range()
    heavy_persistence = pipe.heavy_persistence()
    assert value_range > 0.0, "value range should be positive"
    assert np.isfinite(heavy_persistence), "heavy persistence should be finite"
    print(f"[PASS] heavy_lift: value_range={value_range:.4f} "
          f"heavy_persistence={heavy_persistence:.4f}")

    # --- Phase B: select persistence + build merge tree ---------------------
    pipe.set_persistence(heavy_persistence)
    assert np.isfinite(pipe.current_persistence())
    tree = json.loads(pipe.merge_tree_json())
    leaves = _count_leaves(tree)
    assert len(tree["roots"]) >= 1, "merge tree should have at least one root"
    assert leaves >= 3, f"expected a leaf per shell (>=3), got {leaves}"
    print(f"[PASS] merge_tree: roots={len(tree['roots'])} leaves={leaves}")

    # --- Phase B: run a segmentation ----------------------------------------
    # cut at 0 (fully separate the basins); intensity threshold picks the shells.
    seg8, ids = pipe.segment(0.0, 0.4)  # cut_threshold, background_threshold
    assert seg8.shape == vol.shape and seg8.dtype == np.uint8
    assert ids.shape == vol.shape and ids.dtype == np.int32
    foreground = int((seg8 & 8 > 0).sum())
    membrane = int((seg8 & 2 > 0).sum())
    nonbg_asc = int((seg8 & 1 > 0).sum())
    n_ids = int(np.unique(ids).size)
    assert foreground > 0, "intensity foreground (bit 8) should be non-empty"
    assert membrane > 0, "cleaned membrane (bit 2) should be non-empty"
    assert nonbg_asc > 0, "non-background ascending (bit 1) should be non-empty"
    assert n_ids >= 2, "id volume should have multiple distinct regions"
    print(f"[PASS] segment: foreground={foreground} membrane={membrane} "
          f"nonbg_ascending={nonbg_asc} distinct_ids={n_ids}")

    print("cellseg smoke test OK")


def test_cellseg_smoke() -> None:
    """pytest entry point."""
    run()


def main() -> int:
    try:
        run()
    except AssertionError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
