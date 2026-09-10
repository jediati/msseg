"""``mspath-gui --selftest``: build the real app and drive it headlessly.

The pytest suites cover the pure pieces (keys, the label layer, the positional
mapping). This covers the thing they cannot: that ``MsPathApp`` actually
composes with ``ViewerShell`` -- that the hooks are the ones the shell calls,
that a session round-trips, that navigation reaches an item, and that a prime
followed by a persistence change produces a record and an overlay.

It needs a display for Tk, and a slide only if one is available: without a
real pyramid it builds a **synthetic** one (a small tiled multi-level TIFF via
tifffile, or the ARPA-H slide when it is present), so the test runs anywhere
the labeler's does.
"""
from __future__ import annotations

import os
import sys
import tempfile

SLIDE_ENV = "MSPATH_TEST_SLIDE"
KNOWN_SLIDE = (r"C:\Users\jediati\Desktop\JEDIATI\data\arpa-h\topo-data\WSI"
               r"\Subset1_Test_10.tiff")


def _synthetic_slide(folder, name="synthetic.tiff", w=1024, h=768, levels=4):
    """A small tiled, multi-level RGB TIFF: enough of a pyramid for the reader,
    small enough to prime in a second."""
    import numpy as np
    import tifffile

    yy, xx = np.mgrid[0:h, 0:w]
    blobs = (np.sin(xx / 37.0) * np.cos(yy / 29.0) * 0.5 + 0.5)
    base = np.stack([(blobs * 200 + 30), (blobs * 120 + 60), (255 - blobs * 180)], -1)
    base = np.ascontiguousarray(base.astype(np.uint8))
    path = os.path.join(folder, name)
    with tifffile.TiffWriter(path, bigtiff=False) as tw:
        opts = dict(tile=(256, 256), photometric="rgb", compression=None)
        tw.write(base, subifds=0, **opts)
        cur = base
        for _ in range(levels - 1):
            cur = np.ascontiguousarray(cur[::2, ::2])
            tw.write(cur, **opts)
    return path


def _pick_slide(tmp):
    env = os.environ.get(SLIDE_ENV)
    if env and os.path.exists(env):
        return env, os.path.dirname(env), "env"
    if os.path.exists(KNOWN_SLIDE):
        return KNOWN_SLIDE, os.path.dirname(KNOWN_SLIDE), "known"
    folder = os.path.join(tmp, "slides")
    os.makedirs(folder, exist_ok=True)
    return _synthetic_slide(folder), folder, "synthetic"


def run_selftest():
    import tkinter as tk

    from msseg.labeler.pyramid import backends_available
    from . import items as I
    from .app import MsPathApp

    if not backends_available():
        print("selftest SKIPPED: no pyramid backend installed "
              "(pip install openslide-python)")
        return 0

    tmp = tempfile.mkdtemp(prefix="mspath-selftest-")
    slide, folder, origin = _pick_slide(tmp)
    print(f"[mspath] selftest slide ({origin}): {slide}")

    try:
        root = tk.Tk()
    except Exception as exc:
        print(f"selftest SKIPPED: no display ({exc})")
        return 0
    root.withdraw()
    app = MsPathApp(root, autosave=False)

    # -- the session browser -------------------------------------------- #
    app._add_folder_path(folder)
    assert app.folders, "the folder was not added"
    assert slide in app.all_files, f"{os.path.basename(slide)} was not listed"
    # a preview needs no Run: the pyramid IS the preview
    app._preview_file(slide)
    assert app.viewer.has_base, "the preview did not reach the canvas"
    src = app.viewer.source
    assert src.levels >= 2 and src.channels == 3, f"unexpected source {src}"

    # -- a slide becomes a sequence -------------------------------------- #
    app.file_list.selection_clear(0, "end")
    app.file_list.selection_set(app.all_files.index(slide))
    app._make_subsequence()
    assert len(app.subsequences) == 1, app.subsequences

    # keep the selftest quick whatever slide it got: a level of a few Mpx
    deepest = max(0, src.levels - 3)
    app.level_var.set(deepest)
    app._on_level_change()

    item = app._item_at(0, 0)
    assert item is not None and item.is_overview, item
    assert item.level == deepest
    assert I.parse_key(item.key) == item, "the key does not round-trip"
    key = app.catalogue.key_of(0, 0)
    assert key == item.key
    assert app.catalogue.index_of(key) == (0, 0)
    assert app.catalogue.group_of(key) == item.slide, "CV groups must be whole slides"
    tree = app.catalogue.tree()
    assert len(tree) == 1 and len(tree[0]["children"]) == 1, tree

    # -- run, synchronously ---------------------------------------------- #
    profile = app._profile_for_compute()
    app.engine.reset()
    app.engine.prime_item(item, profile, halo=0)
    rec = app.engine.ensure_record(item.key, profile)
    assert rec is not None and rec["stats"].n_rows >= 1, rec
    assert app._slice_msc_mark(0, 0) == "Y"
    app._rebuild_flat_slices()
    assert app.flat_slices == [(0, 0)], app.flat_slices
    assert app.regions.keys() == [item.key]
    assert app.regions.record(item.key) is rec

    # positions are SLIDE coordinates, so they must reach past the level's own
    # extent whenever the level is coarser than 1:1
    scale = rec["scale"]
    if scale > 1:
        lh, lw = rec["labels"].shape[:2]
        assert rec["stats"].column("ext_x").max() > lw, \
            "positional columns were not mapped into slide coordinates"

    # -- the label layer lives in slide coordinates ---------------------- #
    layer = app.regions.label_layer(item.key)
    assert layer is not None and layer.rev == app.engine.commit_id
    assert layer.shape == tuple(src.level_shape(0)), (layer.shape, src.level_shape(0))
    assert layer.full() is None or scale == 1.0, "a coarse layer must not claim a full raster"
    crop = layer.crop(0, 0, 0, 32, 32)
    assert crop.shape == (32, 32)
    outside = layer.id_at(-5, -5)
    assert outside == -1, outside

    # -- render + hover --------------------------------------------------- #
    app._goto_slice(0)
    app._refresh_render()
    overlays = app._seg_overlays(item.key, rec, __import__("numpy"),
                                 __import__("msseg.viz", fromlist=["min_colors"]).min_colors)
    assert overlays and overlays[0]["lut"].shape[0] == layer.n_ids, \
        "the LUT must be sized from the layer's declared id count"
    ext = rec["stats"].row_of_feature(int(rec["stats"].column("feature_id")[0]))
    app._on_hover(int(ext["ext_x"]), int(ext["ext_y"]))
    assert "region" in app.hover_var.get(), app.hover_var.get()

    # -- persistence is a select, not a prime ----------------------------- #
    before = app.engine.commit_id
    n_before = rec["stats"].n_rows
    app.persist_var.set(min(40.0, float(app.persist_var.get()) + 20.0))
    app._on_persistence_change()
    assert app.engine.commit_id > before, "a threshold change must bump the commit"
    rec2 = app.engine.record(item.key)
    assert rec2 is not None and rec2 is not rec, "the record was not recomputed"
    assert rec2["stats"].n_rows <= n_before, "more persistence must not add regions"
    # the pin is per level and set once
    assert set(app.engine.persistence_abs) == {deepest}, app.engine.persistence_abs

    # -- profiles + session round-trip ------------------------------------ #
    p = app._profile_from_ui()
    assert p["slide"]["overview_level"] == deepest
    app.level_var.set(0); app.halo_var.set(0)
    app._apply_profile_to_ui(p, lambda v, x: v.set(x), [])
    assert app.level_var.get() == deepest and app.halo_var.get() == p["slide"]["halo"]

    doc = app._session_doc()
    app2 = MsPathApp(tk.Toplevel(root), autosave=False)
    app2._apply_session_doc(doc, source="selftest")
    assert len(app2.subsequences) == 1, app2.subsequences
    assert app2._item_at(0, 0).key == item.key, "the item key did not survive the session"

    root.destroy()
    print("selftest OK: pyramid preview, slide->sequence, overview item + key round-trip, "
          "prime + record, slide-coordinate positions, label layer, render + hover, "
          "persistence select, profile + session round-trip")
    return 0


if __name__ == "__main__":
    sys.exit(run_selftest())
