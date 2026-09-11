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
    from . import app as app_mod
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
    overlays = app._seg_overlays(0, 0, rec, None, __import__("numpy"),
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

    # -- the ROI tier ------------------------------------------------------ #
    # A slide's items are its overview and whatever has been cut from it.
    assert app._enumerate_items.__self__ is app
    assert list(app._enumerate_items()) == [(0, 0)], "a fresh slide has one item"
    sh0, sw = src.level_shape(0)
    roi_level = max(0, deepest - 2)
    app.roi_level_var.set(roi_level)
    side = int(256 * src.level_scale(roi_level))       # 256 px AT the roi level
    added = app._add_roi(0, roi_level, sw // 4, sh0 // 4, side, side)
    assert added is not None and len(app._rois_of(0)) == 1, app._rois_of(0)

    # a rect that is nothing at its own level is refused, not silently kept:
    # a 1x1 raster would pin that level's threshold for everything after it
    assert app._add_roi(0, roi_level, 0, 0, 4, 4) is None
    assert len(app._rois_of(0)) == 1, "a degenerate ROI was accepted"
    assert list(app._enumerate_items()) == [(0, 0), (0, 1)]
    roi = app._item_at(0, 1)
    assert roi is not None and not roi.is_overview and roi.level == app.roi_level_var.get()
    assert I.parse_key(roi.key) == roi
    assert app.catalogue.index_of(roi.key) == (0, 1)
    # the overview and the ROI are different items of the SAME slide, so they
    # share a cross-validation group
    assert app.catalogue.group_of(roi.key) == app.catalogue.group_of(item.key)
    rows = app._sequence_item_labels(0)
    assert len(rows) == 2 and rows[0].startswith("overview"), rows

    # an ROI is capped rather than allowed to be a half-hour of compute
    app._add_roi(0, 0, 0, 0, sw, sh0)
    big = app._rois_of(0)[-1]
    assert big["w"] * big["h"] <= app_mod.MAX_ROI_PX + 1, big
    assert big["w"] < sw, "a whole-slide ROI at level 0 must be capped"

    # priming the ROI keeps the thresholds already pinned for other levels
    pins_before = dict(app.engine.persistence_abs)
    app.engine.prime_item(roi, app._profile_for_compute(), halo=app._halo())
    rrec = app.engine.ensure_record(roi.key, app._profile_for_compute())
    assert rrec is not None and rrec["stats"].n_rows >= 1
    for level, v in pins_before.items():
        assert app.engine.persistence_abs[level] == v, "an added ROI re-pinned a level"
    # its labels are the ROI's own, placed on the slide
    assert rrec["origin"] == (roi.rect[0], roi.rect[1]), rrec["origin"]
    rlayer = app.engine.label_layer(roi.key)
    assert rlayer.shape == tuple(src.level_shape(0))
    assert rlayer.id_at(roi.rect[0] - 5, roi.rect[1] - 5) == -1, "outside the ROI is background"

    app._remove_roi_at(0, 2)
    assert len(app._rois_of(0)) == 1

    doc = app._session_doc()
    assert doc["sequences"][0]["rois"], "ROI geometry did not reach the session document"
    app2 = MsPathApp(tk.Toplevel(root), autosave=False)
    app2._apply_session_doc(doc, source="selftest")
    assert len(app2.subsequences) == 1, app2.subsequences
    assert app2._item_at(0, 0).key == item.key, "the item key did not survive the session"
    assert len(app2._rois_of(0)) == 1, "the ROI did not survive the session"
    assert app2._item_at(0, 1).key == roi.key, "the ROI's key did not survive the session"

    root.destroy()
    print("selftest OK: pyramid preview, slide->sequence, overview item + key round-trip, "
          "prime + record, slide-coordinate positions, label layer, render + hover, "
          "persistence select, profile + session round-trip")
    return 0


if __name__ == "__main__":
    sys.exit(run_selftest())


def run_labeler_selftest():
    """``mspath-labeler --selftest``: the annotation layer over a placed item.

    The point of interest is the one the coupon labeler cannot exercise: the
    canvas draws in slide coordinates while the region raster is the item's, so
    every gesture, preview and training row has to cross that gap. A whole
    level-4-ish overview is used precisely because its scale is not 1 -- an
    identity placement would pass a level-0 test and fail here.
    """
    import numpy as np
    import tkinter as tk

    from msseg.labeler.pyramid import backends_available
    from msseg.labeler.labeling import touched_ids_over
    from .labeler import LabelerApp

    if not backends_available():
        print("selftest SKIPPED: no pyramid backend installed")
        return 0

    tmp = tempfile.mkdtemp(prefix="mspath-labeler-selftest-")
    slide, folder, origin = _pick_slide(tmp)
    print(f"[mspath] labeler selftest slide ({origin}): {slide}")
    try:
        root = tk.Tk()
    except Exception as exc:
        print(f"selftest SKIPPED: no display ({exc})")
        return 0
    root.withdraw()
    app = LabelerApp(root, autosave=False)

    app._add_folder_path(folder)
    app.file_list.selection_clear(0, "end")
    app.file_list.selection_set(app.all_files.index(slide))
    app._make_subsequence()
    src = app.engine.source(app._item_at(0, 0).slide)
    deepest = max(0, src.levels - 3)
    app.level_var.set(deepest)
    app._on_level_change()

    item = app._item_at(0, 0)
    profile = app._profile_for_compute()
    app.engine.prime_item(item, profile, halo=0)
    rec = app.engine.ensure_record(item.key, profile)
    app._rebuild_flat_slices()
    app._goto_slice(0)
    assert rec is not None and rec["stats"].n_rows >= 2, rec

    # -- the placement is the item's, not the identity -------------------- #
    place = app._region_placement()
    assert place.scale == rec["scale"] and (place.ox, place.oy) == tuple(rec["origin"])
    if rec["scale"] > 1:
        assert not place.identity, "a coarse item must not claim an identity placement"
    lh, lw = rec["labels"].shape[:2]
    # a slide point well past the raster's own extent still maps inside it
    far_x = int((lw - 1) * rec["scale"])
    rx, ry = place.to_raster(far_x + rec["origin"][0], rec["origin"][1])
    assert 0 <= int(rx) < lw and 0 <= int(ry) < lh, (rx, ry)

    # -- a gesture in SLIDE coordinates paints the regions under it ------- #
    layer = app.regions.label_layer(item.key)
    stats = rec["stats"]
    fid = stats.column("feature_id").astype(int)
    ex = stats.column("ext_x"); ey = stats.column("ext_y")
    target = int(fid[0])
    px, py = float(ex[0]), float(ey[0])
    assert layer.id_at(int(px), int(py)) == target, "the extremum is in slide coordinates"

    app.active_class_var.set(1)
    app._commit_interaction("taps", [(px, py)])
    assert len(app.store.interactions) == 1
    it = app.store.interactions[0]
    assert it.slice_key == item.key, it.slice_key
    assert touched_ids_over(it, layer, np) == {target}, "the gesture missed its region"

    # the class LUT resolves the same way the commit did
    entry = app._labels_cache_for(0, 0, rec, np)
    region_class = entry[5]
    assert region_class is not None and region_class[target] == 1, \
        "the class layer disagrees with the gesture"
    assert entry[4][1] == 1, entry[4][:3]

    # -- a box over the whole item paints every region -------------------- #
    x0, y0 = rec["origin"]
    x1 = x0 + lw * rec["scale"]; y1 = y0 + lh * rec["scale"]
    app.active_class_var.set(2)
    app._commit_interaction("box", [(float(x0), float(y0)), (float(x1), float(y1))])
    entry = app._labels_cache_for(0, 0, rec, np)
    counts = entry[4]
    assert counts[2] >= max(1, stats.n_rows - 1), (counts[:3], stats.n_rows)

    # -- a training row carries the drawn class --------------------------- #
    from msseg.labeler.training import TrainingSetBuilder
    builder = TrainingSetBuilder(app.FIELDS)
    cls = builder.row_classes(app.store.for_slice(item.key), rec["labels"], fid, np, layer)
    assert (cls > 0).any(), "no training row picked up a label"
    assert int(cls[0]) == 2, "the later gesture must win on a region both touch"

    # -- the model is pinned to the level --------------------------------- #
    assert app._feature_scope() == f"L{deepest}"
    names = app._expected_feature_names()
    assert names and "mean_base" in names and "ext_x" not in names
    assert app._check_model_compat(names, "selftest", f"L{deepest}") is None
    other = app._check_model_compat(names, "selftest", "L99")
    assert other and "L99" in other, other

    # -- undo puts it back ------------------------------------------------ #
    app._undo()
    assert len(app.store.interactions) == 1
    app._undo()
    assert not app.store.interactions

    root.destroy()
    print("labeler selftest OK: placement, slide-coordinate gestures, class layer, "
          "box over the item, training rows, level-scoped compat gate, undo")
    return 0
