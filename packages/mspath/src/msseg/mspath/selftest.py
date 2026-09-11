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

    # -- persistence is a select, not a prime, and the slider is LIVE ------ #
    before = app.engine.commit_id
    n_before = rec["stats"].n_rows
    abs_before = app.engine.persistence_abs[deepest]
    app.persist_var.set(min(40.0, float(app.persist_var.get()) + 20.0))
    app._on_persistence_change()
    assert app.engine.commit_id > before, "a threshold change must bump the commit"
    rec2 = app.engine.record(item.key)
    assert rec2 is not None and rec2 is not rec, "the record was not recomputed"
    # Not merely "no more regions": the threshold itself must have moved. The
    # first version cached the absolute per level, which left the slider doing
    # nothing while this assertion (<=) still passed.
    assert app.engine.persistence_abs[deepest] > abs_before, "the slider did not move the threshold"
    assert rec2["stats"].n_rows < n_before, "a 3x threshold must merge regions away"
    # the REFERENCE range is what stays pinned per level
    assert set(app.engine.level_range) == {deepest}, app.engine.level_range
    app.persist_var.set(10.0); app._on_persistence_change()
    assert app.engine.record(item.key)["stats"].n_rows == n_before, "and back again"

    # -- the Image dropdown shows the primed channels, placed on the slide -- #
    from .sources import PlacedImageSource
    assert list(app.background_combo.cget("values")) == list(app.CHANNELS)
    for channel in ("filtered", "base"):
        app.background_var.set(channel)
        app._refresh_render()
        shown = app.viewer.source
        assert isinstance(shown, PlacedImageSource), f"{channel}: {shown!r} is not the channel"
        assert shown.level_shape(0) == tuple(src.level_shape(0)), "placed on the whole slide"
        assert shown.scale == rec["scale"] and (shown.ox, shown.oy) == tuple(rec["origin"])
        lo, hi = shown.value_range()
        assert hi > lo, f"{channel} has no range -- a blank raster"
        r0 = rec["stats"].row_of_feature(int(rec["stats"].column("feature_id")[0]))
        app._on_hover(int(r0["ext_x"]), int(r0["ext_y"]))
        assert f"{channel}=" in app.hover_var.get(), app.hover_var.get()
    app.background_var.set("slide")
    app._refresh_render()
    # the ENGINE's pyramid, not the preview's: same file, a different handle
    back = app.viewer.source
    assert back is app.engine.source(item.slide) and back.path == src.path,         "slide shows the pyramid again"

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

    # Adding an ROI navigated to it, and navigating to an unprimed item primes
    # it ON THE WORKER -- the real on-demand path. Drive it the way the pump
    # would: wait for the worker, drain its events through the shell. (An
    # earlier version of this test primed the ROI synchronously here as well,
    # which ran two primes of one item on two threads over pipelines that are
    # not re-entrant, and left the worker's "done" undrained so pending_work()
    # stayed True for the rest of the test.)
    pins_before = dict(app.engine.level_range)
    assert app.engine._worker is not None and app.engine.pending_work(),         "no on-demand prime started"

    def settle(limit=6):
        """Run the pump by hand until the engine is idle. Draining one prime's
        events can start the next -- the primed handler primes whatever is on
        screen if it is not yet -- so this loops, as the real pump does."""
        for _ in range(limit):
            if not app.engine.pending_work():
                return
            w = app.engine._worker
            if w is not None:
                w.join(timeout=120)
                assert not w.is_alive(), "a prime did not finish"
            for ev in app.engine.poll():
                app._handle_event(ev)
        assert not app.engine.pending_work(), "the engine never went idle"

    settle()
    # Only the item ON SCREEN is selected after a prime, and the second
    # _add_roi moved the view to `big`. Go back to `roi`: its pipeline is live,
    # so this is the inline select, not another prime.
    app._goto_slice(app.flat_slices.index((0, 1)))
    settle()
    rrec = app.engine.record(roi.key)
    assert rrec is not None and rrec["stats"].n_rows >= 1, "the on-demand prime produced no record"
    for level, v in pins_before.items():
        assert app.engine.level_range[level] == v, "an added ROI re-pinned a level"
    # its labels are the ROI's own, placed on the slide
    assert rrec["origin"] == (roi.rect[0], roi.rect[1]), rrec["origin"]
    rlayer = app.engine.label_layer(roi.key)
    assert rlayer.shape == tuple(src.level_shape(0))
    assert rlayer.id_at(roi.rect[0] - 5, roi.rect[1] - 5) == -1, "outside the ROI is background"

    # -- a "primed" event must not throw away the item being looked at ----- #
    # The shell's flat rebuild resets navigation to item 0 (the overview).
    # Before this was caught, an ROI primed on demand was dropped from view
    # before its regions were ever computed -- the log showed the prime and
    # then never a region count.
    app._goto_slice(app.flat_slices.index((0, 1)))
    settle()
    assert app._current() == (0, 1)
    app.engine.commit_selection()                 # its record is now stale
    assert app.engine.record(roi.key) is None
    app._handle_compute_event(("primed",))
    assert app._current() == (0, 1), "the primed event moved the view back to the overview"
    assert app.engine.record(roi.key) is not None, "the ROI's regions were never computed"

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
          "live persistence slider, channel dropdown (slide/base/filtered), ROI tier, "
          "primed event keeps the current item, profile + session round-trip")
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

    def pytest_approx(v, tol=1e-9):
        class _A:
            def __eq__(self, other):
                return abs(float(other) - float(v)) <= tol
        return _A()

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

    # -- the loop: classify the overview, then propose where to look ------ #
    # Needs at least two classes with labels; the box above painted class 2
    # over everything, so give a couple of regions back to class 1.
    from msseg.mspath import propose as PR
    ids_by_area = np.argsort(-np.asarray(stats.column("area")))[:3]
    app.active_class_var.set(1)
    app._commit_interaction("taps", [(float(ex[i]), float(ey[i])) for i in ids_by_area])
    app._train_classifier()          # returns None on success; the model is the result
    if app._clf is None:
        print(f"[mspath] (classifier unavailable: {app.status_var.get()!r} "
              f"- skipping the proposal loop)")
    else:
        app._classify()
        pred = app._pred.get(item.key)
        assert pred is not None and pred[0] == rec["commit"], "the overview is not classified"
        # indexed by REGION ID, not by table row: that is the space a LUT
        # indexes, and it is what propose.candidates gathers through
        assert pred[2] is not None and pred[2].shape[0] > int(fid.max()),             "the probability matrix must span the region-id space"

        before = len(app._rois_of(0))
        app.roi_level_var.set(max(0, deepest - 2))
        app.propose_count_var.set(3)
        app.propose_size_var.set(256)
        app._propose_rois()
        added = len(app._rois_of(0)) - before
        assert added >= 1, "the proposal added nothing"
        assert "proposed" in app.status_var.get(), app.status_var.get()

        # every proposal is a real item, inside the slide, at the asked level
        sh0, sw0 = src.level_shape(0)
        for r in app._rois_of(0)[before:]:
            assert r["level"] == app.roi_level_var.get()
            assert 0 <= r["x"] and r["x"] + r["w"] <= sw0
            assert 0 <= r["y"] and r["y"] + r["h"] <= sh0
        keys = [app._item_at(0, li).key for li in range(1, 1 + len(app._rois_of(0)))]
        assert len(set(keys)) == len(keys), "a proposal duplicated an existing item"

        # and the ranking is the uncertainty, not the order of the table
        scores = PR.region_scores(pred[2], np, "entropy")
        cands = PR.candidates(rec["stats"], scores, np, app.FIELDS, min_area=4.0)
        assert cands and cands[0][0] >= cands[-1][0]
        # every candidate is a real region of the table, scored by its own id
        assert {c[3] for c in cands} <= set(fid.tolist())
        assert cands[0][0] == pytest_approx(scores[cands[0][3]])

    # -- undo puts it back ------------------------------------------------ #
    n = len(app.store.interactions)
    app._undo()
    assert len(app.store.interactions) == n - 1
    while app.store.interactions:
        app._undo()
    assert not app.store.interactions

    root.destroy()
    print("labeler selftest OK: placement, slide-coordinate gestures, class layer, "
          "box over the item, training rows, level-scoped compat gate, "
          "classify + propose ROIs, undo")
    return 0
