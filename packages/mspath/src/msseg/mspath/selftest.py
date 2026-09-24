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

import json
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


def _settle(app, limit=8):
    """Run the work-queue pump by hand until the engine is idle.

    Draining one prime's events can start the next -- the primed handler
    primes whatever is on screen if it is not yet -- so this loops, as the
    real pump does. Every selftest step that can leave a worker running
    (navigating to an ROI, proposing ROIs) settles before asserting, or the
    next step's `pending_work()` guard silently changes what it does.
    """
    for _ in range(limit):
        if not app.engine.pending_work():
            return
        w = app.engine._worker
        if w is not None:
            w.join(timeout=300)
            assert not w.is_alive(), "a prime did not finish"
        for ev in app.engine.poll():
            app._handle_event(ev)
    assert not app.engine.pending_work(), "the engine never went idle"


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

    # -- the session browser: a list of slides ---------------------------- #
    # A slide is added as a FILE; its folder is registered underneath so the
    # shell's session document keeps working, but no folder or file list is
    # shown (the shell's ones exist, unpacked, because the shell writes to them).
    assert app._add_slide_path(slide), "the slide was not added"
    assert not app._add_slide_path(slide), "adding a slide twice must be a no-op"
    assert len(app.subsequences) == 1 and app.folders, app.subsequences
    # the scrolled listboxes pack inside their own holders, which live in one
    # container that is never packed -- that container is the thing to check
    assert app._session_hidden.winfo_manager() == "", "the folder/file lists must stay unpacked"
    for hidden in (app.folder_list, app.file_list):
        assert hidden.winfo_toplevel() is app.subseq_list.winfo_toplevel()
        assert app._session_hidden in (hidden.master, hidden.master.master)
    assert app.subseq_list.winfo_manager() == "pack"
    assert app._sequence_row_text(app.subsequences[0]) == os.path.basename(slide)
    # a preview needs no Run: the pyramid IS the preview
    app._preview_file(slide)
    assert app.viewer.has_base, "the preview did not reach the canvas"
    src = app.viewer.source
    assert src.levels >= 2 and src.channels == 3, f"unexpected source {src}"
    # the slide's brightness window is measured from the pyramid at once
    lo, hi = app._channel_windows["slide"]
    assert 0.0 <= lo < hi <= 1.0 and app._window_channel == "slide"
    assert (float(app.vmin_var.get()), float(app.vmax_var.get())) == (lo, hi)

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
    # A layer's rev is its record's identity (record_keys.py), not a generation.
    assert layer is not None and layer.rev == rec["commit"]
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
    # the entry is the control (Enter / focus-out), as in the coupon viewer
    app.persist_live_var.set(f"{min(40.0, float(app.persist_var.get()) + 20.0):g}")
    app._on_persistence_change()
    assert float(app.persist_var.get()) == 30.0
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
    app.persist_live_var.set("10"); app._on_persistence_change()
    assert app.persist_label.get().startswith("= "), app.persist_label.get()
    app.persist_live_var.set("not a number"); app._on_persistence_change()
    assert float(app.persist_var.get()) == 10.0, "garbage in the entry must change nothing"
    assert app.engine.record(item.key)["stats"].n_rows == n_before, "and back again"

    # -- the Image dropdown shows the primed channels, placed on the slide -- #
    from .sources import PlacedImageSource
    # The picker offers the three rasters, then one channel per chain stage --
    # the state the chain passes through after it, which is what makes a chain
    # judgeable by looking.
    offered = list(app.background_combo.cget("values"))
    assert offered[:len(app.CHANNELS)] == list(app.CHANNELS), offered
    stages = offered[len(app.CHANNELS):]
    assert stages == app._stage_channels(), (stages, app._stage_channels())
    assert len(stages) == len([c for c in app.filter_cards
                               if c.get("operation") not in ("", "none")]), stages

    # A card's `show` radio ENTERS the walk; F continues it and steps off the
    # end back to what was being looked at. F elsewhere is still the framework's
    # A/B flip, which the block further down covers.
    app.background_var.set("slide"); app._on_image_channel_change()
    app.background_var.set(stages[0]); app._on_stage_radio()
    seen = [app._swap_image() for _ in range(len(stages))]
    assert seen[:len(stages) - 1] == stages[1:], seen
    assert seen[-1] == "slide", "stepping off the end returns to what was on screen"

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

    # -- each channel keeps its own brightness window, measured on first sight
    # and kept once moved.
    #
    # F used to be the framework's A/B flip (original <-> the derived channel
    # last shown). mspath overrides it to WALK the chain instead -- asserted
    # above -- because a chain is tuned by stepping along it, so the channel is
    # set directly here and the walk is tested on its own. ------------------- #
    for ch in ("slide", "base", "filtered"):
        lo, hi = app._channel_windows[ch]
        assert 0.0 <= lo < hi <= 1.0, (ch, lo, hi)
    assert app._original_channel() == "slide"
    app.background_var.set("filtered"); app._on_image_channel_change()
    assert (float(app.vmin_var.get()), float(app.vmax_var.get())) == app._channel_windows["filtered"]
    app.vmin_var.set(0.25); app._on_window_change()
    assert app._channel_windows["filtered"][0] == 0.25, "the slider edits the channel on screen"
    app.background_var.set("slide"); app._on_image_channel_change()
    assert app._channel_windows["filtered"][0] == 0.25 and float(app.vmin_var.get()) != 0.25,         "the slide's own window is back on the sliders"
    app.background_var.set("base"); app._on_image_channel_change()
    assert not app._typing(), "the dropdown hands the keyboard back"
    app._on_swap_key()
    assert app.background_var.get() == "slide"
    app._on_swap_key()
    assert app.background_var.get() == "base", "F returns to the channel picked last"
    view = app._view_state()
    assert set(view["windows"]) >= {"slide", "base", "filtered"} and view["swap_channel"] == "base"
    app.background_var.set("slide"); app._on_image_channel_change()

    # -- the live preview: a chain edit repaints without a Run ------------- #
    # The item is primed, so `filtered` normally shows the primed raster. An
    # edit puts the NEW chain on screen in its place -- computed on exactly
    # the pixels a prime reads, placed where the item sits -- and the region
    # overlays come off, because they are about a different field.
    app._preview_sync = True
    app._primed_chain = app._chain_fingerprint()
    chain_before = json.loads(json.dumps(app.filter_cards))
    app.background_var.set("filtered")
    app._refresh_render()
    primed_src = app._channel_source(item.key, "filtered", src)[0]
    assert primed_src is app.engine.primed[item.key].channel_sources["filtered"]
    app._launch_preview()                        # the chain as primed
    assert not app._preview_is_stale(), "the chain as primed is not stale"
    assert app._launch_preview() is None, "and asking again is free"
    assert app._channel_source(item.key, "filtered", src)[0] is not primed_src, \
        "though the live raster is what is on screen once computed"
    app.filter_cards[0]["operation"] = "blur"
    app.filter_cards[0]["params"] = {"sigma": 2.0}
    tok = app._launch_preview()
    assert tok == app._preview_token and tok > 0, "an edit recomputes"
    live = app._channel_source(item.key, "filtered", src)[0]
    assert live is not primed_src and live is app._preview_shown[3], "the live raster wins"
    assert (live.ox, live.oy, live.scale) == (primed_src.ox, primed_src.oy,
                                              primed_src.scale), \
        "placed where the item is"
    assert live.raster.shape == primed_src.raster.shape, "and the halo is trimmed"
    assert app._preview_is_stale() and app.viewer._hud_mode == "stale"
    import numpy as _np
    from msseg.viz import min_colors as _min_colors
    assert app._seg_overlays(0, 0, app.engine.record(item.key), None, _np,
                             _min_colors) == [], "the primed region overlays come off"
    assert app._preview_pending is None

    # -- a stage channel: the intermediate the chain passes through --------- #
    # `base` and `filtered` come from the primed record; a stage has no primed
    # counterpart, so picking one has to ASK for it, or the channel changes and
    # the picture does not. And a stage that carries colour must reach the
    # canvas as RGB -- PlacedImageSource stores (H, W, 3), which is what the
    # canvas hands to PIL; a CHW stack placed itself three pixels tall.
    app.filter_cards[0]["operation"] = "edges"
    app.filter_cards[0]["params"] = {"sigma": 1.0}
    app.filter_cards[1:] = [{"operation": "adapt",
                             "params": {"mode": "reduce", "how": "max"}}]
    app._rebuild_filter_cards("topo")
    stages = app._stage_channels()
    assert len(stages) == 2, stages

    from msseg.mscoupon import config_io
    plan = config_io.chain_plan(app.filter_cards, 3, "luminance")
    assert plan["stages"][0]["lifted"] and plan["stages"][0]["out"] == 3, plan
    assert plan["out"] == 1, "the chain reduces on its own, so no conversion is inserted"

    app.background_var.set(stages[0])
    app._on_image_channel_change()
    shown = app._channel_source(item.key, stages[0], src)[0]
    assert shown is not src, "picking a stage must not fall back to the slide"
    assert isinstance(shown, PlacedImageSource), shown
    assert shown.channels == 3, "a lifted edges is three planes, shown as RGB"
    region = shown.read_region(0, 0, 0, 8, 6)
    assert region.shape == (6, 8, 3), region.shape

    app.background_var.set(stages[1])
    app._on_image_channel_change()
    reduced = app._channel_source(item.key, stages[1], src)[0]
    assert reduced.channels == 1 and reduced is not shown, "after the reduction, one plane"
    assert reduced.read_region(0, 0, 0, 8, 6).shape == (6, 8)

    # Showing the stages submitted work of their own, so the token the next
    # block compares against is the one AFTER them.
    app.filter_cards[:] = json.loads(json.dumps(chain_before))
    app._rebuild_filter_cards("topo")
    app.background_var.set("filtered")
    app._on_image_channel_change()
    app._launch_preview()
    tok = app._preview_token

    # The slide channel is not a chain output, so it never recomputes.
    app.background_var.set("slide")
    assert app._launch_preview() is None and app._preview_token == tok
    assert app._channel_source(item.key, "slide", src)[0] is src
    # Retyping the old chain comes back from the cache; the live raster stays
    # on screen (it is what the chain on the panel produces either way), and
    # being no longer stale, the overlays come back.
    app.background_var.set("filtered")
    app.filter_cards[:] = json.loads(json.dumps(chain_before))
    assert app._launch_preview() is None, "the earlier chain is still cached"
    assert app._preview_token == tok, "so nothing was submitted"
    assert not app._preview_is_stale(), "back to the chain that was primed"
    live2 = app._channel_source(item.key, "filtered", src)[0]
    assert live2 is app._preview_shown[3] and live2 is not live, "the earlier raster"
    assert live2.raster.shape == primed_src.raster.shape, "the item's own extent"
    assert app.viewer._hud_mode != "stale"
    assert app._seg_overlays(0, 0, app.engine.record(item.key), None, _np,
                             _min_colors), "and the region overlays come back"
    # A Run clears the preview outright.
    app._preview_shown = ("x", "filtered", ("filtered", "{}"), primed_src)
    app._handle_compute_event(("primed",))
    assert app._preview_shown is None and app._primed_chain == app._chain_fingerprint()
    app._preview_sync = False
    app._primed_chain = None
    app.background_var.set("slide"); app._on_image_channel_change()

    # -- profiles + session round-trip ------------------------------------ #
    p = app._profile_from_ui()
    assert p["slide"]["overview_level"] == deepest
    app.level_var.set(0); app.halo_var.set(0)
    app._apply_profile_to_ui(p, lambda v, x: v.set(x), [])
    assert app.level_var.get() == deepest and app.halo_var.get() == p["slide"]["halo"]
    # the GPU switch is exposed, rides the profile, and reaches the params
    assert p["msc"]["use_gpu_gradient"] is False
    app.gpu_var.set(True)
    assert app._profile_from_ui()["msc"]["use_gpu_gradient"] is True
    assert app._profile_for_compute()["msc"].get("use_gpu_gradient") is True
    app.gpu_var.set(False)
    app._apply_profile_to_ui({**p, "msc": {**p["msc"], "use_gpu_gradient": True}},
                             lambda v, x: v.set(x), [])
    assert app.gpu_var.get()
    app.gpu_var.set(False)

    # -- the statistics panel is the feature vector, and it is in Processing - #
    assert app.stats_frame.master is app._processing_parent("stats")
    base_fields = set(app._stat_channel_names())
    assert base_fields == {"base"}
    on, sigmas, _src = app.stat_kind_vars["blur"]
    on.set(True); sigmas.set("1.5, 3.0"); app._on_stat_spec_change()
    names = app._stat_channel_names()
    assert "blur_s1.5" in names and "blur_s3" in names, names
    assert "channels" in app.stat_summary_var.get()
    prof = app._profile_from_ui()
    kinds = [c.get("kind") if isinstance(c, dict) else c for c in prof["statistics"]["channels"]]
    assert "blur" in kinds, prof["statistics"]
    # and it round-trips through the profile
    on.set(False); app._on_stat_spec_change()
    assert app._stat_channel_names() == ["base"]
    app._apply_profile_to_ui(prof, lambda v, x: v.set(x), [])
    assert app.stat_kind_vars["blur"][0].get() and "blur_s3" in app._stat_channel_names()
    on.set(False); app._on_stat_spec_change()

    # -- layout: ROI section in the left pane, Run at the shell's slot ------ #
    assert app.roi_hint_parent.winfo_toplevel() is app.left.winfo_toplevel()
    assert app.roi_hint_parent.master is app._left_section_parent("roi")
    assert app.run_frame.master is app._left_section_parent("run")
    assert not hasattr(app, "regions_check"), "no extra regions toggle in the viewer"

    # -- the busy badge follows the engine --------------------------------- #
    app.viewer.set_hud("busy", "Training")          # what the labeler's badge does
    app._update_busy()
    assert app.viewer.hud[0] is None, "an idle engine must take the badge down"

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

    def settle():
        _settle(app)

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
    app.engine.records.drop(roi.key)              # its record is gone
    assert app.engine.record(roi.key) is None
    app._handle_compute_event(("primed",))
    assert app._current() == (0, 1), "the primed event moved the view back to the overview"
    assert app.engine.record(roi.key) is not None, "the ROI's regions were never computed"

    # -- double-click brings the item into view ---------------------------- #
    app.viewer.canvas.winfo_width = lambda: 400
    app.viewer.canvas.winfo_height = lambda: 300
    app.viewer.set_view(0.0, 0.0, scale=1.0)               # far from the ROI
    app.subseq_list.selection_set("q0:1")
    app._on_seq_tree_double()
    v = app.viewer
    rx, ry, rw, rh = roi.rect
    assert app._current() == (0, 1)
    assert v.view_x <= rx and v.view_y <= ry, "the ROI's origin must be on screen"
    assert v.view_x + 400 * v.scale >= rx + rw and v.view_y + 300 * v.scale >= ry + rh, \
        "the ROI's far corner must be on screen"
    assert v.scale >= max(rw / 400.0, rh / 300.0), "zoomed to fit, not closer"
    app.subseq_list.selection_set("q0:0")
    app._on_seq_tree_double()                                # the overview: the slide fits
    assert v.view_x <= 0 and v.view_y <= 0 and v.scale >= max(sw / 400.0, sh0 / 300.0)

    app._remove_roi_at(0, 2)
    assert len(app._rois_of(0)) == 1

    doc = app._session_doc()
    assert doc["sequences"][0]["rois"], "ROI geometry did not reach the session document"

    # a folder on the command line means every slide in it, as one-file sequences
    app3 = MsPathApp(tk.Toplevel(root), autosave=False, initial=folder)
    assert len(app3.subsequences) >= 1
    assert all(len(sq["files"]) == 1 for sq in app3.subsequences)
    assert any(os.path.normpath(sq["files"][0]) == os.path.normpath(slide)
               for sq in app3.subsequences), "the folder's slides were not added"
    app2 = MsPathApp(tk.Toplevel(root), autosave=False)
    app2._apply_session_doc(doc, source="selftest")
    assert len(app2.subsequences) == 1, app2.subsequences
    assert app2._item_at(0, 0).key == item.key, "the item key did not survive the session"
    assert len(app2._rois_of(0)) == 1, "the ROI did not survive the session"
    assert app2._item_at(0, 1).key == roi.key, "the ROI's key did not survive the session"

    root.destroy()
    print("selftest OK: pyramid preview, slide->sequence, overview item + key round-trip, "
          "prime + record, slide-coordinate positions, label layer, render + hover, "
          "persistence entry, channel dropdown (slide/base/filtered), "
          "live preview on a chain edit, ROI tier, "
          "primed event keeps the current item, double-click views the item, "
          "profile + session round-trip")
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

    assert app._add_slide_path(slide)
    # Adding a slide works nothing (design note §5): the first task starts
    # empty, and the overview is enrolled only by choice.
    assert app._task.enrolled == {} and app.flat_slices == [] and app._current() is None
    assert app._browse_row == (0, None), "a new slide is browsed, not worked"
    assert app._enrol_row(0, 0) and app.flat_slices == [(0, 0)]
    assert app._task.enrolled == {app._item_at(0, 0).slide: {"overview": None}}
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

    # -- seam tools on a placed item: gestures in slide coordinates ---------- #
    import types as _types
    from msseg.labeler.seams import SEAM_BOUNDARY, SEAM_INTERIOR
    g_s = app.regions.seams(item.key, np)
    assert g_s is not None and g_s.n_seams > 0, "the overview has seams"
    assert g_s.placement.scale == rec["scale"]
    assert (g_s.placement.ox, g_s.placement.oy) == tuple(rec["origin"])
    v = app.viewer
    ctrl = v.tool
    view_before = (v.view_x, v.view_y, v.scale)
    undo_before = list(app._undo_stack)
    # one screen px == one raster px, screen (0, 0) == the item's origin
    v.view_x, v.view_y, v.scale = (float(rec["origin"][0]), float(rec["origin"][1]),
                                   float(rec["scale"]))

    def _ev(x, y):
        return _types.SimpleNamespace(x=int(x), y=int(y), x_root=int(x), y_root=int(y), state=0)

    app.tool_var.set("scope"); app.seam_class_var.set(SEAM_INTERIOR)
    assert ctrl.on_press(_ev(0, 0)) and ctrl.on_move(_ev(lw, lh)) and ctrl.on_release(_ev(lw, lh))
    sc = app.store.seams[-1]
    assert sc.tool == "scope"
    assert abs(sc.points[0][0] - rec["origin"][0]) < 1e-6 and abs(sc.points[0][1] - rec["origin"][1]) < 1e-6
    assert abs(sc.points[-1][0] - (rec["origin"][0] + lw * rec["scale"])) < 1e-6
    _g, cls_s = app._seam_classes_for(0, 0, np)
    assert (cls_s == SEAM_INTERIOR).all(), "a scope over the whole item labels every seam"
    open_s = np.flatnonzero(g_s.j0 >= 0)
    if len(open_s):
        s_long = int(open_s[np.argmax(g_s.lengths(np)[open_s])])
        pts_s = g_s.seam_points(s_long)
        (ax, ay), (bx, by) = pts_s[0].tolist(), pts_s[-1].tolist()
        app.tool_var.set("trace"); app.seam_class_var.set(SEAM_BOUNDARY)
        app.seam_toll_var.set("geometric")
        assert ctrl.on_press(_ev(ax, ay)) and ctrl.trace.active
        ctrl.trace.on_hover(*g_s.placement.to_image(bx, by))
        assert ctrl.trace._s["hover"][1] is not None, "the far corner is reachable"
        assert ctrl.on_press(_ev(bx, by)) and len(ctrl.trace._s["lw"].legs) == 1
        assert ctrl.trace.commit()
        tr = app.store.seams[-1]
        assert tr.tool == "trace" and tr.class_id == SEAM_BOUNDARY
        ix0, iy0 = g_s.placement.to_image(ax, ay)
        assert tr.points[0] == (ix0, iy0), "stored in slide coordinates"
        _g, cls_s = app._seam_classes_for(0, 0, np)
        assert (cls_s == SEAM_BOUNDARY).any(), "the trace labels the seams it runs along"
        app.store.remove_many([tr.uid])
    # leave nothing behind: the later sections count gestures and undo steps
    app.store.remove_many([sc.uid])
    app._rebuild_class_panels()
    app._undo_stack[:] = undo_before
    app._redo_stack.clear()
    app.tool_var.set("squiggle"); app.seam_toll_var.set("feature")
    v.view_x, v.view_y, v.scale = view_before

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
    # Keyed by the SLIDE, with the drawing item's scale of intent recorded.
    assert it.slice_key == item.slide, it.slice_key
    assert (it.si, it.li) == (0, 0) and it.bound
    assert it.meta["level"] == item.level and it.meta["scale"] == rec["scale"]
    assert it.meta["px"] == v.scale
    assert app._gestures_for_key(item.key) == [it] and app._gesture_on_item(it, 0, 0)
    assert touched_ids_over(it, layer, np) == {target}, "the gesture missed its region"
    # A gesture drawn much coarser than this item works at is marked, and
    # said once: a warning, not a refusal.
    assert app._annotation_mark(0, 0, 1) == "1"
    coarse = app.store.add("taps", [(px, py)], 1, item.slide, 0, 0,
                           meta={"level": item.level + 2, "scale": 4.0 * rec["scale"]})
    assert app._annotation_mark(0, 0, 2) == "2!" and len(app._coarse_gestures(0, 0)) == 1
    app.status_var.set("")
    app._coarse_notice(0, 0)
    assert "coarser" in app.status_var.get()
    app.status_var.set("")
    app._coarse_notice(0, 0)
    assert app.status_var.get() == "", "once per item and task"
    app.store.remove(coarse.uid)
    app._rebuild_class_panels()

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
    cls = builder.row_classes(app._gestures_for_key(item.key), rec["labels"], fid, np, layer)
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
    # An "-> edges" kind first: its pair model groups edges by the catalogue's
    # group, which here is the slide NAME. int("WSI/a.tiff") took the first
    # real Train down.
    app.model_kind_var.set("custom FC -> edges")
    app._train_classifier()
    assert app._clf is not None, f"edges kind failed to train: {app.status_var.get()!r}"
    assert app._edge_model is not None, "no edge model came out of an -> edges kind"
    app.model_kind_var.set("dense FC")
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
        _settle(app)          # proposing navigated to a new ROI, which primes on demand

        # every proposal is a real item, inside the slide, at the asked level
        sh0, sw0 = src.level_shape(0)
        for r in app._rois_of(0)[before:]:
            assert r["level"] == app.roi_level_var.get()
            assert 0 <= r["x"] and r["x"] + r["w"] <= sw0
            assert 0 <= r["y"] and r["y"] + r["h"] <= sh0
            # a place asked for by THIS task's model, and worked by it
            o = r["origin"]
            assert o["task"] == app._task.uid and o["reason"] == app.propose_method_var.get()
            assert isinstance(o["score"], float)
            assert app._task.enrolled[app._item_at(0, 0).slide][r["uid"]] == r["level"]
        keys = [app._item_at(0, li).key for li in range(1, 1 + len(app._rois_of(0)))]
        assert len(set(keys)) == len(keys), "a proposal duplicated an existing item"

        # -- a refused magic-fill press says so where the eye is -------------- #
        import types
        v = app.viewer
        # proposing navigated to the last ROI it cut; the press is on the
        # classified OVERVIEW
        app._goto_slice(app.flat_slices.index((0, 0)))
        v.set_view(px - 10, py - 10, scale=1.0)
        e = types.SimpleNamespace(x=10, y=10, state=0)
        app.tool_var.set("magic"); app.magic_metric_var.set("learned")
        app.status_var.set("")
        assert v.tool.on_press(e) is False, "learned without an edge model must be refused"
        assert "edge model" in app.status_var.get()
        assert v.hud[0] == "stale" and "edge model" in v.hud[1],             "the refusal must reach the canvas HUD, not only the status bar"
        app._update_busy()
        assert v.hud[0] == "stale", "a repaint must not clear a fresh notice"
        app._notice_until = 0.0
        app._update_busy()
        assert v.hud[0] is None, "an expired notice is cleared"

        # -- and proba works once classified, and SURVIVES visiting an ROI ---- #
        app.magic_metric_var.set("proba")
        assert v.tool.on_press(e) is True, "proba refused after Classify"
        v.tool.on_release(e); app._end_preview()
        pred_before = app._pred.get(item.key)
        assert pred_before is not None
        # an on-demand ROI prime is INCREMENTAL: it must not drop the
        # overview's predictions the way a Run does
        sh0, sw = src.level_shape(0)
        new_roi = app._add_roi(0, app.roi_level_var.get(), sw // 3, sh0 // 3,
                               int(256 * src.level_scale(app.roi_level_var.get())),
                               int(256 * src.level_scale(app.roi_level_var.get())))
        assert new_roi is not None
        _settle(app)
        assert app.engine.record(new_roi.key) is not None, "the ROI did not prime"
        assert app._pred.get(item.key) is pred_before,             "an incremental prime wiped the overview's predictions"
        app._goto_slice(app.flat_slices.index((0, 0)))
        v.set_view(px - 10, py - 10, scale=1.0)
        assert v.tool.on_press(e) is True, "proba refused after visiting an ROI"
        v.tool.on_release(e); app._end_preview()

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

    # -- tree rows: slide / overview / ROI, ownership, guarded removal ------ #
    from unittest import mock as _mock
    from tkinter import messagebox
    from msseg.labeler.labeling import LabelStore
    _settle(app)
    n_rois = len(app._rois_of(0))
    lvl = max(0, deepest - 1)
    app.roi_level_var.set(lvl)
    side = int(64 * src.level_scale(lvl))
    cut = app._add_roi(0, lvl, 0, 0, side, side)
    assert cut is not None
    _settle(app)                       # navigating there primed it on demand
    li = 1 + len(app._rois_of(0)) - 1
    assert app._current() == (0, li) and app.engine.record(cut.key) is not None
    assert (app._row_kind(0, None), app._row_kind(0, 0), app._row_kind(0, li)) == \
        ("slide", "overview", "ROI")
    assert app._remove_target(0, 0) == (0, None), "the overview goes with its slide"
    assert app._remove_target(0, li) == (0, li)
    labels = [e[0] if e else None for e in app._seq_tree_menu_entries(0, 0)]
    assert labels == ["Go to", "Unenrol from 'task 1'", None, "Clear annotations",
                      "Remove slide\u2026"], labels
    roi_labels = [e[0] for e in app._seq_tree_menu_entries(0, li) if e]
    assert roi_labels[-1] == "Remove ROI\u2026" and "Unenrol from 'task 1'" in roi_labels
    assert "Note\u2026" in roi_labels and any(l.startswith("Work at L") for l in roi_labels)
    assert "ROI" in app._row_description(0, li) and "overview" in app._row_description(0, None)
    # one annotation on the overview, one inside the ROI: both are the
    # SLIDE's (docs/design_multi_model_tasks.md \u00a78) -- each item sees those
    # meeting its place -- and a row keyed by an item, as an older store
    # wrote them, rebases to the slide with the item's level recorded
    app.active_class_var.set(1)
    app._goto_slice(app.flat_slices.index((0, 0)))
    app._commit_interaction("taps", [(px, py)])
    app._goto_slice(app.flat_slices.index((0, li)))
    # Inside the ROI's rect (an extremum can sit in the compute halo, which
    # the rect query rightly does not count as the ROI's).
    cx, cy = cut.rect[0] + cut.rect[2] / 2.0, cut.rect[1] + cut.rect[3] / 2.0
    app._commit_interaction("taps", [(float(cx), float(cy))])
    a_ovw, a_roi = app.store.interactions[-2:]
    assert a_ovw.slice_key == a_roi.slice_key == item.slide, "gestures are the slide's"
    assert (a_roi.si, a_roi.li) == (0, 0), "the hints name the slide's first row"
    assert a_roi.meta["level"] == lvl and a_roi.meta["scale"] == app.engine.record(cut.key)["scale"]
    assert a_roi.meta["px"] == app.viewer.scale
    old = app.store.add("taps", [(1.0, 1.0)], 1, f"{item.slide}@{lvl}#1,2,64,64")
    app._rebind_store(app.store)
    assert old.slice_key == item.slide and old.bound, "an item-keyed row moves to its slide"
    assert old.meta["level"] == lvl and \
        old.meta["scale"] == float(app.engine.source(item.slide).level_scale(lvl))
    n_roi = len(app._gestures_for(0, li))
    assert n_roi >= 2 and app._gesture_on_item(a_roi, 0, li) and app._gesture_on_item(old, 0, li)
    assert len(app._gestures_for(0, 0)) == 3 and app._gesture_on_item(a_ovw, 0, 0), \
        "the overview sees every gesture on the slide"
    assert app._row_owns_key(0, None, item.slide) and not app._row_owns_key(0, li, item.slide)
    assert app._row_owns_key(0, None, cut.key), "a slide row still owns item keys"
    assert len(app._row_interactions(0, None)) == 3 and len(app._row_interactions(0, li)) == n_roi
    labels = [e[0] if e else None for e in app._seq_tree_menu_entries(0, None)]
    assert labels == ["Go to", "Enrol every place in 'task 1'", None,
                      "Clear annotations\u2026 (3)", "Remove slide\u2026"], labels
    # the ROI: removing it dooms nothing -- its gestures are the slide's and
    # stay, visible on the overview -- the engine forgets it and the view
    # lands on a neighbour; a re-cut of the same rect sees them again with no
    # rebind at all
    msg = app._remove_rows_message([(0, li)])
    assert "ROI" in msg and "belong to the slide" in msg, msg
    assert app._doomed_interactions([(0, li)]) == []
    with _mock.patch.object(messagebox, "askyesno", return_value=False):
        assert not app._remove_rows_guarded([(0, li)])
    assert len(app._rois_of(0)) == n_rois + 1
    with _mock.patch.object(messagebox, "askyesno", return_value=True):
        assert app._remove_rows_guarded([(0, li)])
    _settle(app)
    assert len(app._rois_of(0)) == n_rois
    assert app.engine.record(cut.key) is None and cut.key not in app.engine.primed
    assert len(app.store.interactions) == 3 and all(it.bound for it in app.store.interactions)
    assert len(app._gestures_for(0, 0)) == 3, "still the slide's, still on the overview"
    cur = app._current()
    assert cur is not None and cur[0] == 0 and cur[1] < li
    again = app._add_roi(0, lvl, 0, 0, side, side)
    assert again is not None and again.key == cut.key, "the key is the place"
    _settle(app)
    assert len(app._gestures_for(0, li)) == n_roi, "a re-cut sees its gestures again"
    doc_before = app.store.to_json()
    app._install_store(LabelStore.from_json(doc_before))
    assert all(it.bound for it in app.store.interactions), \
        "slide keys bind through the catalogue on a load"
    assert app.store.to_json() == doc_before, "a load of slide-keyed rows changes nothing"
    # the slide: everything on it goes, the pyramid is closed and the canvas
    # cleared
    assert len(app._doomed_interactions([(0, None)])) == 3
    with _mock.patch.object(messagebox, "askyesno", return_value=True):
        assert app._remove_rows_guarded([app._remove_target(0, 0)])
    assert app.subsequences == [] and app.flat_slices == []
    assert not app.store.interactions and not app.engine.primed and not app.engine.slices
    assert item.slide not in app.engine.sources and item.slide not in app.engine.paths
    assert not app.viewer.has_base, "the canvas shows nothing of a closed slide"
    app._undo()
    assert len(app.store.interactions) == 3 and not any(it.bound for it in app.store.interactions)

    # Tasks: a second detector over the same slides has its own store and
    # model; a profile built from a model binds to the active task; both
    # tasks ride the session document into a fresh window.
    t1 = app._task
    clf1 = app._clf
    assert clf1 is not None
    t2 = app._task_new("stroma")
    assert app._task is t2 and app._clf is None and app.store.interactions == []
    assert app._pred == {} and t2.workflow == t1.workflow
    assert str(app.classify_btn.cget("state")) == "disabled"
    assert app._activate_task(t1) and app._clf is clf1 and len(app.store.interactions) == 3
    assert str(app.classify_btn.cget("state")) == "normal"
    n_prof = len(app.profiles)
    prof = app._profile_from_model(os.path.join("C:", "x", "gland_model.pkl"), {"channels": ["base"]})
    assert len(app.profiles) == n_prof + 1 and app.active_profile_idx == n_prof
    assert t1.workflow == prof["name"] and t2.workflow != prof["name"], \
        "a profile from a model becomes the active task's workflow only"
    doc_t = app._session_doc()
    assert doc_t["session_version"] == 3 and len(doc_t["tasks"]) == 2
    assert doc_t["active_task"] == t1.uid and doc_t["tasks"][0]["workflow"] == prof["name"]
    app2 = LabelerApp(tk.Toplevel(root), autosave=False)
    app2._apply_session_doc(doc_t, "tasks")
    assert [t.name for t in app2.tasks] == [t1.name, "stroma"] and app2._task.uid == t1.uid
    assert len(app2.store.interactions) == 3 and app2.tasks[1].gesture_count == 0
    assert app2.profiles[app2.active_profile_idx]["name"] == prof["name"]
    assert app2._feature_scope() == app._feature_scope()

    # -- Places and enrolment (design note §5): a place is the slide's, the
    # work on it the task's; nothing is worked unless enrolled ------------- #
    import json as _json
    import re as _re
    from . import places as _places
    app3 = LabelerApp(tk.Toplevel(root), autosave=False)
    assert app3._add_slide_path(slide)
    tA = app3._task
    assert tA.enrolled == {} and app3.flat_slices == [] and app3._browse_row == (0, None)
    app3.active_class_var.set(1)
    app3._commit_interaction("taps", [(10.0, 10.0)])
    assert app3.store.interactions == [], "nothing is annotated on a browsed slide"
    src3 = app3.engine.source(app3._item_at(0, 0).slide)
    lvl = max(1, src3.levels - 2)
    side = int(96 * src3.level_scale(lvl))
    p1 = app3._add_roi(0, lvl, 0, 0, side, side)       # cut from the view: enrols A only
    _settle(app3)
    assert p1 is not None and app3.flat_slices == [(0, 1)] and app3._current() == (0, 1)
    place = app3._rois_of(0)[0]
    uid = place["uid"]
    assert _re.fullmatch(r"p_[0-9a-f]{6}", uid)
    assert place["origin"] == {"reason": "view", "task": tA.uid}
    sid3 = p1.slide
    assert tA.enrolled == {sid3: {uid: lvl}}
    again3 = app3._add_roi(0, lvl, 0, 0, side, side)
    assert again3.key == p1.key and len(app3._rois_of(0)) == 1, "the same rect is one place"
    # The whole slide shows the places: the task's solid with its level.
    app3._browse(0, None)
    assert app3._current() is None
    c3 = app3.viewer.canvas
    assert len(c3.find_withtag("place_worked")) == 1 and not c3.find_withtag("place_other")
    assert c3.itemcget(c3.find_withtag("place_label")[0], "text") == f"L{lvl}"
    x0, y0, x1, y1 = c3.coords(c3.find_withtag("place_worked")[0])
    app3.viewer.set_view(scale=app3.viewer.scale / 2.0)
    app3._redraw_hover_geometry()
    X0, Y0, X1, Y1 = c3.coords(c3.find_withtag("place_worked")[0])
    assert abs((X1 - X0) - 2 * (x1 - x0)) < 2, "the outline follows a zoom"
    # Task B works nothing: the place is listed greyed, drawn dashed, and a
    # click on it browses -- no current item, nothing primed.
    tB = app3._task_new("stroma")
    assert tB.enrolled == {} and app3.flat_slices == [] and app3._current() is None
    assert app3._row_tags(0, 1) == (app3.UNENROLLED_TAG,)
    assert app3._row_tags(0, None) == (app3.UNENROLLED_TAG,)
    app3._browse(0, None)
    assert len(c3.find_withtag("place_other")) == 1 and not c3.find_withtag("place_worked")
    assert app3._goto_row(0, 1) and app3._current() is None and app3._browse_row == (0, 1)
    # Enrolled at another level in B: one place, two items.
    lvl2 = lvl - 1
    assert app3._enrol_row(0, 1, lvl2)
    _settle(app3)
    assert app3.flat_slices == [(0, 1)] and app3._current() == (0, 1), "enrolling a browsed row selects it"
    kB = app3.catalogue.key_of(0, 1)
    assert kB != p1.key and f"@{lvl2}#" in kB and tB.enrolled == {sid3: {uid: lvl2}}
    assert app3._row_tags(0, 1) == () and app3._tasks_working(0, 1) == [tA.name, tB.name]
    # A level that makes a place degenerate is refused, not shrunk.
    app3._rois_of(0).append({"level": 0, "x": 400, "y": 300, "w": 40, "h": 40})
    app3._ensure_place_uids()
    assert not app3._enrol_row(0, 2, src3.levels - 1)
    assert "too small" in app3.status_var.get(), app3.status_var.get()
    app3._rois_of(0).pop()
    app3._rebuild_flat_slices()
    # A's work is untouched, and its caches survive a switch away and back.
    assert app3._activate_task(tA) and app3.catalogue.key_of(0, 1) == p1.key
    app3._pred[p1.key] = ("kept",)
    assert app3._activate_task(tB) and app3._activate_task(tA)
    assert app3._pred.get(p1.key) == ("kept",), "a switch does not drop predictions"
    app3._pred.clear()
    # Run task primes the active task's items, Run all every task's on the workflow.
    assert [i.key for i in app3._prime_items("task")] == [p1.key]
    assert sorted(i.key for i in app3._prime_items("all")) == sorted([p1.key, kB])
    # The session keeps places and enrolments; a document written before
    # enrolment reads as "works its places", never the overview.
    doc3 = app3._session_doc()
    assert doc3["sequences"][0]["rois"][0]["uid"] == uid
    assert [t.get("enrolled") for t in doc3["tasks"]] == [{sid3: {uid: lvl}}, {sid3: {uid: lvl2}}]
    app4 = LabelerApp(tk.Toplevel(root), autosave=False)
    app4._apply_session_doc(_json.loads(_json.dumps(doc3)), "enrolment")
    assert app4._rois_of(0)[0]["uid"] == uid
    assert [t.enrolled for t in app4.tasks] == [{sid3: {uid: lvl}}, {sid3: {uid: lvl2}}]
    legacy = _json.loads(_json.dumps(doc3))
    for t in legacy["tasks"]:
        t.pop("enrolled")
    for r in legacy["sequences"][0]["rois"]:
        r.pop("uid")
    notes4 = []
    app5 = LabelerApp(tk.Toplevel(root), autosave=False)
    app5._apply_session_doc(legacy, "legacy", notes4)
    uid5 = app5._rois_of(0)[0]["uid"]
    assert [t.enrolled for t in app5.tasks] == [{sid3: {uid5: lvl}}] * 2, \
        "a legacy task works its places at their own level"
    assert not _places.overview_enrolled(app5.tasks[0].enrolled, sid3)
    assert any("overview" in n for n in notes4), notes4
    # Removing the place removes it for every task, and says who worked it.
    msg3 = app3._remove_rows_message([(0, 1)])
    assert f"'{tA.name}'" in msg3 and f"'{tB.name}'" in msg3, msg3
    with _mock.patch.object(messagebox, "askyesno", return_value=True):
        assert app3._remove_rows_guarded([(0, 1)])
    _settle(app3)
    assert app3._rois_of(0) == [] and all(not t.enrolled for t in app3.tasks)
    assert p1.key not in app3.engine.primed and kB not in app3.engine.primed

    # -- the MSC and the statistics are cached apart ------------------------ #
    # The statistics live on the Features tab. An edit there re-measures the
    # item on screen on the worker: its labels stay, its columns follow the
    # spec, and nothing is primed; a profile that differs only in its
    # statistics keeps the live items across the switch.
    app = LabelerApp(tk.Toplevel(root), autosave=False)
    assert app._add_slide_path(slide) and app._enrol_row(0, 0)
    app.level_var.set(deepest)
    app._on_level_change()
    app._goto_slice(0)
    assert app.stats_frame.master.master is app._processing_parent("stats")   # a Collapsible
    assert app._processing_parent("stats") is app.feat_col
    assert app._processing_parent("filters") is app.proc_col
    kk = app.catalogue.key_of(*app._current())
    app.engine.prime_item(app._item_at(*app._current()), app._profile_for_compute(), halo=0)

    # -- the stage strip: msc -> stats -> classified <- model --------------- #
    rec_s = app.engine.ensure_record(kk, app._profile_for_compute())
    app._refresh_stages()
    assert app.viewer.stages == {"msc": "ok", "stats": "ok", "model": "none",
                                 "classified": "none"}, app.viewer.stages
    for box, tab in (("msc", "Processing"), ("stats", "Features"), ("model", "Model"),
                     ("classified", "Annotation")):
        _k, x0, y0, x1, y1, _tip = [b for b in app.viewer._stage_boxes if b[0] == box][0]
        ev = _types.SimpleNamespace(x=(x0 + x1) // 2, y=(y0 + y1) // 2)
        app.viewer._drag_start(ev); app.viewer._drag_end(ev)
        assert app._center_tab_name() == tab, (box, app._center_tab_name())
    expected = app._expected_names_for(None)
    if expected is not None:
        # A model trained here (a stand-in estimator: the strip never predicts).
        with _mock.patch.object(app, "_refresh_model_readout"), \
                _mock.patch.object(app, "_refresh_edge_readout"), \
                _mock.patch.object(app, "_refresh_confusion"):
            app._install_model(object(), list(expected), "random forest")
        app._pred[kk] = (rec_s["commit"], np.ones(int(rec_s["n_ids"]), np.int64), None)
        app._refresh_stages()
        assert app.viewer.stages == {"msc": "ok", "stats": "ok", "model": "ok",
                                     "classified": "ok"}, app.viewer.stages
        # An annotation edit since training: the model, and what it classified,
        # are out of date.
        app.store.rev += 1
        app._refresh_stages()
        st = app.viewer.stages
        assert st["model"] == "stale" and st["classified"] == "stale" and st["msc"] == "ok", st
        assert "Train again" in app.viewer.stage_tip("model")
        app._task.model.trained_rev = app.store.rev
        # An un-Run topology edit: msc and everything downstream of it.
        edited = _json.loads(_json.dumps(app._profile_for_compute()))
        edited["filters"] = list(edited.get("filters") or []) + [
            {"operation": "blur", "params": {"sigma": 5.0}}]
        with _mock.patch.object(app, "_profile_for_compute", return_value=edited):
            app._refresh_stages()
            st = app.viewer.stages
        assert st == {"msc": "stale", "stats": "stale", "model": "ok",
                      "classified": "stale"}, st
        assert "Run task" in app.viewer.stage_tip("msc")
        # A compute badge spins its box, and clears.
        app._compute_badge("Training 2/5")
        assert app.viewer.stages["model"] == "busy" and app.viewer.hud[0] is None
        app._clear_compute_badge()
        assert app.viewer.stages["model"] == "ok"
        app._task.model.reset()
        app._pred.clear()
        app._refresh_stages()
    strip_checked = "stage strip: states, clicks -> tabs, staleness, spinner"
    if app.engine.can_remeasure(kk):
        rec0 = app.engine.ensure_record(kk, app._profile_for_compute())
        labels0 = np.array(rec0["labels"], copy=True)
        live0 = app.engine.primed[kk]
        primes, seen = [], []
        orig_prime, orig_handle = app.engine.prime_item, app._handle_event
        app.engine.prime_item = lambda *a, **k: primes.append(1) or orig_prime(*a, **k)
        app._handle_event = lambda ev: seen.append(ev[0]) or orig_handle(ev)
        blur_on, blur_sig, _src = app.stat_kind_vars["blur"]
        blur_on.set(True); blur_sig.set("2.0")
        app._on_stat_spec_change()
        app._refresh_stages()
        assert app.viewer.stages["stats"] == "stale", "an unsettled Features edit"
        assert app._stat_edit_after is not None, "the edit settles before it measures"
        app.root.after_cancel(app._stat_edit_after)
        app._stat_edit_settled()
        assert app.engine.running_kind == "measure" or not app.engine.pending_work()
        _settle(app)
        rec1 = app.engine.record(kk) or app.engine.ensure_record(kk, app._profile_for_compute())
        assert "mean_blur_s2" in rec1["stats"].names, rec1["stats"].names[:8]
        assert np.array_equal(rec1["labels"], labels0), "a re-measure keeps the regions"
        assert primes == [] and "primed" not in seen, (primes, seen)
        assert app.engine.primed[kk] is live0 and live0.live
        app._refresh_stages()
        assert app.viewer.stages["stats"] == "ok", "re-measured: the stats box is green"
        # A profile that differs only in its statistics: kept, re-measured.
        app._snapshot_active_profile()
        other = _json.loads(_json.dumps(app.profiles[app.active_profile_idx]))
        other["name"] = "stats only"
        other["statistics"]["channels"] = ["base"]
        app.profiles.append(other)
        app._switch_profile(len(app.profiles) - 1)
        _settle(app)
        assert app.engine.primed.get(kk) is live0 and live0.live, "the switch kept the pipe"
        rec2 = app.engine.record(kk) or app.engine.ensure_record(kk, app._profile_for_compute())
        assert "mean_blur_s2" not in rec2["stats"].names
        assert np.array_equal(rec2["labels"], labels0) and primes == []
        # Task switches keep what a task computed. A (this task, on "stats
        # only") holds a prediction on rec2; B, a copy of A, works the same
        # items on the blur workflow. Back in A the record is A's again --
        # the same id, so A's prediction is valid -- with no re-measure.
        tA = app._task
        fake_final = np.ones(int(rec2["n_ids"]), np.int64)            # every region: class 1
        app._pred[kk] = (rec2["commit"], fake_final, None)
        tB = app._task_duplicate("B")
        assert tB is not None and app._task is tB
        app._switch_profile(len(app.profiles) - 2)             # B on the blur workflow
        _settle(app)
        recB = app.engine.record(kk) or app.engine.ensure_record(kk, app._profile_for_compute())
        assert "mean_blur_s2" in recB["stats"].names and recB["commit"] == rec1["commit"], \
            "the blur record was found again, not re-measured into a new one"
        measures = []
        orig_rm = app.engine.remeasure_item
        app.engine.remeasure_item = lambda *a, **k: measures.append(1) or orig_rm(*a, **k)
        assert app._activate_task(tA)
        _settle(app)
        assert app.engine.record(kk)["commit"] == rec2["commit"] == tA.caches.pred[kk][0], \
            "back in A, A's record and so A's prediction are current again"
        assert measures == [] and primes == []
        app.engine.remeasure_item = orig_rm
        app.engine.prime_item, app._handle_event = orig_prime, orig_handle
        # Run all tasks primes a task on ANOTHER field under its own workflow,
        # into its own slot; switching to it then needs no prime, and A's pipe
        # is still there when A comes back.
        from msseg.mscoupon.fingerprints import field_fingerprint_of as _ffp
        coarse = _json.loads(_json.dumps(app.profiles[app._profile_index(tA.workflow)]))
        coarse["name"] = "coarse"
        coarse["filters"] = list(coarse.get("filters") or []) + [
            {"operation": "blur", "params": {"sigma": 3.0}}]
        app.profiles.append(coarse)
        tC = app._task_duplicate("C")
        app._switch_profile(len(app.profiles) - 1)           # C on "coarse": no prime
        _settle(app)
        assert app.engine.primed.get(kk) is None, "a new field starts empty -- and unprimed"
        assert app._activate_task(tA)
        _settle(app)
        assert app.engine.primed.get(kk) is live0, "A's field slot kept its pipe"
        jobs = app._prime_items("all")
        assert any(len(j) == 2 and isinstance(j[1], dict) and j[0].key == kk          # (Item, params)
                   and j[1]["filters"][-1]["params"]
                   == {"sigma": 3.0} for j in jobs), jobs
        app._run("all")
        _settle(app)
        assert app._activate_task(tC)
        _settle(app)
        pC = app.engine.primed.get(kk)
        coarse_doc = app._profile_for_compute()
        assert pC is not None and pC.field == _ffp(coarse_doc), "C's item was primed by Run all"
        n_primes = []
        app.engine.prime_item = lambda *a, **k: n_primes.append(1) or orig_prime(*a, **k)
        recC = app.engine.record(kk) or app.engine.ensure_record(kk, coarse_doc)
        assert recC is not None and n_primes == []
        assert app._activate_task(tA)
        _settle(app)
        assert app.engine.primed.get(kk) is live0 and live0.live and n_primes == []
        app.engine.prime_item = orig_prime
        remeasured = ("statistics re-measured without a prime, stats-only switch keeps "
                      "pipes, a task switch back finds the task's record and prediction, "
                      "Run all tasks primes another field into its own slot")
    else:
        remeasured = "re-measure SKIPPED (extension predates it)"

    root.destroy()
    print("labeler selftest OK: placement, slide-coordinate gestures, class layer, "
          "box over the item, training rows, level-scoped compat gate, "
          "classify + propose ROIs, refusal notice on the HUD, "
          "predictions survive an incremental prime, undo, tree rows: "
          "slide/overview/ROI menu + guarded removal, slide-bound gestures: "
          "the slide's key + scale of intent + rect query + item-key rebase + "
          "an ROI's removal keeps them + the coarse mark, "
          "seam tools in slide coordinates, tasks: own store + model, "
          "profile-from-model binds the active task, two tasks into a fresh window, "
          f"{strip_checked}, {remeasured}, "
          "places + enrolment: nothing worked until enrolled, browse, outlines, "
          "one place at two levels, Run task / all, round trip + legacy, removal")
    return 0
