"""mscoupon interactive labeler: annotate MSC regions with classes.

A fork of the viewer (``mscoupon-gui``) for fast class annotation: arm a class
(left-click its color swatch, or the numeric hotkey; right-click the swatch
recolors it), then draw over the slice --

    squiggle   every living MSC region under the polyline gets the class
    box        every region intersecting the dragged rectangle
    lasso      every region under the filled (auto-closed) drag polygon

Class 0 is "no label"; while it is armed the left mouse button pans as in the
viewer (middle-drag always pans). Each gesture is an *interaction*, listed in
its class's subpanel on the right; interactions resolve in creation order (a
later gesture paints over an earlier one), can be dragged between class
subpanels (or moved via right-click), and are stored as raw geometry -- so
changing persistence and hitting Rerun re-resolves them against the new
regions for free.

The class layer renders through the canvas's label-overlay path: one RGBA LUT
per slice (region id -> class color), rebuilt only when an interaction or the
segmentation commit changes, gathered over the visible crop at render time.
The ``regions`` checkbox says whether that layer is drawn; the dropdown beside
it says how it is *colored* -- the region-id palette, ``P(class k)``, or
prediction ``uncertainty`` (1 - the top-two probability margin). The scalar
modes replace the id LUT rather than tinting it, and appear only once Classify
has filled the probability cache. ``outlines`` paints every on-slice gesture
at once instead of only what the pointer is over, and a right-CLICK on the
image plane (a right-DRAG still pans) offers the same move/delete menu the
class-list rows do.

Train freezes a prediction per region; the confusion matrix under the class
stack then counts frozen-prediction against live-label, so labeling more moves
only the true axis -- "old prediction, new value". Clicking a cell highlights
its regions on the current slice. Early counts are resubstitution (the truth
IS the training set); the reading that matters is what moves after a Train.

Loading a classifier saved under different ``statistics`` no longer just
refuses: the v2 pickle carries the statistics it was trained under, and the
labeler offers to build a profile from them (keeping the active profile's
filters/MSC/selection) and switch to it -- which drops the primed data, since
the per-slice feature table is baked at prime time.

Everything else -- sequences, filter chains, statistics, priming, persistence,
config export, session autosave -- is inherited from the viewer. The exported
folder additionally receives ``annotations.json`` (the raw gestures), and the
session autosaves under its own file (``mscoupon-labeler``), never the
viewer's.

Run:  mscoupon-labeler [folder]     |     mscoupon-labeler --selftest
"""
from __future__ import annotations

import os
import sys
import json
import math
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from . import config_io
from . import session
from .app import MscouponApp
from msseg.labeler.annotate import AnnotationShell
from msseg.labeler.defaults import *  # noqa: F401,F403  (the selftest reads the tunables)
from msseg.labeler.tools import DrawController, MagicFillController, _extremum_points  # noqa: F401
from .common import log
from msseg.labeler.widgets import ScrollFrame, attach_tooltip
from msseg.labeler.labeling import (LabelStore, MAX_CLASSES, TOOLS,
                       resolve_slice, resolve_sets, touched_sets, class_lut,
                       scalar_lut, line_pixels, polygon_mask, preview_lut)
from msseg.labeler import magic_fill
from msseg.labeler import model_search
from msseg.labeler import edge_model
from msseg.labeler import bundle as model_bundle
from msseg.labeler import fields
from msseg.labeler.training import TrainingSetBuilder, TrainingProblem

class LabelerApp(AnnotationShell, MscouponApp):
    SESSION_APP = "mscoupon-labeler"
    APP_TITLE = "mscoupon labeler"
    WINDOW_TITLE = "mscoupon labeler"
    MODEL_APP_TAG = model_bundle.DEFAULT_APP_TAG     # the classifier pickle "app" tag
    FIELDS = fields.DEFAULT                          # column conventions of the statistics table

    def _default_profile(self, name="default"):
        return session.default_profile(name, relevance=False)

    def _on_regions_toggle(self):
        self.seg_source_var.set("msc" if self.show_regions_var.get() else "none")
        self._on_seg_source_change()

    def _rerun_selection(self):
        """A new region commit invalidates every frozen region prediction."""
        self._pred.clear()
        self._cm_cell = None
        self._refresh_region_modes()
        super()._rerun_selection()
        self._refresh_confusion()

    def _build_live_panel(self, parent):
        # persistence: region identity depends on it, so it stays adjustable;
        # commit is the Rerun button exactly as in the viewer. These controls
        # live directly under the image controls rather than in a subpanel.
        row = ttk.Frame(parent); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Persistence %:").pack(side="left")
        self.persist_entry = ttk.Entry(row, textvariable=self.persist_live_var, width=8)
        self.persist_entry.pack(side="left", padx=4)
        self.persist_entry.bind("<Return>", self._on_persistence_change)
        self.persist_entry.bind("<FocusOut>", self._on_persistence_change)
        self.persist_value_label = ttk.Label(row, text="")
        self.persist_value_label.pack(side="left", padx=4)

        # No per-slice selection / pixel trim / connectivity: the labeler works
        # on the full pre-filter MSC labeling (the guarded _rebuild_*_cards
        # no-op without their frames).
        self.rerun_btn = ttk.Button(row, text="update region simplification", state="disabled",
                                    command=self._rerun_selection)
        self.rerun_btn.pack(side="left", padx=4)

    def _feature_schema_now(self):
        try:
            return config_io.feature_schema(self._params_json())
        except Exception:
            return None

    # -- model <-> profile compatibility --------------------------------- #
    def _expected_feature_names(self):
        """The feature set the ACTIVE profile's statistics produce -- a pure
        schema call, no priming. None when the compiled extension is absent
        (the fallback field list would falsely block)."""
        try:
            from msseg import mscoupon as ext
            if not getattr(ext, "_HAVE_EXTENSION", False):
                return None
        except ImportError:
            return None
        fields = config_io.query_fields(self._params_json())
        return [n for n in fields if n not in _NON_FEATURE_FIELDS]

    # -- model provenance ------------------------------------------------ #
    def _stats_brief(self, stats):
        """One line for a statistics block's channels, e.g. `base,blur(0.7,1.5)
        x4` -- the two numbers (channels, reductions) that decide how wide the
        model's per-feature row is."""
        try:
            doc = config_io.statistics_from_json(stats or {})
        except Exception:
            return "?"
        parts = []
        for c in doc["channels"]:
            sig = c.get("sigmas") or []
            parts.append(c["kind"] + (("(" + ",".join(f"{s:g}" for s in sig) + ")")
                                      if sig else ""))
        return ",".join(parts) + f" x{len(doc['reductions'])}"

    # -- the stage strip's two app boxes (panels/stages.py) ---------------- #
    def _measure_key(self):
        from . import fingerprints
        try:
            return fingerprints.measure_fingerprint_of(self._params_json(cores=1))
        except Exception:
            return None

    def _stage_field(self, key):
        from . import fingerprints
        if self.engine.run_active:
            return ("busy", "Priming the stack...")
        idx = self.catalogue.index_of(key)
        if not self.primed or idx is None or idx[0] >= len(self.primed):
            return ("none", "Not primed -- Run to prime this sequence.")
        if (getattr(self, "_primed_fingerprint", None) is not None
                and fingerprints.field_fingerprint_of(self._params_json(cores=1))
                != self._primed_fingerprint):
            return ("stale", "The topology chain or MSC changed since the prime -- "
                             "Run to re-prime (what is shown is a preview).")
        return ("ok", "Primed: the MSC and its regions are live.")

    def _stage_measure(self, key):
        idx = self.catalogue.index_of(key)
        if not self.primed or idx is None or idx[0] >= len(self.primed):
            return ("none", "No statistics: not primed.")
        si, li = idx
        cur = self._current()
        if cur is not None and cur == (si, li) and self._is_current_busy():
            return ("busy", "Re-measuring..." if self._current_measure_stale()
                    else "Computing regions and statistics...")
        if self._measurement_moved():
            return ("stale", "Features edited -- re-measuring.")
        if self._selection_dirty:
            return ("stale", "Persistence or selection changed -- click Rerun selection.")
        if self.engine.record(si, li) is None:
            return ("none", "Not computed at these parameters yet.")
        return ("ok", "Regions and statistics current for this workflow's Features.")

    def _profile_from_model(self, path, statistics):
        """Append a profile that keeps the active one's filters/MSC/selection
        but MEASURES what the model was trained on, and activate it.

        A new profile rather than an edit of the active one: profiles have no
        undo stack. The new profile shares the active one's FIELD, so the
        switch keeps the primed stack and the slices re-measure under the
        model's statistics as they are read -- no Run."""
        self._snapshot_active_profile()
        base = {}
        if 0 <= self.active_profile_idx < len(self.profiles):
            base = json.loads(json.dumps(self.profiles[self.active_profile_idx]))
        stats = json.loads(json.dumps(statistics))
        base["statistics"] = stats
        # profile_from_json takes the radius from msc, not from the statistics
        # block, so carry the model's across or the round-trip would reset it.
        base.setdefault("msc", {})["extremum_sample_radius"] = max(
            0, int(stats.get("extremum_sample_radius") or 0))
        base["name"] = session.dedupe_profile_name(
            f"from {os.path.basename(path)}", [p["name"] for p in self.profiles])
        notes = []
        # The round-trip re-validates feature_filters against the field
        # universe the NEW statistics produce, dropping the stale ones.
        self.profiles.append(session.profile_from_json(base, notes))
        for msg in notes:
            log(msg)
        self._switch_profile(len(self.profiles) - 1)

    def _make_training_set(self):
        """Pick a folder; write `train/` (the raw input TIFFs) and `labels/`
        (per-pixel class-id masks from the classifier, with user annotations
        winning where they disagree) -- the raw material for a UNet-style
        image model later."""
        if not self.primed:
            self.status_var.set("Run first - the masks need computed regions.")
            return
        if self.engine.asm_running or self.engine.asm_pending is not None:
            self.status_var.set("Busy computing - try again in a moment.")
            return
        out = filedialog.askdirectory(title="Choose a folder for the training set")
        if not out:
            return
        written, skipped = self._write_training_set(out)
        msg = (f"Training set: {written} image(s) -> {os.path.join(out, 'train')} "
               f"+ masks -> {os.path.join(out, 'labels')}")
        if skipped:
            msg += f" ({skipped} slice(s) skipped - no labels or predictions)"
        self.status_var.set(msg)

    def _write_training_set(self, out_dir):
        import shutil
        import numpy as np
        from PIL import Image
        train_dir = os.path.join(out_dir, "train")
        labels_dir = os.path.join(out_dir, "labels")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        # Gate the model ONCE: under a mismatched profile the masks fall back
        # to annotations alone rather than silently wrong predictions.
        use_model = (self._clf is not None and
                     self._check_model_compat(self._clf_names,
                                              "training-set export") is None)
        written = skipped = 0
        try:
            for si, p in enumerate(self.primed):
                for li in range(len(p["pipes"])):
                    self._compute_badge(f"Exporting {si}:{li}")
                    rec = self._ensure_slice_record(si, li)
                    key = self._slice_key(si, li)
                    if rec is None or rec.get("labels") is None or key is None:
                        skipped += 1
                        continue
                    labels = rec["labels"]
                    K = int(labels.max()) + 1 if labels.size else 1
                    combined = np.zeros(max(K, 1), np.uint8)
                    if use_model:
                        pred = self._predict_slice(si, li, rec, np)
                        if pred is not None:
                            combined = pred.copy()
                    user = resolve_slice(self._gestures_for_key(key), labels, np)
                    combined[user > 0] = user[user > 0]    # annotations win
                    if not combined.any():
                        skipped += 1                       # nothing to teach
                        continue
                    mask = np.zeros(labels.shape, np.uint8)
                    valid = labels >= 0
                    mask[valid] = combined[labels[valid]]
                    src = p["files"][li]
                    folder = "seq"
                    if si < len(self.subsequences):
                        folder = self.subsequences[si].get("folder") or "seq"
                    safe = folder.replace("/", "_").replace("\\", "_")
                    base = os.path.basename(src)
                    stem, _ext = os.path.splitext(base)
                    try:
                        shutil.copy2(src, os.path.join(train_dir, f"{safe}__{base}"))
                    except OSError as exc:
                        log(f"training set: could not copy {src}: {exc}")
                        skipped += 1
                        continue
                    Image.fromarray(mask).save(
                        os.path.join(labels_dir, f"{safe}__{stem}.tiff"))
                    written += 1
        finally:
            self._clear_compute_badge()
        return written, skipped

    def _export_csv(self):
        """Resolved region -> class table: one row per living MSC region of
        every slice whose labels are cached at the current commit (class 0 =
        unlabeled, so the file carries negatives for training too)."""
        if not self.primed:
            self.status_var.set("Nothing primed - Run first, then export.")
            return
        path = filedialog.asksaveasfilename(title="Export resolved labels as CSV",
                                            defaultextension=".csv",
                                            initialfile="labels.csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        covered, skipped = self._write_labels_csv(path)
        msg = f"Wrote {path}: {covered} slice(s)"
        if skipped:
            msg += (f", {skipped} skipped (labels not computed at this commit - "
                    "browse them or Rerun first)")
        self.status_var.set(msg)

    def _write_labels_csv(self, path):
        import numpy as np
        covered = skipped = 0
        # Every statistics column the spec produced rides along (the feature
        # row IS the region's future design vector). The schema is spec-driven
        # and identical across slices of a run; the first cached table fixes
        # the column order, later tables are aligned by name.
        stat_names = None
        rows = []           # (slice_key, region_id, class, predicted, table, row_idx)
        for si, p in enumerate(self.primed):
            for li in range(len(p["pipes"])):
                rec = self.regions.record(self.catalogue.key_of(si, li))
                key = self._slice_key(si, li)
                if rec is None or rec.get("labels") is None or key is None:
                    skipped += 1
                    continue
                labels = rec["labels"]
                rc = resolve_slice(self._gestures_for_key(key), labels, np)
                # Classifier predictions ride along when they exist for this
                # commit; blank otherwise (a prediction of class 0 never
                # happens -- the model only knows labeled classes).
                pr = self._pred.get(self.catalogue.key_of(si, li))
                pred = pr[1] if pr is not None and pr[0] == rec.get("commit") else None
                table = rec.get("stats")
                if getattr(table, "values", None) is None:
                    table = None
                row_of = {}
                if table is not None:
                    if stat_names is None:
                        stat_names = [n for n in table.names if n != "feature_id"]
                    fids = table.column("feature_id")
                    if fids is not None:
                        row_of = {int(v): r for r, v in enumerate(fids)}
                ids = np.unique(labels)
                for i in ids[ids >= 0]:
                    p_val = ("" if pred is None or i >= len(pred)
                             else str(int(pred[i])))
                    rows.append((key, int(i), int(rc[i]), p_val, table,
                                 row_of.get(int(i))))
                covered += 1
        lines = ["slice,region_id,class,predicted" +
                 ("," + ",".join(stat_names) if stat_names else "")]
        for key, rid, cls, p_val, table, r in rows:
            line = f"{key},{rid},{cls},{p_val}"
            if stat_names:
                vals = []
                for n in stat_names:
                    col = table.column(n) if (table is not None and r is not None) else None
                    vals.append("" if col is None else f"{float(col[r]):.10g}")
                line += "," + ",".join(vals)
            lines.append(line)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(lines) + "\n")
        return covered, skipped

    def _write_configs(self, out_dir):
        paths = super()._write_configs(out_dir)
        path = os.path.join(out_dir, "annotations.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.store.to_json(), f, indent=2)
        paths.append(path)
        return paths

    # ------------------------------------------------------------------ #
    # AnnotationShell hooks: the coupon profile and the viewer's region layer
    # ------------------------------------------------------------------ #
    def _workflow_summary(self, profile):
        return session.profile_summary(profile)

    def _workflow_hint_tooltip(self):
        codes = ", ".join(f"{v}={k}" for k, v in session.OP_CODES.items())
        return ("The active compute profile. Line 1: the topology field the MSC "
                "runs on and the MSC setting (manifold, persistence; mf = merge "
                "forest). Line 2: the base chain the statistics are measured on "
                "and channels × reductions. Click to open the Processing tab."
                + chr(10) + "Codes: " + codes)

    def _set_region_layer_visible(self, on):
        """One toggle instead of the viewer's five seg sources: the labeler
        works on the full MSC labeling ("msc"), shown faintly under the class
        layer. seg_source stays a valid viewer value so _needed_level() is
        always "slice" and every inherited path keeps working; the mask is
        never on."""
        if not hasattr(self, "mask_var"):
            self.mask_var = tk.BooleanVar(value=False)
        self.seg_source_var.set("msc" if on else "none")

    def _region_layer_visible(self):
        return self.seg_source_var.get() == "msc"




# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def main():
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        return _selftest()
    initial = args[0] if args else None
    root = tk.Tk()
    LabelerApp(root, initial)
    root.mainloop()


class _FakeEvent:
    """Enough of a Tk event for the canvas's click-vs-drag arithmetic and
    the drawing tool's modifier check."""

    def __init__(self, x, y, state=0):
        self.x, self.y = x, y
        self.x_root, self.y_root = x, y
        self.state = state


def _selftest():
    """Exercise the labeler's wiring headlessly (no engine/render needed);
    the pure geometry/ordering/LUT math is covered by tests/test_labeling.py."""
    import tempfile
    import numpy as np
    from msseg.labeler import labeling

    root = tk.Tk()
    root.withdraw()
    # autosave=False: this builds a real app, and a test must not overwrite the
    # user's saved session.
    app = LabelerApp(root, autosave=False)

    stats0 = config_io.statistics_from_json(app.profiles[0]["statistics"])
    assert not stats0["relevance"], "new labeler profiles default relevance off"
    assert app.rerun_btn.cget("text") == "update region simplification"
    assert app.show_pred_var.get(), "classification rendering is on by default"

    # Window layout: three columns -- data navigation, `app.right` (the canvas
    # and its controls), and a notebook of everything that is edited. The left
    # column keeps the profile PICKER, the session and Run; the profile is
    # EDITED on the Processing tab as one column of collapsible groups; the
    # annotations and the classifier are the Annotation tab; the model kind
    # lives on the Model tab and is gone from the classifier panel.
    def _under(w, ancestor):
        while w is not None:
            if w is ancestor:
                return True
            w = getattr(w, "master", None)
        return False

    tabs = [str(app.center.tab(t, "text")) for t in app.center.tabs()]
    assert tabs == list(_CENTER_TABS) == ["Processing", "Features", "Annotation",
                                          "Model", "Analysis"], tabs
    assert [str(p) for p in app.paned.panes()] \
        == [str(app.left_pane), str(app.right), str(app.center)], "data | view | tabs"
    assert _under(app.viewer.canvas, app.right), "the canvas is the middle column"
    assert _under(app.label_pane, app.annot_tab), "the annotations are a tab now"
    assert app._center_tab_name() == "Processing"
    # Only the viewer pane has a weight, so a resize goes to the picture.
    weights = [int(app.paned.pane(p, weight=None)) for p in app.paned.panes()]
    assert weights == [0, 1, 0], weights
    # The sashes are persisted as fractions (restored with the session doc
    # below), and F9 folds the tab column away. A withdrawn root lays nothing
    # out, so the fractions are the WANTED ones throughout -- what is checked
    # here is the bookkeeping, not the pixels.
    assert app._view_state()["panes"] == list(app._DEFAULT_PANES)
    app._apply_pane_fractions([0.2, 0.5])
    app.root.update_idletasks()
    assert app._pane_fractions() == [0.2, 0.5]
    app._apply_pane_fractions([0.5, 0.2])                  # sashes cannot cross
    assert app._pane_fractions() == [0.2, 0.5], "an out-of-order pair is ignored"
    app._apply_pane_fractions([9.0, 9.0])
    assert app._pane_fractions() == [0.2, 0.5], "an absurd fraction is ignored"
    assert app._toggle_center_tabs() == "break" and app._panes_collapsed
    app._toggle_center_tabs(); app.root.update_idletasks()
    assert not app._panes_collapsed
    assert app._pane_fractions() == [0.2, 0.5], "F9 remembers where they were"
    # Processing is ONE column of collapsible groups, and which are folded
    # rides the session. The measurement -- the base channel, the statistics
    # and the ext sample radius -- is on the Features tab (still folding
    # groups under the same keys).
    for w in (app.filters_frame, app.msc_frame, app.profile_load_btn):
        assert _under(w, app.processing_tab), w
    for w in (app.filters_frame, app.msc_frame):
        assert _under(w, app.proc_col), w
    for w in (app.stats_frame, app.base_frame):
        assert _under(w, app.feat_col) and _under(w, app.features_tab), w
        assert not _under(w, app.processing_tab), w
    radius_entries = [w for w in app.stats_frame.winfo_children()
                      for w in w.winfo_children()
                      if isinstance(w, ttk.Entry)
                      and str(w.cget("textvariable")) == str(app.ext_radius_var)]
    assert len(radius_entries) == 1, "the ext sample radius is a statistics field"
    assert set(app._proc_groups) >= {"filters", "base", "msc", "stats"}
    app._refresh_hints()
    assert app.features_hint_var.get() == "stats: base→1ch×4", app.features_hint_var.get()
    app._show_center_tab("Features")
    assert app._center_tab_name() == "Features"
    app._show_center_tab("Processing")
    assert app._proc_open_state() == {}, "everything starts open"
    app._proc_groups["msc"].toggle()
    assert not app._proc_groups["msc"].is_open()
    assert app._view_state()["proc_open"] == {"msc": False}
    assert app._proc_groups["msc"].body.winfo_manager() == "", "the body is folded"
    app._apply_proc_open({"msc": True})
    assert app._proc_groups["msc"].is_open() and app._proc_open_state() == {}
    assert _under(app.profile_combo, app.left) and _under(app.run_btn, app.left_pane)
    assert not _under(app.profile_load_btn, app.left), "profile tools moved"
    # Left pane: Run pinned at the bottom (packed first, side=bottom), the
    # session's three lists as the panes of a vertical paned window.
    assert _under(app.run_frame, app.left_bottom) and not _under(app.run_btn, app.left)
    assert app.left_bottom.pack_info()["side"] == "bottom"
    assert app.left_pane.pack_slaves()[0] is app.left_bottom, "Run claims its space first"
    assert len(app.session_paned.panes()) == 3
    assert [str(app._session_groups[n]) for n in ("folders", "files", "sequences")] == \
        [str(p) for p in app.session_paned.panes()]
    assert _under(app.folder_list, app._session_groups["folders"])
    assert _under(app.file_list, app._session_groups["files"])
    assert _under(app.subseq_list, app._session_groups["sequences"])
    for btn, lst in ((app.folder_btn_row, app.folder_list), (app.make_seq_btn, app.file_list),
                     (app.seq_btn_row, app.subseq_list)):
        slaves = lst.master.master.pack_slaves()
        assert btn.pack_info()["side"] == "bottom"
        assert slaves.index(btn) < slaves.index(lst.master), "buttons packed before the list"
        assert lst.master.pack_info()["expand"] in (1, "1", True)
    assert _under(app.model_kind_combo, app.model_tab)
    assert not _under(app.model_kind_combo, app.label_pane)
    ml_rows = [c for row in app.confusion_holder.master.winfo_children()
               for c in row.winfo_children()]
    assert not any(isinstance(c, ttk.Combobox) for c in ml_rows), \
        "no kind combobox left in the classifier panel"
    # The notebook never binds plain Tab, so the overlay hotkey still fires
    # (and still consumes the key) from every tab.
    assert "<Key-Tab>" not in app.center.bind_class("TNotebook")
    assert app._on_tab_toggle() == "break"
    assert not app.show_overlay_var.get()
    app._on_tab_toggle()
    assert app.show_overlay_var.get()
    # The architecture readout follows the kind, formatted from the constants
    # _make_model builds with.
    for kind in _MODEL_KINDS:
        app.model_kind_var.set(kind)
        assert app.model_arch_var.get() == app._model_description_now(kind), kind
    assert str(_FOREST_TREES) in _model_description("random forest")
    assert str(_MLP_HIDDEN) in _model_description("dense FC")
    assert str(_DENSE_TOP_N["dense-top-16"]) in _model_description("dense-top-16")
    app.model_kind_var.set("dense FC")

    # Toolbar hints: the selected workflow as a compact chain and the active
    # model, each a click away from its tab.
    app._refresh_hints()
    assert app.workflow_hint_var.get().splitlines() == \
        ["topo field: base→msc(asc, 10%, mf)", "stats: base→1ch×4"], app.workflow_hint_var.get()
    assert app.model_hint_var.get() == "model: dense FC · not trained"
    app._on_filter_op_change(0, "blur"); app.filter_cards[0]["params"]["sigma"] = 1.5
    app._on_filter_op_change(1, "edges"); app.filter_cards[1]["params"]["sigma"] = 0.7
    app._refresh_hints()
    assert "base→b(1.5)→e(0.7)" in app.workflow_hint_var.get(), app.workflow_hint_var.get()
    app.filter_cards = [app._new_filter_card()]
    app._rebuild_filter_cards()
    app._refresh_hints()
    assert app.workflow_hint_var.get().splitlines()[0] == "topo field: base→msc(asc, 10%, mf)"
    app._show_center_tab("Processing")
    assert app._center_tab_name() == "Processing"
    app._show_center_tab("Model")
    assert app._center_tab_name() == "Model"
    app._show_center_tab("nope")
    assert app._center_tab_name() == "Model", "an unknown name is ignored"
    app._show_center_tab("View")
    assert app._center_tab_name() == "Model", "the retired View name is ignored"
    app.center.select(app.processing_tab)
    app._clf, app._clf_names, app._clf_kind = object(), ["a", "b"], "random forest"
    assert app._model_hint_text().startswith("model: random forest · 2 feats")
    app._clf = None
    app._refresh_hints()
    assert app.model_hint_var.get() == "model: dense FC · not trained"
    # Placement: the workflow hint heads the Run section; the model hint
    # heads the classifier section, above Train/Classify.
    assert app.workflow_hint.master is app.run_frame
    assert app.run_frame.pack_slaves()[0] is app.workflow_hint
    ml = app.confusion_holder.master
    # What the regions are measured by (-> Features), then the model (-> Model).
    assert ml.pack_slaves()[0] is app.features_hint
    assert app.model_hint.master is ml and ml.pack_slaves()[1] is app.model_hint
    train_row = ml.pack_slaves()[2]
    assert any(isinstance(w, ttk.Button) and str(w.cget("text")).startswith("Train")
               for w in train_row.winfo_children()), "Train/Classify right under the hint"

    # `app.right` is still the frame the viewer area packs into, so the scan
    # below reads unchanged.
    right_rows = list(app.right.winfo_children())
    assert not any(isinstance(w, ttk.LabelFrame)
                   and str(w.cget("text")) == "Live parameters"
                   for w in right_rows), "live controls must not have a subpanel"
    direct_labels = [child for row in right_rows
                     for child in row.winfo_children()
                     if isinstance(child, ttk.Label)]
    assert any(str(w.cget("text")) == "Overlay alpha:" for w in direct_labels)
    direct_checks = [child for row in right_rows
                     for child in row.winfo_children()
                     if isinstance(child, ttk.Checkbutton)]
    assert any(str(w.cget("text")) == "Show Classification"
               for w in direct_checks)
    assert any(str(w.cget("text")) == "Show GT" for w in direct_checks)
    overlay_children = list(app.overlay_master_check.master.winfo_children())
    master_idx = overlay_children.index(app.overlay_master_check)
    assert isinstance(overlay_children[master_idx + 1], ttk.Separator), \
        "a vertical separator must follow the overlay master"
    assert overlay_children[master_idx + 2] is app.show_regions_check
    # One brightness window per channel: the sliders edit the channel on
    # screen, a channel with nothing on the canvas yet is not cached, and F
    # flips base <-> filtered (nothing derived remembered yet).
    assert app.viewer.source is None
    assert app._window_for("edges_s1") == (0.0, 1.0)
    assert "edges_s1" not in app._channel_windows, "no source: nothing to measure, nothing cached"
    assert app._window_channel == "edges_s1"
    app.vmin_var.set(0.2); app.vmax_var.set(0.8)
    app._on_window_change()
    assert app._channel_windows["edges_s1"] == (0.2, 0.8), "a moved slider is kept for that channel"
    assert app._window_for("edges_s1") == (0.2, 0.8)
    assert app._window_for("base") == (0.0, 1.0) and app.vmin_var.get() == 0.0, \
        "the sliders follow the channel"
    assert app.background_var.get() == "base"
    assert app._on_swap_key() is None and app.background_var.get() == "filtered"
    assert "(F: base)" in app.status_var.get(), app.status_var.get()
    app._on_swap_key()
    assert app.background_var.get() == "base" and app._swap_channel == "filtered"
    app._channel_windows.clear(); app._window_channel = None; app._swap_channel = None
    app.vmin_var.set(0.0); app.vmax_var.set(1.0)

    # The labeler never needs more than the per-slice tier.
    assert app._needed_level() == "slice", "regions view must stay on the slice tier"
    app.show_regions_var.set(False); app._on_regions_toggle()
    assert app._needed_level() == "slice"
    app.show_regions_var.set(True); app._on_regions_toggle()

    # Fake one primed slice: 4 blocks with SPARSE living ids, -1 border. The
    # session has one (nonexistent) folder; slice identity is folder-qualified.
    lab = np.full((20, 20), -1, np.int32)
    lab[2:10, 2:10] = 0; lab[2:10, 10:18] = 2
    lab[10:18, 2:10] = 5; lab[10:18, 10:18] = 9
    data_dir = r"C:\labdata"
    files = [os.path.join(data_dir, "s0.tiff")]
    app.folders = [{"path": data_dir, "name": "data"}]
    app.active_folder_idx = 0
    app._refresh_folder_list()
    app.subsequences = [{"name": "seq1", "folder": "data", "files": files}]
    zeros = np.zeros((20, 20), np.float32)
    app.primed = [{"files": files, "base": [zeros], "filtered": [zeros],
                   "pipes": [None], "normalizers": [[]]}]
    app._rebuild_flat_slices()
    from .common import FeatureTable
    table = FeatureTable(["feature_id", "area", "mean_base"],
                         np.array([[0.0, 64.0, 1.5], [2.0, 80.0, 2.5],
                                   [5.0, 96.0, 3.5], [9.0, 112.0, 4.5]]))
    rec = {"commit": app._commit_id, "labels": lab, "stats": table,
           "kept": set(), "cc": None, "n_feat": 4}
    app._slices[(0, 0)] = rec

    # Gestures commit through the same path the DrawController uses.
    app.active_class_var.set(1)
    app._commit_interaction("box", [(3.0, 3.0), (16.0, 16.0)])      # all 4 -> 1
    app.active_class_var.set(2)
    # A commit must DIFF the panels, not rebuild them: destroying and
    # recreating the class frames, lists, swatches and confusion cells is what
    # flashed the whole right pane black on every gesture release (and, through
    # the paned window's re-layout, the panes beside it).
    keep = (dict(app._class_panels), dict(app._class_swatches),
            dict(app._class_lists), dict(app._cm_cells["all"]),
            dict(app._row_widgets))
    app._commit_interaction("squiggle", [(3.0, 5.0), (15.0, 5.0)])  # {0,2} -> 2
    assert [it.uid for it in app.store.interactions] == [1, 2]
    assert all(app._class_panels[k] is w for k, w in keep[0].items())
    assert all(app._class_swatches[k] is w for k, w in keep[1].items())
    assert all(app._class_lists[k] is w for k, w in keep[2].items())
    assert all(app._cm_cells["all"][c] is w for c, w in keep[3].items()), \
        "the confusion grid must not be rebuilt for a count change"
    assert all(app._row_widgets[u] is r for u, r in keep[4].items()), \
        "existing interaction rows survive a commit"
    assert set(app._row_widgets) - set(keep[4]) == {2}, "exactly one row added"
    assert [w for w in app._class_lists[2].inner.pack_slaves()] == \
        [app._row_widgets[2]["frame"]]
    # ...but a structural change (more classes, a new colour) still rebuilds.
    was = app._class_color_hex(1)
    app.store.set_color(1, "#0b0b0b")
    app._rebuild_class_panels()
    assert app._class_panels[1] is not keep[0][1], "a colour change rebuilds"
    app.store.set_color(1, was)          # restored directly: no history entry
    app._rebuild_class_panels()
    assert app.store.interactions[0].slice_key == "data/s0.tiff", \
        "slice identity is folder-qualified"
    # The sequence tree's columns track priming + per-slice annotations, and
    # a count change updates the cells IN PLACE -- a delete-and-reinsert
    # repaints the left pane and drops the selection on every commit.
    app._refresh_subseq_list()
    assert tuple(app.subseq_list.item("q0:0", "values")) == ("Y", "2")
    assert tuple(app.subseq_list.item("q0", "values")) == ("Y", "2")
    app.subseq_list.selection_set("q0:0")
    app._commit_interaction("squiggle", [(4.0, 6.0)])
    assert tuple(app.subseq_list.item("q0:0", "values")) == ("Y", "3")
    assert app.subseq_list.selection() == ("q0:0",), "selection survives a commit"
    app._delete_interaction(app.store.interactions[-1].uid)
    assert tuple(app.subseq_list.item("q0:0", "values")) == ("Y", "2")
    # A sequence whose files changed still forces the full rebuild.
    app.subsequences[0]["files"] = [os.path.join(data_dir, "s9.tiff")]
    assert not app._update_subseq_values()
    app.subsequences[0]["files"] = files
    app._refresh_subseq_list()

    # The class panels list ON-SLICE interactions only (they swap with the
    # slice); the titles carry ALL-slice annot/region totals.
    other = app.store.add("squiggle", [(0.0, 0.0)], 1, "data/other.tiff")
    app._rebuild_class_panels()
    assert len(app._visible_interactions()) == 2, "on-slice interactions only"
    t1 = str(app._class_title_labels[1].cget("text"))
    assert t1 == "1 · 2a · 2r", t1            # box + off-slice; {5,9}
    t2 = str(app._class_title_labels[2].cget("text"))
    assert t2 == "2 · 1a · 2r", t2            # squiggle; {0,2}
    app.store.remove(other.uid)
    app._rebuild_class_panels()

    # User-picked class colors: reach the LUT (rev bump rebuilds the cache),
    # ride the store, and undo like any other edit.
    app._push_history()
    app.store.set_color(1, "#123456")
    app._rebuild_class_panels()
    assert app._class_color_hex(1) == "#123456"
    lut_c = app._class_lut_for(0, 0, rec, np)
    assert tuple(lut_c[5]) == (0x12, 0x34, 0x56, 255), "picked color in the LUT"
    assert app.store.to_json()["classes"][0]["color"] == "#123456"
    app._undo()
    assert app._class_color_hex(1) != "#123456", "color change is undoable"

    # The swatch is the arm control (the per-class "draw" radiobutton is gone),
    # and the trace rings exactly the armed one.
    for k, frame in app._class_panels.items():
        assert not [w for w in frame.winfo_children()
                    if w.winfo_class() == "Radiobutton"], \
            f"class {k} still has a draw radiobutton"
    app._class_swatches[2].event_generate("<ButtonRelease-1>")
    assert app.active_class_var.get() == 2, "swatch left-click arms the class"
    armed = [k for k, w in app._class_swatches.items()
             if str(w.cget("relief")) == "sunken"]
    assert armed == [2], armed
    app.active_class_var.set(1)          # the trace, not the swatch, repaints
    armed = [k for k, w in app._class_swatches.items()
             if str(w.cget("relief")) == "sunken"]
    assert armed == [1], armed
    app._rebuild_class_panels()          # a rebuild restores the ring
    assert str(app._class_swatches[1].cget("relief")) == "sunken"

    # Resolution + LUT: later interaction painted over the earlier one.
    lut = app._class_lut_for(0, 0, rec, np)
    assert lut is not None and lut.shape == (10, 4)
    assert tuple(lut[0]) == labeling.CLASS_COLORS[2]     # repainted by squiggle
    assert tuple(lut[2]) == labeling.CLASS_COLORS[2]
    assert tuple(lut[5]) == labeling.CLASS_COLORS[1]     # box only
    assert tuple(lut[9]) == labeling.CLASS_COLORS[1]
    assert lut[1, 3] == 0, "id 1 is not a living id -> transparent"

    # Memoization keys on (commit, store.rev).
    assert app._class_lut_for(0, 0, rec, np) is lut, "cache hit expected"
    app._move_interaction(2, 2)          # same class: no rev bump, still cached
    assert app._class_lut_for(0, 0, rec, np) is lut
    app.store.set_class(1, 2)            # mutation -> rev bump -> rebuilt
    lut2 = app._class_lut_for(0, 0, rec, np)
    assert lut2 is not lut and tuple(lut2[9]) == labeling.CLASS_COLORS[2]
    app.engine.commit_selection()        # Rerun path: fresh commit, fresh labels
    rec2 = dict(rec, commit=app._commit_id)
    lut3 = app._class_lut_for(0, 0, rec2, np)
    assert lut3 is not lut2, "a new commit must re-resolve"

    # The overlay stack: dimmed region layer under the opaque class layer.
    from msseg.viz import min_colors
    ovs = app._seg_overlays(0, 0, rec2, None, np, min_colors)
    assert len(ovs) == 2
    assert int(ovs[0]["lut"][:, 3].max()) <= _REGION_ALPHA
    assert int(ovs[1]["lut"][:, 3].max()) == 255
    app.show_gt_var.set(False)
    assert len(app._seg_overlays(0, 0, rec2, None, np, min_colors)) == 1, \
        "Show GT controls the drawn-label layer independently"
    app.show_gt_var.set(True)
    # The master overlay switch (Tab) blanks the whole stack.
    app._on_tab_toggle()
    assert not app.show_overlay_var.get()
    assert all(str(w.cget("state")) == "disabled"
               for w, _state in app._overlay_dependents)
    assert app._seg_overlays(0, 0, rec2, None, np, min_colors) == []
    app._on_tab_toggle()
    assert app.show_overlay_var.get()
    assert str(app.region_mode_combo.cget("state")) == "readonly"
    assert all(str(w.cget("state")) == state
               for w, state in app._overlay_dependents)

    # Class-count change clamps orphans; arm state resets when it vanishes.
    app.active_class_var.set(2)
    app.n_classes_var.set(2)
    app._on_n_classes_change()
    assert all(it.class_id == 1 for it in app.store.interactions)
    assert app.active_class_var.get() == 0

    # Session round-trip: the store rides the v2 session doc, and so does
    # the selected center tab (by name).
    app.center.select(app.model_tab)
    kind_pick = app.model_kind_var.get()
    app.model_kind_var.set(_CUSTOM_EDGE_KIND)
    sdoc = app._session_doc()
    assert sdoc["view"]["center_tab"] == "Model"
    assert sdoc["view"]["panes"] == [0.2, 0.5], sdoc["view"]["panes"]
    # The Model-tab settings and the gestures are the TASK's (session v3):
    # one task here, and nothing duplicated at the top level.
    assert sdoc["session_version"] == 4 and len(sdoc["tasks"]) == 1
    assert sdoc["active_task"] == sdoc["tasks"][0]["uid"] == app._task.uid
    assert "model_kind" not in sdoc["view"], "the picked kind is the task's, not the window's"
    assert sdoc["tasks"][0]["view"]["model_kind"] == _CUSTOM_EDGE_KIND, \
        "the picked kind rides the task's view"
    app.model_kind_var.set("dense FC")
    app.center.select(app.processing_tab)
    assert "labels" not in sdoc and "annotations" not in sdoc and "models" not in sdoc, \
        "the gesture geometry is the task's 'annotations' now"
    assert sdoc["tasks"][0]["annotations"]["n_classes"] == 2
    assert len(sdoc["tasks"][0]["annotations"]["interactions"]) == 2
    assert sdoc["tasks"][0]["workflow"] == app.profiles[app.active_profile_idx]["name"]
    assert sdoc["sequences"][0]["folder"] == "data"
    # A labeler document whose tasks do not say their kind is refused whole.
    old = json.loads(json.dumps(sdoc))
    for _t in old["tasks"]:
        _t.pop("kind")
    n_before = len(app.store.interactions)
    notes_old = app._apply_session_doc(old, "older")
    assert notes_old and "older labeler" in notes_old[-1], notes_old
    assert len(app.store.interactions) == n_before, "nothing of a refused document applies"
    app.store = LabelStore()             # clobber
    app._apply_session_doc(sdoc, "test")
    assert app._center_tab_name() == "Model", "the center tab restores by name"
    app.root.update_idletasks()
    assert app._pane_fractions() == [0.2, 0.5], "the sashes restore too"
    assert app.model_kind_var.get() == _CUSTOM_EDGE_KIND, "the picked kind restores"
    bad = json.loads(json.dumps(sdoc))
    bad["tasks"][0]["view"]["model_kind"] = "no such kind"
    app._apply_session_doc(bad, "test")
    assert app.model_kind_var.get() == _CUSTOM_EDGE_KIND, "an unknown kind is ignored"
    sdoc["tasks"][0]["view"]["model_kind"] = kind_pick   # later re-applies keep the default
    app.model_kind_var.set(kind_pick)
    app.center.select(app.processing_tab)
    assert len(app.store.interactions) == 2
    assert all(it.bound for it in app.store.interactions), \
        "rebind by folder-qualified key"

    # ...and a session written BEFORE the rename still restores: the key moved,
    # the document under it did not.
    # (A pre-task v2 document: no tasks, the store under the old key, the
    # Model-tab keys in the window view.)
    legacy = {k: v for k, v in sdoc.items() if k not in ("tasks", "active_task")}
    legacy["session_version"] = 2
    legacy["labels"] = sdoc["tasks"][0]["annotations"]
    legacy["view"] = dict(sdoc["view"], center_tab="bogus",      # unknown -> untouched
                          **sdoc["tasks"][0]["view"])
    app.store = LabelStore()
    app._apply_session_doc(legacy, "legacy key")
    assert app._center_tab_name() == "Processing", "an unknown tab name is ignored"
    assert len(app.store.interactions) == 2, \
        "a pre-rename session's 'labels' key still loads"
    assert len(app.tasks) == 1 and app._task.name == app.profiles[app.active_profile_idx]["name"], \
        "a v2 session reads as one task named after its active profile"
    assert app.model_kind_var.get() == kind_pick, "…with the window's Model-tab keys as its view"
    # ...and so does a session written before the View tab retired.
    app.center.select(app.model_tab)
    app._apply_session_doc(dict(legacy, view=dict(sdoc["view"], center_tab="View")),
                           "pre-split view")
    assert app._center_tab_name() == "Model", "the retired View name leaves the tab alone"
    app.center.select(app.processing_tab)

    # Tasks: several detectors over the same data, one active. Gestures, undo
    # history, class vocabulary and Model-tab settings are per task; the
    # session document carries every task; a task points at a profile by
    # name and follows it through renames and deletions.
    from unittest import mock as _mock
    t1 = app._task
    n1 = len(app.store.interactions)
    assert app.tasks == [t1] and tuple(app.task_tree.selection()) == (t1.uid,)
    assert t1.workflow == app.profiles[app.active_profile_idx]["name"]
    assert app._rename_class(1, "gland") and app.store.name(1) == "gland"
    assert app._class_title_labels[1].cget("text").startswith("1 gland ·")
    assert not app._rename_class(1, "gland"), "an unchanged name is a no-op"
    app.model_kind_var.set(_CUSTOM_EDGE_KIND)
    t2 = app._task_new("stroma")
    assert t2 is app._task and app.tasks == [t1, t2] and t2.name == "stroma"
    assert app.store is t2.store and app.store.interactions == [], "a new task starts empty"
    assert app.store.n_classes == t1.store.n_classes and not app.store.has_name(1)
    assert app._undo_stack == [] and app._pred == {}
    assert tuple(app.task_tree.selection()) == (t2.uid,)
    assert t1.view["model_kind"] == _CUSTOM_EDGE_KIND, "the outgoing task's Model tab was stashed"
    assert app.model_kind_var.get() == _CUSTOM_EDGE_KIND, "a new task inherits what is on screen"
    app.model_kind_var.set("dense FC")
    app._push_history()
    app.store.add("squiggle", [(3.0, 3.0), (4.0, 4.0)], 1, "data/s0.tiff")
    app._rebuild_class_panels()
    assert len(app.store.interactions) == 1 and len(t1.store.interactions) == n1
    assert app.task_tree.set(t2.uid, "annot") == "1" and app.task_tree.set(t1.uid, "annot") == str(n1)
    assert app._activate_task(t1) and app._task is t1
    assert len(app.store.interactions) == n1 and app.store.name(1) == "gland"
    assert app.model_kind_var.get() == _CUSTOM_EDGE_KIND, "each task's Model tab comes back"
    assert t2.view["model_kind"] == "dense FC"
    assert len(app._undo_stack) == 1, "task 1's history: the class rename, and not task 2's gesture"
    assert tuple(app.task_tree.selection()) == (t1.uid,)
    # Clicking a row activates; a rename repaints in place and dedupes.
    app.task_tree.selection_set(t2.uid)
    app.root.update()
    assert app._task is t2, "selecting a task row activates it"
    assert app._task_rename("stroma detector")
    assert app.task_tree.item(t2.uid, "text") == "stroma detector"
    assert app._task_rename(t1.name) and t2.name == f"{t1.name} (2)", "names stay unique"
    assert app._task_rename("stroma detector")
    # The session document: both tasks, the active one marked, both restored.
    sdoc_t = app._session_doc()
    assert sdoc_t["session_version"] == 4 and sdoc_t["active_task"] == t2.uid
    assert [t["name"] for t in sdoc_t["tasks"]] == [t1.name, "stroma detector"]
    assert sdoc_t["tasks"][0]["annotations"]["classes"][0]["name"] == "gland"
    assert sdoc_t["tasks"][0]["view"]["model_kind"] == _CUSTOM_EDGE_KIND
    app._apply_session_doc(sdoc_t, "tasks")
    assert [t.name for t in app.tasks] == [t1.name, "stroma detector"]
    assert app._task.uid == t2.uid and len(app.store.interactions) == 1
    assert len(app.tasks[0].store.interactions) == n1 and app.tasks[0].store.name(1) == "gland"
    assert app.model_kind_var.get() == "dense FC"
    assert all(it.bound for it in app.tasks[0].store.interactions), "every task's store rebinds"
    t1, t2 = app.tasks
    # A search in flight refuses a switch: its finish installs into the
    # active task, so a switch would land the winner in the wrong one.
    app._search = object()
    assert not app._activate_task(t1) and app._task is t2
    assert tuple(app.task_tree.selection()) == (t2.uid,)
    app._search = None
    # Profiles are the session's pool; a task points at one by name.
    app._profile_new()
    new_prof = app.profiles[app.active_profile_idx]["name"]
    assert t2.workflow == new_prof and t1.workflow != new_prof
    assert app.task_tree.set(t2.uid, "workflow") == new_prof
    with _mock.patch("tkinter.simpledialog.askstring", return_value="renamed wf"):
        app._profile_rename()
    assert t2.workflow == "renamed wf" and app.task_tree.set(t2.uid, "workflow") == "renamed wf"
    with _mock.patch.object(messagebox, "askyesno", return_value=True):
        app._profile_delete()
    assert "renamed wf" not in [p["name"] for p in app.profiles]
    assert t2.workflow == app.profiles[app.active_profile_idx]["name"], \
        "a task on a deleted profile moves to the survivor"
    assert app._activate_task(t1) and app.profiles[app.active_profile_idx]["name"] == t1.workflow, \
        "activating a task activates its workflow"
    # Delete: the neighbour takes over; the last task stays.
    assert app._activate_task(t2) and app._task_delete(confirm=True)
    assert app.tasks == [t1] and app._task is t1 and len(app.store.interactions) == n1
    assert not app._task_delete(confirm=True), "the last task stays"
    assert tuple(app.task_tree.selection()) == (t1.uid,)
    # Back to the single-task state the rest of the selftest expects.
    assert app._rename_class(1, "") and not app.store.has_name(1)
    app.model_kind_var.set(kind_pick)

    # Export writes annotations.json alongside the config(s).
    with tempfile.TemporaryDirectory() as td:
        paths = app._write_configs(td)
        assert os.path.basename(paths[-1]) == "annotations.json"
        with open(paths[-1], encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["version"] == 2 and len(doc["interactions"]) == 2

    # The round-trip's apply reset the engine; re-fake the primed slice
    # for the interactive-behavior checks below.
    app.primed = [{"files": files, "base": [zeros], "filtered": [zeros],
                   "pipes": [None], "normalizers": [[]]}]
    app._rebuild_flat_slices()
    rec3 = dict(rec, commit=app._commit_id)
    app._slices[(0, 0)] = rec3

    # Undo/redo: snapshots cover adds, class moves, and class-count changes;
    # a fresh edit wipes the redo branch.
    n0 = len(app.store.interactions)
    app.active_class_var.set(1)
    app._commit_interaction("squiggle", [(12.0, 12.0)])        # single-tap gesture
    assert len(app.store.interactions) == n0 + 1
    assert app.store.interactions[-1].points == [(12.0, 12.0)]
    app._undo()
    assert len(app.store.interactions) == n0, "undo removes the tap"
    app._redo()
    assert len(app.store.interactions) == n0 + 1, "redo restores it"
    app._undo()
    app._commit_interaction("squiggle", [(5.0, 12.0)])
    assert not app._redo_stack, "a new edit wipes the redo branch"
    app._undo()
    assert len(app.store.interactions) == n0
    app.n_classes_var.set(3)
    app._on_n_classes_change()
    uid0 = app.store.interactions[0].uid
    app._move_interaction(uid0, 2)
    assert app.store.get(uid0).class_id == 2
    app._undo()
    assert app.store.get(uid0).class_id == 1, "undo covers class moves"
    app._undo()
    assert app.store.n_classes == 2, "undo covers class-count changes"

    # Hover shows the gesture's geometry; a click recenters at constant zoom.
    if app.viewer is not None:
        app._show_interaction_geometry(uid0)
        assert app.viewer.canvas.find_withtag("ihover"), "hover geometry drawn"
        app._hide_interaction_geometry()
        assert not app.viewer.canvas.find_withtag("ihover")
        # A view change (zoom/pan) must re-project the geometry, not drop it.
        app._show_interaction_geometry(uid0)
        app._redraw_hover_geometry()
        assert app.viewer.canvas.find_withtag("ihover"), \
            "geometry survives a view change"
        app._hide_interaction_geometry()
        s0 = app.viewer.scale
        app._on_row_click(uid0)
        assert app.viewer.scale == s0, "recenter keeps the zoom level"
        it0 = app.store.get(uid0)
        cx = sum(x for x, _y in it0.points) / len(it0.points)
        w = max(app.viewer.canvas.winfo_width(), 1)
        assert abs(app.viewer.view_x - (cx - (w / 2) * s0)) < 1e-6
        # Hovering a LABELED pixel draws every gesture touching its region;
        # a background pixel clears again.
        app._on_hover(5, 5)                  # inside region 0
        assert app.viewer.canvas.find_withtag("ihover"), "pixel hover draws gestures"
        app._on_hover(0, 0)                  # -1 background
        assert not app.viewer.canvas.find_withtag("ihover")

        # The persistent annotation view: every on-slice gesture at once, on
        # its own tag so an overlay repaint (which drops "ihover") keeps it.
        assert not app.viewer.canvas.find_withtag("ipersist")
        app.show_annot_var.set(True)
        app._refresh_annotation_layer()
        n_persist = len(app.viewer.canvas.find_withtag("ipersist"))
        assert n_persist >= len(app._visible_interactions()) > 0, n_persist
        from msseg.viz import min_colors as _mc0
        app._seg_overlays(0, 0, rec, None, np, _mc0)     # drops "ihover" only
        assert len(app.viewer.canvas.find_withtag("ipersist")) == n_persist
        app._redraw_hover_geometry()                     # zoom/pan re-projects
        assert len(app.viewer.canvas.find_withtag("ipersist")) == n_persist
        app.show_annot_var.set(False)
        app._refresh_annotation_layer()
        assert not app.viewer.canvas.find_withtag("ipersist")

        # Right-CLICK on the image plane resolves the annotation under it;
        # a right-DRAG is a pan, so it must NOT reach the menu.
        assert app._interaction_at(5, 5) is not None, "region 0 is annotated"
        assert app._interaction_at(0, 0) is None, "background offers no menu"
        popped = []
        app._interaction_menu = lambda e, uid: popped.append(uid)
        canvas = app.viewer               # the SliceCanvas, not its tk.Canvas
        canvas.on_context = app._canvas_menu
        # Pin the view so screen == image: a withdrawn root has a 1x1 canvas,
        # and _on_row_click just recentred on that.
        canvas.view_x, canvas.view_y, canvas.scale = 0.0, 0.0, 1.0
        sx, sy = 5, 5                        # inside region 0
        want = app._interaction_at(*canvas.screen_to_image(sx, sy))
        assert want is not None, canvas.screen_to_image(sx, sy)
        canvas._context_press(_FakeEvent(sx, sy))
        canvas._context_release(_FakeEvent(sx + 40, sy))     # dragged -> pan
        assert not popped, "a right-drag pans, it does not open a menu"
        canvas._context_press(_FakeEvent(sx, sy))
        canvas._context_release(_FakeEvent(sx + 1, sy))      # a click
        assert popped == [want], (popped, want)
        del app._interaction_menu                            # back to the real one
        assert app.viewer.on_context is not None

    # Export as CSV: one row per living region (class 0 = unlabeled), carrying
    # every statistics column the spec produced for the region.
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "labels.csv")
        covered, skipped = app._write_labels_csv(csv_path)
        assert (covered, skipped) == (1, 0)
        with open(csv_path, encoding="utf-8") as f:
            rows = f.read().strip().splitlines()
        assert rows[0] == "slice,region_id,class,predicted,area,mean_base"
        assert len(rows) == 5
        assert rows[1] == "data/s0.tiff,0,1,,64,1.5"   # blank pre-classify
        assert rows[4] == "data/s0.tiff,9,1,,112,4.5"

    # Train + classify (skipped when scikit-learn isn't installed).
    try:
        import sklearn  # noqa: F401
        have_sklearn = True
    except ImportError:
        have_sklearn = False
        print("labeler selftest: scikit-learn not installed - classifier "
              "checks skipped")
    if have_sklearn:
        from msseg.viz import min_colors as _mc
        from unittest import mock as _mock
        # The fake FeatureTable's schema is not the real profile's, so pin the
        # expected-feature source to it (the real _expected_feature_names is a
        # pure schema call; the check logic below is what's under test).
        app._expected_feature_names = lambda: ["area", "mean_base"]
        assert app.model_kind_var.get() == "dense FC", "dense FC is the default"
        app.model_kind_var.set("random forest")
        app.n_classes_var.set(3)
        app._on_n_classes_change()
        app.active_class_var.set(2)
        app._commit_interaction("squiggle", [(12.0, 12.0), (15.0, 15.0)])  # {9} -> 2

        # Model operations are stack-wide even when only the visible slice has
        # a cached record. Simulate a Rerun (all old records stale), then let a
        # fake synchronous materializer stand in for the compiled pipe.
        file1 = os.path.join(data_dir, "s1.tiff")
        pair_files = [files[0], file1]
        app.subsequences[0]["files"] = pair_files
        app.primed = [{"files": pair_files, "base": [zeros, zeros],
                       "filtered": [zeros, zeros], "pipes": [None, None],
                       "normalizers": [[], []]}]
        app._rebuild_flat_slices()
        second_uids = [
            app.store.add("box", [(3.0, 3.0), (16.0, 16.0)], 1,
                          "data/s1.tiff", 0, 1).uid,
            app.store.add("squiggle", [(12.0, 12.0), (15.0, 15.0)], 2,
                          "data/s1.tiff", 0, 1).uid,
        ]
        original_ensure = app._ensure_slice_record
        app.engine.commit_selection()
        materialized = []
        templates = [rec3, rec3]

        def _materialize(si, li):
            materialized.append((si, li))
            fresh = dict(templates[li], commit=app._commit_id)
            app._slices[(si, li)] = fresh
            return fresh

        app._ensure_slice_record = _materialize
        app._train_classifier()
        assert app._clf is not None, "training must produce a model"
        assert app._clf_kind == "random forest"
        assert "min_x" not in app._clf_names and "ext_x" not in app._clf_names
        assert set(materialized) == {(0, 0), (0, 1)}, \
            "training must materialize every primed slice after a new commit"
        assert "on 8 labeled regions" in app.status_var.get(), app.status_var.get()
        rec3 = app.engine.record(0, 0)
        materialized.clear()
        app._classify()
        pr = app._pred.get(app.catalogue.key_of(0, 0))
        pr_second = app._pred.get(app.catalogue.key_of(0, 1))
        assert pr is not None and pr[0] == app._commit_id
        assert pr_second is not None and pr_second[0] == app._commit_id
        assert set(materialized) == {(0, 0), (0, 1)}, \
            "classification must score every primed slice"
        assert pr[1].shape == (10,)
        assert set(int(v) for v in pr[1][[0, 2, 5, 9]]) <= {1, 2}
        assert app.show_pred_var.get()
        if app.viewer is not None:
            assert app.viewer._hud_mode is None, "computing badge cleared"
        ovs = app._seg_overlays(0, 0, rec3, None, np, _mc)
        # ...plus the seam overlay: two classes drawn on adjacent regions
        # derive boundary / interior seam labels (derive.py) with no seam
        # gesture at all, and the seams layer shows them.
        assert len(ovs) == 4, "regions + prediction layer + drawn labels + derived seams"
        assert int(ovs[1]["lut"][:, 3].max()) < 255, "prediction layer is translucent"

        # Probabilities ride the same cache; the hard label IS their argmax.
        assert len(pr) == 3, "_pred caches (commit, region_class, region_proba)"
        proba = pr[2]
        assert proba.shape == (10, labeling.MAX_CLASSES)
        for r in (0, 2, 5, 9):
            assert abs(float(proba[r].sum()) - 1.0) < 1e-5, r
            assert int(proba[r].argmax()) == int(pr[1][r]), "hard label = argmax"
        assert float(proba[1].sum()) == 0.0, "id 1 is not a living region"
        app._hover_ctx = {"si": 0, "li": 0, "base": zeros,
                          "filt": zeros, "data": None}
        app._on_hover(3, 3)
        hover = app.hover_var.get()
        assert "P(class 1)=" in hover and "P(class 2)=" in hover, hover
        assert all(name not in hover for name in ("MSC=", "CC=", "global=", "mask=")), hover
        app._on_hover(0, 0)
        assert "class probabilities: -" in app.hover_var.get()

        # Regions coloring modes appear only once probabilities exist.
        app._refresh_region_modes()
        modes = list(app.region_mode_combo.cget("values"))
        assert modes[0] == "label id" and "uncertainty" in modes, modes
        assert "P(class 1)" in modes and "P(class 2)" in modes, modes
        for mode in ("P(class 1)", "uncertainty"):
            app.region_mode_var.set(mode)
            ovs_m = app._seg_overlays(0, 0, rec3, None, np, _mc)
            # The scalar layer REPLACES the id layer, so the stack is the same
            # height; it is the bottom one, and only living regions are opaque.
            assert len(ovs_m) == 4, mode
            lut_m = ovs_m[0]["lut"]
            assert lut_m.shape == (10, 4)
            assert int(lut_m[[0, 2, 5, 9], 3].min()) > 0, mode
            assert int(lut_m[1, 3]) == 0, "unscored region stays invisible"
        app.region_mode_var.set("label id")
        app._pred.clear()
        app._refresh_region_modes()
        assert list(app.region_mode_combo.cget("values")) == ("label id",) or \
            list(app.region_mode_combo.cget("values")) == ["label id"], \
            "modes collapse when the cache is dropped"
        app._classify()
        pr = app._pred.get(app.catalogue.key_of(0, 0))

        # -- "-> edges" kinds: an edge model on top of a dense base --------- #
        # The two fake slices carry the block raster (ids 0,2 -> class 2 and
        # 5,9 -> class 1), so pixel adjacency gives four arcs per slice: two
        # same-class, two crossing. Train fits the custom base, then the edges.
        kind_before = app.model_kind_var.get()
        assert _CUSTOM_KIND in _MODEL_KINDS and _CUSTOM_EDGE_KIND in _MODEL_KINDS
        assert _base_kind(_CUSTOM_EDGE_KIND) == _CUSTOM_KIND
        assert _edge_kind_of(_TUNED_KIND) == _TUNED_EDGE_KIND and _edge_kind_of("dense FC") is None
        assert _under(app.edge_eval_btn, app.model_tab) and _under(app.edge_tree, app.model_tab)
        assert _under(app.edge_readout, app.confusion_holder.master)
        app.custom_hidden_var.set("4")
        app.model_kind_var.set(_CUSTOM_EDGE_KIND)
        assert "not trained yet" in app.model_arch_var.get(), app.model_arch_var.get()
        assert "not trained" in app.edge_readout_var.get()
        app._train_classifier()
        assert app._clf_kind == _CUSTOM_EDGE_KIND and app._clf_spec is not None
        assert tuple(app._clf_spec.hidden) == (4,)
        edge = app._edge_model
        assert edge is not None, app.status_var.get()
        assert edge.n_edges == 8 and edge.n_diff == 4, (edge.n_edges, edge.n_diff)
        assert not edge.used_saddle and edge.width == 4 and edge.n_in == 10
        assert "-> edges" in app.status_var.get(), app.status_var.get()
        assert "-> edges" in app.model_strip_var.get() and "-> edges" in app.model_hint_var.get()
        assert "edges: logistic" in app.model_arch_var.get()
        # The saddle-free pair model: `contact` (log shared boundary length,
        # from the label raster) in, `barrier` out. Off by default, so the
        # widths above are the historical ones.
        assert not app.edge_feat_vars["contact"].get() and edge.spec.features == \
            edge_model.DEFAULT_FEATURES
        app.edge_feat_vars["contact"].set(True); app.edge_feat_vars["barrier"].set(False)
        app._on_edge_settings_change()
        assert "edge settings changed" in app.model_arch_var.get()
        app._train_classifier()
        edge_sf = app._edge_model
        assert edge_sf is not None and edge_sf.spec.features == ("absdiff", "prod", "contact")
        assert edge_sf.n_in == 9 and edge_sf.used_contact and not edge_sf.used_saddle
        assert "contact lengths" in edge_sf.describe()
        assert app.regions.arcs("data/s0.tiff", np).get("length") is not None, \
            "contact lengths derived onto the arcs once"
        app._classify()
        pr_sf = app._pred.get(app.catalogue.key_of(0, 0))
        assert len(pr_sf) == 4 and np.isfinite(app._pred_aux(pr_sf)["pdiff"]).all()
        app.edge_feat_vars["contact"].set(False); app.edge_feat_vars["barrier"].set(True)
        app._on_edge_settings_change()
        app._train_classifier()
        edge = app._edge_model
        assert edge.n_in == 10 and edge.spec.features == edge_model.DEFAULT_FEATURES
        app._classify()
        pr = app._pred.get(app.catalogue.key_of(0, 0))
        assert len(pr) == 4, "the cache entry grows an aux dict with an edge model"
        aux = app._pred_aux(pr)
        assert aux is not None and aux["pdiff"].shape == (4,) and np.isfinite(aux["pdiff"]).all()
        assert aux["source"] == "pixels" and aux["active"] and aux["raw"].shape == (10,)
        assert set(int(v) for v in pr[1][[0, 2, 5, 9]]) <= {1, 2}
        assert int(pr[1][1]) == 0, "a non-living id stays class 0 after voting"
        assert "voting on" in app.edge_readout_var.get() and "flipped" in app.edge_readout_var.get()
        # Coloring modes for the edge model.
        app._refresh_region_modes()
        modes = list(app.region_mode_combo.cget("values"))
        assert _MODE_FLIPPED in modes and _MODE_PDIFF in modes, modes
        for mode in (_MODE_FLIPPED, _MODE_PDIFF):
            app.region_mode_var.set(mode)
            vals, mask = app._region_scalar(pr, np)
            assert vals.shape == (10,) and bool(mask[0]) and not bool(mask[1]), mode
        app.region_mode_var.set(_MODE_PDIFF)
        vals, _m = app._region_scalar(pr, np)
        touching0 = [float(aux["pdiff"][i]) for i in range(4)
                     if 0 in (int(aux["la"][i]), int(aux["lb"][i]))]
        assert np.isclose(float(vals[0]), max(touching0))
        app.region_mode_var.set("label id")
        # N flips to the base kind: raw predictions, voting off, no forward pass.
        app._on_edge_key()
        assert app.model_kind_var.get() == _CUSTOM_KIND
        pr0 = app._pred.get(app.catalogue.key_of(0, 0))
        assert np.array_equal(pr0[1], app._pred_aux(pr0)["raw"]) and not app._pred_aux(pr0)["active"]
        assert "voting off" in app.edge_readout_var.get()
        app._on_edge_key()
        assert app.model_kind_var.get() == _CUSTOM_EDGE_KIND
        assert app._pred_aux(app._pred[app.catalogue.key_of(0, 0)])["active"]
        # lambda 0 is the base answer even with voting on; the entry re-votes in place.
        app.edge_lam_var.set("0")
        app._on_edge_settings_change()
        pr0 = app._pred[app.catalogue.key_of(0, 0)]
        assert np.array_equal(pr0[1], app._pred_aux(pr0)["raw"]) and app._pred_aux(pr0)["lam"] == 0.0
        app.edge_lam_var.set("1")
        app._on_edge_settings_change()
        # The learned magic metric: the flood reads p(diff) from the cache and
        # is refused without it.
        tool = app.viewer.tool if app.viewer is not None else None
        if tool is not None:
            app.magic_metric_var.set("learned")
            tool_before, cls_before = app.tool_var.get(), app.active_class_var.get()
            app.tool_var.set("magic")
            app.active_class_var.set(1)
            assert tool.on_press(_FakeEvent(5, 5)), app.status_var.get()
            assert tool.magic._s["ladder"].metric == "learned"
            assert tool.magic._s["ladder"].n_reach == 4
            assert tool.magic.cancel()
            saved_pred = app._pred
            app._pred = {}
            assert not tool.on_press(_FakeEvent(5, 5))
            assert "Classify" in app.status_var.get(), app.status_var.get()
            app._pred = saved_pred
            app.magic_metric_var.set("mean")
            app.tool_var.set(tool_before)
            app.active_class_var.set(cls_before)
        # Freeze base: Train keeps the base net and refits only the edges.
        clf_id = id(app._clf)
        app.freeze_base_var.set(True)
        app._train_classifier()
        assert id(app._clf) == clf_id and app._edge_model is not None and app._edge_model is not edge
        assert "(base frozen)" in app.status_var.get(), app.status_var.get()
        app.freeze_base_var.set(False)
        # Pickle v4 carries the edge model; the session record flags it.
        with tempfile.TemporaryDirectory() as td:
            p4 = os.path.join(td, "edges.pkl")
            app._save_classifier_to(p4)
            assert app.models[-1]["edge"] is True and app.models[-1]["kind"] == _CUSTOM_EDGE_KIND
            sdoc4 = session.session_doc_from_json(app._session_doc())
            assert sdoc4["models"][-1]["edge"] is True
            app._edge_model = None
            app.model_kind_var.set("dense FC")
            app._load_classifier_from(p4)
            assert app._edge_model is not None and app._clf_kind == _CUSTOM_EDGE_KIND
            assert app.model_kind_var.get() == _CUSTOM_EDGE_KIND
            assert app.custom_hidden_var.get() == "4"
        # Evaluate edges runs inline; with two slices every fold is stratified
        # inside them, so no fold has training edges and all are skipped -- the
        # report still lands and the buttons come back.
        app._evaluate_edges(sync=True)
        assert app._search is None and app.edge_progress_var.get(), app.edge_progress_var.get()
        assert str(app.edge_eval_btn.cget("state")) == "normal"
        # The stack settings ride the task's view and are validated on the way in.
        v = app._task_view_from_ui()["neighbours"]
        assert v["custom_hidden"] == "4" and v["edge_spec"]["lam"] == 1.0 and v["freeze_base"] is False
        app._apply_neighbours_view({"custom_hidden": "8-4", "freeze_base": True,
                                    "edge_spec": {"lam": 0.5, "rounds": 2, "features": ["absdiff"]}})
        assert app._custom_hidden() == (8, 4) and app.freeze_base_var.get()
        assert app._edge_spec_from_ui().lam == 0.5 and app._edge_spec_from_ui().features == ("absdiff",)
        app._apply_neighbours_view({"custom_hidden": "bad!"})
        assert app.custom_hidden_var.get() == "8-4", "an unreadable ladder is ignored"
        app.freeze_base_var.set(False)
        app.custom_hidden_var.set(_DEFAULT_CUSTOM_HIDDEN)
        app._apply_edge_spec(edge_model.EdgeSpec())
        # The region list behind a confusion cell, and double-click navigation.
        app._classify()                                # the pickle load cleared _pred
        app._refresh_confusion()
        counts = app._cm_counts["all"]
        cell = max(counts, key=counts.get)
        app._on_confusion_open(*cell)
        assert app._center_tab_name() == "Analysis" and app._cm_cell == cell
        assert len(app.errors_tree.get_children()) == counts[cell] == len(app._error_rows)
        assert str(cell[0]) in app.errors_header_var.get()
        assert all(d["true"] == cell[0] and d["pred"] == cell[1] for d in app._error_rows)
        first = app._error_rows[0]
        assert first["area"] >= app._error_rows[-1]["area"], "largest first within a slice"
        app.errors_tree.selection_set("0")
        app._on_error_row_open()
        app.root.update_idletasks()                    # the deferred centring
        assert app._center_tab_name() == "Analysis", \
            "the canvas is always on screen, so the list stays open"
        assert app._current() == (first["si"], first["li"])
        assert app._hover_key == (first["si"], first["li"], first["region"])
        assert app.status_var.get().startswith(f"region {first['region']}")
        assert not app._goto_region(7, 7, 0), "an unprimed slice is refused"
        app._on_confusion_click(*cell)              # toggles the cell off
        assert app._cm_cell is None and not app.errors_tree.get_children()
        assert "No confusion cell" in app.errors_header_var.get()
        # Back to the plain kind: the edge model goes with the base it sat on.
        app.model_kind_var.set(kind_before)
        app._train_classifier()
        app._classify()
        pr = app._pred.get(app.catalogue.key_of(0, 0))
        assert app._edge_model is None and len(pr) == 3

        # Confusion matrix: frozen predictions against live labels.
        app._refresh_confusion()
        counts = dict(app._cm_counts["current"])
        counts_all = dict(app._cm_counts["all"])
        truth = app._truth_from_cache(0, 0, rec3, np)
        expect = {}
        for r in (0, 2, 5, 9):
            t, p = int(truth[r]), int(pr[1][r])
            if t >= 1 and p >= 1:
                expect[(t, p)] = expect.get((t, p), 0) + 1
        assert counts == expect, (counts, expect)
        assert sum(counts.values()) == 4, counts
        assert sum(counts_all.values()) == 8, counts_all
        assert str(app._cm_cells["current"][(1, 1)].cget("text")) == \
            str(counts.get((1, 1), 0))
        assert str(app._cm_cells["all"][(1, 1)].cget("text")) == \
            str(counts_all.get((1, 1), 0))
        # A new label moves the TRUE axis without re-Classify.
        moved = next(iter(expect))
        app.active_class_var.set(3 if app.store.n_classes > 3 else 2)
        before = {scope: dict(values) for scope, values in app._cm_counts.items()}
        assert before["current"], before
        app._commit_interaction("squiggle", [(2.0, 2.0)])     # repaints id 0
        assert app._cm_counts["current"] != before["current"], \
            "labeling moves the matrix with the predictions frozen"
        assert sum(app._cm_counts["current"].values()) == \
            sum(before["current"].values()), \
            "the same regions, redistributed across the true axis"
        assert sum(app._cm_counts["all"].values()) == sum(before["all"].values()), \
            "the global table retains every slice while truth moves"
        assert app._pred.get(app.catalogue.key_of(0, 0))[1] is pr[1], "predictions stay frozen"
        app._undo()
        assert app._cm_counts == before, "undo restores both tables"

        # Clicking either shared cell highlights the same regions/current cell
        # in both grids and reports current/global counts.
        app._on_confusion_click(*moved)
        assert app._cm_cell == moved
        assert app._cm_cells["all"][moved].cget("background") == "#cde8ff"
        assert app._cm_cells["current"][moved].cget("background") == "#cde8ff"
        hits = app._confusion_hits()
        assert hits == {r for r in (0, 2, 5, 9)
                        if (int(truth[r]), int(pr[1][r])) == moved}, hits
        assert (f"{before['current'].get(moved, 0)} on this slice / "
                f"{before['all'].get(moved, 0)} total") in app.status_var.get()
        ovs_h = app._seg_overlays(0, 0, rec3, None, np, _mc)
        assert len(ovs_h) == 5, "highlight rides on top (of regions, prediction, labels, seams)"
        assert set(np.nonzero(ovs_h[-1]["lut"][:, 3])[0].tolist()) == hits
        app._on_confusion_click(*moved)                       # click again clears
        assert app._cm_cell is None
        assert len(app._seg_overlays(0, 0, rec3, None, np, _mc)) == 4

        # Navigating changes only the current-slice table; the all-slice table
        # remains the aggregate over both records.
        all_before_nav = dict(app._cm_counts["all"])
        app._goto_slice(1)
        rec_second = app.engine.record(0, 1)
        truth_second = app._truth_from_cache(0, 1, rec_second, np)
        expect_second = {}
        for r in (0, 2, 5, 9):
            t, p = int(truth_second[r]), int(pr_second[1][r])
            if t >= 1 and p >= 1:
                expect_second[(t, p)] = expect_second.get((t, p), 0) + 1
        assert app._cm_counts["current"] == expect_second
        assert app._cm_counts["all"] == all_before_nav
        app._goto_slice(0)

        # Return the rest of the broad selftest to its original one-slice fixture.
        for uid in second_uids:
            app.store.remove(uid)
        app.subsequences[0]["files"] = files
        app.primed = [{"files": files, "base": [zeros], "filtered": [zeros],
                       "pipes": [None], "normalizers": [[]]}]
        app._slices = {(0, 0): rec3}
        app._pred = {(0, 0): pr}
        app._ensure_slice_record = original_ensure
        app._rebuild_flat_slices()
        app._rebuild_class_panels()

        # Dense FC kind: same buttons, scaler inside the pickled pipeline.
        app.model_kind_var.set("dense FC")
        app._train_classifier()
        assert app._clf_kind == "dense FC"
        app._classify()
        assert app._pred, "dense FC classifies"

        # RF-selected dense variants consume the full profile fingerprint but
        # train the MLP on only the selected dimensions. This two-column
        # fixture also exercises the top-N cap for profiles narrower than N.
        assert set(_DENSE_TOP_N) <= set(_MODEL_KINDS)
        assert app._make_model("dense-top-16", 8, 40).named_steps[
            "select"].max_features == 16
        assert app._make_model("dense-top-32", 8, 40).named_steps[
            "select"].max_features == 32
        for dense_kind in ("dense-top-16", "dense-top-32"):
            app.model_kind_var.set(dense_kind)
            app._train_classifier()
            assert app._clf_kind == dense_kind
            selector = app._clf.named_steps["select"]
            assert int(selector.get_support().sum()) == 2
            assert selector.max_features == 2
            assert app._clf_names == ["area", "mean_base"], \
                "the full profile fingerprint must survive selection"
            app._classify()
            before_load = app._pred[app.catalogue.key_of(0, 0)][1].copy()
            with tempfile.TemporaryDirectory() as td:
                dense_path = os.path.join(td, dense_kind + ".pkl")
                app._save_classifier_to(dense_path)
                app._clf = None
                app._clf_names = None
                app.classify_btn.config(state="disabled")
                app._load_classifier_from(dense_path)
                assert app._clf_kind == dense_kind
                assert app.model_kind_var.get() == dense_kind
                app._classify()
                assert np.array_equal(app._pred[app.catalogue.key_of(0, 0)][1], before_load), \
                    "loaded dense-top model must preserve predictions"
        # "dense (tuned)": Train with no search yet builds the baseline behind a
        # FeatureSubset; Optimize (run inline here) installs the search winner,
        # whose spec rides the pickle and the session model record.
        assert _TUNED_KIND in _MODEL_KINDS
        assert _under(app.optimize_btn, app.model_tab)
        assert str(app.cancel_search_btn.cget("state")) == "disabled"
        app.model_kind_var.set(_TUNED_KIND)
        assert "No search yet" in app.model_arch_var.get()
        app._train_classifier()
        assert app._clf_kind == _TUNED_KIND and app._clf_spec is None
        assert app._clf.named_steps["dense"].hidden_layer_sizes == _MLP_HIDDEN
        assert app._clf_names == ["area", "mean_base"], "full fingerprint kept"
        app._classify()
        assert app._pred, "tuned baseline classifies"
        # Unscorable labels (one class) refuse with a reason, not a traceback.
        original_training_set = app._training_set
        app._training_set = lambda: (np.zeros((6, 2)), np.ones(6, int),
                                     np.arange(6), ["area", "mean_base"])
        app._optimize_network(sync=True)
        assert app._search is None and "Cannot optimize" in app.status_var.get()
        # A synthetic labeled set -- six slices, three classes -- runs the
        # search to completion inline, grouped by slice.
        rng = np.random.RandomState(0)
        Xs = rng.normal(size=(90, 2))
        ys = rng.randint(1, 4, size=90)
        gs = rng.randint(0, 6, size=90)
        Xs[:, 1] += 2.0 * ys
        app._training_set = lambda: (Xs, ys, gs, ["area", "mean_base"])
        app.search_trials_var.set("3")
        app.search_timeout_var.set("60")            # minutes
        models_td = tempfile.mkdtemp()
        app._models_dir = models_td                 # never the user's folder
        app._optimize_network(sync=True)
        assert app._search is None, "the pump drains a sync search to completion"
        # The winner is pickled + recorded on finish, so a restore finds it.
        saved = [f for f in os.listdir(models_td) if f.startswith("tuned_")]
        assert len(saved) == 1, saved
        assert app.models[-1]["path"] == os.path.join(models_td, saved[0])
        assert app.models[-1]["kind"] == _TUNED_KIND
        assert "saved" in app.status_var.get()
        assert app._clf_spec.max_iter == _MLP_MAX_ITER, "refit with the full budget"
        assert str(app.optimize_btn.cget("state")) == "normal"
        assert app._clf_kind == _TUNED_KIND and app._clf_spec is not None
        spec = app._clf_spec
        assert spec.n_trials == 3 and spec.cv_kind == "slices" and spec.n_groups == 6
        assert spec.cv_score is not None and spec.cv_score <= spec.baseline_score
        assert app._search_spec is spec
        assert app.model_arch_var.get() == _model_description(_TUNED_KIND, spec, 2)
        assert "CV bacc" in app.model_hint_var.get(), app.model_hint_var.get()
        assert "CV bacc" in app.model_strip_var.get(), app.model_strip_var.get()
        assert app._pred, "the winner is classified immediately"
        assert "trials" in app.search_progress_var.get()
        # Train on the tuned kind rebuilds the same spec on the current labels.
        app._pred.clear()
        app._train_classifier()
        assert app._clf_spec == spec and app._clf_kind == _TUNED_KIND
        # The spec rides the pickle (v3) and the session's model record.
        with tempfile.TemporaryDirectory() as td:
            tuned_path = os.path.join(td, "tuned.pkl")
            app._save_classifier_to(tuned_path)
            assert app.models[-1]["kind"] == _TUNED_KIND
            assert app.models[-1]["spec"]["hidden"] == list(spec.hidden)
            sdoc_t = session.session_doc_from_json(app._session_doc())
            assert sdoc_t["models"][-1]["spec"]["hidden"] == list(spec.hidden)
            app._clf = None
            app._clf_spec = None
            app._search_spec = None
            app.model_kind_var.set("dense FC")
            app._load_classifier_from(tuned_path)
            assert app._clf_kind == _TUNED_KIND
            assert app._clf_spec == spec and app._search_spec == spec
            assert app.model_kind_var.get() == _TUNED_KIND
            assert app.model_arch_var.get() == _model_description(_TUNED_KIND, spec, 2)
        # The search settings ride the session view, clamped on the way in.
        assert app._task_view_from_ui()["model_search"] == \
            {"trials": 3, "timeout_s": 3600, "seed": 0, "feature_search": True,
             "backend": "auto",
             "sweep_sizes": "64-32, 32-16, 16-8, 8-4, 4", "sweep_trials": _SWEEP_TRIALS}, \
            "the entry is minutes, the session seconds"
        app._apply_search_view({"trials": 12, "timeout_s": 5, "seed": 7,
                                "feature_search": False, "backend": "sklearn", "junk": 1})
        assert app._search_settings() == (12, 5, 7, False, "sklearn")
        app.search_timeout_var.set("0.5")
        assert app._search_settings()[1] == 30, "fractional minutes"
        app.search_timeout_var.set("junk")
        assert app._search_settings()[1] == _SEARCH_TIMEOUT_S, "malformed = default"
        # Size sweep: one fixed-size search per rung, a report row per rung,
        # the best rung installed + saved, any rung installable from the table.
        app.search_timeout_var.set("60")
        assert _under(app.sweep_tree, app.analysis_tab) and _under(app.sweep_btn, app.analysis_tab)
        assert _under(app.errors_tree, app.analysis_tab)
        assert model_search.parse_sizes(" 64-32, 16x8;4 ") == [(64, 32), (16, 8), (4,)]
        assert model_search.n_params((4,), 2, 3) == (2 + 1) * 4 + (4 + 1) * 3
        app.sweep_sizes_var.set("8-4, 4")
        app.sweep_trials_var.set("2")
        assert app._sweep_settings() == ([(8, 4), (4,)], 2)
        app.sweep_sizes_var.set("8-4, banana")
        app._sweep_network(sync=True)
        assert app._search is None and "cannot read" in app.status_var.get()
        app.sweep_sizes_var.set("8-4, 4")
        n_models_before = len([f for f in os.listdir(models_td) if f.startswith("tuned_")])
        app._sweep_network(sync=True)
        assert app._search is None and app._sweep is not None
        res = app._sweep
        assert [r.size for r in res.rows] == ["8-4", "4"] and all(r.n_trials == 2 for r in res.rows)
        assert all(r.spec.hidden == r.hidden and r.spec.baseline_hidden == r.hidden
                   for r in res.rows), "each rung searched at its own fixed size"
        assert all(r.spec.max_iter == _MLP_MAX_ITER for r in res.rows)
        assert len(app.sweep_tree.get_children()) == 2
        assert app.sweep_tree.item("0", "values")[0] == "8-4"
        assert app.sweep_summary_var.get().startswith("best: ")
        assert "rung" in app.search_progress_var.get()
        best = res.rows[res.best_index()]
        assert app._clf_spec == best.spec and app._clf_kind == _TUNED_KIND
        assert app._clf is best.estimator and app._pred, "best rung installed + classified"
        assert len([f for f in os.listdir(models_td) if f.startswith("tuned_")]) == n_models_before + 1
        assert len([f for f in os.listdir(models_td) if f.startswith("sweep_")]) == 1
        lines = res.report_lines()
        assert lines[0].startswith("size") and len(lines) == 2 + len(res.rows)
        assert res.to_dict()["rows"][0]["hidden"] == [8, 4]
        # Install the OTHER rung from the table.
        other = 1 - res.best_index()
        app.sweep_tree.selection_set(str(other))
        app._refresh_sweep_buttons()
        assert str(app.sweep_install_btn.cget("state")) == "normal"
        app._sweep_install_selected()
        assert app._clf is res.rows[other].estimator and app._clf_spec == res.rows[other].spec
        assert app._search_spec == res.rows[other].spec
        assert "Installed " + res.rows[other].size in app.status_var.get()
        assert app.model_arch_var.get() == _model_description(_TUNED_KIND, res.rows[other].spec, 2)
        # The sweep settings ride the session view and are validated on the way in.
        app._apply_search_view({"sweep_sizes": "nope!", "sweep_trials": 7})
        assert app.sweep_sizes_var.get() == "8-4, 4" and app._sweep_settings()[1] == 7
        app._apply_search_view({"sweep_sizes": "16-8, 4"})
        assert app.sweep_sizes_var.get() == "16-8, 4"
        app.sweep_sizes_var.set(", ".join(model_search.size_text(h)
                                          for h in model_search.DEFAULT_SIZE_LADDER))
        app.sweep_trials_var.set(str(_SWEEP_TRIALS))
        app._sweep = None
        app._refresh_sweep_report([])
        app._apply_search_view({"backend": "nonsense"})
        assert app._search_settings()[4] == "sklearn", "unknown backend ignored"
        # A forced sklearn backend builds sklearn's MLP even when torch exists;
        # the spec records the resolved backend and the readout names it.
        app.search_trials_var.set("2")
        app._optimize_network(sync=True)
        assert app._search is None and app._clf_spec.backend == "sklearn"
        assert type(app._clf.named_steps["dense"]).__name__ == "MLPClassifier"
        assert "sklearn (cpu)" in app.model_arch_var.get()
        app.search_backend_var.set("auto")
        app.search_trials_var.set("abc")
        assert app._search_settings()[0] == _SEARCH_TRIALS, "malformed = default"
        app.search_trials_var.set(str(_SEARCH_TRIALS))
        app.search_timeout_var.set(str(_SEARCH_TIMEOUT_MIN))
        import shutil
        shutil.rmtree(models_td, ignore_errors=True)
        app._models_dir = None
        app.search_seed_var.set("0")
        app.search_features_var.set(True)
        app.search_backend_var.set("auto")
        app._training_set = original_training_set
        app._search_spec = None
        app.model_kind_var.set("random forest")

        # 'R' = train + immediate reclassify in one call, without resetting
        # visualization choices during the transient empty prediction cache.
        app.region_mode_var.set(_MODE_UNCERTAINTY)
        app.show_pred_var.set(False)
        app._cm_cell = (1, 1)
        app._pred.clear()
        app._train_and_classify()
        assert app._clf_kind == "random forest" and app._pred, \
            "'R' retrains and reclassifies"
        assert app.region_mode_var.get() == _MODE_UNCERTAINTY
        assert not app.show_pred_var.get(), "R preserves Show Classification"
        assert app._cm_cell == (1, 1), "R preserves the confusion selection"

        # 'C' = classify with the current model and also leaves visibility alone.
        app._pred.clear()
        app._on_classify_key()
        assert app._pred, "'C' reclassifies"
        assert not app.show_pred_var.get(), "Classify does not force its layer on"
        app.show_pred_var.set(True)

        # SHIFT-accept: the predictions under a box become one "taps"
        # interaction per predicted class -- geometric, and ONE undo step.
        n_before = len(app.store.interactions)
        app._accept_predictions([(2.0, 2.0), (17.0, 17.0)])   # covers all 4
        added = app.store.interactions[n_before:]
        assert added and all(it.tool == "taps" for it in added)
        assert sum(len(it.points) for it in added) == 4, "all regions accepted"
        rc_now = labeling.resolve_slice(app.store.for_slice("data/s0.tiff"),
                                        lab, np)
        pr_now = app._pred[app.catalogue.key_of(0, 0)][1]
        for r in (0, 2, 5, 9):
            assert rc_now[r] == pr_now[r], "accepted labels match predictions"
        app._undo()
        assert len(app.store.interactions) == n_before, "batch accept = one undo"

        # Classifier save/load round trip, session model refs, and the CSV.
        with tempfile.TemporaryDirectory() as td:
            clf_path = os.path.join(td, "classifier.pkl")
            app._save_classifier_to(clf_path)
            assert app.models and app.models[-1]["fingerprint"] == ["area", "mean_base"]
            assert app.models[-1]["kind"] == "random forest"
            sdoc2 = app._session_doc()
            models2 = sdoc2["tasks"][0]["models"]
            assert models2 and models2[-1]["path"] == os.path.abspath(clf_path)
            app._clf = None
            app._clf_names = None
            app.classify_btn.config(state="disabled")
            # The compatibility gate: an incompatible profile refuses the load
            # outright, and Classify blocks with a message instead of silently
            # skipping slices.
            app._expected_feature_names = lambda: ["area", "mean_base", "std_blur_s1.5"]
            try:
                app._load_classifier_from(clf_path)
                raise AssertionError("load must refuse an incompatible model")
            except ValueError as exc:
                assert "std_blur_s1.5" in str(exc), exc
            assert app._clf is None, "nothing installed on refusal"

            # Track A: interactively, a v2 pickle's OWN statistics can be
            # applied as a new profile instead of that flat refusal.
            def _expected_from_profile():
                """Stand-in for the real schema call (which needs the compiled
                extension): derive the field set from the ACTIVE profile's
                statistics, so applying the model's own really changes it."""
                doc = config_io.statistics_from_json(
                    app.profiles[app.active_profile_idx].get("statistics"))
                out = ["area"]
                for c in doc["channels"]:
                    for s in (c.get("sigmas") or [None]):
                        out.append("mean_" + c["kind"]
                                   + ("" if s is None else f"_s{s:g}"))
                return out

            def _reprime():
                """Put the fake primed stack back, its record at the current
                commit, for the rest of the selftest."""
                app.primed = [{"files": files, "base": [zeros],
                               "filtered": [zeros], "pipes": [None],
                               "normalizers": [[]]}]
                app._rebuild_flat_slices()
                rec["commit"] = app._commit_id
                app._slices[(0, 0)] = rec

            app._expected_feature_names = _expected_from_profile
            # The pickle was saved under base-only statistics; move the ACTIVE
            # profile off them so the load really mismatches.
            app.stat_kind_vars["blur"][0].set(True)
            app.stat_kind_vars["blur"][1].set("1.5")
            app._snapshot_active_profile()
            assert app._check_model_compat(["area", "mean_base"], "t") is not None
            n_prof = len(app.profiles)
            with _mock.patch.object(messagebox, "askyesno", return_value=False):
                try:
                    app._load_classifier_from(clf_path, interactive=True)
                    raise AssertionError("declining the offer must still refuse")
                except ValueError:
                    pass
            assert len(app.profiles) == n_prof, "declining creates no profile"
            assert app._clf is None, "nothing installed when the offer is declined"
            # The primed stack's field is the new profile's (only the
            # statistics differ), so the switch keeps it and re-assembles --
            # re-measures -- the slice on screen instead of dropping it.
            from . import fingerprints as _fp
            app._primed_fingerprint = _fp.field_fingerprint_of(app._params_json(cores=1))
            primed_before = app.primed
            prev_idx = app.active_profile_idx
            with _mock.patch.object(messagebox, "askyesno", return_value=True), \
                    _mock.patch.object(app, "_request_assembly") as _req:
                app._load_classifier_from(clf_path, interactive=True)
            assert app.primed is primed_before, "a statistics-only switch keeps the primes"
            assert _req.called, "the slice on screen is re-assembled under the new statistics"
            # The commit is the identity of the parameters: back and forth
            # between the two profiles (what a task switch does) returns each
            # one's commit, so their records and predictions are found again.
            c_model = app._commit_id
            with _mock.patch.object(app, "_request_assembly"):
                app._switch_profile(prev_idx)
                c_prev = app._commit_id
                app._switch_profile(n_prof)
            assert c_prev != c_model and app._commit_id == c_model, \
                "returning to a profile returns its commit"
            # The stage strip: four boxes; a loaded model makes no claim about
            # what it was trained on; a stamp that moved turns it stale; a box
            # opens its tab.
            app._refresh_stages()
            assert set(app.viewer.stages) == {"msc", "stats", "model", "classified"}
            assert app._task.model.trained_rev is None
            assert "Loaded model" in (app.viewer.stage_tip("model") or "") or \
                app.viewer.stages["model"] == "stale"
            app._task.model.trained_rev = app.store.rev - 1
            app._refresh_stages()
            assert app.viewer.stages["model"] == "stale"
            app._task.model.trained_rev = None
            app._on_stage_click("model")
            assert app._center_tab_name() == "Model"
            app._on_stage_click("stats")
            assert app._center_tab_name() == "Features"
            app._show_center_tab("Processing")
            assert len(app.profiles) == n_prof + 1, "accepting adds a profile"
            assert app.active_profile_idx == n_prof, "and switches to it"
            assert app.profiles[-1]["name"] == "from classifier.pkl"
            assert not app.stat_kind_vars["blur"][0].get(), \
                "the model's statistics reached the stats panel"
            assert app._clf is not None and not app._pred, \
                "model installed, predictions dropped with the old parameters"
            strip = app.model_strip_var.get()
            assert "random forest" in strip and "2 feats" in strip, strip
            assert "mismatch" not in strip, strip
            _reprime()
            # Headless (no compiled extension) the offer never fires: the real
            # gate returns None, so the load succeeds as before.
            app._expected_feature_names = lambda: ["area", "mean_base"]
            app._load_classifier_from(clf_path)
            assert app._clf is not None and app._clf_names == ["area", "mean_base"]
            assert str(app.classify_btn.cget("state")) == "normal"
            app._expected_feature_names = lambda: ["area", "mean_base", "extra"]
            msg = app._check_model_compat(app._clf_names, "test")
            assert msg is not None and "extra" in msg
            with _mock.patch.object(messagebox, "showerror") as _err:
                app._classify()
                assert _err.called, "classify must block on mismatch"
            app._expected_feature_names = lambda: ["area", "mean_base"]
            app._classify()
            csv2 = os.path.join(td, "labels2.csv")
            covered2, _sk = app._write_labels_csv(csv2)
            assert covered2 == 1
            with open(csv2, encoding="utf-8") as f:
                rows2 = f.read().strip().splitlines()
            preds = [r.split(",")[3] for r in rows2[1:]]
            assert all(v in ("1", "2") for v in preds), "predicted column filled"

            # Training-set export: raw TIFF into train/, class-id mask into
            # labels/ (classifier predictions, user annotations winning).
            from PIL import Image as _Img
            src_dir = os.path.join(td, "src")
            os.makedirs(src_dir)
            src_tif = os.path.join(src_dir, "s0.tiff")
            _Img.fromarray(zeros).save(src_tif)
            app.folders = [{"path": src_dir, "name": "data"}]
            app.subsequences = [{"name": "seq1", "folder": "data",
                                 "files": [src_tif]}]
            app.primed = [{"files": [src_tif], "base": [zeros],
                           "filtered": [zeros], "pipes": [None],
                           "normalizers": [[]]}]
            app._rebuild_flat_slices()
            rec5 = dict(rec3, commit=app._commit_id)
            app._slices[(0, 0)] = rec5
            out_dir = os.path.join(td, "tset")
            written, skipped_ts = app._write_training_set(out_dir)
            assert (written, skipped_ts) == (1, 0), (written, skipped_ts)
            assert os.path.isfile(os.path.join(out_dir, "train", "data__s0.tiff"))
            mask = np.asarray(_Img.open(
                os.path.join(out_dir, "labels", "data__s0.tiff")))
            assert mask.shape == (20, 20) and mask.dtype == np.uint8
            assert mask[0, 0] == 0, "background (-1) stays class 0"
            assert mask[5, 5] > 0, "labeled region carries its class id"
            # User annotations win over predictions where they disagree.
            rc_user = labeling.resolve_slice(
                app.store.for_slice("data/s0.tiff"), lab, np)
            for r, px in ((0, (5, 5)), (9, (12, 12))):
                if rc_user[r] > 0:
                    assert mask[px] == rc_user[r]

    # -- Drawing-tool state machine: gesture previews + magic fill ----------- #
    # Fresh one-slice fixture at the current commit (earlier sections swapped
    # the record). No engine: adjacency comes from the pixel fallback and the
    # table has mean_base only (so the extremum points fall back to the first
    # pixel of each region).
    rec_m = {"commit": app._commit_id, "labels": lab, "stats": table,
             "kept": set(), "cc": None, "n_feat": 4}
    app._slices[(0, 0)] = rec_m
    app._pred = {}
    v = app.viewer
    v.view_x, v.view_y, v.scale = 0.0, 0.0, 1.0
    ctrl = v.tool
    assert isinstance(ctrl, DrawController)
    n_before = len(app.store.interactions)

    def lit():
        return set(np.flatnonzero(v._transient[2][:, 3] > 0).tolist())

    # Squiggle: incremental, exact -- from region 0 into region 2.
    app.tool_var.set("squiggle"); app.active_class_var.set(1)
    assert ctrl.on_press(_FakeEvent(5, 5))
    assert v._transient is not None and app._hover_suppressed
    assert lit() == {0}, lit()
    assert ctrl.on_move(_FakeEvent(15, 5))
    assert lit() == {0, 2}, lit()
    assert ctrl.on_release(_FakeEvent(15, 5))
    assert v._transient is None and not app._hover_suppressed
    assert len(app.store.interactions) == n_before + 1
    assert app.store.interactions[-1].tool == "squiggle"

    # Box across all four blocks, then Escape: nothing committed, class kept.
    app.tool_var.set("box")
    assert ctrl.on_press(_FakeEvent(3, 3))
    assert ctrl.on_move(_FakeEvent(16, 16))
    assert lit() == {0, 2, 5, 9}, lit()
    app._on_escape()
    assert v._transient is None and ctrl._pts is None
    assert len(app.store.interactions) == n_before + 1, "Escape commits nothing"
    assert app.active_class_var.get() == 1, "Escape mid-gesture keeps the class"
    app._on_escape()
    assert app.active_class_var.get() == 0, "Escape with nothing in flight disarms"

    # SHIFT-accept box: the preview shows what WOULD be accepted, each region
    # in its predicted class color; unpredicted regions stay dark.
    app.active_class_var.set(1)
    k_hi = app.store.n_classes - 1
    pred_rc = np.zeros(10, np.uint8); pred_rc[0] = 1; pred_rc[9] = k_hi
    app._pred[app.catalogue.key_of(0, 0)] = (app._commit_id, pred_rc,
                         np.zeros((10, MAX_CLASSES), np.float32))
    assert ctrl.on_press(_FakeEvent(3, 3, state=0x0001))
    assert ctrl.on_move(_FakeEvent(16, 16, state=0x0001))
    assert lit() == {0, 9}, lit()
    pure = np.asarray(app.store.rgba(k_hi)[:3], np.int32)
    shown = v._transient[2][9, :3].astype(np.int32)
    assert (shown >= pure).all() and v._transient[2][9, 3] == 255, (shown, pure)
    app._on_escape()
    app._pred = {}

    # Magic fill: seed region 0, mean/anchor on base. The ladder is
    # d = [0, 1, 2, 3] / std(mean_base), so a long drag up sweeps all four
    # regions and a drag down leaves the seed alone.
    app.tool_var.set("magic"); app.active_class_var.set(1)
    assert app.magic_metric_var.get() == "mean"
    assert ctrl.on_press(_FakeEvent(5, 5))
    assert ctrl.magic.active and v._transient is not None
    assert rec_m.get("arcs") is not None and rec_m["arcs"]["source"] == "pixels"
    assert set(zip(rec_m["arcs"]["a"].tolist(), rec_m["arcs"]["b"].tolist())) == \
        {(0, 2), (0, 5), (2, 9), (5, 9)}
    assert v._hud_mode == "info" and "magic" in v._hud_text
    assert lit() == {0}, lit()
    assert ctrl.on_move(_FakeEvent(5, 5 - 400))
    assert lit() == {0, 2, 5, 9}, lit()
    assert ctrl.on_move(_FakeEvent(5, 5 + 400))
    assert lit() == {0}, lit()
    assert ctrl.on_move(_FakeEvent(5, 5 - 400))
    assert ctrl.on_release(_FakeEvent(5, 5 - 400))
    assert v._transient is None and v._hud_mode is None and not ctrl.magic.active
    it_m = app.store.interactions[-1]
    assert it_m.tool == "taps" and it_m.meta and it_m.meta["tool"] == "magic"
    assert it_m.meta["n_regions"] == 4 and it_m.meta["seed_id"] == 0
    assert it_m.meta["extent"] is True, "a released fill is an extent by default"
    assert len(it_m.points) == 4
    rc_m = labeling.resolve_slice([it_m], lab, np)
    assert all(rc_m[r] == 1 for r in (0, 2, 5, 9))
    # Provenance in the session doc; the options ride the view state; undo
    # removes the whole fill in one step.
    app._rebuild_class_panels()
    doc_m = app._session_doc()
    assert doc_m["view"]["magic"]["metric"] == "mean"
    assert any(d.get("meta", {}).get("tool") == "magic"
               for d in doc_m["tasks"][0]["annotations"]["interactions"])
    n_now = len(app.store.interactions)
    app._undo()
    assert len(app.store.interactions) == n_now - 1
    assert not any(it.meta for it in app.store.interactions)
    # A second press starts from the threshold last released (all four).
    app.active_class_var.set(1)
    assert ctrl.on_press(_FakeEvent(5, 5))
    assert lit() == {0, 2, 5, 9}, lit()
    assert ctrl.magic.cancel()
    assert v._transient is None
    # Escape mid-fill abandons it and keeps the class armed; a press on the
    # background is refused (falls through to the pan).
    assert ctrl.on_press(_FakeEvent(12, 12))
    app._on_escape()
    assert not ctrl.magic.active and app.active_class_var.get() == 1
    assert not ctrl.on_press(_FakeEvent(0, 0))
    # The chain mode and bhattacharyya need std_ columns: a table without
    # them reports instead of raising out of the press.
    app.magic_metric_var.set("bhattacharyya")
    assert not ctrl.on_press(_FakeEvent(5, 5))
    assert "std_base" in app.status_var.get(), app.status_var.get()
    app.magic_metric_var.set("mean")

    # Blobber: the magic core in the active class, its immediate neighbours
    # in the ring class ("next" -> class 2 for class 1). The last released
    # threshold (all four) is remembered, so drag DOWN to the seed alone.
    if app.store.n_classes < 3:
        app.store.set_n_classes(3); app.n_classes_var.set(3)
    app.tool_var.set("blobber"); app.active_class_var.set(1)
    assert ctrl.magic.ring_class(1) == 2
    app.blob_ring_var.set("1")                 # same as the active class -> next
    assert ctrl.magic.ring_class(1) == 2
    app.blob_ring_var.set(str(app.store.n_classes - 1))
    assert ctrl.magic.ring_class(1) == app.store.n_classes - 1
    app.blob_ring_var.set("next")
    assert ctrl.on_press(_FakeEvent(5, 5))
    assert ctrl.magic.active and "ring" in v._hud_text
    assert ctrl.on_move(_FakeEvent(5, 5 + 400))
    assert lit() == {0, 2, 5}, lit()           # core {0}, ring = its neighbours
    lut = v._transient[2]
    assert lut[0].tolist() == list(app.store.rgba(1)), "seed keeps the pure core color"
    c1 = np.asarray(app.store.rgba(1)[:3], int); c2 = np.asarray(app.store.rgba(2)[:3], int)
    for r in (2, 5):
        d1 = np.abs(lut[r, :3].astype(int) - c1).sum()
        d2 = np.abs(lut[r, :3].astype(int) - c2).sum()
        assert d2 < d1, "ring regions preview in the ring class color"
    n0 = len(app.store.interactions)
    assert ctrl.on_release(_FakeEvent(5, 5 + 400))
    assert len(app.store.interactions) == n0 + 2, "ring + core, two interactions"
    ring_it, core_it = app.store.interactions[-2], app.store.interactions[-1]
    assert ring_it.class_id == 2 and ring_it.meta["part"] == "ring" \
        and ring_it.meta["tool"] == "blobber" and len(ring_it.points) == 2
    assert core_it.class_id == 1 and core_it.meta["part"] == "core" \
        and len(core_it.points) == 1
    # The core is an extent, the ring a sample (derive.py).
    assert core_it.meta["extent"] is True and "extent" not in ring_it.meta
    rc_b = labeling.resolve_slice([ring_it, core_it], lab, np)
    assert rc_b[0] == 1 and rc_b[2] == 2 and rc_b[5] == 2 and rc_b[9] == 0
    app._rebuild_class_panels()
    assert app._session_doc()["view"]["magic"]["ring"] == "next"
    # ...and with no seam gesture at all the seams between the core and its
    # ring are boundaries, derived from the region gestures (derive.py);
    # the ring's outer seams stay unknown (a sample says nothing about its
    # neighbours), and the readout says so.
    from msseg.labeler import derive as _derive
    from msseg.labeler.seams import SEAM_BOUNDARY as _SB
    _gb, cls_b = app._seam_classes_for(0, 0, np)
    idx_b = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(_gb.a, _gb.b))}
    assert cls_b[idx_b[(0, 2)]] == _SB and cls_b[idx_b[(0, 5)]] == _SB, "core | ring"
    # The other seams follow the gestures earlier sections left on this
    # slice; the app's answer is the derivation module's, exactly.
    key_b = app.catalogue.key_of(0, 0)
    _rc, sl_b, _bo, _di = _derive.labels_for_item(
        app._gestures_for_key(key_b), app._seam_gestures_for_key(key_b), lab, _gb, None, np)
    assert cls_b.tolist() == sl_b.cls.tolist(), (cls_b.tolist(), sl_b.cls.tolist())
    assert sl_b.derived.any() and "derived" in app._seam_counts_text(), app._seam_counts_text()
    app._undo()
    assert len(app.store.interactions) == n0, "one undo step removes ring + core"
    # Grown to everything: no ring left, only the core commits.
    assert ctrl.on_press(_FakeEvent(5, 5))
    assert ctrl.on_move(_FakeEvent(5, 5 - 400))
    assert lit() == {0, 2, 5, 9}, lit()
    assert ctrl.on_release(_FakeEvent(5, 5 - 400))
    assert len(app.store.interactions) == n0 + 1
    assert app.store.interactions[-1].meta["part"] == "core"
    app._undo()

    # Hop gain + drag entries: lenient parsing, clamped, session round-trip
    # as floats, restored with validation; the gain reaches the ladder, the
    # HUD, the provenance and the last-threshold key.
    app.tool_var.set("magic")
    app.magic_gain_var.set("abc"); assert ctrl.magic.hop_gain() == _DEFAULT_HOP_GAIN
    app.magic_gain_var.set("0.5"); assert ctrl.magic.hop_gain() == 1.0
    app.magic_gain_var.set("9");   assert ctrl.magic.hop_gain() == 2.0
    app.magic_drag_var.set("x");   assert ctrl.magic.drag_px() == _DEFAULT_DRAG_PX
    app.magic_gain_var.set("1.5"); app.magic_drag_var.set("8")
    vs = app._session_doc()["view"]["magic"]
    assert vs["hop_gain"] == 1.5 and vs["drag_px"] == 8.0
    assert vs["extent"] is True, "the extent checkbox rides the view"
    n0 = len(app.store.interactions)
    assert ctrl.on_press(_FakeEvent(5, 5))
    assert "g1.5" in v._hud_text, v._hud_text
    assert ctrl.magic._s["ladder"].hop_gain == 1.5
    assert ctrl.magic._s["drag_px"] == 8.0
    assert ctrl.on_move(_FakeEvent(5, 5 - 400))
    assert ctrl.on_release(_FakeEvent(5, 5 - 400))
    assert len(app.store.interactions) == n0 + 1
    assert app.store.interactions[-1].meta["hop_gain"] == 1.5
    assert any(k[3] == 1.5 for k in ctrl.magic._last), list(ctrl.magic._last)
    app._undo()
    app._apply_magic_view({"hop_gain": 1.25, "drag_px": 6, "metric": "cosine",
                           "mode": "chain"})
    assert app.magic_gain_var.get() == "1.25" and app.magic_drag_var.get() == "6"
    assert app.magic_metric_var.get() == "cosine" and app.magic_mode_var.get() == "chain"
    app._apply_magic_view({"hop_gain": "abc", "drag_px": 0.5, "metric": "nope"})
    assert app.magic_gain_var.get() == "1.25" and app.magic_drag_var.get() == "6"
    assert app.magic_metric_var.get() == "cosine"
    app._apply_magic_view({"extent": False})
    assert app.magic_extent_var.get() is False
    app._apply_magic_view({"extent": "no"})              # not a bool: ignored
    assert app.magic_extent_var.get() is False
    # With the checkbox off a fill is a sample: no `extent` on its taps.
    app.magic_gain_var.set("1")
    n0 = len(app.store.interactions)
    assert ctrl.on_press(_FakeEvent(5, 5))
    assert ctrl.on_move(_FakeEvent(5, 5 - 400))
    assert ctrl.on_release(_FakeEvent(5, 5 - 400))
    assert len(app.store.interactions) == n0 + 1
    assert app.store.interactions[-1].meta["extent"] is False
    app._undo()
    app._apply_magic_view({"extent": True})
    assert app.magic_extent_var.get() is True
    # cosine runs on the whole row (here area + mean_base) without channels.
    app.magic_gain_var.set("1")
    assert ctrl.on_press(_FakeEvent(5, 5))
    assert ctrl.magic._s["ladder"].metric == "cosine"
    assert ctrl.magic.cancel()
    app.magic_mode_var.set("anchor")

    # A lasso is an extent unless Ctrl was held at press (derive.py).
    app.tool_var.set("polygon"); app.active_class_var.set(1)
    n0 = len(app.store.interactions)
    assert ctrl.on_press(_FakeEvent(3, 3))
    assert ctrl.on_move(_FakeEvent(16, 3)) and ctrl.on_move(_FakeEvent(16, 16))
    assert ctrl.on_release(_FakeEvent(16, 16))
    assert len(app.store.interactions) == n0 + 1
    it_l = app.store.interactions[-1]
    assert it_l.tool == "polygon" and it_l.meta == {"extent": True}, it_l.meta
    assert app._interaction_row_text(it_l).endswith(" ext")
    app._undo()
    assert ctrl.on_press(_FakeEvent(3, 3, state=0x0004))        # Ctrl: a sample
    assert ctrl.on_move(_FakeEvent(16, 3)) and ctrl.on_move(_FakeEvent(16, 16))
    assert ctrl.on_release(_FakeEvent(16, 16))
    assert app.store.interactions[-1].tool == "polygon" and app.store.interactions[-1].meta is None
    app._undo()
    assert len(app.store.interactions) == n0
    app.tool_var.set("magic")

    # proba: refused without predictions or with stale ones; with a fake
    # prediction the seed's class joins first, unscored regions last.
    app.magic_metric_var.set("proba")
    app._pred = {}
    assert not ctrl.on_press(_FakeEvent(5, 5))
    assert "Classify first" in app.status_var.get(), app.status_var.get()
    proba = np.zeros((10, MAX_CLASSES), np.float32)
    proba[0, 1] = 1.0; proba[2, 1] = 1.0; proba[5, 2] = 1.0      # 9 unscored
    rc_p = np.zeros(10, np.uint8)
    app._pred[app.catalogue.key_of(0, 0)] = (app._commit_id + 1, rc_p, proba)          # stale
    assert not ctrl.on_press(_FakeEvent(5, 5))
    app._pred[app.catalogue.key_of(0, 0)] = (app._commit_id, rc_p, proba)
    assert ctrl.on_press(_FakeEvent(5, 5))
    lad_p = ctrl.magic._s["ladder"]
    assert lad_p.ids[lad_p.order].tolist() == [0, 2, 5, 9], lad_p.ids[lad_p.order].tolist()
    assert np.allclose(lad_p.join, [0.0, 0.0, 1.0, 1.0]), lad_p.join
    assert ctrl.magic.cancel()
    app._pred = {}
    app.magic_metric_var.set("mean")
    app.magic_gain_var.set(f"{_DEFAULT_HOP_GAIN:g}")
    app.magic_drag_var.set(f"{_DEFAULT_DRAG_PX:g}")
    app.tool_var.set("blobber")
    # Two-class store: no ring class exists, the press is refused.
    saved_n = app.store.n_classes
    app.store.set_n_classes(2); app.n_classes_var.set(2)
    assert not ctrl.on_press(_FakeEvent(5, 5))
    assert "second class" in app.status_var.get()
    app.store.set_n_classes(saved_n); app.n_classes_var.set(saved_n)
    app.active_class_var.set(0)
    app.tool_var.set("squiggle")

    # Rows of the sequence tree: the context menu's entries, a removal that
    # asks first and takes the row's annotations and primed data with it
    # (indices shift, keys do not), undo bringing the annotations back, and
    # "clear annotations" per row and per class.
    from unittest import mock as _mock
    fx = list(app.subsequences[0]["files"])        # the fixture as it stands now
    dx = os.path.dirname(fx[0])
    rec_cur = app.engine.record(0, 0)
    assert rec_cur is not None, "the fixture record is stale"
    assert app.primed[0]["files"] == fx, "the fixture is not primed as listed"
    n_on = len(app.store.for_slice("data/s0.tiff"))
    n_all = len(app.store.interactions)
    assert n_on >= 2
    assert app._seq_tree_row("q1:0") == (1, 0) and app._seq_tree_row("q1") == (1, None)
    assert app._seq_tree_row("x1") is None and app._seq_tree_row("") is None
    assert app._seq_tree_context(_FakeEvent(0, -1)) is None, "no row under the pointer: no menu"
    labels = [e[0] if e else None for e in app._seq_tree_menu_entries(0, 0)]
    assert labels == ["Go to", None, f"Clear annotations\u2026 ({n_on})", "Remove slice\u2026"], labels
    labels = [e[0] if e else None for e in app._seq_tree_menu_entries(0, None)]
    assert labels[-1] == "Remove sequence\u2026" and labels[0] == "Go to", labels
    assert app._row_owns_key(0, 0, "data/s0.tiff") and app._row_owns_key(0, None, "data/s0.tiff")
    assert not app._row_owns_key(0, 0, "data/other.tiff") and not app._row_owns_key(0, 0, None)
    # A second sequence holding the SAME slice: the key is the slice, so the
    # annotations belong to both and go only when the last one does.
    app.subsequences.append({"name": "seq2", "folder": "data", "files": list(fx)})
    app.primed.append({"files": list(fx), "base": [zeros], "filtered": [zeros],
                       "pipes": [None], "normalizers": [[]]})
    app._slices[(1, 0)] = dict(rec_cur)
    app._rebuild_flat_slices(); app._refresh_subseq_list(); app._goto_slice(0)
    assert app._doomed_interactions([(1, None)]) == [], "the slice is still in seq1"
    assert "No annotations" in app._remove_rows_message([(1, None)])
    assert len(app._doomed_interactions([(0, None), (1, None)])) == n_on
    assert f"The {n_on} annotation(s)" in app._remove_rows_message([(0, None), (1, None)])
    assert "slice 's0.tiff' of sequence" in app._remove_rows_message([(0, 0)])
    with _mock.patch.object(messagebox, "askyesno", return_value=False):
        assert not app._remove_rows_guarded([(1, None)]), "declined: nothing happens"
    assert len(app.subsequences) == 2 and len(app.primed) == 2
    app.engine.asm_running = True
    with _mock.patch.object(messagebox, "askyesno",
                            side_effect=AssertionError("must not ask while busy")):
        assert not app._remove_rows_guarded([(1, None)])
    assert "Busy" in app.status_var.get()
    app.engine.asm_running = False
    with _mock.patch.object(messagebox, "askyesno", return_value=True):
        assert app._remove_rows_guarded([(1, None)])
    assert len(app.subsequences) == 1 and len(app.primed) == 1
    assert app.flat_slices == [(0, 0)] and (1, 0) not in app._slices
    assert len(app.store.interactions) == n_all, "shared slice: the annotations stayed"
    # The item on screen survives a removal in FRONT of it: its record, its
    # annotations' hints and the navigation all shift down with it.
    z = [os.path.join(dx, "z.tiff")]
    app.subsequences.insert(0, {"name": "seq0", "folder": "data", "files": z})
    app.primed.insert(0, {"files": z, "base": [zeros], "filtered": [zeros],
                          "pipes": [None], "normalizers": [[]]})
    app._slices = {(1, 0): rec_cur}
    app._rebuild_flat_slices(); app._goto_slice(1)
    app.store.rebind(app.subsequences)
    assert app._current() == (1, 0) and app.store.for_slice("data/s0.tiff")[0].si == 1
    assert app._slice_msc_mark(0, 0) == "Y"
    assert app._remove_rows([(0, None)]) == 1
    assert app._current() == (0, 0) and app.subsequences[0]["name"] == "seq1"
    assert app.primed[0]["files"] == fx and app._slices == {(0, 0): rec_cur}, \
        "the primed entry and the record moved with their sequence"
    assert all(it.si == 0 for it in app.store.for_slice("data/s0.tiff")), "hints rebound"
    assert len(app.store.interactions) == n_all
    # A slice of a primed sequence: its raster and record go, the rest shift.
    s1 = os.path.join(dx, "s1.tiff")
    two = fx + [s1]
    app.subsequences[0]["files"] = list(two)       # a copy: the removal deletes from it
    app.primed[0] = {"files": list(two), "base": [zeros, zeros], "filtered": [zeros, zeros],
                     "pipes": [None, None], "normalizers": [[], []]}
    app._slices = {(0, 0): {"commit": app._commit_id, "labels": None},
                   (0, 1): rec_cur}
    app.engine.assembly[0] = {"_commit": app._commit_id}
    app._rebuild_flat_slices(); app._goto_slice(1)
    assert app._remove_rows([(0, 0)]) == 1
    assert app.subsequences[0]["files"] == [s1] and app.primed[0]["files"] == [s1]
    assert len(app.primed[0]["pipes"]) == 1 and app._slices == {(0, 0): rec_cur}
    assert 0 not in app.engine.assembly, "the 3D assembly spanned the removed slice"
    assert app._current() == (0, 0) and app._slice_msc_mark(0, 0) == "Y"
    app.subsequences[0]["files"] = list(fx)     # a copy: the removal deletes from it
    app.primed[0]["files"] = list(fx)
    app._rebuild_flat_slices(); app._goto_slice(0)
    app.store.rebind(app.subsequences)
    # Removing the slice itself takes its annotations; undo brings them back,
    # greyed, since the slice is gone.
    with _mock.patch.object(messagebox, "askyesno", return_value=True):
        assert app._remove_rows_guarded([(0, 0)])
    assert app.subsequences == [] and app.primed == [] and app.flat_slices == []
    assert len(app.store.interactions) == n_all - n_on
    assert "annotation(s)" in app.status_var.get()
    app._undo()
    assert len(app.store.interactions) == n_all
    assert not any(it.bound for it in app.store.for_slice("data/s0.tiff"))
    # The fixture, back.
    app.subsequences = [{"name": "seq1", "folder": "data", "files": fx}]
    app.primed = [{"files": fx, "base": [zeros], "filtered": [zeros],
                   "pipes": [None], "normalizers": [[]]}]
    app._slices = {(0, 0): rec_cur}
    app._rebuild_flat_slices(); app._goto_slice(0)
    app.store.rebind(app.subsequences)
    app._rebuild_class_panels()
    assert all(it.bound for it in app.store.for_slice("data/s0.tiff"))
    # Clear per class (every item) and per row: guarded, one undo step each,
    # and the tree's annot column follows.
    app.active_class_var.set(1); app._commit_interaction("taps", [(5.0, 5.0)])
    app.active_class_var.set(2); app._commit_interaction("taps", [(15.0, 15.0)])
    app.active_class_var.set(0)
    n_on = len(app.store.for_slice("data/s0.tiff"))
    n_all = len(app.store.interactions)
    k1, k2 = len(app.store.for_class(1)), len(app.store.for_class(2))
    assert k1 and k2
    assert set(app._class_clear_buttons) == set(range(1, app.store.n_classes))
    with _mock.patch.object(messagebox, "askyesno", return_value=False):
        assert not app._clear_class_guarded(1)
    assert len(app.store.for_class(1)) == k1
    with _mock.patch.object(messagebox, "askyesno", return_value=True):
        assert app._clear_class_guarded(1)
    assert not app.store.for_class(1) and len(app.store.for_class(2)) == k2
    assert not app._clear_class_guarded(1) and "no annotations" in app.status_var.get()
    app._undo()
    assert len(app.store.for_class(1)) == k1 and len(app.store.interactions) == n_all
    assert app.subseq_list.item("q0:0", "values")[1] == str(n_on)
    with _mock.patch.object(messagebox, "askyesno", return_value=True):
        assert app._clear_row_annotations_guarded(0, 0)
    assert len(app.store.interactions) == n_all - n_on
    assert app.subseq_list.item("q0:0", "values")[1] == ""
    assert not app._clear_row_annotations_guarded(0, 0) and "No annotations" in app.status_var.get()
    app._undo()
    assert len(app.store.interactions) == n_all
    assert app.subseq_list.item("q0:0", "values")[1] == str(n_on)

    # Neighbourhood context (msseg.labeler.context): ring columns join every
    # row by name, so the model, the gate, the pickle and the session all see
    # them as ordinary columns -- and an empty spec leaves the fingerprint
    # exactly what it was.
    from msseg.labeler import context as _cx
    app._expected_feature_names = lambda: ["area", "mean_base"]
    app.model_kind_var.set("dense FC")
    ctx_spec = _cx.ContextSpec(kinds=("ring_mean", "ring_contrast"),
                               weights=("uniform", "contact"))
    app._apply_context_spec(ctx_spec)
    assert app._context_spec_from_ui() == ctx_spec
    assert "ring(mean,contrast)" in app.model_arch_var.get()
    app._train_classifier()
    assert app._clf_names[:2] == ["area", "mean_base"], app._clf_names
    assert app._clf_names[2:] == _cx.column_names(ctx_spec, ["area", "mean_base"]), \
        app._clf_names
    assert "ring_mean[contact]__mean_base" in app._clf_names
    assert app._clf_context == ctx_spec, "the spec the model was fit under rides with it"
    assert "ctx: ring(mean,contrast) [uniform,contact]" in app.model_strip_var.get()
    assert "mismatch" not in app.model_strip_var.get(), \
        "the gate expects the model's own context columns"
    assert app._check_model_compat(app._clf_names, "t") is None
    arcs_now = app.regions.arcs("data/s0.tiff", np)
    assert arcs_now.get("length") is not None, "contact lengths derived once and cached"
    app._classify()
    assert app._pred and len(next(iter(app._pred.values()))) == 3, "plain entries"
    # The pickle carries the spec (only when there is one) and a load restores
    # it -- onto the model AND the Model tab -- after the gate has passed.
    ctx_td = tempfile.mkdtemp()
    ctx_path = os.path.join(ctx_td, "ctx.pkl")
    app._save_classifier_to(ctx_path)
    assert app.models[-1]["context"] == ctx_spec.to_dict()
    assert model_bundle.ModelBundle.load(ctx_path).stack["context"] == ctx_spec.to_dict()
    app._apply_context_spec(_cx.ContextSpec())               # the tab moves off
    assert "context changed" in app.model_arch_var.get()
    app._load_classifier_from(ctx_path)
    assert app._clf_context == ctx_spec and app._context_spec_from_ui() == ctx_spec
    assert app.context_kind_vars["ring_mean"].get() and app.context_weight_vars["contact"].get()
    app._classify()
    assert app._pred, "a reloaded context model classifies"
    # A different spec on the tab is the NEXT model, not this one: the
    # readout says so and the gate still judges the loaded model by its own.
    app.context_kind_vars["ring_std"].set(True); app._on_context_settings_change()
    assert "context changed" in app.model_arch_var.get()
    assert app._check_model_compat(app._clf_names, "t") is None
    assert app._task_view_from_ui()["context"]["kinds"] == ["ring_mean", "ring_std", "ring_contrast"], \
        "the picked context rides the session view, kinds in canonical order"
    # A context-free pickle refuses the fingerprint of a context model, and
    # vice versa: the columns are part of the schema.
    assert app._check_model_compat(["area", "mean_base"], "t", ctx=ctx_spec) is not None
    assert app._check_model_compat(app._clf_names, "t", ctx=_cx.ContextSpec()) is not None
    # Back to a plain model: the fingerprint is exactly the old one and the
    # pickle document has no context key at all.
    app._apply_context_spec(_cx.ContextSpec())
    app._train_classifier()
    assert app._clf_names == ["area", "mean_base"] and app._clf_context.empty()
    plain_path = os.path.join(ctx_td, "plain.pkl")
    app._save_classifier_to(plain_path)
    assert "context" not in model_bundle.ModelBundle.load(plain_path).stack
    assert "context" not in app.models[-1]
    # The latent-ring head: a second net on the base's embedding of the ring,
    # fit after the base (a dense net embeds; a forest cannot). It makes the
    # prediction, adds nothing to the fingerprint, rides the pickle as
    # stack["latent"] and comes back with the base it was fit on.
    lat_spec = _cx.ContextSpec(latent=_cx.LatentSpec(weight="latent", h0=True))
    app._apply_context_spec(lat_spec)
    assert app.context_latent_var.get() and app._context_spec_from_ui() == lat_spec
    assert "context changed" in app.model_arch_var.get()
    app._train_classifier()
    lat = app._context_model
    assert lat is not None, app.status_var.get()
    assert app._clf_names == ["area", "mean_base"], "latent columns are not in the fingerprint"
    assert lat.names[:2] == ["area", "mean_base"] and lat.names[-4:] == list(_cx.H0_NAMES)
    assert lat.width == _MLP_HIDDEN[-1] and lat.n_in == 2 + lat.width + 4
    assert "latent(mean[latent],h0)" in app.status_var.get(), app.status_var.get()
    assert "latent head on" in app.model_arch_var.get()
    assert app._check_model_compat(app._clf_names, "t") is None
    app._classify()
    assert app._pred and len(next(iter(app._pred.values()))) == 3
    lat_path = os.path.join(ctx_td, "latent.pkl")
    app._save_classifier_to(lat_path)
    assert "latent" in model_bundle.ModelBundle.load(lat_path).stack
    app._context_model = None
    app._load_classifier_from(lat_path)
    assert app._context_model is not None and app._clf_context == lat_spec
    assert app._context_model.net_hash == lat.net_hash
    app._classify()
    assert app._pred, "a reloaded latent head classifies"
    # A forest has no embedding: the head is skipped with a note, the base predicts.
    app.model_kind_var.set("random forest")
    app._train_classifier()
    assert app._context_model is None and "latent head skipped" in app.status_var.get()
    app._classify()
    assert app._pred
    app.model_kind_var.set("dense FC")
    app._apply_context_spec(_cx.ContextSpec())
    app._train_classifier()
    assert app._context_model is None and app._clf_context.empty()
    # Labels as context: the ring's annotated classes are columns (the
    # region's own never is), the fingerprint grows by them, and predictions
    # are dropped the moment the store changes, since they depend on it.
    app._apply_context_spec(_cx.ContextSpec(labels=_cx.LabelSpec(dropout=0.0)))
    got_spec = app._context_spec_from_ui()
    assert got_spec.labels is not None and got_spec.labels.dropout == 0.0
    app._train_classifier()
    lab_names = _cx.column_names(got_spec, ["area", "mean_base"])
    assert app._clf_names == ["area", "mean_base"] + lab_names and lab_names[-1] == "nbr_class__any"
    assert "labels(p=0)" in app.model_strip_var.get()
    assert app._check_model_compat(app._clf_names, "t") is None
    app._classify()
    assert app._pred and app._pred_store_rev == app.store.rev
    rev0 = app.store.rev
    app.active_class_var.set(1); app._commit_interaction("taps", [(5.0, 5.0)])
    app.active_class_var.set(0)
    assert app.store.rev != rev0 and not app._pred, "annotating drops label-context predictions"
    app._undo()
    assert not app._pred
    app._classify()
    assert app._pred, "Classify rebuilds them from the current annotations"
    app._apply_context_spec(_cx.ContextSpec())
    app._train_classifier()
    assert app._clf_names == ["area", "mean_base"] and app._clf_context.empty()
    app._classify()
    app.active_class_var.set(1); app._commit_interaction("taps", [(5.0, 5.0)])
    app.active_class_var.set(0)
    assert app._pred, "without labels in the context, annotating keeps the predictions"
    app._undo()

    # A re-prime (engine "done") bumps the commit, so every commit-keyed cache
    # (per-slice records, class LUTs, predictions) self-invalidates -- the
    # "stale overlays after adding a folder and re-running" regression.
    c0 = app._commit_id
    app.engine.work_q.put(("done", []))
    app.engine.poll()
    assert app._commit_id != c0, "re-prime must move the commit"

    # -- Seam tools: scope box, livewire trace, resolution, overlay, session -- #
    # A fresh one-slice fixture (the four sparse-id blocks) at the current
    # commit; the seam graph comes from the compiled extension when present,
    # else the numpy reference, through the provider.
    from msseg.labeler.seams import SEAM_BOUNDARY, SEAM_INTERIOR
    from .common import FeatureTable as _FT
    lab_s = np.full((20, 20), -1, np.int32)
    lab_s[2:10, 2:10] = 0; lab_s[2:10, 10:18] = 2
    lab_s[10:18, 2:10] = 5; lab_s[10:18, 10:18] = 9
    ids_s = np.array([0, 2, 5, 9], np.float64)
    table_s = _FT(["feature_id", "area", "mean_base", "std_base", "ext_filtered",
                   "ext_x", "ext_y"],
                  np.stack([ids_s, np.full(4, 64.0), np.array([0.0, 1.0, 5.0, 6.0]),
                            np.ones(4), np.array([0.0, 1.0, 5.0, 6.0]),
                            np.array([5.0, 13.0, 5.0, 13.0]),
                            np.array([5.0, 5.0, 13.0, 13.0])], axis=1))
    rec_s = {"commit": app._commit_id, "labels": lab_s, "stats": table_s,
             "kept": set(), "cc": None, "n_feat": 4}
    app._slices[(0, 0)] = rec_s
    for _sl in app.flat_slices:           # every listed slice has a record
        app._slices.setdefault(tuple(_sl), dict(rec_s))
    app._pred = {}
    app._clear_seam_caches()
    # Seams are a POLYLINE task's: its own kind, on the same workflow; the
    # tabs, tools and keys follow the kind.
    t_region = app._task
    t_walls = app._task_new("walls", kind="polyline")
    assert app._task is t_walls and app._task_kind() == "polyline"
    assert app.tool_var.get() == "trace", "a polyline task offers trace / scope"
    assert app.seam_frame.winfo_manager() == "pack" and not app.annot_frame.winfo_manager()
    assert not app.region_ml_frame.winfo_manager()
    assert app._model_polyline.winfo_manager() and not app._model_region.winfo_manager()
    app.tool_var.set("magic")
    assert app.tool_var.get() == "trace", "a region tool is refused in a polyline task"
    assert app._for_kind("region")(lambda e: "ran")() is None
    assert app.task_tree.set(t_walls.uid, "kind") == app._KIND_GLYPH["polyline"]
    key_s = app.catalogue.key_of(0, 0)
    v = app.viewer
    v.view_x, v.view_y, v.scale = 0.0, 0.0, 1.0
    ctrl = v.tool
    tc = ctrl.trace
    g_s = app.regions.seams(key_s, np)
    assert g_s is not None and g_s.n_seams == 4 and rec_s.get("_seams") is g_s
    assert app.regions.seams(key_s, np) is g_s, "the seam graph is cached on the record"
    n_seams0 = len(app.store.seams)
    from msseg.viz import min_colors as _mc_s

    class _KeyEv:            # a toplevel key event whose widget is the canvas
        widget = v.canvas

    # Scope: a box over everything -> every seam interior, previewed on the
    # transient layer with the HUD saying how many.
    app.tool_var.set("scope"); app.seam_class_var.set(SEAM_INTERIOR)
    assert ctrl.on_press(_FakeEvent(1, 1)) and tc.active
    assert ctrl.on_move(_FakeEvent(19, 19))
    assert v._transient is not None and v._hud_mode == "info", (v._hud_mode, v._hud_text)
    assert "scope: 4 seams" in v._hud_text, v._hud_text
    assert ctrl.on_release(_FakeEvent(19, 19))
    assert v._transient is None and not tc.active and v._hud_mode is None
    assert len(app.store.seams) == n_seams0 + 1 and app.store.seams[-1].tool == "scope"
    assert app.store.seams[-1].meta["seams"] == 4
    _g, cls_s = app._seam_classes_for(0, 0, np)
    assert (cls_s == SEAM_INTERIOR).all()
    ovs_s = app._seg_overlays(0, 0, rec_s, None, np, _mc_s)
    seam_ov = ovs_s[-1]
    assert int(seam_ov["lut"][:, 3].max()) == 255 and seam_ov["labels"].shape == (20, 20)
    assert seam_ov["labels"][5, 9] >= 0 and seam_ov["labels"][5, 10] >= 0, "both flanks lit"
    assert seam_ov["labels"][5, 5] < 0

    # Trace: a press near the 0|2 seam (x = 10) anchors; hovering shows the
    # path; a click at the centre junction and one at the right edge add
    # legs; Enter commits ONE trace in the boundary class.
    app.tool_var.set("trace"); app.seam_class_var.set(SEAM_BOUNDARY)
    assert app.seam_toll_var.get() == "feature"
    assert ctrl.on_press(_FakeEvent(10, 3)) and tc.active
    assert v._hud_mode == "info" and "trace feature" in v._hud_text, v._hud_text
    tc.on_hover(10, 9)
    assert tc._s["hover"] is not None and tc._s["hover"][1] is not None
    assert len(v.canvas.find_withtag("trace")) >= 2, "the hover leg and the anchor are drawn"
    assert ctrl.on_press(_FakeEvent(10, 10)) and len(tc._s["lw"].legs) == 1
    assert ctrl.on_press(_FakeEvent(18, 10)) and len(tc._s["lw"].legs) == 2
    assert app._on_trace_commit_key(_KeyEv()) == "break"
    assert not tc.active and v._transient is None and v._hud_mode is None
    assert len(app.store.seams) == n_seams0 + 2
    tr_s = app.store.seams[-1]
    assert tr_s.tool == "trace" and tr_s.class_id == SEAM_BOUNDARY
    assert tr_s.points == [(10.0, 3.0), (10.0, 10.0), (18.0, 10.0)], tr_s.points
    assert tr_s.meta["seams"] == 2 and tr_s.meta["anchors"] == 3
    assert tr_s.meta["toll"] == "feature"
    assert tr_s.meta["scoped"], "anchored inside the scope that covers everything"
    _g, cls_s = app._seam_classes_for(0, 0, np)
    idx_s = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(_g.a, _g.b))}
    assert cls_s[idx_s[(0, 2)]] == SEAM_BOUNDARY and cls_s[idx_s[(2, 9)]] == SEAM_BOUNDARY
    assert cls_s[idx_s[(0, 5)]] == SEAM_INTERIOR and cls_s[idx_s[(5, 9)]] == SEAM_INTERIOR
    assert "2 boundary, 2 interior" in app.seam_readout_var.get(), app.seam_readout_var.get()
    assert len(app._seam_rows) == 2, "the panel lists this slice's seam gestures"
    # Undo / redo take the whole trace.
    app._undo()
    assert len(app.store.seams) == n_seams0 + 1
    app._redo()
    assert len(app.store.seams) == n_seams0 + 2
    # A trace anchored inside a scope is confined to it (the scope covers
    # everything here, so the search is merely flagged as scoped).
    assert ctrl.on_press(_FakeEvent(10, 3)) and tc._s["restricted"]
    # BackSpace drops a leg, Escape abandons; a press far from any seam and a
    # toll without its input are refused (fall through to the pan).
    assert ctrl.on_press(_FakeEvent(10, 10)) and len(tc._s["lw"].legs) == 1
    assert app._on_trace_back_key(_KeyEv()) == "break" and len(tc._s["lw"].legs) == 0
    app._on_escape()
    assert not tc.active and v._transient is None
    assert not ctrl.on_press(_FakeEvent(0, 0)), "no seam within reach"
    app.seam_toll_var.set("model")
    assert not ctrl.on_press(_FakeEvent(10, 3)) and "seam model" in app.status_var.get()
    app.seam_toll_var.set("feature")
    assert app._on_trace_commit_key(_KeyEv()) is None, "Enter with no trace in flight passes"
    # The session document carries the seams (v3) and the view state the toll.
    doc_s = app._session_doc()
    walls_doc = next(t for t in doc_s["tasks"] if t["uid"] == t_walls.uid)
    assert walls_doc["kind"] == "polyline"
    ann_s = walls_doc["annotations"]
    assert ann_s["version"] == 3 and len(ann_s["seams"]) == 2
    assert doc_s["view"]["seams"]["toll"] == "feature" and doc_s["view"]["tool"] == "trace"
    app.seam_toll_var.set("barrier"); app.show_seams_var.set(False)
    app._apply_seams_view(doc_s["view"]["seams"])
    assert app.seam_toll_var.get() == "feature" and app.show_seams_var.get()
    # The seam model: fit on the two boundary + two interior seams, every
    # seam scored, the 'model' toll then accepted, the boundaryness colouring
    # shown, and the export written.
    try:
        import sklearn  # noqa: F401
    except ImportError:
        sklearn = None
    if sklearn is not None:
        app._train_seam_model()
        assert app._seam_model is not None, app.status_var.get()
        assert key_s in app._seam_pred and len(app._seam_pred[key_s][1]) == 4
        app.seam_toll_var.set("model")
        assert ctrl.on_press(_FakeEvent(10, 3)) and tc.active
        app._on_escape()
        app.seam_toll_var.set("feature")
        app.seam_color_var.set(_SEAM_MODE_BOUNDARYNESS)
        ovs_b = app._seg_overlays(0, 0, rec_s, None, np, _mc_s)
        assert ovs_b[-1]["lut"].shape == (4, 4) and int(ovs_b[-1]["lut"][:, 3].max()) > 0
        app.seam_color_var.set(_SEAM_MODE_CLASS)
        with tempfile.TemporaryDirectory() as td_s:
            n_it, n_sm = app._export_seams_to(td_s)
            assert n_it >= 1 and n_sm >= 4, (n_it, n_sm)
            assert os.path.isfile(os.path.join(td_s, "seams_summary.csv"))
            assert any(f.startswith("seams_") and f.endswith(".json") for f in os.listdir(td_s))
        assert "logistic" in app.seam_readout_var.get(), app.seam_readout_var.get()
    # Row helpers see seam gestures too.
    assert len(app._row_interactions(0, 0)) >= 2
    assert app._activate_task(t_region) and app._task_kind() == "region"
    assert app.tool_var.get() == "squiggle", "back in a region task: its first tool"
    assert app.annot_frame.winfo_manager() == "pack" and not app.seam_frame.winfo_manager()

    # -- The outline: a closed livewire loop fills what it encloses ---------- #
    # A 3x3 grid of blocks (ids 0..8) so a loop of seams exists: the centre
    # block 4 is ringed by the seams 1|4, 4|5, 4|7 and 3|4, with junctions
    # at the four corners. (The earlier scope + trace go first: a scope
    # containing the anchor would confine the search.)
    from msseg.labeler import derive as _derive
    app.store.remove_many([s.uid for s in app.store.seams])
    lab_e = np.full((32, 32), -1, np.int32)
    for r_ in range(3):
        for c_ in range(3):
            lab_e[1 + 10 * r_:11 + 10 * r_, 1 + 10 * c_:11 + 10 * c_] = 3 * r_ + c_
    ids_e = np.arange(9, dtype=np.float64)
    table_e = _FT(["feature_id", "area", "mean_base", "std_base", "ext_filtered",
                   "ext_x", "ext_y"],
                  np.stack([ids_e, np.full(9, 100.0), ids_e, np.ones(9), ids_e,
                            np.array([6 + 10 * (i % 3) for i in range(9)], float),
                            np.array([6 + 10 * (i // 3) for i in range(9)], float)], axis=1))
    rec_e = {"commit": app._commit_id, "labels": lab_e, "stats": table_e,
             "kept": set(), "cc": None, "n_feat": 9}
    app._slices[(0, 0)] = rec_e
    app._pred = {}
    app._clear_seam_caches(); app._class_luts.clear()
    g_e = app.regions.seams(key_s, np)
    assert g_e is not None and g_e.n_seams == 12, g_e.n_seams
    n_i0, n_s0 = len(app.store.interactions), len(app.store.seams)
    app.tool_var.set("outline")
    assert app.tool_var.get() == "outline"
    app.seam_toll_var.set("geometric")

    def _loop():
        assert ctrl.on_press(_FakeEvent(16, 11)) and tc.active              # on 1|4 (y = 11)
        assert ctrl.on_press(_FakeEvent(21, 16)) and len(tc._s["lw"].legs) == 1   # 4|5
        assert ctrl.on_press(_FakeEvent(16, 21)) and len(tc._s["lw"].legs) == 2   # 4|7
        assert ctrl.on_press(_FakeEvent(11, 16)) and len(tc._s["lw"].legs) == 3   # 3|4
        assert ctrl.on_press(_FakeEvent(16, 11))            # the first anchor again: closed
    # No class armed: the outline does not start.
    app.active_class_var.set(0)
    assert not ctrl.on_press(_FakeEvent(16, 11)) and not tc.active
    assert "Arm a class" in app.status_var.get(), app.status_var.get()
    # With class 1: the closed loop fills its one enclosed region at once.
    app.active_class_var.set(1)
    _loop()
    assert not tc.active and v._hud_mode is None
    assert len(app.store.seams) == n_s0, "an outline stores no seam gesture"
    assert len(app.store.interactions) == n_i0 + 1
    enc = app.store.interactions[-1]
    assert enc.tool == "taps" and enc.class_id == 1 and len(enc.points) == 1
    assert enc.meta["tool"] == "outline" and enc.meta["extent"] is True
    assert enc.meta["n_regions"] == 1 and enc.meta["anchors"] == 5
    assert enc.meta.get("outline"), "the loop rides along as the outline"
    assert labeling.resolve_slice([enc], lab_e, np)[4] == 1
    assert "outline (1) ext" in app._interaction_row_text(enc)
    app._undo()
    assert len(app.store.interactions) == n_i0, "one gesture, one undo step"
    app._redo()
    enc = app.store.interactions[-1]
    _g2, cls_e = app._seam_classes_for(0, 0, np)
    # No seam gesture was stored, so every seam -- the ring included -- is
    # what the derivation makes of the extent and whatever region gestures
    # earlier sections left on the slice.
    _rc, sl_e, _bo, _di = _derive.labels_for_item(
        app._gestures_for_key(key_s), app._seam_gestures_for_key(key_s), lab_e, _g2, None, np)
    assert cls_e.tolist() == sl_e.cls.tolist()
    # An open outline does not commit: Enter says so and keeps it in flight;
    # Escape abandons it with the class still armed.
    assert ctrl.on_press(_FakeEvent(16, 11)) and ctrl.on_press(_FakeEvent(21, 16))
    assert app._on_trace_commit_key(_KeyEv()) == "break" and tc.active
    assert "closes on its first anchor" in app.status_var.get(), app.status_var.get()
    app._on_escape()
    assert not tc.active and app.active_class_var.get() == 1
    assert len(app.store.seams) == n_s0 and len(app.store.interactions) == n_i0 + 1
    # Back to the four-block fixture, without the gestures of this block.
    app.store.remove_many([s.uid for s in app.store.seams] + [enc.uid])
    app._slices[(0, 0)] = rec_s
    app._pred = {}
    app._clear_seam_caches(); app._class_luts.clear()
    app.seam_toll_var.set("feature"); app.tool_var.set("squiggle")
    app._rebuild_class_panels()

    # New session: data, sequences, primed stacks and annotations go; the
    # profiles and the model selection stay when asked to, and the model in
    # memory survives the apply (it is not in the document).
    app.model_kind_var.set(_CUSTOM_EDGE_KIND)
    app.custom_hidden_var.set("4")
    prof_names = [p["name"] for p in app.profiles]
    n_cls = app.store.n_classes
    app.store.add("squiggle", [(3.0, 3.0)], 1, "data/s0.tiff")
    assert app.store.interactions
    clf_keep = app._clf
    # ...and the tasks stay, emptied: a second one with its own vocabulary.
    task_names = [t.name for t in app.tasks] + ["bubbles"]
    tb = app._task_new("bubbles")
    assert tb is app._task and app._clf is None
    app._rename_class(1, "bubble")
    app._push_history()
    app.store.add("box", [(1.0, 1.0), (5.0, 5.0)], 1, "data/s0.tiff")
    assert app._activate_task(app.tasks[0]) and app._clf is clf_keep
    assert [o[0] for o in app._new_session_options()] == ["profiles", "model"]
    assert app._new_session(keep={"profiles": True, "model": True})
    assert app.folders == [] and app.subsequences == [] and not app.primed
    assert not app.flat_slices and not app._pred
    assert app.store.interactions == [] and app.store.n_classes == n_cls
    assert app.store.seams == [], "a new session drops the seam gestures too"
    assert [t.name for t in app.tasks] == task_names and app._task is app.tasks[0]
    assert all(t.gesture_count == 0 and t.undo == [] for t in app.tasks), "every store emptied"
    assert app.tasks[-1].store.name(1) == "bubble", "vocabularies stay"
    assert [p["name"] for p in app.profiles] == prof_names
    assert app.model_kind_var.get() == _CUSTOM_EDGE_KIND
    assert app.custom_hidden_var.get() == "4"
    assert app._clf is clf_keep, "the in-memory model survives"
    assert app._session_owned and "kept" in app.status_var.get()
    assert app._new_session(keep={"profiles": False, "model": False})
    assert len(app.profiles) == 1 and app.profiles[0]["name"] == "default"
    assert [t.name for t in app.tasks] == task_names, "tasks survive; their models do not"
    assert all(t.model.empty and t.models == [] and t.workflow == "default" for t in app.tasks)
    assert app._clf is None and app._edge_model is None and app.models == []
    assert app.model_kind_var.get() == "dense FC"
    assert app.custom_hidden_var.get() == _DEFAULT_CUSTOM_HIDDEN
    assert str(app.classify_btn.cget("state")) == "disabled"
    assert "nothing kept" in app.status_var.get()

    # The labeler's session file never collides with the viewer's.
    assert config_io.session_path(app=app.SESSION_APP) != config_io.session_path()

    root.destroy()
    print("labeler selftest OK: tiers, gestures, resolution order, LUT cache, "
          "overlay stack, class-count clamp, session round-trip (qualified "
          "keys), export, undo/redo, tap-squiggle, hover geometry, "
          "click-to-center, CSV, model kinds + compat gate + 'R', "
          "profile-from-model + provenance strip, swatch arming, "
          "proba cache + coloring modes, confusion matrix + highlight, "
          "persistent outlines + canvas right-click, gesture previews, "
          "magic fill, blobber, hop gain + drag + cosine/proba metrics, "
          "center notebook + model tab, toolbar hints, optimize network, size sweep, "
          "edge kinds, analysis tab + region list, panel diffing (no rebuild "
          "on a commit), tree context menu + guarded row removal + clear per "
          "row / per class, context columns + their pickle/gate/session ride, "
          "saddle-free contact edges, latent ring head, labels as context, "
          "seam tools: scope + livewire trace + resolution + overlay + seam model + export, "
          "tasks: per-task stores / vocabularies / Model tabs, session v3 + v2-as-one-task, "
          "workflow binding through profile rename / delete, New session keeps tasks, "
          "extents: fill / blob core / lasso (Ctrl = sample) + the checkbox in the view, "
          "derived seam labels with no seam gesture, the outline -> enclosure in one step, "
          "region / polyline task kinds (tabs, tools, keys; older documents refused)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
