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

    def _image_window(self, _channel):
        """One fractional window follows whichever image channel is active."""
        return self.vmin_var.get(), self.vmax_var.get()

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

    def _profile_from_model(self, path, statistics):
        """Append a profile that keeps the active one's filters/MSC/selection
        but MEASURES what the model was trained on, and activate it.

        A new profile rather than an edit of the active one: profiles have no
        undo stack, and _switch_profile already drops the primed data -- which
        a statistics change needs, since the per-slice feature table is baked
        at prime time (a selection rerun would not rebuild it)."""
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
                    user = resolve_slice(self.store.for_slice(key), labels, np)
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
                rc = resolve_slice(self.store.for_slice(key), labels, np)
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

    # Window layout: a tabbed center. The left pane keeps the profile PICKER,
    # the session and Run; the profile is EDITED on the Processing tab (two
    # columns); the View tab is `app.right`; the model kind lives on the Model
    # tab and is gone from the classifier panel.
    def _under(w, ancestor):
        while w is not None:
            if w is ancestor:
                return True
            w = getattr(w, "master", None)
        return False

    tabs = [str(app.center.tab(t, "text")) for t in app.center.tabs()]
    assert tabs == list(_CENTER_TABS) == ["Processing", "View", "Model", "Analysis"], tabs
    assert app.center.nametowidget(app.center.select()) is app.right, \
        "the View tab is selected at start"
    assert app._center_tab_name() == "View"
    for w in (app.filters_frame, app.base_frame, app.msc_frame, app.stats_frame,
              app.profile_load_btn):
        assert _under(w, app.processing_tab), w
    assert app.filters_frame.master is app.base_frame.master is app.proc_col_a
    assert app.msc_frame.master is app.stats_frame.master is app.proc_col_b
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
    app.center.select(app.right)
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
    assert app.model_hint.master is ml and ml.pack_slaves()[0] is app.model_hint
    train_row = ml.pack_slaves()[1]
    assert any(isinstance(w, ttk.Button) and str(w.cget("text")).startswith("Train")
               for w in train_row.winfo_children()), "Train/Classify right under the hint"

    # `app.right` IS the View tab, so the scan below reads unchanged.
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
    app.vmin_var.set(0.2); app.vmax_var.set(0.8)
    assert app._image_window("base") == (0.2, 0.8)
    assert app._image_window("filtered") == (0.2, 0.8)
    assert app._image_window("edges_s1") == (0.2, 0.8)
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
    app._commit_interaction("squiggle", [(3.0, 5.0), (15.0, 5.0)])  # {0,2} -> 2
    assert [it.uid for it in app.store.interactions] == [1, 2]
    assert app.store.interactions[0].slice_key == "data/s0.tiff", \
        "slice identity is folder-qualified"
    # The sequence tree's columns track priming + per-slice annotations.
    app._refresh_subseq_list()
    assert tuple(app.subseq_list.item("q0:0", "values")) == ("Y", "2")
    assert tuple(app.subseq_list.item("q0", "values")) == ("Y", "2")

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
    assert sdoc["view"]["model_kind"] == _CUSTOM_EDGE_KIND, "the picked kind rides the view"
    app.model_kind_var.set("dense FC")
    app.center.select(app.right)
    assert "labels" not in sdoc, "the gesture geometry is 'annotations' now"
    assert sdoc["annotations"]["n_classes"] == 2
    assert len(sdoc["annotations"]["interactions"]) == 2
    assert sdoc["sequences"][0]["folder"] == "data"
    app.store = LabelStore()             # clobber
    app._apply_session_doc(sdoc, "test")
    assert app._center_tab_name() == "Model", "the center tab restores by name"
    assert app.model_kind_var.get() == _CUSTOM_EDGE_KIND, "the picked kind restores"
    bad = json.loads(json.dumps(sdoc))
    bad["view"]["model_kind"] = "no such kind"
    app._apply_session_doc(bad, "test")
    assert app.model_kind_var.get() == _CUSTOM_EDGE_KIND, "an unknown kind is ignored"
    sdoc["view"]["model_kind"] = kind_pick        # later re-applies keep the default
    app.model_kind_var.set(kind_pick)
    app.center.select(app.right)
    assert len(app.store.interactions) == 2
    assert all(it.bound for it in app.store.interactions), \
        "rebind by folder-qualified key"

    # ...and a session written BEFORE the rename still restores: the key moved,
    # the document under it did not.
    legacy = {k: v for k, v in sdoc.items() if k != "annotations"}
    legacy["labels"] = sdoc["annotations"]
    legacy["view"] = dict(sdoc["view"], center_tab="bogus")   # unknown -> untouched
    app.store = LabelStore()
    app._apply_session_doc(legacy, "legacy key")
    assert app._center_tab_name() == "View", "an unknown tab name is ignored"
    assert len(app.store.interactions) == 2, \
        "a pre-rename session's 'labels' key still loads"

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
        assert len(ovs) == 3, "regions + prediction layer + drawn labels"
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
            assert len(ovs_m) == 3, mode
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
        # The stack settings ride the session view and are validated on the way in.
        v = app._view_state()["neighbours"]
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
        assert app._center_tab_name() == "View"
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
        assert len(ovs_h) == 4, "highlight rides on top"
        assert set(np.nonzero(ovs_h[-1]["lut"][:, 3])[0].tolist()) == hits
        app._on_confusion_click(*moved)                       # click again clears
        assert app._cm_cell is None
        assert len(app._seg_overlays(0, 0, rec3, None, np, _mc)) == 3

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
        assert app._view_state()["model_search"] == \
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
            assert sdoc2["models"] and sdoc2["models"][-1]["path"] == \
                os.path.abspath(clf_path)
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
                """The engine.reset() inside the profile switch drops the fake
                primed stack; put it back for the rest of the selftest."""
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
            with _mock.patch.object(messagebox, "askyesno", return_value=True):
                app._load_classifier_from(clf_path, interactive=True)
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
    assert len(it_m.points) == 4
    rc_m = labeling.resolve_slice([it_m], lab, np)
    assert all(rc_m[r] == 1 for r in (0, 2, 5, 9))
    # Provenance in the session doc; the options ride the view state; undo
    # removes the whole fill in one step.
    app._rebuild_class_panels()
    doc_m = app._session_doc()
    assert doc_m["view"]["magic"]["metric"] == "mean"
    assert any(d.get("meta", {}).get("tool") == "magic"
               for d in doc_m["annotations"]["interactions"])
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
    rc_b = labeling.resolve_slice([ring_it, core_it], lab, np)
    assert rc_b[0] == 1 and rc_b[2] == 2 and rc_b[5] == 2 and rc_b[9] == 0
    app._rebuild_class_panels()
    assert app._session_doc()["view"]["magic"]["ring"] == "next"
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
    # cosine runs on the whole row (here area + mean_base) without channels.
    app.magic_gain_var.set("1")
    assert ctrl.on_press(_FakeEvent(5, 5))
    assert ctrl.magic._s["ladder"].metric == "cosine"
    assert ctrl.magic.cancel()
    app.magic_mode_var.set("anchor")

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

    # A re-prime (engine "done") bumps the commit, so every commit-keyed cache
    # (per-slice records, class LUTs, predictions) self-invalidates -- the
    # "stale overlays after adding a folder and re-running" regression.
    c0 = app._commit_id
    app.engine.work_q.put(("done", []))
    app.engine.poll()
    assert app._commit_id == c0 + 1, "re-prime must bump the commit"

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
    assert [o[0] for o in app._new_session_options()] == ["profiles", "model"]
    assert app._new_session(keep={"profiles": True, "model": True})
    assert app.folders == [] and app.subsequences == [] and not app.primed
    assert not app.flat_slices and not app._pred
    assert app.store.interactions == [] and app.store.n_classes == n_cls
    assert [p["name"] for p in app.profiles] == prof_names
    assert app.model_kind_var.get() == _CUSTOM_EDGE_KIND
    assert app.custom_hidden_var.get() == "4"
    assert app._clf is clf_keep, "the in-memory model survives"
    assert app._session_owned and "kept" in app.status_var.get()
    assert app._new_session(keep={"profiles": False, "model": False})
    assert len(app.profiles) == 1 and app.profiles[0]["name"] == "default"
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
          "edge kinds, analysis tab + region list")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
