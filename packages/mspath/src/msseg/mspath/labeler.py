"""``mspath-labeler`` -- the whole-slide labeler.

``LabelerApp`` is to ``MsPathApp`` what ``mscoupon-labeler`` is to
``mscoupon-gui``: the same viewer with ``AnnotationShell`` composed on top, so
the annotation store, the drawing tools, the magic fill, the classifier, the
size sweep and the edge model all arrive already built. What mspath has to
supply is only what the framework cannot know:

* **where the region raster sits** (``_region_placement``). The canvas draws in
  slide coordinates whatever item is on screen, so a gesture at (32100, 41000)
  has to become an index into a raster that may start at (32000, 40000), or be
  a level-4 overview at 1/16. Everything the tools rasterize goes through this;
  what a gesture *stores* stays in slide coordinates, because that is the
  geometry annotations.json keeps and it has to outlive a re-decomposition.
* **the feature scope** (inherited: ``"L<level>"``), so a model trained on one
  pyramid level is refused on another -- the feature names are identical and
  nothing else would catch it.
* **the statistics schema** the active profile produces, for the compatibility
  gate, which comes from the same ``feature_fields`` the coupon labeler uses.

Cross-validation leaves whole SLIDES out (``SlideCatalogue.group_of``): two
ROIs on one slide share tissue, staining and scanner, so holding one out while
training on the other measures memorisation.

    mspath-labeler [folder]
    mspath-labeler --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    import tkinter as tk
except Exception:                                   # headless import
    tk = None

from msseg.labeler import fields
from msseg.labeler.annotate import AnnotationShell
from msseg.labeler.labeling import Placement
from msseg.mscoupon import session as coupon_session

from .app import MsPathApp
from .common import log


class LabelerApp(AnnotationShell, MsPathApp):
    SESSION_APP = "mspath-labeler"
    APP_TITLE = "mspath labeler"
    WINDOW_TITLE = "mspath labeler -- whole-slide region annotation"
    LOG_PREFIX = "mspath"
    MODEL_APP_TAG = "mspath-labeler-classifier"

    # The statistics table's column names. mspath measures the same schema the
    # coupon pipeline does -- it is the same compiled pipeline -- so the
    # framework's defaults are already right.
    FIELDS = fields.DEFAULT

    # ------------------------------------------------------------------ #
    # Where the ids are
    # ------------------------------------------------------------------ #
    def _region_placement(self):
        """The current item's raster, placed on the slide.

        Without this every tool would index a 2940 x 5625 overview raster with
        a slide coordinate up to 90 000 -- out of bounds, which at least fails
        loudly rather than painting the wrong regions.
        """
        cur = self._current()
        item = self._item_at(*cur) if cur is not None else None
        rec = self.engine.record(item.key) if item is not None else None
        if rec is None:
            return Placement()
        return Placement(origin=rec["origin"], scale=rec["scale"])

    # ------------------------------------------------------------------ #
    # The compatibility gate
    # ------------------------------------------------------------------ #
    def _expected_feature_names(self):
        """The feature names the ACTIVE profile produces, from the extension's
        own schema rather than a hand-kept mirror -- so a profile change and
        the gate cannot disagree. None (skipping the gate) when the compiled
        pipeline is unavailable."""
        try:
            from msseg.mscoupon import mscoupon_py as ext
            names = ext.feature_fields(json.dumps(self._profile_for_compute()))
        except Exception as exc:
            log(f"feature schema unavailable: {type(exc).__name__}: {exc}")
            return None
        return [n for n in names if n not in self.FIELDS.positional]

    def _feature_schema_now(self):
        try:
            from msseg.mscoupon import mscoupon_py as ext
            return ext.feature_schema(json.dumps(self._profile_for_compute()))
        except Exception:
            return None

    def _stats_brief(self, stats):
        """One line describing a statistics block, for the model strip."""
        if not stats:
            return "?"
        try:
            return coupon_session.stats_width(stats)
        except Exception:
            return f"{len(stats)} keys"

    def _workflow_summary(self, profile):
        """Two lines naming the active workflow, plus the level -- which for a
        slide is as much a part of the workflow as the filter chain, and is the
        thing a model is pinned to."""
        try:
            text = coupon_session.profile_summary(profile)
        except Exception:
            text = "topo field: ?\nstats: ?"
        level = (profile.get("slide") or {}).get("overview_level",
                                                 self._overview_level())
        return text.replace("topo field: ", f"topo field: L{level} ", 1)

    def _workflow_hint_tooltip(self):
        return ("The workflow this session is annotating: the pyramid level, the "
                "topology field's filter chain, and the statistics the classifier "
                "sees. A model is valid at one level only.")

    def _profile_from_model(self, path, statistics):
        """Offer a profile rebuilt from a loaded model's statistics block."""
        prof = self._default_profile(os.path.splitext(os.path.basename(path))[0])
        if statistics:
            prof["statistics"] = dict(statistics)
        self.profiles.append(prof)
        self.active_profile_idx = len(self.profiles) - 1
        self._refresh_profile_combo()
        self._apply_profile_to_ui(prof, lambda v, x: v.set(x), [])
        return prof

    # ------------------------------------------------------------------ #
    # Export: annotations -> per-pixel masks
    # ------------------------------------------------------------------ #
    def _make_training_set(self):
        """Write `train/` (each item's own pixels) and `labels/` (per-pixel
        class-id masks) -- the raw material for an image model later.

        Per ITEM rather than per slide: a slide has no single resolution, and
        an item is exactly the rect-at-a-level the regions were computed on, so
        the image and its mask are the same pixels by construction.
        """
        from tkinter import filedialog
        if not self.regions.keys():
            self.status_var.set("Run first - the masks need computed regions.")
            return
        if self.regions.pending():
            self.status_var.set("Busy computing - try again in a moment.")
            return
        out = filedialog.askdirectory(title="Choose a folder for the training set")
        if not out:
            return
        written, skipped = self._write_training_set(out)
        msg = (f"Training set: {written} item(s) -> {os.path.join(out, 'train')} "
               f"+ masks -> {os.path.join(out, 'labels')}")
        if skipped:
            msg += f" ({skipped} item(s) skipped - no labels or predictions)"
        self.status_var.set(msg)

    def _write_training_set(self, out_dir):
        """(written, skipped). Annotations win over predictions wherever they
        disagree; a region with neither stays 0 (unlabeled)."""
        import numpy as np
        from PIL import Image
        from msseg.labeler.labeling import resolve_slice

        train_dir = os.path.join(out_dir, "train")
        labels_dir = os.path.join(out_dir, "labels")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        # Gate the model ONCE: under a mismatched profile (or another pyramid
        # level) the masks fall back to annotations alone rather than to
        # silently wrong predictions.
        use_model = (self._clf is not None
                     and self._check_model_compat(self._clf_names,
                                                  "training-set export") is None)
        written = skipped = 0
        for key in self.regions.keys():
            rec = self.regions.ensure_record(key)
            if rec is None or rec.get("labels") is None:
                skipped += 1
                continue
            labels = rec["labels"]
            n_ids = int(rec.get("n_ids") or (int(labels.max()) + 1 if labels.size else 1))
            region_class = np.zeros(n_ids, np.uint8)
            pr = self._pred.get(key) if use_model else None
            if pr is not None and pr[0] == rec.get("commit"):
                take = min(len(pr[1]), n_ids)
                region_class[:take] = np.asarray(pr[1][:take], np.uint8)
            drawn = resolve_slice(self.store.for_slice(key), labels, np,
                                  self.regions.label_layer(key))
            take = min(len(drawn), n_ids)
            m = np.asarray(drawn[:take]) > 0
            region_class[:take][m] = np.asarray(drawn[:take], np.uint8)[m]
            if not region_class.any():
                skipped += 1
                continue
            mask = np.where(labels >= 0, region_class[np.clip(labels, 0, n_ids - 1)], 0)

            item = self.catalogue and __import__("msseg.mspath.items",
                                                 fromlist=["parse_key"]).parse_key(key)
            level, lx, ly, lw, lh, _origin, _scale = self.engine.item_geometry(item)
            tile = self.engine.source(item.slide).read_region(level, lx, ly, lw, lh)
            stem = key.replace("/", "_").replace("@", "_L").replace("#", "_")
            Image.fromarray(np.asarray(tile, np.uint8)).save(
                os.path.join(train_dir, f"{stem}.png"))
            Image.fromarray(mask.astype(np.uint8)).save(
                os.path.join(labels_dir, f"{stem}.png"))
            written += 1
        return written, skipped

    def _export_csv(self):
        """One row per labelled region, across every computed item."""
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            title="Export labelled regions", defaultextension=".csv",
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        n = self._write_labels_csv(path)
        self.status_var.set(f"Exported {n} labelled region(s) -> {path}")

    def _write_labels_csv(self, path):
        """`item, class, <every statistics column>` for each labelled region.

        The item key leads, and it names the slide, the level and the rect, so
        a row stays interpretable without the session that produced it -- and
        the positional columns are already slide coordinates, so two rows from
        different items are talking about the same map.
        """
        import csv
        import numpy as np
        from msseg.labeler.labeling import resolve_slice

        written = 0
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = None
            for key in self.regions.keys():
                rec = self.regions.ensure_record(key)
                table = None if rec is None else rec.get("stats")
                if rec is None or getattr(table, "values", None) is None:
                    continue
                drawn = resolve_slice(self.store.for_slice(key), rec["labels"], np,
                                      self.regions.label_layer(key))
                fid = table.column(self.FIELDS.id_field)
                if fid is None:
                    continue
                if writer is None:
                    writer = csv.writer(fh)
                    writer.writerow(["item", "class"] + list(table.names))
                ids = np.asarray(fid, int)
                ok = (ids >= 0) & (ids < len(drawn))
                for row in np.flatnonzero(ok):
                    cls = int(drawn[ids[row]])
                    if cls <= 0:
                        continue
                    writer.writerow([key, cls] + [f"{v:.10g}" for v in table.values[row]])
                    written += 1
        return written

    # ------------------------------------------------------------------ #
    # The region layer under the class layer
    # ------------------------------------------------------------------ #
    def _set_region_layer_visible(self, on):
        self.regions_var.set(bool(on))

    def _region_layer_visible(self):
        return bool(self.regions_var.get())


def main(argv=None):
    ap = argparse.ArgumentParser(description="mspath whole-slide labeler")
    ap.add_argument("folder", nargs="?", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="run the headless integration test and exit")
    args = ap.parse_args(argv)
    if args.selftest:
        from .selftest import run_labeler_selftest
        return run_labeler_selftest()
    if tk is None:
        print("tkinter is unavailable", file=sys.stderr)
        return 2
    root = tk.Tk()
    LabelerApp(root, initial=args.folder)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
