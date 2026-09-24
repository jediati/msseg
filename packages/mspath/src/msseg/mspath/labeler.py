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

from . import propose
from msseg.labeler.widgets import attach_tooltip

from . import places
from .app import MAX_ROI_PX, MIN_ROI_SIDE, SLIDE_PLANES, MsPathApp
from .items import roi as roi_item
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

    def _draw_meta(self, si, li):
        """The scale of intent a gesture drawn on item (si, li) records: the
        item's level, its slide pixels per raster pixel, and the slide pixels
        per screen pixel at draw time. A gesture is the SLIDE's, so an item
        at another level applying it needs to know how coarse it was."""
        item = self._item_at(si, li)
        if item is None:
            return None
        rec = self.engine.record(item.key)
        level = int(rec["level"]) if rec is not None else int(item.level)
        if rec is not None:
            scale = float(rec["scale"])
        else:
            try:
                scale = float(self.engine.source(item.slide).level_scale(level))
            except Exception:
                scale = float(2 ** level)
        meta = {"level": level, "scale": scale}
        v = self.viewer
        if v is not None:
            try:
                meta["px"] = float(v.scale)
            except (AttributeError, TypeError, ValueError):
                pass
        return meta

    # ------------------------------------------------------------------ #
    # Enrolment: which places the ACTIVE task works (design note §5)
    # ------------------------------------------------------------------ #
    # A place (an ROI record, now with a uid) belongs to the slide; the task
    # says which places it works and at which level. What the task does not
    # enrol is listed (greyed) and browsable, never worked: not primed, not
    # trained on, not classified, not annotatable. The overview is enrolled
    # only by choice -- viewing the whole slide is browsing.
    ENROLMENT = True
    # The fast path: a place is classified when it is selected, and model
    # operations never prime (ensure_record would, synchronously).
    CLASSIFY_ON_ARRIVAL = True

    def _stream_ready(self, key):
        """Computed already: a current record, or a live pipeline that can
        produce one without a prime."""
        if self.engine.record(key) is not None:
            return True
        p = self.engine.primed.get(key)
        return p is not None and getattr(p, "pipe", None) is not None

    def _slide_places(self):
        """``{slide id: its places}`` over the session's slides."""
        out = {}
        for si in range(len(self.subsequences)):
            sid, _p = self._slide_of(si)
            if sid is not None:
                out[sid] = self._rois_of(si)
        return out

    def _ensure_place_uids(self, notes=None):
        """Every place gets a uid, unique over the session (idempotent)."""
        taken = []
        for si in range(len(self.subsequences)):
            rois = self._rois_of(si)
            places.ensure_uids(rois, taken, notes)
            taken.extend(r["uid"] for r in rois)

    def _enrolment(self):
        """The active task's enrolment, materialised if it was None (a task
        from before enrolment works its places -- never the overview)."""
        task = self._task
        if task.enrolled is None:
            self._ensure_place_uids()
            task.enrolled = places.materialise_all(self._slide_places())
        return task.enrolled

    def _place_enrolled(self, si, li):
        sid, _p = self._slide_of(si)
        if sid is None:
            return False
        en = self._task.enrolled
        if li <= 0:
            return places.overview_enrolled(en, sid)
        rois = self._rois_of(si)
        if li - 1 >= len(rois):
            return False
        return places.level_of(en, sid, rois[li - 1]) is not None

    def _place_level(self, si, li):
        rois = self._rois_of(si)
        if not (1 <= li <= len(rois)):
            return 0
        place = rois[li - 1]
        sid, _p = self._slide_of(si)
        lvl = places.level_of(self._task.enrolled, sid, place) if sid is not None else None
        return int(place["level"]) if lvl is None else int(lvl)

    def _row_tags(self, si, li):
        if li is None:
            n = 1 + len(self._rois_of(si))
            worked = any(self._place_enrolled(si, k) for k in range(n))
            return () if worked else (self.UNENROLLED_TAG,)
        return () if self._place_enrolled(si, li) else (self.UNENROLLED_TAG,)

    def _find_place(self, si, rect, level):
        """Reuse a place with the same rect: one piece of tissue is one
        place, whichever tasks work it at whichever levels."""
        return places.find_by_rect(self._rois_of(si), rect)

    def _on_place_added(self, si, li, level, origin):
        """A place cut from the view or proposed (or an existing one reused)
        is enrolled in the ACTIVE task only; its origin names who asked."""
        self._ensure_place_uids()
        place = self._rois_of(si)[li - 1]
        if not isinstance(place.get("origin"), dict):
            place["origin"] = dict(origin or {"reason": "view"})
        place["origin"].setdefault("task", self._task.uid)
        sid, _p = self._slide_of(si)
        places.enrol(self._enrolment(), sid, place["uid"], level)

    def _normalize_enrolment(self, task, notes=None):
        """After a session's tasks install: places get uids; a task written
        before enrolment works its places (the overview is not worked until
        enrolled -- said once); enrolments of places that are gone drop."""
        self._ensure_place_uids(notes)
        slides = self._slide_places()
        if task.enrolled is None:
            task.enrolled = places.materialise_all(slides)
            if notes is not None and slides:
                n = sum(len(v) for v in task.enrolled.values())
                notes.append(f"task {task.name!r}: works its {n} place(s); the overview "
                             "is no longer worked until you enrol it (tree menu)")
        else:
            places.drop_dangling(task.enrolled, slides, notes, f"task {task.name!r}")

    def _level_problem(self, si, place, level):
        """Why `place` cannot be worked at `level` (degenerate or over the
        pixel budget there), or None. A shared place is never shrunk to fit."""
        sid, _p = self._slide_of(si)
        try:
            src = self.engine.source(sid)
            level = max(0, min(int(level), src.levels - 1))
            scale = float(src.level_scale(level))
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        lw, lh = float(place["w"]) / scale, float(place["h"]) / scale
        if min(lw, lh) < MIN_ROI_SIDE:
            return (f"At level {level} this place is {lw:.0f}x{lh:.0f} px - too small to "
                    f"segment (minimum {MIN_ROI_SIDE}).")
        if lw * lh > MAX_ROI_PX:
            return (f"At level {level} this place is {lw:.0f}x{lh:.0f} px - over the "
                    f"{MAX_ROI_PX / 1e6:.0f} Mpx budget; pick a coarser level.")
        return None

    def _enrolment_busy(self):
        if self.regions.pending() or self._search is not None:
            self._notify("Wait for the computation to finish before changing what the task works.")
            return True
        return False

    def _enrol_row(self, si, li, level=None):
        """Enrol tree row (si, li) -- li 0 the overview -- in the active task,
        an ROI at `level` (default: the place's own). Headless-callable.
        Returns True when the task now works it."""
        if self._enrolment_busy():
            return False
        sid, _p = self._slide_of(si)
        if sid is None:
            return False
        en = self._enrolment()
        if li <= 0:
            places.enrol(en, sid, places.OVERVIEW)
        else:
            self._ensure_place_uids()
            rois = self._rois_of(si)
            if li - 1 >= len(rois):
                return False
            place = rois[li - 1]
            lvl = int(place["level"]) if level is None else int(level)
            problem = self._level_problem(si, place, lvl)
            if problem:
                self._notify(problem)
                return False
            places.enrol(en, sid, place["uid"], lvl)
        browsing_it = self._browse_row == (si, li)
        self._enrolment_changed()
        if browsing_it and (si, li) in self.flat_slices:
            self._goto_slice(self.flat_slices.index((si, li)))     # selected: work it
        self.status_var.set(f"Task '{self._task.name}' now works "
                            f"{self._row_description(si, li)}.")
        return True

    def _unenrol_row(self, si, li):
        """Stop working row (si, li) in the active task. The place, its
        gestures and any prime stay; other tasks are untouched."""
        if self._enrolment_busy():
            return False
        sid, _p = self._slide_of(si)
        if sid is None:
            return False
        en = self._enrolment()
        if li <= 0:
            changed = places.unenrol(en, sid, places.OVERVIEW)
        else:
            rois = self._rois_of(si)
            changed = li - 1 < len(rois) and places.unenrol(en, sid, rois[li - 1].get("uid"))
        if changed:
            self._enrolment_changed()
            self.status_var.set(f"Task '{self._task.name}' no longer works "
                                f"{self._row_description(si, li)}.")
        return bool(changed)

    def _enrol_slide_places(self, si):
        """Enrol every place on slide `si` at its own level -- not the
        overview. Places that cannot be worked at their level are skipped."""
        if self._enrolment_busy():
            return 0
        sid, _p = self._slide_of(si)
        if sid is None:
            return 0
        self._ensure_place_uids()
        en = self._enrolment()
        n = 0
        for place in self._rois_of(si):
            if self._level_problem(si, place, int(place["level"])) is None:
                places.enrol(en, sid, place["uid"], int(place["level"]))
                n += 1
        self._enrolment_changed()
        self.status_var.set(f"Task '{self._task.name}' works {n} place(s) on this slide.")
        return n

    # -- the tree: enrol / unenrol / level / note -------------------------- #
    def _seq_tree_menu_entries(self, si, li):
        """The labeler's entries plus what the ACTIVE task works: spliced in
        after "Go to", so Remove stays last."""
        entries = super()._seq_tree_menu_entries(si, li)
        name = self._task.name
        extra = []
        if li is None:
            n = len(self._rois_of(si))
            extra.append((f"Enrol every place in '{name}'",
                          lambda: self._enrol_slide_places(si), n > 0))
        else:
            if self._place_enrolled(si, li):
                extra.append((f"Unenrol from '{name}'", lambda: self._unenrol_row(si, li), True))
            else:
                extra.append((f"Enrol in '{name}'", lambda: self._enrol_row(si, li), True))
            if li > 0:
                lvl = max(0, int(self.roi_level_var.get()))
                extra.append((f"Work at L{lvl}", lambda: self._enrol_row(si, li, lvl),
                              self._place_level(si, li) != lvl
                              or not self._place_enrolled(si, li)))
                extra.append(("Note…", lambda: self._note_place(si, li), True))
        return entries[:1] + extra + entries[1:]

    def _note_place(self, si, li, text=None):
        """Attach a free-text note to a place (every task sees it). `text`
        given -> no dialog (headless); an empty note clears it."""
        rois = self._rois_of(si)
        if not (1 <= li <= len(rois)):
            return False
        place = rois[li - 1]
        if text is None:
            from tkinter import simpledialog
            text = simpledialog.askstring(self.APP_TITLE, "Note for this place:",
                                          initialvalue=place.get("note", ""), parent=self.root)
            if text is None:
                return False
        text = str(text).strip()
        if text:
            place["note"] = text
        else:
            place.pop("note", None)
        self._refresh_subseq_list()
        return True

    def _sequence_item_labels(self, si):
        """Row text: the place's level for the ACTIVE task, and its note."""
        rows = super()._sequence_item_labels(si)
        rois = self._rois_of(si)
        for k, place in enumerate(rois, start=1):
            if k < len(rows) and place.get("note"):
                rows[k] = f"{rows[k]} - {place['note']}"
        return rows

    # -- the places on the whole-slide view --------------------------------- #
    PLACE_COLOR = "#00e5ff"
    PLACE_OTHER_COLOR = "#a0a0a0"

    def _outline_slide(self):
        """The slide whose places are outlined: the one browsed, or the one
        whose OVERVIEW is on screen. None on an ROI (it is a place itself)."""
        if self._browse_row is not None:
            return self._browse_row[0]
        cur = self._current()
        if cur is not None and cur[1] == 0:
            return cur[0]
        return None

    def _redraw_place_outlines(self):
        """Every place on the slide as a box: solid with its level for the
        places the active task works, dashed grey for the rest. Screen-space
        canvas items (tag "places"), re-projected on every zoom / pan."""
        v = self.viewer
        if v is None:
            return
        c = v.canvas
        c.delete("places")
        si = self._outline_slide()
        if si is None:
            return
        sid, _p = self._slide_of(si)
        if sid is None:
            return
        en = self._task.enrolled
        for place in self._rois_of(si):
            x, y, w, h = places.rect_of(place)
            x0, y0 = (x - v.view_x) / v.scale, (y - v.view_y) / v.scale
            x1, y1 = (x + w - v.view_x) / v.scale, (y + h - v.view_y) / v.scale
            lvl = places.level_of(en, sid, place)
            if lvl is not None:
                c.create_rectangle(x0, y0, x1, y1, outline=self.PLACE_COLOR, width=2,
                                   tags=("draw", "places", "place_worked"))
                c.create_text(x0 + 3, y0 + 2, text=f"L{lvl}", anchor="nw",
                              fill=self.PLACE_COLOR, tags=("draw", "places", "place_label"))
            else:
                c.create_rectangle(x0, y0, x1, y1, outline=self.PLACE_OTHER_COLOR, width=1,
                                   dash=(4, 3), tags=("draw", "places", "place_other"))

    def _after_browse(self, si, li):
        self._redraw_place_outlines()

    def _refresh_render(self):
        super()._refresh_render()
        self._redraw_place_outlines()

    def _redraw_hover_geometry(self, *args, **kwargs):
        out = super()._redraw_hover_geometry(*args, **kwargs)
        self._redraw_place_outlines()
        return out

    # -- removal: a place goes for every task ---------------------------- #
    def _tasks_working(self, si, li):
        """The names of the tasks that work tree row (si, li)."""
        sid, _p = self._slide_of(si)
        if sid is None:
            return []
        if li <= 0:
            return [t.name for t in self.tasks if places.overview_enrolled(t.enrolled, sid)]
        rois = self._rois_of(si)
        if li - 1 >= len(rois):
            return []
        return [t.name for t in self.tasks
                if places.level_of(t.enrolled, sid, rois[li - 1]) is not None]

    def _remove_rows_guarded(self, rows):
        if self._search is not None:
            self._notify("A search is running - Cancel it before removing places.")
            return False
        return super()._remove_rows_guarded(rows)

    def _remove_rows_message(self, rows):
        msg = super()._remove_rows_message(rows)
        names = sorted({n for si, li in rows if li is not None and li > 0
                        for n in self._tasks_working(si, li)})
        if names:
            msg += ("\n\nThe place is removed for every task; worked by: "
                    + ", ".join(repr(n) for n in names) + ".")
        return msg

    def _remove_item_at(self, si, li):
        """Removing an ROI removes the PLACE: from every task's enrolment,
        and its computed items at every level any task worked it at (unless
        another place with the same rect still needs one)."""
        rois = self._rois_of(si)
        if li <= 0 or li - 1 >= len(rois):
            return super()._remove_item_at(si, li)
        sid, _p = self._slide_of(si)
        place = rois[li - 1]
        rect = places.rect_of(place)
        levels = {int(place["level"])}
        for t in self.tasks:
            lvl = places.level_of(t.enrolled, sid, place)
            if lvl is not None:
                levels.add(int(lvl))
        uid = place.get("uid")
        super()._remove_item_at(si, li)
        for t in self.tasks:
            if uid:
                places.drop_place(t.enrolled, sid, uid)
        if places.find_by_rect(rois, rect) is None:
            for lvl in levels:
                try:
                    self.engine.forget(roi_item(sid, lvl, *rect).key)
                except Exception:
                    pass

    def _remove_sequence_at(self, si):
        sid, _p = self._slide_of(si)
        super()._remove_sequence_at(si)
        if sid is not None:
            for t in self.tasks:
                places.drop_slide(t.enrolled, sid)

    # -- Run: the active task's items, or every task's on this workflow ---- #
    def _build_run_section(self):
        from tkinter import ttk
        super()._build_run_section()
        self.run_note.config(text="Primes the items the task works: its enrolled "
                                  "overviews and places.")
        self.run_btn.config(text="Run task", command=lambda: self._run("task"))
        self.run_all_btn = ttk.Button(self.run_frame, text="Run all tasks",
                                      command=lambda: self._run("all"))
        self.run_all_btn.pack(fill="x", padx=6, pady=(0, 4))
        attach_tooltip(self.run_all_btn,
                       "Every item any task on this workflow works, primed once: "
                       "switching between those tasks then needs no Run, and each "
                       "level's persistence threshold no longer depends on which "
                       "task ran first. (Tasks on another workflow wait for their "
                       "own Run.)")

    def _items_of_task(self, task):
        """The items `task` works, without making it the active one."""
        out = []
        for si in range(len(self.subsequences)):
            sid, _p = self._slide_of(si)
            if sid is None:
                continue
            if places.overview_enrolled(task.enrolled, sid):
                item = self._item_at(si, 0)
                if item is not None:
                    out.append(item)
            for place in self._rois_of(si):
                lvl = places.level_of(task.enrolled, sid, place)
                if lvl is not None:
                    out.append(roi_item(sid, lvl, int(place["x"]), int(place["y"]),
                                        int(place["w"]), int(place["h"])))
        return out

    def _workflow_params(self, name):
        """The params document a task on workflow `name` primes with (the
        stored profile, composed the way ``_profile_for_compute`` composes the
        panel's), or None when no such profile exists."""
        idx = self._profile_index(name)
        if idx is None:
            return None
        return json.loads(coupon_session.profile_params_json(self.profiles[idx], 1,
                                                             SLIDE_PLANES))

    def _prime_items(self, scope="task"):
        """"task": the active task's items. "all": the union over every task
        -- tasks on the active workflow as plain items (the panel's
        parameters), tasks on another workflow as ``(item, params)`` jobs the
        engine primes under THAT workflow, into its own field slot. Only the
        LRU's few pipes stay live, but every item keeps its record, so a
        later switch to that task shows and classifies it without a Run."""
        if scope != "all":
            return super()._prime_items(scope)
        self._snapshot_active_profile()
        wf = self._task.workflow
        seen, out, per_wf = set(), [], {}
        for t in self.tasks:
            other = t is not self._task and t.workflow != wf
            params = self._workflow_params(t.workflow) if other else None
            if other and params is None:
                log(f"RUN all: task '{t.name}' names no known workflow - skipped")
                continue
            for item in self._items_of_task(t):
                tag = (t.workflow if other else wf, item.key)
                if tag in seen:
                    continue
                seen.add(tag)
                out.append((item, params) if other else item)
                per_wf[tag[0]] = per_wf.get(tag[0], 0) + 1
        if len(per_wf) > 1:
            log("RUN all: " + ", ".join(f"{n} item(s) on '{w}'" for w, n in per_wf.items()))
        return out

    def _enrolment_changed(self):
        """The active task's items changed (a switch, enrol / unenrol): the
        navigation, catalogue and tree follow, the current item is kept when
        still worked, else the first worked item on the same slide, else the
        slide is browsed. Never a prime -- a place is primed when it is
        selected, not when it becomes workable."""
        cur = self._current()
        key = self.catalogue.key_of(*cur) if cur is not None else None
        browse = self._browse_row
        slide_si = cur[0] if cur is not None else (browse[0] if browse else None)
        self._browse_row = None
        self._rebuild_flat_slices()
        target = None
        if key is not None:
            target = self.catalogue.index_of(key)
            if target is not None and target not in self.flat_slices:
                target = None
        if target is None and slide_si is not None:
            target = next((p for p in self.flat_slices if p[0] == slide_si), None)
        if target is not None:
            self.slice_var.set(self.flat_slices.index(target))
            self._sync_slice_combo()
        elif slide_si is not None and slide_si < len(self.subsequences):
            self._browse(slide_si, browse[1] if browse else None)
        elif not self.flat_slices:
            self.slice_var.set(-1)
            self._sync_slice_combo()
            if self.viewer is not None:
                self.viewer.set_overlays([])
        self._refresh_subseq_list()
        self._update_roi_hint()
        if self._current() is not None:
            self._refresh_render()

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
        # Through the switch, so the active task's workflow follows the
        # profile (and the compute state is reset, as any switch does).
        self._bind_workflow(prof["name"])
        return prof

    # ------------------------------------------------------------------ #
    # Where to look next
    # ------------------------------------------------------------------ #
    def _build_roi_section(self):
        """The viewer's ROI controls, plus the one a model makes possible."""
        super()._build_roi_section()
        import tkinter.ttk as ttk
        frame = self.roi_hint_parent
        row = ttk.Frame(frame); row.pack(fill="x", padx=4, pady=(0, 3))
        ttk.Button(row, text="Propose from model",
                   command=self._propose_rois).pack(side="left", padx=4)
        ttk.Label(row, text="count:").pack(side="left")
        ttk.Spinbox(row, from_=1, to=64, width=4,
                    textvariable=self.propose_count_var).pack(side="left", padx=(2, 8))
        ttk.Label(row, text="size:").pack(side="left")
        ttk.Spinbox(row, from_=256, to=4096, increment=256, width=6,
                    textvariable=self.propose_size_var).pack(side="left", padx=2)
        ttk.Combobox(row, textvariable=self.propose_method_var, width=11,
                     state="readonly", values=list(propose.METHODS)).pack(side="left",
                                                                          padx=4)

    def _init_variables(self):
        super()._init_variables()
        self.propose_count_var = tk.IntVar(value=8)
        self.propose_size_var = tk.IntVar(value=2048)
        self.propose_method_var = tk.StringVar(value="entropy")
        self.propose_edges_var = tk.DoubleVar(value=0.0)

    def _propose_rois(self):
        """Cut ROIs where the classifier is least sure about the overview.

        The overview is one prime over the whole slide, so once it is
        classified the model has an opinion everywhere -- and where that
        opinion is weakest is where a full-resolution look is worth its
        seconds. This is the loop HistomicsML runs; what it does NOT do is
        move any label down a level, because the coarse and fine
        decompositions of the same tissue are not nested.
        """
        import numpy as np
        cur = self._current()
        if cur is None:
            self.status_var.set("Open a slide first.")
            return
        si = cur[0]
        item = self._item_at(si, 0)                      # propose from the OVERVIEW
        if item is None:
            return
        key = item.key
        rec = self.engine.record(key)
        pred = self._pred.get(key)
        if rec is None:
            self.status_var.set("Run first - the overview has no regions yet.")
            return
        if pred is None or pred[0] != rec.get("commit"):
            self.status_var.set("Classify the overview first (its predictions are "
                                "what the proposal ranks).")
            return
        proba = pred[2]
        if proba is None:
            self.status_var.set("This model reports no probabilities to rank by.")
            return
        aux = pred[3] if len(pred) > 3 else None
        weight = float(self.propose_edges_var.get() or 0.0)
        arcs = self.regions.arcs(key, np) if weight > 0 else None
        pdiff = (aux or {}).get("pdiff") if weight > 0 else None

        src = self.engine.source(item.slide)
        level = max(0, int(self.roi_level_var.get()))
        rois = propose.propose(
            rec["stats"], proba, np, self.FIELDS, arcs=arcs, pdiff=pdiff,
            method=self.propose_method_var.get(), boundary_weight=weight,
            count=int(self.propose_count_var.get()),
            level=level, size=int(self.propose_size_var.get()),
            level_scale=src.level_scale(level), slide_shape=src.level_shape(0),
            min_area=4.0, max_px=MAX_ROI_PX)
        taken = []
        method = self.propose_method_var.get()
        for r in rois:
            # A proposal is a place asked for BY this task's model: the
            # origin says so, with the score, and only this task enrols it.
            origin = {"task": self._task.uid, "reason": method,
                      "score": round(float(r["score"]), 4)}
            if self._add_roi(si, r["level"], r["x"], r["y"], r["w"], r["h"],
                             origin=origin) is not None:
                taken.append(r)
                log(f"  proposed ROI at region {r['region']} "
                    f"(score {r['score']:.3f}): L{r['level']} "
                    f"({r['x']},{r['y']}) {r['w']}x{r['h']}")
        self.status_var.set(propose.summarise(taken, method))

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
            drawn = resolve_slice(self._gestures_for_key(key), labels, np,
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
                drawn = resolve_slice(self._gestures_for_key(key), rec["labels"], np,
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
