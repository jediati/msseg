"""``mspath-gui`` -- the whole-slide viewer.

The coupon viewer and this one are the same application over different data,
which is what ``msseg.labeler`` was extracted to make true. ``MsPathApp`` is a
``ViewerShell`` binding: it supplies the compute (``SlideEngine``), the two
seams (``SlideCatalogue`` / ``SlideRegionProvider``), the parameter panel, and
the render -- and inherits the window, the session browser, profiles,
navigation, the work-queue pump and the session document unchanged.

What is different from the coupon viewer, and why:

* **A session's sequences are slides, one item each.** The shell's data model
  is folders -> sequences -> items, and a slide maps onto it as a sequence
  holding exactly one thing: its *overview*, a whole coarse pyramid level.
  Level 4 of a 90 000-row slide is 16.5 Mpx -- one ordinary prime, ~9 s -- so
  the overview tier needs no new machinery at all. (Rects at finer levels are
  the next tier; ``items.py`` already keys them.)
* **The base image is never resident.** ``PyramidImageSource`` is handed to the
  canvas directly, so panning reads tiles rather than holding 4 gigapixels.
* **Region rasters are served in slide coordinates.** An item covers part of
  the slide, and its ``RoiLabelLayer`` places its ids there, ``-1`` elsewhere.

Run it against a folder of slides:

    mspath-gui [folder]
    mspath-gui --selftest        # headless integration test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception:                                   # headless import
    tk = ttk = messagebox = None

from msseg.labeler.shell import ViewerShell
from msseg.mscoupon import session as coupon_session
from msseg.mscoupon import config_io
from msseg.mscoupon.app import format_hist_ranges, parse_hist_ranges
from msseg.mscoupon.common import _format_sigmas, _parse_sigmas
from msseg.mscoupon.config_io import (FILTER_OPERATIONS, FILTER_SCHEMA, COLOR_METHODS,
                                      filter_param_schema, filters_to_json)
from msseg.mscoupon.engine import preview_job, preview_raster
from msseg.mscoupon.fingerprints import field_fingerprint_of
from msseg.labeler.defaults import _PREVIEW_PUMP_MS

from .adapters import SlideCatalogue, SlideRegionProvider
from .common import SLIDE_EXTENSIONS, list_slides, log
from .engine import SlideEngine, default_color_method
from .sources import PlacedImageSource, RoiLabelLayer
from .items import overview, parse_key, roi as roi_item, slide_id

# The level a slide's overview is taken at. 4 (1/16) puts a 90 000 x 47 040
# slide at 5625 x 2940 = 16.5 Mpx, which primes in ~9 s and ~1 GB -- the size
# of one coupon slice. Level 3 is 66 Mpx and ~6.6 GB, the practical ceiling;
# level 0 overflows MSCEER's int32 cell indexing outright, so this is not a
# performance preference but the only way the slide is computable at all.
DEFAULT_OVERVIEW_LEVEL = 4

# Compute-only padding around an item. The overview is a whole level and has no
# neighbours to borrow, so it is 0 there; an ROI wants ~64 px (measured in
# experiments/roi_bench.py -- and note that is far more than the filter kernels
# need, because a basin near the cut can drain to an extremum outside it).
DEFAULT_HALO = 64

# A slide read through the pyramid is RGB. Declared rather than measured because
# the statistics schema, and the chain plan the cards draw, both have to resolve
# before any raster is in hand.
SLIDE_PLANES = 3

# The Image channel that means "the chain's state after config stage k".
STAGE_CHANNEL = "stage:"

# The largest ROI worth offering, measured rather than guessed
# (experiments/roi_bench.py): 4096^2 is 8.7 s and ~1.9 GB peak, 8192^2 is 35 s
# and 7.2 GB. Past this an "ROI" stops being something you wait for.
MAX_ROI_PX = 4096 * 4096

# The smallest ROI worth computing, in pixels AT ITS OWN LEVEL. A rect that is
# a few hundred slide pixels is nothing at all once a coarse level has divided
# it by 32, and a 1x1 raster produces one region and a meaningless value range
# -- which then PINS that level's persistence threshold for everything after it.
MIN_ROI_SIDE = 32


def _clean_rois(raw, notes=None):
    """ROI records from a session document, dropping anything unusable -- a
    half-written rect must not become an item whose key names nowhere."""
    out = []
    for r in (raw or []):
        try:
            rec = {"level": int(r["level"]), "x": int(r["x"]), "y": int(r["y"]),
                   "w": int(r["w"]), "h": int(r["h"])}
        except (TypeError, KeyError, ValueError):
            if notes is not None:
                notes.append(f"unusable ROI dropped: {r!r}")
            continue
        if rec["w"] > 0 and rec["h"] > 0 and rec["level"] >= 0:
            # A PLACE's identity and provenance ride along when present
            # (docs/design_multi_model_tasks.md §5): a record without them is
            # the ROI it always was.
            if isinstance(r.get("uid"), str) and r["uid"]:
                rec["uid"] = r["uid"]
            if isinstance(r.get("note"), str) and r["note"]:
                rec["note"] = r["note"]
            if isinstance(r.get("origin"), dict) and r["origin"]:
                rec["origin"] = dict(r["origin"])
            out.append(rec)
        elif notes is not None:
            notes.append(f"empty ROI dropped: {rec}")
    return out


def _roi_row_text(item):
    x, y, w, h = item.rect
    return f"L{item.level} {w}x{h} @({x},{y})"


def _id_lut(n_ids, min_colors, np):
    """RGBA LUT indexed by region id (the canvas treats id < 0 as transparent).

    Sized from the layer's declared id count rather than ``raster.max()``: over
    a 16-megapixel overview that scan is image-sized work on every frame, and
    the layer already knows the answer.
    """
    k = max(int(n_ids), 1)
    lut = np.zeros((k, 4), np.uint8)
    lut[:, :3] = (min_colors(np.arange(k)) * 255).astype(np.uint8)
    lut[:, 3] = 255
    return lut


class MsPathApp(ViewerShell):
    SESSION_APP = "mspath"
    APP_TITLE = "mspath viewer"
    WINDOW_TITLE = "mspath -- whole-slide MSC viewer"
    LOG_PREFIX = "mspath"

    # ------------------------------------------------------------------ #
    # Identity / profiles
    # ------------------------------------------------------------------ #
    def _log(self, msg):
        log(msg)

    def _default_profile(self, name="default"):
        """The coupon profile plus the two things a slide adds: which pyramid
        level the overview is taken at, and the compute halo."""
        p = coupon_session.default_profile(name, relevance=False)
        p["filters"] = [{"operation": "color", "params": {"method": "optical_density"}},
                        {"operation": "blur", "params": {"sigma": 1.5}},
                        {"operation": "edges", "params": {"sigma": 1.0,
                                                          "output": "magnitude"}}]
        p["base_filters"] = [{"operation": "color", "params": {"method": "optical_density"}}]
        p["slide"] = {"overview_level": DEFAULT_OVERVIEW_LEVEL, "halo": DEFAULT_HALO}
        return p

    def _make_catalogue(self):
        return SlideCatalogue(self)

    def _make_region_provider(self):
        return SlideRegionProvider(self)

    # ------------------------------------------------------------------ #
    # Compute state
    # ------------------------------------------------------------------ #
    def _init_compute(self):
        self.engine = SlideEngine()
        self.filter_cards = [{"operation": "color",
                              "params": {"method": "optical_density"}},
                             {"operation": "blur", "params": {"sigma": 1.5}},
                             {"operation": "edges", "params": {"sigma": 1.0,
                                                               "output": "magnitude"}},
                             self._new_filter_card()]
        self.base_cards = [{"operation": "color", "params": {"method": "optical_density"}},
                           self._new_filter_card()]
        self._normalize_readouts = []
        self._preview_src = None            # the pyramid shown before any Run
        # Live preview of the chains (an edit, no Run). The READ is what costs
        # seconds at a deep level, so the array is cached per (item, level,
        # halo) and a chain edit does NOT invalidate it; the chain output is
        # cached per (channel, chain) beside it.
        self._preview_arr = None            # (stamp, arr, geom)
        self._preview_shown = None          # (item key, channel, chain key, source)
        self._preview_chan = {}             # (channel, chain key) -> (raster, source)
        self._preview_worker = None
        self._preview_pending = None
        self._preview_sync = False          # selftests run the worker inline
        self._primed_chain = None           # the chains the primed items used
        self._run_active = False

    def _init_variables(self):
        self.persist_var = tk.DoubleVar(value=10.0)
        self.persist_live_var = tk.StringVar(value="10")
        self.manifold_var = tk.StringVar(value="ascending")
        self.simplification_var = tk.StringVar(value=coupon_session.DEFAULT_SIMPLIFICATION)
        self.accurate_var = tk.BooleanVar(value=False)
        self.gpu_var = tk.BooleanVar(value=False)        # msc.use_gpu_gradient
        self.level_var = tk.IntVar(value=DEFAULT_OVERVIEW_LEVEL)
        self.halo_var = tk.IntVar(value=DEFAULT_HALO)
        self.regions_var = tk.BooleanVar(value=True)
        self.roi_level_var = tk.IntVar(value=0)
        # The statistics spec: which channels a region is measured on, with
        # which reductions. This IS the classifier's feature vector, so it is
        # the coupon panel, ported as it stands.
        self.stat_base_var = tk.BooleanVar(value=True)
        self.stat_filtered_var = tk.BooleanVar(value=False)
        self.stat_color_var = tk.BooleanVar(value=False)
        self.stat_kind_vars = {}       # kind -> (BooleanVar on, StringVar sigmas, StringVar source)
        self.stat_reduction_vars = {}  # reduction -> BooleanVar
        self.stat_extremum_var = tk.BooleanVar(value=True)
        self.hist_on_var = tk.BooleanVar(value=False)
        self.hist_bins_var = tk.StringVar(value="16")
        self.hist_channels_var = tk.StringVar(value="base")
        self.hist_ranges_var = tk.StringVar(value="*: 0, 1")

    # ------------------------------------------------------------------ #
    # Data-model factories (copied from the coupon viewer -- the card model
    # is the same, only the data behind it differs)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _new_filter_card():
        return {"operation": "none", "params": {}}

    def _chain(self, chain):
        """(card list, containing frame) for one of the two filter chains.

        "topo" builds the field the MSC runs on; "base" preprocesses the channel
        statistics are read from. They share all the card machinery -- only the
        list and the frame differ."""
        if chain == "base":
            return self.base_cards, self.base_frame
        return self.filter_cards, self.filters_frame

    def _set_chain_cards(self, chain, cards):
        if chain == "base":
            self.base_cards = cards
        else:
            self.filter_cards = cards

    def _rebuild_filter_cards(self, chain="topo"):
        cards, frame = self._chain(chain)
        for w in list(frame.winfo_children()):
            if isinstance(w, ttk.Label):            # the chain's explanatory label
                continue
            w.destroy()
        # The conversion a slide's RGB planes get when a chain does not start
        # with one is DERIVED: re-planned every rebuild, never stored in `cards`,
        # so it reaches neither the profile nor a config. A slide read through
        # the pyramid is RGB, which is the same 3 _profile_for_compute declares.
        try:
            plan = config_io.chain_plan(cards, SLIDE_PLANES, default_color_method(
                self._profile_from_ui()))
            head = plan["stages"][0] if plan["stages"] else None
        except Exception:
            head = None
        if head is not None and head["synthesized"]:
            self._build_auto_color_card(frame, head, chain)
        for idx, card in enumerate(cards):
            self._build_filter_card(idx, card, chain)
        if chain == "topo":
            try:
                self._refresh_stage_channels()
            except Exception:
                pass

    def _plan_record(self, chain, idx):
        """The planned arity of config stage `idx`, or None if it cannot be
        planned (a chain mid-edit often cannot)."""
        try:
            cards, _frame = self._chain(chain)
            plan = config_io.chain_plan(cards, SLIDE_PLANES,
                                        default_color_method(self._profile_from_ui()))
            for rec in plan["stages"]:
                if rec["index"] == idx:
                    return rec
        except Exception:
            return None
        return None

    def _build_auto_color_card(self, parent, stage, chain):
        """Draw the conversion the runner inserts, greyed and read-only.

        Not one of `self.filter_cards`, so not exported, not saved, not editable
        in place. "Pin" makes it a real card at the head of the chain."""
        frame = ttk.Frame(parent, relief="groove", borderwidth=1)
        frame.pack(fill="x", padx=4, pady=2)
        top = ttk.Frame(frame); top.pack(fill="x")
        method = str(stage["params"].get("method", "luminance"))
        ttk.Label(top, text=f"(auto) color / {method}", foreground="#777",
                  width=22).pack(side="left", padx=2, pady=2)
        ttk.Label(top, text=f"{stage['in']}→{stage['out']}",
                  foreground="#777").pack(side="left", padx=2)
        ttk.Button(top, text="Pin", width=5,
                   command=lambda c=chain, m=method: self._pin_auto_color(c, m)
                   ).pack(side="right", padx=2)

    def _pin_auto_color(self, chain, method):
        """Make the synthesized conversion an ordinary card at index 0."""
        cards, _frame = self._chain(chain)
        cards.insert(0, {"operation": "color", "params": {"method": method}})
        self._set_chain_cards(chain, cards)
        self._rebuild_filter_cards(chain)
        self._notify_profile_edit()

    def _build_filter_card(self, idx, card, chain="topo"):
        cards, parent = self._chain(chain)
        frame = ttk.Frame(parent, relief="groove", borderwidth=1)
        frame.pack(fill="x", padx=4, pady=2)
        top = ttk.Frame(frame); top.pack(fill="x")
        op_var = tk.StringVar(value=card["operation"])
        # `color` consumes the input planes, so only the head of a chain may be one.
        ops = config_io.filter_operations_at(idx)
        combo = ttk.Combobox(top, textvariable=op_var, values=ops, state="readonly", width=20)
        combo.pack(side="left", padx=2, pady=2)
        combo.bind("<<ComboboxSelected>>",
                   lambda e, i=idx, v=op_var, c=chain: self._on_filter_op_change(i, v.get(), c))
        # The plan's arity for this stage, so lifting is never silent: a card
        # reading `3->3 (per plane)` is the whole reason automatic lifting is
        # acceptable where the leading reduction was not.
        rec = self._plan_record(chain, idx)
        if rec is not None and (rec["in"] != 1 or rec["out"] != 1):
            text = f"{rec['in']}→{rec['out']}" + (" (per plane)" if rec.get("lifted") else "")
            ttk.Label(top, text=text, foreground="#777").pack(side="left", padx=4)
        # "show": paint this stage's output. A radio on the card rather than a
        # dropdown entry, because the chain is where the eye already is -- and F
        # walks the same set in order.
        chan_var = getattr(self, "background_var", None)
        if (chan_var is not None and chain == "topo"
                and str(card.get("operation") or "") not in ("", "none")):
            ttk.Radiobutton(top, text="show", value=f"{STAGE_CHANNEL}{idx}",
                            variable=chan_var,
                            command=self._on_stage_radio).pack(side="right", padx=2)
        if idx < len(cards) - 1 or card["operation"] != "none":
            ttk.Button(top, text="✕", width=3,
                       command=lambda i=idx, c=chain: self._remove_filter_card(i, c)
                       ).pack(side="right", padx=2)
        if card["operation"] == "color":
            self._build_color_method_row(frame, card, chain)
            for pname, kind, default in filter_param_schema("color", card["params"])[1:]:
                self._build_param_row(frame, card["params"], pname, kind, default)
        else:
            for pname, kind, default in FILTER_SCHEMA.get(card["operation"], []):
                self._build_param_row(frame, card["params"], pname, kind, default)

    def _build_color_method_row(self, frame, card, chain):
        """The `method` picker of a colour card. Switching it swaps the card's
        parameter rows, so the card is rebuilt with only the new method's keys."""
        params = card["params"]
        if params.get("method") not in COLOR_METHODS:
            params["method"] = "optical_density"
        row = ttk.Frame(frame); row.pack(fill="x", padx=6, pady=1)
        ttk.Label(row, text="method", width=16).pack(side="left")
        var = tk.StringVar(value=params["method"])
        combo = ttk.Combobox(row, textvariable=var, values=COLOR_METHODS,
                             state="readonly", width=16)
        combo.pack(side="left")

        def on_method(_e=None, c=card, v=var, ch=chain):
            method = v.get()
            if method == c["params"].get("method"):
                return
            c["params"] = {"method": method}
            self._rebuild_filter_cards(ch)
            self._notify_profile_edit()
        combo.bind("<<ComboboxSelected>>", on_method)

    def _build_param_row(self, parent, params, pname, kind, default):
        row = ttk.Frame(parent); row.pack(fill="x", padx=6, pady=1)
        ttk.Label(row, text=pname, width=16).pack(side="left")
        if pname not in params:
            params[pname] = default
        # Every commit reports the edit, so the live preview repaints the
        # chain output (debounced) instead of waiting for a Run.
        def _set(p, value):
            params[p] = value
            self._notify_profile_edit()

        if kind == "bool":
            var = tk.BooleanVar(value=bool(params[pname]))
            var.trace_add("write", lambda *_: _set(pname, var.get()))
            ttk.Checkbutton(row, variable=var).pack(side="left")
        elif kind.startswith("choice:"):
            choices = kind.split(":", 1)[1].split(",")
            var = tk.StringVar(value=str(params[pname]))
            var.trace_add("write", lambda *_: _set(pname, var.get()))
            ttk.Combobox(row, textvariable=var, values=choices, state="readonly",
                         width=12).pack(side="left")
        elif kind in ("str", "floats", "numstr"):
            var = tk.StringVar(value=str(params[pname]))
            var.trace_add("write", lambda *_: _set(pname, var.get()))
            ttk.Entry(row, textvariable=var,
                      width=18 if kind == "floats" else 14).pack(side="left")
        elif kind in ("optfloat", "nullfloat"):
            var = tk.StringVar(value=str(params[pname]))

            def commit_opt(*_, p=pname, v=var):
                text = v.get().strip()
                if not text:
                    _set(p, "")
                    return
                try:
                    _set(p, float(text))
                except ValueError:
                    pass
            var.trace_add("write", commit_opt)
            ttk.Entry(row, textvariable=var, width=10).pack(side="left")
        else:                                        # float | int
            var = tk.StringVar(value=str(params[pname]))

            def commit(*_, p=pname, k=kind, v=var):
                try:
                    _set(p, int(v.get()) if k == "int" else float(v.get()))
                except ValueError:
                    pass
            var.trace_add("write", commit)
            ttk.Entry(row, textvariable=var, width=10).pack(side="left")

    def _on_filter_op_change(self, idx, op, chain="topo"):
        cards, _ = self._chain(chain)
        cards[idx]["operation"] = op
        cards[idx]["params"] = {}
        cards = [c for c in cards if c["operation"] != "none"]
        cards.append(self._new_filter_card())
        self._set_chain_cards(chain, cards)
        self._rebuild_filter_cards(chain)
        self._notify_profile_edit()

    def _remove_filter_card(self, idx, chain="topo"):
        cards, _ = self._chain(chain)
        if 0 <= idx < len(cards):
            del cards[idx]
        if not cards or cards[-1]["operation"] != "none":
            cards.append(self._new_filter_card())
        self._set_chain_cards(chain, cards)
        self._rebuild_filter_cards(chain)
        self._notify_profile_edit()

    # ------------------------------------------------------------------ #
    # The item model: one slide, one overview
    # ------------------------------------------------------------------ #
    def _overview_level(self):
        try:
            return max(0, int(self.level_var.get()))
        except Exception:
            return DEFAULT_OVERVIEW_LEVEL

    def _feature_scope(self):
        """A model is valid at ONE pyramid level.

        Every derived channel's sigma is in pixels, so `mean_blur_s1.5` at
        level 4 measures a neighbourhood sixteen times wider than at level 0 --
        and the feature NAMES are identical, so nothing else would catch a
        model being applied at the wrong resolution. Declaring the level as the
        scope makes the compatibility gate refuse it (see
        `msseg.labeler.bundle.compat_message`).
        """
        return f"L{self._overview_level()}"

    def _halo(self):
        try:
            return max(0, int(self.halo_var.get()))
        except Exception:
            return DEFAULT_HALO

    def _slide_of(self, si):
        """(slide id, path) for sequence `si`, registered with the engine."""
        try:
            s = self.subsequences[si]
            path = s["files"][0]
        except (IndexError, KeyError, TypeError):
            return None, None
        sid = slide_id(s.get("folder", ""), path)
        self.engine.register(sid, path)
        return sid, path

    def _rois_of(self, si):
        """The sequence's ROI records, ``[{level, x, y, w, h}]``. Stored on the
        sequence beside its file because an ROI belongs to a slide, and it
        rides the session document there (see `_session_doc`)."""
        try:
            return self.subsequences[si].setdefault("rois", [])
        except (IndexError, KeyError, TypeError, AttributeError):
            return []

    def _item_at(self, si, li):
        """The Item at a shell address.

        A sequence is one slide, and its items are the overview at index 0 and
        its ROIs after it. Index 0 is the overview rather than the first ROI so
        that a slide always has one item -- the tier that is computable before
        anyone has decided where to look.
        """
        sid, _path = self._slide_of(si)
        if sid is None:
            return None
        if li <= 0:
            # Clamped to the levels the file HAS, in the key itself: a small
            # image asked for level 6 computes at its top level either way,
            # and a key that said @6 while @5 computed the same pixels would
            # let one level change detach every annotation on it.
            level = self._overview_level()
            try:
                level = min(level, self.engine.source(sid).levels - 1)
            except Exception:
                pass
            return overview(sid, max(0, level))
        rois = self._rois_of(si)
        if li - 1 >= len(rois):
            return None
        r = rois[li - 1]
        return roi_item(sid, self._place_level(si, li), int(r["x"]), int(r["y"]),
                        int(r["w"]), int(r["h"]))

    # -- which places are worked, at which level --------------------------- #
    # A place (an ROI record) belongs to the slide; the level it is worked at
    # and whether it is worked at all belong to whoever works it. The viewer
    # works everything at the place's own level; the labeler asks the active
    # task (docs/design_multi_model_tasks.md §5).
    def _place_enrolled(self, si, li):
        """Whether item (si, li) -- li 0 the overview -- is worked."""
        return True

    def _place_level(self, si, li):
        """The level place (si, li >= 1) is worked at: its own by default."""
        try:
            return int(self._rois_of(si)[li - 1]["level"])
        except (IndexError, KeyError, TypeError, ValueError):
            return 0

    def _enumerate_items(self):
        for si, _s in enumerate(self.subsequences):
            for li in range(1 + len(self._rois_of(si))):
                if self._place_enrolled(si, li):
                    yield (si, li)

    def _sequence_item_labels(self, si):
        out = []
        for li in range(1 + len(self._rois_of(si))):
            item = self._item_at(si, li)
            out.append("overview L%d" % item.level if item is not None and item.is_overview
                       else (_roi_row_text(item) if item is not None else "?"))
        return out

    def _slice_msc_mark(self, si, li):
        key = self.catalogue.key_of(si, li)
        return "Y" if key and self.engine.record(key) is not None else ""

    def _slice_nav_text(self, si, li):
        item = self._item_at(si, li)
        return item.label() if item is not None else ""

    # ------------------------------------------------------------------ #
    # Panels
    # ------------------------------------------------------------------ #
    def _list_files(self, folder):
        return list_slides(folder)

    # ------------------------------------------------------------------ #
    # Session browser: loaded slides, not folders and sequences
    # ------------------------------------------------------------------ #
    # The shell's session is folders -> files -> sequences, which is what a
    # coupon stack is. A slide session is a list of files. The shell's data
    # model is kept underneath -- every slide is a one-file sequence in a
    # registered folder, so the session document, Load/Restore, the tree's
    # in-place update and navigation all work unchanged -- and only the
    # browser is replaced: one tree of slides with their items, and an
    # "Add slides…" that takes files. The folder and file lists still exist,
    # because the shell writes to them, but they are never packed.
    def _build_session_section(self):
        c = ttk.LabelFrame(self._left_section_parent("session"), text="1. Slides")
        c.pack(fill="x", padx=6, pady=4)
        self.session_frame = c
        hidden = self._session_hidden = ttk.Frame(c)     # never packed
        self.folder_list = self._scrolled_listbox(hidden, height=1, exportselection=False)
        self.file_list = self._scrolled_listbox(hidden, selectmode="extended", height=1,
                                                exportselection=False)
        self.folder_btn_row = ttk.Frame(hidden)
        self.make_seq_btn = ttk.Button(hidden, text="")
        self.seq_btn_row = ttk.Frame(hidden)

        g = self._session_group("slides", c)
        holder = ttk.Frame(g); holder.pack(fill="both", expand=True, padx=4, pady=(2, 0))
        self.subseq_list = ttk.Treeview(holder, columns=("msc", "annot"), height=8,
                                        selectmode="browse")
        self.subseq_list.heading("#0", text="slide / item")
        self.subseq_list.heading("msc", text="msc")
        self.subseq_list.heading("annot", text="annot")
        self.subseq_list.column("#0", width=190, stretch=True)
        self.subseq_list.column("msc", width=38, anchor="center", stretch=False)
        self.subseq_list.column("annot", width=44, anchor="center", stretch=False)
        sb = ttk.Scrollbar(holder, orient="vertical", command=self.subseq_list.yview)
        self.subseq_list.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.subseq_list.pack(side="left", fill="both", expand=True)
        self.subseq_list.bind("<<TreeviewSelect>>", self._on_seq_tree_select)
        self.subseq_list.bind("<Double-1>", self._on_seq_tree_double)
        self.subseq_list.bind("<Button-3>", self._seq_tree_context)
        row = ttk.Frame(g); row.pack(side="bottom", fill="x", padx=4, pady=2)
        ttk.Button(row, text="Add slides…", command=self._add_slides).pack(
            side="left", fill="x", expand=True)
        ttk.Button(row, text="Remove", command=self._remove_subsequence).pack(
            side="left", padx=(4, 0))

    def _on_seq_tree_double(self, _event=None):
        """Double-click on a row: go there AND bring it into view. A single
        click selects the item; the canvas keeps looking wherever it was,
        which on a 90 000-pixel slide can be nowhere near a 2000-pixel ROI."""
        sel = self.subseq_list.selection()
        if not sel or ":" not in sel[0]:
            return
        left, li = sel[0].split(":")
        try:
            si, li = int(left[1:]), int(li)
        except ValueError:
            return
        if (si, li) in self.flat_slices:
            idx = self.flat_slices.index((si, li))
            if idx != int(round(float(self.slice_var.get()))):
                self._goto_slice(idx)
            self._view_item(si, li)
        else:
            self._browse(si, li)          # not worked: look, do not work
        return "break"

    def _view_item(self, si, li):
        """Fit the item on the canvas: the whole slide for an overview, the
        rect for an ROI (with a little margin, so its edge is visible)."""
        v = self.viewer
        item = self._item_at(si, li)
        if v is None or item is None:
            return
        if item.rect is None:
            v.fit()
            return
        x, y, w, h = item.rect
        cw = max(v.canvas.winfo_width(), 1)
        ch = max(v.canvas.winfo_height(), 1)
        margin = 1.1
        scale = max(w * margin / cw, h * margin / ch, 1e-6)
        v.set_view(x + (w - cw * scale) / 2.0, y + (h - ch * scale) / 2.0, scale=scale)

    def _sequence_row_text(self, s):
        files = s.get("files") or []
        return os.path.basename(files[0]) if files else str(s.get("name") or "slide")

    # -- tree rows: slides, their overview and their ROIs ------------------ #
    ITEM_NOUN = "item"

    def _row_kind(self, si, li):
        if li is None:
            return "slide"
        return "overview" if li == 0 else "ROI"

    def _row_description(self, si, li):
        name = self._row_name(si, None)
        if li is None:
            n = len(self._rois_of(si))
            return f"slide '{name}' with its overview and {n} ROI(s)"
        if li == 0:
            return f"the overview of slide '{name}'"
        return f"ROI {self._row_name(si, li)} of slide '{name}'"

    def _remove_target(self, si, li):
        """The overview is the slide: removing it means removing the slide."""
        return (si, None) if li == 0 else (si, li)

    def _row_owns_key(self, si, li, key):
        """A slide row owns every key ON the slide -- including an ROI that
        was cut away earlier and whose annotations were kept for a re-cut;
        they go with the slide."""
        if li is None:
            sid, _path = self._slide_of(si)
            if sid is None:
                return False
            if key == sid:                    # a gesture's key: the slide itself
                return True
            item = parse_key(key)
            return item is not None and item.slide == sid
        return super()._row_owns_key(si, li, key)

    _browse_row = None

    def _goto_row(self, si, li):
        """Go there AND bring it into view, as a double-click does. A row
        that is not worked (the labeler's active task does not enrol it) is
        BROWSED instead: see `_browse`."""
        if (si, 0 if li is None else li) not in self.flat_slices:
            return self._browse(si, li)
        ok = super()._goto_row(si, li)
        if ok:
            self._view_item(si, 0 if li is None else li)
        return ok

    def _browse(self, si, li):
        """Look at slide `si` -- fitted to row `li`'s place -- without working
        it: no item is current (`_current()` is None, so no gesture can be
        stored and nothing is primed or classified), the region overlays are
        off, and the view is the slide's own pyramid. A click on any worked
        row ends it (`_goto_slice`)."""
        sid, path = self._slide_of(si)
        if sid is None:
            return False
        self._browse_row = (si, li)
        self.slice_var.set(-1)
        self._sync_slice_combo()
        self._hover_ctx = None
        v = self.viewer
        if v is not None:
            try:
                src = self.engine.source(sid)
            except Exception as exc:
                self.status_var.set(f"{type(exc).__name__}: {exc}")
                return False
            v.set_source(src, path=path)
            v.set_overlays([])
            v.set_window(*self._window_for("slide"))
            if li is None or li <= 0:
                v.fit()
            else:
                self._view_item(si, li)
        self._after_browse(si, li)
        what = "the slide" if li is None or li <= 0 else "this place"
        self.status_var.set(f"Browsing {os.path.basename(str(path))} - {what} is not worked "
                            "here (nothing is computed or annotated on it).")
        return True

    def _after_browse(self, si, li):
        """Hook: the labeler draws the places' outlines on a browsed slide."""

    def _browsed_slide(self):
        """The slide index being browsed, or None."""
        return None if self._browse_row is None else self._browse_row[0]

    def _goto_slice(self, idx):
        self._browse_row = None
        super()._goto_slice(idx)

    def _rebuild_flat_slices(self):
        browsing = self._browse_row is not None
        super()._rebuild_flat_slices()
        if browsing:                      # the rebuild resets to row 0; stay put
            self.slice_var.set(-1)
            self._sync_slice_combo()

    def _remove_item_at(self, si, li):
        if li <= 0:
            return                       # the overview goes with its slide
        key = self.catalogue.key_of(si, li)
        rois = self._rois_of(si)
        if li - 1 < len(rois):
            gone = rois.pop(li - 1)
            log(f"ROI removed: {gone}")
        if key is not None:
            self.engine.forget(key)

    def _remove_sequence_at(self, si):
        sid, _path = self._slide_of(si)
        if sid is not None:
            n = self.engine.forget_slide(sid)
            log(f"slide removed: {sid} ({n} computed item(s) dropped)")
        super()._remove_sequence_at(si)

    def _after_rows_removed(self, cur_key, pos):
        super()._after_rows_removed(cur_key, pos)
        self._update_roi_hint()
        if not self.flat_slices and self.viewer is not None:
            self.viewer.clear()          # its pyramid is closed

    def _bind_preview(self):
        pass                                        # no file list to click

    def _add_slides(self):
        from tkinter import filedialog
        paths = filedialog.askopenfilenames(
            title="Add slides to the session",
            filetypes=[("Slides", " ".join("*" + e for e in SLIDE_EXTENSIONS)),
                       ("All files", "*.*")])
        added = [p for p in paths if self._add_slide_path(p)]
        if added:
            self.status_var.set(f"Added {len(added)} slide(s).")

    def _ensure_folder(self, folder):
        """Register a folder with the shell without adding its slides."""
        norm = os.path.normpath(folder)
        for i, f in enumerate(self.folders):
            if os.path.normpath(f["path"]) == norm:
                return f["name"]
        ViewerShell._add_folder_path(self, folder)
        return self.folders[-1]["name"]

    def _add_slide_path(self, path):
        """One slide becomes one sequence. Returns True when it was new."""
        path = os.path.normpath(str(path))
        if not os.path.isfile(path):
            self.status_var.set(f"Not a file: {path}")
            return False
        for s in self.subsequences:
            if any(os.path.normpath(f) == path for f in (s.get("files") or [])):
                return False
        folder = self._ensure_folder(os.path.dirname(path))
        self.subsequences.append({"name": os.path.basename(path), "folder": folder,
                                  "files": [path], "rois": []})
        self._rebuild_flat_slices_keeping_current()
        self._refresh_subseq_list()
        self._update_roi_hint()
        # The first slide of a session goes on screen at once: a session with
        # slides in it and a blank canvas reads as broken, and there is nothing
        # to wait for -- the pyramid is the preview.
        if self.viewer is not None and not self.viewer.has_base:
            if self.flat_slices:
                self._goto_slice(0)
            else:                          # nothing worked yet (the labeler): look at it
                self._browse(len(self.subsequences) - 1, None)
        return True

    def _add_folder_path(self, path):
        """A folder on the command line means "every slide in it"."""
        idx = super()._add_folder_path(path)
        for f in list_slides(path):
            self._add_slide_path(f)
        return idx

    def _build_processing_sections(self):
        parent = self._processing_parent("filters")

        slide = self._group(parent, "1. Slide", key="slide")
        row = ttk.Frame(slide); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="overview level:").pack(side="left")
        ttk.Spinbox(row, from_=0, to=9, width=4, textvariable=self.level_var,
                    command=self._on_level_change).pack(side="left", padx=4)
        ttk.Label(row, text="halo px:").pack(side="left", padx=(12, 0))
        ttk.Spinbox(row, from_=0, to=512, increment=16, width=5,
                    textvariable=self.halo_var).pack(side="left", padx=4)
        self.level_hint = tk.StringVar(value="")
        ttk.Label(slide, textvariable=self.level_hint, foreground="#666").pack(
            anchor="w", padx=6)

        topo = self._group(parent, "2. Topology field (the MSC runs on this)",
                           key="filters")
        self.filters_frame = ttk.Frame(topo); self.filters_frame.pack(fill="x")
        self._rebuild_filter_cards("topo")

        base = self._group(parent, "3. Base channel (statistics are read from this)",
                           key="base")
        ttk.Label(base, text="Derived from the raw slide, not chained onto the topology field.",
                  foreground="#666").pack(anchor="w", padx=6)
        self.base_frame = ttk.Frame(base); self.base_frame.pack(fill="x")
        self._rebuild_filter_cards("base")

        msc = self._group(parent, "4. MSC", key="msc")
        row = ttk.Frame(msc); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="manifold:").pack(side="left")
        ttk.Combobox(row, textvariable=self.manifold_var, state="readonly", width=12,
                     values=["ascending", "descending"]).pack(side="left", padx=4)
        row = ttk.Frame(msc); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="simplification:").pack(side="left")
        ttk.Radiobutton(row, text="merge forest", variable=self.simplification_var,
                        value="merge_forest").pack(side="left", padx=4)
        ttk.Radiobutton(row, text="MSC hierarchy", variable=self.simplification_var,
                        value="msc").pack(side="left", padx=4)
        ttk.Checkbutton(msc, text="accurate gradient (slower, ~12x the memory; "
                                  "run-to-run nondeterministic)",
                        variable=self.accurate_var).pack(anchor="w", padx=6)
        # The discrete gradient on CUDA (bit-identical labels), and with it the
        # statistics accumulation: a twelve-channel bank over a 16 Mpx ROI is
        # 2.7-8.5 s on the CPU path and never materializes on the device one.
        ttk.Checkbutton(msc, text="GPU gradient + statistics (CUDA; bit-identical results)",
                        variable=self.gpu_var).pack(anchor="w", padx=6)

        self._build_statistics_panel()

    # ------------------------------------------------------------------ #
    # Statistics channels (ported from the coupon viewer: the feature vector)
    # ------------------------------------------------------------------ #
    def _build_statistics_panel(self):
        """Which channels a region is measured on, and with which reductions.

        A derived channel is a Gaussian-derivative response computed on the
        base raster; naming several sigmas on one row is the cross-product. They
        are measure-only: the topology field is still `filters`. Sigmas are in
        PIXELS AT THE ITEM'S LEVEL, which is why a model is pinned to a level.
        """
        c = self._group(self._processing_parent("stats"), "5. Statistics channels",
                        key="stats")
        self.stats_frame = c

        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=2)
        ttk.Checkbutton(row, text="base", variable=self.stat_base_var,
                        command=self._on_stat_spec_change).pack(side="left")
        ttk.Checkbutton(row, text="filtered", variable=self.stat_filtered_var,
                        command=self._on_stat_spec_change).pack(side="left", padx=8)
        ttk.Checkbutton(row, text="color planes", variable=self.stat_color_var,
                        command=self._on_stat_spec_change).pack(side="left", padx=8)

        for kind in config_io.DERIVED_CHANNEL_KINDS:
            row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=1)
            on = tk.BooleanVar(value=False)
            sigmas = tk.StringVar(value="0.7, 1.5, 3.0")
            color_only = kind in config_io.COLOR_ONLY_KINDS
            ttk.Checkbutton(row, text=kind, variable=on, width=10,
                            command=self._on_stat_spec_change).pack(side="left")
            ttk.Label(row, text="sigmas:").pack(side="left")
            entry = ttk.Entry(row, textvariable=sigmas, width=16)
            entry.pack(side="left", padx=2)
            entry.bind("<Return>", lambda e: self._on_stat_spec_change())
            entry.bind("<FocusOut>", lambda e: self._on_stat_spec_change())
            source = tk.StringVar(value="color" if color_only else "base")
            src = ttk.Combobox(row, textvariable=source, values=config_io.STAT_SOURCES,
                               state="disabled" if color_only else "readonly", width=6)
            src.pack(side="left", padx=2)
            src.bind("<<ComboboxSelected>>", lambda e: self._on_stat_spec_change())
            if kind in config_io.TWO_SLOT_KINDS:
                ttk.Label(row, text="(largest + smallest)").pack(side="left")
            self.stat_kind_vars[kind] = (on, sigmas, source)

        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=(4, 2))
        ttk.Label(row, text="reductions:").pack(side="left")
        for reduction in config_io.STAT_REDUCTIONS:
            var = tk.BooleanVar(value=True)
            self.stat_reduction_vars[reduction] = var
            ttk.Checkbutton(row, text=reduction, variable=var,
                            command=self._on_stat_spec_change).pack(side="left", padx=2)
        ttk.Checkbutton(c, text="seeding extremum (ext_* per channel)",
                        variable=self.stat_extremum_var,
                        command=self._on_stat_spec_change).pack(anchor="w", padx=4)
        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=(4, 1))
        ttk.Checkbutton(row, text="histogram", variable=self.hist_on_var, width=10,
                        command=self._on_stat_spec_change).pack(side="left")
        ttk.Label(row, text="bins:").pack(side="left")
        for var, width in ((self.hist_bins_var, 4), (self.hist_channels_var, 14)):
            e = ttk.Entry(row, textvariable=var, width=width); e.pack(side="left", padx=2)
            e.bind("<Return>", lambda ev: self._on_stat_spec_change())
            e.bind("<FocusOut>", lambda ev: self._on_stat_spec_change())
            if var is self.hist_bins_var:
                ttk.Label(row, text="channels:").pack(side="left")
        row = ttk.Frame(c); row.pack(fill="x", padx=4, pady=1)
        ttk.Label(row, text="ranges (name: lo, hi; ...):").pack(side="left")
        e = ttk.Entry(row, textvariable=self.hist_ranges_var, width=22); e.pack(side="left", padx=2)
        e.bind("<Return>", lambda ev: self._on_stat_spec_change())
        e.bind("<FocusOut>", lambda ev: self._on_stat_spec_change())
        ttk.Button(row, text="measure", width=8,
                   command=self._measure_hist_ranges).pack(side="left", padx=2)
        self.stat_summary_var = tk.StringVar(value="")
        ttk.Label(c, textvariable=self.stat_summary_var, foreground="#555",
                  wraplength=330, justify="left").pack(anchor="w", padx=4, pady=(0, 3))
        self._refresh_stat_summary()

    def _apply_stat_state(self, state, setvar):
        """Restore the `statistics` block into the panel's controls. One row per
        kind, so a config naming a kind twice has its sigma lists merged."""
        channels = state.get("stat_channels")
        if channels is None:
            return
        setvar(self.stat_base_var, any(c.get("kind") == "base" for c in channels))
        setvar(self.stat_filtered_var, any(c.get("kind") == "filtered" for c in channels))
        setvar(self.stat_color_var, any(c.get("kind") == "color" for c in channels))
        by_kind, by_source = {}, {}
        for card in channels:
            kind = card.get("kind")
            if kind in ("base", "filtered", "color") or kind not in self.stat_kind_vars:
                continue
            by_source.setdefault(kind, card.get("source") or "base")
            for sigma in card.get("sigmas") or []:
                by_kind.setdefault(kind, [])
                if sigma not in by_kind[kind]:
                    by_kind[kind].append(float(sigma))
        for kind, (on, sigmas, source) in self.stat_kind_vars.items():
            values = by_kind.get(kind)
            setvar(on, bool(values))
            if values:
                setvar(sigmas, _format_sigmas(sorted(values)))
            setvar(source, "color" if kind in config_io.COLOR_ONLY_KINDS
                   else by_source.get(kind, "base"))
        reductions = state.get("stat_reductions")
        if reductions is not None:
            for name, var in self.stat_reduction_vars.items():
                setvar(var, name in reductions)
        if state.get("stat_extremum") is not None:
            setvar(self.stat_extremum_var, bool(state["stat_extremum"]))
        hist = state.get("stat_histogram")
        if hist is not None:
            setvar(self.hist_on_var, bool(hist.get("bins")))
            if hist.get("bins"):
                setvar(self.hist_bins_var, str(int(hist["bins"])))
                setvar(self.hist_channels_var, ", ".join(hist.get("channels") or ["base"]))
                setvar(self.hist_ranges_var, format_hist_ranges(hist.get("ranges") or {}))
        self._refresh_stat_summary()

    def _stat_sources(self):
        """The active profile's `statistics.sources`, carried through unchanged.

        There is no editor for them yet: a source is a filter chain a config or
        another tool declares. But the statistics block is rebuilt from UI state
        on every export, so without this a profile that declares a source would
        lose it the first time the GUI wrote the profile back -- and its channels
        would then stop resolving. Preserving what a profile brought is the whole
        contract until there is a panel."""
        try:
            prof = self.profiles[self.active_profile_idx]
        except (AttributeError, IndexError):
            return None
        stats = prof.get("statistics")
        if not isinstance(stats, dict):
            return None
        src = stats.get("sources")
        return src if isinstance(src, dict) and src else None

    def _stat_channel_cards(self):
        """The `statistics.channels[]` model, in slot order."""
        cards = []
        if self.stat_base_var.get():
            cards.append({"kind": "base"})
        if self.stat_filtered_var.get():
            cards.append({"kind": "filtered"})
        if self.stat_color_var.get():
            cards.append({"kind": "color"})
        for kind, (on, sigmas, source) in self.stat_kind_vars.items():
            if not on.get():
                continue
            values = _parse_sigmas(sigmas.get())
            if values:
                card = {"kind": kind, "sigmas": values}
                if kind in config_io.COLOR_ONLY_KINDS or source.get() == "color":
                    card["source"] = "color"
                cards.append(card)
        return cards

    def _stat_reductions(self):
        return [r for r, v in self.stat_reduction_vars.items() if v.get()]

    def _hist_channel_names(self):
        return [c.strip() for c in self.hist_channels_var.get().replace(";", ",").split(",")
                if c.strip()]

    def _stat_histogram(self):
        if not self.hist_on_var.get():
            return None
        try:
            bins = int(float(self.hist_bins_var.get()))
        except (ValueError, tk.TclError):
            bins = 16
        return {"bins": bins, "channels": self._hist_channel_names(),
                "ranges": parse_hist_ranges(self.hist_ranges_var.get())}

    def _measure_hist_ranges(self):
        """Fill the "*" range from the item on screen: the min/max over its
        primed base / filtered rasters (the only channels held per item)."""
        import numpy as np
        cur = self._current()
        key = self.catalogue.key_of(*cur) if cur is not None else None
        p = self.engine.primed.get(key) if key else None
        if p is None:
            self.status_var.set("histogram ranges: prime the item on screen first")
            return
        lo, hi = np.inf, -np.inf
        for name in self._hist_channel_names() or ["base"]:
            raster = getattr(p, name, None) if name in ("base", "filtered") else None
            if raster is None:
                continue
            r = np.asarray(raster, np.float64)
            r = r[np.isfinite(r)]
            if r.size:
                lo, hi = min(lo, float(r.min())), max(hi, float(r.max()))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            self.status_var.set("histogram ranges: only base / filtered can be measured here")
            return
        self.hist_ranges_var.set(f"*: {lo:.6g}, {hi:.6g}")
        self._on_stat_spec_change()

    def _stat_channel_names(self):
        try:
            return [c["name"] for c in config_io.stat_channels(
                json.dumps(self._profile_for_compute()))]
        except Exception:
            return ["base"]

    def _refresh_stat_summary(self):
        """The resolved channel and field counts: what decides how wide every
        per-region row is, and therefore what the classifier sees."""
        var = getattr(self, "stat_summary_var", None)
        if var is None:
            return
        try:
            params = json.dumps(self._profile_for_compute())
            n_ch = len(config_io.stat_channels(params))
            n_fields = len(config_io.query_fields(params))
        except Exception:
            n_ch, n_fields = 0, 0
        var.set(f"{n_ch} channels -> {n_fields} selectable fields")

    def _on_stat_spec_change(self):
        """The measurement spec changed: anything primed under the old spec
        measures the wrong things. Nothing is dropped -- the records stay on
        screen -- and the next Run re-measures the live items rather than
        priming them (the labeler re-measures the item on screen at once)."""
        self._refresh_stat_summary()
        if self.engine.primed:
            self.status_var.set("Statistics changed - Run re-measures the new channels "
                                "(no re-prime).")

    def _build_run_section(self):
        parent = self._left_section_parent("run")
        # `run_frame` is part of the shell's contract, not decoration: the
        # workflow hint packs itself above this section's first child.
        self.run_frame = frame = ttk.LabelFrame(parent, text="6. Run")
        frame.pack(fill="x", padx=4, pady=4)
        self.run_note = ttk.Label(frame, text="Primes every listed slide's overview and ROIs.",
                                  foreground="#666")
        self.run_note.pack(anchor="w", padx=6, pady=(2, 0))
        self.run_btn = ttk.Button(frame, text="Run", command=self._run)
        self.run_btn.pack(fill="x", padx=6, pady=4)
        self.run_all_btn = None           # the labeler adds "Run all tasks"

    def _set_run_buttons(self, state):
        for b in (getattr(self, "run_btn", None), getattr(self, "run_all_btn", None)):
            if b is not None:
                b.config(state=state)

    def _prime_items(self, scope="task"):
        """The items a Run primes. The viewer primes everything it lists;
        the labeler's "task" scope is the items the active task works and
        its "all" scope the union over the tasks on the active workflow."""
        out = []
        for si, li in self._enumerate_items():
            item = self._item_at(si, li)
            if item is not None:
                out.append(item)
        return out

    def _build_left(self):
        super()._build_left()
        self._build_roi_section()

    def _build_roi_section(self):
        parent = self._left_section_parent("roi")
        frame = ttk.LabelFrame(parent, text="2. Regions of interest")
        frame.pack(fill="x", padx=4, pady=4)
        ttk.Label(frame, text="Full-resolution work happens in ROIs: the whole slide "
                              "cannot be\nsegmented at level 0 at all.",
                  foreground="#666").pack(anchor="w", padx=6, pady=(2, 0))
        row = ttk.Frame(frame); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="level:").pack(side="left")
        ttk.Spinbox(row, from_=0, to=9, width=4,
                    textvariable=self.roi_level_var).pack(side="left", padx=4)
        ttk.Button(row, text="Add from view",
                   command=self._add_roi_from_view).pack(side="left", padx=4)
        ttk.Button(row, text="Remove", command=self._remove_roi).pack(side="left")
        self.roi_hint = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.roi_hint, foreground="#666").pack(
            anchor="w", padx=6, pady=(0, 3))
        # The labeler adds a row here (proposing ROIs needs a model, which the
        # viewer has not got), so the frame is part of the hook's contract.
        self.roi_hint_parent = frame

    def _add_roi_from_view(self):
        """Cut an ROI from what is on screen, at the ROI level.

        The viewport is the selection: it is already the rect the user chose by
        navigating there, and it needs no new canvas tool to express. Clamped
        to the slide and capped at MAX_ROI_PX -- a request for a quarter of a
        slide at level 0 is a request for half an hour of compute.
        """
        cur = self._current()
        si = (cur[0] if cur is not None else self._browsed_slide())
        if si is None:
            si = 0 if self.subsequences else None
        if si is None:
            self.status_var.set("Add a slide first.")
            return
        sid, _path = self._slide_of(si)
        v = self.viewer
        if v is None or sid is None:
            return
        try:
            src = self.engine.source(sid)
        except Exception as exc:
            self.status_var.set(f"{type(exc).__name__}: {exc}")
            return
        sh, sw = src.level_shape(0)
        cw = max(v.canvas.winfo_width(), 1)
        ch = max(v.canvas.winfo_height(), 1)
        x0 = max(0, int(v.view_x)); y0 = max(0, int(v.view_y))
        x1 = min(sw, int(v.view_x + cw * v.scale))
        y1 = min(sh, int(v.view_y + ch * v.scale))
        self._add_roi(si, max(0, int(self.roi_level_var.get())),
                      x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    def _find_place(self, si, rect, level):
        """An existing place to reuse for a new ROI of this rect, as its index
        in the slide's list, or None to append one. The viewer never reuses
        (a rect at a second level is a second record, as it always was); the
        labeler reuses the place and enrols it at the level asked for."""
        return None

    def _on_place_added(self, si, li, level, origin):
        """A place was added or reused at tree row (si, li): the labeler
        enrols it in the active task. Runs before the rebuild and the
        navigation, so the new item is already worked when they look."""

    def _add_roi(self, si, level, x, y, w, h, origin=None):
        """Add one ROI to slide `si`, in slide coordinates. Headless-callable;
        `_add_roi_from_view` is the UI that computes the rect.

        Refuses a rect that is degenerate at its own level and caps one that is
        too big; returns the new item, or None. `origin` (``{"task",
        "reason"[, "score"]}``) records who asked for the place."""
        sid, _path = self._slide_of(si)
        if sid is None:
            return None
        try:
            src = self.engine.source(sid)
        except Exception as exc:
            self.status_var.set(f"{type(exc).__name__}: {exc}")
            return None
        level = max(0, min(int(level), src.levels - 1))
        scale = src.level_scale(level)
        x, y, w, h = int(x), int(y), max(1, int(w)), max(1, int(h))
        lw, lh = w / scale, h / scale
        if min(lw, lh) < MIN_ROI_SIDE:
            self.status_var.set(
                f"That rect is {lw:.0f}x{lh:.0f} px at level {level} - too small to "
                f"segment (minimum {MIN_ROI_SIDE}). Zoom in, or pick a finer level.")
            return None
        if lw * lh > MAX_ROI_PX:
            k = (MAX_ROI_PX / (lw * lh)) ** 0.5
            nw, nh = max(1, int(w * k)), max(1, int(h * k))
            x += (w - nw) // 2
            y += (h - nh) // 2
            w, h = nw, nh
            lw, lh = w / scale, h / scale
            self.status_var.set(
                f"ROI capped to {lw:.0f}x{lh:.0f} px at level {level} "
                f"({MAX_ROI_PX / 1e6:.0f} Mpx budget).")
        rois = self._rois_of(si)
        found = self._find_place(si, (x, y, w, h), level)
        if found is None:
            rec = {"level": level, "x": x, "y": y, "w": w, "h": h}
            if origin:
                rec["origin"] = dict(origin)
            rois.append(rec)
            li = len(rois)
        else:
            li = int(found) + 1
        self._on_place_added(si, li, level, origin)
        self._rebuild_flat_slices()
        self._refresh_subseq_list()
        self._update_roi_hint()
        log(f"ROI {'added' if found is None else 'reused'} on {sid}: L{level} ({x},{y}) "
            f"{w}x{h} slide px = {lw:.0f}x{lh:.0f} at the level")
        item = self._item_at(si, li)
        try:
            self._goto_slice(self.flat_slices.index((si, li)))
        except ValueError:
            pass
        return item

    def _remove_roi(self):
        """Drop the ROI on screen, after asking -- the same path as the tree's
        context menu, so the labeler takes its annotations with it."""
        cur = self._current()
        if cur is None or cur[1] <= 0:
            self.status_var.set("Select an ROI to remove (the overview stays).")
            return
        self._remove_rows_guarded([cur])

    def _remove_roi_at(self, si, li):
        """The removal itself, unguarded (headless callers)."""
        if li > 0:
            self._remove_rows([(si, li)])

    def _update_roi_hint(self):
        hint = getattr(self, "roi_hint", None)
        if hint is None:
            return
        n = sum(len(self._rois_of(si)) for si in range(len(self.subsequences)))
        primed = sum(1 for k in self.regions.keys() if self.engine.record(k) is not None)
        hint.set(f"{n} ROI(s) over {len(self.subsequences)} slide(s); "
                 f"{primed} item(s) computed")

    def _build_live_panel(self, parent):
        live = ttk.LabelFrame(parent, text="Live parameters")
        live.pack(side="bottom", fill="x", padx=6, pady=4)
        # Persistence is a numeric entry committed on Enter / focus-out, as in
        # the coupon viewer: a select is cheap, but re-resolving on every
        # slider tick bumps the commit and drops every cache keyed on it.
        row = ttk.Frame(live); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Persistence %:").pack(side="left")
        self.persist_entry = ttk.Entry(row, textvariable=self.persist_live_var, width=8)
        self.persist_entry.pack(side="left", padx=4)
        self.persist_entry.bind("<Return>", self._on_persistence_change)
        self.persist_entry.bind("<FocusOut>", self._on_persistence_change)
        self.persist_label = tk.StringVar(value="")
        ttk.Label(row, textvariable=self.persist_label).pack(side="left", padx=4)

    def _build_segmentation_controls(self, chan):
        ttk.Label(chan, textvariable=getattr(self, "level_hint", tk.StringVar()),
                  foreground="#666").pack(side="left", padx=8)

    CHANNELS = ("slide", "base", "filtered")
    _pre_stage_channel = "slide"

    def _original_channel(self):
        """What F flips back to: the slide itself."""
        return "slide"

    def _after_layout(self):
        self._update_level_hint()
        self._update_roi_hint()
        # The shell offers base|filtered, which for a coupon slice are the two
        # rasters there are. A slide has a third thing to look at -- itself --
        # and it is the default: the scalar channels only exist once an item
        # is primed.
        self._refresh_stage_channels()
        # The cards were built before the picker existed, so their `show` radios
        # were skipped; now that it does, draw them.
        try:
            self._rebuild_filter_cards("topo")
        except Exception:
            pass

    def _channel_source(self, key, channel, slide_src):
        """The ImageSource for `channel` on the current item: the pyramid for
        "slide"; for "base" / "filtered" the primed item's scalar raster placed
        on the slide, or the pyramid again with a note when nothing is primed.

        Cached on the Primed record per channel, because building the source
        measures the raster's range and a repaint must not scan 16 Mpx."""
        if channel == "slide":
            return slide_src, None
        live = self._live_channel_source(key, channel)
        if live is not None:
            return live, None
        if channel in self._stage_channels():
            # A stage is never primed; it is computed on demand and shown
            # through the live-preview path above. Reaching here means the
            # compute has not landed yet (or failed), so say that rather than
            # "Run first", which would not help.
            return slide_src, f"{self._stage_label(channel)}: computing..."
        p = self.engine.primed.get(key)
        raster = None if p is None else getattr(p, channel, None)
        if raster is None:
            return slide_src, f"{channel}: Run first - nothing is primed for this item."
        cache = p.channel_sources
        src = cache.get(channel)
        if src is None:
            src = cache[channel] = PlacedImageSource(
                raster, origin=p.origin, scale=p.scale,
                slide_shape=slide_src.level_shape(0), path=None)
        return src, None

    # ------------------------------------------------------------------ #
    # The live preview: a chain edit repaints without a Run
    # ------------------------------------------------------------------ #
    _PREVIEW_CHAN_MAX = 4                   # (channel, chain) rasters + sources

    def _on_image_channel_change(self, *a, **kw):
        cur = self.background_var.get() or ""
        stages = self._stage_channels()
        if cur not in stages:
            self._last_non_stage_channel = cur
            self._pre_stage_channel = cur or "slide"
        out = super()._on_image_channel_change(*a, **kw)
        # `base` and `filtered` come from the PRIMED record, so picking one just
        # repaints. A stage has no primed counterpart -- it is an intermediate
        # the chain passes through and nothing stores -- so it exists only as a
        # preview, and picking one has to ask for it. Without this the channel
        # changed and the picture did not.
        if cur in stages:
            self._launch_preview()
        return out

    def _stage_channels(self):
        """One channel per chain stage: the state the chain passes through
        after it. A stage yielding a stack renders as RGB, which is the whole
        point -- `edges` lifted over three planes IS three planes, and looking
        at them is how a chain gets judged."""
        return [f"{STAGE_CHANNEL}{i}" for i, c in enumerate(self.filter_cards)
                if str(c.get("operation") or "") not in ("", "none")]

    def _refresh_stage_channels(self):
        """Keep the Image picker in step with the chain."""
        var = getattr(self, "background_var", None)
        if var is None:
            return
        combo = getattr(self, "background_combo", None)
        offered = list(self.CHANNELS) + self._stage_channels()
        if combo is not None:
            combo.config(values=offered)
        if var.get() not in offered:
            var.set("slide")

    def _stage_label(self, channel):
        idx = self._stage_index(channel)
        if idx is None or idx >= len(self.filter_cards):
            return str(channel)
        return f"after {self.filter_cards[idx].get('operation')}"

    def _on_stage_radio(self):
        """A card's `show` was picked: the Image channel IS that stage.

        Entering the walk from an ordinary channel remembers it, so stepping
        off the end of the chain comes back to what was being looked at."""
        cur = self.background_var.get() or ""
        stages = self._stage_channels()
        prev = getattr(self, "_last_non_stage_channel", None)
        if prev and prev not in stages:
            self._pre_stage_channel = prev
        self._on_image_channel_change()
        self.status_var.set(f"showing {self._stage_label(cur)}")

    def _swap_image(self):
        """F: while a stage is on screen, step to the next one.

        A card's `show` radio ENTERS the walk; F continues it, and the last
        stage comes back to whatever was on screen before. Anywhere else F is
        the framework's own A/B flip (original <-> the derived channel last
        shown), which is a different and still useful thing -- so this overrides
        only the case the framework has no opinion about."""
        cur = self.background_var.get() or ""
        stages = self._stage_channels()
        if cur not in stages:
            return super()._swap_image()
        nxt = stages.index(cur) + 1
        target = stages[nxt] if nxt < len(stages) else (self._pre_stage_channel or "slide")
        self.background_var.set(target)
        self._on_image_channel_change()
        self.status_var.set(f"showing {self._stage_label(target)}"
                            if target in stages else f"showing {target}")
        return target

    def _reduce_at(self):
        """Where the conversion goes when the chain does not reduce itself."""
        prof = self._profile_from_ui()
        col = ((prof.get("input") or {}).get("color") or {})
        return "end" if col.get("reduce_at") == "end" else "front"

    @staticmethod
    def _stage_index(channel):
        try:
            return int(str(channel).split(":", 1)[1])
        except (IndexError, ValueError):
            return None

    def _cycle_stage_view(self, _event=None):
        """`f`: step through the chain's stages, then back to the picture.

        A chain is judged by looking at what each step does to it, and the
        stages are where the eye already is -- so this walks them in order and
        wraps round to whatever channel was on screen before."""
        cards = [c for c in self.filter_cards if c.get("operation") not in ("none", "", None)]
        if not cards:
            return "break"
        cur = self.background_var.get() or ""
        if cur.startswith(STAGE_CHANNEL):
            nxt = (self._stage_index(cur) or 0) + 1
            target = f"{STAGE_CHANNEL}{nxt}" if nxt < len(cards) else (self._pre_stage_channel or "slide")
        else:
            self._pre_stage_channel = cur or "slide"
            target = f"{STAGE_CHANNEL}0"
        self.background_var.set(target)
        self._on_background_change()
        return "break"

    def _preview_chain_key(self, channel):
        """(cache key, job) for previewing `channel`, or None when the channel
        is not a chain output.

        The key is the chain that produces it and nothing else, so "does what
        is on screen still match the panel?" is one tuple comparison."""
        if channel not in ("base", "filtered") and not channel.startswith(STAGE_CHANNEL):
            return None
        prof = self._profile_from_ui()
        base_filters = prof.get("base_filters") or []
        filters = prof.get("filters") or []
        method = default_color_method(prof)
        if channel.startswith(STAGE_CHANNEL):
            # An intermediate depends on the WHOLE chain -- the plan decides what
            # runs before it -- plus where the conversion goes, so both are in
            # the key.
            key = (channel, json.dumps({"chain": filters, "default": method,
                                        "at": self._reduce_at()}, sort_keys=True))
            return key, (base_filters, filters, method)
        chain = base_filters if channel == "base" else filters
        key = (channel, json.dumps({"chain": chain, "default": method},
                                   sort_keys=True))
        return key, (base_filters, filters, method)

    def _preview_item(self):
        """(item, raw array, geometry) the live preview computes on.

        The item on screen, read at its own level with the halo -- exactly the
        pixels a prime would run the chains over, which is what makes the
        preview honest. Cached, because that read is seconds at a deep level
        and a sigma edit must not repeat it."""
        cur = self._current()
        item = self._item_at(*cur) if cur is not None else None
        if item is None:
            return None, None, None
        stamp = (item.key, int(item.level), self._halo())
        cached = self._preview_arr
        if cached is not None and cached[0] == stamp:
            return item, cached[1], cached[2]
        try:
            arr, geom, dt = self.engine.read_item(item, stamp[2])
        except Exception as exc:
            log(f"preview read failed for {item.key}: {exc}")
            return None, None, None
        log(f"preview read {item.key}: {1e3 * dt:.0f}ms")
        self._preview_arr = (stamp, arr, geom)
        return item, arr, geom

    def _live_channel_source(self, key, channel):
        """The live preview's source for `channel` on item `key`, but only
        while it still matches the panel; otherwise None, and the primed
        raster (or the pyramid) answers as before."""
        shown = self._preview_shown
        if shown is None or shown[0] != key or shown[1] != channel:
            return None
        want = self._preview_chain_key(channel)
        return shown[3] if want is not None and want[0] == shown[2] else None

    def _launch_preview(self):
        """A chain parameter settled: recompute the shown channel off-thread.

        Returns the token submitted, or None when there is nothing to do --
        the slide channel is not a chain output at all, and a channel whose
        chain did not move is one tuple comparison."""
        if self.viewer is None:
            return None
        channel = self.background_var.get() or "slide"
        want = self._preview_chain_key(channel)
        if want is None:
            return None
        key, (base_filters, filters, method) = want
        shown = self._preview_shown
        if shown is not None and shown[1] == channel and shown[2] == key:
            return None                     # already on screen
        item, arr, _geom = self._preview_item()
        if arr is None:
            return None
        hit = self._preview_chan.get(key)
        if hit is not None:                 # retyping the old sigma: instant
            self._show_live(item.key, channel, key, hit[1])
            return None
        try:
            from msseg.mscoupon import mscoupon_py as ext
        except Exception:
            return None
        if channel.startswith(STAGE_CHANNEL):
            job = preview_job("stage", base_filters, filters, method,
                              planar=getattr(arr, "ndim", 2) == 3, label=item.key,
                              upto=self._stage_index(channel), reduce_at=self._reduce_at())
        else:
            job = preview_job("base" if channel == "base" else "filtered",
                              base_filters, filters, method,
                              planar=getattr(arr, "ndim", 2) == 3, label=item.key)
        self._preview_token += 1
        token = self._preview_token
        self._preview_pending = (item, channel, key)
        self.viewer.set_hud("busy", f"Previewing {channel}")
        self.status_var.set(f"preview: computing {channel} on {item.key}...")
        self._worker().submit(token,
                              lambda stop: preview_raster(ext, arr, job, log, stop),
                              sync=self._preview_sync)
        return token

    def _worker(self):
        if self._preview_worker is None:
            from msseg.labeler.preview import PreviewWorker
            self._preview_worker = PreviewWorker(
                self.root, self._on_preview_result, on_error=self._on_preview_error,
                log=log, pump_ms=_PREVIEW_PUMP_MS, name="mspath-preview")
        return self._preview_worker

    def _on_preview_result(self, token, out):
        """A chain recompute landed. Placing a raster measures its range, so
        the source is cached with it -- a repaint must not rescan 16 Mpx."""
        if token != self._preview_token or self._preview_pending is None:
            return
        item, channel, key = self._preview_pending
        self._preview_pending = None
        raster = (out or {}).get("base" if channel == "base" else "filtered")
        geom = self._preview_arr[2] if self._preview_arr else None
        if raster is None or geom is None:      # stopped, or an empty chain
            self._update_busy()
            return
        src = self._place_preview(item, raster, geom)
        if src is None:
            self._update_busy()
            return
        self._preview_chan[key] = (raster, src)
        while len(self._preview_chan) > self._PREVIEW_CHAN_MAX:
            self._preview_chan.pop(next(iter(self._preview_chan)))
        self._show_live(item.key, channel, key, src)

    def _place_preview(self, item, raster, geom):
        """Put a preview raster where its item sits on the slide.

        The halo is compute-only -- a prime trims it off the rasters it stores
        -- so this trims it too, or the picture would sit `halo` pixels up and
        to the left of the thing it is a picture of."""
        import numpy as np
        _level, _lx, _ly, lw, lh, origin, scale = geom
        halo = self._halo()
        if halo:
            # A stage preview may be a plane STACK, (C, h, w). Trim the spatial
            # axes, whichever they are.
            raster = np.ascontiguousarray(
                raster[..., halo:halo + lh, halo:halo + lw] if getattr(raster, "ndim", 2) == 3
                else raster[halo:halo + lh, halo:halo + lw])
        try:
            shape = self.engine.source(item.slide).level_shape(0)
        except Exception as exc:
            log(f"preview placement failed for {item.key}: {exc}")
            return None
        return PlacedImageSource(raster, origin=origin, scale=scale,
                                 slide_shape=shape, path=None)

    def _show_live(self, item_key, channel, key, src):
        self._preview_shown = (item_key, channel, key, src)
        self._refresh_render()
        self._update_busy()
        self.status_var.set(f"preview: {item_key}  [{channel}]")

    def _on_preview_error(self, token, msg):
        if token == self._preview_token:
            self._preview_pending = None
        log(f"preview failed: {msg}")
        self.status_var.set(f"preview failed: {msg}")
        self._update_busy()                 # the canvas keeps what it has

    def _preview_is_stale(self):
        """True while the live preview shows a chain the primed items were
        not built with -- so the region overlays describe a different field."""
        if self._preview_shown is None or self._primed_chain is None:
            return False
        return self._chain_fingerprint() != self._primed_chain

    # ------------------------------------------------------------------ #
    # Parameters
    # ------------------------------------------------------------------ #
    def _profile_for_compute(self):
        """The params the engine primes with.

        Composed by ``session.profile_params_json`` -- the coupon composer, not
        a second one -- so a slide is primed by exactly the pipeline a coupon
        config describes, and the filter cards go through ``filters_to_json``,
        which drops the "not set" blanks the parameter rows leave behind (an
        empty ``stain`` reaches C++ as ``\"\"`` and is refused).

        The plane count is declared as 3: a slide read through the pyramid is
        RGB, and the statistics schema has to resolve before any raster is in
        hand.
        """
        return json.loads(coupon_session.profile_params_json(
            self._profile_from_ui(), 1, SLIDE_PLANES))

    def _profile_from_ui(self):
        active = self.profiles[self.active_profile_idx]
        return {
            "name": active.get("name", "default"),
            # mspath has no colour-input section, so this block is only ever what
            # a loaded profile brought with it -- but it has to survive the UI
            # round trip, or `input.color.default_method` is dropped here and
            # every reader downstream falls back to luminance. The reader is
            # total, so a profile without the block gets the default one.
            "input": {"color": coupon_session.color_input_from_json(active.get("input"))},
            "filters": filters_to_json(self.filter_cards),
            "base_filters": filters_to_json(self.base_cards),
            "msc": {"manifold": self.manifold_var.get(),
                    "persistence_percent": float(self.persist_var.get()),
                    "accurate": bool(self.accurate_var.get()),
                    "use_gpu_gradient": bool(self.gpu_var.get()),
                    "simplification": self.simplification_var.get()},
            "statistics": config_io.statistics_to_json(
                self._stat_channel_cards(), self._stat_reductions(),
                bool(self.stat_extremum_var.get()), 0, False,
                self._stat_histogram(), self._stat_sources()),
            "slide": {"overview_level": self._overview_level(), "halo": self._halo()},
        }

    def _apply_profile_to_ui(self, profile, setvar, notes):
        msc = profile.get("msc") or {}
        setvar(self.manifold_var, msc.get("manifold", "ascending"))
        setvar(self.persist_var, float(msc.get("persistence_percent", 10.0) or 10.0))
        setvar(self.persist_live_var, f"{float(self.persist_var.get()):g}")
        setvar(self.accurate_var, bool(msc.get("accurate")
                                       or msc.get("accurate_ascending")))
        setvar(self.gpu_var, bool(msc.get("use_gpu_gradient")))
        setvar(self.simplification_var,
               str(msc.get("simplification") or coupon_session.DEFAULT_SIMPLIFICATION))
        sl = profile.get("slide") or {}
        setvar(self.level_var, int(sl.get("overview_level", DEFAULT_OVERVIEW_LEVEL)))
        setvar(self.halo_var, int(sl.get("halo", DEFAULT_HALO)))
        stats = config_io.statistics_from_json(profile.get("statistics"), notes)
        self._apply_stat_state({"stat_channels": stats["channels"],
                                "stat_reductions": stats["reductions"],
                                "stat_extremum": stats["extremum"],
                                "stat_histogram": stats.get("histogram")}, setvar)
        self.filter_cards = [dict(c) for c in (profile.get("filters") or [])]
        self.filter_cards.append(self._new_filter_card())
        self.base_cards = [dict(c) for c in (profile.get("base_filters") or [])]
        self.base_cards.append(self._new_filter_card())
        if getattr(self, "filters_frame", None) is not None:
            self._rebuild_filter_cards("topo")
            self._rebuild_filter_cards("base")
        self._update_level_hint()

    def _profile_to_file_doc(self, profile):
        return dict(profile)

    def _profile_from_file_doc(self, doc, notes):
        out = self._default_profile(str((doc or {}).get("name") or "default"))
        out.update({k: v for k, v in (doc or {}).items() if k in
                    ("name", "filters", "base_filters", "msc", "statistics", "slide")})
        return out

    def _session_doc(self):
        """The shell's document, plus each slide's ROIs.

        `build_session_doc` keeps a sequence's name, folder and files and drops
        everything else, which is right for a stack of slices. A slide's ROIs
        are not files -- they are places on one -- so they are re-attached
        here, and read back in `_session_doc_from_json`. Without this a session
        would reload with the annotations intact (their keys carry the
        geometry) and nothing to attach them to.
        """
        doc = super()._session_doc()
        for si, sd in enumerate(doc.get("sequences") or []):
            rois = self._rois_of(si)
            if rois:
                sd["rois"] = [dict(r) for r in rois]
        return doc

    def _apply_session_doc(self, doc, source="session", notes=None):
        """The shell's apply, then the ROIs put back.

        Step 4 of the apply rebuilds each sequence as a fresh
        ``{name, folder, files}`` -- right for a stack of slices, and it drops
        anything else. So the ROIs are re-attached afterwards, matched by
        (folder, slide file) rather than by position: a sequence whose folder
        went missing is skipped, and matching by index would then hand one
        slide's ROIs to another.
        """
        notes = notes if notes is not None else []
        super()._apply_session_doc(doc, source, notes)
        wanted = {}
        for sd in ((doc or {}).get("sequences") or []):
            files = [f for f in (sd.get("files") or []) if isinstance(f, str)]
            if not files:
                continue
            wanted[(str(sd.get("folder") or ""), os.path.basename(files[0]))] =                 _clean_rois(sd.get("rois"), notes)
        for s in self.subsequences:
            files = s.get("files") or []
            if not files:
                continue
            s["rois"] = wanted.get((str(s.get("folder") or ""),
                                    os.path.basename(files[0])), [])
        self._rebuild_flat_slices()
        self._refresh_subseq_list()
        self._update_roi_hint()

    def _run_settings(self):
        return {"overview_level": self._overview_level(), "halo": self._halo()}

    def _apply_run_settings(self, run, setvar, notes):
        if not run:
            return
        setvar(self.level_var, int(run.get("overview_level", DEFAULT_OVERVIEW_LEVEL)))
        setvar(self.halo_var, int(run.get("halo", DEFAULT_HALO)))

    # ------------------------------------------------------------------ #
    # Running
    # ------------------------------------------------------------------ #
    def _on_level_change(self, *_a):
        """A level change makes DIFFERENT items -- the keys carry the level --
        so the primed ones no longer describe anything on screen."""
        self.engine.reset()
        self._rebuild_flat_slices()
        self._update_level_hint()
        self._refresh_render()

    def _update_level_hint(self):
        hint = getattr(self, "level_hint", None)
        if hint is None:
            return
        item = None
        cur = self._current()
        if cur is not None:
            item = self._item_at(*cur)
        if item is None and self.subsequences:
            item = self._item_at(0, 0)
        if item is None:
            hint.set("")
            return
        try:
            src = self.engine.source(item.slide)
            h, w = src.level_shape(self._overview_level())
            hint.set(f"level {self._overview_level()}: {w} x {h} "
                     f"({w * h / 1e6:.1f} Mpx, 1/{src.level_scale(self._overview_level()):g})")
        except Exception as exc:
            hint.set(f"({type(exc).__name__}: {exc})")

    def _rebuild_flat_slices_keeping_current(self):
        cur = self._current()
        key = self.catalogue.key_of(*cur) if cur is not None else None
        self._rebuild_flat_slices()
        if key is not None:
            idx = self.catalogue.index_of(key)
            if idx is not None and idx in self.flat_slices:
                self.slice_var.set(self.flat_slices.index(idx))
                self._sync_slice_combo()

    def _run(self, scope="task"):
        if not self.subsequences:
            self.status_var.set("Add a folder and make a slide list first.")
            return
        items_to_prime = self._prime_items(scope)
        if not items_to_prime:
            self.status_var.set("Nothing to prime - no item is worked "
                                "(enrol the overview or cut an ROI).")
            return
        profile = self._profile_for_compute()
        # A live pipe built under this field (chains, colour, MSC, halo) is
        # kept -- re-measured on the worker if only the statistics moved --
        # and everything else is dropped, as a reset would.
        kept = self.engine.keep_field(profile, self._halo())
        self._run_active = True
        self._set_run_buttons("disabled")
        self._set_load_enabled(False)
        self.status_var.set(f"Priming {len(items_to_prime)} item(s)…")
        n_ov = sum(1 for it in items_to_prime if it.is_overview)
        log(f"RUN: {n_ov} overview(s) at level {self._overview_level()} + "
            f"{len(items_to_prime) - n_ov} ROI(s), halo {self._halo()}")
        log(f"  filters: {[f['operation'] for f in profile['filters']] or ['(none)']}")
        log(f"  base_filters: {[f['operation'] for f in profile['base_filters']] or ['(none)']}")
        if kept:
            log(f"  {kept} live item(s) kept (same field): re-measured where the "
                "statistics moved, not primed")
        if not self.engine.start_run(items_to_prime, profile, halo=self._halo(),
                                     reset_pins=not kept):
            self.status_var.set("A prime is already running.")
            return
        self._ensure_pump()

    def _request_item(self, key):
        """Navigation asked for an item.

        A primed one only needs selecting at the current persistence, which is
        milliseconds and happens inline. One that is not primed -- an ROI just
        cut, or one the live-pipeline LRU released -- is primed on the worker,
        because at 2-9 s it is not something to do on the UI thread.
        """
        if self.engine.record(key) is not None or self.engine.pending_work():
            return
        p = self.engine.primed.get(key)
        if p is not None and p.live:
            profile = self._profile_for_compute()
            if self.engine.measure_stale(key, profile) and self.engine.can_remeasure(key):
                # The statistics moved under a live pipe: re-measure it on the
                # worker (a read and the chains, then the rows -- no MSC).
                self.status_var.set(f"Measuring {p.item.label()}…")
                if self.engine.start_run([p.item], profile, halo=self._halo(),
                                         reset_pins=False, incremental=True,
                                         remeasure_only=True):
                    self._ensure_pump()
                    self._update_busy()
                return
            try:
                self.engine.ensure_record(key, self._profile_for_compute())
            except Exception as exc:
                log(f"{key}: {type(exc).__name__}: {exc}")
            return
        item = parse_key(key)
        if item is None:
            return
        if item.is_overview:
            # Navigating to a slide shows it; Run primes it. An ROI is different
            # -- it exists because the user asked to look there -- and primes
            # on demand. Priming an overview on a click would start ten
            # seconds of compute from a browse.
            return
        self.status_var.set(f"Priming {item.label()}…")
        # reset_pins=False: the thresholds already resolved for this session
        # stay put, or every item primed earlier would silently re-threshold.
        if self.engine.start_run([item], self._profile_for_compute(),
                                 halo=self._halo(), reset_pins=False, incremental=True):
            self._ensure_pump()
            self._update_busy()

    def _handle_compute_event(self, ev):
        kind = ev[0]
        if kind in ("primed", "item_primed"):
            self._run_active = False
            self._set_run_buttons("normal")
            self._set_load_enabled(True)
            # The shell's rebuild resets the navigation to item 0. For a coupon
            # run that is the first slice of a fresh stack; here it is the
            # overview, and the item that was just primed on demand -- the ROI
            # the user navigated to -- would be dropped before its regions were
            # ever computed. Keep the current key across the rebuild.
            # The chains these items were primed with: what a live preview
            # compares itself against. The primed rasters are the truth again.
            self._primed_chain = self._chain_fingerprint()
            self._preview_shown = None
            self._rebuild_flat_slices_keeping_current()
            self._refresh_subseq_list()
            self.status_var.set("Primed.")
            cur = self._current()
            if cur is not None:
                self._request_item(self.catalogue.key_of(*cur))
            self._refresh_render()
        elif kind == "item_done":
            self._refresh_subseq_list()
            self._update_roi_hint()
            self._update_busy()
        elif kind == "item_error":
            _key, msg = ev[1], ev[2]
            self.status_var.set(f"Could not prime {_key}: {msg.splitlines()[0]}")
            self._refresh_subseq_list()
            self._update_busy()

    def _keep_compute_for(self, profile):
        """Keep the live items the new profile's field can reuse (see
        ``SlideEngine.keep_field``); a statistics-only difference then costs a
        re-measure of the item on screen, not a Run."""
        if self._run_active or self.engine.pending_work() or not self.engine.primed:
            return False
        try:
            doc = json.loads(coupon_session.profile_params_json(profile, 1, SLIDE_PLANES))
        except Exception:
            return False
        return self.engine.keep_field(doc, self._halo()) > 0

    def _remeasure_current(self):
        """A new generation; the item on screen is re-selected, or re-measured
        on the worker when the statistics moved (``_request_item``)."""
        if not self.engine.primed or self.engine.pending_work():
            return
        profile = self._profile_for_compute()
        cur = self._current()
        key = self.catalogue.key_of(*cur) if cur is not None else None
        p = self.engine.primed.get(key) if key is not None else None
        if p is not None and p.field == field_fingerprint_of(profile):
            # The primed items follow the panel's statistics from here on.
            self._primed_chain = self._chain_fingerprint()
        self.engine.commit_selection()
        if key is not None:
            self._request_item(key)
        self._update_busy()
        self._refresh_render()

    def _reset_compute(self):
        self.engine.reset()
        # Nothing is primed, so there is nothing for a preview to be stale
        # against; the read cache goes too (a new profile may change the level).
        self._primed_chain = None
        self._preview_shown = None
        self._preview_arr = None
        self._preview_chan.clear()

    def _settle_controls(self):
        self._set_run_buttons("disabled" if self._run_active else "normal")

    def _update_busy(self):
        """The canvas badge: 'Priming' while the item on screen is being
        primed on the worker, else nothing. The labeler's own badges (Training,
        Classifying) call back here when they finish, so this is also what
        takes them down -- with no implementation they stayed up forever."""
        if self.viewer is None:
            return
        cur = self._current()
        key = self.catalogue.key_of(*cur) if cur is not None else None
        if key is not None and self.engine.pending_work() and key in self.engine.running_keys:
            self.viewer.set_hud("busy", "Measuring" if self.engine.running_kind == "measure"
                                else "Priming")
        elif self.engine.pending_work() and self._run_active:
            self.viewer.set_hud("busy", "Priming")
        elif self._preview_pending is not None:
            self.viewer.set_hud("busy", f"Previewing {self._preview_pending[1]}")
        elif self._notice_active():
            return                                  # a _notify is still on screen
        elif self._preview_is_stale():
            self.viewer.set_hud("stale", "Preview - filters changed, Run to re-prime")
        else:
            self.viewer.set_hud(None)

    def _on_persistence_change(self, _event=None):
        """A new threshold is a new parameter generation: records fall stale by
        commit and the visible one is recomputed. Cheap -- the pipelines are
        alive, so this is a select, not a prime."""
        try:
            pct = float(self.persist_live_var.get())
        except (ValueError, tk.TclError):
            self.status_var.set("Persistence must be a number (percent of the field's range).")
            return
        pct = max(0.0, min(100.0, pct))
        if abs(pct - float(self.persist_var.get())) < 1e-9 and self.engine.slices:
            return                                  # nothing changed: no commit bump
        self.persist_var.set(pct)
        self.engine.commit_selection()
        cur = self._current()
        if cur is not None:
            self._request_item(self.catalogue.key_of(*cur))
        self._refresh_render()

    def _update_persist_label(self):
        """The absolute the percentage resolves to for the item on screen."""
        cur = self._current()
        item = self._item_at(*cur) if cur is not None else None
        pin = self.engine.persistence_abs.get(item.level) if item is not None else None
        self.persist_label.set(f"= {pin:.4g}" if pin is not None else "")

    # ------------------------------------------------------------------ #
    # Render
    # ------------------------------------------------------------------ #
    def _region_overlay(self, labels, lut, np, visible=True):
        """Place a region raster on the slide.

        An item's raster is its own -- 2048 square at level 0, or a whole level
        4 at 1/16 -- and the canvas draws in slide pixels, so handing the raster
        over as the framework's default does would draw it at the slide's origin
        at 1:1. Wrapping it in the item's ``RoiLabelLayer`` is what makes the
        class layer, the predictions and a gesture preview land where the
        regions actually are.

        The LUT's length is the id count: it was built for exactly these ids,
        and asking the raster for its maximum on every overlay of every frame is
        the image-sized work the layer exists to avoid.
        """
        cur = self._current()
        item = self._item_at(*cur) if cur is not None else None
        rec = self.engine.record(item.key) if item is not None else None
        if rec is None or labels is None:
            return super()._region_overlay(labels, lut, np, visible)
        try:
            slide_shape = self.engine.source(item.slide).level_shape(0)
        except Exception:
            slide_shape = None
        layer = RoiLabelLayer(labels, origin=rec["origin"], scale=rec["scale"],
                              slide_shape=slide_shape, rev=int(rec["commit"]),
                              n_ids=int(lut.shape[0]))
        return {"layer": layer, "lut": lut, "visible": bool(visible)}

    def _seg_overlays(self, si, li, rec, data, np, min_colors):
        """Overlay list for one item. The framework's signature -- the labeler
        mixin chains to it -- even though `data` (the coupon 3D assembly) has
        no counterpart here."""
        if not self.regions_var.get() or rec is None or self._preview_is_stale():
            # A live preview of an edited chain is on screen: the regions came
            # from a different field, so drawing their boundaries over it
            # would not be slightly stale but simply wrong.
            return []
        key = self.catalogue.key_of(si, li)
        layer = self.regions.label_layer(key) if key else None
        if layer is None:
            return []
        return [self._region_overlay(rec["labels"],
                                     _id_lut(layer.n_ids, min_colors, np), np)]

    def _refresh_render(self):
        self._update_persist_label()
        self._update_busy()
        if self.viewer is None:
            return
        cur = self._current()
        if cur is None:
            return
        try:
            import numpy as np
            from msseg.viz import min_colors
        except Exception:
            return
        item = self._item_at(*cur)
        if item is None:
            return
        try:
            src = self.engine.source(item.slide)
        except Exception as exc:
            self.status_var.set(f"{type(exc).__name__}: {exc}")
            return
        key = item.key
        rec = self.engine.record(key)
        overlays = self._seg_overlays(cur[0], cur[1], rec, None, np, min_colors)

        channel = self.background_var.get() or "slide"
        shown, note = self._channel_source(key, channel, src)
        if note:
            self.status_var.set(note)
        first = not self.viewer.has_base
        self.viewer.set_source(shown, path=self.engine.paths.get(item.slide))
        # Windowed as what is on screen: the slide when the channel fell back
        # to it, else the channel -- each with its own window, first taken
        # from the source's percentiles.
        self.viewer.set_window(*self._window_for(channel if shown is not src else "slide"))
        self.viewer.set_overlays(overlays)
        self.viewer.set_alpha(self.alpha_var.get())
        self._hover_ctx = {"key": key, "rec": rec, "src": src,
                           "channel": channel if shown is not src else "slide",
                           "shown": shown}
        if first:
            self.viewer.fit()
        else:
            self.viewer.render()

    def _preview_file(self, path):
        """Show a slide before any Run: the pyramid IS the preview, so this is
        just pointing the canvas at it."""
        if self.viewer is None:
            return
        try:
            from msseg.labeler.pyramid import PyramidImageSource
            src = PyramidImageSource(str(path))
        except Exception as exc:
            self.status_var.set(f"cannot open {os.path.basename(str(path))}: {exc}")
            return
        self._preview_src = src
        self.viewer.set_source(src, path=str(path))
        self.viewer.set_overlays([])
        self.viewer.set_window(*self._window_for("slide"))
        self.viewer.fit()
        h, w = src.level_shape(0)
        self.status_var.set(f"{os.path.basename(str(path))}: {w} x {h} x {src.channels}, "
                            f"{src.levels} levels ({src.be.name})")

    def _bind_preview(self):
        try:
            self.file_list.bind("<<ListboxSelect>>", self._on_filelist_click)
        except Exception:
            pass

    def _on_filelist_click(self, _event=None):
        sel = list(self.file_list.curselection())
        if sel:
            self._preview_file(self.all_files[sel[0]])

    def _on_hover(self, ix, iy=None):
        ctx = self._hover_ctx
        if ix is None or ctx is None:
            self.hover_var.set("")
            return
        parts = [f"({int(ix)}, {int(iy)})"]
        shown = ctx.get("shown")
        if ctx.get("channel", "slide") != "slide" and hasattr(shown, "value_at"):
            v = shown.value_at(int(ix), int(iy))
            parts.append(f"{ctx['channel']}={v:.4g}" if v is not None else f"{ctx['channel']}=-")
        rec = ctx.get("rec")
        if rec is not None:
            layer = self.regions.label_layer(ctx["key"])
            rid = layer.id_at(int(ix), int(iy)) if layer is not None else -1
            parts.append(f"region {rid}" if rid >= 0 else "region -")
            if rid >= 0:
                row = rec["stats"].row_of_feature(rid)
                if row:
                    for name in ("area", "mean_base"):
                        if name in row:
                            parts.append(f"{name}={row[name]:.4g}")
        self.hover_var.set("   ".join(parts))


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="mspath whole-slide viewer")
    ap.add_argument("folder", nargs="?", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="run the headless integration test and exit")
    args = ap.parse_args(argv)
    if args.selftest:
        from .selftest import run_selftest
        return run_selftest()
    if tk is None:
        print("tkinter is unavailable", file=sys.stderr)
        return 2
    root = tk.Tk()
    MsPathApp(root, initial=args.folder)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
