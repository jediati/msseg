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
from msseg.labeler.widgets import jump_scale
from msseg.mscoupon import session as coupon_session
from msseg.mscoupon.config_io import (FILTER_OPERATIONS, FILTER_SCHEMA, COLOR_METHODS,
                                      filter_param_schema, filters_to_json)

from .adapters import SlideCatalogue, SlideRegionProvider
from .common import list_slides, log
from .engine import SlideEngine
from .items import overview, parse_key, slide_id

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
        self._run_active = False

    def _init_variables(self):
        self.persist_var = tk.DoubleVar(value=10.0)
        self.manifold_var = tk.StringVar(value="ascending")
        self.simplification_var = tk.StringVar(value=coupon_session.DEFAULT_SIMPLIFICATION)
        self.accurate_var = tk.BooleanVar(value=False)
        self.level_var = tk.IntVar(value=DEFAULT_OVERVIEW_LEVEL)
        self.halo_var = tk.IntVar(value=DEFAULT_HALO)
        self.regions_var = tk.BooleanVar(value=True)

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
        for idx, card in enumerate(cards):
            self._build_filter_card(idx, card, chain)

    def _build_filter_card(self, idx, card, chain="topo"):
        cards, parent = self._chain(chain)
        frame = ttk.Frame(parent, relief="groove", borderwidth=1)
        frame.pack(fill="x", padx=4, pady=2)
        top = ttk.Frame(frame); top.pack(fill="x")
        op_var = tk.StringVar(value=card["operation"])
        # `color` consumes the input planes, so only the head of a chain may be one.
        ops = FILTER_OPERATIONS if idx == 0 else [o for o in FILTER_OPERATIONS if o != "color"]
        combo = ttk.Combobox(top, textvariable=op_var, values=ops, state="readonly", width=20)
        combo.pack(side="left", padx=2, pady=2)
        combo.bind("<<ComboboxSelected>>",
                   lambda e, i=idx, v=op_var, c=chain: self._on_filter_op_change(i, v.get(), c))
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
        combo.bind("<<ComboboxSelected>>", on_method)

    def _build_param_row(self, parent, params, pname, kind, default):
        row = ttk.Frame(parent); row.pack(fill="x", padx=6, pady=1)
        ttk.Label(row, text=pname, width=16).pack(side="left")
        if pname not in params:
            params[pname] = default
        if kind == "bool":
            var = tk.BooleanVar(value=bool(params[pname]))
            var.trace_add("write", lambda *_: params.__setitem__(pname, var.get()))
            ttk.Checkbutton(row, variable=var).pack(side="left")
        elif kind.startswith("choice:"):
            choices = kind.split(":", 1)[1].split(",")
            var = tk.StringVar(value=str(params[pname]))
            var.trace_add("write", lambda *_: params.__setitem__(pname, var.get()))
            ttk.Combobox(row, textvariable=var, values=choices, state="readonly",
                         width=12).pack(side="left")
        elif kind in ("str", "floats", "numstr"):
            var = tk.StringVar(value=str(params[pname]))
            var.trace_add("write", lambda *_: params.__setitem__(pname, var.get()))
            ttk.Entry(row, textvariable=var,
                      width=18 if kind == "floats" else 14).pack(side="left")
        elif kind in ("optfloat", "nullfloat"):
            var = tk.StringVar(value=str(params[pname]))

            def commit_opt(*_, p=pname, v=var):
                text = v.get().strip()
                if not text:
                    params[p] = ""
                    return
                try:
                    params[p] = float(text)
                except ValueError:
                    pass
            var.trace_add("write", commit_opt)
            ttk.Entry(row, textvariable=var, width=10).pack(side="left")
        else:                                        # float | int
            var = tk.StringVar(value=str(params[pname]))

            def commit(*_, p=pname, k=kind, v=var):
                try:
                    params[p] = int(v.get()) if k == "int" else float(v.get())
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

    def _remove_filter_card(self, idx, chain="topo"):
        cards, _ = self._chain(chain)
        if 0 <= idx < len(cards):
            del cards[idx]
        if not cards or cards[-1]["operation"] != "none":
            cards.append(self._new_filter_card())
        self._set_chain_cards(chain, cards)
        self._rebuild_filter_cards(chain)

    # ------------------------------------------------------------------ #
    # The item model: one slide, one overview
    # ------------------------------------------------------------------ #
    def _overview_level(self):
        try:
            return max(0, int(self.level_var.get()))
        except Exception:
            return DEFAULT_OVERVIEW_LEVEL

    def _halo(self):
        try:
            return max(0, int(self.halo_var.get()))
        except Exception:
            return DEFAULT_HALO

    def _item_at(self, si, li):
        """The Item at a shell address. A sequence is one slide and (for now)
        carries exactly its overview, so `li` is always 0."""
        try:
            s = self.subsequences[si]
            path = s["files"][li]
        except (IndexError, KeyError, TypeError):
            return None
        sid = slide_id(s.get("folder", ""), path)
        self.engine.register(sid, path)
        return overview(sid, self._overview_level())

    def _enumerate_items(self):
        for si, s in enumerate(self.subsequences):
            for li in range(len(s.get("files") or [])):
                yield (si, li)

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

    def _build_processing_sections(self):
        parent = self._processing_parent("filters")

        slide = ttk.LabelFrame(parent, text="1. Slide")
        slide.pack(fill="x", padx=4, pady=4)
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

        topo = ttk.LabelFrame(parent, text="2. Topology field (the MSC runs on this)")
        topo.pack(fill="x", padx=4, pady=4)
        self.filters_frame = ttk.Frame(topo); self.filters_frame.pack(fill="x")
        self._rebuild_filter_cards("topo")

        base = ttk.LabelFrame(parent, text="3. Base channel (statistics are read from this)")
        base.pack(fill="x", padx=4, pady=4)
        ttk.Label(base, text="Derived from the raw slide, not chained onto the topology field.",
                  foreground="#666").pack(anchor="w", padx=6)
        self.base_frame = ttk.Frame(base); self.base_frame.pack(fill="x")
        self._rebuild_filter_cards("base")

        msc = ttk.LabelFrame(parent, text="4. MSC")
        msc.pack(fill="x", padx=4, pady=4)
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

    def _build_run_section(self):
        parent = self._processing_parent("run")
        frame = ttk.LabelFrame(parent, text="5. Run")
        frame.pack(fill="x", padx=4, pady=4)
        ttk.Label(frame, text="Primes the overview of every listed slide.",
                  foreground="#666").pack(anchor="w", padx=6, pady=(2, 0))
        self.run_btn = ttk.Button(frame, text="Run", command=self._run)
        self.run_btn.pack(fill="x", padx=6, pady=4)

    def _build_live_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Live")
        frame.pack(side="bottom", fill="x")
        row = ttk.Frame(frame); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="persistence %:").pack(side="left")
        self.persist_label = tk.StringVar(value="10.0%")
        ttk.Label(row, textvariable=self.persist_label, width=22).pack(side="right")
        jump_scale(frame, from_=0.0, to=50.0, orient="horizontal",
                   variable=self.persist_var, command=self._on_persistence_change
                   ).pack(fill="x", padx=6)
        ttk.Checkbutton(frame, text="show regions", variable=self.regions_var,
                        command=self._refresh_render).pack(anchor="w", padx=6)

    def _build_segmentation_controls(self, chan):
        ttk.Label(chan, textvariable=getattr(self, "level_hint", tk.StringVar()),
                  foreground="#666").pack(side="left", padx=8)

    def _after_layout(self):
        self._update_level_hint()

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
            self._profile_from_ui(), 1, 3))

    def _profile_from_ui(self):
        return {
            "name": self.profiles[self.active_profile_idx].get("name", "default"),
            "filters": filters_to_json(self.filter_cards),
            "base_filters": filters_to_json(self.base_cards),
            "msc": {"manifold": self.manifold_var.get(),
                    "persistence_percent": float(self.persist_var.get()),
                    "accurate": bool(self.accurate_var.get()),
                    "simplification": self.simplification_var.get()},
            "statistics": {"channels": ["base"],
                           "reductions": ["mean", "min", "max", "std"],
                           "extremum": True},
            "slide": {"overview_level": self._overview_level(), "halo": self._halo()},
        }

    def _apply_profile_to_ui(self, profile, setvar, notes):
        msc = profile.get("msc") or {}
        setvar(self.manifold_var, msc.get("manifold", "ascending"))
        setvar(self.persist_var, float(msc.get("persistence_percent", 10.0) or 10.0))
        setvar(self.accurate_var, bool(msc.get("accurate")
                                       or msc.get("accurate_ascending")))
        setvar(self.simplification_var,
               str(msc.get("simplification") or coupon_session.DEFAULT_SIMPLIFICATION))
        sl = profile.get("slide") or {}
        setvar(self.level_var, int(sl.get("overview_level", DEFAULT_OVERVIEW_LEVEL)))
        setvar(self.halo_var, int(sl.get("halo", DEFAULT_HALO)))
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

    def _run(self):
        if not self.subsequences:
            self.status_var.set("Add a folder and make a slide list first.")
            return
        items_to_prime = []
        for si, li in self._enumerate_items():
            item = self._item_at(si, li)
            if item is not None:
                items_to_prime.append(item)
        if not items_to_prime:
            self.status_var.set("Nothing to prime.")
            return
        profile = self._profile_for_compute()
        self.engine.reset()
        self._run_active = True
        self.run_btn.config(state="disabled")
        self._set_load_enabled(False)
        self.status_var.set(f"Priming {len(items_to_prime)} overview(s)…")
        log(f"RUN: {len(items_to_prime)} overview(s) at level {self._overview_level()}, "
            f"halo {self._halo()}")
        log(f"  filters: {[f['operation'] for f in profile['filters']] or ['(none)']}")
        log(f"  base_filters: {[f['operation'] for f in profile['base_filters']] or ['(none)']}")
        if not self.engine.start_run(items_to_prime, profile, halo=0):
            self.status_var.set("A prime is already running.")
            return
        self._ensure_pump()

    def _request_item(self, key):
        """Navigation asked for an item. Everything the overview tier needs is
        primed by Run, so this only has to select at the current persistence --
        which is milliseconds -- and is done inline."""
        if self.engine.record(key) is not None or self.engine.pending_work():
            return
        p = self.engine.primed.get(key)
        if p is None or not p.live:
            return
        try:
            self.engine.ensure_record(key, self._profile_for_compute())
        except Exception as exc:
            log(f"{key}: {type(exc).__name__}: {exc}")

    def _handle_compute_event(self, ev):
        kind = ev[0]
        if kind == "primed":
            self._run_active = False
            self.run_btn.config(state="normal")
            self._set_load_enabled(True)
            self._rebuild_flat_slices()
            self._refresh_subseq_list()
            self.status_var.set("Primed.")
            cur = self._current()
            if cur is not None:
                self._request_item(self.catalogue.key_of(*cur))
            self._refresh_render()
        elif kind == "item_done":
            self._refresh_subseq_list()

    def _reset_compute(self):
        self.engine.reset()

    def _settle_controls(self):
        if getattr(self, "run_btn", None) is not None:
            self.run_btn.config(state="disabled" if self._run_active else "normal")

    def _update_busy(self):
        pass

    def _on_persistence_change(self, _event=None):
        """A new threshold is a new parameter generation: records fall stale by
        commit and the visible one is recomputed. Cheap -- the pipelines are
        alive, so this is a select, not a prime."""
        self.engine.commit_selection()
        cur = self._current()
        if cur is not None:
            self._request_item(self.catalogue.key_of(*cur))
        self._refresh_render()

    def _update_persist_label(self):
        pct = float(self.persist_var.get())
        pins = self.engine.persistence_abs
        level = self._overview_level()
        pin = pins.get(level)
        self.persist_label.set(f"{pct:.1f}%" + (f" = {pin:.4g}" if pin is not None else ""))

    # ------------------------------------------------------------------ #
    # Render
    # ------------------------------------------------------------------ #
    def _seg_overlays(self, key, rec, np, min_colors):
        """Overlay list for one item (a subclass hook: the labeler appends its
        class layer here)."""
        if not self.regions_var.get() or rec is None:
            return []
        layer = self.regions.label_layer(key)
        if layer is None:
            return []
        return [{"layer": layer, "lut": _id_lut(layer.n_ids, min_colors, np),
                 "visible": True}]

    def _refresh_render(self):
        self._update_persist_label()
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
        overlays = self._seg_overlays(key, rec, np, min_colors)

        first = not self.viewer.has_base
        self.viewer.set_source(src, path=self.engine.paths.get(item.slide))
        lo, hi = src.value_range()
        span = (hi - lo) or 1.0
        self.viewer.set_window(float(self.vmin_var.get()), float(self.vmax_var.get()))
        self.viewer.set_overlays(overlays)
        self.viewer.set_alpha(self.alpha_var.get())
        self._hover_ctx = {"key": key, "rec": rec, "src": src}
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
