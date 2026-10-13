"""Foundation-model features in the labeler (docs/design_foundation_features.md).

One Features-tab group ("8. Foundation features") describes a
``FoundationSpec`` for the ACTIVE task's next model. A region task carries it
inside its ``ContextSpec`` (so the pickle, the model record, the session view,
the compatibility gate and Optimize's feature mask all see it); a polyline
task carries it inside its ``SeamSpec``. Either way the columns are built
here, because they need pixels:

* the token grid comes from ``TokenCache`` (memory + ``<app data>/fm_cache``),
  computed on a miss from ``RegionProvider.item_pixels`` -- an OPTIONAL
  provider method; an app without it refuses the columns with a message;
* the projection basis is fitted at the first Train over the task's ready
  items, frozen onto the spec (``_fm_fitted``) and kept by the model: its
  digest is in every column name, so a refit is a new feature set and the
  name-set gate refuses a model built over another one;
* a region row gains the spec's pools (``_foundation_table``), a seam the
  crossing block (``_foundation_seam_block``).

Everything runs on the Tk thread with the compute badge, like streaming the
records does; the first extraction loads the backbone (seconds), later ones
are a GPU pass per uncached item.
"""
from __future__ import annotations

import os
import time
import tkinter as tk
from tkinter import ttk

from ..foundation import extract as fx
from ..foundation import pool as fp
from ..foundation.cache import TokenCache, cache_key
from ..foundation.project import fit_basis, project
from ..foundation.spec import MODELS, POOLS, FoundationSpec
from ..widgets import attach_tooltip

FM_GROUP_TITLE = "8. Foundation features"
_POOL_TIPS = {
    "mean": "The area-weighted mean token over the cells the region covers.",
    "purity": "The same, each cell weighted by how much of it the region covers "
              "(a cell straddling a boundary counts less).",
    "interior": "Only the cells the region covers wholly (falls back to mean).",
    "boundary": "Only the cells it shares with a neighbour (falls back to mean).",
}


class FoundationMixin:
    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def _init_foundation_vars(self, root):
        self.fm_enabled_var = tk.BooleanVar(master=root, value=False)
        self.fm_model_var = tk.StringVar(master=root, value=FoundationSpec().model)
        self.fm_layers_var = tk.StringVar(master=root, value="-1")
        self.fm_scale_var = tk.StringVar(master=root, value="1")
        self.fm_dims_var = tk.StringVar(master=root, value=str(FoundationSpec().dims))
        self.fm_pool_vars = {p: tk.BooleanVar(master=root, value=p in FoundationSpec().pools)
                             for p in POOLS}
        self.fm_status_var = tk.StringVar(master=root, value="")
        self._fm_fitted = None          # FoundationSpec with a basis, or None
        self._fm_cache = None           # TokenCache, created on first use
        self._fm_projected = {}         # (key, halo, hash8) -> (z, stride, offset)
        # Builds the backbone for a spec (tests inject a fake one).
        self._fm_backbone_for = fx.load_backbone

    def _foundation_spec_from_ui(self):
        """The group's FoundationSpec (None when off), carrying the fitted
        basis when it was fitted for exactly these extraction settings."""
        if not self.fm_enabled_var.get():
            return None
        try:
            layers = tuple(int(x) for x in self.fm_layers_var.get().replace(" ", "").split(",") if x)
        except ValueError:
            layers = (-1,)
        d = {"model": self.fm_model_var.get(), "layers": list(layers or (-1,)),
             "scale": self.fm_scale_var.get(), "dims": self.fm_dims_var.get(),
             "pools": [p for p in POOLS if self.fm_pool_vars[p].get()] or ["mean"]}
        fs = FoundationSpec.from_dict(d)
        fit = self._fm_fitted
        if (fit is not None and fit.extract_key() == fs.extract_key()
                and fit.basis is not None and fit.basis.dims == fs.dims):
            fs = fs.with_basis(fit.basis)
        return fs

    def _apply_foundation_spec(self, fs):
        """Push a FoundationSpec (or None = off) onto the group; a fitted one
        becomes the basis the next Train reuses."""
        self.fm_enabled_var.set(fs is not None)
        if fs is not None:
            self.fm_model_var.set(fs.model)
            self.fm_layers_var.set(",".join(str(i) for i in fs.layers))
            self.fm_scale_var.set(str(fs.scale))
            self.fm_dims_var.set(str(fs.dims))
            for p, var in self.fm_pool_vars.items():
                var.set(p in fs.pools)
            if fs.basis is not None:
                self._fm_fitted = fs
        self._refresh_foundation_status()

    def _on_foundation_change(self, *_a):
        self._refresh_foundation_status()
        try:
            self._refresh_model_readout()
            self._refresh_model_strip()
        except (AttributeError, tk.TclError):
            pass

    def _refit_foundation(self):
        """Forget the fitted basis: the next Train fits a new one (and so a
        new feature set -- models built over the old one are refused)."""
        self._fm_fitted = None
        self._fm_projected.clear()
        self._on_foundation_change()
        self.status_var.set("Foundation projection cleared - the next Train fits a new one.")

    def _refresh_foundation_status(self):
        fs = self._foundation_spec_from_ui()
        if fs is None:
            text = "off"
        elif fs.basis is None:
            text = f"{fs.brief()} - the next Train fits the projection"
        else:
            text = f"{fs.brief()} - columns fm{fs.hash8}__…"
        cache = self._fm_cache
        if cache is not None:
            text += f"  (grids: {cache.misses} computed, {cache.hits + cache.disk_hits} reused)"
        try:
            self.fm_status_var.set(text)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def _build_foundation_group(self):
        body = self._group(self.feat_col, FM_GROUP_TITLE, key="foundation")
        ttk.Label(body, foreground="#666", wraplength=380, justify="left",
                  text="A pretrained vision model's patch tokens, pooled over each region "
                       "(region tasks) or over the cells a seam runs through (polyline "
                       "tasks), as extra columns of the next model. Needs the "
                       "[foundation] extra and an app that hands out pixels.").pack(
            anchor="w", padx=4, pady=(2, 2))
        row = ttk.Frame(body); row.pack(fill="x", padx=4, pady=(0, 2))
        chk = ttk.Checkbutton(row, text="use", variable=self.fm_enabled_var,
                              command=self._on_foundation_change)
        chk.pack(side="left")
        ttk.Label(row, text=" model:").pack(side="left")
        combo = ttk.Combobox(row, textvariable=self.fm_model_var, state="readonly",
                             values=tuple(MODELS), width=18)
        combo.pack(side="left", padx=2)
        combo.bind("<<ComboboxSelected>>", lambda e: (self._unfocus_entries(),
                                                      self._on_foundation_change()))
        attach_tooltip(combo, "DINOv2 (Apache-2.0) or DINOv3 (Meta's licence: accept it on "
                              "the Hugging Face page first). Weights download on first use.")
        row = ttk.Frame(body); row.pack(fill="x", padx=4, pady=(0, 2))
        for label, var, width, tip in (
                ("layers:", self.fm_layers_var, 6,
                 "Block indices, comma-separated, negative from the end (-1 = last). "
                 "Several are concatenated; earlier blocks are more local."),
                ("scale:", self.fm_scale_var, 3,
                 "Upsample the image before the model: 2 halves the patch, for "
                 "regions smaller than a token (slower, off the model's training scale)."),
                ("dims:", self.fm_dims_var, 4,
                 "The projection size (PCA over the task's tokens, fitted at Train).")):
            ttk.Label(row, text=label).pack(side="left")
            en = ttk.Entry(row, textvariable=var, width=width)
            en.pack(side="left", padx=(2, 8))
            en.bind("<Return>", lambda e: (self._unfocus_entries(), self._on_foundation_change()))
            en.bind("<FocusOut>", lambda e: self._on_foundation_change())
            attach_tooltip(en, tip)
        row = ttk.Frame(body); row.pack(fill="x", padx=4, pady=(0, 2))
        ttk.Label(row, text="pools:").pack(side="left")
        for p in POOLS:
            cb = ttk.Checkbutton(row, text=p, variable=self.fm_pool_vars[p],
                                 command=self._on_foundation_change)
            cb.pack(side="left", padx=(2, 4))
            attach_tooltip(cb, _POOL_TIPS[p] + " (Region tasks; a seam always pools "
                                               "the cells it runs through.)")
        row = ttk.Frame(body); row.pack(fill="x", padx=4, pady=(0, 4))
        btn = ttk.Button(row, text="Refit projection", command=self._refit_foundation)
        btn.pack(side="left")
        attach_tooltip(btn, "Forget the fitted PCA basis; the next Train fits a new one. "
                            "The basis is part of the column names, so this is a new "
                            "feature set.")
        ttk.Label(body, textvariable=self.fm_status_var, foreground="#333",
                  wraplength=380, justify="left").pack(anchor="w", padx=4, pady=(0, 4))
        self._refresh_foundation_status()

    # ------------------------------------------------------------------ #
    # grids
    # ------------------------------------------------------------------ #
    def _fm_token_cache(self):
        if self._fm_cache is None:
            root = None
            try:
                from ..session_doc import app_data_dir
                root = os.path.join(app_data_dir(self.SESSION_APP), "fm_cache")
            except Exception:
                root = None
            self._fm_cache = TokenCache(root, max_items=8)
        return self._fm_cache

    def _fm_halo(self, key, rec):
        probe = getattr(self.regions, "pixels_halo", None)
        return int(probe(key, rec)) if probe is not None else 0

    def _fm_grid(self, key, rec, fs, np):
        """``(grid, stride, halo)``: the item's raw token grid, from the cache
        or computed now (ValueError when this app cannot supply pixels or the
        model cannot be loaded)."""
        read = getattr(self.regions, "item_pixels", None)
        if read is None:
            raise ValueError("foundation features need the item's pixels, and this "
                             "app's region provider does not hand them out")
        halo = self._fm_halo(key, rec)
        ck = cache_key(key, rec.get("level"), halo, fs.extract_key())

        def compute():
            try:
                backbone = self._fm_backbone_for(fs)
            except ImportError as exc:
                raise ValueError(str(exc))
            arr, _h = read(key, rec, np)
            t0 = time.perf_counter()
            out = fx.extract_grid(backbone, arr, fs)
            self._log(f"foundation grid {key}: {out[0].shape} in "
                      f"{time.perf_counter() - t0:.2f} s")
            return out
        grid, stride = self._fm_token_cache().get_or_compute(ck, compute)
        return grid, stride, halo

    def _fm_proj(self, key, rec, fs, np):
        """``(z, stride, offset)`` -- the projected grid, memoised per item."""
        halo = self._fm_halo(key, rec)
        mk = (key, halo, fs.hash8)
        hit = self._fm_projected.get(mk)
        if hit is not None:
            return hit
        grid, stride, halo = self._fm_grid(key, rec, fs, np)
        out = (project(grid, fs.basis), stride, (halo, halo))
        if len(self._fm_projected) >= 4:
            self._fm_projected.pop(next(iter(self._fm_projected)))
        self._fm_projected[mk] = out
        return out

    def _fm_fit(self, fs, action="Fitting the foundation projection"):
        """`fs` with a basis fitted over every ready item's grid (computed
        as needed); it becomes the group's fitted spec."""
        import numpy as np
        keys = [k for k in self.regions.keys() if self._stream_ready(k)]
        if not keys:
            raise ValueError("no computed item to fit the foundation projection on")
        grids = []
        self._compute_badge(f"{action} 0/{len(keys)}")
        try:
            for n, key in enumerate(keys, start=1):
                self._compute_badge(f"{action} {n}/{len(keys)}")
                si, li = self.catalogue.index_of(key)
                rec = self._ensure_slice_record(si, li)
                if rec is None or rec.get("labels") is None:
                    continue
                grids.append(self._fm_grid(key, rec, fs, np)[0])
        finally:
            self._clear_compute_badge()
        if not grids:
            raise ValueError("no item could be read for the foundation projection")
        fitted = fs.with_basis(fit_basis(grids, fs.dims))
        self._fm_fitted = fitted
        self._fm_projected.clear()
        self._log(f"foundation projection fitted over {len(grids)} item(s): "
                  f"{fitted.brief()} -> fm{fitted.hash8}")
        self._refresh_foundation_status()
        return fitted

    def _fm_resolved(self, fs):
        """`fs` ready to build columns: fitted now when it has no basis."""
        if fs is None or fs.basis is not None:
            return fs
        return self._fm_fit(fs)

    # ------------------------------------------------------------------ #
    # columns
    # ------------------------------------------------------------------ #
    def _foundation_table(self, key, rec, table, fs, np):
        """`table` + the spec's region pools (a NEW table)."""
        if fs.basis is None:
            raise ValueError("the foundation projection is not fitted - Train fits it")
        z, stride, offset = self._fm_proj(key, rec, fs, np)
        return fp.columns(table, rec["labels"], z, stride, fs, offset, self.FIELDS.id_field)

    def _foundation_seam_block(self, key, rec, graph, fs, np):
        """``(F float32[S, dims + 1], names)``: per seam, the cells it runs through."""
        if fs.basis is None:
            raise ValueError("the foundation projection is not fitted - Train fits it")
        z, stride, offset = self._fm_proj(key, rec, fs, np)
        return (fp.pool_seams(graph, z, stride, offset).astype(np.float32),
                fp.seam_column_names(fs))
