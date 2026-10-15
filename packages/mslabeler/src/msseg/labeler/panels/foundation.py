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

Extraction (and the projection fit) runs OFF the Tk thread (``foundation.worker.GridWorker``): a
model operation that meets an uncached grid enqueues what it needs -- every
ready item for Train / Classify all / Evaluate, the item on screen for
classify-current -- returns at once with a status line, and re-runs itself
when those grids have landed (``fm_deferred`` on its entry point,
``_fm_defer`` here). Outside such an operation a miss enqueues the item and
reads as "not available yet". ``_fm_sync`` (the selftests) computes inline.
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
from ..foundation.worker import GridWorker, GridsPending
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
        self._fm_worker = None          # GridWorker, created on first use
        self._fm_retries = {}           # operation name -> (retry, cache keys waited on)
        self._fm_sync = False           # True: extract inline (the selftests)
        self._fm_fit_jobs = {}          # worker key of a basis fit -> (spec, n items)

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
        w = self._fm_worker
        if w is not None and w.pending():
            text += f"  - extracting {w.pending()} item(s)"
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
        cbtn = ttk.Button(row, text="Cancel extraction", command=self._cancel_foundation)
        cbtn.pack(side="left", padx=(6, 0))
        attach_tooltip(cbtn, "Drop the token grids still queued (the one being computed "
                             "finishes); the operation waiting on them does not run.")
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

    def _fm_key(self, key, rec, fs):
        level = rec.get("level") if rec is not None else None
        return cache_key(key, level, self._fm_halo(key, rec), fs.extract_key())

    def _fm_job(self, key, rec, fs):
        """The extraction of one item's grid, as a closure the worker runs:
        nothing in it touches Tk or the cache."""
        import numpy as np
        read = self.regions.item_pixels
        factory = self._fm_backbone_for
        log = self._log

        def job():
            try:
                backbone = factory(fs)
            except ImportError as exc:
                raise ValueError(str(exc))
            arr, _h = read(key, rec, np)
            t0 = time.perf_counter()
            out = fx.extract_grid(backbone, arr, fs)
            log(f"foundation grid {key}: {out[0].shape} in {time.perf_counter() - t0:.2f} s")
            return out
        return job

    def _fm_grid(self, key, rec, fs, np):
        """``(grid, stride, halo)``: the item's raw token grid from the cache.
        On a miss the extraction is queued on the worker and this raises
        ``GridsPending`` inside a deferred operation (ValueError outside one);
        ValueError too when this app cannot supply pixels."""
        if getattr(self.regions, "item_pixels", None) is None:
            raise ValueError("foundation features need the item's pixels, and this "
                             "app's region provider does not hand them out")
        halo = self._fm_halo(key, rec)
        ck = self._fm_key(key, rec, fs)
        cache = self._fm_token_cache()
        hit = cache.get(ck)
        if hit is None and self._fm_sync:
            hit = cache.put(ck, *self._fm_job(key, rec, fs)())
        if hit is not None:
            return hit[0], hit[1], halo
        scopes = self.__dict__.get("_fm_scopes") or []
        wanted = self._fm_request(fs, key, rec, everything=bool(scopes) and scopes[-1] == "all")
        if scopes:
            raise GridsPending(wanted)
        raise ValueError("its foundation tokens are being extracted - try again in a moment")

    def _fm_worker_get(self):
        if self._fm_worker is None:
            self._fm_worker = GridWorker(self.root, self._fm_job_done, self._fm_job_error,
                                         log=self._log)
        return self._fm_worker

    def _fm_request(self, fs, key, rec, everything=False):
        """Queue `key`'s grid (and, with `everything`, every other ready
        item's that is not cached): the set of cache keys to wait for."""
        worker = self._fm_worker_get()
        cache = self._fm_token_cache()
        wanted = set()
        jobs = [(key, rec)]
        if everything:
            jobs += [(k, self.regions.record(k)) for k in self.regions.keys()
                     if k != key and self._stream_ready(k)]
        for k, r in jobs:
            ck = self._fm_key(k, r, fs)
            if ck in cache and not worker.is_pending(ck):
                continue
            worker.submit(ck, self._fm_job(k, r, fs), label=str(k))
            wanted.add(ck)
        self._refresh_foundation_status()
        return wanted

    def _fm_defer(self, name, retry, keys):
        """Run `retry` once every grid in `keys` is cached (the same
        operation asked twice keeps only its latest call)."""
        worker = self._fm_worker_get()
        keys = {k for k in keys if worker.is_pending(k)}
        if not keys:
            self.root.after(0, retry)
            return
        self._fm_retries[name] = (retry, keys)
        n = worker.pending()
        self.status_var.set(f"Extracting foundation tokens for {n} item(s) on the GPU - "
                            f"{name.strip('_').replace('_', ' ')} continues when they are ready.")
        self._refresh_foundation_status()

    def _fm_job_done(self, ck, out):
        fit = self._fm_fit_jobs.pop(ck, None)
        if fit is not None:
            self._fm_install_basis(fit[0], out, fit[1])
        else:
            grid, stride = out
            self._fm_token_cache().put(ck, grid, stride)
        ready = []
        for name, (retry, keys) in list(self._fm_retries.items()):
            keys.discard(ck)
            if not keys:
                ready.append(retry)
                del self._fm_retries[name]
        w = self._fm_worker
        if w is not None and w.pending() and self._fm_retries:
            self.status_var.set(f"Extracting foundation tokens: {w.pending()} item(s) to go.")
        self._refresh_foundation_status()
        for retry in ready:
            self.root.after(0, retry)

    def _fm_job_error(self, ck, msg):
        dropped = [name for name, (_r, keys) in self._fm_retries.items() if ck in keys]
        for name in dropped:
            del self._fm_retries[name]
        self._log(f"foundation extraction failed ({ck}): {msg}")
        self.status_var.set(f"Foundation extraction failed: {msg}"
                            + (f" - {', '.join(n.strip('_') for n in dropped)} did not run"
                               if dropped else ""))
        self._refresh_foundation_status()

    def _cancel_foundation(self):
        n = self._fm_worker.cancel() if self._fm_worker is not None else 0
        waiting = len(self._fm_retries)
        self._fm_retries.clear()
        self._fm_fit_jobs.clear()
        self.status_var.set(f"Foundation extraction cancelled ({n} item(s) dropped"
                            + (f", {waiting} operation(s) will not run)." if waiting else ")."))
        self._refresh_foundation_status()

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
        if self._fm_sync or not self.__dict__.get("_fm_scopes"):
            return self._fm_install_basis(fs, fit_basis(grids, fs.dims), len(grids))
        # The PCA is seconds over a large sample: the worker fits it too, and
        # the operation re-runs with the basis in place.
        ck = f"fit|{fs.extract_key()}|{fs.dims}|{len(grids)}"
        worker = self._fm_worker_get()
        if not worker.is_pending(ck):
            self._fm_fit_jobs[ck] = (fs, len(grids))
            worker.submit(ck, lambda g=grids, d=fs.dims: fit_basis(g, d), label="projection")
        raise GridsPending({ck})

    def _fm_install_basis(self, fs, basis, n_items):
        fitted = fs.with_basis(basis)
        self._fm_fitted = fitted
        self._fm_projected.clear()
        self._log(f"foundation projection fitted over {n_items} item(s): "
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
