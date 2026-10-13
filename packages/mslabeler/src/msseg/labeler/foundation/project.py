"""The projection from model width (768 / 1024 / ...) to ``spec.dims``.

A PCA over a seeded sample of tokens, whitened so every projected component
has unit variance over that sample (the region head z-scores anyway; unit
scale keeps a pooled column readable). It is unsupervised and FROZEN onto
the spec once fitted -- the basis digest is part of ``spec.hash8`` and hence
of every column name, so refitting is a new feature set by construction.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .spec import Basis


def fit_basis(grids: Iterable[np.ndarray], dims: int, max_tokens: int = 200_000,
              seed: int = 0) -> Basis:
    """Fit over the cells of `grids` (each ``(Gh, Gw, D)``), sampling at most
    `max_tokens` of them in proportion to each grid's size."""
    grids = [np.asarray(g) for g in grids]
    if not grids:
        raise ValueError("no token grids to fit a projection on")
    D = int(grids[0].shape[-1])
    sizes = np.array([g.shape[0] * g.shape[1] for g in grids], np.int64)
    total = int(sizes.sum())
    rng = np.random.default_rng(seed)
    take = np.minimum(sizes, np.ceil(sizes * min(1.0, max_tokens / max(total, 1))).astype(np.int64))
    rows = []
    for g, k in zip(grids, take):
        flat = g.reshape(-1, D)
        idx = rng.choice(flat.shape[0], size=int(k), replace=False) if k < flat.shape[0] else slice(None)
        rows.append(np.asarray(flat[idx], np.float32))
    X = np.concatenate(rows)
    mean = X.mean(0)
    Xc = X - mean
    k = int(min(dims, D, X.shape[0]))
    _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    comps = vt[:k]
    scale = s[:k] / np.sqrt(max(X.shape[0] - 1, 1))
    scale = np.where(scale > 1e-12, scale, 1.0)
    return Basis(mean=mean.astype(np.float32), components=comps.astype(np.float32),
                 scale=scale.astype(np.float32))


def project(grid: np.ndarray, basis: Optional[Basis]) -> np.ndarray:
    """``(Gh, Gw, k)`` float32; with no basis the raw grid as float32."""
    g = np.asarray(grid, np.float32)
    if basis is None:
        return g
    flat = g.reshape(-1, g.shape[-1])
    z = ((flat - basis.mean) @ basis.components.T) / basis.scale
    return z.reshape(g.shape[0], g.shape[1], -1).astype(np.float32)
