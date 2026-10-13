"""The token grid: a ViT's per-patch output tokens, stitched over an item.

A ViT reads a TILE (``spec.tile`` px, e.g. 448) and keeps one token per
PATCH (14 or 16 px) through every block -- the tile is only how the model is
fed; the patch is the resolution. ``extract_grid`` tiles the item with
overlap, keeps from each tile the tokens nearest its centre (an interior tile
edge loses ``overlap_tokens`` of context, so those are taken from the
neighbour that sees them better) and returns ONE ``(Gh, Gw, D)`` grid whose
cell ``(i, j)`` covers pixels ``[i*s, (i+1)*s) x [j*s, (j+1)*s)`` of the
input, ``s = patch / scale``. CLS and register tokens are never returned.

The model is reached through a tiny backbone protocol so the tiling is
testable without a download:

    backbone.patch -> int
    backbone.mean, backbone.std -> 3 floats (input normalisation)
    backbone(x: (B, 3, h, w) tensor) -> list[(B, D_l, h/patch, w/patch)], one per layer

``load_backbone`` builds the real one from timm. Torch and timm are imported
lazily: the rest of the package is numpy-only.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .spec import FoundationSpec, timm_name

_BACKBONES: Dict[Tuple[str, Optional[str], Tuple[int, ...], str], "TimmBackbone"] = {}


def log(msg: str) -> None:
    print(f"[foundation] {msg}", flush=True)


class TimmBackbone:
    """A timm ViT behind the backbone protocol (``forward_intermediates``)."""

    def __init__(self, model, layers: Sequence[int]):
        self.model = model
        n_blocks = len(getattr(model, "blocks", ())) or None
        idx = []
        for i in layers:
            i = int(i)
            if i < 0:
                if n_blocks is None:
                    raise ValueError("negative layer index needs a model with .blocks")
                i += n_blocks
            idx.append(i)
        self.indices = idx
        ps = model.patch_embed.patch_size
        self.patch = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
        cfg = getattr(model, "pretrained_cfg", None) or {}
        self.mean = tuple(cfg.get("mean", (0.485, 0.456, 0.406)))
        self.std = tuple(cfg.get("std", (0.229, 0.224, 0.225)))

    def __call__(self, x):
        out = self.model.forward_intermediates(x, indices=self.indices,
                                               intermediates_only=True, output_fmt="NCHW")
        return list(out)


def default_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_backbone(spec: FoundationSpec, device: Optional[str] = None) -> TimmBackbone:
    """The spec's model on `device`, memoised per process. The first call may
    download the weights into timm's / Hugging Face's cache (logged)."""
    device = device or default_device()
    key = (spec.model, spec.weights, tuple(spec.layers), device)
    bb = _BACKBONES.get(key)
    if bb is not None:
        return bb
    try:
        import timm
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("foundation features need timm and torch: "
                          "pip install 'msseg-labeler[foundation]'") from e
    name = timm_name(spec.model)
    kw = {"pretrained": True, "dynamic_img_size": True}
    if spec.weights:
        kw["pretrained_cfg_overlay"] = {"file": spec.weights}
        log(f"loading {name} from {spec.weights}")
    else:
        log(f"loading {name} (downloads into the hub cache on first use)")
    t0 = time.perf_counter()
    model = timm.create_model(name, **kw).eval().to(device)
    bb = TimmBackbone(model, spec.layers)
    log(f"loaded {name}: patch {bb.patch}, blocks {len(getattr(model, 'blocks', ()))}, "
        f"layers {bb.indices} ({time.perf_counter() - t0:.1f} s)")
    _BACKBONES[key] = bb
    return bb


def release() -> None:
    """Drop every memoised backbone and return its VRAM to the allocator's
    pool and the driver (the diffg bank and torch_mlp share the card)."""
    _BACKBONES.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# tiling
# --------------------------------------------------------------------------- #
def spans(G: int, n: int, overlap: int) -> List[Tuple[int, int, int]]:
    """Tiles along one axis of a ``G``-token grid, ``n`` tokens per tile:
    ``(start, lo, hi)`` -- the tile covers ``[start, start + min(n, G))`` and
    OWNS ``[lo, hi)``. The owned ranges partition ``[0, G)``, and a boundary
    between two tiles sits mid-way through their overlap, so a kept token is
    at least ~``overlap`` tokens from an interior tile edge."""
    if G <= n:
        return [(0, 0, G)]
    step = max(1, n - 2 * int(overlap))
    starts = list(range(0, G - n + 1, step))
    if starts[-1] != G - n:
        starts.append(G - n)
    out = []
    lo = 0
    for i, s in enumerate(starts):
        hi = G if i == len(starts) - 1 else (starts[i + 1] + s + n) // 2
        out.append((s, lo, hi))
        lo = hi
    return out


def _as_rgb(img) -> np.ndarray:
    """(3, H, W) float32 from (H, W), (C, H, W) or (H, W, C) with C in 1..4
    (gray is replicated, a 4th / alpha plane dropped)."""
    a = np.asarray(img, np.float32)
    if a.ndim == 2:
        a = a[None]
    elif a.ndim == 3 and a.shape[0] not in (1, 2, 3, 4) and a.shape[2] in (1, 2, 3, 4):
        a = np.transpose(a, (2, 0, 1))
    if a.ndim != 3:
        raise ValueError(f"expected an image, got shape {a.shape}")
    if a.shape[0] in (1, 2):
        a = np.repeat(a[:1], 3, axis=0)
    elif a.shape[0] >= 4:
        a = a[:3]
    return np.ascontiguousarray(a)


def grid_stride(backbone, spec: FoundationSpec) -> float:
    """Pixels of the INPUT image per grid cell."""
    return backbone.patch / float(spec.scale)


def extract_grid(backbone, img, spec: FoundationSpec, device: Optional[str] = None,
                 batch: int = 8, dtype: str = "float16") -> Tuple[np.ndarray, float]:
    """``(grid, stride)``: the ``(Gh, Gw, D)`` token grid of `img` (``D`` the
    concatenated widths of the spec's layers, stored as `dtype`) and the
    input pixels per cell. ``Gh = ceil(H * scale / patch)``; the last row and
    column of cells may overhang the image (it is edge-padded)."""
    import torch
    import torch.nn.functional as F

    device = device or default_device()
    p = int(backbone.patch)
    if spec.tile % p:
        raise ValueError(f"tile {spec.tile} is not a multiple of the patch {p}")
    n = spec.tile // p
    x = torch.from_numpy(_as_rgb(img)).to(device)
    x = x / float(spec.value_range)
    mean = torch.tensor(backbone.mean, dtype=x.dtype, device=device).view(3, 1, 1)
    std = torch.tensor(backbone.std, dtype=x.dtype, device=device).view(3, 1, 1)
    x = ((x - mean) / std)[None]
    if spec.scale != 1:
        x = F.interpolate(x, scale_factor=float(spec.scale), mode="bilinear", align_corners=False)
    H, W = x.shape[-2:]
    Gh, Gw = math.ceil(H / p), math.ceil(W / p)
    if Gh * p != H or Gw * p != W:
        x = F.pad(x, (0, Gw * p - W, 0, Gh * p - H), mode="replicate")
    sy, sx = spans(Gh, n, spec.overlap_tokens), spans(Gw, n, spec.overlap_tokens)
    th, tw = min(n, Gh), min(n, Gw)
    jobs = [(a, b) for a in sy for b in sx]
    use_amp = device.startswith("cuda") and dtype == "float16"
    out = None
    i, bs = 0, max(1, int(batch))
    with torch.inference_mode():
        while i < len(jobs):
            chunk = jobs[i:i + bs]
            tiles = torch.cat([x[..., a[0] * p:(a[0] + th) * p, b[0] * p:(b[0] + tw) * p]
                               for a, b in chunk])
            try:
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    feats = backbone(tiles)
            except torch.cuda.OutOfMemoryError:
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                torch.cuda.empty_cache()
                log(f"out of memory; batch -> {bs}")
                continue
            f = torch.cat([t.float() for t in feats], dim=1)          # (B, D, th, tw)
            if f.shape[-2:] != (th, tw):
                raise ValueError(f"backbone returned a {tuple(f.shape[-2:])} token grid "
                                 f"for a {th}x{tw}-token tile")
            f = f.permute(0, 2, 3, 1).to("cpu").numpy()
            if out is None:
                out = np.zeros((Gh, Gw, f.shape[-1]), dtype=dtype)
            for k, ((y0, ylo, yhi), (x0, xlo, xhi)) in enumerate(chunk):
                out[ylo:yhi, xlo:xhi] = f[k, ylo - y0:yhi - y0, xlo - x0:xhi - x0]
            i += len(chunk)
    return out, grid_stride(backbone, spec)
