"""``FoundationSpec``: which foundation model, read how, pooled how -- as plain
data (the experiment's variants; later the pickle's ``stack["foundation"]``
and the model record).

Two digests split what the spec controls:

* ``extract_key()`` -- what the TOKEN GRID is a function of (model, weights,
  layers, tile, overlap, scale, value range). A grid cache keys on it, so a
  new projection or pool choice reuses every grid already computed.
* ``hash8`` -- everything a COLUMN is a function of: the extract key plus the
  projection basis. It sits in the column names (``fm<hash8>__mean_07``), so
  the name-set compatibility gate refuses a model trained over another
  model's / basis's columns with no new logic (the ``EncoderBundle`` trick).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

PREFIX = "fm"
SEP = "__"

# Short names -> timm model names. Any other timm name is accepted verbatim as
# ``timm:<name>``. DINOv2 is Apache-2.0; DINOv3's weights carry Meta's own
# licence (accept it on the Hugging Face model page before the first load).
MODELS: Dict[str, str] = {
    "dinov2_vits14_reg": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov2_vitb14_reg": "vit_base_patch14_reg4_dinov2.lvd142m",
    "dinov2_vitl14_reg": "vit_large_patch14_reg4_dinov2.lvd142m",
    "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",
    "dinov3_vitl16": "vit_large_patch16_dinov3.lvd1689m",
}
DEFAULT_MODEL = "dinov2_vitb14_reg"

# How a region gathers its tokens (see pool.py).
POOLS = ("mean", "purity", "interior", "boundary")


def timm_name(model: str) -> str:
    if model.startswith("timm:"):
        return model[5:]
    if model in MODELS:
        return MODELS[model]
    raise ValueError(f"unknown foundation model {model!r}; known: {sorted(MODELS)} "
                     "or 'timm:<name>'")


@dataclass
class Basis:
    """A PCA projection: ``(x - mean) @ components.T``, whitened by ``scale``."""
    mean: np.ndarray            # (D,)
    components: np.ndarray      # (k, D)
    scale: np.ndarray           # (k,)

    @property
    def dims(self) -> int:
        return int(self.components.shape[0])

    def digest(self) -> str:
        h = hashlib.sha1()
        for a in (self.mean, self.components, self.scale):
            h.update(np.ascontiguousarray(a, np.float32).tobytes())
        return h.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": np.asarray(self.mean, np.float32).tolist(),
                "components": np.asarray(self.components, np.float32).tolist(),
                "scale": np.asarray(self.scale, np.float32).tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Basis":
        return cls(mean=np.asarray(d["mean"], np.float32),
                   components=np.asarray(d["components"], np.float32),
                   scale=np.asarray(d["scale"], np.float32))


@dataclass
class FoundationSpec:
    model: str = DEFAULT_MODEL
    weights: Optional[str] = None           # local checkpoint path; None = timm's hub cache
    layers: Tuple[int, ...] = (-1,)         # block indices, concatenated (negative = from the end)
    tile: int = 448                         # model input side, pixels (a multiple of the patch)
    overlap_tokens: int = 2                 # tokens dropped at an interior tile edge
    scale: int = 1                          # image upsampling before the model (2 = half-size patches)
    value_range: float = 255.0              # raw intensity that maps to 1.0
    dims: int = 64                          # projection size
    pools: Tuple[str, ...] = ("mean", "purity")
    basis: Optional[Basis] = field(default=None, compare=False)

    # ------------------------------------------------------------------ #
    def extract_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "weights": self.weights, "layers": list(self.layers),
                "tile": int(self.tile), "overlap_tokens": int(self.overlap_tokens),
                "scale": int(self.scale), "value_range": float(self.value_range)}

    def extract_key(self) -> str:
        text = json.dumps(self.extract_dict(), sort_keys=True)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    @property
    def hash8(self) -> str:
        h = hashlib.sha1(self.extract_key().encode("utf-8"))
        if self.basis is not None:
            h.update(self.basis.digest().encode("utf-8"))
        return h.hexdigest()[:8]

    def with_basis(self, basis: Optional[Basis]) -> "FoundationSpec":
        return replace(self, basis=basis, dims=basis.dims if basis is not None else self.dims)

    # ------------------------------------------------------------------ #
    def to_dict(self, include_basis: bool = True) -> Dict[str, Any]:
        d = self.extract_dict()
        d.update(dims=int(self.dims), pools=list(self.pools))
        if include_basis and self.basis is not None:
            d["basis"] = self.basis.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Any) -> "FoundationSpec":
        """Tolerant: unknown pools are dropped, bad numbers fall back to the
        defaults; an unknown model name is kept (it fails at load, loudly)."""
        d = dict(d or {}) if isinstance(d, dict) else {}
        dflt = cls()

        def num(key, kind, lo):
            try:
                return max(lo, kind(d.get(key, getattr(dflt, key))))
            except (TypeError, ValueError):
                return getattr(dflt, key)

        try:
            layers = tuple(int(x) for x in (d.get("layers") or dflt.layers))
        except (TypeError, ValueError):
            layers = dflt.layers
        pools = tuple(p for p in dict.fromkeys(str(p) for p in (d.get("pools") or ())) if p in POOLS)
        basis = d.get("basis")
        return cls(model=str(d.get("model") or dflt.model),
                   weights=(str(d["weights"]) if d.get("weights") else None),
                   layers=layers or dflt.layers,
                   tile=num("tile", int, 14), overlap_tokens=num("overlap_tokens", int, 0),
                   scale=num("scale", int, 1), value_range=num("value_range", float, 1e-6),
                   dims=num("dims", int, 1), pools=pools or dflt.pools,
                   basis=Basis.from_dict(basis) if isinstance(basis, dict) else None)

    def brief(self) -> str:
        """`fm: dinov2_vitb14_reg L-1 x1 -> 64 (mean,purity)`."""
        layers = ",".join(str(i) for i in self.layers)
        x = f" x{self.scale}" if self.scale != 1 else ""
        fit = "" if self.basis is not None else " (unfitted)"
        return f"fm: {self.model} L{layers}{x} -> {self.dims} ({','.join(self.pools)}){fit}"


def column_prefix(spec: FoundationSpec) -> str:
    return f"{PREFIX}{spec.hash8}{SEP}"


def column_names(spec: FoundationSpec) -> list:
    """The columns ``pool.columns`` appends: per pool ``<pool>_<kk>``, then the
    two reliability columns ``tokens`` and ``purity_frac``."""
    pre = column_prefix(spec)
    w = max(2, len(str(max(int(spec.dims) - 1, 0))))
    out = [f"{pre}{p}_{k:0{w}d}" for p in spec.pools for k in range(int(spec.dims))]
    return out + [f"{pre}tokens", f"{pre}purity_frac"]


def schema_entries(spec: FoundationSpec) -> list:
    """``{name, channel, reduction}`` per column, each pool ONE channel so
    Optimize's feature mask toggles a pool as a group (the two reliability
    columns ride as their own group)."""
    pre = column_prefix(spec)
    out = []
    for n in column_names(spec):
        tail = n[len(pre):]
        if tail in ("tokens", "purity_frac"):
            out.append({"name": n, "channel": f"{PREFIX}{spec.hash8}:reliability", "reduction": tail})
        else:
            pool, _, k = tail.rpartition("_")
            out.append({"name": n, "channel": f"{PREFIX}{spec.hash8}:{pool}", "reduction": f"z{k}"})
    return out


def is_foundation_column(name: str) -> bool:
    return name.startswith(PREFIX) and SEP in name and len(name.partition(SEP)[0]) == len(PREFIX) + 8
