"""The frozen region encoder: a small net over the statistics row, applied
with numpy.

An ``EncoderBundle`` is what ``embedding_train`` produces and what a labeler
loads: the input convention (which columns, in what order, which are
log-transformed, their z-score mean and std), a stack of dense layers, and
the whitening of the latent measured over the harvest. ``embed`` takes a
matrix whose columns are named -- any order, extra columns ignored, a missing
one an error -- and returns ``(n, d)`` float32 unit-scale latents.

Inference is three small matmuls, so this module imports nothing beyond
numpy: a collaborator on the pure-Python wheel with no torch loads and
applies an encoder exactly as the machine that trained it. The bundle is a
zip of ``weights.npz`` + ``meta.json`` (see ``save``) so it is inspectable
without this code.

The latent is exposed to the labeler as columns ``emb<hash8>__z00 ..`` (see
``columns``); the hash of the weights sits in the name on purpose, so the
compatibility gate -- a set comparison of names -- refuses a model trained
over another encoder's columns with no new logic. The prefix starts with
none of the conventions' ``mean_`` / ``std_`` / ``hist`` spellings, so no
phantom channel appears. Design: ``docs/design_region_autoencoder.md``.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .table import FeatureTable

BUNDLE_VERSION = 1
PREFIX = "emb"
SEP = "__"
# Activations a layer may name. Inference must not need torch, so the set is
# what numpy expresses in one line.
ACTIVATIONS = ("silu", "relu", "tanh", None)
STD_FLOOR = 1e-6


def _act(name: Optional[str], h: np.ndarray) -> np.ndarray:
    if name is None or name == "none":
        return h
    if name == "silu":
        return h / (1.0 + np.exp(-h))
    if name == "relu":
        return np.maximum(h, 0.0)
    if name == "tanh":
        return np.tanh(h)
    raise ValueError(f"unknown activation {name!r}")


@dataclass
class Layer:
    W: np.ndarray            # (n_in, n_out)
    b: np.ndarray            # (n_out,)
    act: Optional[str] = None


def weights_hash(layers: Sequence[Layer]) -> str:
    h = hashlib.sha1()
    for L in layers:
        h.update(np.ascontiguousarray(L.W, np.float32).tobytes())
        h.update(np.ascontiguousarray(L.b, np.float32).tobytes())
    return h.hexdigest()


def prepare_inputs(X, mean, std, log_mask) -> np.ndarray:
    """The input convention, in one place for training and inference: log1p
    on the flagged columns (clamped at 0), then z-score, then non-finite ->
    0 (which is the column mean after standardisation)."""
    Z = np.array(X, dtype=np.float64, copy=True)
    if Z.ndim != 2:
        raise ValueError("expected a 2-D matrix")
    lm = np.asarray(log_mask, bool)
    if lm.any():
        Z[:, lm] = np.log1p(np.maximum(Z[:, lm], 0.0))
    Z = (Z - np.asarray(mean, np.float64)) / np.asarray(std, np.float64)
    Z[~np.isfinite(Z)] = 0.0
    return Z


@dataclass
class EncoderBundle:
    names: List[str]                  # the input columns, in the encoder's order
    mean: np.ndarray                  # per-column z-score (after the log)
    std: np.ndarray
    log_mask: np.ndarray              # bool per column: log1p first
    layers: List[Layer]
    latent_mean: np.ndarray           # whitening of z measured over the harvest
    latent_std: np.ndarray
    scope: Optional[str] = None       # e.g. "L4"; None when the app has no scope
    meta: Dict[str, Any] = field(default_factory=dict)   # provenance, free-form

    # ------------------------------------------------------------------ #
    @property
    def dim(self) -> int:
        return int(self.layers[-1].W.shape[1]) if self.layers else 0

    @property
    def hash(self) -> str:
        return weights_hash(self.layers)

    @property
    def hash8(self) -> str:
        return self.hash[:8]

    @property
    def n_in(self) -> int:
        return len(self.names)

    def column_names(self) -> List[str]:
        w = max(2, len(str(max(self.dim - 1, 0))))
        return [f"{PREFIX}{self.hash8}{SEP}z{i:0{w}d}" for i in range(self.dim)]

    def describe(self) -> str:
        arch = " -> ".join([str(self.n_in)] + [str(L.W.shape[1]) for L in self.layers])
        return f"{self.meta.get('arch', 'mlp')} {arch} ({self.hash8}, scope {self.scope or 'none'})"

    # ------------------------------------------------------------------ #
    def encode_prepared(self, Z: np.ndarray, whiten: bool = True) -> np.ndarray:
        h = np.asarray(Z, np.float64)
        for L in self.layers:
            h = _act(L.act, h @ np.asarray(L.W, np.float64) + np.asarray(L.b, np.float64))
        if whiten:
            h = (h - self.latent_mean) / self.latent_std
        return h.astype(np.float32)

    def gather(self, X, names: Sequence[str]) -> np.ndarray:
        """The bundle's columns out of `X` (whose columns are `names`), in the
        bundle's order. Raises on a missing one."""
        pos = {str(n): i for i, n in enumerate(names)}
        missing = [n for n in self.names if n not in pos]
        if missing:
            raise ValueError(f"encoder needs column(s) {missing[:6]}"
                             + ("…" if len(missing) > 6 else ""))
        X = np.asarray(X)
        return X[:, [pos[n] for n in self.names]]

    def embed(self, X, names: Sequence[str]) -> np.ndarray:
        """``float32[n, dim]`` unit-scale latents for the rows of `X`."""
        Z = prepare_inputs(self.gather(X, names), self.mean, self.std, self.log_mask)
        return self.encode_prepared(Z)

    def columns(self, table) -> FeatureTable:
        """A NEW table: the record's columns plus the latent as
        ``emb<hash8>__z..`` columns. The input table is untouched."""
        z = self.embed(table.values, list(table.names)).astype(np.float64)
        values = np.concatenate([np.asarray(table.values, np.float64), z], axis=1)
        return FeatureTable(list(table.names) + self.column_names(), values)

    def schema_entries(self) -> List[Dict[str, str]]:
        """One ``{name, channel, reduction}`` per latent column, the block as
        ONE channel so Optimize's feature mask toggles the whole latent."""
        chan = f"{PREFIX}{self.hash8}"
        return [{"name": n, "channel": chan, "reduction": f"z{i}"}
                for i, n in enumerate(self.column_names())]

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def to_meta(self) -> Dict[str, Any]:
        return {"version": BUNDLE_VERSION,
                "names": list(self.names),
                "log_columns": [n for n, m in zip(self.names, self.log_mask) if m],
                "layers": [{"act": L.act, "n_in": int(L.W.shape[0]), "n_out": int(L.W.shape[1])}
                           for L in self.layers],
                "dim": self.dim,
                "scope": self.scope,
                "hash": self.hash,
                "meta": dict(self.meta)}

    def save(self, path: str) -> str:
        buf = io.BytesIO()
        arrays = {"mean": np.asarray(self.mean, np.float64),
                  "std": np.asarray(self.std, np.float64),
                  "log_mask": np.asarray(self.log_mask, bool),
                  "latent_mean": np.asarray(self.latent_mean, np.float64),
                  "latent_std": np.asarray(self.latent_std, np.float64)}
        for i, L in enumerate(self.layers):
            arrays[f"W{i}"] = np.asarray(L.W, np.float32)
            arrays[f"b{i}"] = np.asarray(L.b, np.float32)
        np.savez(buf, **arrays)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("weights.npz", buf.getvalue())
            zf.writestr("meta.json", json.dumps(self.to_meta(), indent=1, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: str) -> "EncoderBundle":
        with zipfile.ZipFile(path, "r") as zf:
            meta = json.loads(zf.read("meta.json").decode("utf-8"))
            with np.load(io.BytesIO(zf.read("weights.npz"))) as npz:
                arrays = {k: npz[k] for k in npz.files}
        if int(meta.get("version", 0)) > BUNDLE_VERSION:
            raise ValueError(f"encoder bundle version {meta['version']} is newer than this reader")
        layers = []
        for i, spec in enumerate(meta["layers"]):
            layers.append(Layer(arrays[f"W{i}"], arrays[f"b{i}"], spec.get("act")))
        names = [str(n) for n in meta["names"]]
        b = cls(names=names, mean=arrays["mean"], std=arrays["std"], log_mask=arrays["log_mask"],
                layers=layers, latent_mean=arrays["latent_mean"], latent_std=arrays["latent_std"],
                scope=meta.get("scope"), meta=dict(meta.get("meta") or {}))
        if meta.get("hash") and meta["hash"] != b.hash:
            raise ValueError("encoder bundle: the weights do not match the recorded hash")
        return b


# --------------------------------------------------------------------------- #
# helpers shared with the trainer
# --------------------------------------------------------------------------- #
def input_convention(X, names: Sequence[str], log_columns: Sequence[str]
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mean, std, log_mask)`` measured over `X` for the bundle's inputs.
    A constant column gets std 1 so it standardises to 0 rather than NaN."""
    names = [str(n) for n in names]
    log_mask = np.array([n in set(log_columns) for n in names], bool)
    Z = np.array(X, dtype=np.float64, copy=True)
    if log_mask.any():
        Z[:, log_mask] = np.log1p(np.maximum(Z[:, log_mask], 0.0))
    Z[~np.isfinite(Z)] = np.nan
    mean = np.nanmean(Z, axis=0)
    std = np.nanstd(Z, axis=0)
    mean[~np.isfinite(mean)] = 0.0
    std = np.where(np.isfinite(std) & (std > STD_FLOOR), std, 1.0)
    return mean, std, log_mask


def is_embedding_column(name: str) -> bool:
    return str(name).startswith(PREFIX) and SEP in str(name)
