"""The nets, and the PCA baseline.

``RegionEncoder`` is three linears to the latent (SiLU between), plus a
projection head the InfoNCE loss is taken on and a linear reconstruction
head; both heads are discarded when the bundle is written (``to_layers``
exports the encoder alone, as numpy). ``pca_layer`` is the one-linear-layer
baseline -- trial zero -- and needs no torch.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..embedding import Layer


def pca_layer(Z: np.ndarray, dim: int, max_rows: Optional[int] = 200_000, seed: int = 0
              ) -> Tuple[Layer, np.ndarray]:
    """``(layer, explained_variance_ratio)`` of the top-`dim` principal
    components of the standardised inputs `Z` (zero-mean by construction, so
    the bias is zero). A seeded subsample keeps the SVD bounded."""
    Z = np.asarray(Z, np.float64)
    n = Z.shape[0]
    if max_rows is not None and n > int(max_rows):
        rng = np.random.default_rng(int(seed))
        Z = Z[rng.choice(n, int(max_rows), replace=False)]
    Zc = Z - Z.mean(axis=0)
    _u, s, vt = np.linalg.svd(Zc, full_matrices=False)
    d = max(1, min(int(dim), vt.shape[0]))
    var = s ** 2
    ratio = var[:d] / max(float(var.sum()), 1e-12)
    W = np.ascontiguousarray(vt[:d].T, np.float32)               # (n_in, d)
    return Layer(W, np.zeros(d, np.float32), None), ratio


def build_torch_model(n_in: int, hidden: Sequence[int], dim: int, proj: int = 32):
    """The torch ``RegionEncoder``; imported lazily so the module loads
    without torch."""
    import torch
    from torch import nn

    class RegionEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            enc: List[nn.Module] = []
            last = int(n_in)
            for h in hidden:
                enc += [nn.Linear(last, int(h)), nn.SiLU()]
                last = int(h)
            enc.append(nn.Linear(last, int(dim)))
            self.enc = nn.Sequential(*enc)
            self.proj = nn.Sequential(nn.Linear(int(dim), int(proj)), nn.SiLU(),
                                      nn.Linear(int(proj), int(proj)))
            self.rec = nn.Linear(int(dim), int(n_in))

        def forward(self, x):
            z = self.enc(x)
            return z, self.proj(z), self.rec(z)

        def encoder_layers(self) -> List[Layer]:
            out = []
            mods = list(self.enc)
            i = 0
            while i < len(mods):
                lin = mods[i]
                act = None
                if i + 1 < len(mods) and isinstance(mods[i + 1], nn.SiLU):
                    act = "silu"
                    i += 1
                W = lin.weight.detach().cpu().numpy().T.astype(np.float32)   # (n_in, n_out)
                b = lin.bias.detach().cpu().numpy().astype(np.float32)
                out.append(Layer(np.ascontiguousarray(W), b, act))
                i += 1
            return out

    torch.manual_seed(0)
    return RegionEncoder()
