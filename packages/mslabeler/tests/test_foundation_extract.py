"""The tiled token grid (msseg.labeler.foundation.extract) against a tiny fake
backbone -- no download, no timm. Skipped without torch; CPU is enough.

    pytest packages/mslabeler/tests/test_foundation_extract.py
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from msseg.labeler.foundation import extract as fx
from msseg.labeler.foundation.spec import FoundationSpec


class LocalBackbone:
    """A patch-local 'ViT': each token is a fixed linear map of its own patch,
    so a correctly stitched grid is independent of the tiling."""
    def __init__(self, patch=4, dim=6, layers=1):
        self.patch, self.mean, self.std = patch, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        g = torch.Generator().manual_seed(0)
        self.convs = [torch.nn.Conv2d(3, dim, patch, stride=patch) for _ in range(layers)]
        for c in self.convs:
            with torch.no_grad():
                c.weight.copy_(torch.randn(c.weight.shape, generator=g))
                c.bias.zero_()
        self.calls = []

    def __call__(self, x):
        self.calls.append(tuple(x.shape))
        return [c(x) for c in self.convs]


@pytest.mark.parametrize("G,n,o", [(5, 8, 2), (8, 8, 2), (9, 8, 2), (30, 8, 2), (30, 8, 0), (31, 5, 1)])
def test_spans_partition(G, n, o):
    sp = fx.spans(G, n, o)
    owned = [t for _s, lo, hi in sp for t in range(lo, hi)]
    assert owned == list(range(G))
    for s, lo, hi in sp:
        size = min(n, G)
        assert s <= lo and hi <= s + size and 0 <= s and s + size <= G


@pytest.mark.parametrize("shape,scale", [((37, 53), 1), ((3, 37, 53), 1), ((20, 26, 3), 2), ((9, 9), 1)])
def test_stitched_grid_equals_one_pass(shape, scale):
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, size=shape).astype(np.float32)
    bb = LocalBackbone(patch=4, dim=6, layers=2)
    tiled = FoundationSpec(tile=16, overlap_tokens=1, scale=scale)
    whole = FoundationSpec(tile=4 * 64, overlap_tokens=0, scale=scale)
    g1, s1 = fx.extract_grid(bb, img, tiled, device="cpu", batch=3, dtype="float32")
    g2, s2 = fx.extract_grid(bb, img, whole, device="cpu", dtype="float32")
    assert s1 == s2 == 4 / scale
    h, w = (shape[:2] if len(shape) == 2 or shape[-1] == 3 else shape[1:])
    assert g1.shape == (int(np.ceil(h * scale / 4)), int(np.ceil(w * scale / 4)), 12)
    np.testing.assert_allclose(g1, g2, rtol=1e-5, atol=1e-4)


def test_tiles_are_batched_and_sized():
    bb = LocalBackbone(patch=4)
    img = np.zeros((3, 100, 100), np.float32)
    fx.extract_grid(bb, img, FoundationSpec(tile=32, overlap_tokens=2), device="cpu", batch=4)
    assert all(c[1:] == (3, 32, 32) for c in bb.calls)
    assert max(c[0] for c in bb.calls) <= 4
    with pytest.raises(ValueError):
        fx.extract_grid(bb, img, FoundationSpec(tile=30), device="cpu")


def test_timm_backbone_resolves_negative_layers():
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(12)])
            self.patch_embed = type("P", (), {"patch_size": (14, 14)})()
            self.pretrained_cfg = {"mean": (0.1, 0.2, 0.3), "std": (1, 1, 1)}
    bb = fx.TimmBackbone(Fake(), (-1, -4, 2))
    assert bb.indices == [11, 8, 2] and bb.patch == 14 and bb.mean == (0.1, 0.2, 0.3)
