"""Train a learned edge filter (the `unet` chain stage) from exported crops.

    python train_edge_filter.py CROPS [CROPS ...] --name sim_edges_v1 [--steps 4000]

Each CROPS is a folder (every labeled crop in it) or one crop, named by any of
its files (``<stem>.json``, ``<stem>_mask.png``, ``<stem>.png``). A crop is what the mspath tree's "Export for labeling..." writes:
``<stem>.json`` sidecars, ``<stem>_raw.tif`` pixels and ``<stem>_mask.png``
layers drawn in GIMP. A mask pixel is an EDGE where its alpha is > 0.5 and it
is not blue; BLUE (b > r and b > g) is "unsure" and ignored; everything else
is "no edge". The flanks of every stroke (3 px) are ignored too -- a stroke
says where an edge is to within its width, not to the pixel.

The net is small (a 3-level U-Net, `--widths`) and trained with heavy
augmentation (any rotation, 0.7-1.4x scale, shear, defocus over the patch or
one side of a random seam with a gain step across it, a fade toward black,
gamma/gain/offset, vertical/horizontal banding, signal-dependent noise): the
annotation is rough, so the point is an image FILTER that responds to edges
the way the strokes do, judged by eye on unseen regions -- not a held-out
score. The bundle lands in the folder the `unet` card lists
(`learned_filters.models_dir()`, or --out).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np


def sidecars(paths):
    """The ``<stem>.json`` sidecars a list of folders / crop files names, in
    order, each once."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(glob.glob(os.path.join(p, "*.json")))
        else:
            stem = os.path.splitext(p)[0]
            for suffix in ("_mask", "_raw"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
            if not os.path.exists(stem + ".json"):
                raise SystemExit(f"{p}: no sidecar {stem}.json next to it")
            found = [stem + ".json"]
        out += [f for f in found if f not in out]
    return out


def load_crops(paths):
    import tifffile
    from PIL import Image
    from scipy import ndimage as ndi
    from msseg.mscoupon import learned_filters as lf
    crops = []
    for side in sidecars(paths):
        with open(side, encoding="utf-8") as fh:
            doc = json.load(fh)
        if doc.get("format") != "msseg-edge-label":
            continue
        base = os.path.dirname(side)
        mask_path = os.path.join(base, doc["files"]["mask"])
        raw_path = os.path.join(base, doc["files"]["raw"])
        if not (os.path.exists(mask_path) and os.path.exists(raw_path)):
            continue
        m = np.asarray(Image.open(mask_path).convert("RGBA")).astype(np.int16)
        painted = m[..., 3] > 127
        blue = painted & (m[..., 2] > m[..., 0] + 40) & (m[..., 2] > m[..., 1] + 40)
        pos = painted & ~blue
        if pos.sum() == 0:
            print(f"skip {os.path.basename(side)}: no edge strokes")
            continue
        raw = (tifffile.imread(raw_path) if raw_path.endswith(".tif")
               else np.load(raw_path)).astype(np.float32)
        img = raw if raw.ndim == 2 else raw[..., 0] if np.ptp(raw.std((0, 1))) == 0 else raw.mean(-1)
        if img.shape != pos.shape:
            raise SystemExit(f"{side}: mask {pos.shape} does not match raw {img.shape}")
        ignore = (ndi.binary_dilation(pos, iterations=3) & ~pos) | blue
        crops.append(dict(name=os.path.basename(side)[:-5], level=doc.get("level"),
                          x=lf.normalise(img, {}, np), pos=pos, ignore=ignore))
        print(f"crop {crops[-1]['name']}: {img.shape}, edge {pos.mean():.3f}, ignore {ignore.mean():.3f}")
    return crops


def make_batcher(crops, P, B, dev):
    import torch
    import torch.nn.functional as F
    T = [(torch.tensor(c["x"], device=dev)[None, None],
          torch.tensor(c["pos"], dtype=torch.float32, device=dev)[None, None],
          torch.tensor(~c["ignore"], dtype=torch.float32, device=dev)[None, None]) for c in crops]
    weights = np.array([c["x"].size for c in crops], float); weights /= weights.sum()

    def blur(t, sig):
        out = torch.empty_like(t)
        for i in range(t.shape[0]):
            s = float(sig[i])
            if s < 0.3:
                out[i] = t[i]; continue
            r = int(math.ceil(3 * s)); k = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
            k = torch.exp(-k ** 2 / (2 * s * s)); k = k / k.sum()
            v = F.conv2d(F.pad(t[i:i + 1], (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
            out[i] = F.conv2d(F.pad(v, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))[0]
        return out

    def sample(X, Y, M, n):
        H, W = X.shape[-2:]
        th = torch.rand(n, device=dev) * 2 * math.pi
        sc = torch.exp(torch.empty(n, device=dev).uniform_(math.log(0.7), math.log(1.4)))
        sh = torch.empty(n, device=dev).uniform_(-0.15, 0.15)
        cx = torch.empty(n, device=dev).uniform_(-0.8, 0.8); cy = torch.empty(n, device=dev).uniform_(-0.8, 0.8)
        c, s = torch.cos(th), torch.sin(th)
        ax, ay = P / W * sc, P / H * sc
        A = torch.zeros(n, 2, 3, device=dev)
        A[:, 0, 0] = c * ax; A[:, 0, 1] = (-s + sh) * ax; A[:, 0, 2] = cx
        A[:, 1, 0] = s * ay; A[:, 1, 1] = c * ay; A[:, 1, 2] = cy
        g = F.affine_grid(A, (n, 1, P, P), align_corners=False)
        kw = dict(padding_mode="reflection", align_corners=False)
        return (F.grid_sample(X.expand(n, -1, -1, -1), g, mode="bilinear", **kw),
                F.grid_sample(Y.expand(n, -1, -1, -1), g, mode="bilinear", **kw) > 0.5,
                F.grid_sample(M.expand(n, -1, -1, -1), g, mode="nearest", **kw))

    yy, xx = torch.meshgrid(torch.linspace(-1, 1, P, device=dev), torch.linspace(-1, 1, P, device=dev),
                            indexing="ij")

    def batch():
        which = np.random.choice(len(T), B, p=weights)
        parts = [sample(*T[k], int((which == k).sum())) for k in range(len(T)) if (which == k).any()]
        xb = torch.cat([p[0] for p in parts]); yb = torch.cat([p[1] for p in parts])
        mb = torch.cat([p[2] for p in parts])
        # defocus over the patch or one side of a random seam, gain step across it
        blurred = blur(xb, torch.rand(B, device=dev) ** 2 * 3.0)
        ang = torch.rand(B, device=dev) * math.pi; off = torch.empty(B, device=dev).uniform_(-0.7, 0.7)
        side = (xx[None] * torch.cos(ang).view(-1, 1, 1) + yy[None] * torch.sin(ang).view(-1, 1, 1)) \
            > off.view(-1, 1, 1)
        side = (side | (torch.rand(B, device=dev) < 0.5).view(-1, 1, 1))[:, None]
        xb = torch.where(side, blurred, xb)
        seam = (torch.rand(B, device=dev) < 0.4).view(-1, 1, 1, 1)
        xb = torch.where(seam & side, xb * torch.exp(torch.randn(B, 1, 1, 1, device=dev) * 0.3)
                         + torch.randn(B, 1, 1, 1, device=dev) * 0.15, xb)
        # fade toward black, gamma, gain/offset
        fld = F.interpolate(torch.randn(B, 1, 4, 4, device=dev), size=(P, P), mode="bicubic",
                            align_corners=False)
        xb = xb * torch.exp(fld * (torch.rand(B, device=dev) * 0.8).view(-1, 1, 1, 1))
        lo = xb.amin((2, 3), keepdim=True)
        xb = (xb - lo).clamp_min(0) ** torch.exp(torch.randn(B, 1, 1, 1, device=dev) * 0.3) + lo
        xb = xb * torch.exp(torch.randn(B, 1, 1, 1, device=dev) * 0.25) \
            + torch.randn(B, 1, 1, 1, device=dev) * 0.15
        # banding
        amp = (torch.rand(B, device=dev) * 0.12).view(-1, 1, 1, 1)
        per = torch.empty(B, 1, 1, 1, device=dev).uniform_(3, 40)
        sinus = torch.sin(2 * math.pi * torch.arange(P, device=dev).view(1, 1, 1, -1) / per
                          + torch.rand(B, 1, 1, 1, device=dev) * 6.28)
        stripe = (torch.randn(B, 1, 1, P, device=dev) * 0.5 + sinus) * amp
        stripe = torch.where((torch.rand(B, device=dev) < 0.5).view(-1, 1, 1, 1), stripe,
                             stripe.transpose(2, 3))
        xb = xb + stripe * (torch.rand(B, device=dev) < 0.6).view(-1, 1, 1, 1)
        # noise
        xb = xb + torch.randn_like(xb) * (torch.rand(B, 1, 1, 1, device=dev) * 0.08) \
            * (1 + xb.abs()).sqrt()
        return xb, yb.float(), mb

    return batch


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("crops", nargs="+", help="folders and/or crop files")
    ap.add_argument("--name", required=True, help="bundle name (the dropdown entry)")
    ap.add_argument("--out", default=None, help="models folder (default: the app's)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--widths", default="16,32,48,64")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F
    from msseg.mscoupon import learned_filters as lf

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    crops = load_crops(args.crops)
    if not crops:
        raise SystemExit(f"no labeled crops in {' '.join(args.crops)}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = {"arch": "unet", "widths": [int(v) for v in args.widths.split(",")], "in_channels": 1,
            "groups": 4, "norm": {"kind": "percentile", "q": [1, 50, 99]},
            "trained_level": crops[0]["level"], "trained_on": [c["name"] for c in crops],
            "steps": args.steps, "trained": time.strftime("%Y-%m-%d %H:%M")}
    net = lf.build_net(meta).to(dev)
    print(f"{sum(p.numel() for p in net.parameters())} params on {dev}")
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=args.steps)
    n_pos = sum(c["pos"].sum() for c in crops)
    n_neg = sum((~c["pos"] & ~c["ignore"]).sum() for c in crops)
    wpos = torch.tensor(float(np.clip(n_neg / max(n_pos, 1), 1, 8)), device=dev)
    batch = make_batcher(crops, args.patch, args.batch, dev)
    t0 = time.time()
    for it in range(args.steps):
        xb, yb, mb = batch()
        loss = (F.binary_cross_entropy_with_logits(net(xb), yb, reduction="none", pos_weight=wpos)
                * mb).sum() / mb.sum()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % 500 == 0 or it == args.steps - 1:
            print(f"step {it} loss {loss.item():.4f}  {time.time() - t0:.0f}s", flush=True)
    path = os.path.join(args.out or lf.models_dir(), args.name + ".pt")
    lf.save_bundle(path, net.eval(), meta)
    print("wrote", path)


if __name__ == "__main__":
    main()
