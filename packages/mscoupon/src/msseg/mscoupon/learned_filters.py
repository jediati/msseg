"""Learned filters: the `unet` chain stage, a pre-trained network as an image
filter.

A stage ``{"operation": "unet", "params": {"model": NAME, "output":
"probability" | "logit", "device": "auto" | "cuda" | "cpu"}}`` maps a scalar
raster to the network's per-pixel response -- an edge probability, for the
edge filters trained from hand-drawn crops -- so it can be the topology field
the MSC runs on, or a base-chain stage. It is a PACKAGE-ONLY operation
(`config_io.PACKAGE_ONLY_OPERATIONS`): the C++ core never sees it, the engine
applies it between core batches, and the CLI cannot run a config that uses it.

Models are single-file bundles (``<name>.pt``, `save_bundle`) in
`models_dir()`: the folder picked on the card (`set_models_dir`, kept in the
session view) when there is one, else ``$MSSEG_FILTER_MODELS`` when set, else
``<app data>/filters``. A bundle carries its architecture, its input
normalisation and the level it was trained at, beside the weights, so this
module rebuilds the net without the training code. Torch is imported lazily:
listing models needs none, applying one does.
"""
from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional, Tuple

FORMAT = "msseg-learned-filter"
VERSION = 1
ENV_DIR = "MSSEG_FILTER_MODELS"
OUTPUTS = ("probability", "logit")
DEVICES = ("auto", "cuda", "cpu")
# Inference tile edge (px). Large on purpose: GroupNorm normalises over each
# input's whole extent, so a tile's statistics -- and so its output -- depend
# on what else is in the tile; big tiles keep those near the whole item's.
TILE = 2048
MARGIN = 64            # context each tile reads beyond what it keeps


_CHOSEN: Optional[str] = None      # the card's "Folder..." pick, None = default


def set_models_dir(path: Optional[str]) -> None:
    """Pick the models folder (the card's Folder... button); None or ""
    goes back to the default (`$MSSEG_FILTER_MODELS`, else app data)."""
    global _CHOSEN
    _CHOSEN = str(path) if path else None


def chosen_models_dir() -> Optional[str]:
    return _CHOSEN


def models_dir() -> str:
    if _CHOSEN:
        return _CHOSEN
    env = os.environ.get(ENV_DIR)
    if env:
        return env
    from msseg.labeler.session_doc import app_data_dir
    return os.path.join(app_data_dir("mscoupon"), "filters")


def models_report(folder: Optional[str] = None) -> str:
    """Where the model list comes from, in one line, for the card's hint and
    the log: the folder, why it is that folder, whether it exists, what is in
    it, and which copy of this module answered."""
    try:
        folder = folder or models_dir()
    except Exception as exc:                 # app_data_dir unavailable
        return f"unet models: cannot resolve the folder ({type(exc).__name__}: {exc})"
    why = ("chosen" if _CHOSEN else f"${ENV_DIR}" if os.environ.get(ENV_DIR)
           else "app data")
    if not os.path.isdir(folder):
        return f"unet models: folder {folder} ({why}) does not exist [{__file__}]"
    try:
        entries = sorted(os.listdir(folder))
    except OSError as exc:
        return f"unet models: cannot list {folder}: {exc} [{__file__}]"
    names = list_models(folder)
    other = [e for e in entries if not e.lower().endswith(".pt")]
    return (f"unet models: {len(names)} in {folder} ({why})"
            + (f": {', '.join(names)}" if names else "")
            + (f"; ignored {other}" if other else "") + f" [{__file__}]")


def list_models(folder: Optional[str] = None) -> List[str]:
    """Bundle names (file stems) in the models folder, sorted; [] when the
    folder does not exist. Cheap: a directory listing, no torch."""
    folder = folder or models_dir()
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(os.path.splitext(n)[0] for n in names if n.lower().endswith(".pt"))


# -- the architecture ------------------------------------------------------ #
def build_net(meta: dict):
    """The network a bundle's metadata describes (``arch`` "unet": a 3-level
    U-Net with GroupNorm, ``widths`` = 4 channel counts)."""
    import torch.nn as nn
    import torch
    import torch.nn.functional as F

    if meta.get("arch") != "unet":
        raise ValueError(f"unknown learned-filter architecture {meta.get('arch')!r}")
    c = tuple(int(v) for v in meta.get("widths", (16, 32, 48, 64)))
    cin = int(meta.get("in_channels", 1))
    groups = int(meta.get("groups", 4))

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(groups, o), nn.ReLU(True),
                             nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(groups, o), nn.ReLU(True))

    class UNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.d = nn.ModuleList([block(cin, c[0]), block(c[0], c[1]), block(c[1], c[2])])
            s.mid = block(c[2], c[3])
            s.up = nn.ModuleList([nn.ConvTranspose2d(c[3], c[2], 2, 2),
                                  nn.ConvTranspose2d(c[2], c[1], 2, 2),
                                  nn.ConvTranspose2d(c[1], c[0], 2, 2)])
            s.u = nn.ModuleList([block(2 * c[2], c[2]), block(2 * c[1], c[1]),
                                 block(2 * c[0], c[0])])
            s.head = nn.Conv2d(c[0], 1, 1)

        def forward(s, t):
            sk = []
            for d in s.d:
                t = d(t); sk.append(t); t = F.max_pool2d(t, 2)
            t = s.mid(t)
            for up, u, k in zip(s.up, s.u, reversed(sk)):
                t = u(torch.cat([up(t), k], 1))
            return s.head(t)

    return UNet()


def normalise(arr, meta: dict, np):
    """The input normalisation the net was trained with: ``percentile``
    (default) maps the raster's own (q50) to 0 and (q99 - q1) to 1."""
    norm = meta.get("norm") or {"kind": "percentile", "q": [1, 50, 99]}
    a = np.asarray(arr, dtype=np.float32)
    if norm.get("kind", "percentile") != "percentile":
        raise ValueError(f"unknown learned-filter normalisation {norm.get('kind')!r}")
    finite = a[np.isfinite(a)]
    if not finite.size:
        return np.zeros_like(a)
    q1, q50, q99 = np.percentile(finite, [float(v) for v in norm.get("q", [1, 50, 99])])
    return ((np.nan_to_num(a, nan=q50) - q50) / max(float(q99 - q1), 1e-6)).astype(np.float32)


def save_bundle(path: str, net, meta: dict) -> str:
    """Write a bundle: the metadata keys plus ``state_dict``."""
    import torch
    doc = dict(meta)
    doc.update(format=FORMAT, version=VERSION,
               state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(doc, path)
    return path


# -- loading + inference --------------------------------------------------- #
_CACHE: Dict[Tuple[str, float, str], tuple] = {}
_LOCK = threading.Lock()


def _device(want: str):
    import torch
    if want == "cpu" or (want == "auto" and not torch.cuda.is_available()):
        return "cpu"
    if want == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("unet: device 'cuda' asked for, but torch sees no CUDA device")
    return "cuda"


def load(name: str, device: str = "auto", folder: Optional[str] = None):
    """(net, meta, device) for a bundle, cached by path + mtime + device."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("the unet filter needs torch (pip install torch)") from exc
    if not name:
        raise RuntimeError(f"unet: no model chosen (models live in {folder or models_dir()})")
    path = os.path.join(folder or models_dir(), name + ".pt")
    if not os.path.exists(path):
        raise RuntimeError(f"unet: model {name!r} not found at {path}")
    dev = _device(device)
    key = (path, os.path.getmtime(path), dev)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            return hit
        doc = torch.load(path, map_location="cpu", weights_only=False)
        if doc.get("format") != FORMAT:
            raise RuntimeError(f"unet: {path} is not a learned-filter bundle")
        meta = {k: v for k, v in doc.items() if k != "state_dict"}
        net = build_net(meta)
        net.load_state_dict(doc["state_dict"])
        net.eval().to(dev)
        _CACHE.clear()                     # one model resident at a time
        _CACHE[key] = (net, meta, dev)
        return _CACHE[key]


def apply(arr, params: dict, np, log=lambda _m: None):
    """Run the stage on a scalar raster; returns float32 of the same shape.
    Tiled (`TILE` + `MARGIN` of context per side), so an item of any size
    fits in memory."""
    import torch
    params = params or {}
    output = str(params.get("output") or "probability")
    if output not in OUTPUTS:
        raise RuntimeError(f"unet: output must be one of {OUTPUTS} (got {output!r})")
    name = str(params.get("model") or "")
    if not name:
        # What the card shows for an unset model: its default, the first
        # bundle. A stage saved while the folder was empty (or before the card
        # was built) would otherwise fail every prime with nothing to pick.
        names = list_models()
        if names:
            name = names[0]
            log(f"  unet: no model named in the stage; using {name!r} (the card's default)")
    net, meta, dev = load(name, str(params.get("device") or "auto"))
    a = np.asarray(arr)
    if a.ndim != 2:
        raise RuntimeError(f"unet: expects one plane, got shape {a.shape}")
    x = normalise(a, meta, np)
    h, w = x.shape
    out = np.empty((h, w), np.float32)
    step = 8                                # the net's total downsampling
    with torch.no_grad():
        for y0 in range(0, h, TILE):
            for x0 in range(0, w, TILE):
                y1, x1 = min(h, y0 + TILE), min(w, x0 + TILE)
                ya, xa = max(0, y0 - MARGIN), max(0, x0 - MARGIN)
                yb, xb = min(h, y1 + MARGIN), min(w, x1 + MARGIN)
                t = x[ya:yb, xa:xb]
                th, tw = t.shape
                ph, pw = (-th) % step, (-tw) % step
                mode = "reflect" if min(th, tw) > max(ph, pw) else "edge"
                t = np.pad(t, ((0, ph), (0, pw)), mode=mode)
                r = net(torch.from_numpy(np.ascontiguousarray(t))[None, None].to(dev))[0, 0]
                if output == "probability":
                    r = torch.sigmoid(r)
                r = r.float().cpu().numpy()
                out[y0:y1, x0:x1] = r[y0 - ya:y1 - ya, x0 - xa:x1 - xa]
    return out


def describe(name: str, folder: Optional[str] = None) -> str:
    """One line for a log or tooltip: what the bundle says about itself."""
    try:
        _net, meta, dev = load(name, "cpu", folder)
    except Exception as exc:
        return f"{name}: {exc}"
    lvl = meta.get("trained_level")
    src = meta.get("trained_on")
    return (f"{name}: {meta.get('arch')} {tuple(meta.get('widths', ()))}"
            + (f", trained at L{lvl}" if lvl is not None else "")
            + (f" on {len(src)} crop(s)" if isinstance(src, list) else ""))
