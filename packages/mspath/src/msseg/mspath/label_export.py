"""Export a place for hand labeling in an external editor (GIMP).

A place exports as four files, one stem each:

``<stem>.png``
    The place at its level, windowed to 8 bits -- the image to draw over.
``<stem>_mask.png``
    A fully transparent RGBA canvas of the same size. Drag it onto the image
    as a layer, paint on it, and save it back in place: a separate layer can be
    toggled to check the drawing against the tissue, and it is the only file
    the import reads. Never overwritten once it exists -- re-exporting a place
    must not destroy the drawing on it.
``<stem>_raw.tif``
    The same pixels in the file's own dtype and every channel, for training.
``<stem>.json``
    The sidecar: which slide, level and rect the pixels came from (level and
    slide coordinates), the display window, and the mask convention.

The mask convention is RED = edge, BLUE = unsure (ignore); every other pixel
of the crop is "no edge" -- a densely labeled crop is a complete box, so its
unpainted pixels are real negatives.

Headless: the caller hands in the pixel source (anything with
``level_scale`` / ``read_region``) and the rect.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional, Sequence

FORMAT = "msseg-edge-label"
VERSION = 1
MASK_CONVENTION = {"edge": "#ff0000", "ignore": "#0000ff", "complete": True}
DISPLAY_PERCENTILES = (0.5, 99.5)


def export_stem(slide_id: str, level: int, rect: Sequence[int]) -> str:
    """``<slide name>_L<level>_x<x>_y<y>_<w>x<h>`` in SLIDE pixels: unique per
    place and level, and readable in a file browser."""
    name = os.path.splitext(os.path.basename(str(slide_id)))[0] or "slide"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    x, y, w, h = (int(v) for v in rect)
    return f"{name}_L{int(level)}_x{x}_y{y}_{w}x{h}"


def display_image(arr, np, percentiles=DISPLAY_PERCENTILES):
    """8-bit view of `arr` (``(h, w)`` or ``(h, w, C)``), windowed per channel
    at the given percentiles. Returns ``(uint8 image, windows)``: gray for
    one channel, RGB for three, the first three for more (two: R, G, and G
    again would mislead, so the second plane goes to magenta)."""
    a = np.asarray(arr, dtype=np.float32)
    planes = [a] if a.ndim == 2 else [a[..., c] for c in range(a.shape[2])]
    out, windows = [], []
    for p in planes[:3]:
        finite = p[np.isfinite(p)]
        if finite.size:
            lo, hi = (float(v) for v in np.percentile(finite, percentiles))
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        windows.append([lo, hi])
        q = np.clip((np.nan_to_num(p, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
        out.append((q * 255.0 + 0.5).astype(np.uint8))
    if len(out) == 1:
        return out[0], windows
    if len(out) == 2:                     # gray + magenta, not red + green
        g, m = out
        return np.stack([np.maximum(g, m), g, np.maximum(g, m)], -1), windows
    return np.stack(out, -1), windows


def _write_raw(path, arr, np):
    """The raw crop as a TIFF (tifffile, every channel, own dtype); falls back
    to ``.npy`` when tifffile is missing. Returns the path written."""
    try:
        import tifffile
    except ImportError:
        path = os.path.splitext(path)[0] + ".npy"
        np.save(path, np.asarray(arr))
        return path
    tifffile.imwrite(path, np.ascontiguousarray(arr))
    return path


def export_place(src, *, slide_id: str, level: int, rect: Sequence[int],
                 out_dir: str, np, slide_path: Optional[str] = None,
                 place_uid: Optional[str] = None) -> dict:
    """Write the four files for one place; returns the sidecar dict.

    `rect` is ``(x, y, w, h)`` in slide (level-0) pixels, read at `level`
    through ``src.read_region`` with the source's own downsample.
    """
    from PIL import Image

    level = int(level)
    scale = float(src.level_scale(level)) or 1.0
    x, y, w, h = (int(v) for v in rect)
    lx, ly = int(round(x / scale)), int(round(y / scale))
    lw, lh = max(1, int(round(w / scale))), max(1, int(round(h / scale)))
    arr = np.asarray(src.read_region(level, lx, ly, lw, lh))
    os.makedirs(out_dir, exist_ok=True)
    stem = export_stem(slide_id, level, (x, y, w, h))
    base = os.path.join(out_dir, stem)

    view, windows = display_image(arr, np)
    Image.fromarray(view).save(base + ".png")
    mask_path = base + "_mask.png"
    mask_kept = os.path.exists(mask_path)
    if not mask_kept:
        Image.new("RGBA", (arr.shape[1], arr.shape[0]), (0, 0, 0, 0)).save(mask_path)
    raw_path = _write_raw(base + "_raw.tif", arr, np)

    doc = {
        "format": FORMAT, "version": VERSION,
        "slide": str(slide_id),
        "path": None if slide_path is None else str(slide_path),
        "place": place_uid,
        "level": level, "scale": scale,
        "rect_slide": [x, y, w, h],
        "rect_level": [lx, ly, lw, lh],
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "channels": 1 if arr.ndim == 2 else int(arr.shape[2]),
        "dtype": str(arr.dtype),
        "display": {"percentiles": list(DISPLAY_PERCENTILES), "windows": windows},
        "mask": dict(MASK_CONVENTION),
        "files": {"image": os.path.basename(base + ".png"),
                  "mask": os.path.basename(mask_path),
                  "raw": os.path.basename(raw_path)},
    }
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    doc["mask_kept"] = mask_kept          # for the caller's message, not the file
    return doc
