"""Shared pure-Python helpers for the mscoupon Tk apps (viewer + labeler).

Everything here is UI-free (or takes widgets as arguments) and imports lazily,
so the module stays importable in headless environments.
"""
from __future__ import annotations

import os
import re

# Moved to the labeler framework; re-exported so `from .common import FeatureTable`
# (engine, tests) and the wheel helpers keep their old home.
from msseg.labeler.table import FeatureTable  # noqa: F401


def log(msg):
    """Command-line stage logging (alongside MSCEER's own stdout, which is left
    verbose on purpose -- it reports the data's critical-point/cancellation
    structure). Launch mscoupon-gui from a terminal to see all of it."""
    print(f"[mscoupon] {msg}", flush=True)


def natural_key(path: str):
    """Natural sort key (so asdf_2 < asdf_10)."""
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_tiffs(folder: str):
    """Naturally-sorted .tif/.tiff files in `folder`."""
    try:
        entries = [os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".tif", ".tiff"))]
    except OSError:
        return []
    return sorted(entries, key=natural_key)


def load_slice(path, alpha="drop", engine=None, log=None):
    """One TIFF as float32: (h,w) for a grayscale file, planar (C,h,w) for a
    colour one (alpha dropped by default, like the CLI).

    Prefers the extension's `read_tiff_planes` -- the CLI's own TinyTIFF reader,
    so the GUI loads exactly what a batch run will -- and falls back to Pillow
    (which also handles the compressed files TinyTIFF cannot), transposing its
    (H,W,C) into the planar layout every C++ entry point takes.
    """
    import numpy as np
    if engine is None:
        try:
            from msseg import mscoupon as engine
        except Exception:
            engine = None
    if engine is not None and hasattr(engine, "read_tiff_planes"):
        try:
            arr = engine.read_tiff_planes(str(path), alpha)
            return np.ascontiguousarray(arr[0] if arr.shape[0] == 1 else arr)
        except Exception as exc:
            if log is not None:
                log(f"read_tiff_planes failed for {os.path.basename(str(path))} "
                    f"({exc}); falling back to Pillow")
    from PIL import Image
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        if alpha == "drop" and arr.shape[2] in (2, 4):
            arr = arr[..., :-1]
        arr = np.transpose(arr, (2, 0, 1))
        if arr.shape[0] == 1:
            arr = arr[0]
    return np.ascontiguousarray(arr, dtype=np.float32)


def compact_planes(arr):
    """The smallest integer dtype that holds `arr` losslessly (uint8, uint16),
    else the array itself. Colour planes are kept per primed slice for the 3D
    assembly and the Image dropdown, and three float32 planes at 3232^2 are
    125 MB a slice; the file's own 8-bit samples are a quarter of that."""
    import numpy as np
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.dtype.kind in "ui":
        return a
    if not np.isfinite(a).all():
        return a
    lo, hi = float(a.min()), float(a.max())
    if lo < 0 or hi > 65535 or not np.array_equal(a, np.rint(a)):
        return a
    return a.astype(np.uint8 if hi <= 255 else np.uint16)


def _id_lut(raster, min_colors, np):
    """RGBA color LUT indexed by non-negative label id (the canvas treats id<0 as
    transparent background). Colors come from the shared min_colors palette."""
    K = int(raster.max()) + 1 if raster.size else 1
    lut = np.zeros((max(K, 1), 4), np.uint8)
    ids = np.arange(max(K, 1))
    lut[:, :3] = (min_colors(ids) * 255).astype(np.uint8)
    lut[:, 3] = 255
    return lut


def _parse_sigmas(text):
    """Parse a "0.7, 1.5, 3" sigma list. Silently drops anything unparseable or
    non-positive, so a half-typed entry never raises mid-keystroke."""
    out = []
    for piece in str(text).replace(";", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            value = float(piece)
        except ValueError:
            continue
        if value > 0.0:
            out.append(value)
    return out


def _format_sigmas(values):
    return ", ".join(f"{v:g}" for v in values)


def group_contiguous(indices):
    """Group a sorted list of indices into contiguous runs -> list of lists."""
    runs, cur = [], []
    for i in sorted(indices):
        if cur and i == cur[-1] + 1:
            cur.append(i)
        else:
            if cur:
                runs.append(cur)
            cur = [i]
    if cur:
        runs.append(cur)
    return runs
