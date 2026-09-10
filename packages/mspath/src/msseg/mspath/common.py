"""Shared UI-free helpers for the mspath apps.

Deliberately thin: everything mspath needs that is not slide-specific already
exists in ``msseg.labeler`` (the framework) or ``msseg.mscoupon`` (the compiled
2D pipeline and the profile/config layer), and is imported rather than copied.
"""
from __future__ import annotations

import os
import re

from msseg.labeler.table import FeatureTable  # noqa: F401  (re-exported)

# Extensions a slide can arrive as. The pyramid backends decide what they can
# actually open; this only decides what is offered in a folder listing.
SLIDE_EXTENSIONS = (".svs", ".tif", ".tiff", ".ndpi", ".scn", ".mrxs", ".vms",
                    ".vmu", ".bif", ".qptiff")


def log(msg):
    """Stage logging, alongside MSCEER's own stdout (left verbose on purpose --
    it reports the data's critical-point structure). Launch mspath-gui from a
    terminal to see all of it."""
    print(f"[mspath] {msg}", flush=True)


def natural_key(path: str):
    """Natural sort key (so slide_2 < slide_10)."""
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_slides(folder: str):
    """Naturally-sorted slide files in `folder`."""
    try:
        entries = [os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(SLIDE_EXTENSIONS)]
    except OSError:
        return []
    return sorted(entries, key=natural_key)


def human_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"
