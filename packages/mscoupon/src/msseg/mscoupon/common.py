"""Shared pure-Python helpers for the mscoupon Tk apps (viewer + labeler).

Everything here is UI-free (or takes widgets as arguments) and imports lazily,
so the module stays importable in headless environments.
"""
from __future__ import annotations

import os
import re


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


def _wheel_delta(event):
    """Scroll units for one wheel event, normalized across platforms.

    Windows/macOS report a signed `delta` (120 per notch); X11 sends Button-4/5
    with no delta at all. Positive result = scroll down.
    """
    num = getattr(event, "num", 0)
    if num == 4:
        return -1
    if num == 5:
        return 1
    return int(-getattr(event, "delta", 0) / 120) or 0


def _bind_click_to_value(scale):
    """Make a ttk.Scale jump to the clicked/dragged position (absolute) instead of
    stepping toward it by a page increment. Approximates the trough as the full
    widget width; the endpoints clamp to from/to."""
    def jump(event):
        w = scale.winfo_width()
        if w <= 1:
            return None
        frac = min(max(event.x / w, 0.0), 1.0)
        lo = float(scale.cget("from"))
        hi = float(scale.cget("to"))
        scale.set(lo + frac * (hi - lo))
        return "break"
    scale.bind("<Button-1>", jump)
    scale.bind("<B1-Motion>", jump)


def _id_lut(raster, min_colors, np):
    """RGBA color LUT indexed by non-negative label id (the canvas treats id<0 as
    transparent background). Colors come from the shared min_colors palette."""
    K = int(raster.max()) + 1 if raster.size else 1
    lut = np.zeros((max(K, 1), 4), np.uint8)
    ids = np.arange(max(K, 1))
    lut[:, :3] = (min_colors(ids) * 255).astype(np.uint8)
    lut[:, 3] = 255
    return lut


class FeatureTable:
    """One slice's per-feature statistics, columnar: names once + an (n, f) block.

    Mirrors ``mscoupon::FeatureTable``. Nothing here builds a dict per feature --
    that is the whole point. With a twelve-channel scale-space stack a row is
    ~60 fields, so a dict per feature meant tens of thousands of Python strings
    and dict entries on every persistence commit; as a block it is one buffer.

    The one place a dict is still convenient is the hover readout, which shows a
    single feature, so ``row_of_feature`` builds exactly one.
    """

    __slots__ = ("names", "values", "_col", "_row_of_id")

    def __init__(self, names, values):
        self.names = names
        self.values = values
        self._col = {n: i for i, n in enumerate(names)}
        self._row_of_id = None

    @property
    def n_rows(self):
        return int(self.values.shape[0]) if self.values is not None else 0

    def column(self, name):
        """One field across every feature, or None if the spec excluded it."""
        i = self._col.get(name)
        return None if i is None else self.values[:, i]

    def row_of_feature(self, feature_id):
        """One feature's fields as a name -> value dict, or None if unknown.

        The id -> row index is built once, lazily: a hover that never happens
        should not cost a pass over the table.
        """
        if self._row_of_id is None:
            ids = self.column("feature_id")
            self._row_of_id = ({} if ids is None
                               else {int(v): r for r, v in enumerate(ids)})
        r = self._row_of_id.get(int(feature_id))
        if r is None:
            return None
        return {n: float(self.values[r, i]) for n, i in self._col.items()}

    def rows(self):
        """Every feature as a dict. Only for the fallback path against an
        extension too old to know the columnar evaluator."""
        return [{n: float(self.values[r, i]) for n, i in self._col.items()}
                for r in range(self.n_rows)]


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
