"""Interaction model for the mscoupon labeler: pure data + resolution.

An *interaction* is one drawing gesture (squiggle / box / lasso polygon) in
image coordinates, bound to one slice and one class. The store keeps them in
creation order; that order IS the resolution order, so a later interaction
"paints over" an earlier one on any region both touch.

Nothing here touches Tk or the compute engine. Region rasters come in as the
living MSC labeling (``Msc2DPipeline.labels()``: int32 per pixel, -1 =
background, ids SPARSE -- a subset of the compact base ids), and everything is
sized ``labels.max()+1``, the established LUT pattern.

Persistence: ``LabelStore.to_json()`` <-> ``from_json()`` round-trips the raw
geometry (labels.json). The file basename is the authoritative slice identity
so a session survives folder moves; ``(si, li)`` are session-local hints,
recomputed by ``rebind()``.
"""
from __future__ import annotations

import os

# Class 0 is "no label" (transparent). Classes 1..4 get fixed, saturated colors
# chosen to stay distinguishable over both the grayscale base image and the
# golden-ratio region palette (msseg.viz.min_colors) drawn beneath at low alpha.
CLASS_COLORS = [
    (0, 0, 0, 0),          # 0: no label -> fully transparent
    (230, 25, 75, 255),    # 1: red
    (60, 180, 75, 255),    # 2: green
    (67, 99, 216, 255),    # 3: blue
    (245, 130, 49, 255),   # 4: orange
]
MAX_CLASSES = len(CLASS_COLORS)      # includes class 0

# "taps" is a set of INDEPENDENT sample points (no connecting segments): each
# point picks exactly the region under it. It is what the SHIFT-accept gesture
# records -- one taps interaction per predicted class, one point per accepted
# region -- so accepted predictions stay geometric and re-resolve after a
# recompute like every other gesture. It is not offered in the tool selector.
TOOLS = ("squiggle", "box", "polygon", "taps")


def class_color_hex(class_id):
    r, g, b, _a = CLASS_COLORS[class_id]
    return f"#{r:02x}{g:02x}{b:02x}"


class Interaction:
    """One drawing gesture. ``uid`` is the creation-order id (monotonic,
    unique within a store) and doubles as the resolution order."""

    __slots__ = ("uid", "slice_key", "si", "li", "tool", "points", "class_id")

    def __init__(self, uid, slice_key, si, li, tool, points, class_id):
        self.uid = int(uid)
        self.slice_key = str(slice_key)          # file basename
        self.si = si                             # session-local hints (or None)
        self.li = li
        self.tool = str(tool)
        self.points = [(float(x), float(y)) for x, y in points]
        self.class_id = int(class_id)

    @property
    def bound(self):
        """False when rebind() could not match slice_key against the current
        subsequences (the interaction is kept, shown greyed, never dropped)."""
        return self.si is not None and self.li is not None


class LabelStore:
    """The ordered interaction list + class count. Every mutation bumps
    ``rev`` so render-side caches (the per-slice class LUTs) can key on it."""

    def __init__(self, n_classes=3):
        self.n_classes = int(n_classes)
        self.interactions = []                   # uid order == creation order
        self.rev = 0
        self._next_uid = 1

    # -- mutations (each bumps rev) ------------------------------------ #
    def add(self, tool, points, class_id, slice_key, si=None, li=None):
        if tool not in TOOLS:
            raise ValueError(f"unknown tool {tool!r}")
        if not (1 <= int(class_id) < self.n_classes):
            raise ValueError(f"class {class_id} out of range 1..{self.n_classes - 1}")
        it = Interaction(self._next_uid, slice_key, si, li, tool, points, class_id)
        self._next_uid += 1
        self.interactions.append(it)
        self.rev += 1
        return it

    def remove(self, uid):
        n = len(self.interactions)
        self.interactions = [it for it in self.interactions if it.uid != uid]
        if len(self.interactions) != n:
            self.rev += 1

    def set_class(self, uid, class_id):
        if not (1 <= int(class_id) < self.n_classes):
            raise ValueError(f"class {class_id} out of range 1..{self.n_classes - 1}")
        for it in self.interactions:
            if it.uid == uid and it.class_id != int(class_id):
                it.class_id = int(class_id)
                self.rev += 1
                return

    def set_n_classes(self, n):
        """Change the class count (2..MAX_CLASSES). Interactions whose class no
        longer exists are clamped to the new highest class rather than dropped;
        returns their uids so the UI can say so."""
        n = int(n)
        if not (2 <= n <= MAX_CLASSES):
            raise ValueError(f"n_classes must be 2..{MAX_CLASSES}, got {n}")
        changed = []
        for it in self.interactions:
            if it.class_id >= n:
                it.class_id = n - 1
                changed.append(it.uid)
        self.n_classes = n
        self.rev += 1
        return changed

    # -- queries -------------------------------------------------------- #
    def for_slice(self, slice_key):
        """The slice's interactions in creation (= resolution) order."""
        return [it for it in self.interactions if it.slice_key == slice_key]

    def get(self, uid):
        for it in self.interactions:
            if it.uid == uid:
                return it
        return None

    # -- persistence ---------------------------------------------------- #
    # version 2: slice keys are folder-qualified ("folder/basename"); v1 files
    # carried bare basenames and are migrated on rebind() when unambiguous.
    def to_json(self):
        return {
            "version": 2,
            "app": "mscoupon-labeler",
            "n_classes": self.n_classes,
            "classes": [{"id": k, "color": class_color_hex(k)}
                        for k in range(1, self.n_classes)],
            "interactions": [
                {"uid": it.uid, "slice": it.slice_key, "si": it.si, "li": it.li,
                 "tool": it.tool, "class": it.class_id,
                 "points": [[x, y] for x, y in it.points]}
                for it in self.interactions
            ],
        }

    @classmethod
    def from_json(cls, doc):
        store = cls(n_classes=int(doc.get("n_classes", 3)))
        for d in doc.get("interactions", []):
            it = Interaction(d["uid"], d["slice"], d.get("si"), d.get("li"),
                             d.get("tool", "squiggle"), d.get("points", []),
                             d.get("class", 1))
            # A file written under a larger class count still loads: clamp.
            it.class_id = min(max(it.class_id, 1), store.n_classes - 1)
            store.interactions.append(it)
        store.interactions.sort(key=lambda it: it.uid)
        store._next_uid = 1 + max((it.uid for it in store.interactions), default=0)
        store.rev += 1
        return store

    def rebind(self, subsequences):
        """Re-derive the (si, li) hints by matching each interaction's
        folder-qualified slice key (``"folder/basename"``) against
        ``subsequences`` ([{"name", "folder", "files"}]).

        Legacy (v1) bare-basename keys match by basename ONLY when the
        basename is unambiguous across the session, and are upgraded in place
        to the qualified form -- the next save writes v2 keys. An ambiguous or
        missing key leaves the interaction unbound (kept, shown greyed) rather
        than silently binding to the first match, which is what the old
        first-wins behaviour did. Returns the number left unbound."""
        by_key = {}                      # "folder/basename" -> (si, li)
        by_base = {}                     # basename -> [(qualified_key, si, li)]
        for si, s in enumerate(subsequences):
            folder = str(s.get("folder") or "")
            for li, path in enumerate(s.get("files", [])):
                base = os.path.basename(path)
                key = f"{folder}/{base}"
                by_key.setdefault(key, (si, li))
                by_base.setdefault(base, []).append((key, si, li))
        unbound = 0
        for it in self.interactions:
            hit = by_key.get(it.slice_key)
            if hit is None and "/" not in it.slice_key:
                candidates = by_base.get(it.slice_key, [])
                if len(candidates) == 1:            # unambiguous: migrate
                    key, si, li = candidates[0]
                    it.slice_key = key
                    hit = (si, li)
            it.si, it.li = hit if hit is not None else (None, None)
            unbound += hit is None
        self.rev += 1
        return unbound


# --------------------------------------------------------------------------- #
# Resolution (pure numpy/PIL; labels = living MSC ids, -1 background)
# --------------------------------------------------------------------------- #
def line_pixels(x0, y0, x1, y1, w, h, np):
    """(ys, xs) of the raster pixels under the segment (x0,y0)-(x1,y1).

    Dense linspace sampling at max(|dx|,|dy|)+1 points touches every unit step
    -- Bresenham-equivalent coverage for region picking without a skimage
    dependency. Out-of-bounds samples are clipped away, so a stroke that runs
    off the image just contributes nothing there.
    """
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.rint(np.linspace(x0, x1, n)).astype(np.intp)
    ys = np.rint(np.linspace(y0, y1, n)).astype(np.intp)
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    return ys[ok], xs[ok]


def touched_ids(interaction, labels, np):
    """The set of region ids the gesture touches on `labels` (background -1
    is never included).

    squiggle: every pixel under the polyline's segments.
    box:      every region intersecting the rectangle spanned by the first and
              last point (a region merely overlapping the box counts).
    polygon:  every region under the filled (auto-closed) lasso, outline
              included so a degenerate sliver still picks what it was drawn on.
    taps:     each point sampled independently (no connecting segments).
    """
    h, w = labels.shape
    pts = interaction.points
    if not pts:
        return set()
    if interaction.tool == "taps":
        vals = []
        for (x, y) in pts:
            ys, xs = line_pixels(x, y, x, y, w, h, np)
            if len(ys):
                vals.append(labels[ys, xs])
        if not vals:
            return set()
        vals = np.unique(np.concatenate(vals))
    elif interaction.tool == "box":
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        xa, xb = sorted((int(round(x0)), int(round(x1))))
        ya, yb = sorted((int(round(y0)), int(round(y1))))
        xa, xb = max(xa, 0), min(xb, w - 1)
        ya, yb = max(ya, 0), min(yb, h - 1)
        if xa > xb or ya > yb:
            return set()
        vals = np.unique(labels[ya:yb + 1, xa:xb + 1])
    elif interaction.tool == "polygon":
        if len(pts) < 3:
            return set()
        from PIL import Image, ImageDraw
        im = Image.new("L", (w, h), 0)
        ImageDraw.Draw(im).polygon([(x, y) for x, y in pts], fill=1, outline=1)
        mask = np.asarray(im, dtype=bool)
        if not mask.any():
            return set()
        vals = np.unique(labels[mask])
    else:  # squiggle
        vals = []
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            ys, xs = line_pixels(x0, y0, x1, y1, w, h, np)
            if len(ys):
                vals.append(labels[ys, xs])
        if len(pts) == 1:        # a click: sample the single point
            ys, xs = line_pixels(pts[0][0], pts[0][1], pts[0][0], pts[0][1], w, h, np)
            if len(ys):
                vals.append(labels[ys, xs])
        if not vals:
            return set()
        vals = np.unique(np.concatenate(vals))
    return set(int(v) for v in vals if v >= 0)


def touched_sets(interactions, labels, np):
    """[(interaction, touched region-id set)] in creation (= resolution) order.

    The per-interaction sets are what resolution consumes, and callers that
    also need the reverse question (which interactions touch region r?) get it
    from the same single rasterization pass."""
    return [(it, touched_ids(it, labels, np))
            for it in sorted(interactions, key=lambda it: it.uid)]


def resolve_slice(interactions, labels, np):
    """region_class: uint8 array sized labels.max()+1, 0 = unlabeled.

    Interactions apply in creation (uid) order, so a later gesture paints over
    an earlier one on any region both touch."""
    return resolve_sets(touched_sets(interactions, labels, np), labels, np)


def resolve_sets(sets, labels, np):
    """resolve_slice over already-computed touched_sets output."""
    K = int(labels.max()) + 1 if labels.size else 1
    region_class = np.zeros(max(K, 1), np.uint8)
    for it, ids in sets:
        if ids:
            region_class[list(ids)] = it.class_id
    return region_class


def class_lut(region_class, np):
    """(K, 4) uint8 RGBA LUT for the canvas label overlay: region id -> its
    class color. Class-0 rows have alpha 0, so unlabeled regions are invisible
    (the canvas also treats raster values < 0 as transparent background)."""
    colors = np.asarray(CLASS_COLORS, np.uint8)
    return colors[region_class]
