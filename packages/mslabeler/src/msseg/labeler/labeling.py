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
geometry (annotations.json -- labels.json before the rename; the document
itself is unchanged, so old files still load). Note the two senses of "label"
that rename separated: the ANNOTATIONS here are gestures, while ``labels`` in
the compute path is the MSC label raster this module rasterizes them onto.
A gesture is keyed by the SLIDE it was drawn on (coupon: the folder-qualified
slice file, which is its own slide; mspath: ``"folder/slide.svs"``), never by
the item -- a place at a resolution -- it happened to be drawn on: the
geometry is a statement about tissue at a location, so every item covering
that location, at any level, queries it (``for_item``). ``(si, li)`` are
session-local hints, recomputed by ``rebind()``.

``meta`` never selects WHAT a gesture paints on the level it was drawn at;
the geometry alone does. Off that level, ``meta["level"]`` / ``["scale"]`` /
``["px"]`` (the scale of intent: the drawing item's level, its slide pixels
per raster pixel, and the slide pixels per screen pixel at draw time) and
``meta["outline"]`` (an extent's closed loops) choose HOW the same geometry
is applied -- a stroke keeps the width the user saw, an extent resolves by
its outline rather than by seeds that would land in different regions.
"""
from __future__ import annotations

import math
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
# Seam gestures (docs/seam_labeling.md) label the polylines BETWEEN regions,
# not regions, and live in ``LabelStore.seams``, a separate list: a "scope"
# is a box inside which every seam is labelled (interior unless a trace says
# otherwise) and a "trace" is a livewire path along seams (class boundary).
# Kept apart from ``interactions`` so an older reader, which rasterizes any
# unknown tool as a squiggle, never paints a trace onto regions.
SEAM_TOOLS = ("scope", "trace")


def class_color_hex(class_id):
    r, g, b, _a = CLASS_COLORS[class_id]
    return f"#{r:02x}{g:02x}{b:02x}"


class Interaction:
    """One drawing gesture. ``uid`` is the creation-order id (monotonic,
    unique within a store) and doubles as the resolution order."""

    __slots__ = ("uid", "slice_key", "si", "li", "tool", "points", "class_id",
                 "meta")

    def __init__(self, uid, slice_key, si, li, tool, points, class_id, meta=None):
        self.uid = int(uid)
        self.slice_key = str(slice_key)          # the slide (coupon: the slice file)
        self.si = si                             # session-local hints (or None)
        self.li = li
        self.tool = str(tool)
        self.points = [(float(x), float(y)) for x, y in points]
        self.class_id = int(class_id)
        # Optional provenance and scale of intent, JSON-safe values only (a
        # magic fill records the tool that produced its taps, the seed,
        # threshold and metric; a placed labeler records level / scale / px;
        # an extent records its outline). On the drawing level resolution
        # never reads it -- the geometry alone decides what a gesture paints,
        # so a re-decomposition cannot be steered by stale metadata. Off that
        # level it only chooses how the geometry is applied (module docstring).
        self.meta = dict(meta) if meta else None

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
        self.seams = []                          # seam gestures (SEAM_TOOLS), same uid space
        self.colors = {}                         # class_id -> "#rrggbb" override
        self.names = {}                          # class_id -> display name (a task's vocabulary)
        self.rev = 0
        self._next_uid = 1

    # -- class colors (user-pickable; defaults from CLASS_COLORS) -------- #
    def color(self, class_id):
        return self.colors.get(class_id, class_color_hex(class_id))

    def rgba(self, class_id):
        hexv = self.colors.get(class_id)
        if hexv is None:
            return (CLASS_COLORS[class_id]
                    if 0 <= class_id < MAX_CLASSES else (0, 0, 0, 0))
        h = str(hexv).lstrip("#")
        try:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
        except (ValueError, IndexError):
            return CLASS_COLORS[class_id % MAX_CLASSES]

    def set_color(self, class_id, hexv):
        if not (1 <= int(class_id) < MAX_CLASSES):
            return
        hexv = str(hexv)
        if self.colors.get(class_id) == hexv:
            return
        self.colors[int(class_id)] = hexv
        self.rev += 1                    # render LUT caches key on rev

    # -- class names (a task's vocabulary; display only, ids stay the wire
    #    format for every raster, LUT and probability column) ------------ #
    def name(self, class_id):
        """The class's display name, ``"class k"`` when none was given."""
        return self.names.get(int(class_id)) or f"class {int(class_id)}"

    def has_name(self, class_id):
        return bool(self.names.get(int(class_id)))

    def set_name(self, class_id, name):
        """Name a class; an empty name clears it. A real change bumps rev
        so undo snapshots and the session autosave see it."""
        k = int(class_id)
        if not (1 <= k < MAX_CLASSES):
            return
        name = str(name or "").strip()
        if (self.names.get(k) or "") == name:
            return
        if name:
            self.names[k] = name
        else:
            self.names.pop(k, None)
        self.rev += 1

    # -- mutations (each bumps rev) ------------------------------------ #
    def add(self, tool, points, class_id, slice_key, si=None, li=None, meta=None):
        if tool not in TOOLS:
            raise ValueError(f"unknown tool {tool!r}")
        if not (1 <= int(class_id) < self.n_classes):
            raise ValueError(f"class {class_id} out of range 1..{self.n_classes - 1}")
        it = Interaction(self._next_uid, slice_key, si, li, tool, points, class_id,
                         meta=meta)
        self._next_uid += 1
        self.interactions.append(it)
        self.rev += 1
        return it

    def add_seam(self, tool, points, class_id, slice_key, si=None, li=None, meta=None):
        """A seam gesture (``SEAM_TOOLS``) -- a polyline task's. Its class is
        one of the store's own (in a polyline task the vocabulary IS the seam
        vocabulary; class 1 is the "not a boundary" role); uids are shared
        with the region gestures."""
        if tool not in SEAM_TOOLS:
            raise ValueError(f"unknown seam tool {tool!r}")
        if not (1 <= int(class_id) < self.n_classes):
            raise ValueError(f"seam class {class_id} out of range 1..{self.n_classes - 1}")
        it = Interaction(self._next_uid, slice_key, si, li, tool, points, class_id,
                         meta=meta)
        self._next_uid += 1
        self.seams.append(it)
        self.rev += 1
        return it

    def remove(self, uid):
        n = len(self.interactions) + len(self.seams)
        self.interactions = [it for it in self.interactions if it.uid != uid]
        self.seams = [it for it in self.seams if it.uid != uid]
        if len(self.interactions) + len(self.seams) != n:
            self.rev += 1

    def remove_many(self, uids):
        """Drop several interactions (region or seam gestures) as ONE mutation
        (one rev bump, so the render caches rebuild once, not per gesture).
        Returns how many went."""
        uids = {int(u) for u in uids}
        n = len(self.interactions) + len(self.seams)
        self.interactions = [it for it in self.interactions if it.uid not in uids]
        self.seams = [it for it in self.seams if it.uid not in uids]
        removed = n - len(self.interactions) - len(self.seams)
        if removed:
            self.rev += 1
        return removed

    def set_class(self, uid, class_id):
        if not (1 <= int(class_id) < self.n_classes):
            raise ValueError(f"class {class_id} out of range 1..{self.n_classes - 1}")
        for it in self.interactions + self.seams:
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
        for it in self.interactions + self.seams:
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

    def for_class(self, class_id):
        """Every gesture of one class (region or seam), across all slices."""
        return [it for it in self.interactions + self.seams if it.class_id == int(class_id)]

    def for_slice_seams(self, slice_key):
        """The slice's seam gestures in creation order."""
        return [it for it in self.seams if it.slice_key == slice_key]

    def for_item(self, slice_key, rect=None):
        """The gestures an ITEM sees: those keyed by its slide whose extent
        (points, outline, stroke width) meets `rect` -- ``(x, y, w, h)`` in
        image pixels, the item's place on the slide -- or every one of the
        slide's when `rect` is None (a whole slice / the overview). Creation
        order, like ``for_slice``."""
        return [it for it in self.interactions
                if it.slice_key == slice_key and gesture_meets(it, rect)]

    def for_item_seams(self, slice_key, rect=None):
        """``for_item`` over the seam gestures."""
        return [it for it in self.seams
                if it.slice_key == slice_key and gesture_meets(it, rect)]

    def get(self, uid):
        for it in self.interactions:
            if it.uid == uid:
                return it
        for it in self.seams:
            if it.uid == uid:
                return it
        return None

    # -- persistence ---------------------------------------------------- #
    # version 2: slice keys are folder-qualified ("folder/basename"); v1 files
    # carried bare basenames and are migrated on rebind() when unambiguous.
    # version 3: a "seams" list of seam gestures -- written, and the version
    # raised, ONLY when there are any, so a store without seams serializes
    # byte-identically to v2 and an older reader (which ignores the key)
    # still loads everything it understands.
    @staticmethod
    def _row(it):
        return {"uid": it.uid, "slice": it.slice_key, "si": it.si, "li": it.li,
                "tool": it.tool, "class": it.class_id,
                "points": [[x, y] for x, y in it.points],
                # "meta" only when present: a file without it is byte-identical
                # to what older versions wrote, and older readers ignore it.
                **({"meta": it.meta} if it.meta else {})}

    def to_json(self):
        doc = {
            "version": 3 if self.seams else 2,
            "app": "mscoupon-labeler",
            "n_classes": self.n_classes,
            # "name" only when one was given: a store without names is
            # byte-identical to what older versions wrote, and older readers
            # ignore the key.
            "classes": [{"id": k, "color": self.color(k),
                         **({"name": self.names[k]} if self.names.get(k) else {})}
                        for k in range(1, self.n_classes)],
            "interactions": [self._row(it) for it in self.interactions],
        }
        if self.seams:
            doc["seams"] = [self._row(it) for it in self.seams]
        return doc

    @classmethod
    def from_json(cls, doc):
        store = cls(n_classes=int(doc.get("n_classes", 3)))
        for c in doc.get("classes") or []:
            if isinstance(c, dict) and c.get("id") and c.get("color"):
                store.colors[int(c["id"])] = str(c["color"])
            if isinstance(c, dict) and c.get("id") and isinstance(c.get("name"), str) \
                    and c["name"].strip():
                store.names[int(c["id"])] = c["name"].strip()
        for d in doc.get("interactions", []):
            meta = d.get("meta")
            it = Interaction(d["uid"], d["slice"], d.get("si"), d.get("li"),
                             d.get("tool", "squiggle"), d.get("points", []),
                             d.get("class", 1),
                             meta=meta if isinstance(meta, dict) else None)
            # A file written under a larger class count still loads: clamp.
            it.class_id = min(max(it.class_id, 1), store.n_classes - 1)
            store.interactions.append(it)
        store.interactions.sort(key=lambda it: it.uid)
        for d in doc.get("seams") or []:
            if not isinstance(d, dict) or d.get("tool") not in SEAM_TOOLS:
                continue
            meta = d.get("meta")
            it = Interaction(d["uid"], d["slice"], d.get("si"), d.get("li"),
                             d["tool"], d.get("points", []), d.get("class", 2),
                             meta=meta if isinstance(meta, dict) else None)
            it.class_id = max(it.class_id, 1)
            store.seams.append(it)
        store.seams.sort(key=lambda it: it.uid)
        store._next_uid = 1 + max((it.uid for it in store.interactions + store.seams),
                                  default=0)
        store.rev += 1
        return store

    def rebind(self, subsequences, resolve=None, rebase=None):
        """Re-derive the (si, li) hints by matching each interaction's
        folder-qualified slice key (``"folder/basename"``) against
        ``subsequences`` ([{"name", "folder", "files"}]).

        Legacy (v1) bare-basename keys match by basename ONLY when the
        basename is unambiguous across the session, and are upgraded in place
        to the qualified form -- the next save writes v2 keys. An ambiguous or
        missing key leaves the interaction unbound (kept, shown greyed) rather
        than silently binding to the first match, which is what the old
        first-wins behaviour did. Returns the number left unbound.

        ``rebase(key) -> (slide_key, level, scale) | None`` upgrades a key
        written when gestures were bound to ITEMS (a slide's
        ``"folder/name@level#rect"``) to the slide's own key, in place, the
        way v1 basenames are -- and records the item's level and scale as
        the gesture's scale of intent (``meta``) unless it already has one.
        ``resolve(key) -> (si, li) | None`` is then asked about a key the
        file matching cannot place -- an app whose keys are not
        ``"folder/basename"`` binds through its catalogue this way; without
        it every such key stays unbound for good. Both are idempotent, so a
        second pass (an undo snapshot, a task switch) changes nothing."""
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
        for it in self.interactions + self.seams:
            hit = by_key.get(it.slice_key)
            if hit is None and "/" not in it.slice_key:
                candidates = by_base.get(it.slice_key, [])
                if len(candidates) == 1:            # unambiguous: migrate
                    key, si, li = candidates[0]
                    it.slice_key = key
                    hit = (si, li)
            if hit is None and rebase is not None:
                try:
                    rb = rebase(it.slice_key)
                except Exception:
                    rb = None
                if rb is not None:                  # an item key: move to its slide
                    new_key, level, scale = rb
                    it.slice_key = str(new_key)
                    meta = dict(it.meta or {})
                    if level is not None:
                        meta.setdefault("level", int(level))
                    if scale is not None:
                        meta.setdefault("scale", float(scale))
                    it.meta = meta or None
                    hit = by_key.get(it.slice_key)
            if hit is None and resolve is not None:
                try:
                    found = resolve(it.slice_key)
                except Exception:
                    found = None
                if found is not None:
                    hit = (int(found[0]), int(found[1]))
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


def polygon_mask(pts, w, h, np):
    """Rasterize the auto-closed polygon `pts` (image coords) over its own
    bounding box, clipped to the w x h raster: ``(mask, ya, xa)`` with
    ``mask`` a bool (bh, bw) array whose [0, 0] is raster pixel (ya, xa), or
    None when the box falls entirely outside the image.

    Bounding-box rasterization is what makes a lasso cheap to re-resolve (and
    to preview while it is still being drawn): a small lasso on a 3232^2 slice
    used to allocate and scan the full 10 Mpx mask."""
    from PIL import Image, ImageDraw
    xs = [x for x, _y in pts]
    ys = [y for _x, y in pts]
    xa, xb = max(int(min(xs)) - 1, 0), min(int(max(xs)) + 2, w)
    ya, yb = max(int(min(ys)) - 1, 0), min(int(max(ys)) + 2, h)
    if xa >= xb or ya >= yb:
        return None
    im = Image.new("L", (xb - xa, yb - ya), 0)
    ImageDraw.Draw(im).polygon([(x - xa, y - ya) for x, y in pts], fill=1, outline=1)
    return np.asarray(im, dtype=bool), ya, xa


def stroke_mask(pts, width, w, h, np):
    """The polyline drawn `width` raster pixels wide with round joints and
    ends (a click is a disk), as ``(mask, ya, xa)`` over its own bbox crop of
    an ``h x w`` raster -- or None when the crop is empty. PIL draws it, as
    it does the lasso, so the cost is the area and not the width."""
    from PIL import Image, ImageDraw
    if not pts:
        return None
    r = float(width) / 2.0
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    xa = max(0, int(math.floor(min(xs) - r)) - 1)
    ya = max(0, int(math.floor(min(ys) - r)) - 1)
    xb = min(int(w), int(math.ceil(max(xs) + r)) + 2)
    yb = min(int(h), int(math.ceil(max(ys) + r)) + 2)
    if xb <= xa or yb <= ya:
        return None
    img = Image.new("L", (xb - xa, yb - ya), 0)
    d = ImageDraw.Draw(img)
    local = [(x - xa, y - ya) for x, y in pts]
    if len(local) > 1:
        d.line(local, fill=1, width=max(1, int(round(width))), joint="curve")
    for x, y in local:
        d.ellipse([x - r, y - r, x + r, y + r], fill=1)
    return np.asarray(img, dtype=bool), ya, xa


def touched_ids(interaction, labels, np, width=None, off_level=False):
    """The set of region ids the gesture touches on `labels` (background -1
    is never included).

    squiggle: every pixel under the polyline's segments -- a hairline, or a
              stroke `width` raster pixels wide when that exceeds 1.5 (the
              width the user saw: ``meta["px"]`` slide px per screen px,
              divided by the raster's scale; ``touched_ids_over`` passes it).
    box:      every region intersecting the rectangle spanned by the first and
              last point (a region merely overlapping the box counts).
    polygon:  every region under the filled (auto-closed) lasso, outline
              included so a degenerate sliver still picks what it was drawn on.
    taps:     each point sampled independently (no connecting segments); an
              extent (``meta["outline"]``) applied `off_level` -- on a raster
              other than the one it was drawn on -- resolves by its outline
              instead, since seeds land in different regions at another level.
    """
    h, w = labels.shape
    pts = interaction.points
    if not pts:
        return set()
    meta = interaction.meta or {}
    if interaction.tool == "taps" and off_level and meta.get("outline"):
        from .extents import extent_mask
        em = extent_mask(meta["outline"], w, h, np)
        if em is None:
            return set()
        mask, ya, xa = em
        if not mask.any():
            return set()
        sub = labels[ya:ya + mask.shape[0], xa:xa + mask.shape[1]]
        vals = np.unique(sub[mask])
    elif interaction.tool == "taps":
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
        pm = polygon_mask(pts, w, h, np)
        if pm is None:
            return set()
        mask, ya, xa = pm
        if not mask.any():
            return set()
        sub = labels[ya:ya + mask.shape[0], xa:xa + mask.shape[1]]
        vals = np.unique(sub[mask])
    elif width is not None and float(width) > 1.5:   # a squiggle with its width
        sm = stroke_mask(pts, float(width), w, h, np)
        if sm is None:
            return set()
        mask, ya, xa = sm
        if not mask.any():
            return set()
        sub = labels[ya:ya + mask.shape[0], xa:xa + mask.shape[1]]
        vals = np.unique(sub[mask])
    else:  # squiggle, a hairline
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


class Placement:
    """Where an item's region raster sits in the image the canvas draws.

    Gestures arrive in image coordinates. For an item that IS the image -- a
    coupon slice -- those are already raster indices and this is the identity.
    For an item that covers part of a larger image, or covers it at a coarser
    resolution (a whole-slide ROI at level 0, an overview at 1/16), a raster
    index is ``(image - origin) / scale``, and indexing the raster with an image
    coordinate is out of bounds rather than merely wrong -- which is the good
    news, because it fails loudly.

    Only the drawing tools need this: they rasterize a gesture against the ids
    to preview and to seed. What a gesture STORES stays in image coordinates
    (that is the geometry annotations.json keeps, and it must survive a change
    of level), and resolution goes through ``touched_ids_over``, which asks the
    layer instead.
    """

    __slots__ = ("ox", "oy", "scale")

    def __init__(self, origin=(0, 0), scale=1.0):
        self.ox, self.oy = float(origin[0]), float(origin[1])
        self.scale = float(scale) or 1.0

    @property
    def identity(self):
        return (self.ox, self.oy, self.scale) == (0.0, 0.0, 1.0)

    def to_raster(self, x, y):
        """Image point -> raster coordinates (floats; the caller rounds)."""
        return (float(x) - self.ox) / self.scale, (float(y) - self.oy) / self.scale

    def to_image(self, x, y):
        """Raster coordinates -> image point."""
        return float(x) * self.scale + self.ox, float(y) * self.scale + self.oy

    def points_to_raster(self, pts):
        return [self.to_raster(x, y) for x, y in pts]

    def __repr__(self):
        return f"Placement(origin=({self.ox:g}, {self.oy:g}), scale={self.scale:g})"


IDENTITY = Placement()


def gesture_extent(interaction, pad=1):
    """Inclusive ``(x0, y0, x1, y1)`` in image pixels covering everything the
    gesture can touch -- its points, every ``meta["outline"]`` loop, and half
    its recorded stroke width (``meta["px"]``) -- or None when it has no
    points. `pad` widens it by a pixel on every side so a click, a horizontal
    squiggle or a zero-area box still names a rect with something in it; a
    gesture without a recorded width or outline gets exactly that pad, so
    the numbers older callers saw are unchanged."""
    pts = list(interaction.points)
    meta = interaction.meta or {}
    for loop in meta.get("outline") or ():
        pts.extend((float(x), float(y)) for x, y in loop)
    if not pts:
        return None
    px = meta.get("px")
    if isinstance(px, (int, float)) and not isinstance(px, bool) and px > 0:
        pad = max(pad, 1 + int(math.ceil(px / 2.0)))
    xs = [int(math.floor(x)) for x, _ in pts]
    ys = [int(math.floor(y)) for _, y in pts]
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def gesture_meets(interaction, rect):
    """Whether the gesture's extent meets `rect` ``(x, y, w, h)``; a None
    rect (a whole slide) meets everything that has geometry."""
    ext = gesture_extent(interaction)
    if ext is None:
        return False
    if rect is None:
        return True
    x, y, w, h = rect
    x0, y0, x1, y1 = ext
    return x0 < x + w and x1 >= x and y0 < y + h and y1 >= y


def gesture_bbox(interaction, np, pad=1):
    """(x, y, w, h) covering a gesture's extent, or None when it has none
    (see ``gesture_extent``; `np` is kept for the callers' sake)."""
    ext = gesture_extent(interaction, pad)
    if ext is None:
        return None
    x0, y0, x1, y1 = ext
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def transformed(interaction, fn):
    """The same gesture with every point -- its own and its outline's --
    mapped through ``fn(x, y) -> (x, y)``: how a gesture in image coordinates
    becomes one in a raster crop's."""
    meta = interaction.meta
    if meta and meta.get("outline"):
        meta = dict(meta)
        meta["outline"] = [[list(fn(float(x), float(y))) for x, y in loop]
                           for loop in meta["outline"]]
    return Interaction(interaction.uid, interaction.slice_key, interaction.si,
                       interaction.li, interaction.tool,
                       [fn(x, y) for x, y in interaction.points],
                       interaction.class_id, meta)


def shifted(interaction, dx, dy):
    """The same gesture with its points moved by (dx, dy)."""
    return transformed(interaction, lambda x, y: (x + dx, y + dy))


def resolve_opts(interaction, raster_scale):
    """How a gesture's geometry applies on a raster of `raster_scale`
    (image px per raster px), from its scale of intent: the stroke width in
    raster px (``meta["px"]``, slide px per screen px at draw time), and
    whether this is a raster other than the one it was drawn on (its
    ``meta["scale"]``), where an extent resolves by outline. A gesture with
    no such meta -- every coupon gesture, every older one -- gets the
    hairline and the seeds it always had."""
    meta = interaction.meta or {}
    sc = float(raster_scale) or 1.0
    width = None
    px = meta.get("px")
    if isinstance(px, (int, float)) and not isinstance(px, bool) and px > 0:
        width = float(px) / sc
    drawn = meta.get("scale")
    off_level = (isinstance(drawn, (int, float)) and not isinstance(drawn, bool)
                 and not math.isclose(float(drawn), sc, rel_tol=1e-6))
    return {"width": width, "off_level": off_level}


def touched_ids_over(interaction, layer, np):
    """``touched_ids`` against a ``LabelLayer``, materialising only the
    gesture's own bounding box.

    ``touched_ids`` needs a raster, and for an in-memory item there is one --
    ``layer.full()`` hands it over and this is exactly the old call. For a
    layer that has no full raster to give (a whole-slide item, where the ids
    live on part of a 4-gigapixel canvas) the gesture's bbox is cropped instead
    and the points are shifted into it, which is the same answer over a rect
    the size of the gesture rather than the size of the slide.
    """
    full = layer.full()
    if full is not None:
        return touched_ids(interaction, full, np, **resolve_opts(interaction, 1.0))
    box = gesture_bbox(interaction, np)
    if box is None:
        return set()
    # A layer that knows exactly where its raster sits resolves on that
    # raster's own grid: the gesture's points go through the same
    # (p - origin) / scale the drawing tools use, and the bbox is a plain
    # slice of the raster. Exact for any scale -- a 1/128.1 overview and the
    # ladder's 2**7 drift half a pixel apart over 700 pixels, enough for a tap
    # on a region's extremum to land one raster pixel off -- and it costs a
    # slice, not a resample.
    placement = getattr(layer, "placement", None)
    crop_raster = getattr(layer, "crop_raster", None)
    if placement is not None and crop_raster is not None:
        ox, oy, sc = placement()
        sc = float(sc) or 1.0
        x, y, w, h = box
        rx0, ry0 = int(np.floor((x - ox) / sc)), int(np.floor((y - oy) / sc))
        rx1, ry1 = int(np.ceil((x + w - ox) / sc)), int(np.ceil((y + h - oy) / sc))
        rshape = getattr(layer, "raster_shape", None)
        if rshape is not None:                   # never ask for more than there is
            rx0, ry0 = max(0, rx0), max(0, ry0)
            rx1, ry1 = min(int(rshape[1]), rx1), min(int(rshape[0]), ry1)
        if rx1 <= rx0 or ry1 <= ry0:
            return set()
        sub = crop_raster(rx0, ry0, rx1 - rx0, ry1 - ry0)
        if sub is None or sub.size == 0:
            return set()
        exact = transformed(interaction,
                            lambda px, py: ((px - ox) / sc - rx0, (py - oy) / sc - ry0))
        return touched_ids(exact, sub, np, **resolve_opts(interaction, sc))
    # Otherwise crop at the RASTER's resolution, not the image's. A layer whose ids live
    # at 1/128 of the image (a coarse overview) would otherwise be asked for
    # its bbox at level 0: a box over the whole item is the whole slide,
    # 47 040 x 90 000 int32 -- seventeen gigabytes to decide which of 33
    # regions a box touched. At the raster's level the crop is the raster,
    # and a gesture resolves the way it does on a coupon slice: one sample per
    # raster pixel, centre-in-box at the edges.
    level = max(0, int(getattr(layer, "native_level", 0) or 0))
    s = float(2 ** level)
    x, y, w, h = box
    ih, iw = layer.shape
    lx0, ly0 = max(0, int(np.floor(x / s))), max(0, int(np.floor(y / s)))
    lx1 = min(int(np.ceil(iw / s)), int(np.ceil((x + w) / s)))
    ly1 = min(int(np.ceil(ih / s)), int(np.ceil((y + h) / s)))
    if lx1 <= lx0 or ly1 <= ly0:
        return set()
    sub = layer.crop(level, lx0, ly0, lx1 - lx0, ly1 - ly0)
    if s == 1.0:
        return touched_ids(shifted(interaction, -lx0, -ly0), sub, np,
                           **resolve_opts(interaction, s))
    scaled = transformed(interaction, lambda px, py: (px / s - lx0, py / s - ly0))
    return touched_ids(scaled, sub, np, **resolve_opts(interaction, s))


def touched_sets(interactions, labels, np, layer=None):
    """[(interaction, touched region-id set)] in creation (= resolution) order.

    The per-interaction sets are what resolution consumes, and callers that
    also need the reverse question (which interactions touch region r?) get it
    from the same single rasterization pass. Pass `layer` when the item's ids
    are served by a ``LabelLayer`` rather than held as one raster; `labels` is
    then only used to size the result."""
    if layer is not None:
        return [(it, touched_ids_over(it, layer, np))
                for it in sorted(interactions, key=lambda it: it.uid)]
    return [(it, touched_ids(it, labels, np))
            for it in sorted(interactions, key=lambda it: it.uid)]


def resolve_slice(interactions, labels, np, layer=None):
    """region_class: uint8 array sized labels.max()+1, 0 = unlabeled.

    Interactions apply in creation (uid) order, so a later gesture paints over
    an earlier one on any region both touch."""
    return resolve_sets(touched_sets(interactions, labels, np, layer), labels, np)


def resolve_sets(sets, labels, np):
    """resolve_slice over already-computed touched_sets output."""
    K = int(labels.max()) + 1 if labels.size else 1
    region_class = np.zeros(max(K, 1), np.uint8)
    for it, ids in sets:
        if ids:
            region_class[list(ids)] = it.class_id
    return region_class


def class_lut(region_class, np, colors=None):
    """(K, 4) uint8 RGBA LUT for the canvas label overlay: region id -> its
    class color. Class-0 rows have alpha 0, so unlabeled regions are invisible
    (the canvas also treats raster values < 0 as transparent background).
    `colors` overrides the default palette (a (MAX_CLASSES, 4) RGBA table --
    the labeler passes the store's user-picked colors)."""
    table = np.asarray(CLASS_COLORS if colors is None else colors, np.uint8)
    return table[region_class]


def preview_lut(K, ids, colors, np, lerp=0.35, emphasize=None):
    """(K, 4) uint8 RGBA LUT for the "will be painted" preview: the regions in
    `ids` in a brighter (lerped `lerp` of the way toward white), fully opaque
    version of their class color, everything else transparent.

    `colors` is either ONE RGBA tuple (every id gets the armed class's color)
    or a mapping id -> RGBA (the accept preview colors each region by its
    predicted class). An RGBA with alpha 0 (class 0) leaves the row
    transparent. `emphasize` names one id (the magic-fill seed) that keeps the
    pure, un-lerped color so the origin of the fill stays visible."""
    K = max(int(K), 1)
    lut = np.zeros((K, 4), np.uint8)
    ids = np.asarray(list(ids), dtype=np.intp)
    ids = ids[(ids >= 0) & (ids < K)]
    if len(ids) == 0:
        return lut
    if isinstance(colors, dict):
        rgba = np.asarray([colors.get(int(i), (0, 0, 0, 0)) for i in ids], np.float32)
    else:
        rgba = np.broadcast_to(np.asarray(colors, np.float32), (len(ids), 4))
    lerp = float(min(max(lerp, 0.0), 1.0))
    rgb = rgba[:, :3] * (1.0 - lerp) + 255.0 * lerp
    lut[ids, :3] = np.clip(rgb, 0, 255).round().astype(np.uint8)
    lut[ids, 3] = np.where(rgba[:, 3] > 0, 255, 0).astype(np.uint8)
    if emphasize is not None and 0 <= int(emphasize) < K:
        e = int(emphasize)
        col = (colors.get(e) if isinstance(colors, dict) else colors)
        if col is not None and col[3] > 0:
            lut[e, :3] = np.asarray(col[:3], np.uint8)
            lut[e, 3] = 255
    return lut


# A blue -> cyan -> yellow -> red ramp for continuous per-region quantities
# (class probability, prediction uncertainty). Hand-rolled rather than pulled
# from matplotlib: this sits on the render path, and keeping it here -- pure
# numpy, np passed in like everything else in this module -- puts it under
# tests/test_labeling.py instead of only the Tk selftest.
_RAMP_STOPS = ((0.00, (49, 54, 149)),
               (0.33, (69, 180, 190)),
               (0.66, (254, 224, 100)),
               (1.00, (165, 15, 21)))


def scalar_lut(values, np, alpha=255, mask=None):
    """(K, 4) uint8 RGBA LUT mapping a per-region scalar in [0, 1] onto the
    ramp, for the canvas label overlay: `values[i]` colors region id `i`.

    `mask` (a bool array over the same ids) marks the regions that have a
    value at all; everything else gets alpha 0, so regions the classifier
    never saw stay invisible instead of rendering as ramp-zero."""
    v = np.clip(np.asarray(values, np.float32), 0.0, 1.0)
    stops = np.asarray([s for s, _c in _RAMP_STOPS], np.float32)
    cols = np.asarray([c for _s, c in _RAMP_STOPS], np.float32)
    out = np.zeros((len(v), 4), np.uint8)
    for ch in range(3):
        out[:, ch] = np.interp(v, stops, cols[:, ch]).round().astype(np.uint8)
    out[:, 3] = int(max(0, min(255, alpha)))
    if mask is not None:
        out[~np.asarray(mask, bool), 3] = 0
    return out
