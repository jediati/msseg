"""The seams a labeler is built on.

Four protocols describe what the generic shells need from a data source, and
nothing more. The coupon labeler implements them over its TIFF sequences and
the in-memory ``ComputeEngine``; a whole-slide labeler would implement them
over a tiled pyramid, ROI-scoped compute and a catalogue of slides/ROIs. All
four are ``typing.Protocol`` classes: any object with the right methods is an
implementation, no inheritance required.

* ``ItemCatalogue`` -- the things that can be annotated (coupon: slices), each
  under an opaque string ``ItemKey`` that annotations bind to. Keys must be
  stable across sessions: they are what ``annotations.json`` stores.
* ``RegionProvider`` -- the current region decomposition of an item: a label
  raster, its per-region statistics table, the region-adjacency arcs, the
  seam graph (the polylines between regions), and a ``commit`` that every
  cache keys on -- the record's IDENTITY (``record_keys.py``): the interned
  id of what the record is a function of (the prime that made the labels,
  the measurement, the persistence, the selection), so equal ids mean equal
  content and returning to earlier parameters finds the earlier records.
* ``ImageSource`` -- base pixels by level and region, so a canvas can draw a
  gigapixel image without holding it (an in-memory array has one level).
* ``LabelLayer`` -- a region-id raster served by crop, for the same reason.
"""
from __future__ import annotations

from typing import Any, Dict, Hashable, List, Optional, Protocol, Sequence, Tuple, TypedDict, \
    runtime_checkable

ItemKey = str


class RegionRecord(TypedDict, total=False):
    """One item's decomposition; ``commit`` is its identity (equal commits,
    equal labels and rows -- never a hash of parameters alone, since region
    ids differ from one prime to the next). ``labels`` is an int32 raster of
    region ids (-1 = background); ``stats`` a ``FeatureTableLike`` with one row
    per living region; ``arcs`` the region graph as ``{"a", "b", "saddle" |
    None, "source"}`` parallel arrays, or None when only pixel adjacency is
    available (the provider derives and caches it on request)."""
    commit: int
    labels: Any
    stats: Any
    arcs: Optional[Dict[str, Any]]
    kept: Any
    cc: Any


@runtime_checkable
class FeatureTableLike(Protocol):
    names: Sequence[str]
    values: Any                                   # (n_rows, n_fields) float64

    @property
    def n_rows(self) -> int: ...
    def column(self, name: str): ...              # one field over every row, or None
    def row_of_feature(self, feature_id: int): ...  # name -> value for one region, or None


@runtime_checkable
class ItemCatalogue(Protocol):
    def keys(self) -> List[ItemKey]:
        """Every annotatable item, in navigation order."""
        ...

    def key_of(self, *index) -> Optional[ItemKey]:
        """The key for an implementation-specific address (coupon: ``(si, li)``);
        None when the address names nothing."""
        ...

    def index_of(self, key: ItemKey) -> Optional[Tuple]:
        """The inverse of ``key_of``; None for an unknown key."""
        ...

    def label(self, key: ItemKey) -> str:
        """Display text for navigation and messages."""
        ...

    def group_of(self, key: ItemKey) -> Hashable:
        """The cross-validation group of the item's regions: rows that share a
        group are held out together (leave-items-out)."""
        ...

    def tree(self) -> List[Dict[str, Any]]:
        """``[{"key": ItemKey | None, "label": str, "children": [...]}]`` for a
        navigation tree; a node with a None key is a container."""
        ...

    # -- binding: what a gesture on an item is keyed by ------------------ #
    def binding_of(self, key: ItemKey) -> Tuple[str, Optional[Tuple[int, int, int, int]]]:
        """``(slide_key, rect)``: the key gestures drawn on `key` are stored
        under -- the SLIDE, so every item covering the same place shares
        them -- and the item's place on it as ``(x, y, w, h)`` in image
        pixels, or None for the whole slide. An app whose item IS its slide
        (coupon: a slice file) returns ``(key, None)``. ``index_of`` must
        resolve a slide key too (to the slide's first row)."""
        ...

    def rebase(self, key: str) -> Optional[Tuple[str, Optional[int], Optional[float]]]:
        """For a key written when gestures were keyed by ITEM (mspath:
        ``"folder/slide.svs@level#rect"``): ``(slide_key, level, scale)`` --
        the slide's key and the item's level and slide-pixels-per-raster-pixel,
        recorded on the gesture as its scale of intent. None when `key` is
        not such a key (a slide key, a slice file)."""
        ...


@runtime_checkable
class RegionProvider(Protocol):
    @property
    def commit(self) -> int:
        """A generation counter that moves whenever the current parameters
        may have. Informational: which record is current is decided by
        ``record(key)``, and caches compare a record's own ``commit``."""
        ...

    def keys(self) -> List[ItemKey]:
        """The items that can be (or have been) decomposed."""
        ...

    def record(self, key: ItemKey) -> Optional[RegionRecord]:
        """The item's record at the current commit if it is cached, else None.
        Cheap; never computes."""
        ...

    def ensure_record(self, key: ItemKey) -> Optional[RegionRecord]:
        """The item's current record, computing it synchronously when needed;
        None when it cannot be produced."""
        ...

    def request(self, key: ItemKey) -> None:
        """Ask for the item's record asynchronously; completion arrives through
        ``poll()``."""
        ...

    def pending(self) -> bool:
        """True while background work is in flight."""
        ...

    def poll(self) -> List[Tuple]:
        """Drain completed background work into display events."""
        ...

    def arcs(self, key: ItemKey, np) -> Optional[Dict[str, Any]]:
        """The item's region-adjacency arcs at the current commit (the
        record's own, or pixel adjacency derived once and cached); None
        without a record."""
        ...

    def label_layer(self, key: ItemKey) -> Optional["LabelLayer"]:
        """The item's current region raster as a ``LabelLayer`` (its ``rev`` is
        the record's commit); None without a record."""
        ...

    def seams(self, key: ItemKey, np):
        """The item's seam graph (``seams.SeamGraph``, placed like the record's
        raster) at the current commit, derived once and cached on the record;
        None without a record."""
        ...


@runtime_checkable
class ImageSource(Protocol):
    path: Optional[str]

    @property
    def levels(self) -> int: ...                  # 1 for an in-memory array
    @property
    def channels(self) -> int: ...                # 1, or 3+ for colour
    def level_shape(self, level: int) -> Tuple[int, int]: ...      # (h, w) at `level`
    def level_scale(self, level: int) -> float: ...                # full-res px per level px
    def best_level(self, scale: float) -> int: ...                 # coarsest level still finer than `scale`
    def value_range(self) -> Tuple[float, float]: ...              # for windowing
    def read_region(self, level: int, x: int, y: int, w: int, h: int): ...  # (h, w) | (h, w, C) at `level`


@runtime_checkable
class LabelLayer(Protocol):
    @property
    def shape(self) -> Tuple[int, int]: ...       # full-resolution (h, w)
    @property
    def n_ids(self) -> int: ...                   # LUT length: max id + 1
    @property
    def rev(self) -> int: ...                     # cache key (the record's commit)
    def crop(self, level: int, x: int, y: int, w: int, h: int): ...   # int32 ids at `level`, -1 bg
    def id_at(self, x: int, y: int) -> int: ...                       # full-resolution lookup
    def full(self): ...                                               # the whole raster if cheap, else None
