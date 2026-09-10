"""What a whole-slide labeler annotates: a slide, a level, and maybe a rect.

The coupon labeler's item is a slice, and its key is the file it came from.
A slide has no such natural unit -- it is one 4-gigapixel file that cannot be
segmented whole (MSCEER narrows cell indices to int32, so anything past
~537 Mpx overflows before it runs out of memory) -- so the item is a *place on
a slide at a resolution*:

* the **overview**, one per slide: a whole coarse level, ~16 Mpx at level 4 of
  a 90 000-row slide, which is one ordinary prime;
* an **ROI**, any number per slide: a rect at a finer level, usually level 0.

``ItemKey`` is what ``annotations.json`` stores, so it has to be stable across
sessions and readable without the session file. It is therefore
self-describing rather than an index into anything:

    slides/a.svs@4                        the overview: a whole level
    slides/a.svs@0#12000,4000,4096,4096   an ROI: a rect at a level

The rect is in **slide coordinates** -- level-0 pixels -- not level ones. An
ROI is a place on the slide; the level it is worked at is a processing choice
that belongs beside it in the key, not baked into its numbers. And because a
level change makes a different key, it makes a different item: labels drawn at
one resolution never silently reattach to another, which is the honest
behaviour given that the coarse and fine decompositions of the same tissue are
not nested.
"""
from __future__ import annotations

import os
import re
from typing import NamedTuple, Optional, Tuple

_KEY_RE = re.compile(r"^(?P<slide>.+)@(?P<level>\d+)(?:#(?P<rect>-?\d+,-?\d+,\d+,\d+))?$")


class Item(NamedTuple):
    """One annotatable place. `rect` is (x, y, w, h) in slide coordinates, or
    None for the whole level (the overview)."""
    slide: str                       # folder-qualified slide identity
    level: int
    rect: Optional[Tuple[int, int, int, int]] = None

    @property
    def is_overview(self) -> bool:
        return self.rect is None

    @property
    def kind(self) -> str:
        return "overview" if self.rect is None else "roi"

    @property
    def key(self) -> str:
        base = f"{self.slide}@{int(self.level)}"
        if self.rect is None:
            return base
        x, y, w, h = self.rect
        return f"{base}#{int(x)},{int(y)},{int(w)},{int(h)}"

    def label(self) -> str:
        """Display text: the slide's basename, then what of it."""
        name = os.path.basename(self.slide)
        if self.rect is None:
            return f"{name} · overview L{self.level}"
        x, y, w, h = self.rect
        return f"{name} · L{self.level} {w}x{h} @({x},{y})"

    def level_rect(self, level_scale: float) -> Tuple[int, int, int, int]:
        """The item's rect in ITS level's pixels. The overview has no rect, so
        callers ask the source for the level shape instead."""
        if self.rect is None:
            raise ValueError("the overview has no rect; use the level's shape")
        s = float(level_scale) or 1.0
        x, y, w, h = self.rect
        return (int(round(x / s)), int(round(y / s)),
                max(1, int(round(w / s))), max(1, int(round(h / s))))


def slide_id(folder: str, path: str) -> str:
    """A slide's identity inside a session: ``"<folder name>/<file name>"``.

    Folder-qualified for the reason the coupon keys are -- basenames collide
    across a session's folders -- and by folder NAME rather than full path, so
    a session survives the data moving.
    """
    name = os.path.basename(os.path.normpath(str(folder))) if folder else ""
    return f"{name}/{os.path.basename(str(path))}"


def parse_key(key: str) -> Optional[Item]:
    """The Item a key names, or None when it is not one of ours."""
    m = _KEY_RE.match(str(key or ""))
    if m is None:
        return None
    rect = m.group("rect")
    return Item(m.group("slide"), int(m.group("level")),
                tuple(int(v) for v in rect.split(",")) if rect else None)


def overview(slide: str, level: int) -> Item:
    return Item(slide, int(level), None)


def roi(slide: str, level: int, x, y, w, h) -> Item:
    return Item(slide, int(level), (int(x), int(y), int(w), int(h)))
