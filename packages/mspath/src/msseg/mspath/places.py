"""Places on a slide, and which tasks work them.

A **place** is a rect on a slide in level-0 (slide) pixels -- tissue, not
work (docs/design_multi_model_tasks.md §5). It rides the session as the ROI
record it always was, ``{"level", "x", "y", "w", "h"}``, grown by a stable
``uid``, an optional ``note`` and an optional ``origin`` (``{"task", "reason"
[, "score"]}``: which task asked for it and why). Its ``level`` is its
DEFAULT: the level it was cut at, which an enrolment may override.

**Enrolment** belongs to a task (``Task.enrolled``): ``{slide_id:
{"overview": None, "<place uid>": level}}`` -- one entry per place, so a
task works a place at exactly one level; None means "every place at its own
level" (a task written before enrolment). The overview is never enrolled
automatically: working the whole slide is a choice, viewing it is not.

Pure functions over the plain dicts the app keeps (``subsequences[si]
["rois"]`` lists and enrolment dicts); no Tk, no engine.
"""
from __future__ import annotations

import secrets
from typing import Any, Dict, Iterable, List, Optional, Tuple

OVERVIEW = "overview"


def new_uid(taken: Iterable[str] = ()) -> str:
    """A fresh place id, ``p_`` + six hex digits, not in `taken`. Random,
    not a rect hash: one rect may legitimately be two places (an older
    session held the same rect at two levels as two records)."""
    taken = set(taken)
    while True:
        uid = "p_" + secrets.token_hex(3)
        if uid not in taken:
            return uid


def ensure_uids(rois: List[Dict[str, Any]], taken: Iterable[str] = (),
                notes: Optional[List[str]] = None) -> int:
    """Give every place in `rois` a uid, in place; a uid already seen (in
    `taken` or earlier in the list) is replaced, with a note. Returns how
    many were assigned."""
    seen = set(taken)
    n = 0
    for r in rois:
        uid = r.get("uid")
        if not isinstance(uid, str) or not uid or uid in seen:
            if isinstance(uid, str) and uid and notes is not None:
                notes.append(f"duplicate place uid {uid!r} - reassigned")
            r["uid"] = new_uid(seen)
            n += 1
        seen.add(r["uid"])
    return n


def rect_of(place: Dict[str, Any]) -> Tuple[int, int, int, int]:
    return int(place["x"]), int(place["y"]), int(place["w"]), int(place["h"])


def find_by_rect(rois: List[Dict[str, Any]], rect) -> Optional[int]:
    """Index of the first place with exactly this rect, or None."""
    rect = tuple(int(v) for v in rect)
    for i, r in enumerate(rois):
        if rect_of(r) == rect:
            return i
    return None


def find_by_uid(rois: List[Dict[str, Any]], uid: str) -> Optional[int]:
    for i, r in enumerate(rois):
        if r.get("uid") == uid:
            return i
    return None


# --------------------------------------------------------------------------- #
# Enrolment
# --------------------------------------------------------------------------- #
def overview_enrolled(enrolled, slide: str) -> bool:
    """Whether the task works the slide's overview. None ("every place")
    never includes it: the overview is only ever worked by choice."""
    return enrolled is not None and OVERVIEW in (enrolled.get(slide) or {})


def level_of(enrolled, slide: str, place: Dict[str, Any]) -> Optional[int]:
    """The level the task works `place` at, or None when it does not work
    it. None enrolment = every place at its default level."""
    if enrolled is None:
        return int(place["level"])
    lvl = (enrolled.get(slide) or {}).get(place.get("uid"))
    return None if lvl is None else int(lvl)


def enrol(enrolled: Dict, slide: str, key: str, level: Optional[int] = None) -> None:
    """Enrol a place uid (or ``OVERVIEW``, whose level is the workflow's) in
    a task's enrolment, in place; enrolling again re-levels it."""
    enrolled.setdefault(slide, {})[key] = None if key == OVERVIEW else int(level)


def unenrol(enrolled: Dict, slide: str, key: str) -> bool:
    e = enrolled.get(slide)
    if not e or key not in e:
        return False
    del e[key]
    if not e:
        del enrolled[slide]
    return True


def materialise_all(slides: Dict[str, List[Dict[str, Any]]]) -> Dict:
    """The explicit form of None for the places that exist now: every place
    at its default level -- and NOT the overview (it is never enrolled
    automatically). `slides` maps a slide id to its places."""
    out: Dict[str, Dict[str, Optional[int]]] = {}
    for slide, rois in slides.items():
        e = {r["uid"]: int(r["level"]) for r in rois if r.get("uid")}
        if e:
            out[slide] = e
    return out


def drop_place(enrolled, slide: str, uid: str) -> bool:
    """Remove a place from an enrolment (None enrolment: nothing to do)."""
    if enrolled is None:
        return False
    return unenrol(enrolled, slide, uid)


def drop_slide(enrolled, slide: str) -> bool:
    if enrolled is None or slide not in enrolled:
        return False
    del enrolled[slide]
    return True


def drop_dangling(enrolled, slides: Dict[str, List[Dict[str, Any]]],
                  notes: Optional[List[str]] = None, who: str = "task") -> int:
    """Drop enrolments of place uids that no longer exist on a slide the
    session holds (entries for slides not in the session are kept, the way a
    gesture on a missing slide is kept greyed). Returns how many went."""
    if not enrolled:
        return 0
    n = 0
    for slide in list(enrolled):
        if slide not in slides:
            continue
        uids = {r.get("uid") for r in slides[slide]}
        for key in list(enrolled[slide]):
            if key != OVERVIEW and key not in uids:
                del enrolled[slide][key]
                n += 1
                if notes is not None:
                    notes.append(f"{who}: enrolled place {key!r} is gone from {slide!r}")
        if not enrolled[slide]:
            del enrolled[slide]
    return n
