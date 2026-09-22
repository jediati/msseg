"""Gestures keyed by the slide: the per-item query, the extent that drives
it, and the rebind that moves an item-keyed gesture (a stage-1 store) to its
slide while recording the item's level as the scale of intent."""
import json
import os

import numpy as np

from msseg.labeler.labeling import (Interaction, LabelStore, gesture_bbox, gesture_extent,
                                     gesture_meets)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

SEQS = [{"name": "s", "folder": "spears", "files": ["spears/a.tiff", "spears/b.tiff"]}]
SLIDE = "WSI/a.svs"


def _it(uid, pts, meta=None, tool="squiggle"):
    return Interaction(uid, SLIDE, None, None, tool, pts, 1, meta=meta)


def test_extent_is_the_old_bbox_for_a_plain_gesture():
    it = _it(1, [(10.2, 20.7), (14.0, 22.0)])
    assert gesture_extent(it) == (9, 19, 15, 23)
    assert gesture_bbox(it, np) == (9, 19, 7, 5)
    assert gesture_bbox(_it(2, [(5, 5)], tool="taps"), np) == (4, 4, 3, 3), "a click keeps area"
    assert gesture_extent(_it(3, [])) is None and gesture_bbox(_it(3, []), np) is None


def test_extent_grows_with_the_stroke_width_and_the_outline():
    it = _it(1, [(10, 10), (20, 10)], meta={"px": 6.0})
    assert gesture_extent(it) == (10 - 4, 10 - 4, 20 + 4, 10 + 4), "half the width plus one"
    it = _it(2, [(50, 50)], meta={"outline": [[[40, 40], [60, 40], [60, 62], [40, 62]]]}, tool="taps")
    assert gesture_extent(it) == (39, 39, 61, 63), "the outline is part of the extent"
    it = _it(3, [(1, 1)], meta={"px": True})           # a bool is not a width
    assert gesture_extent(it) == (0, 0, 2, 2)


def test_meets_and_for_item():
    store = LabelStore(n_classes=3)
    a = store.add("squiggle", [(100, 100), (110, 100)], 1, SLIDE).uid   # inside the rect
    b = store.add("box", [(990, 990), (1010, 1010)], 2, SLIDE).uid      # straddles its edge
    c = store.add("taps", [(5000, 5000)], 1, SLIDE).uid                 # far away
    d = store.add("taps", [(150, 150)], 1, "WSI/other.svs").uid         # another slide
    e = store.add_seam("scope", [(90, 90), (120, 120)], 2, SLIDE).uid
    store.add_seam("trace", [(3000, 3000), (3010, 3000)], 2, SLIDE)
    rect = (0, 0, 1000, 1000)
    assert [it.uid for it in store.for_item(SLIDE, rect)] == [a, b]
    assert gesture_meets(store.get(c), rect) is False
    assert [it.uid for it in store.for_item(SLIDE, None)] == [a, b, c], "the whole slide"
    assert [it.uid for it in store.for_item("WSI/other.svs", rect)] == [d]
    assert [it.uid for it in store.for_item_seams(SLIDE, rect)] == [e]
    assert len(store.for_item_seams(SLIDE, None)) == 2
    # A rect that only touches the padded extent counts: the pad is what
    # lets a click on the very edge of an ROI belong to it.
    assert [it.uid for it in store.for_item(SLIDE, (111, 95, 10, 10))] == [a]
    assert store.for_item(SLIDE, (113, 95, 10, 10)) == []
    assert store.for_slice(SLIDE) == store.for_item(SLIDE, None), "for_slice stays the exact query"


def _rebase(key):
    """The mspath rule, in miniature: 'slide@level#rect' -> (slide, level, 2**level)."""
    if "@" not in key:
        return None
    slide, rest = key.rsplit("@", 1)
    level = int(rest.split("#", 1)[0])
    return slide, level, float(2 ** level)


def _resolve(key):
    return (0, 0) if key == SLIDE else None


def test_rebind_moves_item_keys_to_the_slide_and_records_the_level():
    store = LabelStore(n_classes=3)
    roi = store.add("taps", [(3, 3)], 1, f"{SLIDE}@0#0,0,64,64").uid
    ovw = store.add("squiggle", [(1, 1), (2, 2)], 1, f"{SLIDE}@4",
                    meta={"level": 9, "tool": "x"}).uid
    sm = store.add_seam("scope", [(0, 0), (9, 9)], 2, f"{SLIDE}@4").uid
    plain = store.add("box", [(0, 0), (1, 1)], 1, "spears/a.tiff").uid
    assert store.rebind(SEQS, resolve=_resolve, rebase=_rebase) == 0
    r, o, s, p = (store.get(u) for u in (roi, ovw, sm, plain))
    assert r.slice_key == o.slice_key == s.slice_key == SLIDE
    assert r.meta == {"level": 0, "scale": 1.0}
    assert o.meta == {"level": 9, "tool": "x", "scale": 16.0}, "an existing level is kept"
    assert s.meta == {"level": 4, "scale": 16.0}
    assert (r.si, r.li) == (0, 0) and o.bound and s.bound, "bound through resolve"
    assert p.slice_key == "spears/a.tiff" and (p.si, p.li) == (0, 0), "a slice key is untouched"
    # Idempotent: a second pass (an undo snapshot) changes nothing.
    doc = store.to_json()
    store.rebind(SEQS, resolve=_resolve, rebase=_rebase)
    assert store.to_json() == doc
    # Without a resolver a rebased key stays unbound but keeps its new key.
    again = LabelStore.from_json(json.loads(json.dumps(doc)))
    again.get(roi).slice_key = f"{SLIDE}@2#8,8,16,16"
    assert again.rebind(SEQS, rebase=_rebase) == 3
    assert again.get(roi).slice_key == SLIDE and again.get(roi).meta == {"level": 0, "scale": 1.0}, \
        "setdefault: the level recorded at the first rebase wins"
    # A rebase that raises binds nothing and breaks nothing.
    def boom(_k):
        raise RuntimeError("no")
    assert again.rebind(SEQS, resolve=_resolve, rebase=boom) == 0


def test_compat_baselines_hold_with_a_rebase():
    for name in ("annotations_v2.json", "annotations_v3.json"):
        with open(os.path.join(DATA, name), "r", encoding="utf-8") as f:
            text = f.read()
        doc = json.loads(text)
        store = LabelStore.from_json(doc)
        # The session the compat tests rebind against (it reproduces the
        # stored hints); a rebase that declines everything is a no-op.
        store.rebind([{"name": "a", "folder": "data", "files": [r"C:\d\s0.tiff", r"C:\d\s1.tiff"]}],
                     rebase=lambda k: None)
        assert json.dumps(store.to_json(), indent=2) == json.dumps(doc, indent=2)
