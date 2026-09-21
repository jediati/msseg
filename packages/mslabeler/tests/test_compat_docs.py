"""annotations.json compatibility: a v2 file round-trips byte-identically and a
v1 file (bare basenames) migrates on rebind() exactly as before the split."""
import json
import os

from msseg.labeler.labeling import LabelStore

DATA = os.path.join(os.path.dirname(__file__), "data")


def _dump(doc):
    return json.dumps(doc, indent=2)


def test_v2_roundtrip_is_byte_identical():
    text = open(os.path.join(DATA, "annotations_v2.json"), encoding="utf-8").read()
    doc = json.loads(text)
    store = LabelStore.from_json(doc)
    assert _dump(store.to_json()) == _dump(doc)
    assert store.n_classes == 4 and store.color(2) == "#123456"
    kinds = [it.tool for it in store.interactions]
    assert kinds == ["squiggle", "box", "polygon", "taps", "squiggle"]
    assert store.interactions[3].meta["tool"] == "magic"
    # rebind against a session that carries the qualified keys: geometry unchanged
    store.rebind([{"name": "a", "folder": "data", "files": [r"C:\d\s0.tiff", r"C:\d\s1.tiff"]}])
    assert _dump(store.to_json()) == _dump(doc)
    assert [it.bound for it in store.interactions] == [True, True, True, True, False]


def test_v1_basenames_migrate_when_unambiguous():
    doc = json.load(open(os.path.join(DATA, "annotations_v1.json"), encoding="utf-8"))
    store = LabelStore.from_json(doc)
    unbound = store.rebind([
        {"name": "a", "folder": "data", "files": [r"C:\d\s0.tiff", r"C:\d\dup.tiff"]},
        {"name": "b", "folder": "more", "files": [r"C:\m\dup.tiff"]}])
    assert unbound == 2
    assert [it.slice_key for it in store.interactions] == ["data/s0.tiff", "dup.tiff", "gone.tiff"]
    assert [(it.si, it.li) for it in store.interactions] == [(0, 0), (None, None), (None, None)]
    out = store.to_json()
    assert out["version"] == 2 and out["interactions"][0]["slice"] == "data/s0.tiff"


def test_v3_roundtrip_is_byte_identical_and_v2_readers_ignore_seams():
    text = open(os.path.join(DATA, "annotations_v3.json"), encoding="utf-8").read()
    doc = json.loads(text)
    store = LabelStore.from_json(doc)
    assert _dump(store.to_json()) == _dump(doc)
    assert doc["version"] == 3
    assert [it.tool for it in store.seams] == ["scope", "trace"]
    assert store.seams[1].meta["toll"] == "feature"
    # The region gestures are exactly the v2 file's: a reader that ignores
    # the "seams" key sees the v2 document (modulo the version number).
    v2 = json.load(open(os.path.join(DATA, "annotations_v2.json"), encoding="utf-8"))
    stripped = {k: v for k, v in doc.items() if k != "seams"}
    stripped["version"] = 2
    assert _dump(stripped) == _dump(v2)
    # Dropping the seams gives back a byte-identical v2 document.
    store.remove_many([it.uid for it in store.seams])
    assert _dump(store.to_json()) == _dump(v2)
