"""Class names on the LabelStore: display-only, per-task vocabulary, and a
document that is byte-identical to before when no name is set."""
import json

from msseg.labeler.labeling import MAX_CLASSES, LabelStore


def test_unnamed_store_document_is_unchanged():
    store = LabelStore(n_classes=3)
    doc = store.to_json()
    assert all(set(c) == {"id", "color"} for c in doc["classes"])
    assert store.name(1) == "class 1" and not store.has_name(1)


def test_names_round_trip_only_when_set():
    store = LabelStore(n_classes=4)
    r0 = store.rev
    store.set_name(1, "  gland ")
    assert store.name(1) == "gland" and store.has_name(1) and store.rev == r0 + 1
    store.set_name(1, "gland")                 # unchanged: no rev bump
    assert store.rev == r0 + 1
    doc = store.to_json()
    assert doc["classes"][0] == {"id": 1, "color": store.color(1), "name": "gland"}
    assert set(doc["classes"][1]) == {"id", "color"}, "unnamed classes keep the old shape"
    back = LabelStore.from_json(json.loads(json.dumps(doc)))
    assert back.name(1) == "gland" and not back.has_name(2)
    assert back.to_json() == doc


def test_clearing_and_bounds():
    store = LabelStore(n_classes=3)
    store.set_name(2, "lumen")
    r = store.rev
    store.set_name(2, "")                      # empty clears
    assert not store.has_name(2) and store.rev == r + 1
    store.set_name(2, None)                    # already clear: no-op
    assert store.rev == r + 1
    store.set_name(0, "bg"); store.set_name(MAX_CLASSES, "x")
    assert store.names == {}
    # A reader ignores blank / non-string names.
    back = LabelStore.from_json({"n_classes": 3, "interactions": [],
                                 "classes": [{"id": 1, "color": "#000000", "name": "  "},
                                             {"id": 2, "color": "#000000", "name": 5}]})
    assert back.names == {}


def test_set_n_classes_keeps_names_of_surviving_classes():
    store = LabelStore(n_classes=4)
    store.set_name(1, "a"); store.set_name(3, "c")
    store.set_n_classes(2)
    assert store.name(1) == "a"
    assert "name" not in store.to_json()["classes"][0] or store.to_json()["classes"][0]["name"] == "a"
    assert len(store.to_json()["classes"]) == 1
