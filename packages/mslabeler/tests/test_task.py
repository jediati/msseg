"""The task object (msseg.labeler.task): plain data, its document form, and
the invariants the shell relies on. Headless."""
import re

from msseg.labeler.context import ContextSpec
from msseg.labeler.labeling import LabelStore
from msseg.labeler.task import (TASK_VIEW_KEYS, ModelStack, Task, TaskCaches,
                                dedupe_task_name, new_uid)


def test_uid_shape_and_collision_retry():
    uid = new_uid()
    assert re.fullmatch(r"t_[0-9a-f]{6}", uid)
    # Retrying past a taken id: with a 24-bit space a collision is unlikely,
    # so force the point by declaring every id taken except by exclusion.
    other = new_uid(taken=[uid])
    assert other != uid
    assert dedupe_task_name("gland", ["gland", "gland (2)"]) == "gland (3)"


def test_model_stack_defaults_empty_and_reset():
    m = ModelStack()
    assert m.empty and m.kind == "dense FC"
    assert isinstance(m.context, ContextSpec) and m.context.empty()
    m.clf = object(); m.names = ["a"]; m.kind = "random forest"; m.scope = "L4"
    m.context = ContextSpec(kinds=("ring_mean",))
    assert not m.empty
    m.reset()
    assert m.empty and m.names is None and m.kind == "dense FC" and m.scope is None
    assert m.context.empty()
    # A seam model alone is still "not empty": there is something to score with.
    m.seam = object()
    assert not m.empty


def test_caches_clear():
    c = TaskCaches()
    c.pred[(0, 0)] = (1, None, None)
    c.seam_pred["k"] = (1, None)
    c.pred_store_rev = 3
    c.cm_cell = (1, 2)
    c.clear()
    assert c.pred == {} and c.seam_pred == {} and c.pred_store_rev is None and c.cm_cell is None


def test_new_task_and_doc_round_trip():
    t = Task.new("gland detector", workflow="H&E L0", n_classes=4)
    assert t.workflow == "H&E L0" and t.store.n_classes == 4 and t.gesture_count == 0
    t.store.set_name(1, "gland")
    t.store.add("squiggle", [(1, 1), (2, 2)], 1, "f/a.tif", 0, 0)
    t.models.append({"path": "/nowhere/m.pkl", "kind": "dense FC", "fingerprint": ["a"]})
    t.view = {"model_kind": "random forest", "model_search": {"trials": 3},
              "panes": [0.2, 0.5],            # not a task key: must not ride
              "context": {"kinds": []}}
    doc = t.to_doc()
    assert list(doc) == ["uid", "name", "workflow", "annotations", "models", "view"]
    assert doc["view"] == {"model_kind": "random forest", "model_search": {"trials": 3},
                           "context": {"kinds": []}}
    assert set(doc["view"]) <= set(TASK_VIEW_KEYS)
    assert doc["annotations"]["classes"][0] == {"id": 1, "color": t.store.color(1), "name": "gland"}
    back = Task.from_doc(doc)
    assert back.uid == t.uid and back.name == t.name and back.workflow == t.workflow
    assert back.store.to_json() == t.store.to_json()
    assert back.models == t.models and back.view == doc["view"]
    assert back.model_pending is False          # the recorded pickle does not exist
    assert back.model.empty and back.undo == [] and back.caches.pred == {}


def test_unbound_workflow_is_omitted_from_the_doc():
    t = Task.new("t")
    assert "workflow" not in t.to_doc()
    assert Task.from_doc(t.to_doc()).workflow is None


def test_duplicate_copies_vocabulary_not_gestures():
    t = Task.new("stroma", workflow="w", n_classes=3)
    t.store.set_color(1, "#123456")
    t.store.set_name(2, "not stroma")
    t.store.add("box", [(0, 0), (3, 3)], 1, "f/a.tif", 0, 0)
    t.view = {"model_kind": "dense (tuned)", "neighbours": {"custom_hidden": "8"}}
    t.model.clf = object()
    d = t.duplicate("stroma (2)")
    assert d.uid != t.uid and d.name == "stroma (2)" and d.workflow == "w"
    assert d.store.n_classes == 3 and d.store.color(1) == "#123456" and d.store.name(2) == "not stroma"
    assert d.gesture_count == 0 and d.model.empty and d.models == []
    assert d.view == t.view and d.view is not t.view
    d.view["neighbours"]["custom_hidden"] = "16"
    assert t.view["neighbours"]["custom_hidden"] == "8", "a structural copy, not a shared dict"


def test_from_doc_is_total(tmp_path):
    t = Task.from_doc({"name": "", "annotations": "junk", "models": "junk", "view": 7})
    assert t.name == "task" and re.fullmatch(r"t_[0-9a-f]{6}", t.uid)
    assert isinstance(t.store, LabelStore) and t.gesture_count == 0
    assert t.models == [] and t.view == {} and t.workflow is None
    notes = []
    bad = Task.from_doc({"uid": "t_000000", "name": "x",
                         "annotations": {"interactions": [{"no_uid": 1}]}}, notes)
    assert bad.uid == "t_000000" and bad.gesture_count == 0
    assert notes and "annotations not restored" in notes[0]
    # A record whose pickle exists marks the task's model as pending a load.
    p = tmp_path / "m.pkl"; p.write_bytes(b"x")
    pend = Task.from_doc({"name": "y", "models": [{"path": str(p), "kind": "dense FC"}]})
    assert pend.model_pending is True
