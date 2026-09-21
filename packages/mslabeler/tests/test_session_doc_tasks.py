"""The session document with tasks (v3), and how a v2 document reads as one
task. Pure; the framework's default (pass-through) profile reader."""
import json
import re

from msseg.labeler import session_doc
from msseg.labeler.session_doc import (SESSION_DOC_VERSION, SESSION_DOC_VERSION_TASKS,
                                       TASK_VIEW_KEYS, build_session_doc, is_session_doc,
                                       session_doc_from_json)
from msseg.labeler.task import Task

UID = re.compile(r"t_[0-9a-f]{6}")


def _base(**kw):
    args = dict(app="labeler", folders=[{"path": "C:/d", "name": "d"}],
                sequences=[{"name": "s", "folder": "d", "files": ["a.tif"]}],
                profiles=[{"name": "p1"}, {"name": "p2"}], active_profile="p2",
                run={"cores_per_slice": 2, "concurrent_slices": 1},
                view={"alpha": 0.5, "model_kind": "random forest",
                      "model_search": {"trials": 4}, "panes": [0.2]})
    args.update(kw)
    return args


def test_tasks_less_document_is_the_v2_one():
    ann = {"version": 2, "n_classes": 3, "interactions": []}
    models = [{"path": "m.pkl", "kind": "dense FC"}]
    doc = build_session_doc(**_base(), annotations=ann, models=models)
    assert doc["session_version"] == SESSION_DOC_VERSION == 2
    assert "tasks" not in doc and "active_task" not in doc
    assert doc["annotations"] is ann and doc["models"] == models
    # The exact key set the viewers have always written.
    assert list(doc) == ["app", "session_version", "folders", "sequences", "profiles",
                         "active_profile", "run", "view", "annotations", "models"]


def test_v2_document_reads_as_one_task_with_the_view_keys_split_out():
    ann = {"version": 2, "n_classes": 3, "interactions": []}
    doc = build_session_doc(**_base(), annotations=ann,
                            models=[{"path": "m.pkl", "kind": "dense FC",
                                     "context": {"kinds": ["ring_mean"]}}])
    back = session_doc_from_json(json.loads(json.dumps(doc)))
    assert back["session_version"] == 2, "no tasks were read"
    assert len(back["tasks"]) == 1
    t = back["tasks"][0]
    assert UID.fullmatch(t["uid"]) and back["active_task"] == t["uid"]
    assert t["name"] == "p2" and t["workflow"] == "p2", "named after the active profile"
    assert t["annotations"] == ann
    assert t["models"][0]["context"] == {"kinds": ["ring_mean"]}, "the record's context survives"
    assert t["view"] == {"model_kind": "random forest", "model_search": {"trials": 4}}
    assert back["view"] == {"alpha": 0.5, "panes": [0.2]}, "task keys moved out of the window view"
    # Compat: the flat keys are the active task's.
    assert back["annotations"] == ann and back["models"] == t["models"]


def test_pre_rename_labels_key_reads_into_the_task():
    doc = {"session_version": 2, "profiles": [{"name": "p"}], "active_profile": "p",
           "labels": {"version": 2, "n_classes": 2, "interactions": []}}
    back = session_doc_from_json(doc)
    assert back["tasks"][0]["annotations"]["n_classes"] == 2
    assert back["annotations"]["n_classes"] == 2


def test_v3_round_trip_two_tasks():
    a = Task.new("gland", workflow="p1", n_classes=3)
    a.store.set_name(1, "gland")
    a.store.add("squiggle", [(0, 0), (1, 1)], 1, "d/a.tif", 0, 0)
    a.models.append({"path": "ga.pkl", "kind": "dense (tuned)", "fingerprint": ["x"],
                     "scope": "L0", "context": {"kinds": ["ring_mean"]}, "seam": True})
    a.view = {"model_kind": "dense (tuned)", "context": {"kinds": ["ring_mean"]}}
    b = Task.new("stroma", workflow="p2", n_classes=2)
    b.store.add_seam("scope", [(0, 0), (5, 5)], 1, "d/a.tif", 0, 0)
    doc = build_session_doc(**_base(), tasks=[a.to_doc(), b.to_doc()], active_task=b.uid,
                            annotations={"should": "not appear"}, models=[{"path": "no"}])
    assert doc["session_version"] == SESSION_DOC_VERSION_TASKS == 3
    assert is_session_doc(doc)
    assert "annotations" not in doc and "models" not in doc, "one source of truth"
    assert doc["active_task"] == b.uid
    notes = []
    back = session_doc_from_json(json.loads(json.dumps(doc)), notes)
    assert notes == []
    assert back["session_version"] == 3 and back["active_task"] == b.uid
    ta, tb = back["tasks"]
    assert (ta["uid"], ta["name"], ta["workflow"]) == (a.uid, "gland", "p1")
    assert (tb["uid"], tb["name"], tb["workflow"]) == (b.uid, "stroma", "p2")
    assert ta["annotations"] == a.store.to_json()
    assert ta["models"] == [{"path": "ga.pkl", "fingerprint": ["x"], "kind": "dense (tuned)",
                             "statistics": {}, "spec": None, "edge": False, "scope": "L0",
                             "context": {"kinds": ["ring_mean"]}, "seam": True}]
    assert ta["view"] == a.view and tb["view"] == {}
    assert back["annotations"] == tb["annotations"] and back["models"] == []
    # A v3 window view is returned as written: the task keys are only moved
    # out of it on the tasks-less (v2) path, where they had nowhere else to be.
    assert back["view"] == _base()["view"]
    # Rebuilding the tasks from the normalized entries reproduces the stores.
    assert Task.from_doc(ta).store.to_json() == a.store.to_json()
    assert Task.from_doc(ta).store.name(1) == "gland"


def test_task_view_keys_are_the_task_modules():
    from msseg.labeler import task
    assert task.TASK_VIEW_KEYS == TASK_VIEW_KEYS


def test_missing_workflow_is_repointed_and_dupes_are_fixed():
    doc = build_session_doc(**_base(), tasks=[
        {"uid": "t_aaaaaa", "name": "x", "workflow": "gone", "annotations": None,
         "models": [], "view": {}},
        {"uid": "t_aaaaaa", "name": "x", "workflow": None, "annotations": None,
         "models": [], "view": {}},
        {"name": "y", "workflow": "p1", "annotations": {"n_classes": 3, "interactions": []},
         "models": [{"kind": "no path"}], "view": {"panes": [1], "model_kind": "k"}},
    ], active_task="t_nobody")
    notes = []
    back = session_doc_from_json(json.loads(json.dumps(doc)), notes)
    t0, t1, t2 = back["tasks"]
    assert t0["workflow"] == "p2" and any("workflow 'gone'" in n for n in notes)
    assert t0["uid"] == "t_aaaaaa" and t1["uid"] != "t_aaaaaa" and UID.fullmatch(t1["uid"])
    assert t1["workflow"] is None, "an unbound task stays unbound"
    assert (t0["name"], t1["name"]) == ("x", "x (2)")
    assert UID.fullmatch(t2["uid"]) and t2["models"] == [] and t2["view"] == {"model_kind": "k"}
    assert back["active_task"] == "t_aaaaaa", "an unknown active task falls back to the first"


def test_reader_is_total_and_always_has_a_task():
    back = session_doc_from_json(None)
    assert back["session_version"] == 2
    assert len(back["tasks"]) == 1 and back["tasks"][0]["name"] == "default"
    assert back["tasks"][0]["workflow"] == "default" and back["annotations"] is None
    junk = session_doc_from_json({"tasks": "nope", "view": "x", "active_task": 3})
    assert len(junk["tasks"]) == 1 and junk["view"] == {}
    assert session_doc.new_task_uid(["t_000000"]) != "t_000000"
