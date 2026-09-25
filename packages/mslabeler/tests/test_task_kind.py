"""A task is region- or polyline-based, fixed at creation, written on every
task (session v4); a labeler document whose tasks do not say is refused."""
import pytest

from msseg.labeler.session_doc import (DEFAULT_TASK_KIND, TASK_KINDS, build_session_doc,
                                       labeler_refusal, normalise_kind, session_doc_from_json)
from msseg.labeler.task import Task


def test_kinds_and_the_default():
    assert TASK_KINDS == ("region", "polyline") and DEFAULT_TASK_KIND == "region"
    assert Task.new("t").kind == "region"
    assert Task.new("walls", kind="polyline").kind == "polyline"
    with pytest.raises(ValueError):
        Task.new("x", kind="squiggly")


def test_the_kind_round_trips_and_survives_duplicate():
    t = Task.new("walls", workflow="p", kind="polyline")
    doc = t.to_doc()
    assert doc["kind"] == "polyline"
    back = Task.from_doc(doc)
    assert back.kind == "polyline"
    assert t.duplicate("walls 2").kind == "polyline"


def test_normalise_kind_notes_junk():
    notes = []
    assert normalise_kind("polyline", notes) == "polyline" and not notes
    assert normalise_kind(None, notes) is None and not notes
    assert normalise_kind("both", notes, "task 'x'") is None and "unknown kind" in notes[0]


def test_the_reader_passes_the_kind_and_a_tasks_less_document_is_one_region_task():
    t = Task.new("walls", workflow="p", kind="polyline")
    doc = build_session_doc(app="labeler", folders=[], sequences=[],
                            profiles=[{"name": "p"}], active_profile="p",
                            run={}, view={}, tasks=[t.to_doc()], active_task=t.uid)
    assert doc["session_version"] == 4
    back = session_doc_from_json(doc)
    assert back["tasks"][0]["kind"] == "polyline"
    assert labeler_refusal(doc) is None
    plain = session_doc_from_json({"profiles": [{"name": "p"}], "active_profile": "p"})
    assert plain["tasks"][0]["kind"] == "region"
    assert labeler_refusal({"profiles": [{"name": "p"}]}) is None, "the viewers' documents load"


def test_an_older_labeler_document_is_refused():
    old = {"session_version": 3, "profiles": [{"name": "p"}], "active_profile": "p",
           "tasks": [{"uid": "t_000001", "name": "a", "annotations": {}}]}
    why = labeler_refusal(old)
    assert why is not None and "older labeler" in why
    mixed = dict(old, tasks=[{"uid": "t_1", "name": "a", "kind": "region"},
                             {"uid": "t_2", "name": "b"}])
    assert labeler_refusal(mixed) is not None
    assert labeler_refusal(None) is None and labeler_refusal({"tasks": "junk"}) is None
