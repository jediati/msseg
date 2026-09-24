"""A task's enrolment in the session document: None (every place, the
legacy and coupon reading) is absent, {} (nothing) is written, and the
reader is total."""
import json

from msseg.labeler.session_doc import (build_session_doc, normalise_enrolled,
                                       session_doc_from_json)
from msseg.labeler.task import Task


def test_normalise_enrolled_forms_and_junk():
    assert normalise_enrolled(None) is None
    assert normalise_enrolled({}) == {}
    good = {"wsi/a.svs": {"overview": 4, "p_aaaaaa": 0, "p_bbbbbb": "2"}}
    assert normalise_enrolled(good) == {"wsi/a.svs": {"overview": None, "p_aaaaaa": 0,
                                                      "p_bbbbbb": 2}}, \
        "the overview's level is the workflow's; levels coerce to int"
    notes = []
    assert normalise_enrolled("all", notes) is None and notes, "junk reads as everything"
    notes = []
    got = normalise_enrolled({"wsi/a.svs": {"p_1": -1, "p_2": "x", "p_3": 1},
                              "": {"overview": None}, "wsi/b.svs": 7}, notes)
    assert got == {"wsi/a.svs": {"p_3": 1}} and len(notes) == 4
    assert normalise_enrolled({"wsi/a.svs": {}}) == {}, "an empty slide entry is dropped"


def test_task_doc_writes_enrolment_only_when_set():
    t = Task.new("gland", workflow="w")
    assert t.enrolled is None
    doc = t.to_doc()
    assert "enrolled" not in doc, "a task without enrolment writes the document it always has"
    assert Task.from_doc(doc).enrolled is None
    t.enrolled = {}
    assert t.to_doc()["enrolled"] == {}, "nothing enrolled is written"
    assert Task.from_doc(t.to_doc()).enrolled == {}
    t.enrolled = {"wsi/a.svs": {"overview": None, "p_aaaaaa": 0}}
    doc = t.to_doc()
    assert list(doc)[-1] == "enrolled", "written last"
    back = Task.from_doc(json.loads(json.dumps(doc)))
    assert back.enrolled == t.enrolled
    notes = []
    assert Task.from_doc(dict(doc, enrolled=["x"]), notes).enrolled is None and notes


def test_duplicate_copies_enrolment():
    t = Task.new("gland", workflow="w")
    t.enrolled = {"wsi/a.svs": {"p_aaaaaa": 0}}
    d = t.duplicate("gland (2)")
    assert d.enrolled == t.enrolled and d.enrolled is not t.enrolled
    d.enrolled["wsi/a.svs"]["p_aaaaaa"] = 2
    assert t.enrolled["wsi/a.svs"]["p_aaaaaa"] == 0, "a structural copy"
    assert Task.new("x").duplicate("y").enrolled is None


def test_session_round_trip_keeps_enrolment():
    a = Task.new("gland", workflow="p")
    a.enrolled = {"wsi/a.svs": {"overview": None, "p_aaaaaa": 0}}
    b = Task.new("stroma", workflow="p")                 # None: every place
    c = Task.new("empty", workflow="p")
    c.enrolled = {}
    doc = build_session_doc(app="labeler", folders=[], sequences=[], profiles=[{"name": "p"}],
                            active_profile="p", run={}, view={},
                            tasks=[a.to_doc(), b.to_doc(), c.to_doc()], active_task=a.uid)
    back = session_doc_from_json(json.loads(json.dumps(doc)))
    ta, tb, tc = back["tasks"]
    assert ta["enrolled"] == a.enrolled
    assert "enrolled" not in tb, "absent stays absent"
    assert tc["enrolled"] == {}
    assert [Task.from_doc(t).enrolled for t in back["tasks"]] == [a.enrolled, None, {}]
