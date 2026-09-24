"""Places (tissue) and enrolment (work): the pure half of stage 4."""
import re

from msseg.mspath import places as P
from msseg.mspath.app import _clean_rois

UID = re.compile(r"p_[0-9a-f]{6}")


def roi(x, y, w=64, h=64, level=0, **kw):
    return dict(level=level, x=x, y=y, w=w, h=h, **kw)


def test_clean_rois_keeps_a_places_identity_and_provenance():
    good = roi(10, 20, uid="p_abcdef", note="duct", origin={"task": "t_1", "reason": "view"})
    assert _clean_rois([good]) == [good]
    junk = roi(10, 20, uid=5, note="", origin="x")
    assert _clean_rois([junk]) == [roi(10, 20)], "junk identity fields are dropped, the rect kept"


def test_uids_are_assigned_once_and_deduped():
    rois = [roi(0, 0), roi(0, 0, level=2), roi(5, 5, uid="p_111111"), roi(9, 9, uid="p_111111")]
    notes = []
    assert P.ensure_uids(rois, notes=notes) == 3
    uids = [r["uid"] for r in rois]
    assert all(UID.fullmatch(u) for u in uids) and len(set(uids)) == 4
    assert uids[2] == "p_111111" and len(notes) == 1, "the first keeps it, the duplicate is renamed"
    assert P.ensure_uids(rois) == 0, "idempotent"
    assert P.find_by_rect(rois, (0, 0, 64, 64)) == 0 and P.find_by_rect(rois, (1, 1, 1, 1)) is None
    assert P.find_by_uid(rois, uids[3]) == 3


def test_enrolment_reads_and_edits():
    a = roi(0, 0, level=0, uid="p_a")
    b = roi(100, 0, level=2, uid="p_b")
    # None: every place at its default level, never the overview.
    assert P.level_of(None, "s", a) == 0 and P.level_of(None, "s", b) == 2
    assert not P.overview_enrolled(None, "s")
    e = {}
    assert P.level_of(e, "s", a) is None and not P.overview_enrolled(e, "s")
    P.enrol(e, "s", "p_a", 1)
    P.enrol(e, "s", P.OVERVIEW)
    assert P.level_of(e, "s", a) == 1 and P.overview_enrolled(e, "s")
    assert e == {"s": {"p_a": 1, "overview": None}}
    P.enrol(e, "s", "p_a", 0)
    assert P.level_of(e, "s", a) == 0, "enrolling again re-levels"
    assert P.unenrol(e, "s", "p_a") and not P.unenrol(e, "s", "p_a")
    assert P.unenrol(e, "s", P.OVERVIEW) and e == {}, "an emptied slide entry goes"


def test_materialise_all_never_enrols_the_overview():
    slides = {"s1": [roi(0, 0, uid="p_a"), roi(9, 9, level=3, uid="p_b")], "s2": []}
    assert P.materialise_all(slides) == {"s1": {"p_a": 0, "p_b": 3}}


def test_drops():
    e = {"s1": {"p_a": 0, "p_b": 1, "overview": None}, "s2": {"p_c": 0}, "gone": {"p_x": 0}}
    assert P.drop_place(e, "s1", "p_a") and not P.drop_place(None, "s1", "p_a")
    assert P.drop_slide(e, "s2") and "s2" not in e and not P.drop_slide(None, "s2")
    slides = {"s1": [roi(0, 0, uid="p_a")]}                   # p_b no longer exists
    notes = []
    assert P.drop_dangling(e, slides, notes) == 1 and notes
    assert e == {"s1": {"overview": None}, "gone": {"p_x": 0}}, \
        "a slide not in the session keeps its entries"
