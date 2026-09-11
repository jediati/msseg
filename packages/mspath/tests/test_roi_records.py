"""ROI records as they survive a session file.

An ROI is the only part of a slide session that is neither a file nor a
profile, so it rides the session document as geometry on its slide. A record
that reads back wrong does not fail loudly -- it becomes an item whose key
names a different place, and annotations bound to the old key quietly detach.
"""
import pytest

from msseg.mspath.app import _clean_rois, MAX_ROI_PX, MIN_ROI_SIDE


GOOD = {"level": 0, "x": 12000, "y": 4000, "w": 4096, "h": 2048}


def test_a_good_record_survives_exactly():
    assert _clean_rois([GOOD]) == [GOOD]


def test_values_are_coerced_to_ints():
    out = _clean_rois([{"level": "2", "x": 10.0, "y": 20.9, "w": 64.5, "h": 64}])
    assert out == [{"level": 2, "x": 10, "y": 20, "w": 64, "h": 64}]
    assert all(isinstance(v, int) for v in out[0].values())


@pytest.mark.parametrize("bad", [
    {},                                                  # nothing at all
    {"level": 0, "x": 1, "y": 2, "w": 3},                # missing a side
    {"level": 0, "x": 1, "y": 2, "w": "wide", "h": 4},   # unparseable
    None,
    "not a record",
])
def test_unusable_records_are_dropped_with_a_note(bad):
    notes = []
    assert _clean_rois([bad], notes) == []
    assert notes, "a dropped ROI must say so"


@pytest.mark.parametrize("bad", [
    {"level": 0, "x": 0, "y": 0, "w": 0, "h": 10},
    {"level": 0, "x": 0, "y": 0, "w": 10, "h": 0},
    {"level": -1, "x": 0, "y": 0, "w": 10, "h": 10},
])
def test_empty_or_impossible_rects_are_dropped(bad):
    notes = []
    assert _clean_rois([bad], notes) == []
    assert notes


def test_a_bad_record_does_not_take_the_good_ones_with_it():
    notes = []
    out = _clean_rois([GOOD, {"w": 1}, dict(GOOD, x=99)], notes)
    assert len(out) == 2 and out[0] == GOOD and out[1]["x"] == 99
    assert len(notes) == 1


def test_negative_origins_are_kept():
    """An ROI may start off the slide's top-left -- the reader zero-fills
    outside a level, which is what lets a halo read across an edge."""
    out = _clean_rois([{"level": 0, "x": -32, "y": -32, "w": 128, "h": 128}])
    assert out and out[0]["x"] == -32


def test_none_and_empty_are_no_rois():
    assert _clean_rois(None) == [] and _clean_rois([]) == []


def test_the_budgets_are_the_measured_ones():
    """4096^2 is 8.7 s and ~1.9 GB peak; 8192^2 is 35 s and 7.2 GB
    (experiments/roi_bench.py). The cap is the first, not the second."""
    assert MAX_ROI_PX == 4096 * 4096
    assert 0 < MIN_ROI_SIDE <= 64
