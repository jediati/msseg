"""FrameProfiler: the accounting behind the canvas' pan timings."""
from msseg.labeler.perf import FrameProfiler


def _prof():
    lines = []
    p = FrameProfiler("canvas", sink=lines.append)
    p.enabled = True
    return p, lines


def test_disabled_is_silent_and_costs_nothing():
    lines = []
    p = FrameProfiler("canvas", sink=lines.append)
    p.enabled = False
    p.event(); p.request(); p.coalesced(); p.add("x", 5.0); p.count("y"); p.set("z", 1)
    with p.frame(out="1x1") as fr:
        assert fr is None
        p.mark("nothing")
    with p.span("also_nothing"):
        pass
    assert lines == []


def test_frame_segments_aggregate_by_name():
    p, lines = _prof()
    with p.frame(out="640x480"):
        p.mark("base.read")
        p.mark("ov.lut")
        p.mark("ov.lut")          # one per overlay
        p.mark("paint")
    head, frame = lines[0], lines[1]
    assert "out=640x480" in head
    assert "base.read" in frame and "ov.lut" in frame and "(2x)" in frame
    assert "paint" in frame


def test_between_frames_work_is_charged_to_the_next_frame():
    """The point of the whole module: per-event work (the annotation redraw)
    is invisible to a timer inside render(), so it lands on the frame that
    finally paints, with the number of events that paid it."""
    p, lines = _prof()
    for _ in range(6):
        p.event()
        with p.span("view_cb"):
            pass
    p.request()
    p.coalesced()
    with p.frame(out="640x480"):
        p.mark("photo")
    joined = "\n".join(lines)
    assert "between frames" in joined
    assert "view_cb" in joined and "(6x)" in joined
    assert "events=6" in joined and "coalesced=1" in joined
    assert "latency(evt->pixels)" in joined


def test_counts_accumulate_and_facts_keep_the_last_value():
    p, lines = _prof()
    p.count("px", 100)
    p.count("px", 50)
    p.set("annot_items", 7)
    p.set("annot_items", 9)      # once per event: the last one is the truth
    with p.frame(out="1x1"):
        p.mark("a")
    tail = lines[-1]
    assert "px=150" in tail and "annot_items=9" in tail


def test_gesture_summary_averages_over_frames():
    p, lines = _prof()
    p.gesture("pan")
    for _ in range(3):
        p.event()
        with p.span("view_cb"):
            pass
        with p.frame(out="1x1"):
            p.mark("paint")
    lines.clear()
    p.end_gesture()
    assert "pan gesture" in lines[0] and "3 frames" in lines[0]
    assert "per-frame average" in lines[1]
    assert "view_cb" in lines[1] and "paint" in lines[1]


def test_gesture_with_no_frames_says_nothing():
    p, lines = _prof()
    p.gesture("pan")
    lines.clear()
    p.end_gesture()
    assert lines == []


def test_set_enabled_toggles_and_reports():
    p, lines = _prof()
    p.add("x", 1.0)
    p.set_enabled(False)
    assert not p.enabled and "off" in lines[-1]
    assert p.set_enabled(True) is True
    assert "ON" in lines[-1]
