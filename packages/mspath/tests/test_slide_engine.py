"""``SlideEngine``: the two things that make items comparable with each other.

Neither needs a slide or the compiled pipeline, and both are the kind of
mistake that produces plausible-looking numbers rather than an error:

* the six positional columns must be mapped into slide coordinates, and
  nothing else may move;
* the persistence threshold must be pinned per LEVEL -- shared across items at
  one resolution, never carried between resolutions.
"""
import numpy as np
import pytest

from msseg.mspath import engine as E
from msseg.mspath import items as I


NAMES = ["feature_id", "area", "bbox_w", "bbox_h", "min_x", "max_x", "min_y", "max_y",
         "mean_base", "ext_x", "ext_y", "ext_base"]


def row(**kw):
    out = np.zeros(len(NAMES), np.float64)
    for k, v in kw.items():
        out[NAMES.index(k)] = v
    return out


class FakePipe:
    def __init__(self, value_range=2.0):
        self._range = float(value_range)
        self.selected = None

    def value_range(self):
        return self._range

    def select_persistence(self, v):
        self.selected = float(v)

    def release_gpu(self):
        pass


def primed(level=4, scale=16.0, origin=(1000, 2000), value_range=2.0, halo=0):
    return E.Primed(I.overview("f/a.svs", level), FakePipe(value_range), None, None,
                    origin, scale, level, (10, 10), halo)


# --------------------------------------------------------------------------- #
# positional mapping
# --------------------------------------------------------------------------- #
def test_only_the_positional_columns_move():
    values = np.stack([row(feature_id=7, area=123, bbox_w=5, min_x=2, max_x=8,
                           min_y=3, max_y=9, mean_base=0.5, ext_x=4, ext_y=6,
                           ext_base=0.25)])
    out = E.SlideEngine._to_slide_coords(values, NAMES, (1000, 2000), 16.0, np)
    got = dict(zip(NAMES, out[0]))
    assert got["min_x"] == 2 * 16 + 1000 and got["max_x"] == 8 * 16 + 1000
    assert got["min_y"] == 3 * 16 + 2000 and got["max_y"] == 9 * 16 + 2000
    assert got["ext_x"] == 4 * 16 + 1000 and got["ext_y"] == 6 * 16 + 2000
    # measurements and sizes stay in the item's own level pixels -- which is
    # exactly why a model is valid at one level only
    assert got["area"] == 123 and got["bbox_w"] == 5
    assert got["mean_base"] == 0.5 and got["ext_base"] == 0.25
    assert got["feature_id"] == 7


def test_the_mapping_does_not_touch_the_input():
    values = np.stack([row(min_x=2, ext_y=6)])
    before = values.copy()
    E.SlideEngine._to_slide_coords(values, NAMES, (10, 20), 2.0, np)
    assert np.array_equal(values, before)


def test_identity_at_level_zero_with_no_origin():
    values = np.stack([row(min_x=2, max_x=8, min_y=3, max_y=9, ext_x=4, ext_y=6)])
    out = E.SlideEngine._to_slide_coords(values, NAMES, (0, 0), 1.0, np)
    assert np.array_equal(out, values)


def test_a_missing_positional_column_is_not_an_error():
    """`statistics.extremum = false` drops ext_x/ext_y from the schema."""
    names = [n for n in NAMES if not n.startswith("ext_")]
    values = np.zeros((2, len(names)), np.float64)
    values[:, names.index("min_x")] = [1.0, 2.0]
    out = E.SlideEngine._to_slide_coords(values, names, (100, 0), 4.0, np)
    assert list(out[:, names.index("min_x")]) == [104.0, 108.0]


# --------------------------------------------------------------------------- #
# persistence pinning
# --------------------------------------------------------------------------- #
def profile(pct=10.0, absolute=None):
    msc = {"persistence_percent": pct}
    if absolute is not None:
        msc["persistence_absolute"] = absolute
    return {"msc": msc}


def test_a_percentage_is_pinned_once_per_level():
    eng = E.SlideEngine()
    first = primed(level=4, value_range=2.0)
    assert eng.resolve_persistence(first, profile(10.0)) == pytest.approx(0.2)
    # a second item at the same level SHARES it, even with a different range --
    # one threshold per level is what makes two items comparable
    second = primed(level=4, value_range=50.0)
    assert eng.resolve_persistence(second, profile(10.0)) == pytest.approx(0.2)


def test_the_slider_moves_the_threshold_but_not_the_reference():
    """The pin is the RANGE; the threshold is derived from the current
    percentage every time. The first version cached the threshold, which
    made the persistence slider do nothing."""
    eng = E.SlideEngine()
    first = primed(level=4, value_range=2.0)
    assert eng.resolve_persistence(first, profile(10.0)) == pytest.approx(0.2)
    assert eng.resolve_persistence(first, profile(25.0)) == pytest.approx(0.5)
    # a later item at the level still resolves against the FIRST range
    later = primed(level=4, value_range=50.0)
    assert eng.resolve_persistence(later, profile(25.0)) == pytest.approx(0.5)
    assert eng.level_range == {4: 2.0}
    assert eng.persistence_abs[4] == pytest.approx(0.5), "the readout follows the slider"


def test_levels_do_not_share_a_pin():
    """A sigma is in pixels, so the topology field at level 4 is not the field
    at level 0; a threshold carried across collapses one of them."""
    eng = E.SlideEngine()
    eng.resolve_persistence(primed(level=4, value_range=0.5), profile(10.0))
    got = eng.resolve_persistence(primed(level=0, value_range=2.0), profile(10.0))
    assert got == pytest.approx(0.2)
    assert set(eng.persistence_abs) == {0, 4}


def test_an_explicit_absolute_wins_and_pins_nothing():
    eng = E.SlideEngine()
    assert eng.resolve_persistence(primed(value_range=2.0),
                                   profile(10.0, absolute=0.75)) == 0.75
    assert eng.persistence_abs == {} and eng.level_range == {}


def test_reset_forgets_the_pins():
    eng = E.SlideEngine()
    eng.resolve_persistence(primed(), profile(10.0))
    before = eng.commit_id
    eng.reset()
    assert eng.persistence_abs == {} and eng.level_range == {} and eng.commit_id > before


# --------------------------------------------------------------------------- #
# generations and the live-pipeline budget
# --------------------------------------------------------------------------- #
def test_records_fall_stale_on_a_commit():
    eng = E.SlideEngine()
    eng.slices["k"] = {"commit": eng.commit_id, "labels": None}
    assert eng.record("k") is not None
    eng.commit_selection()
    assert eng.record("k") is None
    assert "k" not in eng.slices, "a stale record must not be kept as well as missed"


def test_the_live_budget_releases_the_oldest_pipelines(monkeypatch):
    monkeypatch.setenv("MSPATH_LIVE_ITEMS", "2")
    eng = E.SlideEngine()
    for i in range(4):
        eng._install(f"k{i}", primed(level=i))
    live = [k for k, p in eng.primed.items() if p.live]
    assert live == ["k2", "k3"], live
    assert all(k in eng.primed for k in ("k0", "k1")), "an evicted item is still known"


def test_touch_protects_the_item_in_use(monkeypatch):
    monkeypatch.setenv("MSPATH_LIVE_ITEMS", "2")
    eng = E.SlideEngine()
    eng._install("a", primed()); eng._install("b", primed())
    eng.touch("a")                       # 'a' is what the user is looking at
    eng._install("c", primed())
    assert [k for k, p in eng.primed.items() if p.live] == ["a", "c"]


def test_live_budget_reads_the_environment(monkeypatch):
    monkeypatch.setenv("MSPATH_LIVE_ITEMS", "7")
    assert E.live_budget() == 7
    monkeypatch.setenv("MSPATH_LIVE_ITEMS", "not a number")
    assert E.live_budget() == E.DEFAULT_LIVE_ITEMS
    monkeypatch.setenv("MSPATH_LIVE_ITEMS", "0")
    assert E.live_budget() == 1, "a budget of zero would release what is on screen"


# --------------------------------------------------------------------------- #
# one unreadable item does not take the run with it
# --------------------------------------------------------------------------- #
def test_a_failing_item_is_reported_and_the_rest_still_prime(monkeypatch):
    eng = E.SlideEngine()
    good = I.overview("f/a.svs", 4)
    bad = I.overview("f/broken.tiff", 4)
    primed = []

    def fake_prime(item, profile, halo=0, quiet=True):
        if item is bad:
            raise RuntimeError("no pyramid backend could open it")
        primed.append(item.key)
    monkeypatch.setattr(eng, "prime_item", fake_prime)
    eng._run_worker([good, bad, good], {}, 0)
    events = eng.poll()
    kinds = [ev[0] for ev in events]
    assert primed == [good.key, good.key], "the items after the failure must still prime"
    assert kinds.count("item_error") == 1 and kinds[-1] == "primed"
    err = [ev for ev in events if ev[0] == "item_error"][0]
    assert err[1] == bad.key and "no pyramid backend" in err[2]
    assert not eng.pending_work()


def test_a_run_where_nothing_primes_is_an_error(monkeypatch):
    eng = E.SlideEngine()
    bad = I.overview("f/broken.tiff", 4)
    monkeypatch.setattr(eng, "prime_item",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")))
    eng._run_worker([bad], {}, 0)
    kinds = [ev[0] for ev in eng.poll()]
    assert "error" in kinds and "primed" not in kinds


def test_an_incremental_run_ends_with_item_primed_not_primed(monkeypatch):
    """The labeler drops every prediction on "primed". One ROI added to a
    session whose other items are exactly as they were must not do that."""
    eng = E.SlideEngine()
    monkeypatch.setattr(eng, "prime_item", lambda *a, **k: None)
    item = I.roi("f/a.svs", 0, 0, 0, 64, 64)
    eng._incremental = True
    eng._run_worker([item], {}, 0)
    assert [ev[0] for ev in eng.poll()] == ["item_done", "progress", "item_primed"]
    eng._incremental = False
    eng._run_worker([item], {}, 0)
    assert [ev[0] for ev in eng.poll()][-1] == "primed"
