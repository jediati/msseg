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
def test_records_are_found_by_what_they_are_a_function_of():
    """A record's id is its content key: a new generation alone drops
    nothing, other parameters miss, and going back finds the old record."""
    eng = E.SlideEngine()
    p = primed(level=4, value_range=2.0)
    eng.primed[p.item.key] = p
    eng.set_params(profile(10.0))
    rid = eng.record_id(p.item.key, p, eng._peek_persistence(p, {"persistence_percent": 10.0}))
    eng.records.put(p.item.key, rid, {"commit": rid, "labels": None})
    assert eng.record(p.item.key)["commit"] == rid
    eng.commit_selection()
    assert eng.record(p.item.key) is not None, "a generation alone changes nothing"
    eng.commit_selection(profile(20.0))
    assert eng.record(p.item.key) is None, "another persistence is another record"
    eng.commit_selection(profile(10.0))
    assert eng.record(p.item.key)["commit"] == rid, "and going back finds it"
    p.pipe_id += 1
    assert eng.record(p.item.key) is None, "a re-prime never aliases an older record"


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


# --------------------------------------------------------------------------- #
# the measurement, separately from the field (a real prime on a synthetic slide)
# --------------------------------------------------------------------------- #
def _slide_profile(channels=("base",), filters=None, pct=5.0):
    return {"filters": filters if filters is not None else [
                {"operation": "color", "params": {"method": "luminance"}},
                {"operation": "blur", "params": {"sigma": 1.0}}],
            "base_filters": [],
            "input": {"color": {"channels": 3}},
            "msc": {"manifold": "ascending", "persistence_percent": pct,
                    "accurate_ascending": False, "accurate_descending": False,
                    "simplification": "merge_forest"},
            "statistics": {"channels": list(channels), "reductions": ["mean", "std"],
                           "extremum": True}}


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    pytest.importorskip("tifffile")
    ext = pytest.importorskip("msseg.mscoupon.mscoupon_py")
    if not hasattr(ext.Msc2DPipeline, "remeasure"):
        pytest.skip("extension predates Msc2DPipeline.remeasure")
    from msseg.mspath.selftest import _synthetic_slide
    folder = tmp_path_factory.mktemp("slides")
    return _synthetic_slide(str(folder))


def _engine_on(path):
    eng = E.SlideEngine()
    eng.register("f/s.tiff", path)
    return eng, I.roi("f/s.tiff", 0, 64, 64, 256, 192)


def test_a_statistics_change_remeasures_a_live_item(synthetic, monkeypatch):
    eng, item = _engine_on(synthetic)
    a = _slide_profile()
    b = _slide_profile(("base", {"kind": "blur", "sigmas": [2.0], "source": "color"}))
    eng.prime_item(item, a, halo=8)
    rec_a = eng.ensure_record(item.key, a)
    labels_a = np.array(rec_a["labels"], copy=True)
    assert not eng.measure_stale(item.key, a)
    assert eng.measure_stale(item.key, b) and not eng.needs_prime(item.key, b, 8)

    primes = []
    orig = E.SlideEngine.prime_item
    monkeypatch.setattr(E.SlideEngine, "prime_item",
                        lambda self, *x, **k: primes.append(1) or orig(self, *x, **k))
    eng.commit_selection()
    rec_b = eng.ensure_record(item.key, b)
    assert primes == [], "a statistics change re-measures, it does not prime"
    assert np.array_equal(rec_b["labels"], labels_a), "the MSC and its labels are kept"
    assert "mean_blur_c0_s2" in rec_b["stats"].names
    assert "mean_blur_c0_s2" not in rec_a["stats"].names
    assert not eng.measure_stale(item.key, b)

    # ... and the rows are what a fresh prime under b measures
    fresh, _ = _engine_on(synthetic)
    fresh.level_range = dict(eng.level_range)
    monkeypatch.setattr(E.SlideEngine, "prime_item", orig)
    fresh.prime_item(item, b, halo=8)
    rec_f = fresh.ensure_record(item.key, b)
    assert rec_f["stats"].names == rec_b["stats"].names
    assert np.array_equal(rec_f["labels"], rec_b["labels"])
    assert np.allclose(rec_f["stats"].values, rec_b["stats"].values, equal_nan=True)


def test_use_field_keeps_every_field_and_its_pins(synthetic):
    eng, item = _engine_on(synthetic)
    a = _slide_profile()
    eng.prime_item(item, a, halo=8)
    eng.level_range[0] = 1.0
    stats_only = _slide_profile(("base", "color"))
    assert eng.use_field(stats_only, halo=8) == 1, "same field: the same slot"
    assert eng.primed[item.key].live and eng.level_range == {0: 1.0}
    assert eng.use_field(a, halo=4) == 0, "a different halo is a different read"
    assert item.key not in eng.primed and eng.level_range == {0: 1.0}

    eng.prime_item(item, a, halo=8)
    first = eng.primed[item.key]
    other_chain = _slide_profile(filters=[{"operation": "color", "params": {"method": "luminance"}}])
    assert eng.use_field(other_chain, halo=8) == 0
    assert not eng.primed and eng.level_range == {}, "another field: its own slot and pins"
    eng.prime_item(item, other_chain, halo=8)
    eng.ensure_record(item.key, other_chain)
    assert eng.level_range[0] == eng.primed[item.key].value_range
    assert eng.use_field(a, halo=8) == 1, "switching back finds the first field's pipe"
    assert eng.primed[item.key] is first and first.live and eng.level_range == {0: 1.0}


def test_records_by_identity_on_a_real_prime(synthetic, monkeypatch):
    eng, item = _engine_on(synthetic)
    a = _slide_profile()
    b = _slide_profile(("base", {"kind": "blur", "sigmas": [2.0], "source": "color"}))
    eng.prime_item(item, a, halo=8)
    rec_a = eng.ensure_record(item.key, a)
    rec_b = eng.ensure_record(item.key, b)
    assert rec_b["commit"] != rec_a["commit"]
    assert rec_b["labels"] is rec_a["labels"], "one decomposition, two measurements"
    calls = []
    monkeypatch.setattr(eng, "remeasure_item", lambda *x, **k: calls.append(1))
    again = eng.ensure_record(item.key, a)
    assert again is rec_a and calls == [], "back to a: its record, no re-measure"
    eng.set_params(b)
    assert eng.record(item.key) is rec_b
    # a re-prime is a new decomposition: new ids
    monkeypatch.undo()
    eng.prime_item(item, a, halo=8)
    assert eng.ensure_record(item.key, a)["commit"] != rec_a["commit"]


def test_the_record_cache_is_bounded(synthetic, monkeypatch):
    monkeypatch.setenv("MSSEG_RECORDS_PER_ITEM", "2")
    eng, item = _engine_on(synthetic)
    eng.prime_item(item, _slide_profile(pct=2.0), halo=8)
    ids = [eng.ensure_record(item.key, _slide_profile(pct=pct))["commit"]
           for pct in (2.0, 4.0, 6.0)]
    assert len(set(ids)) == 3 and eng.records.count(item.key) == 2
    eng.set_params(_slide_profile(pct=2.0))
    assert eng.record(item.key) is None, "the oldest went first"


def test_run_jobs_prime_into_their_own_field(synthetic):
    eng, item = _engine_on(synthetic)
    a = _slide_profile()
    other = _slide_profile(filters=[{"operation": "color", "params": {"method": "luminance"}}])
    eng.use_field(a, halo=8)
    eng._run_worker([item, (item, other)], a, 8)
    [ev for ev in eng.poll()]
    assert eng.primed[item.key].field == E.field_fingerprint_of(a)
    eng.use_field(other, halo=8)
    assert eng.primed[item.key].field == E.field_fingerprint_of(other)
    assert eng.primed[item.key].pipe_id != eng._slot_dict(E.field_fingerprint_of(a))["primed"][item.key].pipe_id


def test_remeasure_reads_the_chains_the_pipe_was_primed_with(synthetic, monkeypatch):
    """An edited but un-Run chain is a preview: a re-measure must feed the
    pipe the rasters its MSC saw, not the panel's new ones."""
    eng, item = _engine_on(synthetic)
    a = _slide_profile()
    eng.prime_item(item, a, halo=8)
    seen = []
    from msseg.mscoupon.engine import ComputeEngine
    orig = ComputeEngine._leading_color

    def spy(arr, chain, *x, **k):
        seen.append([c["operation"] for c in chain])
        return orig(arr, chain, *x, **k)
    monkeypatch.setattr(ComputeEngine, "_leading_color", staticmethod(spy))
    edited = _slide_profile(("base", "color"), filters=[
        {"operation": "color", "params": {"method": "luminance"}}])
    eng.remeasure_item(item.key, edited)
    assert seen[0] == ["color", "blur"], "the topology chain the pipe was primed with"


def test_a_remeasure_only_run_never_primes(monkeypatch):
    eng = E.SlideEngine()
    item = I.roi("f/a.svs", 0, 0, 0, 64, 64)
    p = E.Primed(item, FakePipe(), None, None, (0, 0), 1.0, 0, (64, 64), 0)
    p.measured = "old"
    eng.primed[item.key] = p
    calls = []
    monkeypatch.setattr(eng, "prime_item", lambda *a, **k: calls.append("prime"))
    monkeypatch.setattr(eng, "remeasure_item", lambda *a, **k: calls.append("measure"))
    monkeypatch.setattr(eng, "can_remeasure", lambda key: True)
    eng._remeasure_only = True
    eng._run_worker([item], {"statistics": {"channels": ["base"]}}, 0)
    assert calls == ["measure"]
    dead = I.roi("f/a.svs", 0, 64, 0, 64, 64)
    calls.clear()
    eng._run_worker([dead], {}, 0)
    assert calls == [], "a released item is skipped, not primed"


def test_a_base_chain_change_is_a_remeasure(synthetic, monkeypatch):
    eng, item = _engine_on(synthetic)
    a = _slide_profile()
    eng.prime_item(item, a, halo=8)
    rec_a = eng.ensure_record(item.key, a)
    base0 = np.array(eng.primed[item.key].base, copy=True)
    b = _slide_profile()
    b["base_filters"] = [{"operation": "color", "params": {"method": "luminance"}},
                         {"operation": "blur", "params": {"sigma": 3.0}}]
    assert not eng.needs_prime(item.key, b, 8), "the base chain is not the field"
    assert eng.measure_stale(item.key, b)
    primes = []
    orig = E.SlideEngine.prime_item
    monkeypatch.setattr(E.SlideEngine, "prime_item",
                        lambda self, *x, **k: primes.append(1) or orig(self, *x, **k))
    rec_b = eng.ensure_record(item.key, b)
    assert primes == [] and rec_b["labels"] is rec_a["labels"]
    col = rec_a["stats"].names.index("mean_base")
    assert not np.allclose(rec_a["stats"].values[:, col], rec_b["stats"].values[:, col])
    assert not np.array_equal(eng.primed[item.key].base, base0)
