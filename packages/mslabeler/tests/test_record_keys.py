"""record_keys: an id per content key, and the bounded per-item cache."""
from msseg.labeler.record_keys import Interner, RecordCache, records_per_item


def test_an_interned_key_keeps_its_id_and_fresh_matches_nothing():
    ids = Interner(start=5)
    a = ids.id_of(("item", 1, "meas", 0.5))
    b = ids.id_of(("item", 2, "meas", 0.5))
    assert a == 5 and b == 6 and ids.id_of(("item", 1, "meas", 0.5)) == a
    f = ids.fresh()
    assert f not in (a, b) and ids.id_of(("new",)) == f + 1 and len(ids) == 3


def test_the_cache_is_bounded_per_item_and_least_recently_used_goes_first():
    c = RecordCache(cap=2)
    c.put("k", 1, {"commit": 1})
    c.put("k", 2, {"commit": 2})
    assert c.get("k", 1)["commit"] == 1          # touch 1: 2 is now the oldest
    c.put("k", 3, {"commit": 3})
    assert c.get("k", 2) is None and c.get("k", 1) and c.get("k", 3)
    assert c.count("k") == 2 and len(c) == 1 and c.latest("k")["commit"] == 3
    c.put("other", 9, {})
    assert len(c) == 2 and sorted(c.items()) == ["k", "other"]
    c.drop("k")
    assert c.get("k", 1) is None and len(c) == 1
    c.clear()
    assert len(c) == 0


def test_shared_decompositions_are_bounded_too():
    c = RecordCache(cap=1)
    labels = object()
    c.share("k", (1, 0.5), (labels, None))
    assert c.shared("k", (1, 0.5))[0] is labels
    c.share("k", (1, 0.7), (object(), None))
    assert c.shared("k", (1, 0.5)) is None


def test_the_cap_reads_the_environment(monkeypatch):
    monkeypatch.setenv("MSSEG_RECORDS_PER_ITEM", "7")
    assert records_per_item() == 7 and RecordCache().cap == 7
    monkeypatch.setenv("MSSEG_RECORDS_PER_ITEM", "junk")
    assert records_per_item() == 4
    monkeypatch.setenv("MSSEG_RECORDS_PER_ITEM", "0")
    assert records_per_item() == 1
