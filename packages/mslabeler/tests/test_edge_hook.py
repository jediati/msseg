"""The edge set's ``arc_labels_of`` hook: the label derivation (derive.py)
decides which arcs are labelled and how, per item, over the arcs' own order;
the default path is unchanged."""
import numpy as np

from msseg.labeler import derive, magic_fill
from msseg.labeler.labeling import LabelStore
from msseg.labeler.training import TrainingSetBuilder

from test_training import items, store_two_classes


def _arcs(key, rec):
    return magic_fill.arcs_from_labels(rec["labels"], np)


def test_hook_overrides_per_item_and_none_keeps_the_default():
    b = TrainingSetBuilder()
    it = items()
    names = b.feature_names(it[0][2])
    store = store_two_classes()
    _X, _c, _g, _e, base, _n = b.edge_set(it, store, names, _arcs, np)
    seen = []

    def hook(key, rec, arcs, region_class, sets):
        seen.append((key, len(arcs["a"]), int(region_class.max()), len(sets)))
        if key == "k1":
            return None                                     # keep the default there
        n = len(arcs["a"])
        return np.ones(n, bool), np.zeros(n, bool)          # k0: every arc labelled, all same

    _X, _c, _g, _e, edges, _n = b.edge_set(it, store, names, _arcs, np, arc_labels_of=hook)
    assert [s[0] for s in seen] == ["k0", "k1"] and seen[0][2] == 2 and seen[0][3] == 2
    k0 = edges["slice"] == 10
    assert edges["both"][k0].all() and not edges["diff"][k0].any()
    assert np.array_equal(edges["both"][~k0], base["both"][~k0])
    assert np.array_equal(edges["diff"][~k0], base["diff"][~k0])
    # A memo hit still asks the hook (annotations may have changed) and
    # `diff` never exceeds `both`.
    def contra(key, rec, arcs, region_class, sets):
        n = len(arcs["a"])
        return np.zeros(n, bool), np.ones(n, bool)
    _X, _c, _g, _e, edges2, _n = b.edge_set(it, store, names, _arcs, np, arc_labels_of=contra)
    assert not edges2["both"].any() and not edges2["diff"].any()


def test_derive_through_the_hook_labels_an_extents_outside_arcs():
    b = TrainingSetBuilder()
    it = items(1)
    names = b.feature_names(it[0][2])
    store = LabelStore(n_classes=3)
    store.add("taps", [(5.0, 5.0), (14.0, 5.0)], 1, "k0", meta={"extent": True})   # blocks 0, 2

    def hook(key, rec, arcs, region_class, sets):
        return derive.arc_labels(arcs, np, region_class, derive.extent_sets(sets))

    _X, _c, _g, _e, edges, _n = b.edge_set(it, store, names, _arcs, np, arc_labels_of=hook)
    rows = {int(f): i for i, f in enumerate(it[0][2].column("feature_id"))}
    pairs = {(int(a), int(b_)): (bool(x), bool(y))
             for a, b_, x, y in zip(edges["a"], edges["b"], edges["both"], edges["diff"])}
    key = lambda p, q: (min(rows[p], rows[q]), max(rows[p], rows[q]))
    assert pairs[key(0, 2)] == (True, False), "inside the extent: same"
    assert pairs[key(0, 5)] == (True, True) and pairs[key(2, 9)] == (True, True), "outside: different"
    assert pairs[key(5, 9)] == (False, False), "neither flank is the extent's"
    # Without the hook the same store labels only the arc inside the extent.
    _X, _c, _g, _e, plain, _n = b.edge_set(it, store, names, _arcs, np)
    assert plain["both"].sum() == 1
