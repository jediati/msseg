"""TrainingSetBuilder: annotations + statistics tables -> (X, y, groups, names)
and the edge model's row/edge arrays. Pure numpy; no Tk, no extension."""
import numpy as np
import pytest

from msseg.labeler import magic_fill
from msseg.labeler.labeling import LabelStore
from msseg.labeler.table import FeatureTable
from msseg.labeler.training import TrainingSetBuilder, TrainingProblem

from test_labeling import blocks_raster

NAMES = ["feature_id", "area", "min_x", "ext_x", "ext_y", "mean_base", "std_base", "ext_filtered"]


def table(offset=0.0):
    rows = [[fid, 100.0, 0.0, 1.0, 1.0, fid + offset, 0.5, fid / 10.0] for fid in (0, 2, 5, 9)]
    return FeatureTable(list(NAMES), np.array(rows, float))


def store_two_classes():
    s = LabelStore(n_classes=3)
    s.add("box", [(3.0, 3.0), (16.0, 16.0)], 1, "k0")          # all four -> 1
    s.add("squiggle", [(3.0, 5.0), (15.0, 5.0)], 2, "k0")      # {0, 2} -> 2
    s.add("taps", [(15.0, 15.0)], 1, "k1")                     # region 9 only
    return s


def items(n=2):
    lab = blocks_raster()
    out = []
    for k in range(n):
        rec = {"commit": 1, "labels": lab, "stats": table(k)}
        out.append((f"k{k}", rec, rec["stats"], 10 + k, f"0:{k}"))
    return out


def test_labeled_set_shapes_and_groups():
    b = TrainingSetBuilder()
    X, y, g, names = b.labeled_set(items(), store_two_classes(), np)
    assert names == ["area", "mean_base", "std_base", "ext_filtered"]   # positional dropped
    assert X.shape == (5, 4)                     # 4 labeled on k0 + 1 on k1
    assert sorted(y.tolist()) == [1, 1, 1, 2, 2]
    assert g.tolist() == [10, 10, 10, 10, 11]
    # the row for region 9 on k1 carries k1's mean_base (fid + 1)
    assert X[-1, 1] == 10.0


def test_row_classes_gathers_by_region_id():
    lab = blocks_raster()
    cls = TrainingSetBuilder.row_classes(store_two_classes().for_slice("k0"), lab,
                                         np.array([0, 2, 5, 9, 42]), np)
    assert cls.tolist() == [2, 2, 1, 1, 0]        # unknown id -> unlabeled


def test_problems_are_reported_with_the_status_text():
    b = TrainingSetBuilder()
    with pytest.raises(TrainingProblem, match="No labeled regions"):
        b.labeled_set(items(), LabelStore(), np)
    one = LabelStore(); one.add("box", [(0.0, 0.0), (19.0, 19.0)], 1, "k0")
    with pytest.raises(TrainingProblem, match="at least 2 classes"):
        b.labeled_set(items(), one, np)
    bad = items(); bad[1][2].names.remove("std_base")     # k1 lacks a column
    bad[1][2]._col.pop("std_base")
    with pytest.raises(TrainingProblem, match="incomplete statistics on slice 0:1"):
        b.labeled_set(bad, store_two_classes(), np)


def test_feature_matrix_zeroes_non_finite():
    t = table(); t.values[1, 5] = np.nan
    m = TrainingSetBuilder.feature_matrix(t, ["mean_base", "area"], np)
    assert m.shape == (4, 2) and m[1, 0] == 0.0 and m[1, 1] == 100.0
    assert TrainingSetBuilder.feature_matrix(t, ["nope"], np) is None


def test_edge_set_matches_gather_edges():
    b = TrainingSetBuilder()
    it = items()
    names = b.feature_names(it[0][2])
    X, cls, grp, ext, edges, out_names = b.edge_set(
        it, store_two_classes(), names,
        lambda key, rec: magic_fill.arcs_from_labels(rec["labels"], np), np)
    assert out_names == names and X.shape == (8, 4)
    assert cls.tolist() == [2, 2, 1, 1, 0, 0, 0, 1]
    assert grp.tolist() == [10] * 4 + [11] * 4
    assert ext is not None and ext.tolist() == [0.0, 0.2, 0.5, 0.9] * 2
    assert edges["n_rows"] == 8
    # four blocks in a 2x2 grid: 4 adjacencies per item, in global row indices
    assert len(edges["a"]) == 8 and edges["a"].max() < 8
    assert sorted(set(edges["slice"].tolist())) == [10, 11]



# --------------------------------------------------------------------------- #
# a retrain repeats the minimum: classes are memoized per item, matrices are
# cut to the rows that are used
# --------------------------------------------------------------------------- #
def test_feature_matrix_rows_subset_and_fast_path_agree():
    t = table()
    names = ["mean_base", "std_base"] if "mean_base" in t.names else [n for n in t.names][:2]
    names = [n for n in NAMES if n not in ("feature_id",)][:2]
    full = TrainingSetBuilder.feature_matrix(t, names, np)
    m = np.zeros(t.n_rows, bool); m[::2] = True
    part = TrainingSetBuilder.feature_matrix(t, names, np, rows=m)
    assert np.array_equal(part, full[m])
    idx = np.array([1, 0, 1])
    assert np.array_equal(TrainingSetBuilder.feature_matrix(t, names, np, rows=idx), full[idx])
    assert full.flags.owndata or full.base is None or not np.shares_memory(full, t.values)

    class Columnar:                        # no `values`: the column path
        names = list(t.names)

        def column(self, n):
            return t.column(n)
    assert np.array_equal(TrainingSetBuilder.feature_matrix(Columnar(), names, np), full)
    assert TrainingSetBuilder.feature_matrix(t, names + ["nope"], np) is None


def test_row_classes_are_memoized_per_item_until_its_annotations_change():
    calls = []
    real = TrainingSetBuilder.row_classes

    class Counting(TrainingSetBuilder):
        @staticmethod
        def row_classes(interactions, labels, fids, np_, layer=None):
            calls.append(1)
            return real(interactions, labels, fids, np_, layer)

    b = Counting()
    labels = blocks_raster()
    store = LabelStore(n_classes=3)
    store.add("taps", [(3.0, 3.0)], 1, "k1"); store.add("taps", [(15.0, 15.0)], 2, "k1")
    store.add("taps", [(3.0, 3.0)], 2, "k2"); store.add("taps", [(15.0, 15.0)], 1, "k2")
    recs = {k: {"labels": labels, "commit": 3} for k in ("k1", "k2")}
    tabs = {k: table() for k in ("k1", "k2")}

    def items():
        return [(k, recs[k], tabs[k], 0, k) for k in ("k1", "k2")]
    X1, y1, _g, _n = b.labeled_set(items(), store, np)
    assert len(calls) == 2
    X2, y2, _g, _n = b.labeled_set(items(), store, np)          # retrain, nothing changed
    assert len(calls) == 2 and np.array_equal(X1, X2) and np.array_equal(y1, y2)
    store.add("taps", [(15.0, 3.0)], 1, "k2")                   # one item's annotations
    X3, y3, _g, _n = b.labeled_set(items(), store, np)
    assert len(calls) == 3 and len(y3) == len(y1) + 1
    recs["k1"] = {"labels": labels.copy(), "commit": 4}         # one item re-primed
    b.labeled_set(items(), store, np)
    assert len(calls) == 4
    b.forget("k1"); b.labeled_set(items(), store, np)
    assert len(calls) == 5
    b.forget(); b.labeled_set(items(), store, np)
    assert len(calls) == 7
