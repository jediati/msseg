"""ModelBundle: the classifier pickle's shape across v1..v4, and the
profile-compatibility message."""
import os
import pickle

import pytest

from msseg.labeler import bundle
from msseg.labeler.bundle import ModelBundle, compat_message, model_record_entry


class Dummy:                       # a picklable stand-in for a fitted estimator
    def __init__(self, w): self.w = w
    def __eq__(self, o): return isinstance(o, Dummy) and o.w == self.w


def test_v4_roundtrip(tmp_path):
    b = ModelBundle(model=Dummy(3), names=["a", "b"], kind="custom FC -> edges",
                    statistics={"channels": ["base"]}, spec={"hidden": [16, 8]},
                    edge={"n_in": 3}, stack={"custom_hidden": "16-8"})
    doc = b.to_doc()
    assert list(doc) == ["app", "version", "kind", "names", "model", "statistics", "spec", "edge", "stack"]
    assert doc["version"] == bundle.PICKLE_VERSION == 4 and doc["app"] == bundle.DEFAULT_APP_TAG
    p = str(tmp_path / "m.pkl"); b.save(p)
    back = ModelBundle.load(p)
    assert back == b
    with open(p, "rb") as f:
        assert pickle.load(f) == doc


@pytest.mark.parametrize("doc,kind,stats,spec,edge", [
    ({"app": bundle.DEFAULT_APP_TAG, "names": ["a"], "model": Dummy(1)}, "random forest", {}, None, None),
    ({"app": bundle.DEFAULT_APP_TAG, "version": 2, "names": ["a"], "model": Dummy(1),
      "kind": "dense FC", "statistics": {"x": 1}}, "dense FC", {"x": 1}, None, None),
    ({"app": bundle.DEFAULT_APP_TAG, "version": 3, "names": ["a"], "model": Dummy(1),
      "kind": "dense (tuned)", "statistics": {}, "spec": {"hidden": [4]}}, "dense (tuned)", {}, {"hidden": [4]}, None),
    ({"app": bundle.DEFAULT_APP_TAG, "version": 4, "names": ["a"], "model": Dummy(1),
      "kind": "custom FC -> edges", "statistics": {}, "spec": None, "edge": {"n_in": 2},
      "stack": {"custom_hidden": "4"}}, "custom FC -> edges", {}, None, {"n_in": 2}),
])
def test_older_layouts_load_by_feature_detection(doc, kind, stats, spec, edge):
    b = ModelBundle.from_doc(doc)
    assert (b.kind, b.statistics, b.spec, b.edge) == (kind, stats, spec, edge)
    assert b.names == ["a"] and b.model == Dummy(1)
    assert b.stack == (doc.get("stack") or {})


@pytest.mark.parametrize("doc", [None, [], {"app": "other", "names": ["a"], "model": 1},
                                 {"app": bundle.DEFAULT_APP_TAG, "names": [], "model": 1},
                                 {"app": bundle.DEFAULT_APP_TAG, "names": ["a"]}])
def test_rejects_foreign_files(doc):
    with pytest.raises(ValueError, match="not a labeler classifier file"):
        ModelBundle.from_doc(doc)


def test_record_entry(tmp_path):
    b = ModelBundle(model=Dummy(1), names=["a", "b"], kind="dense FC", statistics={"s": 1},
                    edge={"n_in": 1})
    e = b.record_entry(str(tmp_path / "x.pkl"))
    assert e == model_record_entry(str(tmp_path / "x.pkl"), ["a", "b"], "dense FC", {"s": 1}, None, True)
    assert os.path.isabs(e["path"]) and e["fingerprint"] == ["a", "b"] and e["edge"] is True


def test_compat_message_is_a_set_comparison():
    assert compat_message(["a", "b"], ["b", "a"], "p", "load") is None
    msg = compat_message(["a", "mean_blur_s1.5"], ["a", "bbox_h", "bbox_w"], "default", "load")
    assert msg == ("Model does not match profile 'default' statistics (load) - "
                   "model needs: mean_blur_s1.5; profile adds: bbox_h, bbox_w. "
                   "Switch profiles or retrain.")
    many = compat_message([], [f"f{i}" for i in range(9)], "p", "train")
    assert many.endswith("profile adds: f0, f1, f2, f3, f4, f5…. Switch profiles or retrain.")
