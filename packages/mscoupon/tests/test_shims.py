"""The `msseg.mscoupon.<module>` shims must be the framework modules themselves.

Pickled classifiers record a step's class as ``msseg.mscoupon.model_search.
FeatureSubset`` / ``msseg.mscoupon.torch_mlp.TorchMLPClassifier``; a pickle
written before the framework split must keep loading. No Tk, no extension."""
import importlib
import pickle

import pytest

PAIRS = [("labeling", "labeling", "LabelStore"),
         ("magic_fill", "magic_fill", "build_ladder"),
         ("model_search", "model_search", "FeatureSubset"),
         ("edge_model", "edge_model", "EdgeModel"),
         ("torch_mlp", "torch_mlp", "TorchMLPClassifier"),
         ("viewer_canvas", "canvas", "SliceCanvas"),
         ("widgets", "widgets", "ScrollFrame")]


@pytest.mark.parametrize("old,new,name", PAIRS)
def test_shim_is_same_object(old, new, name):
    shim = importlib.import_module(f"msseg.mscoupon.{old}")
    real = importlib.import_module(f"msseg.labeler.{new}")
    assert getattr(shim, name) is getattr(real, name)
    # private helpers travel too (tests and the selftest reach a few)
    for k, v in vars(real).items():
        if not k.startswith("__"):
            assert getattr(shim, k) is v, k


def test_common_reexports():
    from msseg.mscoupon.common import FeatureTable
    from msseg.labeler.table import FeatureTable as Real
    assert FeatureTable is Real
    from msseg.mscoupon import session, config_io
    from msseg.labeler import session_doc
    assert session.build_session_doc is session_doc.build_session_doc
    assert config_io.session_path is session_doc.session_path
    assert config_io.app_data_dir is session_doc.app_data_dir


def _dumps_under_old_path(obj, old_module):
    """A pickle stream that names the class by its pre-split module path, the
    way every classifier saved before the framework existed does."""
    cls = type(obj)
    real = cls.__module__
    cls.__module__ = old_module
    try:
        return pickle.dumps(obj, protocol=4)
    finally:
        cls.__module__ = real


def test_old_pickle_path_resolves():
    pytest.importorskip("sklearn")
    from msseg.labeler.model_search import FeatureSubset
    blob = _dumps_under_old_path(FeatureSubset(["a", "b"], ["b"]), "msseg.mscoupon.model_search")
    assert b"msseg.mscoupon.model_search" in blob
    back = pickle.loads(blob)
    assert type(back) is FeatureSubset
    assert back.names == ["a", "b"] and back.keep == ["b"]


def test_old_torch_pickle_path_resolves():
    pytest.importorskip("sklearn"); pytest.importorskip("torch")
    from msseg.labeler.torch_mlp import TorchMLPClassifier
    blob = _dumps_under_old_path(TorchMLPClassifier(hidden_layer_sizes=(4,)), "msseg.mscoupon.torch_mlp")
    assert b"msseg.mscoupon.torch_mlp" in blob
    assert type(pickle.loads(blob)) is TorchMLPClassifier
