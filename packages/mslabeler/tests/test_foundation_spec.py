"""The foundation spec riding ContextSpec (region tasks) and SeamSpec (polyline
tasks): documents without it are unchanged, the view keeps the hash but not
the basis, the pickle form keeps the basis, context kinds never reduce over
foundation columns.

    pytest packages/mslabeler/tests/test_foundation_spec.py
"""
import json

import numpy as np

from msseg.labeler import context as cx
from msseg.labeler import seam_model as sm
from msseg.labeler.foundation.project import fit_basis
from msseg.labeler.foundation.spec import FoundationSpec, column_names


def fitted(dims=4, seed=0):
    rng = np.random.default_rng(seed)
    fs = FoundationSpec(dims=dims, pools=("mean",))
    return fs.with_basis(fit_basis([rng.normal(size=(5, 6, 10))], dims))


def test_context_spec_without_foundation_is_unchanged():
    spec = cx.ContextSpec(kinds=("ring_mean",))
    assert "foundation" not in spec.to_dict() and "foundation" not in spec.to_dict(full=True)
    assert cx.ContextSpec().empty()
    assert cx.ContextSpec.from_dict(spec.to_dict()) == spec
    assert cx.all_column_names(spec, ["mean_base"]) == cx.column_names(spec, ["mean_base"])


def test_context_spec_carries_foundation():
    fs = fitted()
    spec = cx.ContextSpec(foundation=fs)
    assert not spec.empty() and spec.without_foundation().empty()
    view = spec.to_dict()
    assert view["foundation"]["hash8"] == fs.hash8 and "basis" not in view["foundation"]
    json.dumps(spec.to_dict(full=True))                 # the pickle form is plain data
    back = cx.ContextSpec.from_dict(spec.to_dict(full=True))
    assert back.foundation.hash8 == fs.hash8
    unfitted = cx.ContextSpec.from_dict(view)
    assert unfitted.foundation.basis is None            # the view never restores a basis
    assert spec.key() != cx.ContextSpec(foundation=fitted(seed=1)).key(), "basis in the key"
    assert cx.foundation_column_names(spec) == column_names(fs)
    assert cx.foundation_column_names(unfitted) == []   # names unknown until fitted
    assert spec.brief().startswith("fm:")
    assert "ctx: ring(mean) +fm:" in cx.ContextSpec(kinds=("ring_mean",), foundation=fs).brief()
    entries = cx.schema_entries(cx.ContextSpec(kinds=("ring_mean",), foundation=fs), ["mean_base"])
    assert [e["name"] for e in entries][:len(column_names(fs))] == column_names(fs)


def test_context_kinds_skip_foundation_columns():
    fs = fitted()
    names = ["feature_id", "mean_base"] + column_names(fs)
    spec = cx.ContextSpec(kinds=("ring_mean",), foundation=fs)
    assert cx.source_columns(spec, names) == ["mean_base"]
    assert cx.all_column_names(spec, names) == column_names(fs) + ["ring_mean__mean_base"]


def test_seam_spec_carries_foundation():
    plain = sm.SeamSpec()
    assert "foundation" not in plain.to_dict() and "foundation" not in plain.to_dict(full=True)
    fs = fitted()
    spec = sm.SeamSpec(foundation=fs)
    assert "basis" not in spec.to_dict()["foundation"]
    back = sm.SeamSpec.from_dict(spec.to_dict(full=True))
    assert back.foundation.hash8 == fs.hash8 and "fm:" in back.describe()
    model = sm.SeamModel(model=None, spec=spec, n_in=3, feature_names=["a", "b", "c"])
    assert sm.SeamModel.from_dict(model.to_dict()).spec.foundation.hash8 == fs.hash8
