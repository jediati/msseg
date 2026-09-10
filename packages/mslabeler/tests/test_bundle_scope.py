"""The feature SCOPE: what the compatibility gate's names cannot say.

Two models can carry identical feature names and be incomparable, because a
name says what was measured and not what it was measured on. A whole-slide
labeler produces `mean_blur_s1.5` at pyramid level 4 and at level 0, and a
sigma is in pixels -- the level-4 number describes a neighbourhood sixteen
times wider, and both look perfectly plausible. Nothing in the names catches
that, so it is declared separately and the gate refuses a mismatch.

The other half of the contract is that an app WITHOUT the distinction is
untouched, pickle bytes included: `scope=None` must produce the v4 document it
always has.
"""
import pickle

import pytest

from msseg.labeler import bundle
from msseg.labeler.bundle import ModelBundle, compat_message, model_record_entry


class Dummy:
    def __init__(self, n):
        self.n = n

    def __eq__(self, other):
        return isinstance(other, Dummy) and other.n == self.n


NAMES = ["area", "mean_base"]


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def test_no_scope_on_either_side_is_the_old_behaviour():
    assert compat_message(NAMES, NAMES, "p", "load") is None
    assert compat_message(NAMES, NAMES, "p", "load", None, None) is None
    msg = compat_message(NAMES, ["area"], "p", "load")
    assert msg and "mean_base" in msg


def test_matching_scopes_pass():
    assert compat_message(NAMES, NAMES, "p", "load", "L4", "L4") is None


def test_a_scope_mismatch_is_refused_even_though_the_names_match():
    msg = compat_message(NAMES, NAMES, "slide", "classify", "L0", "L4")
    assert msg is not None
    assert "L0" in msg and "L4" in msg
    # the names are identical, so listing them would say nothing about the fault
    assert "mean_base" not in msg and "area" not in msg


def test_a_scope_appearing_or_vanishing_is_a_mismatch():
    """A model saved before the app had scopes must not silently pass once it
    does -- nothing recorded which regime it came from."""
    assert compat_message(NAMES, NAMES, "p", "load", None, "L4") is not None
    assert compat_message(NAMES, NAMES, "p", "load", "L4", None) is not None


def test_empty_string_counts_as_no_scope():
    assert compat_message(NAMES, NAMES, "p", "load", "", None) is None


def test_the_scope_is_checked_before_the_names():
    """Both wrong: report the scope, which is the fault that explains the
    other one (different regimes measure different channels)."""
    msg = compat_message(NAMES, ["area"], "p", "load", "L0", "L4")
    assert msg and "L0" in msg and "not comparable" in msg


# --------------------------------------------------------------------------- #
# the pickle
# --------------------------------------------------------------------------- #
def test_a_scopeless_bundle_is_byte_identical_to_v4(tmp_path):
    b = ModelBundle(model=Dummy(3), names=NAMES, kind="dense FC")
    doc = b.to_doc()
    assert "scope" not in doc
    assert list(doc) == ["app", "version", "kind", "names", "model", "statistics",
                         "spec", "edge", "stack"]
    assert doc["version"] == bundle.PICKLE_VERSION == 4
    p = str(tmp_path / "m.pkl"); b.save(p)
    assert ModelBundle.load(p) == b
    with open(p, "rb") as f:
        assert pickle.load(f) == doc


def test_a_scoped_bundle_declares_v5_and_round_trips(tmp_path):
    b = ModelBundle(model=Dummy(3), names=NAMES, kind="dense FC", scope="L4")
    doc = b.to_doc()
    assert doc["scope"] == "L4" and doc["version"] == bundle.PICKLE_VERSION_SCOPED == 5
    p = str(tmp_path / "m.pkl"); b.save(p)
    back = ModelBundle.load(p)
    assert back == b and back.scope == "L4"


@pytest.mark.parametrize("doc_scope,want", [(None, None), ("L0", "L0")])
def test_older_pickles_read_back_with_no_scope(doc_scope, want):
    doc = {"app": bundle.DEFAULT_APP_TAG, "version": 4, "names": NAMES,
           "model": Dummy(1), "kind": "dense FC"}
    if doc_scope is not None:
        doc["scope"] = doc_scope
    assert ModelBundle.from_doc(doc).scope == want


def test_the_session_record_carries_the_scope(tmp_path):
    plain = model_record_entry(str(tmp_path / "m.pkl"), NAMES, "dense FC", {}, None, False)
    assert "scope" not in plain, "an app without scopes writes the record it always has"
    scoped = model_record_entry(str(tmp_path / "m.pkl"), NAMES, "dense FC", {}, None,
                                False, "L4")
    assert scoped["scope"] == "L4"
    assert ModelBundle(model=Dummy(1), names=NAMES, kind="dense FC",
                       scope="L4").record_entry(str(tmp_path / "m.pkl"))["scope"] == "L4"
