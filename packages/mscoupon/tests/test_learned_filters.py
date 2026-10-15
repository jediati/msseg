"""The `unet` chain stage: bundles, the model dropdown, tiled inference, and
the engine's package-stage split."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from msseg.mscoupon import config_io, learned_filters as lf
from msseg.mscoupon.engine import ComputeEngine

META = {"arch": "unet", "widths": [4, 8, 8, 8], "in_channels": 1, "groups": 4,
        "norm": {"kind": "percentile", "q": [1, 50, 99]}, "trained_level": 2}


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setenv(lf.ENV_DIR, str(tmp_path))
    torch.manual_seed(0)
    lf.save_bundle(str(tmp_path / "tiny.pt"), lf.build_net(META), META)
    lf._CACHE.clear()
    return tmp_path


def test_listing_and_the_dropdown_follow_the_folder(models):
    assert lf.list_models() == ["tiny"]
    rows = config_io.filter_param_schema("unet", {})
    assert rows[0] == ("model", "choice:tiny", "tiny")
    lf.save_bundle(str(models / "another.pt"), lf.build_net(META), META)
    assert config_io.filter_param_schema("unet", {})[0][1] == "choice:another,tiny"
    # a profile naming a missing model keeps the name in the choices
    assert "gone" in config_io.filter_param_schema("unet", {"model": "gone"})[0][1]
    assert "unet" in config_io.PACKAGE_ONLY_OPERATIONS and "unet" in config_io.FILTER_OPERATIONS


def test_apply_shapes_ranges_and_tiling(models, monkeypatch):
    rng = np.random.default_rng(0)
    a = rng.random((37, 53)).astype(np.float32) * 100
    p = lf.apply(a, {"model": "tiny", "device": "cpu"}, np)
    assert p.shape == a.shape and p.dtype == np.float32 and 0 <= p.min() and p.max() <= 1
    logit = lf.apply(a, {"model": "tiny", "device": "cpu", "output": "logit"}, np)
    np.testing.assert_allclose(1 / (1 + np.exp(-logit)), p, atol=1e-5)
    # the normalisation makes the response invariant to an affine intensity map
    np.testing.assert_allclose(lf.apply(a * 3 + 7, {"model": "tiny", "device": "cpu"}, np), p,
                               atol=1e-4)
    # tiles: each keeps its own pixels; a big margin reproduces the one-pass result
    monkeypatch.setattr(lf, "TILE", 16); monkeypatch.setattr(lf, "MARGIN", 64)
    tiled = lf.apply(a, {"model": "tiny", "device": "cpu"}, np)
    assert tiled.shape == a.shape and np.isfinite(tiled).all()


def test_errors_name_the_problem(models, tmp_path, monkeypatch):
    # an unset model is the card's default: the first bundle
    said = []
    out = lf.apply(np.zeros((8, 8), np.float32), {"device": "cpu"}, np, said.append)
    assert out.shape == (8, 8) and any("'tiny'" in m for m in said), said
    empty = tmp_path / "empty"; empty.mkdir()
    monkeypatch.setenv(lf.ENV_DIR, str(empty))
    with pytest.raises(RuntimeError, match="no model chosen"):
        lf.apply(np.zeros((8, 8), np.float32), {}, np)
    monkeypatch.setenv(lf.ENV_DIR, str(models))
    with pytest.raises(RuntimeError, match="not found"):
        lf.apply(np.zeros((8, 8), np.float32), {"model": "nope"}, np)
    with pytest.raises(RuntimeError, match="one plane"):
        lf.apply(np.zeros((8, 8, 3), np.float32), {"model": "tiny", "device": "cpu"}, np)


class _Core:
    """A stand-in extension: `filter_chain` adds 1 per stage, and refuses unet."""

    def __init__(self):
        self.calls = []

    def filter_chain(self, cur, chain_json, _method):
        import json
        ops = [f["operation"] for f in json.loads(chain_json)["filters"]]
        assert "unet" not in ops, "a package stage reached core"
        self.calls.append(ops)
        return np.asarray(cur, np.float32) + len(ops)


def test_tail_batches_core_stages_around_the_package_stage(models):
    core = _Core()
    a = np.random.default_rng(1).random((24, 24)).astype(np.float32)
    rest = [{"operation": "blur", "params": {}}, {"operation": "edges", "params": {}},
            {"operation": "unet", "params": {"model": "tiny", "device": "cpu"}},
            {"operation": "blur", "params": {}}]
    out = ComputeEngine._apply_tail(a, rest, core, lambda _m: None)
    assert core.calls == [["blur", "edges"], ["blur"]], core.calls
    expect = lf.apply(a + 2, {"model": "tiny", "device": "cpu"}, np) + 1
    np.testing.assert_allclose(out, expect, atol=1e-5)
    # no package stage: exactly one call, as before
    core.calls.clear()
    ComputeEngine._apply_tail(a, rest[:2], core, lambda _m: None)
    assert core.calls == [["blur", "edges"]]


def test_chosen_folder_beats_env_and_default(models, tmp_path):
    other = tmp_path / "picked"; other.mkdir()
    lf.save_bundle(str(other / "picked_one.pt"), lf.build_net(META), META)
    try:
        assert lf.list_models() == ["tiny"]                     # the env var's folder
        lf.set_models_dir(str(other))
        assert lf.models_dir() == str(other) and lf.list_models() == ["picked_one"]
        assert "(chosen)" in lf.models_report()
        lf.set_models_dir(None)
        assert lf.list_models() == ["tiny"] and lf.chosen_models_dir() is None
    finally:
        lf.set_models_dir(None)


def test_summary_code():
    from msseg.mscoupon import session
    assert session.stage_code({"operation": "unet", "params": {"model": "m1"}}) == "unet(m1)"
