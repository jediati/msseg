"""The chain plan: the Python mirror, and its parity with the C++ planner.

`config_io.chain_plan` exists because the GUI draws cards with no raster and, in
a headless selftest, no compiled extension. A mirror is a fifth implementation
of chain semantics unless something holds it to the original, so the parity test
below runs every case through both and compares -- skipped, not silently passed,
when the extension is absent.
"""
from __future__ import annotations

import json

import pytest

from msseg.mscoupon import config_io

try:  # the extension is optional in a headless checkout
    from msseg.mscoupon import mscoupon_py as ext
except Exception:  # pragma: no cover - exercised only without a built extension
    ext = None

BLUR = {"operation": "blur", "params": {"sigma": 1.0}}
EDGES = {"operation": "edges", "params": {"sigma": 1.0}}
NORMALIZE = {"operation": "normalize", "params": {"method": "gmm"}}
NONE = {"operation": "none", "params": {}}


def color(method="mean", **params):
    return {"operation": "color", "params": dict(method=method, **params)}


# (chain, channels, default_method) -- every case must agree with C++.
CASES = [
    ([], 1, "luminance"),
    ([], 3, "luminance"),
    ([BLUR], 1, "luminance"),
    ([BLUR], 3, "mean"),
    ([BLUR, EDGES], 3, "mean"),
    ([color("mean")], 3, "luminance"),
    ([color("mean"), BLUR], 3, "luminance"),
    ([color("pick", channel=1), BLUR], 3, "luminance"),
    ([color("mean")], 1, "luminance"),
    ([NORMALIZE, BLUR], 3, "mean"),
    ([NONE, BLUR], 3, "mean"),
]


def test_a_multi_plane_input_shows_its_synthesized_conversion():
    plan = config_io.chain_plan([BLUR], 3, "mean")
    assert [s["operation"] for s in plan["stages"]] == ["color", "blur"]
    head = plan["stages"][0]
    assert head["synthesized"] and head["index"] == -1
    assert head["params"] == {"method": "mean"}
    assert (head["in"], head["out"]) == (3, 1)
    # and the config stage keeps its own index, so a card can find itself
    assert plan["stages"][1]["index"] == 0 and plan["stages"][1]["in"] == 1
    assert plan["in"] == 3 and plan["out"] == 1 and plan["error"] is None


def test_one_plane_plans_no_conversion():
    plan = config_io.chain_plan([BLUR], 1, "luminance")
    assert [s["operation"] for s in plan["stages"]] == ["blur"]
    assert plan["color_stage"] == -1 and plan["out"] == 1


def test_an_explicit_conversion_is_not_synthesized():
    plan = config_io.chain_plan([color("mean"), BLUR], 3, "luminance")
    assert [s["synthesized"] for s in plan["stages"]] == [False, False]
    assert plan["stages"][0]["index"] == 0


def test_color_after_the_head_is_reported_not_raised():
    plan = config_io.chain_plan([BLUR, color("mean")], 3, "luminance")
    assert plan["error"] and "must be the first stage" in plan["error"]


def test_operations_at_index_carry_the_frozen_alias_rule():
    assert "color" in config_io.filter_operations_at(0)
    assert "color" not in config_io.filter_operations_at(1)
    # everything else is offered at every index
    assert set(config_io.filter_operations_at(0)) - {"color"} == set(
        config_io.filter_operations_at(3))


@pytest.mark.skipif(ext is None, reason="needs the compiled mscoupon extension")
@pytest.mark.parametrize("chain,channels,method", CASES)
def test_python_mirror_matches_the_extension(chain, channels, method):
    mine = config_io.chain_plan(chain, channels, method)
    theirs = ext.chain_plan(json.dumps(chain), channels, method)

    assert mine["in"] == theirs["in"]
    assert mine["out"] == theirs["out"]
    assert mine["color_stage"] == theirs["color_stage"]
    assert len(mine["stages"]) == len(theirs["stages"])
    for a, b in zip(mine["stages"], theirs["stages"]):
        assert a["operation"] == b["operation"]
        assert (a["in"], a["out"]) == (b["in"], b["out"])
        assert a["index"] == b["index"]
        assert a["synthesized"] == b["synthesized"]
        assert a["params"] == b["params"]


@pytest.mark.skipif(ext is None, reason="needs the compiled mscoupon extension")
def test_the_mirror_reports_what_the_extension_raises():
    """Same refusals, different shapes: C++ throws, the mirror records, because a
    chain is invalid between keystrokes in the GUI and a card rebuild must not
    depend on the chain being well-formed."""
    chain = [BLUR, color("mean")]
    assert config_io.chain_plan(chain, 3, "luminance")["error"]
    with pytest.raises(Exception):
        ext.chain_plan(json.dumps(chain), 3, "luminance")
