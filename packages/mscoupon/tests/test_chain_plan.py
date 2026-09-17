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
HE = {"operation": "stain_deconvolution", "params": {"preset": "he"}}
OD_PLANES = {"operation": "color", "params": {"method": "optical_density", "output": "planes"}}


def adapt(**params):
    return {"operation": "adapt", "params": params}


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
    # Plane stages: a chain that CARRIES a stack. These cases exist because the
    # mirror was silently wrong for every one of them -- it invented a colour
    # head and reported each stage as 1 -> 1 -- and the suite above did not
    # notice, having been written before the stages were.
    ([HE], 3, "luminance"),
    ([HE, adapt(mode="select", channels=[1])], 3, "luminance"),
    ([HE, adapt(mode="select", channels=[1, 0])], 3, "luminance"),
    ([OD_PLANES], 3, "luminance"),
    ([OD_PLANES, adapt(mode="project", matrix=[[0.1, 0.9, 0.2]])], 3, "luminance"),
    ([OD_PLANES, adapt(mode="project", preset="luminance")], 3, "luminance"),
    ([OD_PLANES, adapt(mode="select", channels=[1]), BLUR], 3, "luminance"),
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


def test_a_plane_stage_head_is_not_given_a_synthesized_conversion():
    """`adapt` and `stain_deconvolution` consume the stack themselves. Handing
    them a scalar made behind their back was the first bug the C++ tests caught
    when plane stages landed."""
    plan = config_io.chain_plan([HE, adapt(mode="select", channels=[1])], 3, "luminance")
    assert [s["operation"] for s in plan["stages"]] == ["stain_deconvolution", "adapt"]
    assert plan["stages"][0]["in"] == 3 and plan["stages"][0]["out"] == 3
    assert plan["stages"][1]["out"] == 1 and plan["out"] == 1


def test_a_scalar_stage_after_a_stack_is_reported():
    plan = config_io.chain_plan([HE, BLUR], 3, "luminance")
    assert plan["error"], plan


def test_plane_stage_cards_round_trip():
    """Without schema rows, filters_to_json drops an operation it does not know --
    so a profile carrying a stain chain would have lost it on export, silently.
    This pins the round trip and the mode-dependent rows."""
    cards = [
        {"operation": "stain_deconvolution", "params": {"preset": "he", "i0": "max", "eps": 0.001}},
        {"operation": "adapt", "params": {"mode": "select", "channels": "1"}},
    ]
    out = config_io.filters_to_json(cards)
    assert [f["operation"] for f in out] == ["stain_deconvolution", "adapt"], out
    assert out[0]["params"]["preset"] == "he"
    assert out[0]["params"]["i0"] == "max"          # a keyword stays a string
    assert out[1]["params"]["channels"] == [1.0]    # a `floats` entry becomes a list

    back = config_io.filters_from_json(out)
    assert [c["operation"] for c in back] == ["stain_deconvolution", "adapt"]

    # mode picks the extra rows, exactly as a colour card's method does
    names = [r[0] for r in config_io.filter_param_schema("adapt", {"mode": "select"})]
    assert names == ["mode", "channels"], names
    names = [r[0] for r in config_io.filter_param_schema("adapt", {"mode": "project"})]
    assert names == ["mode", "matrix", "preset"], names


@pytest.mark.skipif(ext is None, reason="needs the compiled mscoupon extension")
def test_a_flat_projection_row_is_one_row():
    """`matrix` as C numbers is ONE row -- what a single GUI entry can hold, and
    what every stain contrast is. Both sides must read it the same way."""
    chain = [OD_PLANES, adapt(mode="project", matrix=[-2.113, 1.229, -0.641])]
    mine = config_io.chain_plan(chain, 3, "luminance")
    theirs = ext.chain_plan(json.dumps(chain), 3, "luminance")
    assert mine["out"] == theirs["out"] == 1
    assert [s["out"] for s in mine["stages"]] == [s["out"] for s in theirs["stages"]] == [3, 1]


def test_optical_density_output_mode_survives_export():
    """filters_to_json keeps only the params a colour METHOD declares, so a mode
    missing from COLOR_METHOD_PARAMS is dropped on export -- silently, and the
    next stage then meets one plane instead of three. `output` was exactly that
    bug; this pins it."""
    card = {"operation": "color",
            "params": {"method": "optical_density", "output": "planes"}}
    out = config_io.filters_to_json([card])
    assert out[0]["params"].get("output") == "planes", out

    # and the plan agrees before and after the round trip
    chain_before = [card, adapt(mode="project", matrix=[-2.1, 1.2, -0.6])]
    chain_after = out + [adapt(mode="project", matrix=[-2.1, 1.2, -0.6])]
    assert config_io.chain_plan(chain_before, 3, "luminance")["out"] == 1
    assert config_io.chain_plan(chain_after, 3, "luminance")["out"] == 1
    assert "output" in [r[0] for r in config_io.filter_param_schema(
        "color", {"method": "optical_density"})]
