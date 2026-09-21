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
    # Lifting: a scalar stage handed a stack runs per plane, and a chain that
    # reduces on its own gets NO leading conversion.
    ([EDGES, adapt(mode="reduce", how="max")], 3, "luminance"),
    ([HE, EDGES, adapt(mode="reduce", how="mean")], 3, "luminance"),
    ([HE, EDGES, BLUR, adapt(mode="reduce", how="norm")], 3, "luminance"),
    ([EDGES, BLUR], 3, "luminance"),          # no reduction: still converts first
    ([NORMALIZE, BLUR], 3, "luminance"),      # cannot lift: still converts first
    # The promoted colour methods, mid-chain.
    ([HE, {"operation": "dizenzo", "params": {"sigma": 1.5}}], 3, "luminance"),
    ([HE, {"operation": "hsv", "params": {"component": "saturation"}}], 3, "luminance"),
    ([HE, {"operation": "chgradmag", "params": {"sigma": 1.0}}], 3, "luminance"),
    ([{"operation": "optical_density", "params": {"output": "planes"}},
      adapt(mode="reduce", how="max")], 3, "luminance"),
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


def test_color_is_positionally_free():
    """`color` was pinned to index 0 while it was the only plane-consuming stage
    and everything after it was a scalar. A chain carries a stack now, so blur
    LIFTS and the colour stage reduces what reaches it -- which is the only way
    to reach `luminance`, `weighted` or `pick` mid-chain. Lifting the rule was
    strictly widening: it was an error before, so no valid config moved."""
    plan = config_io.chain_plan([BLUR, color("mean")], 3, "luminance")
    assert plan["error"] is None, plan
    assert [(s["operation"], s["in"], s["out"]) for s in plan["stages"]] == [
        ("blur", 3, 3), ("color", 3, 1)]
    assert plan["stages"][0]["lifted"] and plan["out"] == 1


def test_every_operation_is_offered_at_every_index():
    """Arity, not position, is what constrains a stage now -- and the planner
    reports that, so the picker does not have to guess."""
    assert set(config_io.filter_operations_at(0)) == set(config_io.filter_operations_at(3))
    assert "color" in config_io.filter_operations_at(1)


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
    chain = [HE, {"operation": "label_components", "params": {}}]   # cannot lift
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


def test_a_scalar_stage_after_a_stack_lifts():
    """It used to be refused. Lifting discards nothing, so it is automatic --
    and recorded, so the plan and the card drawn from it can say so."""
    plan = config_io.chain_plan([HE, BLUR], 3, "luminance")
    assert plan["error"] is None, plan
    assert plan["stages"][1]["lifted"] and plan["stages"][1]["out"] == 3
    assert plan["out"] == 3, "and the chain still carries a stack, which the MSC will refuse"


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


@pytest.mark.skipif(ext is None, reason="needs the compiled mscoupon extension")
def test_a_chain_that_reduces_gets_no_leading_conversion():
    """The auto-luminance appears only when a chain would not otherwise land on
    one plane. Writing a reduction is how a chain says where that happens."""
    lifted = config_io.chain_plan([EDGES, adapt(mode="reduce", how="max")], 3, "luminance")
    assert [s["operation"] for s in lifted["stages"]] == ["edges", "adapt"]
    assert lifted["stages"][0]["lifted"] and lifted["stages"][0]["out"] == 3
    assert lifted["out"] == 1

    # ...and a chain with no reduction still gets it, as every chain written
    # before plane stages existed does.
    plain = config_io.chain_plan([EDGES, BLUR], 3, "luminance")
    assert plain["stages"][0]["synthesized"] and plain["stages"][0]["operation"] == "color"


def test_a_stage_that_cannot_lift_is_reported():
    """`normalize` measures two landmarks whose meaning is a population; per
    plane they would destroy the cross-plane comparability a projection needs.
    `label_components` yields ids. Neither lifts."""
    plan = config_io.chain_plan([HE, {"operation": "label_components", "params": {}}],
                                3, "luminance")
    assert plan["error"] and "plane by plane" in plan["error"]
    assert not plan["stages"][1]["lifted"]


@pytest.mark.skipif(ext is None, reason="needs the compiled mscoupon extension")
def test_a_promoted_method_equals_the_color_spelling():
    """`dizenzo` the operation and `color{method: dizenzo}` are the same
    computation -- core applies the promoted stage by handing it to the colour
    stage, so there is one implementation, not two."""
    import numpy as np
    rgb = (40 + 180 * np.random.default_rng(3).random((3, 12, 10))).astype(np.float32)
    args = {"sigma": 1.5, "eigen": "largest"}
    alias = ext.filter_chain(rgb, json.dumps({"filters": [
        {"operation": "color", "params": dict(method="dizenzo", **args)}]}), "luminance")
    promoted = ext.filter_chain(rgb, json.dumps({"filters": [
        {"operation": "dizenzo", "params": args}]}), "luminance")
    assert np.array_equal(alias, promoted)

    # ...and both spellings work mid-chain: the operation, and `color{method}`,
    # which is what reaches the methods the promoted set does not cover.
    for stage in ({"operation": "color", "params": {"method": "dizenzo"}},
                  {"operation": "dizenzo", "params": {"sigma": 1.5}}):
        plan = ext.chain_plan(json.dumps([HE, stage]), 3, "luminance")
        assert plan["out"] == 1 and plan["stages"][1]["in"] == 3, plan
