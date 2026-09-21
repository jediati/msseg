"""Named measurement sources: a filter chain over the input planes whose output
becomes measurable.

This is the Stage 3 shape of docs/design_filter_types.md, and deliberately not
the one the note first proposed. Making `base` itself a stack would have changed
what `base` MEANS -- it is the raster `relevance` is measured on, the one the
per-slice CSV reports, and the one a pixel filter names -- for a feature that
needs none of that. A source is measured and nothing else.
"""
from __future__ import annotations

import json

import pytest

from msseg.mscoupon import config_io

try:
    from msseg.mscoupon import mscoupon_py as ext
except Exception:  # pragma: no cover - only without a built extension
    ext = None

HE = {"operation": "stain_deconvolution", "params": {"preset": "he"}}


def params(channels, sources=None):
    return json.dumps({
        "input": {"color": {"channels": 3}},
        "statistics": {"sources": sources or {"he": [HE]},
                       "channels": channels,
                       "reductions": ["mean"], "extremum": False},
    })


def test_sources_round_trip_through_the_config():
    """A source that does not survive export is a source lost -- the same hole
    `optical_density`'s `output` fell into."""
    block = config_io.statistics_to_json(
        [{"kind": "base"}, {"kind": "he"},
         {"kind": "blur", "sigmas": [1.5], "source": "he"}],
        ["mean"], False, 0, True, None, sources={"he": [dict(HE)]})
    assert block["channels"] == ["base", "he", {"kind": "blur", "sigmas": [1.5], "source": "he"}]
    assert list(block["sources"]) == ["he"]
    assert block["sources"]["he"][0]["operation"] == "stain_deconvolution"

    back = config_io.statistics_from_json(block)
    assert [c["kind"] for c in back["channels"]] == ["base", "he", "blur"]
    assert back["channels"][2]["source"] == "he"
    assert list(back["sources"]) == ["he"]


def test_a_spec_without_sources_is_the_document_it_always_was():
    block = config_io.statistics_to_json([{"kind": "base"}], ["mean"], True)
    assert "sources" not in block
    assert config_io.statistics_from_json(block)["sources"] == {}


def test_an_unknown_source_is_noted_not_raised():
    notes: list = []
    block = {"channels": [{"kind": "blur", "sigmas": [1.0], "source": "nope"}]}
    out = config_io.statistics_from_json(block, notes)
    assert out["channels"][0].get("source") in (None, "base")
    assert notes and "nope" in notes[0]


@pytest.mark.skipif(ext is None, reason="needs the compiled mscoupon extension")
def test_a_source_resolves_to_named_channels():
    names = [c["name"] for c in ext.stat_channels(params(["base", "he"]))]
    assert names == ["base", "he_c0", "he_c1", "he_c2"]
    # the plane count is DERIVED from the chain -- nothing declared three
    assert [c["source"] for c in ext.stat_channels(params(["base", "he"]))][1:] == ["he"] * 3


@pytest.mark.skipif(ext is None, reason="needs the compiled mscoupon extension")
def test_a_derived_kind_over_a_source_is_prefixed():
    p = params(["base", {"kind": "blur", "sigmas": [1.5], "source": "he"}])
    names = [c["name"] for c in ext.stat_channels(p)]
    assert names == ["base", "he_blur_c0_s1.5", "he_blur_c1_s1.5", "he_blur_c2_s1.5"]
    # so two sources measured the same way cannot collide
    two = params(["base", {"kind": "blur", "sigmas": [1.5], "source": "he"},
                  {"kind": "blur", "sigmas": [1.5], "source": "hd"}],
                 sources={"he": [HE], "hd": [{"operation": "stain_deconvolution",
                                              "params": {"preset": "hdab"}}]})
    assert len(set(c["name"] for c in ext.stat_channels(two))) == 7


@pytest.mark.skipif(ext is None, reason="needs the compiled mscoupon extension")
def test_the_columns_are_the_feature_schema():
    fields = list(ext.feature_fields(params(["base", "he"])))
    assert "mean_he_c0" in fields and "mean_he_c1" in fields
    # and the GUI's query validation accepts them, since it asks the extension
    assert "mean_base" in fields


@pytest.mark.skipif(ext is None, reason="needs the compiled mscoupon extension")
def test_a_source_slot_carries_its_chains_pixels():
    import numpy as np
    rng = np.random.default_rng(0)
    rgb = (40 + 180 * rng.random((3, 18, 14))).astype(np.float32)
    base = rgb.mean(axis=0)
    names, imgs = ext.stat_channel_images(base, base, params(["base", "he"]), rgb)
    eosin = ext.filter_chain(rgb, json.dumps({"filters": [
        HE, {"operation": "adapt", "params": {"mode": "select", "channels": [1]}}]}), "luminance")
    assert np.allclose(imgs[list(names).index("he_c1")], eosin)


@pytest.mark.skipif(ext is None, reason="needs the compiled mscoupon extension")
def test_an_undeclared_source_is_refused_by_the_extension():
    with pytest.raises(Exception, match="statistics.sources"):
        ext.stat_channels(params(["base", {"kind": "blur", "sigmas": [1.0], "source": "nope"}]))


def test_a_profile_keeps_its_sources_through_a_load():
    """profile_from_json reads the statistics block and writes it back. Without
    `sources` in that hand-off a profile declaring one would lose it on load,
    and its channels would then stop resolving -- the same silent-drop shape as
    `optical_density`'s `output` and, before that, mspath's `color_method`."""
    from msseg.mscoupon import session

    doc = {
        "name": "he",
        "filters": [],
        "base_filters": [],
        "msc": {},
        "statistics": {
            "sources": {"he": [HE]},
            "channels": ["base", "he"],
            "reductions": ["mean"],
            "extremum": True,
        },
    }
    prof = session.profile_from_json(doc)
    assert list(prof["statistics"].get("sources") or {}) == ["he"], prof["statistics"]
    assert "he" in prof["statistics"]["channels"]

    # and it survives a second pass, so repeated saves do not erode it
    again = session.profile_from_json(prof)
    assert list(again["statistics"].get("sources") or {}) == ["he"]


@pytest.mark.skipif(ext is None, reason="needs the compiled mscoupon extension")
def test_the_profiles_params_still_resolve_after_a_load():
    """The point of carrying them: the params a load produces must still name
    channels the extension can resolve."""
    from msseg.mscoupon import session

    doc = {"name": "he", "filters": [], "base_filters": [], "msc": {},
           "input": {"color": {"channels": 3}},
           "statistics": {"sources": {"he": [HE]}, "channels": ["base", "he"],
                          "reductions": ["mean"], "extremum": False}}
    prof = session.profile_from_json(doc)
    pj = session.profile_params_json(prof, 1, 3)
    names = [c["name"] for c in ext.stat_channels(pj)]
    assert names == ["base", "he_c0", "he_c1", "he_c2"], names
