"""Colour input on the Python side: the `color` filter card's JSON round trip,
the planar loader (extension reader and the Pillow fallback), the compact
plane storage, and the workflow summary code.

Headless: nothing here needs the extension except the reader tests, which
skip without it.
"""
import json
import os

import numpy as np
import pytest

from msseg.mscoupon import common, config_io, session


def test_color_card_round_trips_lists_and_keywords():
    cards = [{"operation": "color",
              "params": {"method": "optical_density", "i0": "255", "stain": "1, 0.5, 0",
                         "eps": 0.001, "sigma": 9.0}},     # sigma: a stale row
             {"operation": "blur", "params": {"sigma": 1.5}}]
    doc = config_io.filters_to_json(cards)
    assert doc[0] == {"operation": "color",
                      "params": {"method": "optical_density", "i0": 255.0,
                                 "stain": [1.0, 0.5, 0.0], "eps": 0.001}}
    back = config_io.filters_from_json(doc)
    assert back[0]["params"] == {"method": "optical_density", "i0": "255",
                                 "stain": "1, 0.5, 0", "eps": 0.001}
    # A keyword i0 stays a string; a blank stain is dropped rather than exported.
    kw = config_io.filters_to_json([{"operation": "color",
                                     "params": {"method": "optical_density", "i0": "max",
                                                "stain": ""}}])
    assert kw == [{"operation": "color", "params": {"method": "optical_density", "i0": "max"}}]
    # Every method renders its own rows; the method row is always first.
    for method in config_io.COLOR_METHODS:
        rows = config_io.filter_param_schema("color", {"method": method})
        assert rows[0][0] == "method"
        assert [r[0] for r in rows[1:]] == [r[0] for r in config_io.COLOR_METHOD_PARAMS[method]]
    # The whole config survives json.
    json.dumps(doc)


def test_color_card_from_json_fills_defaults_and_notes_unknowns():
    notes = []
    card = config_io.filter_params_from_json("color", {"method": "dizenzo", "sigma": 2},
                                             notes)
    assert card["params"] == {"method": "dizenzo", "sigma": 2.0, "eigen": "largest"}
    card = config_io.filter_params_from_json("color", {"method": "nope"}, notes)
    assert card["params"]["method"] == "luminance"
    assert any("nope" in n for n in notes)


def test_workflow_summary_names_the_color_stage():
    profile = session.default_profile("p")
    profile["filters"] = [{"operation": "color", "params": {"method": "dizenzo", "sigma": 1.5}},
                          {"operation": "blur", "params": {"sigma": 1.5}}]
    text = session.profile_summary(profile)
    assert text.splitlines()[0].startswith("topo field: base→col(dizenzo)→b(1.5)→msc(")


def test_compact_planes_keeps_integers_small():
    arr = np.array([[[0, 255], [3, 4]]], dtype=np.float32)
    assert common.compact_planes(arr).dtype == np.uint8
    arr16 = np.array([[[0, 300]]], dtype=np.float32)
    assert common.compact_planes(arr16).dtype == np.uint16
    frac = np.array([[[0.5, 1.0]]], dtype=np.float32)
    assert common.compact_planes(frac) is frac
    assert common.compact_planes(None) is None


def _write_rgb(path, planar=False):
    tifffile = pytest.importorskip("tifffile")
    h, w = 3, 5
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        rgb[..., c] = (np.arange(h * w).reshape(h, w) + 50 * c).astype(np.uint8)
    # tifffile takes samples-first data for a separate-planes file.
    data = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))) if planar else rgb
    tifffile.imwrite(path, data, photometric="rgb",
                     planarconfig="separate" if planar else "contig")
    return rgb


def test_load_slice_pillow_fallback_is_planar(tmp_path):
    path = os.path.join(tmp_path, "rgb.tiff")
    rgb = _write_rgb(path)
    arr = common.load_slice(path, engine=False)      # engine=False: no extension
    assert arr.shape == (3, 3, 5) and arr.dtype == np.float32
    assert np.array_equal(arr, np.transpose(rgb, (2, 0, 1)).astype(np.float32))
    # Gray stays 2D.
    gray_path = os.path.join(tmp_path, "gray.tiff")
    pytest.importorskip("tifffile").imwrite(gray_path, rgb[..., 0])
    assert common.load_slice(gray_path, engine=False).shape == (3, 5)
    # RGBA drops alpha by default, keeps it on request.
    rgba_path = os.path.join(tmp_path, "rgba.tiff")
    rgba = np.concatenate([rgb, np.full((3, 5, 1), 7, np.uint8)], axis=2)
    pytest.importorskip("tifffile").imwrite(rgba_path, rgba, photometric="rgb")
    assert common.load_slice(rgba_path, engine=False).shape == (3, 3, 5)
    assert common.load_slice(rgba_path, alpha="keep", engine=False).shape == (4, 3, 5)


def test_load_slice_extension_reader_matches_pillow(tmp_path):
    engine = pytest.importorskip("msseg.mscoupon")
    if not hasattr(engine, "read_tiff_planes"):
        pytest.skip("extension predates read_tiff_planes")
    for planar in (False, True):
        path = os.path.join(tmp_path, f"rgb_{int(planar)}.tiff")
        rgb = _write_rgb(path, planar=planar)
        via_ext = common.load_slice(path, engine=engine)
        via_pil = common.load_slice(path, engine=False)
        assert via_ext.shape == (3, 3, 5)
        assert np.array_equal(via_ext, via_pil), "TinyTIFF and Pillow agree on the planes"
        assert np.array_equal(via_ext[1], rgb[..., 1].astype(np.float32))


def test_filter_slice_reduces_planes_with_the_default_and_an_explicit_stage():
    engine = pytest.importorskip("msseg.mscoupon")
    if not hasattr(engine, "read_tiff_planes"):
        pytest.skip("extension predates the colour bindings")
    planes = np.zeros((3, 4, 6), dtype=np.float32)
    planes[0] = 10.0; planes[1] = 20.0; planes[2] = 30.0
    lum = engine.filter_slice(planes, json.dumps({"filter": {"operation": "color",
                                                              "params": {"method": "luminance"}}}))
    assert lum.shape == (4, 6)
    assert np.allclose(lum, 0.2126 * 10 + 0.7152 * 20 + 0.0722 * 30)
    # A chain with no colour stage gets the default method.
    mean = engine.filter_chain(planes, json.dumps({"filters": []}), "mean")
    assert np.allclose(mean, 20.0)
    picked = engine.filter_chain(planes, json.dumps(
        {"filters": [{"operation": "color", "params": {"method": "pick", "channel": 2}}],
         "input": {"color": {"default_method": "mean"}}}))
    assert np.allclose(picked, 30.0)
    with pytest.raises(RuntimeError):
        engine.filter_chain(planes, json.dumps(
            {"filters": [{"operation": "blur", "params": {"sigma": 1.0}},
                         {"operation": "color", "params": {"method": "mean"}}]}))


def test_statistics_json_carries_colour_sources():
    cards = [{"kind": "base"}, {"kind": "color"},
             {"kind": "blur", "sigmas": [1.5], "source": "color"},
             {"kind": "blur", "sigmas": [0.7]},
             {"kind": "dizenzo", "sigmas": [1.0]}]
    doc = config_io.statistics_to_json(cards, ["mean"], True, 0)
    assert doc["channels"] == ["base", "color",
                               {"kind": "blur", "sigmas": [1.5], "source": "color"},
                               {"kind": "blur", "sigmas": [0.7]},
                               {"kind": "dizenzo", "sigmas": [1.0], "source": "color"}]
    back = config_io.statistics_from_json(doc)
    assert back["channels"] == cards[:2] + [
        {"kind": "blur", "sigmas": [1.5], "source": "color"},
        {"kind": "blur", "sigmas": [0.7]},
        {"kind": "dizenzo", "sigmas": [1.0], "source": "color"}]
    notes = []
    odd = config_io.statistics_from_json(
        {"channels": [{"kind": "blur", "sigmas": [1.0], "source": "sideways"}]}, notes)
    assert "source" not in odd["channels"][0] and notes
    # The width counts planes: base + 3 raw + 3 colour blurs + 1 base blur + 2 dizenzo.
    assert session.stats_width(doc, color_channels=3) == "10ch×1"
    assert session.stats_width(doc, color_channels=0) == "4ch×1"


def test_profile_input_block_rides_only_when_non_default():
    p = session.default_profile("p")
    assert p["input"]["color"] == {"alpha": "drop", "default_method": "luminance", "channels": 0}
    assert "input" not in json.loads(session.profile_params_json(p))
    doc = json.loads(session.profile_params_json(p, color_channels=3))
    assert doc["input"] == {"color": {"channels": 3}}
    p["input"]["color"].update({"alpha": "keep", "default_method": "mean", "channels": 4})
    assert session.profile_from_json(p) == p
    doc = json.loads(session.profile_params_json(p))
    assert doc["input"] == {"color": {"alpha": "keep", "default_method": "mean", "channels": 4}}
    assert session.profile_params_json(p, color_channels=3).count('"channels": 3') == 1
    assert session.profile_summary(p).splitlines()[1] == "stats: base→1ch×4"


def test_extension_measures_colour_channels():
    engine = pytest.importorskip("msseg.mscoupon")
    if not hasattr(engine, "read_tiff_planes"):
        pytest.skip("extension predates the colour bindings")
    rng = np.random.default_rng(3)
    base = rng.random((24, 30), dtype=np.float32)
    planes = np.stack([base, 2 * base + 1, 0.5 * base]).astype(np.float32)
    params = json.dumps({"input": {"color": {"channels": 3}},
                         "statistics": {"channels": ["base", "color",
                                                     {"kind": "blur", "sigmas": [1.5], "source": "color"},
                                                     {"kind": "chgradmag", "sigmas": [1.0]}],
                                        "reductions": ["mean"]}})
    names = [c["name"] for c in engine.stat_channels(params)]
    assert names == ["base", "color_c0", "color_c1", "color_c2", "blur_c0_s1.5", "blur_c1_s1.5",
                     "blur_c2_s1.5", "chgradmag_s1"]
    assert [c["source"] for c in engine.stat_channels(params)][1:] == ["color"] * 7
    got, imgs = engine.stat_channel_images(base, base, params, planes)
    assert list(got) == names and imgs.shape == (8, 24, 30)
    assert np.array_equal(imgs[2], planes[1])
    with pytest.raises(RuntimeError):
        engine.stat_channel_images(base, base, params)      # planes missing
    pipe = engine.prime_slice(base, base, params, planes)
    fields, values = pipe.feature_table()
    fields = list(fields)
    c0, c1 = fields.index("mean_color_c0"), fields.index("mean_color_c1")
    assert np.allclose(values[:, c1], 2 * values[:, c0] + 1, atol=1e-3)
