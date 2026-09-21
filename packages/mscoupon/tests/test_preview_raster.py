"""`engine.preview_raster`: the Tk-free half of a preview channel.

The live preview runs this on a worker thread, and the synchronous
`_preview_channel` runs the same function inline, so what matters is that the
rasters are the pipeline's own -- the chain applied stage by stage, and a
derived channel measured by `stat_channel_images` on the base raster. These
compare it against those calls made directly.

Needs the extension (every branch reaches it); skipped without one.
"""
import json

import numpy as np
import pytest

from msseg.mscoupon.engine import (ComputeEngine, preview_job, preview_raster,
                                   _spec_reads_filtered)


def _slice(h=24, w=32, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = (np.sin(x / 5.0) * np.cos(y / 7.0)).astype(np.float32)
    return np.ascontiguousarray(base + 0.1 * rng.standard_normal((h, w)), np.float32)


def _ext():
    engine = pytest.importorskip("msseg.mscoupon")
    if not hasattr(engine, "filter_slice"):
        pytest.skip("extension not built")
    return engine


def _chain(engine, arr, chain, default="luminance"):
    """The chain applied the way a run applies it."""
    cur, rest = ComputeEngine._leading_color(arr, chain, engine, lambda _m: None, default)
    for f in rest:
        cur = engine.filter_slice(cur, json.dumps({"filter": f}))
    return np.ascontiguousarray(cur, np.float32)


def test_empty_chain_and_blur_match_the_run_calls():
    engine = _ext()
    arr = _slice()
    for chain in ([], [{"operation": "blur", "params": {"sigma": 1.5}}],
                  [{"operation": "blur", "params": {"sigma": 0.7}},
                   {"operation": "edges", "params": {"sigma": 1.5}}]):
        out = preview_raster(engine, arr, preview_job("filtered", [], chain, "luminance"))
        assert out["base"] is None and not out["channels"]
        np.testing.assert_allclose(out["filtered"], _chain(engine, arr, chain), rtol=1e-6)
        # ...and the base chain through its own helper.
        out = preview_raster(engine, arr, preview_job("base", chain, [], "luminance"))
        want, _ = ComputeEngine._apply_base_chain(arr, chain, engine, lambda _m: None)
        assert out["filtered"] is None
        np.testing.assert_allclose(out["base"], want, rtol=1e-6)


def test_colour_slice_reduces_through_the_leading_stage():
    engine = _ext()
    planes = np.ascontiguousarray(np.stack([_slice(seed=k) for k in range(3)]), np.float32)
    chain = [{"operation": "color", "params": {"method": "pick", "channel": 1}},
             {"operation": "blur", "params": {"sigma": 1.0}}]
    out = preview_raster(engine, planes,
                         preview_job("filtered", [], chain, "luminance", planar=True))
    np.testing.assert_allclose(out["filtered"], _chain(engine, planes, chain), rtol=1e-6)
    # No colour stage: the declared default converts instead.
    bare = [{"operation": "blur", "params": {"sigma": 1.0}}]
    out = preview_raster(engine, planes,
                         preview_job("filtered", [], bare, "mean", planar=True))
    np.testing.assert_allclose(out["filtered"], _chain(engine, planes, bare, "mean"),
                               rtol=1e-6)


def _single(kind, sigma, source="base"):
    return json.dumps({"statistics": {"channels": [{"kind": kind, "sigmas": [sigma],
                                                    "source": source}],
                                      "reductions": ["mean"]}})


def test_spec_reads_filtered_only_for_the_aggregate_channel():
    assert not _spec_reads_filtered(_single("blur", 1.5))
    assert _spec_reads_filtered(json.dumps(
        {"statistics": {"channels": ["base", "filtered"]}}))
    assert _spec_reads_filtered(json.dumps(
        {"statistics": {"channels": [{"kind": "filtered"}]}}))
    assert _spec_reads_filtered("not json"), "unreadable: assume it is read"


def test_derived_channel_matches_stat_channel_images():
    engine = _ext()
    if not hasattr(engine, "stat_channel_images"):
        pytest.skip("extension predates stat_channel_images")
    arr = _slice()
    base_chain = [{"operation": "blur", "params": {"sigma": 0.7}}]
    topo = [{"operation": "edges", "params": {"sigma": 1.0}}]
    single = _single("edges", 1.5)
    out = preview_raster(engine, arr,
                         preview_job("derived", base_chain, topo, "luminance",
                                     single=single))
    base, _ = ComputeEngine._apply_base_chain(arr, base_chain, engine, lambda _m: None)
    names, imgs = engine.stat_channel_images(base, base, single)
    assert [n for n, _r in out["channels"]] == list(names)
    for (n, got), want in zip(out["channels"], imgs):
        np.testing.assert_allclose(got, want, rtol=1e-6, err_msg=n)
    # The base raster it had to build rides back too, so one submission fills
    # both cache entries.
    np.testing.assert_allclose(out["base"], base, rtol=1e-6)
    assert topo, "the topology chain is set up but deliberately unused"


def test_hessian_returns_every_plane_of_the_kind():
    engine = _ext()
    if not hasattr(engine, "stat_channel_images"):
        pytest.skip("extension predates stat_channel_images")
    out = preview_raster(engine, _slice(),
                         preview_job("derived", [], [], "luminance",
                                     single=_single("hessian", 1.5)))
    names = [n for n, _r in out["channels"]]
    assert len(names) == 2 and all("hessian" in n for n in names), names


def test_a_derived_channel_ignores_the_topology_chain():
    """A statistics source is base, colour, or a declared source chain -- never
    "filtered" -- so the topology chain cannot move a derived channel, and the
    preview does not run it. That is what makes editing `filters` free while a
    derived channel is on screen."""
    engine = _ext()
    if not hasattr(engine, "stat_channel_images"):
        pytest.skip("extension predates stat_channel_images")
    with pytest.raises(RuntimeError, match="has source 'filtered'"):
        engine.stat_channel_images(_slice(), _slice(), _single("blur", 1.5, "filtered"))
    arr = _slice()
    topo = [{"operation": "blur", "params": {"sigma": 4.0}}]
    a = preview_raster(engine, arr, preview_job("derived", [], [], "luminance",
                                                single=_single("blur", 1.5)))
    b = preview_raster(engine, arr, preview_job("derived", [], topo, "luminance",
                                                single=_single("blur", 1.5)))
    np.testing.assert_allclose(a["channels"][0][1], b["channels"][0][1], rtol=1e-6)
    assert a["filtered"] is None and b["filtered"] is None, \
        "the topology chain is not run for a derived preview"
    # ...but a spec that DOES name the filtered channel still gets it.
    named = json.dumps({"statistics": {"channels": ["filtered",
                                                    {"kind": "blur", "sigmas": [1.5]}],
                                       "reductions": ["mean"]}})
    out = preview_raster(engine, arr, preview_job("derived", [], topo, "luminance",
                                                  single=named))
    np.testing.assert_allclose(out["filtered"], _chain(engine, arr, topo), rtol=1e-6)


def test_have_is_used_instead_of_recomputing():
    engine = _ext()
    if not hasattr(engine, "stat_channel_images"):
        pytest.skip("extension predates stat_channel_images")
    arr = _slice()
    job = preview_job("derived", [{"operation": "blur", "params": {"sigma": 3.0}}],
                      [{"operation": "edges", "params": {"sigma": 1.0}}],
                      "luminance", single=_single("blur", 1.5))
    sentinel = np.zeros_like(arr)
    job["have"] = {"base": sentinel}
    out = preview_raster(engine, arr, job)
    # Nothing was rebuilt, so nothing comes back to be cached...
    assert out["base"] is None and out["filtered"] is None
    # ...and the channel was measured on what was handed over, not on the chain.
    names, imgs = engine.stat_channel_images(sentinel, sentinel, job["single"])
    np.testing.assert_allclose(out["channels"][0][1], imgs[0], rtol=1e-6)


def test_should_stop_bails_out_before_the_chain_runs():
    """The chain is ONE core call now -- it has to be, since planning a fragment
    can insert a conversion the whole chain would not have -- so the point at
    which a superseded preview is abandoned is before it starts, not between
    stages."""
    engine = _ext()
    arr = _slice()
    chain = [{"operation": "blur", "params": {"sigma": 1.0}}] * 3
    out = preview_raster(engine, arr, preview_job("filtered", [], chain, "luminance"),
                         should_stop=lambda: True)
    assert out["filtered"] is None, "a stopped chain produces nothing to paint"


# --------------------------------------------------------------------------- #
# The mirror must compute what ONE core call computes.
#
# It did not, for a while: _leading_color cut the chain at the plane/scalar
# boundary and handed core the fragment before it. Whether a leading conversion
# is synthesized is a property of the WHOLE chain, so `[edges, adapt{reduce}]`
# had its empty prefix planned on its own, got a luminance inserted, and the
# field the GUI primed was silently a different one from the CLI's -- same
# shape, different pixels. The only legitimate cut is at a stage core cannot
# apply (`normalize`), which can only occur where the field is already scalar.
# --------------------------------------------------------------------------- #
def _mirror(engine, arr, chain, method="luminance"):
    import numpy as np
    from msseg.mscoupon.engine import ComputeEngine
    cur, rest = ComputeEngine._leading_color(arr, chain, engine, lambda _m: None, method)
    if rest:
        cur = engine.filter_chain(cur, json.dumps({"filters": rest}), method)
    return np.ascontiguousarray(cur, dtype=np.float32)


_EDGES = {"operation": "edges", "params": {"sigma": 1.0}}
_BLUR = {"operation": "blur", "params": {"sigma": 1.5}}
_REDUCE = {"operation": "adapt", "params": {"mode": "reduce", "how": "max"}}
_HE = {"operation": "stain_deconvolution", "params": {"preset": "he"}}
_ODP = {"operation": "color", "params": {"method": "optical_density", "output": "planes"}}


@pytest.mark.parametrize("chain", [
    [],
    [_BLUR],
    [_BLUR, _EDGES],
    [_EDGES, _REDUCE],                      # the one that broke: lift, then reduce
    [_HE, _EDGES, _REDUCE],
    [_HE, {"operation": "adapt", "params": {"mode": "select", "channels": [1]}}, _BLUR],
    [_ODP, {"operation": "adapt", "params": {"mode": "project", "matrix": [-2.1, 1.2, -0.6]}}],
    [{"operation": "color", "params": {"method": "mean"}}, _BLUR],
    [{"operation": "dizenzo", "params": {"sigma": 1.5}}],
])
def test_the_mirror_computes_what_one_core_call_computes(chain):
    import numpy as np
    engine = _ext()
    if not hasattr(engine, "filter_chain"):
        pytest.skip("extension predates filter_chain")
    rgb = (40 + 180 * np.random.default_rng(5).random((3, 14, 11))).astype(np.float32)
    whole = np.asarray(engine.filter_chain(rgb, json.dumps({"filters": chain}), "luminance"))
    mine = _mirror(engine, rgb, chain)
    assert whole.shape == mine.shape, (whole.shape, mine.shape)
    assert np.allclose(whole, mine), "the mirror and one core call must agree"
