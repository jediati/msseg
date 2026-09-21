"""``mspath-embed``: the pure tiling / rule / tissue helpers, and one real
harvest on the selftest's synthetic slide when the compiled pipeline is
importable (skipped otherwise).
"""
import json
import os

import numpy as np
import pytest

from msseg.labeler.table import FeatureTable
from msseg.mspath import embed as E


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_tile_grid_covers_the_level_or_takes_it_whole():
    assert E.tile_grid((300, 400), 512) == [None]
    tiles = E.tile_grid((300, 700), 256)
    assert tiles == [(0, 0, 256, 256), (256, 0, 256, 256), (512, 0, 188, 256),
                     (0, 256, 256, 44), (256, 256, 256, 44), (512, 256, 188, 44)]
    assert sum(w * h for _x, _y, w, h in tiles) == 300 * 700


def test_rules_parse_and_apply():
    assert E.parse_rule(None) is None
    assert E.parse_rule("mean_base>230") == ("mean_base", ">", 230.0)
    assert E.parse_rule(" ext_base <= -1.5e2 ") == ("ext_base", "<=", -150.0)
    with pytest.raises(ValueError):
        E.parse_rule("mean_base ~ 3")
    t = FeatureTable(["feature_id", "mean_base"], np.array([[0, 100.0], [1, 240.0], [2, 230.0]]))
    assert E.apply_rule(("mean_base", ">", 230.0), t).tolist() == [False, True, False]
    assert E.apply_rule(("mean_base", ">=", 230.0), t).tolist() == [False, True, True]
    with pytest.raises(ValueError):
        E.apply_rule(("nope", ">", 1.0), t)


def test_scaled_profile_scales_percent_or_absolute():
    p = {"msc": {"persistence_percent": 3.0}}
    assert E.scaled_profile(p, 0.5)["msc"]["persistence_percent"] == 1.5
    assert p["msc"]["persistence_percent"] == 3.0
    q = {"msc": {"persistence_percent": 3.0, "persistence_absolute": 8.0}}
    s = E.scaled_profile(q, 2.0)["msc"]
    assert s["persistence_absolute"] == 16.0 and s["persistence_percent"] == 3.0


def test_tissue_fractions_read_the_map_under_each_tile():
    tissue = np.zeros((10, 20), bool)
    tissue[:, :10] = True                        # the left half is tissue
    # the level is 4x finer than the map: level px = map px * 4
    tiles = [(0, 0, 40, 40), (40, 0, 40, 40), (0, 0, 80, 40)]
    f = E.tissue_fractions(tissue, map_scale=4.0, tiles=tiles, level_scale=1.0, level_shape=(40, 80))
    assert f == [1.0, 0.0, 0.5]
    assert E.tissue_fractions(tissue, 4.0, [None], 1.0, (40, 80)) == [0.5]


def test_drop_rows_reindexes_the_arcs():
    keep = np.array([True, False, True, True])
    ia, ib, s = E.drop_rows(keep, [0, 1, 2, 0], [1, 2, 3, 3], [1.0, 2.0, 3.0, 4.0])
    assert ia.tolist() == [1, 0] and ib.tolist() == [2, 2] and s.tolist() == [3.0, 4.0]
    ia, ib, s = E.drop_rows(keep, [], [], None)
    assert len(ia) == 0 and s is None


def test_find_slides_dedupes_and_keeps_order(tmp_path):
    for n in ("b.tiff", "a.svs", "c.txt"):
        (tmp_path / n).write_bytes(b"x")
    got = E.find_slides(str(tmp_path), [str(tmp_path / "b.tiff")], None)
    assert [os.path.basename(p) for p in got] == ["b.tiff", "a.svs"]
    assert [os.path.basename(p) for p in E.find_slides(str(tmp_path), [], "*.svs")] == ["a.svs"]


# --------------------------------------------------------------------------- #
# a real harvest on the synthetic slide
# --------------------------------------------------------------------------- #
PROFILE = {
    "name": "test", "base_filters": [],
    "filters": [{"operation": "color", "params": {"method": "min"}},
                {"operation": "blur", "params": {"sigma": 1.0}}],
    "msc": {"manifold": "ascending", "persistence_percent": 5.0, "accurate": False,
            "simplification": "merge_forest"},
    "slide": {"halo": 8, "overview_level": 2},
    "statistics": {"channels": ["base", {"kind": "blur", "sigmas": [2.0], "source": "color"}],
                   "reductions": ["mean", "std"], "extremum": True},
}


@pytest.fixture(scope="module")
def slide(tmp_path_factory):
    pytest.importorskip("tifffile")
    from msseg.mspath.selftest import _synthetic_slide
    folder = tmp_path_factory.mktemp("slides")
    return _synthetic_slide(str(folder)), str(folder)


def test_harvest_then_train_on_the_synthetic_slide(slide, tmp_path, monkeypatch):
    ext = pytest.importorskip("msseg.mscoupon.mscoupon_py")
    from msseg.labeler.embedding_train.shards import HarvestReader
    path, folder = slide
    prof = tmp_path / "p.profile.json"
    prof.write_text(json.dumps(PROFILE))
    out = tmp_path / "harvest"
    # OpenSlide opens the synthetic file as a single level, so the harvest
    # runs at level 0: the 1024x768 slide is four 512-px tiles
    argv = ["harvest", "--tiff-folder", folder, "--process-profile", str(prof), "--out", str(out),
            "--level", "0", "--roi", "512", "--factors", "1,2", "--min-tissue", "0", "--quiet"]
    assert E.main(argv) == 0
    doc = json.loads((out / "harvest.json").read_text())
    assert doc["level"] == 0 and doc["scope"] == "L0" and doc["factors"] == [1.0, 2.0]
    assert len(doc["shards"]) == 8 and doc["pins"].get("0")
    assert doc["names"] == list(ext.feature_fields(json.dumps(doc["params"])))
    assert doc["schema"] and doc["slides"][0]["tiles"] == 4
    h = HarvestReader(str(out)).load()
    assert h.n_rows > 8 and len(h.shards) == 8
    # the positional columns are in slide coordinates: the second tile's x's
    # start at 512 (level 0 IS slide coordinates here)
    by_key = {}
    for k, m in enumerate(h.shards):
        by_key.setdefault(m["key"], []).append(k)
    key = next(k for k in by_key if "#512,0," in k)
    rows = h.rows[h.row_shard == by_key[key][0]]
    assert rows[:, h.names.index("min_x")].min() >= 512 - 2 * 8    # the halo's straddlers
    # a persistence twice as high leaves fewer regions
    n1 = sum(m["n_rows"] for m in h.shards if m["factor"] == 1.0)
    n2 = sum(m["n_rows"] for m in h.shards if m["factor"] == 2.0)
    assert n2 <= n1
    assert h.n_arcs > 0 and (h.row_shard[h.ia] == h.row_shard[h.ib]).all()

    # a resume primes nothing
    from msseg.mspath import engine as ENG
    calls = []
    orig = ENG.SlideEngine.prime_item

    def counting(self, *a, **kw):
        calls.append(1)
        return orig(self, *a, **kw)
    monkeypatch.setattr(ENG.SlideEngine, "prime_item", counting)
    assert E.main(argv) == 0
    assert calls == []
    # and a --blank rule drops rows into a fresh harvest
    out2 = tmp_path / "harvest2"
    assert E.main(argv[:argv.index("--out") + 1] + [str(out2)] + argv[argv.index("--out") + 2:]
                  + ["--blank", "mean_base>1e9"]) == 0
    assert len(json.loads((out2 / "harvest.json").read_text())["shards"]) == 8

    # train the PCA baseline on it through the same CLI
    enc = tmp_path / "enc.msenc"
    assert E.main(["train", str(out), "--out", str(enc), "--arch", "pca", "--dim", "3",
                   "--quiet"]) == 0
    from msseg.labeler.embedding import EncoderBundle
    b = EncoderBundle.load(str(enc))
    assert b.scope == "L0" and b.dim == 3
    # a level the reader does not have is refused, never clamped
    rc = E.main(argv[:argv.index("--out") + 1] + [str(tmp_path / "h3")]
                + argv[argv.index("--out") + 2:argv.index("--level") + 1] + ["5"]
                + argv[argv.index("--level") + 2:])
    assert rc != 0
    assert b.embed(h.rows, h.names).shape == (h.n_rows, 3)
