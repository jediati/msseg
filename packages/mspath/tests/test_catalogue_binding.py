"""The slide catalogue's binding rule: gestures are keyed by the slide, an
item sees those meeting its rect, and an item key from an older store rebases
to the slide with its level and scale. Headless: a fake app is enough."""
from msseg.mspath import items as I
from msseg.mspath.adapters import SlideCatalogue


class _Src:
    def level_scale(self, level):
        return float(2 ** level) * 1.01          # a real pyramid's downsample is not exact


class _Engine:
    def __init__(self):
        self.sources = {"wsi/a.svs": _Src()}

    def source(self, slide):
        return self.sources.get(slide)


class _App:
    """Two slides; the first has one ROI."""
    def __init__(self):
        self.subsequences = [{"folder": "C:/data/wsi", "files": ["C:/data/wsi/a.svs"],
                              "rois": [{"level": 0, "x": 100, "y": 200, "w": 64, "h": 32}]},
                             {"folder": "C:/data/wsi", "files": ["C:/data/wsi/b.svs"], "rois": []}]
        self.engine = _Engine()
        self.flat_slices = [(0, 0), (0, 1), (1, 0)]

    def _slide_of(self, si):
        s = self.subsequences[si]
        return I.slide_id(s["folder"], s["files"][0]), s["files"][0]

    def _rois_of(self, si):
        return self.subsequences[si]["rois"]

    def _item_at(self, si, li):
        sid, _ = self._slide_of(si)
        if li <= 0:
            return I.overview(sid, 4)
        r = self._rois_of(si)[li - 1]
        return I.roi(sid, r["level"], r["x"], r["y"], r["w"], r["h"])

    def _enumerate_items(self):
        for si in range(len(self.subsequences)):
            for li in range(1 + len(self._rois_of(si))):
                yield si, li


def test_binding_and_rebase():
    cat = SlideCatalogue(_App())
    cat.refresh()
    ovw, roi = cat.key_of(0, 0), cat.key_of(0, 1)
    assert cat.binding_of(ovw) == ("wsi/a.svs", None)
    assert cat.binding_of(roi) == ("wsi/a.svs", (100, 200, 64, 32))
    assert cat.binding_of("wsi/a.svs") == ("wsi/a.svs", None), "a slide key binds to itself"
    assert cat.rebase(ovw) == ("wsi/a.svs", 4, 16.16)
    assert cat.rebase(roi) == ("wsi/a.svs", 0, 1.01)
    assert cat.rebase("wsi/a.svs") is None
    # An unknown slide still rebases, with the power-of-two fallback.
    assert cat.rebase("wsi/zzz.svs@3#0,0,8,8") == ("wsi/zzz.svs", 3, 8.0)


def test_index_of_resolves_slide_keys_to_the_overview_row():
    cat = SlideCatalogue(_App())
    cat.refresh()
    assert cat.index_of("wsi/a.svs") == (0, 0)
    assert cat.index_of("wsi/b.svs") == (1, 0)
    assert cat.index_of("wsi/c.svs") is None
    assert cat.index_of(cat.key_of(0, 1)) == (0, 1), "item keys still resolve exactly"
    assert cat.group_of("wsi/a.svs") == "wsi/a.svs"
