"""label_export (GIMP crops) and the Draw-rect place tool, headless."""
import json
import os

import numpy as np
import pytest
from PIL import Image

from msseg.mspath import label_export
from msseg.mspath.rect_tool import RectPlaceTool


class _Src:
    """A two-level uint16, two-channel source: level 1 is a 2x downsample."""

    def __init__(self):
        yy, xx = np.mgrid[0:200, 0:300]
        self.l0 = np.stack([xx, yy * 3], -1).astype(np.uint16)
        self.reads = []

    def level_scale(self, level):
        return 2.0 ** level

    def read_region(self, level, x, y, w, h):
        self.reads.append((level, x, y, w, h))
        s = int(self.level_scale(level))
        return self.l0[::s, ::s][y:y + h, x:x + w]


def test_export_writes_four_files_and_the_sidecar(tmp_path):
    src = _Src()
    doc = label_export.export_place(src, slide_id="fold/slide one.tiff", level=1,
                                    rect=(40, 20, 100, 60), out_dir=str(tmp_path),
                                    np=np, slide_path="/x/slide one.tiff",
                                    place_uid="p_abc123")
    stem = "slide_one_L1_x40_y20_100x60"
    assert src.reads == [(1, 20, 10, 50, 30)], "slide rect -> level rect"
    for suffix in (".png", "_mask.png", "_raw.tif", ".json"):
        assert (tmp_path / (stem + suffix)).exists(), suffix
    side = json.loads((tmp_path / (stem + ".json")).read_text())
    assert side["format"] == label_export.FORMAT and side["version"] == 1
    assert side["rect_slide"] == [40, 20, 100, 60] and side["rect_level"] == [20, 10, 50, 30]
    assert side["shape"] == [30, 50] and side["channels"] == 2 and side["dtype"] == "uint16"
    assert side["place"] == "p_abc123" and side["mask"]["edge"] == "#ff0000"
    assert "mask_kept" not in side and doc["mask_kept"] is False

    view = Image.open(tmp_path / (stem + ".png"))
    mask = Image.open(tmp_path / (stem + "_mask.png"))
    assert view.size == mask.size == (50, 30)
    assert view.mode == "RGB", "two planes show as gray + magenta"
    assert mask.mode == "RGBA" and np.asarray(mask)[..., 3].max() == 0, \
        "the mask layer starts fully transparent"
    import tifffile
    raw = tifffile.imread(tmp_path / (stem + "_raw.tif"))
    assert raw.dtype == np.uint16 and raw.shape == (30, 50, 2)
    np.testing.assert_array_equal(raw, src.l0[::2, ::2][10:40, 20:70])


def test_reexport_keeps_a_drawn_mask(tmp_path):
    src = _Src()
    kw = dict(slide_id="s.tif", level=0, rect=(0, 0, 64, 32), out_dir=str(tmp_path), np=np)
    label_export.export_place(src, **kw)
    path = tmp_path / "s_L0_x0_y0_64x32_mask.png"
    drawn = np.zeros((32, 64, 4), np.uint8)
    drawn[5, :, :] = (255, 0, 0, 255)
    Image.fromarray(drawn).save(path)
    doc = label_export.export_place(src, **kw)
    assert doc["mask_kept"] is True
    np.testing.assert_array_equal(np.asarray(Image.open(path)), drawn)


def test_display_image_windows_per_channel():
    a = np.zeros((4, 4), np.float32)
    a[0, 0] = 1000.0
    a[1:, :] = np.linspace(0, 1, 12).reshape(3, 4)
    img, win = label_export.display_image(a, np, percentiles=(0, 50))
    assert img.dtype == np.uint8 and img.ndim == 2 and len(win) == 1
    assert img.max() == 255, "values above the window clip, not wrap"
    rgb, win3 = label_export.display_image(np.random.rand(5, 6, 3), np)
    assert rgb.shape == (5, 6, 3) and len(win3) == 3


# -- the rect tool --------------------------------------------------------- #
class _Canvas:
    def __init__(self):
        self.items = []

    def create_rectangle(self, *coords, **kw):
        self.items.append((coords, kw.get("tags")))

    def delete(self, tag):
        self.items = [it for it in self.items if tag not in (it[1] or ())]


class _Viewer:
    def __init__(self):
        self.canvas = _Canvas()
        self.tool = _Prev()
        self.view_x, self.view_y, self.scale = 100.0, 50.0, 4.0

    def screen_to_image(self, sx, sy):
        return int(self.view_x + sx * self.scale), int(self.view_y + sy * self.scale)


class _Prev:
    trace = "the previous tool's trace"


class _E:
    def __init__(self, x, y):
        self.x, self.y = x, y


def test_rect_tool_drag_reports_slide_rect_and_restores_the_tool():
    v = _Viewer()
    prev = v.tool
    got = []
    tool = RectPlaceTool(v, lambda *r: got.append(r))
    assert v.tool is tool and tool.armed
    assert tool.trace == prev.trace, "unknown attributes fall through to the old tool"
    assert tool.on_press(_E(30, 20)) is True
    assert tool.on_move(_E(10, 40)) is True
    assert len(v.canvas.items) == 1, "one feedback rectangle while dragging"
    tool.on_release(_E(10, 40))
    assert got == [(140, 130, 220, 210)], "normalised, in slide px"
    assert v.tool is prev and not v.canvas.items


def test_rect_tool_cancel():
    v = _Viewer()
    prev = v.tool
    off = []
    tool = RectPlaceTool(v, lambda *r: pytest.fail("no rect"), on_cancel=lambda: off.append(1))
    tool.on_press(_E(1, 1))
    tool.on_move(_E(5, 5))
    assert tool.cancel() is True and off == [1] and v.tool is prev
    assert tool.cancel() is False, "a second cancel has nothing to do"
