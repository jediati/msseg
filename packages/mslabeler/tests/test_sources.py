"""ImageSource / LabelLayer implementations and the canvas composite. The
render is compared with a straight re-implementation of the pre-framework
algorithm (crop, window, PIL BILINEAR base; nearest gather + LUT per label
layer; alpha composite; transient at its own alpha), so a regression in either
the sources or the canvas shows up as a pixel difference."""
import numpy as np
import pytest
from PIL import Image

from msseg.labeler import pyramid
from msseg.labeler.sources import ArrayImageSource, ArrayLabelLayer


def scene(rgb=False):
    rng = np.random.default_rng(0)
    h, w = 300, 400
    base = rng.random((h, w), np.float32) * 10 + 3
    if rgb:
        base = np.stack([base, base * 0.5, np.flipud(base)], 0)
    labels = np.full((h, w), -1, np.int32)
    labels[20:150, 30:200] = 0; labels[150:280, 30:200] = 3
    labels[20:280, 200:380] = 7; labels[100:120, 100:120] = -1
    lut = np.zeros((8, 4), np.uint8)
    lut[0] = (255, 0, 0, 200); lut[3] = (0, 255, 0, 120); lut[7] = (0, 0, 255, 255)
    rgba = np.zeros((h, w, 4), np.uint8); rgba[50:100, 50:350] = (255, 255, 0, 128)
    tl = np.zeros((8, 4), np.uint8); tl[3] = (255, 255, 255, 180)
    return base, labels, lut, rgba, tl


def test_array_source_is_the_array():
    base, *_ = scene()
    src = ArrayImageSource(base, path="p")
    assert (src.levels, src.channels, src.level_shape(0), src.level_scale(0), src.best_level(3.0)) == \
        (1, 1, (300, 400), 1.0, 0)
    assert src.value_range() == (float(base.min()), float(base.max()))
    assert np.array_equal(src.read_region(0, 30, 20, 50, 40), base[20:60, 30:80])
    planar = ArrayImageSource(scene(rgb=True)[0])
    assert planar.channels == 3 and planar.array.shape == (300, 400, 3)
    assert np.array_equal(planar.array[..., 2], np.flipud(base))
    two = ArrayImageSource(np.stack([base, base], 0))       # a lone extra plane repeats
    assert two.channels == 3 and np.array_equal(two.array[..., 1], base)


def test_array_label_layer_crops_by_level():
    _, labels, *_ = scene()
    layer = ArrayLabelLayer(labels, rev=5)
    assert layer.shape == (300, 400) and layer.n_ids == 8 and layer.rev == 5
    assert np.array_equal(layer.crop(0, 30, 20, 50, 40), labels[20:60, 30:80])
    assert np.array_equal(layer.full(), labels)
    # level 1: each level pixel samples the full-res pixel under its centre
    c = layer.crop(1, 10, 5, 20, 15)
    ys = np.clip(((5 + np.arange(15) + 0.5) * 2).astype(int), 0, 299)
    xs = np.clip(((10 + np.arange(20) + 0.5) * 2).astype(int), 0, 399)
    assert np.array_equal(c, labels[ys][:, xs])
    assert layer.id_at(35, 25) == 0 and layer.id_at(250, 100) == 7 and layer.id_at(-1, 0) == -1
    assert layer.crop(0, 390, 290, 20, 20).shape == (20, 20)     # clipping never raises


def reference_render(base, overlays, transient, alpha, vmin, vmax, scale, vx, vy, cw, ch):
    """The pre-framework SliceCanvas.render, verbatim in numpy."""
    arr = np.ascontiguousarray(np.transpose(base[:3], (1, 2, 0))) if base.ndim == 3 else base
    H, W = arr.shape[:2]
    left, top = max(0, int(vx)), max(0, int(vy))
    right, bottom = min(W, int(vx + cw * scale)), min(H, int(vy + ch * scale))
    out_w, out_h = max(1, int((right - left) / scale)), max(1, int((bottom - top) / scale))
    bmin, bmax = float(arr.min()), float(arr.max())
    lo = bmin + vmin * (bmax - bmin); hi = bmin + vmax * (bmax - bmin); span = (hi - lo) or 1.0
    norm = np.clip((arr[top:bottom, left:right] - lo) / span, 0, 1)
    if norm.ndim == 3:
        rgb = np.asarray(Image.fromarray((norm * 255).astype(np.uint8), "RGB")
                         .resize((out_w, out_h), Image.BILINEAR), np.uint8).astype(np.float32)
    else:
        gray = (np.asarray(Image.fromarray(norm.astype(np.float32))
                           .resize((out_w, out_h), Image.BILINEAR), np.float32) * 255).astype(np.uint8)
        rgb = np.dstack([gray, gray, gray]).astype(np.float32)
    ys = np.clip((top + (np.arange(out_h) + 0.5) * (bottom - top) / out_h).astype(np.intp), top, bottom - 1)
    xs = np.clip((left + (np.arange(out_w) + 0.5) * (right - left) / out_w).astype(np.intp), left, right - 1)
    entries = list(overlays) + ([transient] if transient is not None else [])
    for i, o in enumerate(entries):
        is_transient = transient is not None and i == len(entries) - 1
        if not o.get("visible", True):
            continue
        if "rgba" in o:
            ov = np.asarray(Image.fromarray(o["rgba"][top:bottom, left:right], mode="RGBA")
                            .resize((out_w, out_h), Image.NEAREST), np.float32)
        else:
            sub = o["labels"][ys][:, xs]
            ov = o["lut"][np.where(sub >= 0, sub, 0)].astype(np.float32); ov[sub < 0] = 0
        a = ov[:, :, 3:4] / 255.0
        if not is_transient:
            a = a * alpha
        rgb = rgb * (1 - a) + ov[:, :, :3] * a
    return rgb.astype(np.uint8)


@pytest.fixture
def canvas_factory():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    from msseg.labeler import canvas as cv
    captured = []

    class _Photo:
        def __init__(self, im): captured.append(np.asarray(im).copy())
    real_photo = cv.ImageTk.PhotoImage
    cv.ImageTk.PhotoImage = _Photo

    def make(cw=320, ch=240):
        sc = cv.SliceCanvas(root)
        sc.canvas.winfo_width = lambda: cw
        sc.canvas.winfo_height = lambda: ch
        sc.canvas.create_image = lambda *a, **k: 0
        return sc
    yield make, captured
    cv.ImageTk.PhotoImage = real_photo
    root.destroy()


@pytest.mark.parametrize("rgb", [False, True])
@pytest.mark.parametrize("view", [(1.25, 0.0, 0.0), (0.5, 40.0, 30.0), (2.0, -50.0, -20.0)])
@pytest.mark.parametrize("use_layer", [False, True])
def test_canvas_render_matches_reference(canvas_factory, rgb, view, use_layer):
    make, captured = canvas_factory
    base, labels, lut, rgba, tl = scene(rgb)
    sc = make()
    sc.set_base(array=base)
    assert sc.has_base and sc.base_is_rgb == rgb and (sc.image_height, sc.image_width) == (300, 400)
    region = {"layer": ArrayLabelLayer(labels)} if use_layer else {"labels": labels}
    sc.set_overlays([{"rgba": rgba, "visible": True}, dict(region, lut=lut, visible=True),
                     {"labels": labels, "lut": lut, "visible": False}])
    sc.set_transient(dict(region, lut=tl))
    sc.set_alpha(0.6); sc.set_window(0.1, 0.9)
    sc.scale, sc.view_x, sc.view_y = view
    captured.clear(); sc.render()
    ref = reference_render(base, [{"rgba": rgba}, {"labels": labels, "lut": lut},
                                  {"labels": labels, "lut": lut, "visible": False}],
                           {"labels": labels, "lut": tl}, 0.6, 0.1, 0.9, *view, 320, 240)
    assert np.array_equal(captured[-1], ref)
    sc.set_transient(None); captured.clear(); sc.render()
    ref = reference_render(base, [{"rgba": rgba}, {"labels": labels, "lut": lut}], None,
                           0.6, 0.1, 0.9, *view, 320, 240)
    assert np.array_equal(captured[-1], ref)


def test_canvas_public_surface(canvas_factory):
    make, captured = canvas_factory
    sc = make()
    assert not sc.has_base and sc.hud == (None, "")
    sc.set_base(array=scene()[0])
    sc.set_view(12.0, 7.0, scale=0.5)
    assert (sc.view_x, sc.view_y, sc.scale) == (12.0, 7.0, 0.5)
    sc.set_view(scale=1000.0)                       # clamped to the image size
    assert sc.scale == 400.0
    sc.set_hud("info", "t 0.5"); assert sc.hud == ("info", "t 0.5")
    sc.set_base(array=None, reset_array=True)
    assert not sc.has_base and sc._base is None


def test_pyramid_source_is_re_exported_and_refuses_an_unreadable_path():
    """One import site for both base-image implementations; the failure to open
    is a RuntimeError whatever backends are installed (see test_pyramid)."""
    from msseg.labeler import sources
    from msseg.labeler.pyramid import PyramidImageSource
    assert sources.PyramidImageSource is PyramidImageSource
    assert sources.HAVE_PYRAMID == bool(pyramid.backends_available())
    with pytest.raises(RuntimeError):
        sources.PyramidImageSource("nope.tiff")


def test_blend_luts_fold_the_per_pixel_alpha_exactly(canvas_factory):
    """The premultiplied LUTs must reproduce the per-pixel float32 arithmetic
    they replace, value for value -- that is what keeps the composite
    bit-identical to the reference above."""
    make, _captured = canvas_factory
    sc = make()
    lut = np.array([[10, 20, 30, 255], [40, 50, 60, 0], [70, 80, 90, 128]], np.uint8)
    premul, one_minus = sc._blend_luts(lut, 0.6)
    a = lut[:, 3].astype(np.float32) / 255.0
    a = a * 0.6
    assert np.array_equal(premul[:3], lut[:, :3].astype(np.float32) * a[:, None])
    # 3 wide so the gather is contiguous and the multiply does not broadcast
    assert premul.shape == one_minus.shape == (4, 3)
    assert np.array_equal(one_minus[:3], np.repeat((1 - a)[:, None], 3, 1))
    # the appended row is what a background id (-1) reaches by negative wrap
    assert not premul[-1].any() and np.array_equal(one_minus[-1], np.ones(3, np.float32))
    # cached per (LUT, alpha): a region LUT has a row per region
    assert sc._blend_luts(lut, 0.6)[0] is premul
    assert sc._blend_luts(lut, 0.3)[0] is not premul
    # bounded, most-recently-used kept: the stable overlay LUT survives a
    # gesture handing over a fresh preview LUT on every drag tick
    for _ in range(sc._LUT_CACHE_MAX + 2):
        sc._blend_luts(lut.copy(), 0.6)
        assert sc._blend_luts(lut, 0.6)[0] is premul
    assert len(sc._lut_cache) <= sc._LUT_CACHE_MAX


def test_repaint_deadline_keeps_a_drag_from_starving_the_render(canvas_factory):
    """Motion events arrive faster than the debounce, so re-arming the timer on
    every one of them means it never fires. Past _MAX_DEFER_MS the armed job
    must be left alone; a completed render restarts the clock."""
    make, _captured = canvas_factory
    sc = make()
    log = []
    sc.after = lambda ms, fn: (log.append(("arm", ms)), f"job{len(log)}")[1]
    sc.after_cancel = lambda job: log.append(("cancel", job))

    sc._MAX_DEFER_MS = 1e9                       # never reached: coalesce freely
    for _ in range(4):
        sc._schedule()
    assert [k for k, _ in log] == ["arm", "cancel", "arm", "cancel", "arm", "cancel", "arm"]

    sc._job = sc._pending_since = None
    log.clear()
    sc._MAX_DEFER_MS = 0                         # every later request is overdue
    for _ in range(4):
        sc._schedule()
    assert [k for k, _ in log] == ["arm"] and sc._job is not None

    sc.render()                                  # no base: returns after the reset
    assert sc._job is None and sc._pending_since is None


class _CropOnlyLayer:
    """A LabelLayer that will not hand over a full raster -- the whole-slide
    case, where the ids cover part of an image far too large to materialise."""

    def __init__(self, labels):
        self._inner = ArrayLabelLayer(labels)

    shape = property(lambda self: self._inner.shape)
    n_ids = property(lambda self: self._inner.n_ids)
    rev = property(lambda self: self._inner.rev)

    def crop(self, level, x, y, w, h):
        return self._inner.crop(level, x, y, w, h)

    def id_at(self, x, y):
        return self._inner.id_at(x, y)

    def full(self):
        return None


@pytest.mark.parametrize("view", [(1.0, 0.0, 0.0), (0.5, 40.0, 30.0), (3.0, -20.0, -10.0)])
def test_a_crop_only_layer_renders(canvas_factory, view):
    """The crop branch of _label_region resizes through PIL, whose buffer is
    READ-ONLY -- and the composite clamps background ids in place. Nothing in
    the tree took that branch until a layer with no full raster existed, so the
    first render of one raised `output array is read-only`."""
    make, captured = canvas_factory
    base, labels, lut, _rgba, tl = scene()
    sc = make()
    sc.set_base(array=base)
    sc.set_overlays([{"layer": _CropOnlyLayer(labels), "lut": lut, "visible": True}])
    sc.set_transient({"layer": _CropOnlyLayer(labels), "lut": tl})
    sc.scale, sc.view_x, sc.view_y = view
    captured.clear()
    sc.render()                                  # must not raise
    assert captured and captured[-1].shape[2] == 3


def test_the_crop_branch_hands_back_a_writable_array(canvas_factory):
    make, _captured = canvas_factory
    _base, labels, *_ = scene()
    sc = make()
    out = sc._label_region(_CropOnlyLayer(labels), 0, 0, 200, 150, 100, 75, {})
    assert out.flags.writeable, "the composite writes into this"
    assert out.dtype == np.int32


def test_the_source_kind_names_what_is_being_read(canvas_factory):
    """A NATIVE pyramid takes the array slot, so "which slot is filled" was
    reporting a gigapixel slide as in-memory."""
    make, _captured = canvas_factory
    base, *_ = scene()
    sc = make()
    assert sc._source_kind() == "none"
    sc.set_base(array=base)
    assert sc._source_kind() == "in-memory"

    class _Pyramid(ArrayImageSource):
        native = True
        levels = 6
    sc.set_source(_Pyramid(base))
    assert sc._source_kind() == "pyramid"


def test_a_fit_before_the_canvas_has_a_size_waits_for_one(canvas_factory):
    """At construction the canvas is 1x1; a fit then is image_width pixels per
    screen pixel -- the whole image in one dot -- and nothing corrects it once
    the window maps. The slide labeler fits on load and showed nothing."""
    make, _captured = canvas_factory
    base, *_ = scene()
    sc = make(cw=1, ch=1)                       # not laid out yet
    sc.set_base(array=base)
    sc.fit()
    assert sc._fit_pending and sc.scale == 1.0, "a 1x1 fit must not set a zoom"
    sc.canvas.winfo_width = lambda: 400
    sc.canvas.winfo_height = lambda: 300
    sc._fit_when_sized()                        # what <Configure> delivers
    assert not sc._fit_pending
    assert sc.scale == pytest.approx(1.0) and (sc.view_x, sc.view_y) == (0.0, 0.0)
    # and a fit on a sized canvas is immediate, as it always was
    sc.canvas.winfo_width = lambda: 200
    sc.fit()
    assert sc.scale == pytest.approx(2.0) and not sc._fit_pending
