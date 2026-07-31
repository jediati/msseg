"""Coupon analysis: 2D Morse-Smale segmentation of a tomography slice stack.

Updated port of the old ``dataops/py/coupon_analysis_f32.ipynb`` notebook onto
the refactored MSSeg Python API. The old notebook drove the legacy ``msc_py``
module (``MakeMSCInstance`` -> ``ComputeMSC`` -> ``SetMSCPersistence`` ->
``GetAsc2Manifolds``) plus ``diffg_py`` for edge filtering; both are replaced by
the single ``msseg.mscoupon`` instance:

    filtered = mscoupon.filter_slice(img, "<filter params json>")   # topo field
    labels   = mscoupon.segment_slice(filtered, "<msc params json>") # int32 labels

The workflow (faithful to notebook cells 5-36):
  1. load a stack of TIFF slices (+ optional binary ground-truth masks),
  2. per slice: edge/gradient-magnitude filter, then a 2D MSC ascending
     2-manifold labeling at a chosen persistence,
  3. per-segment intensity statistics (area / min / max / mean / median),
  4. keep segments whose mean intensity is below a threshold,
  5. link the kept 2D segments into 3D connected components across the stack,
  6. write an interactive multi-layer Bokeh viewer (grayscale / topo /
     segmentation / mask) with a slice slider to a standalone HTML file.

The notebook's critical-point and polyline-graph overlays are intentionally
omitted (no equivalent binding; they belonged to a separate RGB-cell experiment).

Requires (beyond the msseg wheel): numpy, scipy, pandas, tifffile, pillow, bokeh
    pip install numpy scipy pandas tifffile pillow bokeh

Run:
    python coupon_analysis.py --input "F:/data/rec/recon_0009[1-7].tiff" \
        --sigma 0.7 --persistence-abs 20 --threshold 140 \
        --parallelism 4 --concurrent-slices 4 \
        --view-downsample 5 --out coupon_viewer.html --show
        # Slices compute concurrently: --concurrent-slices at a time, each using
        # --parallelism threads (product ~= core count) — this is the main speedup.
        # --view-downsample decimates the DISPLAY only (compute stays full-res);
        # use it for large images so the standalone HTML stays browser-openable.
    python coupon_analysis.py --selftest     # synthetic data, no external files
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import (generate_binary_structure, label, maximum, median,
                           minimum)

from msseg import mscoupon


# --------------------------------------------------------------------------- #
# Parameter JSON builders (old msc_py / diffg_py call arguments -> params_json)
# --------------------------------------------------------------------------- #
def edge_filter_params(sigma=0.7, threads=8, suppress_nonmax=False):
    """diffg edge/gradient-magnitude filter -> the notebook's ``fj.edges(...)``."""
    return json.dumps({"filter": {"operation": "edges",
                                  "params": {"sigma": sigma, "threads": threads,
                                             "suppress_nonmax": suppress_nonmax}}})


def msc_params(persistence_abs=None, persistence_pct=None, manifold="ascending",
               compute_algorithm="serial", parallelism=0):
    """2D MSC over an already-filtered field (filter=none). Mirrors the old
    ``ComputeMSC(..., False, False)`` (non-accurate boundaries) +
    ``SetMSCPersistence`` + ``GetAsc2Manifolds`` / ``GetDsc2Manifolds``.

    ``parallelism`` sets the discrete-gradient thread count (and, in
    "partitioned" mode, the MSC/hierarchy partition count); 0 leaves the
    msc_2d_lib default (8)."""
    msc = {"compute_algorithm": compute_algorithm, "manifold": manifold,
           "accurate_ascending": False, "accurate_descending": False}
    if parallelism:
        msc["requested_parallelism"] = int(parallelism)
    if persistence_abs is not None:
        msc["persistence_absolute"] = float(persistence_abs)
    if persistence_pct is not None:
        msc["persistence_percent"] = float(persistence_pct)
    return json.dumps({"filter": {"operation": "none"}, "msc": msc})


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _natural_key(path):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", os.path.basename(path))]


def resolve_files(pattern):
    """A glob pattern, or a directory (all .tif/.tiff inside), naturally sorted."""
    if pattern is None:
        return []
    if os.path.isdir(pattern):
        files = [p for p in glob.glob(os.path.join(pattern, "*"))
                 if os.path.splitext(p)[1].lower() in (".tif", ".tiff")]
    else:
        files = glob.glob(pattern)
    return sorted(files, key=_natural_key)


def load_stack(input_pattern, mask_pattern=None):
    """Load slices -> list of float32 (h,w); masks -> list of float32 (h,w) or None."""
    import tifffile
    from PIL import Image

    files = resolve_files(input_pattern)
    if not files:
        raise FileNotFoundError(f"no TIFF slices matched: {input_pattern!r}")
    images = [np.ascontiguousarray(np.squeeze(np.asarray(Image.open(f))), dtype=np.float32)
              for f in files]

    masks = None
    mask_files = resolve_files(mask_pattern)
    if mask_files:
        if len(mask_files) != len(files):
            raise ValueError(f"mask count ({len(mask_files)}) != slice count ({len(files)})")
        masks = [np.asarray(tifffile.imread(f)).astype(np.float32) for f in mask_files]
    return images, masks


# --------------------------------------------------------------------------- #
# Segmentation (the mscoupon calls)
# --------------------------------------------------------------------------- #
def segment_stack(images, sigma=0.7, parallelism=4, concurrent_slices=4,
                  suppress_nonmax=False, persistence_abs=None, persistence_pct=None,
                  manifold="ascending", compute_algorithm="serial"):
    """Per slice: filter to a topo field, then MSC-label it.

    Slices are computed concurrently: ``concurrent_slices`` at a time, each using
    ``parallelism`` threads/partitions (so ``concurrent_slices * parallelism``
    should roughly match the core count). ``segment_slice``/``filter_slice``
    release the GIL, so a thread pool runs the C++ compute truly in parallel
    (verified thread-safe). Returns (topo_fields, labels): parallel lists of
    float32 (h,w) and int32 (h,w), in input order.
    """
    if mscoupon is None:
        raise RuntimeError("msseg.mscoupon is not available; build with "
                           "-DMSSEG_BUILD_PYTHON=ON and install the wheel.")
    fparams = edge_filter_params(sigma, parallelism, suppress_nonmax)
    mparams = msc_params(persistence_abs, persistence_pct, manifold, compute_algorithm, parallelism)

    def work(img):
        img = np.ascontiguousarray(img, dtype=np.float32)
        topo = np.asarray(mscoupon.filter_slice(img, fparams), dtype=np.float32)
        lab = np.asarray(mscoupon.segment_slice(topo, mparams), dtype=np.int32)
        return topo, lab

    workers = max(1, int(concurrent_slices))
    if workers == 1:
        results = [work(im) for im in images]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(work, images))  # map preserves input order
    return [r[0] for r in results], [r[1] for r in results]


# --------------------------------------------------------------------------- #
# Per-segment statistics (pure NumPy/SciPy; notebook cell 31)
# --------------------------------------------------------------------------- #
def segment_stats(images, labels):
    """Per slice, a DataFrame of area / min / max / mean / median intensity."""
    tables = []
    for img, lab in zip(images, labels):
        labels_flat = lab.ravel()
        values_flat = img.ravel().astype(np.float64)
        unique_ids = np.unique(labels_flat)
        min_r = minimum(values_flat, labels=labels_flat, index=unique_ids)
        max_r = maximum(values_flat, labels=labels_flat, index=unique_ids)
        med_r = median(values_flat, labels=labels_flat, index=unique_ids)
        sum_r = np.bincount(labels_flat, weights=values_flat)
        cnt_r = np.bincount(labels_flat)
        avg_r = sum_r[unique_ids] / cnt_r[unique_ids]
        tables.append(pd.DataFrame({
            "Segment_ID": unique_ids,
            "Pixel_Count": cnt_r[unique_ids],
            "Min_Intensity": min_r,
            "Max_Intensity": max_r,
            "Avg_Intensity": avg_r,
            "Med_Intensity": med_r,
        }))
    return tables


# --------------------------------------------------------------------------- #
# Threshold-filtered overlays (notebook cell 33)
# --------------------------------------------------------------------------- #
def threshold_overlays(labels, tables, max_mean):
    """Keep segments with mean intensity < max_mean; NaN elsewhere (float overlay)."""
    overlays = []
    for lab, table in zip(labels, tables):
        keep_ids = table.loc[table["Avg_Intensity"] < max_mean, "Segment_ID"].to_numpy()
        overlays.append(np.where(np.isin(lab, keep_ids), lab, np.nan))
    return overlays


# --------------------------------------------------------------------------- #
# 3D connected components across the stack (notebook cell 36)
# --------------------------------------------------------------------------- #
def connected_components_3d(overlays, images):
    """6-connected 3D labeling of the kept overlays; returns (labeled_3d, stats_df)."""
    stack = np.nan_to_num(np.stack(overlays, axis=0), nan=0.0)
    struct6 = generate_binary_structure(rank=3, connectivity=1)
    labeled3d, num = label(stack > 0, structure=struct6)

    labels_flat = labeled3d.ravel()
    values_flat = np.stack(images, axis=0).ravel().astype(np.float64)
    uids = np.unique(labels_flat)
    uids = uids[uids != 0]
    if uids.size:
        min_r = minimum(values_flat, labels=labels_flat, index=uids)
        max_r = maximum(values_flat, labels=labels_flat, index=uids)
        med_r = median(values_flat, labels=labels_flat, index=uids)
        sum_r = np.bincount(labels_flat, weights=values_flat)
        cnt_r = np.bincount(labels_flat)
        avg_r = sum_r[uids] / cnt_r[uids]
        stats = pd.DataFrame({
            "Component_ID": uids, "Voxel_Count": cnt_r[uids],
            "Min_Intensity": min_r, "Max_Intensity": max_r,
            "Avg_Intensity": avg_r, "Med_Intensity": med_r,
        })
    else:
        stats = pd.DataFrame(columns=["Component_ID", "Voxel_Count", "Min_Intensity",
                                      "Max_Intensity", "Avg_Intensity", "Med_Intensity"])
    return labeled3d, num, stats


# --------------------------------------------------------------------------- #
# Interactive Bokeh viewer (notebook cells 35/36) -> standalone HTML
# --------------------------------------------------------------------------- #
def build_viewer(images, topo_fields, overlay_stack, out_html, masks=None, show=False,
                 view_downsample=1):
    """Multi-layer slice viewer (grayscale / topo / segmentation / mask) + slider.

    ``overlay_stack`` is a float (slices, h, w) array with NaN outside segments
    (e.g. the 3D-labeled stack), so segment colors are consistent across slices.

    A standalone HTML embeds every slice of every layer, so for large images set
    ``view_downsample > 1`` to decimate the *displayed* arrays (nearest stride;
    segmentation stays label-exact). Compute is unaffected — coordinates still map
    to full-resolution pixel space because the glyph extent (dw/dh) is unchanged.
    """
    ds = max(1, int(view_downsample))

    def _dec(a):
        return a[::ds, ::ds]
    from bokeh.io import output_file, save
    from bokeh.io import show as bshow
    from bokeh.layouts import column
    from bokeh.models import (ColumnDataSource, CustomJS, HoverTool, Legend,
                              LegendItem, LinearColorMapper, Slider)
    from bokeh.plotting import figure

    n = len(images)
    img_h, img_w = images[0].shape

    # Consistent random palette sized to the number of distinct segment ids.
    import colorsys
    valid = overlay_stack[~np.isnan(overlay_stack)]
    n_ids = int(valid.max()) + 1 if valid.size else 1
    np.random.seed(42)

    def _hex(hue):
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

    palette = [_hex(h) for h in np.random.rand(max(n_ids, 1))]

    # Bokeh image origin is bottom-left -> flip vertically (as the notebook did);
    # decimate for display only (dw/dh below keep the full-resolution extent).
    gray_slices = [np.flipud(_dec(a)) for a in images]
    topo_slices = [np.flipud(_dec(t)) for t in topo_fields]
    seg_slices = [np.flipud(_dec(overlay_stack[i])) for i in range(n)]
    have_masks = masks is not None

    data = dict(gray_channel=[gray_slices[0]], topo_channel=[topo_slices[0]],
                segment_channel=[seg_slices[0]])
    if have_masks:
        # binary mask -> pink only where foreground, NaN (transparent) elsewhere.
        mask_slices = [np.where(np.flipud(_dec(m)) > 0, 1.0, np.nan).astype(np.float32)
                       for m in masks]
        data["mask_channel"] = [mask_slices[0]]
    source = ColumnDataSource(data=data)

    p = figure(width=1000, height=int(1000 * (img_h / img_w)),
               x_range=(0, img_w), y_range=(0, img_h),
               tools="pan,wheel_zoom,box_zoom,reset,save",
               title="Coupon MSC segmentation")

    gray_r = p.image(image="gray_channel", x=0, y=0, dw=img_w, dh=img_h,
                     palette="Greys256", source=source)
    topo_r = p.image(image="topo_channel", x=0, y=0, dw=img_w, dh=img_h,
                     palette="Greys256", source=source)
    seg_mapper = LinearColorMapper(palette=palette, nan_color="#00000000")
    seg_r = p.image(image="segment_channel", x=0, y=0, dw=img_w, dh=img_h,
                    color_mapper=seg_mapper, alpha=0.6, source=source)

    items = [LegendItem(label="Grayscale", renderers=[gray_r]),
             LegendItem(label="Topo field", renderers=[topo_r]),
             LegendItem(label="Segmentation", renderers=[seg_r])]
    if have_masks:  # only create the mask layer when there actually is one
        mask_mapper = LinearColorMapper(palette=["#FF3333", "#FF3333"], low=0, high=1,
                                        nan_color="#00000000")
        mask_r = p.image(image="mask_channel", x=0, y=0, dw=img_w, dh=img_h,
                         color_mapper=mask_mapper, alpha=0.4, source=source)
        items.append(LegendItem(label="Ground-truth mask", renderers=[mask_r]))
    legend = Legend(items=items, location="top_right")
    legend.click_policy = "hide"
    p.add_layout(legend)

    hover = HoverTool(renderers=[gray_r], tooltips=[
        ("X", "$x{0}"), ("Y", "$y{0}"), ("Segment", "@segment_channel{0}"),
        ("Gray", "@gray_channel{0.00000}"), ("Topo", "@topo_channel{0.00000}")])
    p.add_tools(hover)

    slider = Slider(start=0, end=max(n - 1, 0), value=0, step=1, title="Slice", width=600)
    args = dict(source=source, all_gray=gray_slices, all_topo=topo_slices, all_segment=seg_slices)
    code = ("const i = cb_obj.value;\n"
            "source.data['gray_channel'] = [all_gray[i]];\n"
            "source.data['topo_channel'] = [all_topo[i]];\n"
            "source.data['segment_channel'] = [all_segment[i]];\n")
    if have_masks:
        args["all_mask"] = mask_slices
        code += "source.data['mask_channel'] = [all_mask[i]];\n"
    code += "source.change.emit();\n"
    slider.js_on_change("value", CustomJS(args=args, code=code))

    layout = column(p, slider)
    output_file(out_html, title="Coupon MSC segmentation")
    if show:
        bshow(layout)
    else:
        save(layout)
    return out_html


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(images, masks, args):
    import time
    print(f"Loaded {len(images)} slice(s) of shape {images[0].shape}")
    print(f"Segmenting {len(images)} slice(s): {args.concurrent_slices} at a time, "
          f"{args.parallelism} threads each ({args.compute_algorithm})...")
    t0 = time.time()
    topo_fields, labels = segment_stack(
        images, sigma=args.sigma, parallelism=args.parallelism,
        concurrent_slices=args.concurrent_slices, suppress_nonmax=args.suppress_nonmax,
        persistence_abs=args.persistence_abs, persistence_pct=args.persistence_pct,
        manifold=args.manifold, compute_algorithm=args.compute_algorithm)
    print(f"Segmented {len(labels)} slice(s) in {time.time() - t0:.1f}s: "
          f"{sum(len(np.unique(l)) for l in labels)} total 2D segments")

    tables = segment_stats(images, labels)
    overlays = threshold_overlays(labels, tables, args.threshold)
    labeled3d, num, stats3d = connected_components_3d(overlays, images)
    print(f"3D connected components (kept, mean < {args.threshold}): {num}")

    labeled3d_f = labeled3d.astype(np.float32)
    labeled3d_f[labeled3d_f == 0] = np.nan
    out = build_viewer(images, topo_fields, labeled3d_f, args.out, masks=masks,
                       show=args.show, view_downsample=args.view_downsample)
    print(f"Wrote viewer: {out}")
    return topo_fields, labels, tables, overlays, labeled3d, stats3d


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", help="glob or directory of TIFF slices")
    ap.add_argument("--masks", default=None, help="glob or directory of binary masks")
    ap.add_argument("--sigma", type=float, default=0.7, help="edge filter sigma")
    ap.add_argument("--parallelism", type=int, default=4,
                    help="threads/partitions per slice (filter + MSC gradient)")
    ap.add_argument("--concurrent-slices", type=int, default=4,
                    help="slices computed at once; parallelism * concurrent-slices ~= cores")
    ap.add_argument("--compute-algorithm", choices=["serial", "partitioned"], default="serial",
                    help="MSC builder mode (serial reproduces canonical labels; partitioned "
                         "gave no speedup in testing since cross-slice concurrency is the lever)")
    ap.add_argument("--suppress-nonmax", action="store_true", help="non-max suppress edges")
    ap.add_argument("--persistence-abs", type=float, default=20.0,
                    help="absolute MSC simplification persistence")
    ap.add_argument("--persistence-pct", type=float, default=None,
                    help="percent MSC persistence (overrides --persistence-abs if set)")
    ap.add_argument("--manifold", choices=["ascending", "descending"], default="ascending")
    ap.add_argument("--threshold", type=float, default=0.00015,
                    help="keep segments with mean intensity below this")
    ap.add_argument("--out", default="coupon_viewer.html", help="output HTML path")
    ap.add_argument("--view-downsample", type=int, default=1,
                    help="decimate displayed arrays by this factor (compute stays full-res; "
                         "use >1 for large images so the standalone HTML stays browser-openable)")
    ap.add_argument("--show", action="store_true", help="open the viewer in a browser")
    ap.add_argument("--selftest", action="store_true", help="run on synthetic data")
    return ap.parse_args(argv)


def _selftest():
    """Run the full pipeline on a synthetic 3-slice stack; assert shapes."""
    import tempfile
    h, w, n = 48, 48, 3
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    images = []
    for k in range(n):
        img = np.zeros((h, w), np.float32)
        for cy, cx in [(16, 16 + k), (30, 32 - k)]:
            r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            img += np.exp(-((r - 6.0) ** 2) / (2 * 2.0 ** 2))
        images.append(np.ascontiguousarray(img, np.float32))
    masks = [(im > 0.5).astype(np.float32) for im in images]

    class A:  # argparse-like namespace with selftest-appropriate values
        sigma, suppress_nonmax = 1.0, False
        parallelism, concurrent_slices, compute_algorithm = 2, 2, "serial"
        persistence_abs, persistence_pct, manifold = None, 5.0, "ascending"
        threshold = 1e9  # keep everything (synthetic intensities are small)
        out = os.path.join(tempfile.gettempdir(), "coupon_selftest.html")
        view_downsample = 1
        show = False

    topo, labels, tables, overlays, labeled3d, stats3d = run(images, masks, A)
    assert len(topo) == n and topo[0].shape == (h, w) and topo[0].dtype == np.float32
    assert len(labels) == n and labels[0].shape == (h, w) and labels[0].dtype == np.int32
    assert len(tables) == n and "Avg_Intensity" in tables[0].columns
    assert len(overlays) == n and overlays[0].shape == (h, w)
    assert labeled3d.shape == (n, h, w)
    assert os.path.exists(A.out)
    print(f"selftest OK: {n} slices, labels {labels[0].shape}, "
          f"3D components {int(labeled3d.max())}, viewer {A.out}")


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.selftest:
        return _selftest()
    if not args.input:
        print("error: --input is required (or use --selftest)", file=sys.stderr)
        return 2
    # percent persistence, when given, takes precedence over absolute
    if args.persistence_pct is not None:
        args.persistence_abs = None
    images, masks = load_stack(args.input, args.masks)
    run(images, masks, args)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
