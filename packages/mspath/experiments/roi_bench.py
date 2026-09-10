"""What does one ROI of a whole slide cost? -- the measurement P0 exists for.

Everything about the gigapixel design turns on one number per tile size: the
wall time and the peak RSS of priming an ROI. The estimates that motivated the
plan (~100 B/px steady, ~160 B/px peak, ~0.62 us/px of MSC) came from a
3232-square coupon slice with a different filter chain on a different machine,
and a tile budget picked from an estimate is a tile budget that OOMs on the
first real slide. So: read real tiles out of a real pyramid, prime them the way
`ComputeEngine` does, and print the table.

Two experiments, both headless:

* ``sizes`` -- prime 1024/2048/4096/8192-square tiles at level 0 and report
  ms/Mpx, bytes/px and peak RSS, so the ROI budget is chosen from data.
* ``halo`` -- prime the SAME interior rect with a growing halo around it and
  compare the interior labelling against the widest halo's. The filter chain is
  a stack of Gaussian derivatives whose truncation multiple lives inside diffg
  (FetchContent-pinned, not checked out), so the halo that makes a tiled ROI
  agree with an untiled one is measured here rather than read off a constant.
  This is the procedure ilastik documents for picking a block halo.

Tiles are chosen by tissue content, not by coordinate: a slide is mostly white
background, and priming background measures the cost of nothing.

    python packages/mspath/experiments/roi_bench.py sizes --slide <path>
    python packages/mspath/experiments/roi_bench.py halo  --slide <path>

Needs the three source trees on PYTHONPATH (see docs/labeler_framework.md) and
a pyramid backend (`pip install openslide-python`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from msseg.labeler.pyramid import PyramidImageSource
from msseg.mscoupon import mscoupon_py as engine
from msseg.mscoupon.engine import ComputeEngine, prime

DEFAULT_SLIDE = (r"C:\Users\jediati\Desktop\JEDIATI\data\arpa-h\topo-data\WSI"
                 r"\Subset1_Test_10.tiff")

# The workflow a slide ROI actually runs: optical density off the RGB planes
# (tissue is what absorbs), a light blur, and the MSC on the edge field.
# accurate=False is mandatory for anything compared across ROIs -- with the
# accurate discrete gradient the MSC is run-to-run nondeterministic, so a halo
# sweep would measure the noise instead of the halo.
PROFILE = {
    "filters": [{"operation": "color", "params": {"method": "optical_density"}},
                {"operation": "blur", "params": {"sigma": 1.5}},
                {"operation": "edges", "params": {"sigma": 1.0, "output": "magnitude"}}],
    "base_filters": [{"operation": "color", "params": {"method": "optical_density"}}],
    "msc": {"manifold": "ascending", "accurate_ascending": False,
            "accurate_descending": False, "simplification": "merge_forest",
            "compute_algorithm": "serial", "use_gpu_gradient": False},
    "statistics": {"channels": ["base"], "reductions": ["mean", "min", "max", "std"],
                   "extremum": True},
}


def log(msg):
    print(msg, flush=True)


def peak_rss_mb():
    """Peak working set of this process, in MB (Windows: peak_wset)."""
    import psutil
    info = psutil.Process().memory_info()
    return float(getattr(info, "peak_wset", info.rss)) / 1e6


def rss_mb():
    import psutil
    return float(psutil.Process().memory_info().rss) / 1e6


# --------------------------------------------------------------------------- #
# picking somewhere worth priming
# --------------------------------------------------------------------------- #
def find_tissue(src, want_px, level=0, probe_level=None):
    """Top-left (level-0 x, y) of a `want_px`-square window whose tissue
    fraction is highest.

    Tissue is "not near-white": a slide is mostly background, and a tile of
    background has almost no topology, so priming one measures nothing. The
    search runs on a coarse level -- a box filter over the darkness mask via a
    summed-area table, then the argmax.
    """
    if probe_level is None:                      # a probe level of a few Mpx
        probe_level = max(0, min(src.levels - 1,
                                 src.best_level(max(1.0, want_px / 64.0)) + 2))
    h, w = src.level_shape(probe_level)
    a = src.read_region(probe_level, 0, 0, w, h)
    gray = a.mean(axis=2) if a.ndim == 3 else a
    dark = (gray < 0.85 * float(np.max(gray) or 1)).astype(np.float32)
    scale = src.level_scale(probe_level)
    box = max(1, int(round(want_px / scale)))
    if box >= min(h, w):                          # the window covers the probe
        return 0, 0, probe_level, float(dark.mean())
    sat = dark.cumsum(0).cumsum(1)
    sat = np.pad(sat, ((1, 0), (1, 0)))
    tot = (sat[box:, box:] - sat[:-box, box:] - sat[box:, :-box] + sat[:-box, :-box])
    iy, ix = np.unravel_index(int(np.argmax(tot)), tot.shape)
    frac = float(tot[iy, ix]) / float(box * box)
    lx, ly = int(ix * scale), int(iy * scale)     # probe -> level-0 coordinates
    s0 = src.level_scale(level)
    return int(lx / s0), int(ly / s0), probe_level, frac


# --------------------------------------------------------------------------- #
# one prime
# --------------------------------------------------------------------------- #
def prime_tile(src, level, x, y, size_w, size_h, profile=PROFILE, quiet=True,
               persistence_abs=None):
    """Read a rect, run both chains, prime it. Returns the pipe, the labels and
    a timing dict -- the same sequence ComputeEngine walks per slice.

    `persistence_abs` pins the threshold in the field's own units. Leaving it
    None takes 10% of THIS tile's value range, which is what the coupon
    workflow does and what any comparison across differently-sized reads must
    NOT do: a padded tile sees a wider range, so the same percentage is a
    different threshold and the two labellings differ for a reason that has
    nothing to do with the halo.
    """
    say = (lambda _m: None) if quiet else log
    t0 = time.perf_counter()
    tile = src.read_region(level, x, y, size_w, size_h)
    t_read = time.perf_counter()

    # (h, w, C) from the reader -> the planar (C, h, w) float32 the chains want.
    arr = (np.ascontiguousarray(np.transpose(tile, (2, 0, 1)), dtype=np.float32)
           if tile.ndim == 3 else np.ascontiguousarray(tile, dtype=np.float32))

    cur, rest = ComputeEngine._leading_color(arr, profile["filters"], engine, say)
    for f in rest:
        cur = engine.filter_slice(cur, json.dumps({"filter": f}))
    filt = np.ascontiguousarray(cur, dtype=np.float32)
    base, _norms = ComputeEngine._apply_base_chain(arr, profile["base_filters"], engine, say)
    t_filter = time.perf_counter()

    planes = arr if arr.ndim == 3 else None
    params = dict(profile)
    pipe = prime(engine, base, filt, json.dumps(params), planes)
    t_prime = time.perf_counter()

    rng = float(pipe.value_range())
    pipe.select_persistence(rng * 0.10 if persistence_abs is None else float(persistence_abs))
    labels = pipe.labels()
    t_select = time.perf_counter()

    return pipe, labels, {
        "read_ms": 1e3 * (t_read - t0), "filter_ms": 1e3 * (t_filter - t_read),
        "prime_ms": 1e3 * (t_prime - t_filter), "select_ms": 1e3 * (t_select - t_prime),
        "total_ms": 1e3 * (t_select - t0),
    }


# --------------------------------------------------------------------------- #
# experiment 1: how big can an ROI be?
# --------------------------------------------------------------------------- #
def run_sizes(src, sizes, level, out_json=None):
    log(f"slide: {src}")
    log(f"backends: {src.be.name}   level {level} shape {src.level_shape(level)}")
    rows = []
    for n in sizes:
        x, y, probe, frac = find_tissue(src, n * src.level_scale(level), level=level)
        src.clear_cache()
        base_rss = rss_mb()
        try:
            pipe, labels, t = prime_tile(src, level, x, y, n, n)
        except MemoryError as exc:
            log(f"{n:>6}^2  MemoryError: {exc}")
            break
        mpx = n * n / 1e6
        n_feat = len(pipe.feature_stats())
        row = dict(size=n, mpx=mpx, x=x, y=y, tissue=frac, regions=n_feat,
                   rss_mb=rss_mb(), peak_mb=peak_rss_mb(), delta_mb=rss_mb() - base_rss, **t)
        row["ms_per_mpx"] = t["total_ms"] / mpx
        row["bytes_per_px"] = row["delta_mb"] * 1e6 / (n * n)
        rows.append(row)
        log(f"{n:>6}^2 {mpx:7.1f} Mpx  tissue {frac:4.0%}  "
            f"read {t['read_ms']:6.0f}  filter {t['filter_ms']:6.0f}  "
            f"prime {t['prime_ms']:8.0f}  select {t['select_ms']:6.0f}  "
            f"total {t['total_ms']:8.0f} ms  ({row['ms_per_mpx']:6.0f} ms/Mpx)  "
            f"regions {n_feat:>8}  RSS +{row['delta_mb']:7.0f} MB "
            f"({row['bytes_per_px']:5.0f} B/px)  peak {row['peak_mb']:7.0f} MB")
        del pipe, labels
    if out_json:
        with open(out_json, "w") as fh:
            json.dump(rows, fh, indent=2)
        log(f"wrote {out_json}")
    return rows


# --------------------------------------------------------------------------- #
# experiment 2: how much halo does the filter chain need?
# --------------------------------------------------------------------------- #
def run_halo(src, size, halos, level, out_json=None):
    """Prime the same interior rect with a growing halo and compare interiors.

    Region IDS are not comparable between primes (they are assigned in scan
    order over a different raster), so the comparison is the PARTITION: two
    labellings agree where they cut the interior the same way. Measured as the
    fraction of interior pixel pairs -- sampled as horizontal and vertical
    neighbours -- that the two labellings agree about being in the same region.
    A boundary that moves shows up; a renumbering does not.
    """
    halos = sorted(halos)
    x, y, probe, frac = find_tissue(src, size * src.level_scale(level), level=level)
    log(f"interior {size}^2 at level {level} ({x}, {y}), tissue {frac:.0%}")

    # Pin the threshold in field units from the UNPADDED tile, then hold it for
    # every halo. Without this the sweep measures the value range growing with
    # the padding rather than the halo doing its job.
    pipe0, _l0, _t0 = prime_tile(src, level, x, y, size, size)
    pabs = float(pipe0.value_range()) * 0.10
    del pipe0
    log(f"persistence pinned at {pabs:.6g} (10% of the unpadded tile's range)")

    interiors = {}
    for hpx in halos:
        src.clear_cache()
        pipe, labels, t = prime_tile(src, level, x - hpx, y - hpx, size + 2 * hpx,
                                     size + 2 * hpx, persistence_abs=pabs)
        inner = labels[hpx:hpx + size, hpx:hpx + size] if hpx else labels
        interiors[hpx] = np.ascontiguousarray(inner)
        log(f"  halo {hpx:>4}: {t['total_ms']:8.0f} ms   regions in interior "
            f"{len(np.unique(inner)):>7}   range {float(pipe.value_range()):.6g}")
        del pipe, labels

    ref = interiors[halos[-1]]
    rows = []
    log(f"  agreement with the widest halo ({halos[-1]} px):")
    for hpx in halos:
        a = interiors[hpx]
        same = 0.0
        n = 0
        for axis in (0, 1):
            pa = np.take(a, np.arange(a.shape[axis] - 1), axis=axis)
            pb = np.take(a, np.arange(1, a.shape[axis]), axis=axis)
            ra = np.take(ref, np.arange(ref.shape[axis] - 1), axis=axis)
            rb = np.take(ref, np.arange(1, ref.shape[axis]), axis=axis)
            same += float(np.count_nonzero((pa == pb) == (ra == rb)))
            n += pa.size
        agree = same / max(n, 1)
        rows.append({"halo": hpx, "pair_agreement": agree})
        log(f"    halo {hpx:>4}: {agree:8.5%}   (disagreeing pairs "
            f"{int(round((1 - agree) * n)):>9})")
    if out_json:
        with open(out_json, "w") as fh:
            json.dump(rows, fh, indent=2)
        log(f"wrote {out_json}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("experiment", choices=["sizes", "halo"])
    ap.add_argument("--slide", default=DEFAULT_SLIDE)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    ap.add_argument("--halo-size", type=int, default=1024)
    ap.add_argument("--halos", type=int, nargs="+", default=[0, 8, 16, 32, 64, 128])
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if not os.path.exists(args.slide):
        ap.error(f"slide not found: {args.slide}")
    src = PyramidImageSource(args.slide)
    if args.experiment == "sizes":
        run_sizes(src, args.sizes, args.level, args.json)
    else:
        run_halo(src, args.halo_size, args.halos, args.level, args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
