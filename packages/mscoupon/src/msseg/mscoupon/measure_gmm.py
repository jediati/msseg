#!/usr/bin/env python3

"""
measure_gmm.py

Measure a two-component Gaussian mixture for every TIFF in a folder.

Pipeline (all of it now lives in msseg.mscoupon.normalize.gmm, which calls the
C++ fit_gmm; this file is just the CLI):

1. Remove no-data and non-finite pixels.   [--omit-value V / --keep-all]
2. Randomly downsample the remaining pixels.
3. Optionally trim symmetric intensity tails.
4. Fit a 2-component Gaussian mixture.
5. Sort components by mean:
       component 1 = lower-intensity class / air
       component 2 = higher-intensity class / metal
6. Report the fitted mean plus the empirical median and histogram mode of the
   pixels hard-assigned to each class.

Examples
--------

    python measure_gmm.py "F:\\data\\scan" 100 --samples 200:210 \
        --output gmm_cross_stack.csv

    python measure_gmm.py "F:\\data\\scan" 100 --samples 200:210 --trim 0.5 \
        --output gmm_cross_stack.csv

    python measure_gmm.py "F:\\data\\scan2" 100 --trim 0.5 \
        --output gmm_cross_stack.csv --append
"""

import argparse
import sys
from pathlib import Path

from msseg.mscoupon.normalize import CsvWriter, fit_two_gaussians, iter_images, select_files

FIELDNAMES = [
    "folder", "filename", "image_index",
    "downsample_factor", "trim_percent", "trim_lo", "trim_hi",
    "mu_1", "hard_mean_1", "median_1", "mode_1", "sigma_1", "weight_1", "n_class_1",
    "mu_2", "hard_mean_2", "median_2", "mode_2", "sigma_2", "weight_2", "n_class_2",
    "n_valid_pixels", "n_sampled_pixels", "n_fit_pixels",
    "converged", "n_iter",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Measure trimmed two-component Gaussian mixtures for TIFF "
                    "reconstruction images.")
    parser.add_argument("folder", type=Path,
                        help="Input folder containing .tif or .tiff images.")
    parser.add_argument("downsample_factor", type=int,
                        help="Random downsampling factor after zero removal. "
                             "100 uses about 1/100 of the valid pixels.")
    parser.add_argument("--samples", default=None,
                        help="Optional image subsequence START:END (start inclusive, "
                             "end exclusive). Example: --samples 200:210")
    parser.add_argument("--trim", type=float, default=0.0,
                        help="Symmetric percentile trim applied after downsampling and "
                             "before fitting. 0.5 keeps the 0.5th..99.5th percentiles. "
                             "Default: 0.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--append", action="store_true",
                        help="Append to an existing CSV instead of overwriting it. The "
                             "header is written only if the file is new or empty.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed for reproducible downsampling. Default: 0.")
    parser.add_argument("--n-init", type=int, default=3,
                        help="Number of mixture initializations. Default: 3.")
    parser.add_argument("--mode-bins", type=int, default=512,
                        help="Histogram bins for the empirical mode. Default: 512.")
    parser.add_argument("--omit-value", type=float, default=0.0, metavar="V",
                        help="No-data sentinel dropped before measuring (default 0). "
                             "A stack may pad with any constant; dropping the wrong "
                             "value leaves that plateau in as a spurious population.")
    parser.add_argument("--keep-all", action="store_true",
                        help="Measure every pixel (no no-data sentinel).")
    parser.add_argument("--include-zeros", action="store_true",
                        help="Deprecated alias for --keep-all.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.downsample_factor < 1:
        parser.error("downsample_factor must be >= 1")
    if not 0 <= args.trim < 50:
        parser.error("--trim must satisfy 0 <= trim < 50")
    if args.n_init < 1:
        parser.error("--n-init must be >= 1")
    if args.mode_bins < 10:
        parser.error("--mode-bins must be >= 10")

    try:
        selected, start, _ = select_files(args.folder, args.samples)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    folder = args.folder.expanduser().resolve()
    print(f"Folder:             {folder}")
    print(f"Selected TIFFs:     {len(selected)} (indices {start}:{start + len(selected)})")
    print(f"Downsample factor:  {args.downsample_factor}")
    print(f"Trim percent:       {args.trim:g}% each tail")
    print(f"Zeros:              {'included' if args.include_zeros else 'omitted'}")
    print(f"Output:             {args.output}")
    print(f"Mode:               {'append' if args.append else 'overwrite'}")
    print()

    with CsvWriter(args.output, FIELDNAMES, append=args.append) as writer:
        for local_i, image_index, path, image in iter_images(selected, start):
            print(f"[{local_i + 1}/{len(selected)}] index={image_index} {path.name}", flush=True)

            identity = {"folder": folder.name, "filename": path.name,
                        "image_index": image_index,
                        "downsample_factor": args.downsample_factor}
            try:
                # Vary the seed per slice so each draws its own subsample, as the
                # original did by threading one RNG through the whole folder.
                result = fit_two_gaussians(
                    image,
                    downsample_factor=args.downsample_factor,
                    omit_value=None if (args.keep_all or args.include_zeros) else args.omit_value,
                    trim_percent=args.trim,
                    n_init=args.n_init,
                    mode_bins=args.mode_bins,
                    seed=args.seed + image_index,
                )
            except Exception as exc:
                print(f"    ERROR: {exc}", file=sys.stderr)
                writer.write_blank(**identity, trim_percent=args.trim, converged=False)
                continue

            for n in (1, 2):
                print(f"    G{n}: mean={result[f'mu_{n}']:.7g}, "
                      f"hard_mean={result[f'hard_mean_{n}']:.7g}, "
                      f"median={result[f'median_{n}']:.7g}, "
                      f"mode={result[f'mode_{n}']:.7g}, "
                      f"sigma={result[f'sigma_{n}']:.7g}, "
                      f"weight={result[f'weight_{n}']:.4f}")
            print(f"    pixels: valid={result['n_valid_pixels']:,}, "
                  f"sampled={result['n_sampled_pixels']:,}, fit={result['n_fit_pixels']:,}")
            print(f"    trim: {result['trim_lo']:.7g} .. {result['trim_hi']:.7g}")
            print(f"    converged={result['converged']}, iterations={result['n_iter']}")

            writer.write({**identity, **{k: result[k] for k in FIELDNAMES if k in result}})

    print()
    print("Done.")


if __name__ == "__main__":
    main()
