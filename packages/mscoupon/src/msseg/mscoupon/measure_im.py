#!/usr/bin/env python3

"""
measure_im.py

Histogram-based intensity measurement for CT reconstruction TIFFs.

For each image (the work happens in msseg.mscoupon.normalize.histogram, which
calls the C++ measure_histogram; this file is just the CLI):

1. Remove no-data and non-finite pixels.   [--omit-value V / --keep-all]
2. Randomly downsample the remaining pixels.
3. Measure min, max, and the percentile ladder.
4. Build a histogram between the p1 and p99 cut points.
5. Smooth it with a moving average.
6. Find the two strongest separated local maxima.
7. Refine each peak position by parabolic interpolation.

peak_1 is the lower-intensity peak, peak_2 the higher one -- ordered by
intensity, not by height.

Examples
--------

    python measure_im.py "F:\\data\\spears\\tomo_sample_2_051_rec" 100 \
        --samples 200:210 --output im_cross_stack.csv

    python measure_im.py "F:\\data\\spears\\another_substack" 100 \
        --samples 200:210 --output im_cross_stack.csv --append
"""

import argparse
import sys
from pathlib import Path

from msseg.mscoupon.normalize import (PERCENTILE_NAMES, CsvWriter, iter_images,
                                      measure_histogram, select_files)

# The CSV keeps the historical peak_1/peak_2 spelling; the library returns
# peak_low/peak_high, which says the ordering rule out loud.
PEAK_COLUMNS = {
    "peak_1": "peak_low", "peak_2": "peak_high",
    "peak_1_bin": "peak_low_bin", "peak_2_bin": "peak_high_bin",
    "peak_1_height": "peak_low_height", "peak_2_height": "peak_high_height",
}

FIELDNAMES = [
    "folder", "filename", "image_index", "downsample_factor",
    "n_total_pixels", "n_zero_pixels", "n_valid_pixels", "n_sampled_pixels",
    "min", *PERCENTILE_NAMES, "max",
    "hist_lo", "hist_hi",
    "peak_1", "peak_2", "peak_1_bin", "peak_2_bin", "peak_1_height", "peak_2_height",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Measure basic CT image statistics and locate two histogram maxima.")
    parser.add_argument("folder", type=Path, help="Folder containing TIFF images.")
    parser.add_argument("downsample_factor", type=int,
                        help="Random downsampling factor after zero removal. "
                             "100 uses about 1/100 of the valid pixels.")
    parser.add_argument("--samples", default=None,
                        help="Optional START:END subsequence. Example: --samples 200:210")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV.")
    parser.add_argument("--append", action="store_true",
                        help="Append instead of overwriting the output CSV.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random sampling seed. Default: 0.")
    parser.add_argument("--bins", type=int, default=1024,
                        help="Histogram bins. Default: 1024.")
    parser.add_argument("--smooth", type=int, default=11,
                        help="Moving-average smoothing width in bins. Default: 11.")
    parser.add_argument("--peak-window", type=int, default=32,
                        help="Radius in bins defining a stable local maximum. Default: 32.")
    parser.add_argument("--min-peak-distance", type=int, default=64,
                        help="Minimum separation between the selected peaks. Default: 64.")
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
    if args.bins < 32:
        parser.error("--bins must be >= 32")

    try:
        selected, start, _ = select_files(args.folder, args.samples)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    folder = args.folder.expanduser().resolve()
    print(f"Folder:            {folder}")
    print(f"Images:            {start}:{start + len(selected)}")
    print(f"Downsample factor: {args.downsample_factor}")
    print(f"Histogram bins:    {args.bins}")
    print(f"Smoothing width:   {args.smooth}")
    print(f"Peak window:       +/- {args.peak_window} bins")
    print(f"Minimum distance:  {args.min_peak_distance} bins")
    print(f"Zeros:             {'included' if args.include_zeros else 'omitted'}")
    print(f"Output:            {args.output}")
    print()

    with CsvWriter(args.output, FIELDNAMES, append=args.append) as writer:
        for local_i, image_index, path, image in iter_images(selected, start):
            print(f"[{local_i + 1}/{len(selected)}] {path.name}", flush=True)

            identity = {"folder": folder.name, "filename": path.name,
                        "image_index": image_index,
                        "downsample_factor": args.downsample_factor}
            try:
                result = measure_histogram(
                    image,
                    downsample_factor=args.downsample_factor,
                    omit_value=None if (args.keep_all or args.include_zeros) else args.omit_value,
                    bins=args.bins,
                    smooth_width=args.smooth,
                    peak_window=args.peak_window,
                    min_peak_distance=args.min_peak_distance,
                    seed=args.seed + image_index,
                )
            except Exception as exc:
                print(f"    ERROR: {exc}", file=sys.stderr)
                writer.write_blank(**identity)
                continue

            row = {**identity}
            for column in FIELDNAMES:
                if column in identity:
                    continue
                key = PEAK_COLUMNS.get(column, column)
                if key in result:
                    row[column] = result[key]

            print(f"    range: {result['min']:.7g} .. {result['max']:.7g}")
            print(f"    hist:  {result['hist_lo']:.7g} .. {result['hist_hi']:.7g}")
            print(f"    peaks: {result['peak_low']:.7g}, {result['peak_high']:.7g}")

            writer.write(row)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
