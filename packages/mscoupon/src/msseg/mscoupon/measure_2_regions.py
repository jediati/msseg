#!/usr/bin/env python3

"""
measure_2_regions.py

Measure intensity statistics in two fixed rectangular regions for every TIFF in
a folder. The manual counterpart to measure_gmm.py / measure_im.py: instead of
inferring the two populations, you name a rectangle in air and one in metal.

The measurement lives in msseg.mscoupon.normalize.regions (which calls the C++
measure_regions); this file is just the CLI.

Example
-------

python measure_2_regions.py \
    "F:\\data\\spears\\tomo_sample_2_051_rec" \
    --air 250:350 740:840 \
    --metal 600:1200 1400:1600 \
    --output region_stats.csv

Coordinate convention
---------------------

Each region takes ROWS COLUMNS, so

    --air 250:350 740:840

means image[250:350, 740:840]: the first range is Y / rows, the second is
X / columns.

Exact zero values are INCLUDED by default, since these are explicitly chosen
physical regions -- unlike the GMM and histogram measures, which omit them.
Use --exclude-zero to drop them.
"""

import argparse
import sys
from pathlib import Path

from msseg.mscoupon.normalize import (PERCENTILE_NAMES, CsvWriter, iter_images, parse_range,
                                      measure_two_regions, prefixed_stats, select_files)

REGION_STAT_COLUMNS = ["n_pixels", "min", "max", "mean", "std", *PERCENTILE_NAMES]

FIELDNAMES = [
    "folder", "filename", "image_index",
    "air_rows", "air_cols", "metal_rows", "metal_cols",
    *[f"air_{name}" for name in REGION_STAT_COLUMNS],
    *[f"metal_{name}" for name in REGION_STAT_COLUMNS],
]


def rect_arg(parser, text):
    try:
        return parse_range(text)
    except ValueError as exc:
        parser.error(str(exc))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Measure fixed air and metal rectangular regions across a TIFF stack.")
    parser.add_argument("folder", type=Path, help="Folder containing TIFF images.")
    parser.add_argument("--air", nargs=2, required=True, metavar=("ROWS", "COLS"),
                        help="Air rectangle as ROW_START:ROW_END COL_START:COL_END. "
                             "Example: --air 250:350 740:840")
    parser.add_argument("--metal", nargs=2, required=True, metavar=("ROWS", "COLS"),
                        help="Metal rectangle as ROW_START:ROW_END COL_START:COL_END. "
                             "Example: --metal 600:1200 1400:1600")
    parser.add_argument("--samples", default=None,
                        help="Optional image range START:END. Example: --samples 200:210")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV.")
    parser.add_argument("--append", action="store_true",
                        help="Append instead of overwriting the CSV.")
    parser.add_argument("--omit-value", type=float, default=None, metavar="V",
                        help="Drop this no-data value inside the rectangles "
                             "(default: keep every pixel -- these are hand-picked "
                             "physical regions, so a zero there is a measurement).")
    parser.add_argument("--exclude-zero", action="store_true",
                        help="Exclude exact-zero pixels from the ROI statistics. "
                             "By default zeros are included.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    air_rows = rect_arg(parser, args.air[0])
    air_cols = rect_arg(parser, args.air[1])
    metal_rows = rect_arg(parser, args.metal[0])
    metal_cols = rect_arg(parser, args.metal[1])

    try:
        selected, start, _ = select_files(args.folder, args.samples)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    folder = args.folder.expanduser().resolve()
    rect_labels = {
        "air_rows": f"{air_rows[0]}:{air_rows[1]}",
        "air_cols": f"{air_cols[0]}:{air_cols[1]}",
        "metal_rows": f"{metal_rows[0]}:{metal_rows[1]}",
        "metal_cols": f"{metal_cols[0]}:{metal_cols[1]}",
    }

    print(f"Folder:       {folder}")
    print(f"Air ROI:      rows {rect_labels['air_rows']}, cols {rect_labels['air_cols']}")
    print(f"Metal ROI:    rows {rect_labels['metal_rows']}, cols {rect_labels['metal_cols']}")
    print(f"Images:       {start}:{start + len(selected)}")
    print(f"Zeros:        {'excluded' if args.exclude_zero else 'included'}")
    print(f"Output:       {args.output}")
    print()

    with CsvWriter(args.output, FIELDNAMES, append=args.append) as writer:
        for local_i, image_index, path, image in iter_images(selected, start):
            print(f"[{local_i + 1}/{len(selected)}] {path.name}", flush=True)

            identity = {"folder": folder.name, "filename": path.name,
                        "image_index": image_index, **rect_labels}
            try:
                stats = measure_two_regions(image, air_rows, air_cols, metal_rows, metal_cols,
                                            omit_value=(0.0 if args.exclude_zero else args.omit_value))
            except Exception as exc:
                print(f"    ERROR: {exc}", file=sys.stderr)
                writer.write_blank(**identity)
                continue

            row = {**identity}
            row.update(prefixed_stats("air", stats["air"]))
            row.update(prefixed_stats("metal", stats["metal"]))
            writer.write(row)

            for name in ("air", "metal"):
                s = stats[name]
                print(f"    {name + ':':7s}median={s['p50_0']:.7g}, mean={s['mean']:.7g}, "
                      f"p5-p95={s['p5_0']:.7g} .. {s['p95_0']:.7g}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
