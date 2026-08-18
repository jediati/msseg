#!/usr/bin/env python3

"""
calculate_2_gaussian_mixture.py

Fit a two-component Gaussian mixture to every TIFF in a folder and write

    gauss_2_csv_sub_<factor>.csv

into that same folder.

Per image (the fit lives in msseg.mscoupon.normalize.gmm, which calls the C++
fit_gmm; this file is just the CLI):

1. Remove no-data and non-finite pixels.   [--omit-value V / --keep-all]
2. Randomly downsample the remaining valid pixels.
3. Fit a 2-component mixture and sort the components by increasing mean.

The "two_gaussian" preset is the port of what this script used to do with
scikit-learn directly: k-means seeding, max_iter=300, tol=1e-6, no trimming.

Usage:
    python calculate_2_gaussian_mixture.py /path/to/folder 100

Here, a downsample factor of 100 means approximately 1/100
of the valid nonzero pixels are used for the Gaussian fit.
"""

import argparse
import sys
from pathlib import Path

from msseg.mscoupon.normalize import (PRESET_TWO_GAUSSIAN, CsvWriter, find_tiff_files,
                                      fit_two_gaussians, read_tiff)

FIELDNAMES = [
    "filename",
    "mu_1", "sigma_1", "weight_1",
    "mu_2", "sigma_2", "weight_2",
    "n_valid_pixels", "n_sampled_pixels",
]

# compare_gaussian_mixtures.py filters rows on this sentinel, so a failed image
# is recorded as ERROR rather than as a blank row.
FAILED_ROW = {name: "ERROR" for name in FIELDNAMES[1:7]}


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fit a two-component Gaussian mixture to a "
            "random subset of the nonzero pixels in every TIFF."
        )
    )
    parser.add_argument("folder", type=Path, help="Folder containing TIFF images.")
    parser.add_argument("downsample_factor", type=int,
                        help="Random downsampling factor. Example: 100 uses approximately "
                             "1/100 of valid pixels.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducible sampling (default: 0).")
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

    folder = args.folder.expanduser().resolve()

    if not folder.exists():
        print(f"ERROR: Folder does not exist: {folder}", file=sys.stderr)
        sys.exit(1)

    if not folder.is_dir():
        print(f"ERROR: Path is not a folder: {folder}", file=sys.stderr)
        sys.exit(1)

    tiff_files = find_tiff_files(folder)

    if not tiff_files:
        print(f"ERROR: No TIFF files found in: {folder}", file=sys.stderr)
        sys.exit(1)

    factor = args.downsample_factor
    output_file = folder / f"gauss_2_csv_sub_{factor}.csv"

    print(f"Found {len(tiff_files)} TIFF files.")
    print(f"Downsample factor: {factor}")
    print(f"Random seed: {args.seed}")
    print(f"Output: {output_file}")
    print()

    with CsvWriter(output_file, FIELDNAMES) as writer:
        for i, image_path in enumerate(tiff_files):
            print(f"[{i + 1}/{len(tiff_files)}] {image_path.name}", flush=True)

            try:
                # Vary the seed per image so each draws its own subsample, as the
                # original did by threading one RNG through the whole folder.
                result = fit_two_gaussians(
                    read_tiff(image_path),
                    downsample_factor=factor,
                    omit_value=None if (args.keep_all or args.include_zeros) else args.omit_value,
                    preset=PRESET_TWO_GAUSSIAN,
                    seed=args.seed + i,
                )
            except Exception as exc:
                print(f"    ERROR: {exc}", file=sys.stderr)
                writer.write({"filename": image_path.name, **FAILED_ROW})
                continue

            writer.write({"filename": image_path.name,
                          **{k: result[k] for k in FIELDNAMES if k in result}})

            print(f"    valid={result['n_valid_pixels']:,}, "
                  f"sampled={result['n_sampled_pixels']:,}")
            for n in (1, 2):
                print(f"    G{n}: mu={result[f'mu_{n}']:.6g}, "
                      f"sigma={result[f'sigma_{n}']:.6g}, "
                      f"weight={result[f'weight_{n}']:.4f}")

    print()
    print(f"Done: {output_file}")


if __name__ == "__main__":
    main()
