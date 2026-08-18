#!/usr/bin/env python3

"""
calculate_void_probability.py

For every TIFF in a folder (the fit and the posterior both live in
msseg.mscoupon.normalize.gmm; this file is just the CLI):

1. Remove no-data and non-finite pixels.   [--omit-value V / --keep-all]
2. Randomly downsample valid pixels by the requested factor.
3. Fit a 2-component Gaussian mixture. Components come back sorted by mean:
       component 1 = void
       component 2 = solid
4. Convert every valid image pixel to its equal-prior probability of belonging
   to the void Gaussian, and write it as a float32 TIFF:

       probability/void_prob_<original_filename>.tif

5. Record the fitted parameters in probability/gauss_2_mix_sub_<factor>.csv.

Masked-out pixels stay 0.0 in the probability image. The version of this script
that inlined the posterior only delivered that for integer input -- its float
branch masked isfinite alone, so float zeros silently got a real probability.
The library masks both dtypes the same way, under --include-zeros control.

Usage:

    python calculate_void_probability.py /path/to/folder 100

Windows example:

    python calculate_void_probability.py \
        "C:\\Users\\jediati\\Desktop\\JEDIATI\\data\\spears\\tomo_sample_2_051_rec" \
        100
"""

import argparse
import sys
from pathlib import Path

import tifffile

from msseg.mscoupon.normalize import (CsvWriter, find_tiff_files, fit_two_gaussians,
                                      read_tiff, void_probability_image)

FIELDNAMES = [
    "filename",
    "mu_void", "sigma_void", "weight_void",
    "mu_solid", "sigma_solid", "weight_solid",
    "n_valid_pixels", "n_sampled_pixels",
]

# The library names its components by rank; this CSV names them by material.
COMPONENT_COLUMNS = {
    "mu_void": "mu_1", "sigma_void": "sigma_1", "weight_void": "weight_1",
    "mu_solid": "mu_2", "sigma_solid": "sigma_2", "weight_solid": "weight_2",
    "n_valid_pixels": "n_valid_pixels", "n_sampled_pixels": "n_sampled_pixels",
}

FAILED_ROW = {name: "ERROR" for name in FIELDNAMES[1:7]}


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fit a two-Gaussian mixture to each TIFF using "
            "randomized downsampling and write an equal-prior "
            "void-probability TIFF."
        )
    )
    parser.add_argument("folder", type=Path, help="Folder containing TIFF images.")
    parser.add_argument("downsample_factor", type=int,
                        help="Random downsampling factor. Example: 100 uses approximately "
                             "1/100 of valid nonzero pixels for fitting.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducible pixel sampling (default: 0).")
    parser.add_argument("--n-init", type=int, default=3,
                        help="Number of mixture initializations (default: 3).")
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
        print(f"ERROR: Not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    # Output goes into a subfolder, so it is not picked up on future runs.
    tiff_files = find_tiff_files(folder)

    if not tiff_files:
        print(f"ERROR: No TIFF files found in {folder}", file=sys.stderr)
        sys.exit(1)

    output_folder = folder / "probability"
    output_folder.mkdir(exist_ok=True)

    factor = args.downsample_factor
    csv_path = output_folder / f"gauss_2_mix_sub_{factor}.csv"
    omit_value = None if (args.keep_all or args.include_zeros) else args.omit_value

    print(f"Found {len(tiff_files)} TIFF files.")
    print(f"Downsample factor: {factor}")
    print(f"Output directory: {output_folder}")
    print()

    with CsvWriter(csv_path, FIELDNAMES) as writer:
        for i, image_path in enumerate(tiff_files):
            print(f"[{i + 1}/{len(tiff_files)}] {image_path.name}", flush=True)

            try:
                image = read_tiff(image_path)

                # Vary the seed per image so each draws its own subsample, as the
                # original did by threading one RNG through the whole folder.
                result = fit_two_gaussians(
                    image,
                    downsample_factor=factor,
                    omit_value=omit_value,
                    n_init=args.n_init,
                    seed=args.seed + i,
                )
                params = {name: result[key] for name, key in COMPONENT_COLUMNS.items()}

                print(f"    void : mu={params['mu_void']:.7g}, "
                      f"sigma={params['sigma_void']:.7g}")
                print(f"    solid: mu={params['mu_solid']:.7g}, "
                      f"sigma={params['sigma_solid']:.7g}")
                print(f"    fitting pixels: {params['n_sampled_pixels']:,} / "
                      f"{params['n_valid_pixels']:,}")

                probability = void_probability_image(
                    image,
                    mu_void=params["mu_void"],
                    sigma_void=params["sigma_void"],
                    mu_solid=params["mu_solid"],
                    sigma_solid=params["sigma_solid"],
                    omit_value=omit_value,
                )

                output_path = output_folder / ("void_prob_" + image_path.name)
                tifffile.imwrite(output_path, probability)

                writer.write({"filename": image_path.name, **params})
                print(f"    wrote: {output_path.name}")

                # Release before loading the next 3000x3000 TIFF.
                del probability
                del image

            except Exception as exc:
                print(f"    ERROR: {exc}", file=sys.stderr)
                writer.write({"filename": image_path.name, **FAILED_ROW})

    print()
    print("Done.")
    print(f"Probability images: {output_folder}")
    print(f"Gaussian parameters: {csv_path}")


if __name__ == "__main__":
    main()
