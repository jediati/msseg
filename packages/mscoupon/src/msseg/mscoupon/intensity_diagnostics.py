#!/usr/bin/env python3

"""
intensity_diagnostics.py

Diagnostic tool for a single CT reconstruction TIFF.

It:

1. Loads one TIFF.
2. Excludes no-data and non-finite pixels.  [--omit-value V / --keep-all]
3. Optionally random-downsamples valid pixels.
4. Fits a two-component Gaussian mixture (msseg.mscoupon.normalize.gmm, whose
   "measure" preset is the quantile-seeded reg_covar=1e-12 fit this script used
   to build with scikit-learn directly).
5. Prints:
       raw nonzero percentiles
       fitted GMM means / sigmas / weights
       hard-assigned class mean / median / percentiles
6. Repeats the fit after optional percentile trimming.
7. Writes a log-y histogram with:
       raw sampled intensity histogram
       fitted total GMM density
       fitted component densities
       GMM means
       hard-assigned medians

Usage
-----

Basic:

    python intensity_diagnostics.py recon_00200.tiff

100x downsampling:

    python intensity_diagnostics.py recon_00200.tiff --downsample 100

Also test a 0.5%-99.5% trimmed fit:

    python intensity_diagnostics.py recon_00200.tiff \
        --downsample 100 \
        --trim 0.5

Choose output:

    python intensity_diagnostics.py recon_00200.tiff \
        --output diagnostic_00200.png

Dependencies:

    pip install numpy tifffile matplotlib
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from msseg.mscoupon.normalize import (count_zeros, fit_two_gaussians, get_valid_pixels,
                                      random_downsample, read_tiff, trim_pixels)

N_COMPONENTS = 2


def fit(x, n_init):
    """Fit the mixture to pixels that are already masked and subsampled."""
    return fit_two_gaussians(x, omit_zeros=False, n_init=n_init)


def hard_assign(x, result):
    """Most-likely component (0-based) for each sample.

    The library reports summary statistics per class but not the memberships
    themselves, and the diagnostics want each class's full percentile ladder.
    """
    log_density = np.column_stack([
        np.log(result[f"weight_{k}"]) - np.log(result[f"sigma_{k}"])
        - 0.5 * ((x - result[f"mu_{k}"]) / result[f"sigma_{k}"]) ** 2
        for k in range(1, N_COMPONENTS + 1)
    ])
    return np.argmax(log_density, axis=1)


def print_percentiles(label, x):
    percentiles = [0, 0.01, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 99.99, 100]
    values = np.percentile(x, percentiles)

    print()
    print(label)

    for p, value in zip(percentiles, values):
        print(f"  p{p:7g}: {value:.12g}")


def print_gmm_diagnostics(label, x, result):
    print()
    print("=" * 72)
    print(label)
    print("=" * 72)

    print()
    print("GMM fit:")
    print(f"  converged: {result['converged']}")
    print(f"  iterations: {result['n_iter']}")

    labels = hard_assign(x, result)

    for k in range(1, N_COMPONENTS + 1):

        print()
        print(f"Component {k}")

        print(f"  GMM mean:   {result[f'mu_{k}']:.12g}")
        print(f"  GMM sigma:  {result[f'sigma_{k}']:.12g}")
        print(f"  GMM weight: {result[f'weight_{k}']:.12g}")

        print(f"  hard n:     {result[f'n_class_{k}']}")

        xp = x[labels == k - 1]

        if xp.size == 0:
            continue

        hard_mean = result[f"hard_mean_{k}"]
        median = result[f"median_{k}"]

        print(f"  hard mean:  {hard_mean:.12g}")
        print(f"  median:     {median:.12g}")
        print(f"  mean-minus-median: {hard_mean - median:.12g}")
        print(f"  GMM-mean minus hard-mean: {result[f'mu_{k}'] - hard_mean:.12g}")

        ps = [0, 1, 5, 25, 50, 75, 95, 99, 100]
        vals = np.percentile(xp, ps)

        print("  hard-assigned percentiles:")

        for p, value in zip(ps, vals):
            print(f"    p{p:3d}: {value:.12g}")


def normal_pdf(x, mu, sigma):
    return (
        np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        / (sigma * np.sqrt(2.0 * np.pi))
    )


def make_plot(x, result, output_path, bins=1000, trim_result=None, include_zeros=False):
    """Plot histogram in density units with log y-scale."""

    fig, ax = plt.subplots(figsize=(13, 7))

    # Use a robust display range so a few extreme outliers
    # don't make the central peaks microscopic.
    display_lo, display_hi = np.percentile(x, [0.05, 99.95])

    hist_x = x[(x >= display_lo) & (x <= display_hi)]

    ax.hist(
        hist_x,
        bins=bins,
        density=True,
        alpha=0.35,
        label="Nonzero sampled pixels",
    )

    grid = np.linspace(display_lo, display_hi, 5000)

    component_pdfs = []

    for k in range(1, N_COMPONENTS + 1):

        component = result[f"weight_{k}"] * normal_pdf(
            grid, result[f"mu_{k}"], result[f"sigma_{k}"])

        component_pdfs.append(component)

        ax.plot(grid, component, linewidth=1.5, label=f"GMM component {k}")

    ax.plot(grid, sum(component_pdfs), linewidth=2.0, label="Total GMM")

    for k in range(1, N_COMPONENTS + 1):

        ax.axvline(
            result[f"mu_{k}"],
            linestyle="-",
            linewidth=1.0,
            label=f"C{k} GMM mean {result[f'mu_{k}']:.3g}",
        )

        ax.axvline(
            result[f"median_{k}"],
            linestyle="--",
            linewidth=1.0,
            label=f"C{k} hard median {result[f'median_{k}']:.3g}",
        )

    if trim_result is not None:

        for k in range(1, N_COMPONENTS + 1):

            ax.axvline(
                trim_result[f"mu_{k}"],
                linestyle=":",
                linewidth=1.5,
                label=f"Trimmed C{k} mean {trim_result[f'mu_{k}']:.3g}",
            )

    ax.set_yscale("log")

    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density (log scale)")

    ax.set_title(
        "Intensity histogram and 2-Gaussian mixture fit\n"
        + ("exact-zero pixels included" if include_zeros else "exact-zero pixels omitted")
    )

    ax.grid(alpha=0.2)
    ax.legend(fontsize="small", loc="best")

    fig.tight_layout()

    fig.savefig(output_path, dpi=250, bbox_inches="tight")

    plt.close(fig)

    print()
    print(f"Wrote diagnostic plot: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Diagnose two-Gaussian fitting of a CT intensity distribution.")
    parser.add_argument("tiff", type=Path, help="Input TIFF image.")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Random downsample factor after zero removal. "
                             "Default: 1 (all nonzero pixels).")
    parser.add_argument("--trim", type=float, default=None,
                        help="Optional symmetric percentile trimming for a second "
                             "diagnostic fit. Example: --trim 0.5 keeps the "
                             "0.5th through 99.5th percentile.")
    parser.add_argument("--bins", type=int, default=1000,
                        help="Histogram bins for diagnostic plot (default: 1000).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random sampling seed (default: 0).")
    parser.add_argument("--n-init", type=int, default=3,
                        help="GMM initializations (default: 3).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG. Default: <input_stem>_intensity_diagnostics.png")
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

    if args.downsample < 1:
        parser.error("--downsample must be >= 1")

    if args.trim is not None and not 0 <= args.trim < 50:
        parser.error("--trim must satisfy 0 <= trim < 50")

    tiff_path = args.tiff.expanduser().resolve()

    if not tiff_path.exists():
        raise FileNotFoundError(tiff_path)

    if args.output is None:
        output_path = tiff_path.parent / (tiff_path.stem + "_intensity_diagnostics.png")
    else:
        output_path = args.output.expanduser().resolve()

    print(f"Loading: {tiff_path}")

    image = read_tiff(tiff_path)

    n_total = image.size
    n_zero = count_zeros(image)

    pixels = get_valid_pixels(image, omit_value=None if (args.keep_all or args.include_zeros) else args.omit_value)

    print()
    print(f"Total pixels:        {n_total:,}")
    print(f"Exact-zero pixels:   {n_zero:,}")
    print(f"Valid nonzero pixels:{pixels.size:>12,}")
    print(f"Zero fraction:       {n_zero / n_total:.6f}")

    rng = np.random.default_rng(args.seed)

    x = random_downsample(pixels, args.downsample, rng)

    print(f"Sampled pixels:      {x.size:,}")

    print_percentiles("Raw nonzero sampled percentiles:", x)

    result = fit(x, args.n_init)

    print_gmm_diagnostics("UNTRIMMED GMM", x, result)

    trim_result = None

    if args.trim is not None:

        x_trim, lo, hi = trim_pixels(x, args.trim)

        print()
        print(f"Trim diagnostic: keeping {args.trim:g}th through "
              f"{100.0 - args.trim:g}th percentile")
        print(f"Trim limits: {lo:.12g} .. {hi:.12g}")
        print(f"Trimmed n: {x_trim.size:,} / {x.size:,}")

        trim_result = fit(x_trim, args.n_init)

        print_gmm_diagnostics("TRIMMED GMM", x_trim, trim_result)

    make_plot(
        x=x,
        result=result,
        output_path=output_path,
        bins=args.bins,
        trim_result=trim_result,
        include_zeros=args.include_zeros,
    )


if __name__ == "__main__":
    main()
