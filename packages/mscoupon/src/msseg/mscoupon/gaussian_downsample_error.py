#!/usr/bin/env python3

"""
gaussian_downsample_error.py

Monte Carlo study of Gaussian-mixture fitting accuracy and runtime
under randomized pixel downsampling.

For randomly selected TIFF images:

1. Use gauss_2_mix.csv as the full-data reference.
2. Load each TIFF once.
3. Remove zero / NaN / Inf pixels once.
4. For each requested downsampling factor:
       - perform repeated independent random subsampling
       - fit a two-component Gaussian mixture
       - time each randomized fit
       - compare fitted parameters to full-data reference
5. Save:
       - every individual Monte Carlo fit
       - per-image/per-factor summary statistics
       - error boxplots
       - timing plots
       - combined two-axis spaghetti plot

Example:

    python gaussian_downsample_error.py \
        "C:\\Users\\jediati\\Desktop\\JEDIATI\\data\\spears\\tomo_sample_2_051_rec" \
        --n-images 5 \
        --spec 100:10 300:25 1000:100 3000:200

The specification:

    100:10

means:

    downsampling factor = 100
    number of independent randomized fits = 10

Dependencies:

    pip install numpy pandas matplotlib tifffile
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from msseg.mscoupon.normalize import (PARAMETERS, find_tiff_files, fit_two_gaussians,
                                      get_valid_pixels, load_reference_csv, read_tiff)


# ----------------------------------------------------------------------
# Command-line specification parsing
# ----------------------------------------------------------------------

def parse_spec(spec_strings):
    """
    Convert:

        100:10 1000:100

    into:

        [
            (100, 10),
            (1000, 100)
        ]
    """

    result = []

    for text in spec_strings:

        try:
            factor_text, repeats_text = text.split(":")

            factor = int(factor_text)
            repeats = int(repeats_text)

        except Exception:
            raise ValueError(
                f"Invalid specification '{text}'. "
                f"Use FACTOR:REPEATS, e.g. 100:10"
            )

        if factor < 1:
            raise ValueError(
                f"Factor must be >= 1: {text}"
            )

        if repeats < 1:
            raise ValueError(
                f"Repeats must be >= 1: {text}"
            )

        result.append(
            (factor, repeats)
        )

    # Sort by downsampling factor.
    result.sort(
        key=lambda x: x[0]
    )

    return result


# ----------------------------------------------------------------------
# Image selection
# ----------------------------------------------------------------------

def select_images(
    folder,
    reference_df,
    n_images,
    rng,
):
    """
    Select random TIFFs for which valid full-data reference
    fits exist.
    """

    tiff_files = find_tiff_files(folder)

    reference_names = set(
        reference_df["filename"]
    )

    candidates = [
        path
        for path in tiff_files
        if path.name in reference_names
    ]

    if not candidates:
        raise RuntimeError(
            "No TIFFs have valid entries in gauss_2_mix.csv"
        )

    n_select = min(
        n_images,
        len(candidates),
    )

    indices = rng.choice(
        len(candidates),
        size=n_select,
        replace=False,
    )

    return [
        candidates[i]
        for i in indices
    ]


# ----------------------------------------------------------------------
# Error plots
# ----------------------------------------------------------------------

def make_error_boxplot(
    subset,
    parameter,
    filename,
    output_folder,
):
    """
    Boxplot of signed parameter error for one image.
    """

    error_column = f"error_{parameter}"

    factors = sorted(
        subset["downsample_factor"].unique()
    )

    data = []
    labels = []

    for factor in factors:

        values = subset.loc[
            subset["downsample_factor"] == factor,
            error_column,
        ].dropna()

        if len(values) == 0:
            continue

        data.append(
            values.values
        )

        labels.append(
            f"1/{factor}"
        )

    if not data:
        return

    plt.figure(
        figsize=(9, 6)
    )

    plt.boxplot(
        data,
        tick_labels=labels,
        showmeans=True,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Pixel sampling fraction"
    )

    plt.ylabel(
        f"{parameter} estimate - full-data estimate"
    )

    plt.title(
        f"{filename}\nError in {parameter}"
    )

    plt.tight_layout()

    safe_filename = (
        Path(filename)
        .stem
        .replace(" ", "_")
    )

    output_path = (
        output_folder
        / f"{safe_filename}_error_{parameter}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ----------------------------------------------------------------------
# Timing plot for an individual image
# ----------------------------------------------------------------------

def make_timing_plot(
    subset,
    filename,
    output_folder,
):
    """
    Plot mean fitting time versus downsampling factor
    for a single image.
    """

    timing = (
        subset
        .groupby("downsample_factor")["fit_time_ms"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("downsample_factor")
    )

    x = timing["downsample_factor"].values
    y = timing["mean"].values
    yerr = timing["std"].fillna(0).values

    plt.figure(
        figsize=(8, 6)
    )

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=4,
    )

    plt.xscale("log")

    plt.xlabel(
        "Downsampling factor"
    )

    plt.ylabel(
        "Fit time (ms)"
    )

    plt.title(
        f"{filename}\nGaussian-mixture fitting time"
    )

    plt.tight_layout()

    safe_filename = (
        Path(filename)
        .stem
        .replace(" ", "_")
    )

    output_path = (
        output_folder
        / f"{safe_filename}_timing.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ----------------------------------------------------------------------
# Summary statistics
# ----------------------------------------------------------------------

def make_summary(results):
    """
    Compute per-image / per-factor statistics.

    Includes:

        average fit time
        fit-time SD

    plus standard parameter-error statistics.
    """

    rows = []

    grouped = results.groupby(
        [
            "filename",
            "downsample_factor",
        ]
    )

    for (
        filename,
        factor,
    ), group in grouped:

        timing_mean = (
            group["fit_time_ms"].mean()
        )

        timing_sd = (
            group["fit_time_ms"].std(ddof=1)
        )

        timing_min = (
            group["fit_time_ms"].min()
        )

        timing_max = (
            group["fit_time_ms"].max()
        )

        # ----------------------------------------------------------
        # Worst relative error observed across ALL:
        #
        #   - repeats
        #   - mu_1
        #   - sigma_1
        #   - mu_2
        #   - sigma_2
        #
        # expressed as percent.
        # ----------------------------------------------------------

        relative_error_columns = [
            f"relative_error_pct_{p}"
            for p in PARAMETERS
        ]

        worst_case_relative_error_pct = (
            group[
                relative_error_columns
            ]
            .abs()
            .to_numpy()
            .max()
        )

        for parameter in PARAMETERS:

            error_col = (
                f"error_{parameter}"
            )

            relative_col = (
                f"relative_error_pct_{parameter}"
            )

            values = (
                group[error_col]
                .dropna()
            )

            relative_values = (
                group[relative_col]
                .dropna()
            )

            if len(values) == 0:
                continue

            rows.append(
                {
                    "filename":
                        filename,

                    "downsample_factor":
                        factor,

                    "parameter":
                        parameter,

                    "n_repeats":
                        len(values),

                    "mean_error":
                        values.mean(),

                    "sd_error":
                        values.std(ddof=1),

                    "median_error":
                        values.median(),

                    "mean_absolute_error":
                        np.mean(
                            np.abs(values)
                        ),

                    "rmse":
                        np.sqrt(
                            np.mean(
                                values ** 2
                            )
                        ),

                    "mean_absolute_relative_error_pct":
                        np.mean(
                            np.abs(relative_values)
                        ),

                    "max_absolute_relative_error_pct":
                        np.max(
                            np.abs(relative_values)
                        ),

                    "q025":
                        values.quantile(0.025),

                    "q25":
                        values.quantile(0.25),

                    "q75":
                        values.quantile(0.75),

                    "q975":
                        values.quantile(0.975),

                    "mean_fit_time_ms":
                        timing_mean,

                    "sd_fit_time_ms":
                        timing_sd,

                    "min_fit_time_ms":
                        timing_min,

                    "max_fit_time_ms":
                        timing_max,

                    "worst_case_relative_error_pct":
                        worst_case_relative_error_pct,
                }
            )

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Compact image/factor summary
# ----------------------------------------------------------------------

def make_factor_summary(results):
    """
    One row per image / downsampling factor.

    Stores raw fitted mean values and timing statistics.
    """

    rows = []

    for (
        filename,
        factor,
    ), group in results.groupby(
        [
            "filename",
            "downsample_factor",
        ]
    ):

        row = {
            "filename": filename,
            "downsample_factor": factor,
            "n_repeats": len(group),

            "n_sampled_pixels":
                group["n_sampled_pixels"].iloc[0],

            # Timing
            "mean_fit_time_ms":
                group["fit_time_ms"].mean(),

            "sd_fit_time_ms":
                group["fit_time_ms"].std(ddof=1),

            "min_fit_time_ms":
                group["fit_time_ms"].min(),

            "max_fit_time_ms":
                group["fit_time_ms"].max(),
        }

        # ------------------------------------------------------
        # Raw fitted values for the two Gaussian means.
        # ------------------------------------------------------

        for parameter in ["mu_1", "mu_2"]:

            values = group[
                f"estimate_{parameter}"
            ]

            reference = group[
                f"reference_{parameter}"
            ].iloc[0]

            row[
                f"reference_{parameter}"
            ] = reference

            row[
                f"mean_{parameter}"
            ] = values.mean()

            row[
                f"sd_{parameter}"
            ] = values.std(ddof=1)

            row[
                f"min_{parameter}"
            ] = values.min()

            row[
                f"max_{parameter}"
            ] = values.max()

            # Raw-value deviations from the reference.
            errors = (
                values - reference
            )

            row[
                f"mean_error_{parameter}"
            ] = errors.mean()

            row[
                f"max_abs_error_{parameter}"
            ] = np.abs(errors).max()

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "filename",
                "downsample_factor",
            ]
        )
    )


def make_means_spaghetti_plot(
    factor_summary,
    output_folder,
):
    """
    Plot fitted mu_1 and mu_2 versus downsampling factor.

    For every image:

        mu_1 = one line
        mu_2 = one line

    Each point is the mean across randomized repeats.

    A shaded envelope shows the full observed min-to-max
    range across those repeats.

    The full-data reference values are shown at factor=1
    and as horizontal dashed reference lines.
    """

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    filenames = (
        factor_summary["filename"]
        .unique()
    )

    for filename in filenames:

        subset = (
            factor_summary[
                factor_summary["filename"]
                == filename
            ]
            .sort_values(
                "downsample_factor"
            )
        )

        x = (
            subset[
                "downsample_factor"
            ].values
        )

        # ======================================================
        # MU 1
        # ======================================================

        mu1_mean = (
            subset["mean_mu_1"].values
        )

        mu1_min = (
            subset["min_mu_1"].values
        )

        mu1_max = (
            subset["max_mu_1"].values
        )

        ref_mu1 = (
            subset[
                "reference_mu_1"
            ].iloc[0]
        )

        line1, = ax.plot(
            x,
            mu1_mean,
            marker="o",
            linestyle="-",
            label=f"{Path(filename).stem}  mu1",
        )

        color = line1.get_color()

        ax.fill_between(
            x,
            mu1_min,
            mu1_max,
            alpha=0.12,
            color=color,
        )

        # ======================================================
        # MU 2
        # ======================================================

        mu2_mean = (
            subset["mean_mu_2"].values
        )

        mu2_min = (
            subset["min_mu_2"].values
        )

        mu2_max = (
            subset["max_mu_2"].values
        )

        ref_mu2 = (
            subset[
                "reference_mu_2"
            ].iloc[0]
        )

        ax.plot(
            x,
            mu2_mean,
            marker="s",
            linestyle="--",
            color=color,
            label=f"{Path(filename).stem}  mu2",
        )

        ax.fill_between(
            x,
            mu2_min,
            mu2_max,
            alpha=0.12,
            color=color,
        )

        # ------------------------------------------------------
        # Full-data references.
        #
        # Same image color, faint dotted lines.
        # ------------------------------------------------------

        ax.axhline(
            ref_mu1,
            linestyle=":",
            linewidth=0.8,
            alpha=0.45,
            color=color,
        )

        ax.axhline(
            ref_mu2,
            linestyle=":",
            linewidth=0.8,
            alpha=0.45,
            color=color,
        )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        "Downsampling factor"
    )

    ax.set_ylabel(
        "Gaussian mean (raw image value)"
    )

    ax.set_title(
        "Gaussian mixture means vs randomized downsampling\n"
        "solid/circle = mu1, dashed/square = mu2; "
        "shading = min-max over repeats"
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        fontsize="small",
        ncol=2,
    )

    fig.tight_layout()

    output_path = (
        output_folder
        / "downsampling_gaussian_means.png"
    )

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Wrote {output_path.name}"
    )
def make_timing_spaghetti_plot(
    factor_summary,
    output_folder,
):
    """
    Plot fitting time versus downsampling factor.

    One line per image.

    Points show mean timing across repeats.
    Shaded area shows +/- one standard deviation.
    """

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    filenames = (
        factor_summary[
            "filename"
        ]
        .unique()
    )

    for filename in filenames:

        subset = (
            factor_summary[
                factor_summary["filename"]
                == filename
            ]
            .sort_values(
                "downsample_factor"
            )
        )

        x = (
            subset[
                "downsample_factor"
            ].values
        )

        mean_time = (
            subset[
                "mean_fit_time_ms"
            ].values
        )

        sd_time = (
            subset[
                "sd_fit_time_ms"
            ]
            .fillna(0)
            .values
        )

        line, = ax.plot(
            x,
            mean_time,
            marker="o",
            label=Path(filename).stem,
        )

        color = line.get_color()

        lower = np.maximum(
            mean_time - sd_time,
            0,
        )

        upper = (
            mean_time + sd_time
        )

        ax.fill_between(
            x,
            lower,
            upper,
            alpha=0.15,
            color=color,
        )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Downsampling factor"
    )

    ax.set_ylabel(
        "Mean fitting time (ms)"
    )

    ax.set_title(
        "Gaussian-mixture fitting time vs downsampling\n"
        "shading = +/- 1 SD over repeated fits"
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        title="Image",
        fontsize="small",
    )

    fig.tight_layout()

    output_path = (
        output_folder
        / "downsampling_timing.png"
    )

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Wrote {output_path.name}"
    )

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Monte Carlo analysis of Gaussian-mixture "
            "accuracy and runtime under random downsampling."
        )
    )

    parser.add_argument(
        "folder",
        type=Path,
        help=(
            "Folder containing TIFFs and "
            "gauss_2_mix.csv"
        ),
    )

    parser.add_argument(
        "--n-images",
        type=int,
        default=5,
        help=(
            "Number of random TIFF images to study "
            "(default: 5)"
        ),
    )

    parser.add_argument(
        "--spec",
        nargs="+",
        default=[
            "100:10",
            "1000:100",
        ],
        help=(
            "Downsample factor:number of repeats. "
            "Example: "
            "--spec 10:10 100:25 1000:100"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help=(
            "Random seed "
            "(default: 12345)"
        ),
    )

    parser.add_argument(
        "--n-init",
        type=int,
        default=1,
        help=(
            "GMM initializations per fit "
            "(default: 1)"
        ),
    )

    args = parser.parse_args()

    folder = (
        args.folder
        .expanduser()
        .resolve()
    )

    if not folder.is_dir():

        print(
            f"ERROR: Not a folder: {folder}",
            file=sys.stderr,
        )

        sys.exit(1)

    try:

        specs = parse_spec(
            args.spec
        )

    except ValueError as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        sys.exit(1)

    reference_path = (
        folder
        / "gauss_2_mix.csv"
    )

    if not reference_path.exists():

        print(
            "ERROR: gauss_2_mix.csv was not found.",
            file=sys.stderr,
        )

        sys.exit(1)

    reference = load_reference_csv(
        reference_path
    )

    print(
        f"Valid full-data references: "
        f"{len(reference)}"
    )

    rng = np.random.default_rng(
        args.seed
    )

    try:

        selected_files = select_images(
            folder,
            reference,
            args.n_images,
            rng,
        )

    except RuntimeError as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        sys.exit(1)

    output_folder = (
        folder
        / "gaussian_downsample_error"
    )

    output_folder.mkdir(
        exist_ok=True
    )

    print()
    print("Selected images:")

    for path in selected_files:

        print(
            f"  {path.name}"
        )

    print()
    print("Experiment:")

    for factor, repeats in specs:

        print(
            f"  1/{factor}: "
            f"{repeats} repeats/image"
        )

    all_rows = []

    # ==============================================================
    # IMAGES
    # ==============================================================

    for image_number, image_path in enumerate(
        selected_files,
        start=1,
    ):

        print()

        print(
            f"[{image_number}/{len(selected_files)}] "
            f"{image_path.name}"
        )

        # ----------------------------------------------------------
        # TIFF IO is deliberately NOT part of timing.
        # ----------------------------------------------------------

        image = read_tiff(
            image_path
        )

        # ----------------------------------------------------------
        # Zero removal is deliberately NOT part of timing.
        # ----------------------------------------------------------

        pixels = get_valid_pixels(
            image
        )

        del image

        print(
            f"    valid nonzero pixels: "
            f"{len(pixels):,}"
        )

        reference_row = reference.loc[
            reference["filename"]
            == image_path.name
        ].iloc[0]

        # ==========================================================
        # DOWNSAMPLING FACTORS
        # ==========================================================

        for factor, repeats in specs:

            expected_samples = max(
                10,
                len(pixels) // factor,
            )

            print()

            print(
                f"    factor {factor}: "
                f"~{expected_samples:,} pixels/fit, "
                f"{repeats} repeats"
            )

            factor_times = []

            # ======================================================
            # MONTE CARLO REPEATS
            # ======================================================

            for repeat in range(
                1,
                repeats + 1,
            ):

                # Each repeat needs its own subsample, so the seed is drawn from
                # the shared stream: repeats stay independent and --seed still
                # reproduces the whole run.
                repeat_seed = int(
                    rng.integers(2**31)
                )

                # --------------------------------------------------
                # High-resolution wall clock.
                #
                # This includes:
                #   - random index selection
                #   - extracting the random subset
                #   - float conversion
                #   - the EM fit
                #
                # It excludes:
                #   - TIFF loading
                #   - zero elimination
                # --------------------------------------------------

                start_time = (
                    time.perf_counter()
                )

                # The pixels were masked once, before timing, so the fit itself
                # must not drop anything further.
                result = (
                    fit_two_gaussians(
                        pixels,
                        downsample_factor=factor,
                        omit_zeros=False,
                        n_init=args.n_init,
                        seed=repeat_seed,
                    )
                )

                elapsed_seconds = (
                    time.perf_counter()
                    - start_time
                )

                fit_time_ms = (
                    elapsed_seconds
                    * 1000.0
                )

                factor_times.append(
                    fit_time_ms
                )

                row = {
                    "filename":
                        image_path.name,

                    "downsample_factor":
                        factor,

                    "repeat":
                        repeat,

                    "n_valid_pixels":
                        result[
                            "n_valid_pixels"
                        ],

                    "n_sampled_pixels":
                        result[
                            "n_sampled_pixels"
                        ],

                    "fit_time_ms":
                        fit_time_ms,
                }

                # --------------------------------------------------
                # Store estimates and errors.
                # --------------------------------------------------

                for parameter in PARAMETERS:

                    estimate = float(
                        result[
                            parameter
                        ]
                    )

                    truth = float(
                        reference_row[
                            parameter
                        ]
                    )

                    absolute_error = (
                        estimate
                        - truth
                    )

                    # Relative error, percentage.
                    #
                    # If reference parameter somehow equals exactly
                    # zero, record NaN rather than divide by zero.
                    if truth != 0:

                        relative_error_pct = (
                            100.0
                            * absolute_error
                            / abs(truth)
                        )

                    else:

                        relative_error_pct = (
                            np.nan
                        )

                    row[
                        f"reference_{parameter}"
                    ] = truth

                    row[
                        f"estimate_{parameter}"
                    ] = estimate

                    row[
                        f"error_{parameter}"
                    ] = absolute_error

                    row[
                        f"relative_error_pct_{parameter}"
                    ] = relative_error_pct

                all_rows.append(
                    row
                )

                if (
                    repeat == 1
                    or repeat == repeats
                    or repeat % 10 == 0
                ):

                    print(
                        f"        repeat "
                        f"{repeat:4d}/{repeats}   "
                        f"{fit_time_ms:9.2f} ms",
                        flush=True,
                    )

            # ------------------------------------------------------
            # Timing summary immediately after each factor.
            # ------------------------------------------------------

            average_ms = np.mean(
                factor_times
            )

            sd_ms = np.std(
                factor_times,
                ddof=1,
            ) if len(factor_times) > 1 else 0.0

            print(
                f"        mean timing: "
                f"{average_ms:.2f} +/- "
                f"{sd_ms:.2f} ms"
            )

    # ==============================================================
    # SAVE RAW RESULTS
    # ==============================================================

    results = pd.DataFrame(
        all_rows
    )

    raw_path = (
        output_folder
        / "downsample_error_raw.csv"
    )

    results.to_csv(
        raw_path,
        index=False,
    )

    # ==============================================================
    # PARAMETER SUMMARY
    # ==============================================================

    summary = make_summary(
        results
    )

    summary_path = (
        output_folder
        / "downsample_error_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ==============================================================
    # IMAGE/FACTOR SUMMARY
    # ==============================================================

    factor_summary = (
        make_factor_summary(
            results
        )
    )

    factor_summary_path = (
        output_folder
        / "downsample_factor_summary.csv"
    )

    factor_summary.to_csv(
        factor_summary_path,
        index=False,
    )

    # ==============================================================
    # PLOTS
    # ==============================================================

    print()
    print("Generating plots...")

    for filename in results[
        "filename"
    ].unique():

        subset = results[
            results["filename"]
            == filename
        ]

        # Parameter-error boxplots.
        for parameter in PARAMETERS:

            make_error_boxplot(
                subset,
                parameter,
                filename,
                output_folder,
            )

        # Individual timing line.
        make_timing_plot(
            subset,
            filename,
            output_folder,
        )

    # Final combined figure.
    make_means_spaghetti_plot(
        factor_summary,
        output_folder,
    )

    make_timing_spaghetti_plot(
        factor_summary,
        output_folder,
    )

    print()
    print("Done.")

    print(
        f"Output directory: "
        f"{output_folder}"
    )

    print(
        f"Raw Monte Carlo results: "
        f"{raw_path.name}"
    )

    print(
        f"Parameter summary: "
        f"{summary_path.name}"
    )

    print(
        f"Factor summary: "
        f"{factor_summary_path.name}"
    )


if __name__ == "__main__":
    main()