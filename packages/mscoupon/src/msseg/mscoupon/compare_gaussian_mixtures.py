#!/usr/bin/env python3

"""
compare_gaussian_mixtures.py

Compare Gaussian-mixture fitting results produced with different
random downsampling factors.

The script scans a folder for CSV files named like:

    gauss_2_mix.csv
    gauss_2_csv_sub_10.csv
    gauss_2_csv_sub_100.csv
    gauss_2_csv_sub_1000.csv
    ...

It infers the downsampling factor from the filename.

gauss_2_mix.csv is treated as:
    downsample factor = 1
    label = "full"

Rows containing ERROR, missing values, NaN, or invalid numeric values
are ignored independently for each CSV.

Outputs:
    gaussian_mixture_comparison_summary.csv
    boxplot_mu_1.png
    boxplot_sigma_1.png
    boxplot_mu_2.png
    boxplot_sigma_2.png

Usage:
    python compare_gaussian_mixtures.py "C:\\path\\to\\folder"

Dependencies:
    pip install pandas numpy matplotlib
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msseg.mscoupon.normalize import PARAMETERS


def infer_downsample_factor(filename):
    """
    Infer downsample factor from filename.

    Examples:
        gauss_2_mix.csv              -> 1
        gauss_2_csv_sub_10.csv       -> 10
        gauss_2_csv_sub_100.csv      -> 100
        gauss_2_csv_sub_1000.csv     -> 1000

    Returns:
        integer factor, or None if filename does not match.
    """

    name = filename.lower()

    if name == "gauss_2_mix.csv":
        return 1

    match = re.search(r"gauss_2_csv_sub_(\d+)\.csv$", name)

    if match:
        return int(match.group(1))

    return None


def sampling_label(factor):
    return "full" if factor == 1 else f"1/{factor}"


def load_result_file(path):
    """
    Load a Gaussian-mixture result CSV and clean invalid rows.
    """

    factor = infer_downsample_factor(path.name)

    if factor is None:
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: Could not read {path.name}: {exc}")
        return None

    missing_columns = [col for col in PARAMETERS if col not in df.columns]

    if missing_columns:
        print(
            f"WARNING: Skipping {path.name}: "
            f"missing columns {missing_columns}"
        )
        return None

    # "ERROR" and malformed entries become NaN. A row is kept only if all four
    # Gaussian parameters survive.
    for col in PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cleaned = df.loc[np.isfinite(df[PARAMETERS]).all(axis=1)].copy()

    cleaned["downsample_factor"] = factor
    cleaned["sampling_label"] = sampling_label(factor)

    print(f"{path.name}: {len(cleaned):,}/{len(df):,} valid rows")

    return cleaned


def values_by_factor(all_data, parameter):
    """Yield (factor, label, values) for every sampling factor present."""

    for factor in sorted(all_data["downsample_factor"].unique()):

        values = all_data.loc[
            all_data["downsample_factor"] == factor,
            parameter,
        ].dropna()

        if len(values) == 0:
            continue

        yield factor, sampling_label(factor), values


def make_summary(all_data):
    """
    Compute descriptive statistics for each downsampling factor.
    """

    rows = []

    for factor in sorted(all_data["downsample_factor"].unique()):

        subset = all_data[all_data["downsample_factor"] == factor]

        for parameter in PARAMETERS:

            values = subset[parameter].dropna()

            if len(values) == 0:
                continue

            rows.append(
                {
                    "downsample_factor": factor,
                    "sampling_label": sampling_label(factor),
                    "parameter": parameter,
                    "n": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "median": values.median(),
                    "min": values.min(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                    "max": values.max(),
                }
            )

    return pd.DataFrame(rows)


def save_figure(output_folder, name):
    output_path = output_folder / name

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Wrote {output_path.name}")


def make_boxplot(all_data, parameter, output_folder):
    """
    Make one box plot comparing all sampling factors.
    """

    data = []
    labels = []

    for _, label, values in values_by_factor(all_data, parameter):
        data.append(values.values)
        labels.append(label)

    if not data:
        return

    plt.figure(figsize=(9, 6))

    plt.boxplot(data, tick_labels=labels, showmeans=True)

    plt.xlabel("Sampling fraction")
    plt.ylabel(parameter)
    plt.title(f"{parameter}: Gaussian mixture fit vs downsampling")

    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    save_figure(output_folder, f"boxplot_{parameter}.png")


def make_mean_sd_plot(all_data, parameter, output_folder):
    """
    Plot mean +/- 1 SD for each sampling factor.
    """

    means = []
    sds = []
    labels = []

    for _, label, values in values_by_factor(all_data, parameter):
        means.append(values.mean())
        sds.append(values.std(ddof=1))
        labels.append(label)

    if not means:
        return

    x = np.arange(len(means))

    plt.figure(figsize=(9, 6))

    plt.errorbar(x, means, yerr=sds, fmt="o-", capsize=5)

    plt.xticks(x, labels)

    plt.xlabel("Sampling fraction")
    plt.ylabel(parameter)
    plt.title(f"{parameter}: mean +/- SD")

    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    save_figure(output_folder, f"mean_sd_{parameter}.png")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two-Gaussian mixture results "
            "across downsampling factors."
        )
    )
    parser.add_argument("folder", type=Path,
                        help="Folder containing gauss_2_mix.csv and/or "
                             "gauss_2_csv_sub_*.csv files.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    folder = args.folder.expanduser().resolve()

    if not folder.exists():
        print(f"ERROR: Folder does not exist: {folder}", file=sys.stderr)
        sys.exit(1)

    if not folder.is_dir():
        print(f"ERROR: Not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    # Find all CSV files whose filenames encode a recognized downsampling factor.
    csv_files = []

    for path in folder.glob("*.csv"):

        factor = infer_downsample_factor(path.name)

        if factor is not None:
            csv_files.append((factor, path))

    csv_files.sort(key=lambda item: item[0])

    if not csv_files:
        print(
            "ERROR: No matching Gaussian-mixture "
            "CSV files found.",
            file=sys.stderr,
        )
        sys.exit(1)

    print()
    print("Found:")
    for factor, path in csv_files:
        print(f"  {sampling_label(factor):>8} : {path.name}")

    print()
    print("Loading data...")

    datasets = []

    for factor, path in csv_files:

        df = load_result_file(path)

        if df is not None and len(df) > 0:
            datasets.append(df)

    if not datasets:
        print("ERROR: No valid data found.", file=sys.stderr)
        sys.exit(1)

    all_data = pd.concat(datasets, ignore_index=True)

    print()
    print("Generating summary...")

    summary = make_summary(all_data)

    summary_path = folder / "gaussian_mixture_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {summary_path.name}")

    print()
    print("Generating plots...")

    for parameter in PARAMETERS:
        make_boxplot(all_data, parameter, folder)
        make_mean_sd_plot(all_data, parameter, folder)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
