#!/usr/bin/env python3

"""
plot_cross_stack.py

Plot GMM mean and median measurements from gmm_cross_stack.csv (measure_gmm.py).

Creates a broken-y-axis style plot:

    upper panel: metal / high-intensity component
                 mu_2 and median_2

    lower panel: air / low-intensity component
                 mu_1 and median_1

The x-axis is image_index, compressed across unmeasured ranges by
msseg.mscoupon.normalize.plotting; each folder and each contiguous block is
drawn separately so no line ever spans a gap.

Usage:

    python plot_cross_stack.py gmm_cross_stack.csv

Optional output filename:

    python plot_cross_stack.py gmm_cross_stack.csv \
        --output gmm_cross_stack_plot.png

Dependencies:

    pip install pandas numpy matplotlib
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from msseg.mscoupon.normalize.plotting import (compressed_x_axis, draw_x_breaks, draw_y_break,
                                               label_blocks, load_measurement_csv,
                                               padded_limits)

NUMERIC_COLUMNS = ["image_index", "mu_1", "median_1", "mu_2", "median_2"]


def load_data(csv_path):
    """Load and clean the GMM CSV."""

    df = load_measurement_csv(csv_path, required=NUMERIC_COLUMNS)

    if "folder" not in df.columns:
        raise ValueError("CSV is missing required columns: ['folder']")

    return df.sort_values(["folder", "image_index"])


def make_plot(df, output_path):
    """Plot air/metal mean and median with broken y and compressed x axes."""

    fig, (ax_metal, ax_air) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(16, 8),
        gridspec_kw={
            "height_ratios": [1, 1],
            "hspace": 0.06,
        },
    )

    positions, breaks, blocks = compressed_x_axis(df["image_index"].unique())

    for folder in df["folder"].unique():

        folder_df = df[df["folder"] == folder].sort_values("image_index")

        folder_color = None
        first_block = True

        for block in blocks:

            subset = folder_df[folder_df["image_index"].isin(block)]

            if subset.empty:
                continue

            x = [positions[int(i)] for i in subset["image_index"]]

            if folder_color is None:

                line, = ax_air.plot(
                    x,
                    subset["mu_1"],
                    marker="o",
                    markersize=3,
                    linewidth=1.3,
                    label=f"{folder} mean",
                )

                folder_color = line.get_color()

            else:

                ax_air.plot(
                    x,
                    subset["mu_1"],
                    marker="o",
                    markersize=3,
                    linewidth=1.3,
                    color=folder_color,
                )

            ax_air.plot(
                x,
                subset["median_1"],
                marker=".",
                markersize=4,
                linewidth=1.2,
                linestyle="--",
                color=folder_color,
                label=f"{folder} median" if first_block else None,
            )

            ax_metal.plot(
                x,
                subset["mu_2"],
                marker="o",
                markersize=3,
                linewidth=1.3,
                color=folder_color,
            )

            ax_metal.plot(
                x,
                subset["median_2"],
                marker=".",
                markersize=4,
                linewidth=1.2,
                linestyle="--",
                color=folder_color,
            )

            first_block = False

    # Mean and median share a panel, so the limits must cover both.
    ax_air.set_ylim(padded_limits(
        np.concatenate([df["mu_1"].values, df["median_1"].values])))

    ax_metal.set_ylim(padded_limits(
        np.concatenate([df["mu_2"].values, df["median_2"].values])))

    for ax in (ax_metal, ax_air):
        draw_x_breaks(ax, breaks)
        ax.grid(alpha=0.25)

    label_blocks(ax_air, blocks, positions)

    ax_metal.set_ylabel("Metal intensity")
    ax_air.set_ylabel("Air intensity")
    ax_air.set_xlabel("Image number")

    ax_metal.set_title(
        "GMM-derived air and metal intensity across scan\n"
        "solid = GMM mean, dashed = empirical median"
    )

    draw_y_break(ax_metal, ax_air)

    ax_air.legend(fontsize="small", title="Substack / statistic", loc="best")

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.91)

    fig.savefig(output_path, dpi=250, bbox_inches="tight")

    plt.close(fig)

    print(f"Wrote: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Plot air and metal GMM mean/median values "
            "with a broken y-axis."
        )
    )
    parser.add_argument("csv", type=Path, help="Input gmm_cross_stack.csv")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG filename. Default: <input>_plot.png")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    csv_path = args.csv.expanduser().resolve()

    if args.output is None:
        output_path = csv_path.parent / (csv_path.stem + "_plot.png")
    else:
        output_path = args.output.expanduser().resolve()

    df = load_data(csv_path)

    print(f"Loaded {len(df)} valid measurements")
    print(f"Folders: {df['folder'].nunique()}")

    make_plot(df, output_path)


if __name__ == "__main__":
    main()
