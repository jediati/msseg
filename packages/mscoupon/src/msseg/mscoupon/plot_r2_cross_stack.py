#!/usr/bin/env python3

"""
plot_r2_cross_stack.py

Plot median intensities from the two fixed regions measured by
measure_2_regions.py.

- air_p50_0   = median intensity in air ROI
- metal_p50_0 = median intensity in metal ROI
- broken/compressed x-axis for skipped image ranges
- separate y ranges for air and metal

The axis compression lives in msseg.mscoupon.normalize.plotting; each folder and
each contiguous block is drawn separately so no line ever spans a gap.

Usage:

    python plot_r2_cross_stack.py region_stats.csv

Optional output:

    python plot_r2_cross_stack.py region_stats.csv \
        --output region_stats_plot.png

Dependencies:

    pip install numpy pandas matplotlib
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from msseg.mscoupon.normalize.plotting import (compressed_x_axis, draw_x_breaks, draw_y_break,
                                               find_contiguous_blocks, label_blocks,
                                               load_measurement_csv, padded_limits)

NUMERIC_COLUMNS = ["image_index", "air_p50_0", "metal_p50_0"]


def load_data(csv_path):
    """Load and clean region measurement CSV."""

    df = load_measurement_csv(csv_path, required=NUMERIC_COLUMNS)

    if "folder" not in df.columns:
        raise ValueError("CSV is missing required columns: ['folder']")

    return df.sort_values(["folder", "image_index"])


def make_plot(df, output_path):
    """Plot air and metal ROI medians."""

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

    folders = df["folder"].unique()

    positions, breaks, blocks = compressed_x_axis(df["image_index"].unique())

    for folder in folders:

        folder_df = df[df["folder"] == folder].sort_values("image_index")

        folder_color = None

        for block in blocks:

            subset = folder_df[folder_df["image_index"].isin(block)]

            if subset.empty:
                continue

            x = [positions[int(i)] for i in subset["image_index"]]

            if folder_color is None:

                line, = ax_air.plot(
                    x,
                    subset["air_p50_0"],
                    marker="o",
                    markersize=3,
                    linewidth=1.4,
                    label=folder,
                )

                folder_color = line.get_color()

            else:

                ax_air.plot(
                    x,
                    subset["air_p50_0"],
                    marker="o",
                    markersize=3,
                    linewidth=1.4,
                    color=folder_color,
                )

            ax_metal.plot(
                x,
                subset["metal_p50_0"],
                marker="o",
                markersize=3,
                linewidth=1.4,
                color=folder_color,
            )

    ax_air.set_ylim(padded_limits(df["air_p50_0"].values))
    ax_metal.set_ylim(padded_limits(df["metal_p50_0"].values))

    for ax in (ax_metal, ax_air):
        draw_x_breaks(ax, breaks)
        ax.grid(alpha=0.25)

    label_blocks(ax_air, blocks, positions)

    ax_metal.set_ylabel("Metal ROI median")
    ax_air.set_ylabel("Air ROI median")
    ax_air.set_xlabel("Image number")

    ax_metal.set_title("Fixed-region median intensity across scan")

    draw_y_break(ax_metal, ax_air)

    if len(folders) > 1:
        ax_air.legend(fontsize="small", title="Folder", loc="best")

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.91)

    fig.savefig(output_path, dpi=250, bbox_inches="tight")

    plt.close(fig)

    print(f"Wrote: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Plot air and metal ROI median intensities "
            "from measure_2_regions.py."
        )
    )
    parser.add_argument("csv", type=Path, help="Input region_stats.csv")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG. Default: <input>_plot.png")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    csv_path = args.csv.expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    if args.output is None:
        output_path = csv_path.parent / (csv_path.stem + "_plot.png")
    else:
        output_path = args.output.expanduser().resolve()

    df = load_data(csv_path)

    print(f"Loaded {len(df)} measurements")
    print(f"Folders: {df['folder'].nunique()}")
    print(f"Sample blocks: {len(find_contiguous_blocks(df['image_index'].unique()))}")

    make_plot(df, output_path)


if __name__ == "__main__":
    main()
