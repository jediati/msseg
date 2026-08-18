#!/usr/bin/env python3

"""
plot_intensities.py

Load TIFF images from a folder, compute the total pixel intensity for
each selected image, and plot the result as a 1D stack profile.

The folder and --samples arguments match measure_im.py.

Unlike the measure_* scripts this one deliberately sums every pixel, including
exact zeros: the quantity of interest is the raw per-slice flux, so masking the
no-data background would change what is being compared from slice to slice.

Examples
--------

python plot_intensities.py "F:\\data\\spears\\tomo_sample_2_051_rec" \
    --samples 200:210

Optional output filename:

python plot_intensities.py "F:\\data\\spears\\tomo_sample_2_051_rec" \
    --samples 200:210 \
    --output intensities.png

Dependencies
------------

pip install numpy tifffile matplotlib
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from msseg.mscoupon.normalize import find_tiff_files, iter_images, parse_samples
from msseg.mscoupon.normalize.plotting import (compressed_x_axis, draw_x_breaks, label_blocks,
                                               padded_limits)


def measure_intensities(selected, start):
    rows = []

    for local_i, image_index, image_path, image in iter_images(selected, start):

        print(f"[{local_i + 1}/{len(selected)}] {image_path.name}", flush=True)

        total_intensity = float(np.sum(np.asarray(image), dtype=np.float64))

        rows.append(
            {
                "filename": image_path.name,
                "image_index": image_index,
                "total_intensity": total_intensity,
            }
        )

        print(f"    total intensity: {total_intensity:.12g}")

    return rows


def make_plot(rows, folder_name, output_path):
    image_indices = [row["image_index"] for row in rows]
    intensities = [row["total_intensity"] for row in rows]

    positions, breaks, blocks = compressed_x_axis(image_indices)

    x = [positions[int(i)] for i in image_indices]

    fig, ax = plt.subplots(1, 1, figsize=(16, 5))

    ax.plot(x, intensities, marker="o", markersize=3, linewidth=1.3, label=folder_name)

    ax.set_ylim(padded_limits(intensities))

    draw_x_breaks(ax, breaks)
    label_blocks(ax, blocks, positions)

    ax.set_xlabel("Image number")
    ax.set_ylabel("Total pixel intensity")
    ax.set_title("Total pixel intensity across scan")

    ax.grid(alpha=0.25)
    ax.legend(fontsize="small", title="Substack", loc="best")

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.90)

    fig.savefig(output_path, dpi=250, bbox_inches="tight")

    plt.close(fig)

    print(f"Wrote: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Plot total TIFF pixel intensity across a selected "
            "image stack."
        )
    )
    parser.add_argument("folder", type=Path, help="Folder containing TIFF images.")
    parser.add_argument("--samples", type=str, default=None,
                        help="Optional START:END subsequence. Example: --samples 200:210")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG filename. Default: <folder>_intensities.png")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    folder = args.folder.expanduser().resolve()

    if not folder.is_dir():
        print(f"ERROR: invalid folder: {folder}", file=sys.stderr)
        sys.exit(1)

    files = find_tiff_files(folder)

    if not files:
        print("ERROR: no TIFF files found", file=sys.stderr)
        sys.exit(1)

    try:
        start, end = parse_samples(args.samples, len(files))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    selected = files[start:end]

    if not selected:
        print("ERROR: selected range contains no images", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        output_path = (Path.cwd() / f"{folder.name}_intensities.png").resolve()
    else:
        output_path = args.output.expanduser().resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Folder: {folder}")
    print(f"Images: {start}:{end}")
    print(f"Output: {output_path}")
    print()

    rows = measure_intensities(selected, start)

    make_plot(rows, folder.name, output_path)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
