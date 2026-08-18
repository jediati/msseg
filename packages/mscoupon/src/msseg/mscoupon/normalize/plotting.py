"""Shared plotting helpers for the cross-stack measurement plots.

A scan is usually measured in a few disjoint index blocks (``--samples 200:210``
here, ``2500:2510`` there). Plotting those against a linear x-axis wastes most
of the figure on empty space, so the plotters compress the gaps and mark the
breaks. That logic -- along with padded axis limits and the broken-y-axis
decoration -- was duplicated across all four plot scripts; it lives here once.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def padded_limits(values, padding_fraction: float = 0.08) -> Tuple[float, float]:
    """Min/max with a margin, robust to a constant series."""
    values = np.asarray([v for v in np.asarray(values, dtype=float).ravel()
                         if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return 0.0, 1.0

    lo, hi = float(np.min(values)), float(np.max(values))
    if lo == hi:
        pad = abs(lo) * padding_fraction or 1.0
    else:
        pad = (hi - lo) * padding_fraction
    return lo - pad, hi + pad


def find_contiguous_blocks(indices: Sequence[int]) -> List[List[int]]:
    """Split an index list into runs of consecutive values.

    Duplicates are collapsed first: a CSV that appends several folders repeats
    every image_index, and a repeat would otherwise look like the start of a new
    block and split the axis where there is no gap.
    """
    ordered = sorted({int(i) for i in indices})
    if not ordered:
        return []

    blocks: List[List[int]] = [[ordered[0]]]
    for i in ordered[1:]:
        if i == blocks[-1][-1] + 1:
            blocks[-1].append(i)
        else:
            blocks.append([i])
    return blocks


def build_compressed_x(blocks: Sequence[Sequence[int]], block_gap: float = 2.0):
    """Map image indices onto a compressed x-axis.

    Returns ``(positions, breaks)`` where `positions` maps each image index to
    its plot coordinate and `breaks` gives the midpoints between blocks, for
    :func:`draw_x_breaks`.
    """
    positions = {}
    breaks: List[float] = []
    cursor = 0.0

    for b, block in enumerate(blocks):
        if b > 0:
            breaks.append(cursor + block_gap / 2.0)
            cursor += block_gap
        for index in block:
            positions[int(index)] = cursor
            cursor += 1.0
    return positions, breaks


def compressed_x_axis(image_indices: Sequence[int], block_gap: float = 2.0):
    """:func:`find_contiguous_blocks` + :func:`build_compressed_x` in one step."""
    blocks = find_contiguous_blocks(image_indices)
    positions, breaks = build_compressed_x(blocks, block_gap)
    return positions, breaks, blocks


def draw_x_breaks(ax, break_positions: Iterable[float]) -> None:
    """Mark compressed gaps with a dashed vertical rule."""
    for x in break_positions:
        ax.axvline(x, color="0.75", linewidth=0.8, linestyle=(0, (4, 4)), zorder=0)


def label_blocks(ax, blocks: Sequence[Sequence[int]], positions) -> None:
    """Tick each block at its first and last image index."""
    ticks, labels = [], []
    for block in blocks:
        for index in (block[0], block[-1]) if len(block) > 1 else (block[0],):
            ticks.append(positions[int(index)])
            labels.append(str(index))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)


def draw_y_break(ax_top, ax_bottom, size: float = 0.008) -> None:
    """Decorate a shared broken y-axis with the conventional diagonal marks."""
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)

    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=0.8)
    ax_top.plot((-size, +size), (-size, +size), **kwargs)
    ax_top.plot((1 - size, 1 + size), (-size, +size), **kwargs)

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-size, +size), (1 - size, 1 + size), **kwargs)
    ax_bottom.plot((1 - size, 1 + size), (1 - size, 1 + size), **kwargs)


def load_measurement_csv(path, required: Sequence[str] = (), numeric: Sequence[str] = None):
    """Read a measurement CSV, coercing the numeric columns and dropping bad rows.

    `required` columns must be present. `numeric` names the ones to coerce and
    to drop NaN rows on; it defaults to `required`. Pass it explicitly when a
    required column is text -- coercing e.g. "folder" would NaN the whole column
    and the dropna would then empty the frame.
    """
    import pandas as pd

    df = pd.read_csv(path)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    to_coerce = [c for c in (required if numeric is None else numeric) if c in df.columns]
    for col in to_coerce:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if to_coerce:
        df = df.dropna(subset=to_coerce)
    return df.reset_index(drop=True)
