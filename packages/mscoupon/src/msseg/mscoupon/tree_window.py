"""Merge-tree icicle window for the mscoupon viewer.

A Toplevel hosting a matplotlib canvas that draws the voxel-count merge-tree
icicle via the shared ``msseg.viz.draw_icicle`` (same flat {nodes, roots} JSON
the C++ core emits, colored by each subtree's deepest extremum). Mirrors the
cellseg ``TreeWindow`` pattern. Heavy deps (matplotlib, msseg.viz) are imported
here so the main app stays importable without them.
"""
from __future__ import annotations

import json
import tkinter as tk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from msseg.viz import draw_icicle, min_color


class TreeWindow(tk.Toplevel):
    """A resizable window drawing the merge-tree icicle for a slice."""

    def __init__(self, master, tree_json: str):
        super().__init__(master)
        self.title("mscoupon — merge tree")
        self.tree = json.loads(tree_json)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        NavigationToolbar2Tk(self.canvas, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.refresh()

    def refresh(self, tree_json: str | None = None):
        if tree_json is not None:
            self.tree = json.loads(tree_json)
        self.ax.clear()
        draw_icicle(self.ax, self.tree, color_of_min=min_color)
        self.canvas.draw_idle()
