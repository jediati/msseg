# %% [markdown]
# cellseg pipeline demo
#
# Loads a TIFF stack, forces it into a 32-bit float 3D volume, runs the cellseg
# two-phase Morse-Smale pipeline, and plots the minima->1-saddle merge tree with
# edges colour-coded by arc id (the merging 1-saddle's node id).
#
# The script is deliberately linear (each `# %%` block is a self-contained cell)
# so it drops straight into a Jupyter notebook.

# %%
# --- Imports & parameters ---------------------------------------------------
import json
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tifffile

from msseg import cellseg

print("cellseg version:", cellseg.version())

# Path to a TIFF stack (multipage 3D, or a stack of 2D pages). Override on the
# command line: python cellseg_pipeline_demo.py <stack.tif>
TIFF_PATH = sys.argv[1] if len(sys.argv) > 1 else "stack.tif"

# How to reduce an RGB/RGBA channel axis to a single scalar per voxel:
#   "max"/"mean" combine channels, or an int selects one channel (mScarlet ~ red = 0).
CHANNEL = "max"

# Heavy-lift (Phase A) knobs.
BLUR_SIGMA = 3.0            # Gaussian blur sigma (median is a Python pre-pass, if wanted)
PERSISTENCE_PERCENT = 2.0   # simplify to this % of the intensity range

# Phase-B threshold knobs (re-runnable cheaply without repeating the heavy lift).
PERSISTENCE_SELECTION = None       # None -> reuse the heavy-lift persistence
CUT_THRESHOLD = 9.0                # merge-tree cut: cut mergers with saddle value > this
BACKGROUND_PERCENTILE = 80.0       # intensity threshold for the fluorescent-membrane mask

# %%
# --- Load TIFF -> float32 (depth, height, width) ----------------------------
# tifffile stacks pages into an ndarray. A z-stack of RGB pages therefore comes
# in as 4D (Z, H, W, C) (or channels-first (Z, C, H, W)). We (1) squeeze
# singleton axes, (2) collapse the RGB/RGBA channel axis (size 3/4) to one
# scalar per voxel per CHANNEL, (3) collapse any remaining leading axes (e.g.
# time) by taking the first, and (4) promote a lone 2D slice to depth 1 -- so
# the result is always a 3D (z=pages, y, x) volume.
raw = np.squeeze(np.asarray(tifffile.imread(TIFF_PATH)))
print("raw TIFF shape:", raw.shape)

def _reduce_channels(a):
    if a.ndim >= 3 and a.shape[-1] in (3, 4):      # channels-last (..., C)
        axis = a.ndim - 1
    elif a.ndim >= 4 and a.shape[1] in (3, 4):     # channels-first (Z, C, H, W)
        axis = 1
    else:
        return a
    if isinstance(CHANNEL, int):
        return np.take(a, CHANNEL, axis=axis)
    return a.mean(axis=axis) if CHANNEL == "mean" else a.max(axis=axis)

raw = _reduce_channels(raw)
while raw.ndim > 3:      # any remaining leading axis (e.g. time) -> take the first
    raw = raw[0]
if raw.ndim == 2:       # a single 2D slice -> depth 1
    raw = raw[None, ...]
if raw.ndim != 3:
    raise ValueError(f"expected a 2D/3D volume after channel reduction, got shape {raw.shape}")

# Force a contiguous C-order float32 (d, h, w) volume regardless of source dtype.
volume = np.ascontiguousarray(raw, dtype=np.float32)
print("volume:", volume.shape, volume.dtype,
      "range [%.4g, %.4g]" % (float(volume.min()), float(volume.max())))

# %%
# --- Phase A: heavy lift (once) ---------------------------------------------
# read -> blur -> discrete gradient (steepest descent) -> MS complex ->
# simplify -> ascending/descending 3-manifold base decompositions.
heavy_params = json.dumps({
    "blur_sigma": BLUR_SIGMA,
    "persistence_percent": PERSISTENCE_PERCENT,
})
pipe = cellseg.heavy_lift(volume, heavy_params)
print("value range      :", pipe.value_range())
print("heavy persistence:", pipe.heavy_persistence())

# %%
# --- Phase B: select a living persistence and build the merge tree ----------
persistence = pipe.heavy_persistence() if PERSISTENCE_SELECTION is None else PERSISTENCE_SELECTION
pipe.set_persistence(persistence)
tree = json.loads(pipe.merge_tree_json())
print("selected persistence:", pipe.current_persistence())
print("merge-tree roots    :", len(tree["roots"]))

# %%
# --- Lay out the merge tree as a voxel-count icicle -------------------------
# Each node is a box. Its x-extent comes from the feature's voxel count: the
# root spans the whole width, and every split divides the parent's span
# proportionally. Children are ordered left-to-right by their subtree's lowest
# minima value (deepest feature first). Boxes use their full size width and are
# separated only by a 1px white border. y is the function value: a box spans
# [node value, parent value], so its height is the feature's persistence.
#
# Colour is inherited by size (independent of the ordering): the root gets a
# random colour, its largest sub-feature keeps that colour, and every other
# child starts a new random lineage colour.
import colorsys

_rng = np.random.default_rng(0)

def _rand_color():
    h = float(_rng.random())
    s = 0.45 + 0.40 * float(_rng.random())
    v = 0.75 + 0.20 * float(_rng.random())
    return colorsys.hsv_to_rgb(h, s, v)

_all_vals = []
def _collect_vals(node):
    _all_vals.append(node["value"])
    for c in node.get("children", []):
        _collect_vals(c)
for root in tree["roots"]:
    _collect_vals(root)
_vmin, _vmax = min(_all_vals), max(_all_vals)
_vspan = (_vmax - _vmin) or 1.0
_root_top = _vmax + 0.05 * _vspan   # short cap above the final merge

# O(n) post-order pass: each node's lowest-leaf value (min minima value in its
# subtree). Used to order children left-to-right (deepest minimum first).
def _set_lowest_leaf(node):
    kids = node.get("children", [])
    node["_lowest_leaf"] = node["value"] if not kids else min(_set_lowest_leaf(c) for c in kids)
    return node["_lowest_leaf"]
for root in tree["roots"]:
    _set_lowest_leaf(root)

def _layout(node, x0, color, parent_value):
    node["_x0"] = x0
    node["_w"] = float(node["voxel_count"])   # full width; separation is the white border
    node["_color"] = color
    node["_y0"] = node["value"]
    node["_y1"] = parent_value
    kids = node.get("children", [])
    largest = max(kids, key=lambda c: c["voxel_count"], default=None)  # colour heir (by size)
    cursor = x0                                # place children left-to-right by lowest leaf
    for c in sorted(kids, key=lambda c: c["_lowest_leaf"]):
        _layout(c, cursor, color if c is largest else _rand_color(), node["value"])
        cursor += float(c["voxel_count"])

# Roots laid left-to-right by lowest leaf (usually a single root).
roots = sorted(tree["roots"], key=lambda r: r["_lowest_leaf"])
_cursor = 0.0
for r in roots:
    _layout(r, _cursor, _rand_color(), _root_top)
    _cursor += float(r["voxel_count"])
_total_size = _cursor

_n_leaf = 0
def _count_leaves(node):
    global _n_leaf
    if node.get("type") == "leaf":
        _n_leaf += 1
    for c in node.get("children", []):
        _count_leaves(c)
for root in roots:
    _count_leaves(root)
print("leaves (minima):", _n_leaf, "| total feature size:", int(_total_size))

# %%
# --- Plot the voxel-count icicle --------------------------------------------
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(13, 7))

def _draw(node):
    ax.add_patch(Rectangle((node["_x0"], node["_y0"]), node["_w"],
                           node["_y1"] - node["_y0"],
                           facecolor=node["_color"], edgecolor="white", linewidth=1.0))
    for c in node.get("children", []):
        _draw(c)

for root in roots:
    _draw(root)

ax.set_xlim(0, _total_size)
ax.set_ylim(_vmin - 0.02 * _vspan, _root_top)
ax.set_xlabel("voxel count (feature size)")
ax.set_ylabel("function value")
ax.set_title(f"cellseg merge tree (voxel-count icicle) — {_n_leaf} features "
             f"(persistence {pipe.current_persistence():.4g})")
fig.tight_layout()
fig.savefig("cellseg_merge_tree.png", dpi=130)
print("saved cellseg_merge_tree.png")

# %%
# --- Phase B: run a segmentation (bit-flag seg8 + cell-id volumes) ----------
# background threshold = an intensity percentile of the (already blurred-ish)
# input, standing in for "the most likely outer boundary of the dye".
background_threshold = 36 #float(np.percentile(volume, BACKGROUND_PERCENTILE))
seg8, ids = pipe.segment(CUT_THRESHOLD, background_threshold)
print("seg8:", seg8.shape, seg8.dtype, "| ids:", ids.shape, ids.dtype)
print("background threshold:", background_threshold)
for bit, name in [(1, "non-bg ascending"), (2, "cleaned membrane"),
                  (4, "membrane region"), (8, "foreground")]:
    print(f"  bit {bit} ({name}): {int((seg8 & bit > 0).sum())} voxels")
print("distinct cell ids:", int(np.unique(ids).size))

# %%
# --- Show a mid-depth slice of the input, seg8 flags, and cell ids ----------
z = volume.shape[0] // 2
fig2, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(volume[z], cmap="gray"); axs[0].set_title(f"input (z={z})")
axs[1].imshow(seg8[z], cmap="tab20", vmin=0, vmax=15); axs[1].set_title("seg8 bit-flags")
axs[2].imshow(ids[z], cmap="tab20"); axs[2].set_title("cell ids")
for a in axs:
    a.set_xticks([]); a.set_yticks([])
fig2.tight_layout()
fig2.savefig("cellseg_slice.png", dpi=130)
print("saved cellseg_slice.png")

# %%
plt.show()
