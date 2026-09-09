"""Tunables, vocabularies and small helpers shared by the annotation shell's
modules: overlay alphas and coloring modes, the model kinds and their
training/search budgets, the drawing-tool options, and the hint constants.
Everything is re-exported wholesale (``__all__`` lists the underscored names
too) so the modules and the coupon labeler read them as before.
"""
from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from . import edge_model, magic_fill, model_search
from .labeling import MAX_CLASSES, TOOLS


# How faint the inherited region overlay is drawn under the class layer
# (0..255); the class colors themselves stay fully opaque in the LUT and are
# scaled by the shared alpha slider like every overlay.
_REGION_ALPHA = 90
# Classifier predictions render between the region layer and the user's own
# labels, slightly translucent so drawn labels stay distinguishable on top.
_PRED_ALPHA = 170
# A scalar regions layer (class probability / uncertainty) IS the point of the
# view, not orientation under it, so it is drawn much less faintly than the
# region-id layer it replaces.
_SCALAR_ALPHA = 150

# Regions coloring modes; the per-class ones are generated ("P(class 2)").
_MODE_ID = "label id"
_MODE_UNCERTAINTY = "uncertainty"
# ...and, with an edge model, which regions neighbour voting flipped and
# how boundary-like each region is (its max p(diff) over its arcs).
_MODE_FLIPPED = "flipped by neighbours"
_MODE_PDIFF = "boundary p(diff)"

# Classifier kinds are shared by the dropdown, factory, and pickle loader so
# adding one cannot leave a trained model that the UI does not recognize.
_TUNED_KIND = "dense (tuned)"
# "custom FC": a dense net whose hidden sizes are typed on the Model tab.
# The "-> edges" kinds stack an edge model (edge_model.py: a pair classifier
# on the base net's hidden layer over the living-region graph, plus
# neighbour voting) on top of a base net; Train fits the base, then the
# edges. `_clf` is always the BASE pipeline, `_edge_model` the top.
_CUSTOM_KIND = "custom FC"
_TUNED_EDGE_KIND = "dense (tuned) -> edges"
_CUSTOM_EDGE_KIND = "custom FC -> edges"
_EDGE_KINDS = (_TUNED_EDGE_KIND, _CUSTOM_EDGE_KIND)
_MODEL_KINDS = ("random forest", "dense FC", "dense-top-16", "dense-top-32", _TUNED_KIND,
                _CUSTOM_KIND) + _EDGE_KINDS
_DEFAULT_CUSTOM_HIDDEN = "16-8"
_EDGE_LAM_RANGE = (0.0, 5.0)
_EDGE_ROUNDS_RANGE = (0, 10)
_EDGE_C_RANGE = (1e-3, 1e3)


def _is_edge_kind(kind):
    return kind in _EDGE_KINDS


def _base_kind(kind):
    """The base net a kind trains: the kind itself, or the one under `-> edges`."""
    if kind == _TUNED_EDGE_KIND:
        return _TUNED_KIND
    if kind == _CUSTOM_EDGE_KIND:
        return _CUSTOM_KIND
    return kind


def _edge_kind_of(base):
    return _TUNED_EDGE_KIND if base == _TUNED_KIND else (_CUSTOM_EDGE_KIND if base == _CUSTOM_KIND else None)
_DENSE_TOP_N = {"dense-top-16": 16, "dense-top-32": 32}
# The architectures behind the kinds. Module constants rather than literals
# inside _make_model so the Model tab's readout is formatted from the same
# values the estimator is built with and cannot drift from them.
_FOREST_TREES = 200
_OOB_MIN_SAMPLES = 20          # forest OOB score needs a few samples per tree
# ...and shared with model_search, whose search starts from exactly this net.
_MLP_HIDDEN = model_search.BASELINE_HIDDEN
_MLP_MAX_ITER = model_search.DEFAULT_MAX_ITER
# "Optimize network" defaults (the Model tab edits them; they ride the session
# view) and how often the Tk thread drains the worker's progress queue.
# An overnight-sized default: a trial is a 5-fold CV of one candidate, ~5-10 s
# with the search budget (SEARCH_MAX_ITER / SEARCH_PATIENCE), so 300 trials is
# under an hour and the 8 h limit is the real stop for a larger label set.
_SEARCH_TRIALS = 300
_SEARCH_TIMEOUT_MIN = 480                # the entry is in MINUTES
_SEARCH_TIMEOUT_S = _SEARCH_TIMEOUT_MIN * 60
_SEARCH_TRIALS_RANGE = (1, 100000)
_SEARCH_TIMEOUT_RANGE = (0, 7 * 86400)  # seconds; 0 = no time limit
_SEARCH_PUMP_MS = 150
# Trials per rung of the size sweep: enough for alpha / lr / batch / early
# stopping (and the feature mask) to settle at a fixed architecture.
_SWEEP_TRIALS = 20
_SWEEP_TRIALS_RANGE = (1, 10000)


def _hms(seconds):
    """`47s`, `12m 05s`, `3h 07m` for the search progress line."""
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def _model_description(kind, spec=None, n_features=None):
    """One-paragraph, read-only description of what `_make_model(kind)` builds.
    For the tuned kind that is the search winner `spec` (over `n_features`
    columns), or what Train falls back to while no search has run."""
    if kind == _TUNED_KIND:
        if spec is None:
            return ("No search yet: Train builds the dense FC baseline "
                    f"(FeatureSubset(all) -> StandardScaler -> MLPClassifier(hidden "
                    f"layers {_MLP_HIDDEN}, max_iter {_MLP_MAX_ITER})) with balanced "
                    "sample weights. Optimize network (O) searches depth, layer widths, "
                    "L2 alpha, learning rate, batch size, early stopping and a "
                    "per-channel feature subset by leave-slices-out cross-validation "
                    "(Optuna when installed, else random), then installs the winner. "
                    f"Backend: {model_search.backend_label('auto')}.")
        return spec.describe(n_features)
    if kind == "dense FC":
        return (f"StandardScaler -> MLPClassifier(hidden layers {_MLP_HIDDEN}, "
                f"max_iter {_MLP_MAX_ITER}, random_state 0). The scaler lives "
                "inside the pipeline, so the pickle predicts exactly as trained.")
    if kind in _DENSE_TOP_N:
        return (f"SelectFromModel(RandomForest, {_FOREST_TREES} trees, balanced) "
                f"keeps the top {_DENSE_TOP_N[kind]} features by importance "
                "(fewer if the profile has fewer) -> StandardScaler -> "
                f"MLPClassifier(hidden layers {_MLP_HIDDEN}, max_iter "
                f"{_MLP_MAX_ITER}, random_state 0).")
    return (f"RandomForestClassifier: {_FOREST_TREES} trees, class_weight "
            f"balanced, OOB score once >= {_OOB_MIN_SAMPLES} labeled regions, "
            "n_jobs -1, random_state 0.")


# The center notebook's tabs, in order. Named (not indexed) in the session
# view state so a reordered tab list still restores.
_CENTER_TABS = ("Processing", "View", "Model", "Analysis")
# Toolbar hint labels (workflow / model): link-blue, and how often the
# workflow text is re-snapshotted from the panel.
_HINT_COLOR = "#1a4fa0"
_HINT_POLL_MS = 700

# Statistics fields that are POSITIONS, not appearance: where a region sits in
# the slice says nothing about what material it is, and coordinate features
# were exactly what dragged k-means across the label boundary. Never fed to
# the classifier.
_NON_FEATURE_FIELDS = set(magic_fill.POSITIONAL_FIELDS)   # one list, shared with cosine

_TOOL_LABELS = (("squiggle", "squiggle"), ("box", "box"), ("polygon", "lasso"),
                ("magic", "magic"), ("blobber", "blobber"))
# What the tool selector offers (a superset of the STORED tools: "magic" and
# "blobber" are ways of producing "taps" interactions, not gesture types).
# Blobber ring-class choices: the class after the active one, or a fixed id.
_RING_CHOICES = ("next",) + tuple(str(k) for k in range(1, MAX_CLASSES))
# Magic-fill flood options: the per-hop cost multiplier (1 = pure bottleneck;
# clamped to [1, 2] because the cost is w * g^hops and hop depths reach the
# hundreds on a 3232^2 slice) and the drag sensitivity in screen px / region.
_DEFAULT_HOP_GAIN = 1.1
_HOP_GAIN_RANGE = (1.0, 2.0)
_DEFAULT_DRAG_PX = 4.0
_DRAG_PX_RANGE = (1.0, 64.0)


def _bounded_float(text, default, lo, hi):
    """float(text) clamped to [lo, hi], or `default` when it is not a number
    -- an option entry mid-edit must never raise out of a press."""
    try:
        v = float(str(text).strip())
    except (TypeError, ValueError):
        return default
    if v != v:                       # NaN
        return default
    return min(max(v, lo), hi)
_UI_TOOLS = tuple(v for v, _txt in _TOOL_LABELS)

# Focus-widget classes whose keystrokes must not arm classes (typing "1" into
# the persistence entry is not a request to arm class 1).
_TYPING_CLASSES = ("Entry", "TEntry", "Spinbox", "TSpinbox", "TCombobox",
                   "Listbox", "Text")


__all__ = ['_REGION_ALPHA', '_PRED_ALPHA', '_SCALAR_ALPHA', '_MODE_ID', '_MODE_UNCERTAINTY', '_MODE_FLIPPED', '_MODE_PDIFF', '_TUNED_KIND', '_CUSTOM_KIND', '_TUNED_EDGE_KIND', '_CUSTOM_EDGE_KIND', '_EDGE_KINDS', '_MODEL_KINDS', '_DEFAULT_CUSTOM_HIDDEN', '_EDGE_LAM_RANGE', '_EDGE_ROUNDS_RANGE', '_EDGE_C_RANGE', '_is_edge_kind', '_base_kind', '_edge_kind_of', '_DENSE_TOP_N', '_FOREST_TREES', '_OOB_MIN_SAMPLES', '_MLP_HIDDEN', '_MLP_MAX_ITER', '_SEARCH_TRIALS', '_SEARCH_TIMEOUT_MIN', '_SEARCH_TIMEOUT_S', '_SEARCH_TRIALS_RANGE', '_SEARCH_TIMEOUT_RANGE', '_SEARCH_PUMP_MS', '_SWEEP_TRIALS', '_SWEEP_TRIALS_RANGE', '_hms', '_model_description', '_CENTER_TABS', '_HINT_COLOR', '_HINT_POLL_MS', '_NON_FEATURE_FIELDS', '_TOOL_LABELS', '_RING_CHOICES', '_DEFAULT_HOP_GAIN', '_HOP_GAIN_RANGE', '_DEFAULT_DRAG_PX', '_DRAG_PX_RANGE', '_bounded_float', '_UI_TOOLS', '_TYPING_CLASSES']
