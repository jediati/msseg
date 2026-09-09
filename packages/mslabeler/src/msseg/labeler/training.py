"""Annotations + per-region statistics -> the classifier's training problem.

The labeler paints classes onto regions with gestures; the statistics table
describes every living region by a row of named columns. This module joins the
two into ``(X, y, groups, names)`` for the region model and into the row/edge
arrays the edge model consumes -- headless numpy, no Tk, no engine. It is
what used to live as methods on the Tk app, which is why the failure texts are
kept verbatim: the shell shows a ``TrainingProblem`` in the status bar.

``items`` are ``(key, record, table, group, label)`` tuples the shell gathers
from its region provider: the item key, the current record (``labels`` int32
raster, -1 = background), its ``FeatureTable``-like statistics, the value to
group rows by for leave-out cross-validation, and a short label for messages.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

from . import edge_model
from .fields import DEFAULT, FieldConventions
from .labeling import resolve_slice


class TrainingProblem(Exception):
    """The training set cannot be built; ``str(exc)`` is the status-bar text."""


class TrainingSetBuilder:
    """Turns items + a ``LabelStore`` into training arrays.

    ``conv`` names the table's columns (``FieldConventions``); its positional
    set -- the columns that identify or locate a region rather than describe
    it -- never enters the feature matrix (``non_feature_fields`` overrides
    that set alone). The magic fill's ``cosine`` metric excludes the same.
    """

    def __init__(self, conv: Optional[FieldConventions] = None,
                 non_feature_fields: Optional[Iterable[str]] = None):
        self.conv = conv or DEFAULT
        self.non_feature = frozenset(self.conv.positional if non_feature_fields is None
                                     else non_feature_fields)

    def feature_names(self, table) -> List[str]:
        return [n for n in table.names if n not in self.non_feature]

    @staticmethod
    def feature_matrix(table, names: Sequence[str], np):
        """(n_regions, len(names)) float64, or None if a column is missing."""
        cols = [table.column(n) for n in names]
        if any(c is None for c in cols):
            return None
        mat = np.stack(cols, axis=1).astype(np.float64)
        mat[~np.isfinite(mat)] = 0.0
        return mat

    @staticmethod
    def row_classes(interactions, labels, fids, np):
        """Per-row class (0 = unlabeled): the gestures resolved against the
        current raster, gathered by each row's region id."""
        rc = resolve_slice(interactions, labels, np)
        fid = np.asarray(fids).astype(int)
        ok = (fid >= 0) & (fid < len(rc))
        cls = np.zeros(len(fid), int)
        cls[ok] = rc[fid[ok]]
        return cls

    def labeled_set(self, items, store, np, id_field: Optional[str] = None):
        """``(X, y, groups, names)`` over every labeled region of every item."""
        id_field = id_field or self.conv.id_field
        X, y, g, names = [], [], [], None
        for key, rec, table, group, label in items:
            if names is None:
                names = self.feature_names(table)
            mat = self.feature_matrix(table, names, np)
            fids = table.column(id_field)
            if mat is None or fids is None:
                raise TrainingProblem(
                    f"Training stopped: incomplete statistics on slice {label}.")
            cls = self.row_classes(store.for_slice(key), rec["labels"], fids, np)
            m = cls > 0
            if m.any():
                X.append(mat[m])
                y.append(cls[m])
                g.append(np.full(int(m.sum()), group))
        if not X:
            raise TrainingProblem("No labeled regions on computed slices - "
                                  "draw some (and Run/Rerun) first.")
        X = np.concatenate(X)
        y = np.concatenate(y)
        g = np.concatenate(g)
        if len(set(y.tolist())) < 2:
            raise TrainingProblem("Need labels from at least 2 classes to train.")
        return X, y, g, names

    def edge_set(self, items, store, names: Sequence[str], arcs_of: Callable[[Any], Any], np,
                 id_field: Optional[str] = None, ext_field: Optional[str] = None):
        """Every region of every item (class 0 = unlabeled) with its group and
        extremum value, plus the region-graph edges as global row pairs:
        ``(X, cls, groups, ext | None, edges, names)``. ``arcs_of(key, record)``
        supplies the item's arcs (MSC saddles or pixel adjacency)."""
        id_field = id_field or self.conv.id_field
        ext_field = ext_field or self.conv.extremum_value_field
        names = list(names)
        ext_col = names.index(ext_field) if ext_field in names else None
        X, cls, grp, ext, per = [], [], [], [], []
        for key, rec, table, group, label in items:
            mat = self.feature_matrix(table, names, np)
            fids = table.column(id_field)
            if mat is None or fids is None:
                raise TrainingProblem(f"Edges stopped: incomplete statistics on slice {label}.")
            c = self.row_classes(store.for_slice(key), rec["labels"], fids, np)
            fid = np.asarray(fids).astype(int)
            X.append(mat); cls.append(c); grp.append(np.full(len(c), group))
            if ext_col is not None:
                ext.append(mat[:, ext_col])
            per.append((fid, arcs_of(key, rec), c, group))
        edges = edge_model.gather_edges(per)
        return (np.concatenate(X), np.concatenate(cls), np.concatenate(grp),
                np.concatenate(ext) if ext else None, edges, names)
