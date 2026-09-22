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
        # Per-item row classes from the last build, keyed by item: the
        # gestures resolved against the decomposition change only when the
        # item's annotations or its record do, and a retrain is usually
        # neither for most items. See `_classes_for`.
        self._class_memo: dict = {}

    def feature_names(self, table) -> List[str]:
        return [n for n in table.names if n not in self.non_feature]

    @staticmethod
    def feature_matrix(table, names: Sequence[str], np, rows=None):
        """(n_rows, len(names)) float64, or None if a column is missing.

        `rows` (a boolean mask or an index array over the table's rows)
        restricts the copy to those rows -- the labeled few hundred rather
        than every region on a slide. A columnar table (``values`` is the
        (n, f) block) is gathered in one indexing pass; anything else goes
        column by column through ``column()``."""
        values = getattr(table, "values", None)
        tnames = getattr(table, "names", None)
        if values is not None and tnames is not None and getattr(values, "ndim", 0) == 2:
            pos = {n: i for i, n in enumerate(tnames)}
            idx = [pos.get(n) for n in names]
            if any(i is None for i in idx):
                return None
            block = values if rows is None else values[rows]
            mat = np.asarray(block[:, idx], dtype=np.float64)
            if mat.base is not None:                     # a view of the table: copy
                mat = np.array(mat, dtype=np.float64)
        else:
            cols = [table.column(n) for n in names]
            if any(c is None for c in cols):
                return None
            if rows is not None:
                cols = [c[rows] for c in cols]
            mat = np.stack(cols, axis=1).astype(np.float64)
        mat[~np.isfinite(mat)] = 0.0
        return mat

    @staticmethod
    def row_classes(interactions, labels, fids, np, layer=None):
        """Per-row class (0 = unlabeled): the gestures resolved against the
        current decomposition, gathered by each row's region id.

        `layer` is the item's ``LabelLayer`` when the gestures' coordinates are
        not the raster's indices; `labels` then only sizes the result."""
        rc = resolve_slice(interactions, labels, np, layer)
        fid = np.asarray(fids).astype(int)
        ok = (fid >= 0) & (fid < len(rc))
        cls = np.zeros(len(fid), int)
        cls[ok] = rc[fid[ok]]
        return cls

    @staticmethod
    def _annotation_signature(interactions):
        """What a resolved class vector depends on from the store side: the
        interactions' identities, classes and shapes. Undo, a repaint or a
        class change on the item all change it; a retrain after edits
        elsewhere does not."""
        def meta_sig(it):
            # The meta keys that change how the geometry is applied off its
            # drawing level (labeling.py); a rebind may set them in place.
            m = getattr(it, "meta", None) or {}
            outline = m.get("outline") or ()
            return (m.get("level"), m.get("scale"), m.get("px"), len(outline))
        return tuple((getattr(it, "uid", None), getattr(it, "class_id", None),
                      getattr(it, "tool", None), len(getattr(it, "points", ()) or ()),
                      meta_sig(it))
                     for it in interactions)

    def _classes_for(self, key, rec, table, interactions, fids, np, layer):
        """`row_classes` for the item, memoized on (record, annotations, table).

        The gestures are re-resolved against the raster only when the item's
        own annotations or its record changed; every other item on a retrain
        is a dictionary hit. The record is identified by its commit AND its
        label raster (a re-prime at the same commit is a new raster)."""
        labels = rec.get("labels")
        sig = (rec.get("commit"), id(labels), id(table), len(fids),
               self._annotation_signature(interactions))
        hit = self._class_memo.get(key)
        if hit is not None and hit[0] == sig:
            return hit[1]
        cls = self.row_classes(interactions, labels, fids, np, layer)
        self._class_memo[key] = (sig, cls)
        return cls

    def forget(self, key=None):
        """Drop the memoized classes for `key` (None: every item) and the
        edge set's rows, which span every item."""
        self._edge_memo = None
        if key is None:
            self._class_memo.clear()
        else:
            self._class_memo.pop(key, None)

    def labeled_set(self, items, store, np, id_field: Optional[str] = None,
                    layer_of=None, gestures_of=None):
        """``(X, y, groups, names)`` over every labeled region of every item.

        `layer_of(key, rec)` supplies the item's ``LabelLayer`` when its region
        ids are not addressed by the gestures' own coordinates (the whole-slide
        case); the same shape as `edge_set`'s `arcs_of`."""
        id_field = id_field or self.conv.id_field
        X, y, g, names = [], [], [], None
        for key, rec, table, group, label in items:
            if names is None:
                names = self.feature_names(table)
            fids = table.column(id_field)
            if fids is None:
                raise TrainingProblem(
                    f"Training stopped: incomplete statistics on slice {label}.")
            cls = self._classes_for(key, rec, table, (gestures_of or store.for_slice)(key),
                                    fids, np, None if layer_of is None else layer_of(key, rec))
            m = cls > 0
            if not m.any():
                if self.feature_matrix(table, names, np, rows=slice(0, 0)) is None:
                    raise TrainingProblem(
                        f"Training stopped: incomplete statistics on slice {label}.")
                continue
            # Only the labeled rows are copied out: a slide item has hundreds
            # of thousands of regions and a few hundred labels.
            mat = self.feature_matrix(table, names, np, rows=m)
            if mat is None:
                raise TrainingProblem(
                    f"Training stopped: incomplete statistics on slice {label}.")
            X.append(mat)
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

    # The all-rows design matrix and the edge structure of the last edge_set,
    # kept across retrains. They depend on the records (a commit, a label
    # raster, a table) and the column names -- not on the annotations -- so an
    # annotation pass reuses them and only the per-row classes move. Held up
    # to EDGE_CACHE_BYTES of matrix (a whole-slide session's is ~1 GB, the
    # records' own tables are as much again); over that, rebuilt each time.
    EDGE_CACHE_BYTES = 2 << 30
    _edge_memo = None

    def edge_set(self, items, store, names: Sequence[str], arcs_of: Callable[[Any], Any], np,
                 id_field: Optional[str] = None, ext_field: Optional[str] = None,
                 layer_of=None, gestures_of=None):
        """Every region of every item (class 0 = unlabeled) with its group and
        extremum value, plus the region-graph edges as global row pairs:
        ``(X, cls, groups, ext | None, edges, names)``. ``arcs_of(key, record)``
        supplies the item's arcs (MSC saddles or pixel adjacency).

        The rows and the edge structure are memoized (see `_edge_memo`): a
        retrain after an annotation pass re-resolves the classes of the items
        whose annotations changed and recomputes which edges are labeled;
        nothing else. A re-prime, a persistence change, a different item
        list or column set, or arcs that gained their contact lengths since,
        rebuilds everything."""
        id_field = id_field or self.conv.id_field
        ext_field = ext_field or self.conv.extremum_value_field
        names = list(names)
        ext_col = names.index(ext_field) if ext_field in names else None
        items = list(items)
        sig_parts, cls, per_fids, per_arcs = [], [], [], []
        for key, rec, table, group, label in items:
            fids = table.column(id_field)
            if fids is None:
                raise TrainingProblem(f"Edges stopped: incomplete statistics on slice {label}.")
            arcs = arcs_of(key, rec)
            sig_parts.append((key, id(table), rec.get("commit"), id(rec.get("labels")), group,
                              id(arcs), None if arcs is None else "length" in arcs))
            c = self._classes_for(key, rec, table, (gestures_of or store.for_slice)(key),
                                  fids, np, None if layer_of is None else layer_of(key, rec))
            cls.append(c)
            per_fids.append(np.asarray(fids).astype(int))
            per_arcs.append(arcs)
        sig = (tuple(sig_parts), tuple(names))
        cls_all = np.concatenate(cls) if cls else np.zeros(0, int)

        memo = self._edge_memo
        if memo is not None and memo[0] == sig:
            _sig, X_all, grp_all, ext_all, edges0 = memo
        else:
            X, grp, ext, per = [], [], [], []
            for (key, rec, table, group, label), fid, arcs, c in zip(items, per_fids, per_arcs, cls):
                mat = self.feature_matrix(table, names, np)
                if mat is None:
                    raise TrainingProblem(f"Edges stopped: incomplete statistics on slice {label}.")
                X.append(mat); grp.append(np.full(len(c), group))
                if ext_col is not None:
                    ext.append(mat[:, ext_col])
                per.append((fid, arcs, c, group))
            edges0 = edge_model.gather_edges(per)
            X_all = np.concatenate(X) if X else np.zeros((0, len(names)))
            grp_all = np.concatenate(grp) if grp else np.zeros(0, int)
            ext_all = np.concatenate(ext) if ext else None
            self._edge_memo = ((sig, X_all, grp_all, ext_all, edges0)
                               if X_all.nbytes <= self.EDGE_CACHE_BYTES else None)
        # The labeled-ness of an edge is the only thing here that follows the
        # annotations: recomputed from the current classes, the same way
        # gather_edges computes it, into a fresh dict (callers may edit it).
        a, b = edges0["a"], edges0["b"]
        edges = dict(edges0)
        edges["both"] = (cls_all[a] > 0) & (cls_all[b] > 0) if len(a) else np.zeros(0, bool)
        edges["diff"] = (edges["both"] & (cls_all[a] != cls_all[b]) if len(a)
                         else np.zeros(0, bool))
        return X_all, cls_all, grp_all, ext_all, edges, names
