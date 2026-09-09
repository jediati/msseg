"""The columnar per-region statistics table every labeler component reads.

``FeatureTable`` is the Python mirror of ``mscoupon::FeatureTable``: the field
names once, then an ``(n_rows, n_fields)`` float64 block. Anything that needs
one column (a magic-fill metric, the training matrix, a hover readout) asks by
name and gets a view of the block -- nothing here builds a dict per region.
The framework's protocols type it as ``FeatureTableLike`` (duck-typed: ``names``,
``values``, ``column``, ``n_rows``, ``row_of_feature``), so a derived labeler
may hand in any object with that shape.
"""
from __future__ import annotations


class FeatureTable:
    """One slice's per-feature statistics, columnar: names once + an (n, f) block.

    Mirrors ``mscoupon::FeatureTable``. Nothing here builds a dict per feature --
    that is the whole point. With a twelve-channel scale-space stack a row is
    ~60 fields, so a dict per feature meant tens of thousands of Python strings
    and dict entries on every persistence commit; as a block it is one buffer.

    The one place a dict is still convenient is the hover readout, which shows a
    single feature, so ``row_of_feature`` builds exactly one.
    """

    __slots__ = ("names", "values", "_col", "_row_of_id")

    def __init__(self, names, values):
        self.names = names
        self.values = values
        self._col = {n: i for i, n in enumerate(names)}
        self._row_of_id = None

    @property
    def n_rows(self):
        return int(self.values.shape[0]) if self.values is not None else 0

    def column(self, name):
        """One field across every feature, or None if the spec excluded it."""
        i = self._col.get(name)
        return None if i is None else self.values[:, i]

    def row_of_feature(self, feature_id):
        """One feature's fields as a name -> value dict, or None if unknown.

        The id -> row index is built once, lazily: a hover that never happens
        should not cost a pass over the table.
        """
        if self._row_of_id is None:
            ids = self.column("feature_id")
            self._row_of_id = ({} if ids is None
                               else {int(v): r for r, v in enumerate(ids)})
        r = self._row_of_id.get(int(feature_id))
        if r is None:
            return None
        return {n: float(self.values[r, i]) for n, i in self._col.items()}

    def rows(self):
        """Every feature as a dict. Only for the fallback path against an
        extension too old to know the columnar evaluator."""
        return [{n: float(self.values[r, i]) for n, i in self._col.items()}
                for r in range(self.n_rows)]
