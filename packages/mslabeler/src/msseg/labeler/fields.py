"""Column-name conventions of a per-region statistics table.

The framework never hard-codes what a column is called. A ``FieldConventions``
says which column carries the region id, which ones only *locate* a region
(and so never enter a feature vector), how a channel's mean/std columns and
histogram bins are named, and where the seeding extremum's position and value
live. ``DEFAULT`` is the coupon schema (``feature_id``, ``mean_<channel>``,
``hist<kk>_<channel>``, ``ext_x``/``ext_y``/``ext_filtered``); a derived
labeler with another table shape passes its own instance to the magic fill,
the training-set builder and the tools.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class FieldConventions:
    id_field: str = "feature_id"
    # The region's pixel count, when the table has one (the magic fill's HUD).
    area_field: Optional[str] = "area"
    # Columns that say WHERE a region is, not what it looks like: never part of
    # a feature vector or the cosine row.
    positional: FrozenSet[str] = frozenset({"feature_id", "min_x", "max_x", "min_y", "max_y",
                                            "ext_x", "ext_y"})
    # The seeding extremum: its pixel position and its value on the topology
    # field (the `barrier` metric and the edge model's saddle depth read it).
    extremum_xy: Optional[Tuple[str, str]] = ("ext_x", "ext_y")
    extremum_value_field: Optional[str] = "ext_filtered"
    mean_prefix: str = "mean_"
    std_prefix: str = "std_"
    # Histogram bins: one column per bin, group 1 = the channel name.
    hist_pattern: str = r"^hist\d+_(.+)$"
    # A channel listed first when present (the magic fill's default channel).
    preferred_channel: Optional[str] = "base"

    @property
    def hist_re(self):
        return re.compile(self.hist_pattern)       # re caches compiled patterns

    def mean_of(self, channel: str) -> str:
        return f"{self.mean_prefix}{channel}"

    def std_of(self, channel: str) -> str:
        return f"{self.std_prefix}{channel}"

    def channel_names(self, table) -> List[str]:
        """The measurement channels the table carries a mean column for, in
        table order (the preferred channel first when present)."""
        k = len(self.mean_prefix)
        names = [n[k:] for n in table.names if n.startswith(self.mean_prefix)]
        pref = self.preferred_channel
        if pref is not None and pref in names:
            names.remove(pref)
            names.insert(0, pref)
        return names

    def feature_names(self, names: Sequence[str]) -> List[str]:
        """`names` minus the positional columns: what a classifier may see."""
        return [n for n in names if n not in self.positional]

    def is_histogram_column(self, name: str) -> bool:
        return self.hist_re.match(name) is not None

    def histogram_columns(self, table):
        """channel -> [column names] of the table's histogram bins, in bin order."""
        out = {}
        rx = self.hist_re
        for n in table.names:
            m = rx.match(n)
            if m:
                out.setdefault(m.group(1), []).append(n)
        return out


DEFAULT = FieldConventions()
# The coupon spellings, kept as module constants for the code that grew up
# with them (magic_fill re-exports both).
POSITIONAL_FIELDS = DEFAULT.positional
HIST_RE = DEFAULT.hist_re
