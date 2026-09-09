"""The classifier pickle (``ModelBundle``) and the profile-compatibility gate.

A saved classifier is one pickled dict::

    {"app": <tag>, "version": 4, "kind", "names", "model",
     "statistics",      # v2+: the statistics block it was trained under
     "spec",            # v3+: the tuned dense spec (ModelSpec.to_dict) or None
     "edge", "stack"}   # v4:  the edge model (EdgeModel.to_dict) or None, and
                        #      the stack settings (custom hidden sizes, edge spec)

Reading is feature-detecting rather than version-gated, so every earlier
layout (v1: no kind/statistics; v2: no spec; v3: no edge/stack) still loads.
The ``version`` field is written for humans and forward readers; nothing here
branches on it.

The compatibility gate is a SET comparison of feature names: the feature
matrix is assembled by name, so order never matters, but a model trained under
other statistics is refused outright -- per-feature values would silently mean
the wrong thing.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

PICKLE_VERSION = 4
DEFAULT_APP_TAG = "mscoupon-labeler-classifier"


@dataclass
class ModelBundle:
    model: Any
    names: List[str]
    kind: str
    statistics: Dict[str, Any] = field(default_factory=dict)
    spec: Optional[Dict[str, Any]] = None
    edge: Optional[Dict[str, Any]] = None
    stack: Dict[str, Any] = field(default_factory=dict)
    app_tag: str = DEFAULT_APP_TAG

    def to_doc(self) -> Dict[str, Any]:
        return {"app": self.app_tag, "version": PICKLE_VERSION,
                "kind": self.kind, "names": list(self.names),
                "model": self.model, "statistics": dict(self.statistics),
                "spec": self.spec, "edge": self.edge, "stack": dict(self.stack)}

    @classmethod
    def from_doc(cls, doc: Any, app_tag: str = DEFAULT_APP_TAG) -> "ModelBundle":
        if (not isinstance(doc, dict) or doc.get("app") != app_tag
                or "model" not in doc or not doc.get("names")):
            raise ValueError("not a labeler classifier file")
        spec = doc.get("spec")
        edge = doc.get("edge")
        return cls(model=doc["model"], names=list(doc["names"]),
                   kind=str(doc.get("kind") or "random forest"),
                   statistics=dict(doc.get("statistics") or {}),
                   spec=spec if isinstance(spec, dict) else None,
                   edge=edge if isinstance(edge, dict) else None,
                   stack=dict(doc.get("stack") or {}), app_tag=app_tag)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.to_doc(), f)

    @classmethod
    def load(cls, path: str, app_tag: str = DEFAULT_APP_TAG) -> "ModelBundle":
        with open(path, "rb") as f:
            doc = pickle.load(f)
        return cls.from_doc(doc, app_tag)

    def record_entry(self, path: str) -> Dict[str, Any]:
        """The session's model record for this bundle saved at `path`."""
        return model_record_entry(path, self.names, self.kind, self.statistics,
                                  self.spec, self.edge is not None)


def model_record_entry(path: str, names: Sequence[str], kind: str, statistics: Any,
                       spec: Optional[Dict[str, Any]], has_edge: bool) -> Dict[str, Any]:
    """A session ``models[]`` entry: where the pickle is, the feature
    fingerprint it needs, and enough provenance to describe it unloaded."""
    return {"path": os.path.abspath(path),
            "fingerprint": list(names or []),
            "kind": kind,
            "statistics": dict(statistics or {}),
            "spec": spec,
            "edge": bool(has_edge)}


def compat_message(names: Sequence[str], expected: Sequence[str], profile_name: str,
                   context: str) -> Optional[str]:
    """None when `names` matches `expected` as a set; else the blocking
    message naming the exact mismatch (at most six names per side)."""
    want, have = set(names), set(expected)
    if want == have:
        return None
    missing = sorted(want - have)
    extra = sorted(have - want)
    parts = []
    if missing:
        parts.append("model needs: " + ", ".join(missing[:6])
                     + ("…" if len(missing) > 6 else ""))
    if extra:
        parts.append("profile adds: " + ", ".join(extra[:6])
                     + ("…" if len(extra) > 6 else ""))
    return (f"Model does not match profile '{profile_name}' statistics "
            f"({context}) - " + "; ".join(parts)
            + ". Switch profiles or retrain.")
