"""The classifier pickle (``ModelBundle``) and the profile-compatibility gate.

A saved classifier is one pickled dict::

    {"app": <tag>, "version": 4 | 5, "kind", "names", "model",
     "statistics",      # v2+: the statistics block it was trained under
     "spec",            # v3+: the tuned dense spec (ModelSpec.to_dict) or None
     "edge", "stack",   # v4:  the edge model (EdgeModel.to_dict) or None, and
                        #      the stack settings (custom hidden sizes, edge spec,
                        #      and -- only when the names carry neighbourhood
                        #      columns -- "context": the ContextSpec dict)
     "scope"}           # v5:  the app's feature scope, when it has one

A polyline task's seam model is its own pickle (``SeamBundle``): a region
pickle no longer carries one, and each loader refuses the other's file.

Reading is feature-detecting rather than version-gated, so every earlier
layout (v1: no kind/statistics; v2: no spec; v3: no edge/stack; v4: no scope)
still loads. The ``version`` field is written for humans and forward readers;
nothing here branches on it.

The compatibility gate is a SET comparison of feature names: the feature
matrix is assembled by name, so order never matters, but a model trained under
other statistics is refused outright -- per-feature values would silently mean
the wrong thing.

**Scope** covers what names cannot. Two models can carry identical feature
names and still be incomparable, because a name says what was measured and not
what it was measured on: a whole-slide labeler measures ``mean_blur_s1.5`` at
pyramid level 4 and at level 0, and a sigma is in pixels, so the level-4 number
describes a neighbourhood sixteen times wider. Nothing in the names records
that, and the values look perfectly plausible either way. An app with such a
distinction declares it as an opaque scope string and the gate refuses a
mismatch; an app without one (the coupon labeler) leaves it None and nothing
changes -- including the pickle, which grows the key only when a scope exists.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# A scope-less bundle is byte-identical to the v4 document it has always
# been -- the key exists only when there is a scope -- so the version it
# declares stays 4. Only a scoped one is v5, and it says so.
PICKLE_VERSION = 4
PICKLE_VERSION_SCOPED = 5
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
    scope: Optional[str] = None
    app_tag: str = DEFAULT_APP_TAG

    def to_doc(self) -> Dict[str, Any]:
        doc = {"app": self.app_tag,
               "version": PICKLE_VERSION if self.scope is None else PICKLE_VERSION_SCOPED,
               "kind": self.kind, "names": list(self.names),
               "model": self.model, "statistics": dict(self.statistics),
               "spec": self.spec, "edge": self.edge, "stack": dict(self.stack)}
        # Written only when there IS a scope, so an app without the distinction
        # keeps producing the document it always has.
        if self.scope is not None:
            doc["scope"] = str(self.scope)
        return doc

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
                   stack=dict(doc.get("stack") or {}),
                   scope=(str(doc["scope"]) if doc.get("scope") is not None else None),
                   app_tag=app_tag)

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
                                  self.spec, self.edge is not None, self.scope)


SEAM_BUNDLE_VERSION = 1


@dataclass
class SeamBundle:
    """A polyline task's seam model on disk: ``{"app", "version",
    "task_kind": "polyline", "seam": SeamModel.to_dict(), "statistics",
    "classes": {k: name}[, "scope"]}``. The class names are the task's at
    save time (display only: ids are the wire format)."""
    seam: Dict[str, Any]
    statistics: Dict[str, Any] = field(default_factory=dict)
    classes: Dict[int, str] = field(default_factory=dict)
    scope: Optional[str] = None
    app_tag: str = DEFAULT_APP_TAG

    def to_doc(self) -> Dict[str, Any]:
        doc = {"app": self.app_tag, "version": SEAM_BUNDLE_VERSION, "task_kind": "polyline",
               "seam": dict(self.seam), "statistics": dict(self.statistics),
               "classes": {str(int(k)): str(v) for k, v in self.classes.items()}}
        if self.scope is not None:
            doc["scope"] = str(self.scope)
        return doc

    @classmethod
    def from_doc(cls, doc: Any, app_tag: str = DEFAULT_APP_TAG) -> "SeamBundle":
        if (not isinstance(doc, dict) or doc.get("app") != app_tag
                or doc.get("task_kind") != "polyline" or not isinstance(doc.get("seam"), dict)):
            raise ValueError("not a polyline-task (seam) model file")
        classes = {}
        for k, v in (doc.get("classes") or {}).items():
            try:
                classes[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
        return cls(seam=dict(doc["seam"]), statistics=dict(doc.get("statistics") or {}),
                   classes=classes,
                   scope=(str(doc["scope"]) if doc.get("scope") is not None else None),
                   app_tag=app_tag)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.to_doc(), f)

    @classmethod
    def load(cls, path: str, app_tag: str = DEFAULT_APP_TAG) -> "SeamBundle":
        with open(path, "rb") as f:
            doc = pickle.load(f)
        return cls.from_doc(doc, app_tag)

    def record_entry(self, path: str) -> Dict[str, Any]:
        spec = self.seam.get("spec") or {}
        entry = model_record_entry(path, self.seam.get("feature_names") or [],
                                   f"seam {spec.get('model', 'logistic')}", self.statistics,
                                   None, False, self.scope)
        entry["task_kind"] = "polyline"
        return entry


def model_record_entry(path: str, names: Sequence[str], kind: str, statistics: Any,
                       spec: Optional[Dict[str, Any]], has_edge: bool,
                       scope: Optional[str] = None,
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A session ``models[]`` entry: where the pickle is, the feature
    fingerprint it needs, and enough provenance to describe it unloaded.
    `context` is the ContextSpec dict of the neighbourhood columns the
    fingerprint includes, written only when there are any."""
    entry = {"path": os.path.abspath(path),
             "fingerprint": list(names or []),
             "kind": kind,
             "statistics": dict(statistics or {}),
             "spec": spec,
             "edge": bool(has_edge)}
    if scope is not None:
        entry["scope"] = str(scope)
    if context:
        entry["context"] = dict(context)
    return entry


def compat_message(names: Sequence[str], expected: Sequence[str], profile_name: str,
                   context: str, scope: Optional[str] = None,
                   expected_scope: Optional[str] = None) -> Optional[str]:
    """None when `names` matches `expected` as a set AND the scopes agree; else
    the blocking message naming the exact mismatch (at most six names per side).

    A scope mismatch is reported on its own: the names are identical in that
    case, so listing them would say nothing about what is wrong.
    """
    if (scope or None) != (expected_scope or None):
        return (f"Model was trained at {scope or 'no scope'}, but profile "
                f"{profile_name!r} is at {expected_scope or 'no scope'} "
                f"({context}) - the feature names match but the measurements "
                f"are not comparable. Switch back, or retrain here.")
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
