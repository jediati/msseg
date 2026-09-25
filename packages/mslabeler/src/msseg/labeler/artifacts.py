"""Inputs: what a task consumes from somewhere else.

A polyline task labels boundaries, but the strongest evidence it has about a
boundary is often region work: two flanks annotated in different classes are
a boundary whether or not anyone traced it, a region net's hidden layer is a
better pair descriptor than raw statistics, and an edge model already says
p(different) for every arc. That work belongs to a region task; a polyline
task **subscribes** to it through named input **slots**:

========== ======================================= ==============================
slot       what it gives the polyline task          provider capability
========== ======================================= ==============================
labels     derived seam labels (flanks labelled     ``gestures_for(slide, rect)``
           differently -> the boundary class, the
           same -> class 1); shown as derived,
           never stored
embedding  the ``embed`` pair terms of the seam     ``base()`` -> (net, names)
           descriptor
pdiff      the ``edges`` pair feature and toll      ``pdiff_for(...)``
========== ======================================= ==============================

A slot holds a **provider reference**, not a provider: ``{"source": "task",
"uid": "t_ab12cd"}`` today, ``{"source": "library", "id": ...}`` once a model
artifact library exists. ``resolve`` turns a reference into a ``Provider``;
a reference that cannot be resolved (a deleted task, the unbuilt library)
resolves to a ``MissingProvider`` that says why, so a slot never silently
reads as empty. A provider declares what it **requires** of the consuming
workflow (feature columns + scope -- the same compatibility gate a loaded
model passes) and the slot shows the refusal rather than guessing.

A task provider is read WITHOUT activating it: its store is plain data, and
its model is either the live stack (trained or loaded this session) or its
newest saved pickle, read by ``load_model_stack`` into a stack of its own
(never into the task's, whose loading gates against the active workflow).

Headless: numpy is passed in, the estimator modules are imported lazily.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

SLOTS = ("labels", "embedding", "pdiff")
SLOT_TITLES = {"labels": "region labels", "embedding": "region embedding",
               "pdiff": "p(diff)"}
SLOT_HELP = {
    "labels": "A region task's annotations: seams between flanks annotated in "
              "different classes read as the boundary class below, between flanks "
              "of the same class as class 1. Derived, never stored; your own seam "
              "gestures overwrite them.",
    "embedding": "A region task's trained net: its hidden layer describes each "
                 "flank for the seam descriptor's embed terms (Features > Seam "
                 "descriptor, embed = net).",
    "pdiff": "A region task's edge model: p(the two flanks differ) per arc, the "
             "seam descriptor's p(diff) term and the edges toll.",
}
SOURCES = ("task", "library")
DEFAULT_BOUNDARY_CLASS = 2

__all__ = ["SLOTS", "SLOT_TITLES", "SLOT_HELP", "SOURCES", "DEFAULT_BOUNDARY_CLASS",
           "ProviderRef", "Requirement", "Provider", "TaskProvider", "MissingProvider",
           "normalise_inputs", "resolve", "load_model_stack", "boundary_class_of",
           "pdiff_for", "clear_stack_cache"]


# --------------------------------------------------------------------------- #
# References and the document form
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProviderRef:
    """Where a slot's input comes from: a task (by uid) or, later, a library
    artifact (by id)."""
    source: str
    id: str

    def to_doc(self) -> Dict[str, str]:
        return {"source": self.source,
                ("uid" if self.source == "task" else "id"): self.id}

    @classmethod
    def from_doc(cls, d: Any) -> Optional["ProviderRef"]:
        """The reference in a slot entry, or None when there is none."""
        if not isinstance(d, dict) or d.get("source") not in SOURCES:
            return None
        ident = d.get("uid") if d["source"] == "task" else d.get("id")
        if not isinstance(ident, str) or not ident:
            return None
        return cls(str(d["source"]), ident)

    @classmethod
    def task(cls, uid: str) -> "ProviderRef":
        return cls("task", str(uid))


def boundary_class_of(entry: Any) -> int:
    """The class a labels slot gives a derived boundary (>= 2: class 1 is the
    "not a boundary" role)."""
    try:
        k = int((entry or {}).get("boundary_class", DEFAULT_BOUNDARY_CLASS))
    except (TypeError, ValueError, AttributeError):
        return DEFAULT_BOUNDARY_CLASS
    return k if k >= 2 else DEFAULT_BOUNDARY_CLASS


def normalise_inputs(raw: Any, notes: Optional[List[str]] = None,
                     who: str = "task") -> Optional[Dict[str, Dict[str, Any]]]:
    """A task's ``inputs`` in normal form ``{slot: {"source", "uid"|"id"[,
    "boundary_class"]}}``, or None when there are none. Total: an unknown
    slot or an unusable reference is a note and is dropped. Whether a task
    uid still names a task is ``resolve``'s question (the slot shows it)."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        _note(notes, f"{who}: inputs are not a mapping - dropped")
        return None
    out: Dict[str, Dict[str, Any]] = {}
    for slot, entry in raw.items():
        ref = ProviderRef.from_doc(entry)
        if slot not in SLOTS or ref is None:
            _note(notes, f"{who}: unusable input {slot!r}: {entry!r} - dropped")
            continue
        e: Dict[str, Any] = ref.to_doc()
        if slot == "labels" and "boundary_class" in entry:
            e["boundary_class"] = boundary_class_of(entry)
        out[slot] = e
    return out or None


def _note(notes, text):
    if notes is not None:
        notes.append(text)


# --------------------------------------------------------------------------- #
# Requirements (the compatibility gate, for an input)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Requirement:
    """What a provider needs of the consuming workflow: the statistics
    columns its model reads, and the scope they were measured at."""
    columns: Tuple[str, ...] = ()
    scope: Optional[str] = None

    def why_not(self, have: Optional[Sequence[str]], scope: Optional[str] = None,
                workflow: str = "this workflow") -> Optional[str]:
        """None when a workflow producing `have` at `scope` can feed the
        provider; else the reason, naming the columns (at most four). `have`
        None means the app cannot say (no compiled extension): not refused."""
        if (self.scope or None) != (scope or None):
            return (f"measured at {self.scope or 'no scope'}, {workflow} at "
                    f"{scope or 'no scope'}")
        if have is None:
            return None
        missing = [c for c in self.columns if c not in set(have)]
        if not missing:
            return None
        more = f" (+{len(missing) - 4} more)" if len(missing) > 4 else ""
        return f"needs {', '.join(missing[:4])}{more}; {workflow} does not measure it"


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class Provider:
    """What fills a slot. Subclasses say what they can give (``capabilities``),
    what each capability needs (``requires``) and hand it over."""

    ref: ProviderRef
    name: str = "?"

    def capabilities(self) -> frozenset:
        return frozenset()

    def requires(self, slot: str) -> Requirement:
        return Requirement()

    def unavailable(self, slot: str) -> Optional[str]:
        """Why this provider cannot fill `slot` at all (before any workflow
        check), or None."""
        return None if slot in self.capabilities() else f"{self.name} has no {SLOT_TITLES[slot]}"

    def why_not(self, slot: str, have, scope=None, workflow="this workflow") -> Optional[str]:
        """Everything that stops the slot: the capability, then the gate."""
        why = self.unavailable(slot)
        if why:
            return why
        return self.requires(slot).why_not(have, scope, workflow)

    def signature(self, slot: str) -> Any:
        """What the slot's output is a function of -- a cache key."""
        return (self.ref, slot)

    # capabilities (overridden where offered)
    def gestures_for(self, slide, rect):
        return []

    def stack(self):
        return None

    def base(self):
        return None, None


class MissingProvider(Provider):
    """A reference that does not resolve; every slot says why."""

    def __init__(self, ref: ProviderRef, reason: str):
        self.ref = ref
        self.name = ref.id
        self.reason = reason

    def unavailable(self, slot: str) -> Optional[str]:
        return self.reason


class TaskProvider(Provider):
    """A region task as a provider, read without activating it."""

    def __init__(self, task, app_tag: Optional[str] = None):
        self.task = task
        self.ref = ProviderRef.task(task.uid)
        self.name = task.name
        self.app_tag = app_tag

    # -- the model, live or from its newest pickle ------------------------ #
    def stack(self):
        live = self.task.model
        if live.clf is not None:
            return live
        return load_model_stack(self.task, self.app_tag)

    def base(self):
        """``(net, names)`` of the task's region net when it can embed: it
        has hidden layers and reads plain table columns (a net over context
        columns would need the provider's neighbourhood built here)."""
        stack = self.stack()
        if stack is None or stack.clf is None:
            return None, None
        from . import seam_model
        if not seam_model.has_embedding(stack.clf):
            return None, None
        if stack.context is not None and not stack.context.empty():
            return None, None
        return stack.clf, list(stack.names or [])

    def capabilities(self) -> frozenset:
        caps = set()
        if self.task.kind == "region":
            caps.add("labels")
            net, _names = self.base()
            if net is not None:
                caps.add("embedding")
                if self.stack().edge is not None:
                    caps.add("pdiff")
        return frozenset(caps)

    def unavailable(self, slot: str) -> Optional[str]:
        if self.task.kind != "region":
            return f"{self.name} is a {self.task.kind} task"
        if slot == "labels":
            return None
        stack = self.stack()
        if stack is None or stack.clf is None:
            return f"{self.name} has no trained model"
        if slot in ("embedding", "pdiff") and self.base()[0] is None:
            if stack.context is not None and not stack.context.empty():
                return f"{self.name}'s model reads context columns (not usable as an input yet)"
            return f"{self.name}'s model has no hidden layer to embed with"
        if slot == "pdiff" and stack.edge is None:
            return f"{self.name} has no edge model (Model tab, an '-> edges' kind)"
        return None

    def requires(self, slot: str) -> Requirement:
        if slot == "labels":
            return Requirement()              # gestures are geometry
        stack = self.stack()
        if stack is None or stack.clf is None:
            return Requirement()
        return Requirement(tuple(stack.names or ()), stack.scope)

    def signature(self, slot: str) -> Any:
        if slot == "labels":
            return (self.ref, slot, self.task.store.rev)
        stack = self.stack()
        return (self.ref, slot, id(getattr(stack, "clf", None)),
                id(getattr(stack, "edge", None)))

    # -- labels ----------------------------------------------------------- #
    def gestures_for(self, slide, rect):
        """The region gestures the provider's store holds for an item (its
        slide, meeting its rect): gestures are slide-bound geometry, so they
        re-resolve against the consumer's decomposition, whatever workflow
        either task runs."""
        if slide is None:
            return []
        return self.task.store.for_item(slide, rect)


def resolve(ref: Optional[ProviderRef], tasks: Sequence[Any], *, app_tag: Optional[str] = None,
            consumer: Optional[str] = None) -> Optional[Provider]:
    """The provider behind `ref` among the session's `tasks` (None for no
    reference). `consumer` is the subscribing task's uid: a task cannot feed
    itself."""
    if ref is None:
        return None
    if ref.source == "library":
        return MissingProvider(ref, "the model library is not built yet")
    if ref.source != "task":
        return MissingProvider(ref, f"unknown source {ref.source!r}")
    if consumer is not None and ref.id == consumer:
        return MissingProvider(ref, "a task cannot be its own input")
    for t in tasks:
        if t.uid == ref.id:
            return TaskProvider(t, app_tag)
    return MissingProvider(ref, "that task was deleted")


# --------------------------------------------------------------------------- #
# Loading a task's model without activating it
# --------------------------------------------------------------------------- #
_STACKS: Dict[Tuple[str, float], Any] = {}


def clear_stack_cache() -> None:
    _STACKS.clear()


def load_model_stack(task, app_tag: Optional[str] = None):
    """A ``ModelStack`` from the task's newest saved region pickle that still
    exists, or None. Pure: nothing on the task changes and no gate runs (the
    slot gates the columns against the CONSUMING workflow). Cached by path and
    mtime, so a re-save is picked up and a read is paid once."""
    from . import bundle as model_bundle
    from . import context, edge_model
    from .task import ModelStack
    for entry in reversed(getattr(task, "models", None) or []):
        path = str(entry.get("path") or "")
        if entry.get("task_kind") == "polyline" or not os.path.isfile(path):
            continue
        key = (os.path.abspath(path), os.path.getmtime(path))
        if key in _STACKS:
            return _STACKS[key]
        try:
            doc = (model_bundle.ModelBundle.load(path, app_tag) if app_tag
                   else model_bundle.ModelBundle.load(path))
        except Exception:
            return None
        stack = ModelStack(clf=doc.model, names=list(doc.names), kind=doc.kind,
                           scope=doc.scope,
                           context=context.ContextSpec.from_dict(doc.stack.get("context")))
        if doc.edge is not None:
            try:
                em = edge_model.EdgeModel.from_dict(doc.edge)
                if em.names_hash == edge_model.names_hash(stack.names):
                    stack.edge = em
            except Exception:
                pass
        _STACKS[key] = stack
        return stack
    return None


# --------------------------------------------------------------------------- #
# p(diff) from a provider's edge model, on the consumer's record
# --------------------------------------------------------------------------- #
def pdiff_for(stack, table, arcs, labels, np):
    """p(the flanks differ) per arc of the consumer's record (NaN where an
    arc's regions have no row), by the provider's base net + edge model --
    what ``_predict_slice`` computes into ``aux["pdiff"]`` for its own task.
    None when the provider has no edge model or the item has no arcs; a
    ValueError when the table lacks the model's columns."""
    from . import context, edge_model, magic_fill
    from .training import TrainingSetBuilder
    edge = getattr(stack, "edge", None)
    if edge is None or arcs is None or not len(arcs.get("a", ())):
        return None
    names = list(stack.names or [])
    mat = TrainingSetBuilder.feature_matrix(table, names, np)
    fids = table.column("feature_id")
    if mat is None or fids is None:
        raise ValueError("the statistics lack the provider model's columns")
    fid = fids.astype(int)
    ia, ib, keep = magic_fill.index_arcs(arcs, fid, np)
    ext = mat[:, names.index("ext_filtered")] if "ext_filtered" in names else None
    sad = arcs.get("saddle")
    sad = None if sad is None else np.asarray(sad, np.float64)[keep]
    contact = None
    if "contact" in edge.spec.features:
        contact = context.ensure_contact(arcs, labels, np)[keep]
    rows = edge_model.predict_pdiff(edge, stack.clf, mat, ia, ib, sad, ext, contact=contact)
    out = np.full(len(arcs["a"]), np.nan, np.float32)
    out[keep] = rows
    return out
