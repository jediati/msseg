"""A task: one named detector's state, as plain data.

A labeler session used to hold exactly one of everything downstream of the
data -- one annotation store, one class vocabulary, one loaded classifier, one
active compute profile. A **task** is that bundle made into an object so a
session can hold several ("gland detector", "stroma detector", "bubble &
background"), each with its own gestures, classes, workflow and model stack,
over the same folders and sequences (docs/design_multi_model_tasks.md, §5;
stage 1 of §12).

What a task owns:

* ``workflow`` -- the NAME of a compute profile in the session's shared pool.
  Profiles stay session-level (two tasks that share one share its primes for
  free, later); the task only points at one.
* ``store`` -- its ``LabelStore``: region gestures, seam gestures, the class
  count, colours and names. Gestures are geometry, so a store never depends
  on the workflow it was drawn under.
* ``model`` -- the live estimator stack (``ModelStack``): the region
  classifier with its feature names / kind / spec / scope / context columns,
  the edge model and latent head stacked on it, the last search's winner, and
  the seam model.
* ``models`` -- the saved-pickle records the session keeps for this task
  (``bundle.model_record_entry`` dicts), newest last.
* ``view`` -- the Model-tab settings that describe the task's NEXT model
  (``TASK_VIEW_KEYS``), carried per task rather than per window.
* ``undo`` / ``redo`` -- the store's history, per task so Ctrl+Z can never
  resurrect a gesture into another task's vocabulary.
* ``caches`` -- the per-item prediction caches, which are functions of this
  task's store and model and would be wrong under any other.

Identity is a stable ``uid`` (``t_`` + six hex digits), separate from the
display ``name``: later stages cross-reference tasks (subscriptions, composite
rules, model folders) and a rename must not break them. Names are still kept
unique for display, with the profile rule.

Headless: no Tk, no numpy, nothing that is not already a dependency of
``labeling`` and ``context``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .context import ContextSpec
from .labeling import LabelStore
from .session_doc import TASK_VIEW_KEYS, dedupe_profile_name, new_task_uid

# The Model-tab state that is a property of the task's NEXT model, not of the
# window: the picked kind, the search settings, the neighbour (edge) settings
# and the picked context spec (``session_doc.TASK_VIEW_KEYS``). Everything
# else in the session ``view`` -- tool, magic rows, panes, tabs, windows --
# stays per session.
__all__ = ["TASK_VIEW_KEYS", "ModelStack", "Task", "TaskCaches", "dedupe_task_name",
           "new_uid"]

DEFAULT_TASK_NAME = "task"
DEFAULT_MODEL_KIND = "dense FC"

# One dedupe rule for the session: a task name is made unique the way a
# profile name is (``name``, ``name (2)``, ...); one id rule likewise.
dedupe_task_name = dedupe_profile_name
new_uid = new_task_uid


@dataclass
class ModelStack:
    """The live estimators of one task, as one object.

    The same nine fields the labeler used to stash across a New session, plus
    the seam model. ``context`` is never None -- the readers call
    ``.empty()`` on it -- so an unset context is an empty ``ContextSpec``.
    """
    clf: Any = None                      # the fitted region pipeline
    names: Optional[List[str]] = None    # its feature-column order
    kind: str = DEFAULT_MODEL_KIND
    spec: Any = None                     # model_search.ModelSpec (tuned kinds)
    scope: Optional[str] = None          # the regime it was fitted in
    context: ContextSpec = field(default_factory=ContextSpec)
    latent: Any = None                   # context.LatentContextModel
    edge: Any = None                     # edge_model.EdgeModel
    search_spec: Any = None              # the last Optimize winner
    seam: Any = None                     # seam_model.SeamModel

    @property
    def empty(self) -> bool:
        """No estimator at all -- nothing to classify or score with."""
        return self.clf is None and self.seam is None

    def reset(self) -> None:
        fresh = ModelStack()
        for f in fresh.__dataclass_fields__:
            setattr(self, f, getattr(fresh, f))


@dataclass
class TaskCaches:
    """Per-item results that depend on the task's store AND model."""
    pred: Dict[Any, Any] = field(default_factory=dict)       # (si, li) -> (commit, final, proba, aux)
    seam_pred: Dict[Any, Any] = field(default_factory=dict)  # key -> (commit, boundaryness[S])
    pred_store_rev: Any = None
    cm_cell: Any = None                                      # selected confusion cell
    # The catalogue's item keys when the caches were last valid: the caches
    # are keyed by row position, so a row layout that changed while the task
    # was inactive invalidates them on activation.
    keys_sig: Any = None

    def clear(self) -> None:
        self.pred.clear()
        self.seam_pred.clear()
        self.pred_store_rev = None
        self.cm_cell = None


@dataclass
class Task:
    uid: str
    name: str
    workflow: Optional[str] = None        # a profile name; None until bound
    store: LabelStore = field(default_factory=LabelStore)
    model: ModelStack = field(default_factory=ModelStack)
    models: List[Dict[str, Any]] = field(default_factory=list)
    view: Dict[str, Any] = field(default_factory=dict)
    undo: List[Dict[str, Any]] = field(default_factory=list)
    redo: List[Dict[str, Any]] = field(default_factory=list)
    caches: TaskCaches = field(default_factory=TaskCaches)
    # A restored task whose newest saved pickle exists on disk but has not
    # been loaded yet: pickles load on activation, because loading runs the
    # profile-compatibility gate against the ACTIVE workflow.
    model_pending: bool = False
    load_note: Optional[str] = None

    # -- construction ------------------------------------------------------ #
    @classmethod
    def new(cls, name: str, workflow: Optional[str] = None, n_classes: int = 3,
            uid: Optional[str] = None, taken: Sequence[str] = ()) -> "Task":
        return cls(uid=uid or new_uid(taken), name=str(name), workflow=workflow,
                   store=LabelStore(n_classes=n_classes))

    def duplicate(self, name: str, uid: Optional[str] = None,
                  taken: Sequence[str] = ()) -> "Task":
        """A new task with this one's vocabulary (count, colours, names),
        workflow and Model-tab settings -- and no gestures, no model, no
        history."""
        store = LabelStore(n_classes=self.store.n_classes)
        store.colors = dict(self.store.colors)
        store.names = dict(self.store.names)
        return Task(uid=uid or new_uid(taken), name=str(name), workflow=self.workflow,
                    store=store, view=_deep_copy_json(self.view))

    # -- counts for display ------------------------------------------------ #
    @property
    def gesture_count(self) -> int:
        return len(self.store.interactions) + len(self.store.seams)

    # -- the session document form ---------------------------------------- #
    def to_doc(self) -> Dict[str, Any]:
        doc: Dict[str, Any] = {"uid": self.uid, "name": self.name}
        if self.workflow is not None:
            doc["workflow"] = str(self.workflow)
        doc["annotations"] = self.store.to_json()
        doc["models"] = [dict(m) for m in self.models]
        doc["view"] = {k: self.view[k] for k in TASK_VIEW_KEYS if k in self.view}
        return doc

    @classmethod
    def from_doc(cls, d: Dict[str, Any], notes: Optional[List[str]] = None) -> "Task":
        """Total reader over a task entry as ``session_doc_from_json``
        normalises it (uid / name / workflow / annotations / models / view).
        A store that will not parse leaves the task with an empty one and a
        note, never an exception."""
        d = d if isinstance(d, dict) else {}
        name = str(d.get("name") or DEFAULT_TASK_NAME)
        uid = str(d.get("uid") or new_uid())
        workflow = d.get("workflow")
        workflow = str(workflow) if workflow is not None else None
        store = LabelStore()
        ann = d.get("annotations")
        if isinstance(ann, dict):
            try:
                store = LabelStore.from_json(ann)
            except Exception as exc:               # a malformed gesture row
                if notes is not None:
                    notes.append(f"task {name!r}: annotations not restored: {exc}")
        models = [dict(m) for m in (d.get("models") or []) if isinstance(m, dict)]
        view = d.get("view")
        view = {k: view[k] for k in TASK_VIEW_KEYS if k in view} if isinstance(view, dict) else {}
        pending = any(os.path.isfile(str(m.get("path") or "")) for m in models)
        return cls(uid=uid, name=name, workflow=workflow, store=store, models=models,
                   view=view, model_pending=pending)


def _deep_copy_json(value: Any) -> Any:
    """A structural copy of JSON-safe data (dicts / lists / scalars)."""
    if isinstance(value, dict):
        return {k: _deep_copy_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy_json(v) for v in value]
    return value
