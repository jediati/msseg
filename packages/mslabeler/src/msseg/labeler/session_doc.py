"""The session document: folders, sequences, profiles, view state -- and how it
reaches disk.

Generic across labelers. A session is a plain dict; every reader here is
*total* (never raises; problems append to ``notes``). What a *profile* is stays
the app's business -- ``session_doc_from_json`` takes the app's profile reader
and default factory as parameters -- and the ``annotations`` / ``models``
blocks are carried opaquely for the annotation shell.

The disk side (``app_data_dir`` .. ``write_session``) is stdlib-only: a
per-user config directory, a sorted-key serializer so an unchanged session
never looks changed, atomic writes and rotating backups for the auto-save.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
from typing import Any, Callable, Dict, List, Optional, Sequence

SESSION_DOC_VERSION = 2
# A document that carries ``tasks[]`` (several named detectors, each with its
# own annotations, models and Model-tab settings -- msseg.labeler.task)
# declares 3. Without tasks the document is the v2 one it always was.
SESSION_DOC_VERSION_TASKS = 3
# The Model-tab keys that belong to a task rather than to the window (the
# same tuple as ``task.TASK_VIEW_KEYS``; spelled here too so this module
# stays import-free of the task module, which imports it).
TASK_VIEW_KEYS = ("model_kind", "model_search", "neighbours", "context")


# --------------------------------------------------------------------------- #
# Tolerant readers (the config_io style: never raise, note problems)
# --------------------------------------------------------------------------- #
def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: Any, default: Any) -> Any:
    if isinstance(value, bool):
        return default
    try:
        return int(float(value))       # tolerate "3" and 3.0 alike
    except (TypeError, ValueError):
        return default


def _note(notes: Optional[List[str]], msg: str) -> None:
    if notes is not None:
        notes.append(msg)


def _passthrough_profile(doc: Any, notes: Optional[List[str]] = None) -> Dict[str, Any]:
    p = dict(_as_dict(doc))
    p["name"] = str(p.get("name") or "default")
    return p


# --------------------------------------------------------------------------- #
# Profiles, folders and sequences
# --------------------------------------------------------------------------- #
def dedupe_profile_name(name: str, taken: Sequence[str]) -> str:
    taken_set = set(taken)
    if name not in taken_set:
        return name
    i = 2
    while f"{name} ({i})" in taken_set:
        i += 1
    return f"{name} ({i})"


def new_task_uid(taken: Sequence[str] = ()) -> str:
    """A fresh task id, ``t_`` + six hex digits, not in `taken`. A task's
    identity is separate from its display name so a rename can never break a
    cross-reference (see ``msseg.labeler.task``)."""
    taken_set = set(taken)
    while True:
        uid = "t_" + secrets.token_hex(3)
        if uid not in taken_set:
            return uid


# --------------------------------------------------------------------------- #
# Folders and sequences
# --------------------------------------------------------------------------- #
def folder_display_name(path: str, taken: Sequence[str]) -> str:
    """Human-readable unique name for a folder: its basename, qualified with
    trailing parent parts only as needed to dodge a collision."""
    parts = [p for p in os.path.normpath(path).replace("\\", "/").split("/") if p]
    taken_set = set(taken)
    for depth in range(1, len(parts) + 1):
        name = "/".join(parts[-depth:])
        if name not in taken_set:
            return name
    # Everything collides (pathological); make it unique numerically.
    base = "/".join(parts) or "folder"
    return dedupe_profile_name(base, taken)


def sequence_row_text(seq: Dict[str, Any]) -> str:
    """`folder  [first – last] (n)` for the sequences listbox."""
    files = seq.get("files") or []
    def stem(p):
        return os.path.splitext(os.path.basename(p))[0]
    span = ""
    if files:
        span = (f"[{stem(files[0])}]" if len(files) == 1
                else f"[{stem(files[0])} – {stem(files[-1])}]")
    return f"{seq.get('folder', '?')}  {span} ({len(files)})"


def resolve_sequence_files(seq: Dict[str, Any],
                           folders_by_name: Dict[str, Dict[str, Any]],
                           notes: Optional[List[str]] = None) -> List[str]:
    """Doc-form sequence (basenames under a folder name) -> absolute paths.
    A missing folder resolves to nothing (with a note), never half a list."""
    folder = folders_by_name.get(str(seq.get("folder") or ""))
    if folder is None:
        _note(notes, f"sequence {seq.get('name')!r}: folder "
                     f"{seq.get('folder')!r} is not in the session - skipped")
        return []
    root = str(folder.get("path") or "")
    out = []
    for base in _as_list(seq.get("files")):
        if isinstance(base, str) and base:
            out.append(base if os.path.isabs(base) else os.path.join(root, base))
    return out


# --------------------------------------------------------------------------- #
# The session document
# --------------------------------------------------------------------------- #
def build_session_doc(*, app: str,
                      folders: Sequence[Dict[str, Any]],
                      sequences: Sequence[Dict[str, Any]],
                      profiles: Sequence[Dict[str, Any]],
                      active_profile: str,
                      run: Dict[str, Any],
                      view: Dict[str, Any],
                      annotations: Optional[Dict[str, Any]] = None,
                      models: Optional[Sequence[Dict[str, Any]]] = None,
                      tasks: Optional[Sequence[Dict[str, Any]]] = None,
                      active_task: Optional[str] = None) -> Dict[str, Any]:
    """The session document. With ``tasks`` (each a ``Task.to_doc()`` dict:
    uid / name / workflow / annotations / models / view) the document is v3
    and the per-task ``annotations`` / ``models`` live ONLY under their task
    -- one source of truth, so the top-level keys are not written even when
    passed. Without tasks it is the v2 document it always was, byte for
    byte: the viewers, which have no tasks, keep writing exactly that."""
    doc: Dict[str, Any] = {
        "app": str(app),
        "session_version": SESSION_DOC_VERSION_TASKS if tasks is not None else SESSION_DOC_VERSION,
        "folders": [{"path": str(f.get("path") or ""),
                     "name": str(f.get("name") or "")} for f in folders],
        "sequences": [{"name": str(s.get("name") or ""),
                       "folder": str(s.get("folder") or ""),
                       "files": [os.path.basename(p) for p in (s.get("files") or [])]}
                      for s in sequences],
        "profiles": [dict(p) for p in profiles],
        "active_profile": str(active_profile),
        "run": dict(run),
        "view": dict(view),
    }
    if tasks is not None:
        doc["tasks"] = [dict(t) for t in tasks]
        uids = [str(t.get("uid") or "") for t in tasks]
        doc["active_task"] = (str(active_task) if active_task is not None and str(active_task) in uids
                              else (uids[0] if uids else ""))
        return doc
    if annotations is not None:
        doc["annotations"] = annotations
    if models is not None:
        doc["models"] = [dict(m) for m in models]
    return doc


def _first_dict(*candidates: Any) -> Optional[Dict[str, Any]]:
    """The first candidate that is actually a dict, else None -- for reading a
    key that has been renamed, newest spelling first."""
    for c in candidates:
        if isinstance(c, dict):
            return c
    return None


def is_session_doc(doc: Any) -> bool:
    return isinstance(doc, dict) and _as_int(doc.get("session_version"), 0) >= 2


def session_doc_from_json(doc: Any, notes: Optional[List[str]] = None, *,
                          profile_reader: Optional[Callable[[Any, Optional[List[str]]], Dict[str, Any]]] = None,
                          default_profile: Optional[Callable[[], Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Total reader: any dict-ish input -> a fully-populated v2 session dict
    (folders/sequences/profiles normalized, at least one profile, a valid
    active_profile name).

    Profiles are opaque to the framework: ``profile_reader(doc, notes)`` turns a
    stored profile into the app's normalized form (a dict with a ``"name"``) and
    ``default_profile()`` supplies one when the session has none. The defaults
    pass the dict through, so a labeler with no compute profiles still gets a
    valid session; the coupon apps pass ``session.profile_from_json`` /
    ``session.default_profile``."""
    reader = profile_reader or _passthrough_profile
    factory = default_profile or (lambda: {"name": "default"})
    root = _as_dict(doc)

    folders: List[Dict[str, Any]] = []
    taken: List[str] = []
    for f in _as_list(root.get("folders")):
        fd = _as_dict(f)
        path = str(fd.get("path") or "")
        if not path:
            _note(notes, "folder entry without a path - skipped")
            continue
        name = str(fd.get("name") or "") or folder_display_name(path, taken)
        if name in taken:
            name = folder_display_name(path, taken)
        folders.append({"path": path, "name": name})
        taken.append(name)

    sequences: List[Dict[str, Any]] = []
    for s in _as_list(root.get("sequences")):
        sd = _as_dict(s)
        files = [str(b) for b in _as_list(sd.get("files")) if isinstance(b, str) and b]
        if not files:
            _note(notes, f"sequence {sd.get('name')!r} has no files - skipped")
            continue
        sequences.append({"name": str(sd.get("name") or ""),
                          "folder": str(sd.get("folder") or ""),
                          "files": files})

    profiles = [reader(p, notes) for p in _as_list(root.get("profiles"))]
    if not profiles:
        profiles = [factory()]
    names: List[str] = []
    for p in profiles:
        p["name"] = dedupe_profile_name(p["name"], names)
        names.append(p["name"])
    active = str(root.get("active_profile") or "")
    if active not in names:
        active = names[0]

    run = _as_dict(root.get("run"))
    # A copy: the tasks-less path below moves the Model-tab keys out of it.
    view = dict(_as_dict(root.get("view")))

    # Tasks. A v3 document lists them; anything older IS one task -- the
    # session's single annotations block, model registry and Model-tab
    # settings, named after the active profile -- so a reader never has to
    # know which it was given. Either way the result is a list of normalized
    # task entries and the uid of the active one.
    raw_tasks = [t for t in _as_list(root.get("tasks")) if isinstance(t, dict)]
    had_tasks = bool(raw_tasks)
    tasks: List[Dict[str, Any]] = []
    if had_tasks:
        uids: List[str] = []
        tnames: List[str] = []
        for td in raw_tasks:
            uid = str(td.get("uid") or "")
            if not uid or uid in uids:
                fresh = new_task_uid(uids)
                _note(notes, f"task {td.get('name')!r}: "
                             f"{'duplicate' if uid else 'missing'} uid - assigned {fresh}")
                uid = fresh
            name = dedupe_profile_name(str(td.get("name") or "task"), tnames)
            workflow = td.get("workflow")
            workflow = str(workflow) if workflow is not None else None
            if workflow is not None and workflow not in names:
                _note(notes, f"task {name!r}: workflow {workflow!r} is not in the "
                             f"session - using {active!r}")
                workflow = active
            tview = _as_dict(td.get("view"))
            tasks.append({"uid": uid, "name": name, "workflow": workflow,
                          "annotations": _first_dict(td.get("annotations"), td.get("labels")),
                          "models": _models_from_json(td.get("models")),
                          "view": {k: tview[k] for k in TASK_VIEW_KEYS if k in tview}})
            uids.append(uid)
            tnames.append(name)
        active_task = str(root.get("active_task") or "")
        if active_task not in uids:
            active_task = uids[0]
    else:
        tasks.append({"uid": new_task_uid(), "name": active, "workflow": active,
                      # "labels" is what sessions written before the rename
                      # call this, and it is the ONLY thing that key ever
                      # meant here (the raw gesture geometry); elsewhere in
                      # the tree "labels" is the MSC label raster.
                      "annotations": _first_dict(root.get("annotations"), root.get("labels")),
                      "models": _models_from_json(root.get("models")),
                      "view": {k: view.pop(k) for k in TASK_VIEW_KEYS if k in view}})
        active_task = tasks[0]["uid"]
    current = next(t for t in tasks if t["uid"] == active_task)

    return {
        "app": str(root.get("app") or ""),
        "session_version": SESSION_DOC_VERSION_TASKS if had_tasks else SESSION_DOC_VERSION,
        "folders": folders,
        "sequences": sequences,
        "profiles": profiles,
        "active_profile": active,
        "run": {"cores_per_slice": _as_int(run.get("cores_per_slice"), 0) or None,
                "concurrent_slices": _as_int(run.get("concurrent_slices"), 0) or None},
        "view": view,
        "tasks": tasks,
        "active_task": active_task,
        # The active task's, for readers that predate tasks.
        "annotations": current["annotations"],
        "models": current["models"],
    }


def _models_from_json(raw: Any) -> List[Dict[str, Any]]:
    """Saved-model records (``bundle.model_record_entry`` dicts), normalized;
    an entry without a path names nothing and is dropped."""
    models = []
    for m in _as_list(raw):
        md = _as_dict(m)
        if md.get("path"):
            models.append({"path": str(md["path"]),
                           "fingerprint": [str(n) for n in _as_list(md.get("fingerprint"))],
                           "kind": str(md.get("kind") or "random forest"),
                           "statistics": _as_dict(md.get("statistics")),
                           # The tuned dense spec (model_search.ModelSpec as a
                           # dict), opaque here; absent for the other kinds.
                           "spec": _as_dict(md.get("spec")) or None,
                           # Whether an edge model rides the pickle (v4).
                           "edge": bool(md.get("edge")),
                           # Only when present, so an entry without them is
                           # the document it always was.
                           **({"scope": str(md["scope"])} if md.get("scope") is not None else {}),
                           **({"context": _as_dict(md["context"])}
                              if isinstance(md.get("context"), dict) and md["context"] else {}),
                           **({"seam": True} if md.get("seam") else {})})
    return models


# --------------------------------------------------------------------------- #
# Disk: per-user config dir, serializer, atomic write, rotating backups
# --------------------------------------------------------------------------- #
def app_data_dir(app: str = "mscoupon") -> str:
    """Per-user config directory. Stdlib only -- one auto-saved file does not
    justify a dependency. Never raises, never creates."""
    base = os.environ.get("APPDATA") if os.name == "nt" else None
    if not base:
        base = os.environ.get("XDG_CONFIG_HOME")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, app)


def session_path(app: str = "mscoupon", name: str = "last_session.json") -> str:
    return os.path.join(app_data_dir(app), name)


def read_json_file(path: str) -> Optional[Dict[str, Any]]:
    """The parsed top-level object, or None when the file is missing,
    unreadable, malformed, or not an object. Never raises."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def serialize_session(doc: Dict[str, Any]) -> str:
    """`sort_keys` so key ordering can never make an unchanged session look
    changed -- the auto-save compares this text against the last one written."""
    return json.dumps(doc, indent=2, sort_keys=True)


def rotate_session_backups(path: str, keep: int = 3) -> None:
    """Roll `path` back one generation: .1 -> .2 -> ... -> .<keep>, then copy
    the live file to .1. Never raises, and never removes the live file.

    Auto-save is the only writer that runs without the user asking for it, so
    it is the one that needs an undo: a bad automatic write then costs the last
    `keep` states rather than everything. The copy (rather than a move) for the
    newest generation keeps `path` readable even if the write that follows
    fails."""
    if keep < 1 or not os.path.isfile(path):
        return
    root, ext = os.path.splitext(path)
    try:
        for i in range(keep, 1, -1):
            src = f"{root}.{i - 1}{ext}"
            if os.path.isfile(src):
                os.replace(src, f"{root}.{i}{ext}")
        shutil.copy2(path, f"{root}.1{ext}")
    except OSError:
        return          # a full or read-only directory must not stop the save


def write_session_text(text: str, path: str) -> bool:
    """Atomically replace `path` with `text`. False (never an exception) when the
    directory is unwritable, read-only, or full."""
    tmp = path + ".tmp"
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def write_session(doc: Dict[str, Any], path: str) -> bool:
    return write_session_text(serialize_session(doc), path)
