"""Session document model for the mscoupon GUIs (viewer + labeler).

A *session* is the top-level GUI object: the data folders, the sequences made
from them, the named compute profiles (one active), session-level run settings,
view state, and -- for the labeler -- the label interactions and model
references. It is a GUI-only artifact: ``config_io`` remains the AppConfig
boundary (what the C++ CLI parses) and supplies every payload writer/reader
used here, so a profile's ``filters``/``statistics``/``selection`` blocks are
byte-compatible with the corresponding AppConfig blocks.

Profiles are stored in their JSON (writer-shaped) form everywhere -- in memory,
in the session doc, and in single-profile files -- so snapshot/compare/save
are all plain dict operations. ``profile_from_json`` re-normalizes through the
``config_io`` readers, which makes every reader here total (never raises;
problems append to ``notes``), matching the ``config_io`` style.

The v2 session doc is NOT a runnable AppConfig (the v1 autosave was); legacy
v1 sessions and bare config.json files import through
``legacy_docs_to_session``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import config_io
from .config_io import _as_dict, _as_int, _as_list, _note, _opt_float

SESSION_DOC_VERSION = 2
PROFILE_FILE_APP = "mscoupon-profile"
PROFILE_FILE_VERSION = 1


# --------------------------------------------------------------------------- #
# Compute profiles (JSON/writer-shaped dicts)
# --------------------------------------------------------------------------- #
def default_profile(name: str = "default") -> Dict[str, Any]:
    return {
        "name": str(name),
        "filters": [],
        "base_filters": [],
        "msc": {"manifold": "ascending", "persistence_percent": 10.0,
                "accurate": False, "extremum_sample_radius": 0},
        "statistics": config_io.statistics_to_json(
            [{"kind": "base"}], list(config_io.STAT_REDUCTIONS), True, 0),
        "selection": {"feature_filters": [], "pixel_filters": [],
                      "connectivity": 6, "min_area": None},
    }


def profile_from_json(doc: Any, notes: Optional[List[str]] = None) -> Dict[str, Any]:
    """Total reader: any dict-ish input -> a normalized profile (writer-shaped).

    Runs everything through the config_io readers and back through the writers,
    so junk is dropped with a note and a loaded profile is always in the same
    canonical form a UI snapshot produces."""
    root = _as_dict(doc)
    out = default_profile(str(root.get("name") or "profile"))

    out["filters"] = config_io.filters_to_json(
        config_io.filters_from_json(root.get("filters"), notes))
    out["base_filters"] = config_io.filters_to_json(
        config_io.filters_from_json(root.get("base_filters"), notes))

    msc = _as_dict(root.get("msc"))
    manifold = str(msc.get("manifold") or "ascending")
    if manifold not in ("ascending", "descending"):
        _note(notes, f"profile {out['name']}: unknown manifold {manifold!r} - "
                     "using ascending")
        manifold = "ascending"
    pct = _opt_float(msc.get("persistence_percent"))
    out["msc"] = {
        "manifold": manifold,
        "persistence_percent": 10.0 if pct is None else float(pct),
        "accurate": bool(msc.get("accurate")
                         or msc.get("accurate_ascending")
                         or msc.get("accurate_descending")),
        "extremum_sample_radius": max(0, _as_int(msc.get("extremum_sample_radius"), 0)),
    }

    stats = config_io.statistics_from_json(root.get("statistics"), notes)
    out["statistics"] = config_io.statistics_to_json(
        stats["channels"], stats["reductions"], stats["extremum"],
        out["msc"]["extremum_sample_radius"])

    sel = _as_dict(root.get("selection"))
    # Feature filters validate against the schema THIS profile's statistics
    # produce, exactly like config_to_state does for a config document.
    fields = config_io.query_fields(
        json.dumps({"statistics": out["statistics"]}))
    conn = _as_int(sel.get("connectivity"), 6)
    if conn not in (6, 18, 26):
        _note(notes, f"profile {out['name']}: connectivity {conn!r} is not "
                     "6/18/26 - using 6")
        conn = 6
    min_area = sel.get("min_area")
    out["selection"] = {
        "feature_filters": config_io.queries_to_json(
            config_io.queries_from_json(sel.get("feature_filters"), fields, notes)),
        "pixel_filters": config_io.pixel_filters_to_json(
            config_io.pixel_filters_from_json(sel.get("pixel_filters"), notes)),
        "connectivity": conn,
        "min_area": None if min_area is None else max(0, _as_int(min_area, 0)),
    }
    return out


def profile_params_json(profile: Dict[str, Any], cores: int = 1) -> str:
    """THE composer of the priming params JSON: the profile's compute blocks
    plus the session-level core count. cores > 1 selects MSCEER's partitioned
    builder (compute_algorithm/requested_parallelism ride in `msc`, matching
    what the CLI's execution.threads_per_slice implies)."""
    m = _as_dict(profile.get("msc"))
    accurate = bool(m.get("accurate"))
    msc: Dict[str, Any] = {
        "manifold": str(m.get("manifold") or "ascending"),
        "persistence_percent": float(m.get("persistence_percent") or 10.0),
        "accurate_ascending": accurate,
        "accurate_descending": accurate,
    }
    radius = max(0, _as_int(m.get("extremum_sample_radius"), 0))
    if radius > 0:
        msc["extremum_sample_radius"] = radius
    if cores > 1:
        msc["compute_algorithm"] = "partitioned"
        msc["requested_parallelism"] = int(cores)
    return json.dumps({
        "filters": list(profile.get("filters") or []),
        "base_filters": list(profile.get("base_filters") or []),
        "msc": msc,
        "statistics": _as_dict(profile.get("statistics")) or
                      default_profile()["statistics"],
    })


def profile_file_doc(profile: Dict[str, Any]) -> Dict[str, Any]:
    """The document written by 'Save profile…' (one profile per file)."""
    doc = {"app": PROFILE_FILE_APP, "version": PROFILE_FILE_VERSION}
    doc.update(profile)
    return doc


def dedupe_profile_name(name: str, taken: Sequence[str]) -> str:
    taken_set = set(taken)
    if name not in taken_set:
        return name
    i = 2
    while f"{name} ({i})" in taken_set:
        i += 1
    return f"{name} ({i})"


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
                      labels: Optional[Dict[str, Any]] = None,
                      models: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    doc: Dict[str, Any] = {
        "app": str(app),
        "session_version": SESSION_DOC_VERSION,
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
    if labels is not None:
        doc["labels"] = labels
    if models is not None:
        doc["models"] = [dict(m) for m in models]
    return doc


def is_session_doc(doc: Any) -> bool:
    return isinstance(doc, dict) and _as_int(doc.get("session_version"), 0) >= 2


def session_doc_from_json(doc: Any, notes: Optional[List[str]] = None) -> Dict[str, Any]:
    """Total reader: any dict-ish input -> a fully-populated v2 session dict
    (folders/sequences/profiles normalized, at least one profile, a valid
    active_profile name)."""
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

    profiles = [profile_from_json(p, notes) for p in _as_list(root.get("profiles"))]
    if not profiles:
        profiles = [default_profile()]
    names: List[str] = []
    for p in profiles:
        p["name"] = dedupe_profile_name(p["name"], names)
        names.append(p["name"])
    active = str(root.get("active_profile") or "")
    if active not in names:
        active = names[0]

    run = _as_dict(root.get("run"))
    models = []
    for m in _as_list(root.get("models")):
        md = _as_dict(m)
        if md.get("path"):
            models.append({"path": str(md["path"]),
                           "fingerprint": [str(n) for n in _as_list(md.get("fingerprint"))],
                           "kind": str(md.get("kind") or "random forest"),
                           "statistics": _as_dict(md.get("statistics"))})

    return {
        "app": str(root.get("app") or ""),
        "session_version": SESSION_DOC_VERSION,
        "folders": folders,
        "sequences": sequences,
        "profiles": profiles,
        "active_profile": active,
        "run": {"cores_per_slice": _as_int(run.get("cores_per_slice"), 0) or None,
                "concurrent_slices": _as_int(run.get("concurrent_slices"), 0) or None},
        "view": _as_dict(root.get("view")),
        "labels": root.get("labels") if isinstance(root.get("labels"), dict) else None,
        "models": models,
    }


# --------------------------------------------------------------------------- #
# Legacy import: v1 sessions (AppConfig + "_gui") and bare config.json files
# --------------------------------------------------------------------------- #
def _profile_from_state(state: Dict[str, Any], name: str = "imported") -> Dict[str, Any]:
    """A profile from a config_to_state() dict (the v1 load path's shape)."""
    radius = max(0, _as_int(state.get("extremum_sample_radius"), 0))
    pct = state.get("persistence_percent")
    return profile_from_json({
        "name": name,
        "filters": config_io.filters_to_json(state.get("filters") or []),
        "base_filters": config_io.filters_to_json(state.get("base_filters") or []),
        "msc": {"manifold": state.get("manifold") or "ascending",
                "persistence_percent": 10.0 if pct is None else pct,
                "accurate": bool(state.get("accurate")),
                "extremum_sample_radius": radius},
        "statistics": config_io.statistics_to_json(
            state.get("stat_channels") or [{"kind": "base"}],
            state.get("stat_reductions") or list(config_io.STAT_REDUCTIONS),
            bool(state.get("stat_extremum", True)), radius),
        "selection": {
            "feature_filters": config_io.queries_to_json(state.get("feature_filters") or []),
            "pixel_filters": config_io.pixel_filters_to_json(state.get("pixel_filters") or []),
            "connectivity": state.get("connectivity") or 6,
            "min_area": state.get("min_area"),
        },
    })


def legacy_docs_to_session(docs: Sequence[Tuple[str, Any]],
                           notes: Optional[List[str]] = None) -> Dict[str, Any]:
    """[(path, parsed_doc_or_None)] from the v1 world -- one saved session, or a
    multi-select of exported config_N.json -- into ONE v2 session dict.

    Folders are derived from the distinct parent directories of the sequence
    files (D4-named); parameters come from the FIRST readable document, exactly
    like the old _apply_documents."""
    good = [(p, d) for p, d in docs if isinstance(d, dict)]
    for p, d in docs:
        if not isinstance(d, dict):
            _note(notes, f"could not read {os.path.basename(str(p))}")
    if not good:
        return session_doc_from_json({}, notes)

    cfg0, gui0 = config_io.split_session(good[0][1])
    state = config_io.config_to_state(cfg0, notes=notes)

    # Sequences: the _gui block's when a single session doc carries them,
    # else one sequence per document.
    raw: List[Tuple[Optional[str], List[str]]] = []
    saved = gui0.get("subsequences")
    if len(good) == 1 and isinstance(saved, list) and saved:
        for s in saved:
            if isinstance(s, dict):
                files = [f for f in _as_list(s.get("files"))
                         if isinstance(f, str) and f]
                raw.append((s.get("name"), files))
    else:
        for _path, d in good:
            cfg, _gui = config_io.split_session(d)
            one = config_io.config_to_state(cfg)
            raw.append((None, one["files"]))

    folders: List[Dict[str, Any]] = []
    by_dir: Dict[str, str] = {}          # parent dir -> folder name
    sequences: List[Dict[str, Any]] = []
    for name, files in raw:
        files = [f for f in files if f]
        if not files:
            continue
        parent = os.path.dirname(files[0]) or "."
        if parent not in by_dir:
            fname = folder_display_name(parent, [f["name"] for f in folders])
            folders.append({"path": parent, "name": fname})
            by_dir[parent] = fname
        sequences.append({"name": name or "", "folder": by_dir[parent],
                          "files": [os.path.basename(f) for f in files]})
    if not folders:
        legacy_folder = str(gui0.get("folder") or state.get("folder") or "")
        if legacy_folder:
            folders.append({"path": legacy_folder,
                            "name": folder_display_name(legacy_folder, [])})

    view = {k: gui0[k] for k in ("persist_live", "seg_source", "background",
                                 "mask", "alpha", "vmin", "vmax", "vmin_filt",
                                 "vmax_filt", "tool") if k in gui0}
    doc = {
        "app": "",
        "session_version": SESSION_DOC_VERSION,
        "folders": folders,
        "sequences": sequences,
        "profiles": [_profile_from_state(state)],
        "active_profile": "imported",
        "run": {"cores_per_slice": state.get("cores_per_slice"),
                "concurrent_slices": state.get("concurrent_slices")},
        "view": view,
    }
    if isinstance(gui0.get("labels"), dict):
        doc["labels"] = gui0["labels"]
    return session_doc_from_json(doc, notes)
