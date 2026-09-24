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

from . import config_io, fingerprints
from .config_io import _as_dict, _as_int, _as_list, _note, _opt_float

from msseg.labeler.session_doc import SESSION_DOC_VERSION  # noqa: F401
PROFILE_FILE_APP = "mscoupon-profile"
PROFILE_FILE_VERSION = 1


# --------------------------------------------------------------------------- #
# Compute profiles (JSON/writer-shaped dicts)
# --------------------------------------------------------------------------- #
def default_color_input() -> Dict[str, Any]:
    """How a multi-sample TIFF is read: the alpha policy, the colour->scalar
    method a chain without a leading `color` stage gets, and the plane count
    the statistics schema is resolved for (0 = grayscale / unknown; the GUI
    fills it from the slice on screen)."""
    return {"alpha": "drop", "default_method": "luminance", "channels": 0,
            "reduce_at": "front"}


def color_input_from_json(doc: Any) -> Dict[str, Any]:
    col = _as_dict(_as_dict(doc).get("color"))
    method = str(col.get("default_method") or "luminance")
    if method not in config_io.COLOR_METHODS:
        method = "luminance"
    return {"alpha": "keep" if col.get("alpha") == "keep" else "drop",
            "default_method": method,
            "channels": max(0, _as_int(col.get("channels"), 0)),
            # Where the conversion goes when the chain does not reduce itself.
            # `end` lets a chain lift all the way through, so `edges` runs per
            # plane and there is an RGB intermediate to look at.
            "reduce_at": "end" if col.get("reduce_at") == "end" else "front"}


# How persistence simplification is represented when a profile does not say:
# "merge_forest" (MSCEER's extremum merge forest -- no MSC is built during
# priming) or "msc" (the Morse-Smale complex + cancellation hierarchy). One
# name for every Python-side default and read fallback, because the last time
# this was a literal per site the GUI kept one that said "msc" and, since the
# GUI's value is written into the params JSON unconditionally, it overrode the
# C++ defaults and then baked itself into every saved session.
# Mirrors msseg::Msc2DParams::simplification and mscoupon::Config, which cannot
# be imported from here; `MSSEG_SIMPLIFICATION` overrides all of them at run time.
DEFAULT_SIMPLIFICATION = "merge_forest"

SIMPLIFICATIONS = ("merge_forest", "msc")


def default_profile(name: str = "default", relevance: bool = True) -> Dict[str, Any]:
    return {
        "name": str(name),
        "input": {"color": default_color_input()},
        "filters": [],
        "base_filters": [],
        "msc": {"manifold": "ascending", "persistence_percent": 10.0,
                "accurate": False, "extremum_sample_radius": 0,
                "use_gpu_gradient": False,
                "simplification": DEFAULT_SIMPLIFICATION},
        "statistics": config_io.statistics_to_json(
            [{"kind": "base"}], list(config_io.STAT_REDUCTIONS), True, 0,
            relevance),
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

    out["input"] = {"color": color_input_from_json(root.get("input"))}
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
    # Validated like the manifold, and for the same reason: an unrecognised
    # name reaches C++ as a plain string, compares unequal to "merge_forest"
    # and silently selects the MSC hierarchy -- a typo would cost a slower
    # prime with nothing anywhere saying why.
    simplification = str(msc.get("simplification") or DEFAULT_SIMPLIFICATION)
    if simplification not in SIMPLIFICATIONS:
        _note(notes, f"profile {out['name']}: unknown simplification "
                     f"{simplification!r} - using {DEFAULT_SIMPLIFICATION}")
        simplification = DEFAULT_SIMPLIFICATION
    pct = _opt_float(msc.get("persistence_percent"))
    out["msc"] = {
        "manifold": manifold,
        "persistence_percent": 10.0 if pct is None else float(pct),
        "accurate": bool(msc.get("accurate")
                         or msc.get("accurate_ascending")
                         or msc.get("accurate_descending")),
        "extremum_sample_radius": max(0, _as_int(msc.get("extremum_sample_radius"), 0)),
        "use_gpu_gradient": bool(msc.get("use_gpu_gradient")),
        "simplification": simplification,
    }

    stats = config_io.statistics_from_json(root.get("statistics"), notes)
    # `sources` rides through: a profile that declares one and loses it on load
    # is a profile whose channels stop resolving. There is no GUI editor for
    # them yet, so preserving what a profile brought is the whole contract.
    out["statistics"] = config_io.statistics_to_json(
        stats["channels"], stats["reductions"], stats["extremum"],
        out["msc"]["extremum_sample_radius"], stats["relevance"],
        stats.get("histogram"), stats.get("sources"))

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


def profile_params_json(profile: Dict[str, Any], cores: int = 1,
                        color_channels: Optional[int] = None) -> str:
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
    if m.get("use_gpu_gradient"):
        msc["use_gpu_gradient"] = True
    if m.get("simplification"):
        msc["simplification"] = str(m["simplification"])
    if cores > 1:
        msc["compute_algorithm"] = "partitioned"
        msc["requested_parallelism"] = int(cores)
    doc: Dict[str, Any] = {
        "filters": list(profile.get("filters") or []),
        "base_filters": list(profile.get("base_filters") or []),
        "msc": msc,
        "statistics": _as_dict(profile.get("statistics")) or
                      default_profile()["statistics"],
    }
    # `color_channels` (the planes of the slice on screen) overrides the
    # profile's declared count. The block is emitted only when it says
    # something non-default, so a grayscale workflow's params are unchanged.
    col = color_input_from_json(profile.get("input"))
    channels = int(color_channels) if color_channels else int(col["channels"])
    block: Dict[str, Any] = {}
    if col["alpha"] != "drop":
        block["alpha"] = col["alpha"]
    if col["default_method"] != "luminance":
        block["default_method"] = col["default_method"]
    if col.get("reduce_at") == "end":
        block["reduce_at"] = "end"
    if channels > 0:
        block["channels"] = channels
    if block:
        doc["input"] = {"color": block}
    return json.dumps(doc)


def field_fingerprint(profile: Dict[str, Any]) -> str:
    """What a primed slice's MSC and labels depend on (see fingerprints.py):
    two profiles with the same field fingerprint share their primes."""
    return fingerprints.field_fingerprint_of(profile_params_json(profile))


def measure_fingerprint(profile: Dict[str, Any]) -> str:
    """What a primed slice's statistics rows depend on: a profile that differs
    from the primed one only here costs a re-measure, not a re-prime."""
    return fingerprints.measure_fingerprint_of(profile_params_json(profile))


def profile_file_doc(profile: Dict[str, Any]) -> Dict[str, Any]:
    """The document written by 'Save profile…' (one profile per file)."""
    doc = {"app": PROFILE_FILE_APP, "version": PROFILE_FILE_VERSION}
    doc.update(profile)
    return doc


# The generic session document (folders, sequences, doc build/read, disk I/O)
# lives in the labeler framework; the names stay importable from here.
from msseg.labeler import session_doc as _session_doc
from msseg.labeler.session_doc import (dedupe_profile_name, folder_display_name,      # noqa: F401
                                       sequence_row_text, resolve_sequence_files,
                                       build_session_doc, _first_dict, is_session_doc)


def session_doc_from_json(doc: Any, notes: Optional[List[str]] = None) -> Dict[str, Any]:
    """Total reader: any dict-ish input -> a fully-populated v2 session dict
    whose profiles are coupon compute profiles (``profile_from_json``)."""
    return _session_doc.session_doc_from_json(doc, notes, profile_reader=profile_from_json,
                                              default_profile=default_profile)


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
            bool(state.get("stat_extremum", True)), radius,
            bool(state.get("stat_relevance", True)), None,
            state.get("stat_sources")),
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
    if isinstance(gui0.get("labels"), dict):     # v1 spelling, on the way in
        doc["annotations"] = gui0["labels"]
    return session_doc_from_json(doc, notes)


# --------------------------------------------------------------------------- #
# Compact workflow text, two lines: `topo field: base→b(1.5)→e(0.7)→msc(asc, 10%)`
# and `stats: base→norm(gmm)→12ch×4`
# --------------------------------------------------------------------------- #
# One short code per filter operation and the parameter that names the stage.
# The point is a one-line reminder of WHICH workflow is active, legible in a
# toolbar; the full card lives on the Processing tab.
OP_CODES = {
    "blur": "b", "derivative": "d", "laplacian": "lap", "zero_crossings": "zc",
    "hessian_eigenvalues": "hess", "structure_eigenvalues": "struct",
    "edges": "e", "erode": "ero", "dilate": "dil", "open": "open",
    "close": "close", "label_components": "cc", "normalize": "norm",
    "color": "col",
}
# The parameter shown in parentheses; "sigma" unless the operation has none.
HEADLINE_PARAM = {
    "structure_eigenvalues": "smoothing_sigma", "erode": "radius",
    "dilate": "radius", "open": "radius", "close": "radius",
    "label_components": "threshold", "normalize": "method", "color": "method",
}


def _fmt_param(v: Any) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return f"{float(v):g}"
    return str(v)


def stage_code(stage: Dict[str, Any]) -> str:
    """`b(1.5)` for a blur at sigma 1.5, `norm(gmm)`, `ero(2)`; "" for none."""
    op = str(stage.get("operation") or "none")
    if op == "none":
        return ""
    if op == "__msc__":
        return op
    code = OP_CODES.get(op, op)
    params = stage.get("params") or {}
    v = params.get(HEADLINE_PARAM.get(op, "sigma"))
    if v is None or v == "":
        return code
    return f"{code}({_fmt_param(v)})"


def chain_text(stages: Sequence[Dict[str, Any]], start: str = "base",
               arrow: str = "→") -> str:
    """`base→b(1.5)→e(0.7)`; `start` alone when the chain is empty."""
    codes = [c for c in (stage_code(s) for s in stages or []) if c]
    return arrow.join([start] + codes)


def msc_code(msc: Dict[str, Any]) -> str:
    """`msc(asc, 10%)`, with a trailing `mf` for the merge-forest simplifier."""
    manifold = "dsc" if str(msc.get("manifold", "ascending")).startswith("desc") else "asc"
    try:
        pct = f"{float(msc.get('persistence_percent', 10.0)):g}%"
    except (TypeError, ValueError):
        pct = "?%"
    args = [manifold, pct]
    if str(msc.get("simplification") or DEFAULT_SIMPLIFICATION) == "merge_forest":
        args.append("mf")
    return f"msc({', '.join(args)})"


def stats_width(statistics: Any, color_channels: int = 0) -> str:
    """`12ch×4`: how many channels a feature is measured on times the
    reductions -- the width of every per-feature row. A colour source counts
    its planes: the raw planes are `color_channels` channels, and a
    per-channel kind on the colour source yields one response per plane."""
    stats = config_io.statistics_from_json(statistics or {})
    planes = max(0, int(color_channels or 0))
    n_ch = 0
    for c in stats["channels"]:
        kind = c.get("kind")
        if kind == "color":
            n_ch += planes
            continue
        sig = c.get("sigmas") or []
        slots = 2 if kind in config_io.TWO_SLOT_KINDS else 1
        per_plane = (planes if (c.get("source") == "color"
                                and kind not in config_io.COLOR_ONLY_KINDS) else 1)
        n_ch += max(1, len(sig)) * slots * per_plane
    width = f"{n_ch}ch×{len(stats['reductions'])}"
    hist = stats.get("histogram") or {}
    if hist.get("bins"):
        # `+16h×2`: bins times the histogrammed channels (base alone by default).
        width += f"+{int(hist['bins'])}h×{max(1, len(hist.get('channels') or []))}"
    return width


def profile_summary(profile: Dict[str, Any]) -> str:
    """Two lines for a profile, one per pipeline:

        topo field: base→b(1.5)→e(0.7)→msc(asc, 10%)
        stats: base→norm(gmm)→12ch×4

    The first is the field the MSC runs on and the MSC setting; the second
    the channel statistics are measured on (the base chain) and the width of
    the resulting row."""
    topo = chain_text(list(profile.get("filters") or [])
                      + [{"operation": "__msc__"}])
    topo = topo.replace("→__msc__", "→" + msc_code(profile.get("msc") or {}))
    stats_stages = list(profile.get("base_filters") or [])
    try:
        width = stats_width(profile.get("statistics"),
                            color_input_from_json(profile.get("input"))["channels"])
    except Exception:
        width = "?ch"
    stats = chain_text(stats_stages) + "→" + width
    return f"topo field: {topo}\nstats: {stats}"
