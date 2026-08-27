"""Headless tests for the session document model (msseg.mscoupon.session).

No Tk, no compiled extension required (config_io degrades to its fallback
field list when the extension is absent).

    pytest packages/mscoupon/tests/test_session_model.py
"""
import os

from msseg.mscoupon import config_io, session
from msseg.mscoupon.session import (default_profile, profile_from_json,
                                    profile_params_json, profile_file_doc,
                                    dedupe_profile_name, folder_display_name,
                                    sequence_row_text, resolve_sequence_files,
                                    build_session_doc, is_session_doc,
                                    session_doc_from_json,
                                    legacy_docs_to_session)
import json


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #
def test_default_profile_shape():
    p = default_profile()
    assert p["name"] == "default"
    assert p["msc"]["manifold"] == "ascending"
    assert p["statistics"]["channels"] == ["base"]
    assert p["selection"]["connectivity"] == 6 and p["selection"]["min_area"] is None


def test_profile_round_trip_preserves_content():
    p = default_profile("A")
    p["filters"] = config_io.filters_to_json(
        [{"operation": "blur", "params": {"sigma": 2.0}}])
    p["msc"]["manifold"] = "descending"
    p["msc"]["persistence_percent"] = 25.0
    p["statistics"] = config_io.statistics_to_json(
        [{"kind": "base"}, {"kind": "blur", "sigmas": [0.7, 1.5]}],
        ["mean", "max"], True, 0)
    p["selection"]["min_area"] = 12
    back = profile_from_json(p)
    assert back == p


def test_profile_from_json_is_total_and_normalizing():
    notes = []
    p = profile_from_json({"name": "x", "msc": {"manifold": "sideways"},
                           "selection": {"connectivity": 7},
                           "filters": "not-a-list"}, notes)
    assert p["msc"]["manifold"] == "ascending"
    assert p["selection"]["connectivity"] == 6
    assert p["filters"] == []
    assert notes, "junk must be reported, not silently eaten"
    # Garbage in, default profile out.
    assert profile_from_json(None)["msc"]["persistence_percent"] == 10.0


def test_profile_drops_queries_outside_its_own_schema():
    p = default_profile()
    p["selection"]["feature_filters"] = [
        {"field": "mean_base", "op": "gt", "value": 1.0},
        {"field": "mean_blur_s9", "op": "gt", "value": 1.0}]   # not in base-only schema
    notes = []
    back = profile_from_json(p, notes)
    kept = [q["field"] for q in back["selection"]["feature_filters"]]
    assert "mean_base" in kept and "mean_blur_s9" not in kept


def test_profile_params_json_composition():
    p = default_profile()
    p["msc"]["accurate"] = True
    p["msc"]["extremum_sample_radius"] = 2
    doc = json.loads(profile_params_json(p, cores=1))
    assert set(doc) == {"filters", "base_filters", "msc", "statistics"}
    assert doc["msc"]["accurate_ascending"] and doc["msc"]["accurate_descending"]
    assert doc["msc"]["extremum_sample_radius"] == 2
    assert "compute_algorithm" not in doc["msc"]
    doc8 = json.loads(profile_params_json(p, cores=8))
    assert doc8["msc"]["compute_algorithm"] == "partitioned"
    assert doc8["msc"]["requested_parallelism"] == 8


def test_profile_file_doc_and_name_dedupe():
    d = profile_file_doc(default_profile("A"))
    assert d["app"] == session.PROFILE_FILE_APP and d["version"] == 1
    assert dedupe_profile_name("A", ["A", "A (2)"]) == "A (3)"
    assert dedupe_profile_name("B", ["A"]) == "B"


# --------------------------------------------------------------------------- #
# Folders / sequences
# --------------------------------------------------------------------------- #
def test_folder_display_name_collisions():
    assert folder_display_name(r"C:\data\spears", []) == "spears"
    assert folder_display_name(r"C:\other\spears", ["spears"]) == "other/spears"
    assert folder_display_name(r"D:\x\other\spears",
                               ["spears", "other/spears"]) == "x/other/spears"


def test_sequence_row_text():
    s = {"name": "n", "folder": "spears",
         "files": [r"C:\d\recon_00006.tiff", r"C:\d\recon_00042.tiff"]}
    assert sequence_row_text(s) == "spears  [recon_00006 – recon_00042] (2)"
    assert sequence_row_text({"folder": "f", "files": ["a.tif"]}) == "f  [a] (1)"


def test_resolve_sequence_files():
    folders = {"spears": {"path": r"C:\data\spears", "name": "spears"}}
    seq = {"name": "s", "folder": "spears", "files": ["a.tif", "b.tif"]}
    out = resolve_sequence_files(seq, folders)
    assert out == [os.path.join(r"C:\data\spears", "a.tif"),
                   os.path.join(r"C:\data\spears", "b.tif")]
    notes = []
    assert resolve_sequence_files({"folder": "gone", "files": ["a.tif"]},
                                  folders, notes) == []
    assert notes


# --------------------------------------------------------------------------- #
# Session doc
# --------------------------------------------------------------------------- #
def _sample_doc():
    return build_session_doc(
        app="mscoupon-labeler",
        folders=[{"path": r"C:\data\spears", "name": "spears"}],
        sequences=[{"name": "s1", "folder": "spears",
                    "files": [r"C:\data\spears\a.tif", r"C:\data\spears\b.tif"]}],
        profiles=[default_profile("P")],
        active_profile="P",
        run={"cores_per_slice": 4, "concurrent_slices": 1},
        view={"alpha": 0.5},
        labels={"version": 2, "n_classes": 3, "interactions": []},
        models=[{"path": r"C:\m\clf.pkl", "fingerprint": ["area"],
                 "kind": "random forest", "statistics": {}}])


def test_session_doc_round_trip():
    doc = _sample_doc()
    assert is_session_doc(doc) and not is_session_doc({"input": {}})
    back = session_doc_from_json(doc)
    assert back["folders"] == doc["folders"]
    assert back["sequences"][0]["files"] == ["a.tif", "b.tif"]   # basenames in doc
    assert back["active_profile"] == "P"
    assert back["run"] == {"cores_per_slice": 4, "concurrent_slices": 1}
    assert back["labels"]["n_classes"] == 3
    assert back["models"][0]["fingerprint"] == ["area"]


def test_session_doc_reader_is_total():
    back = session_doc_from_json({"folders": [{"nope": 1}],
                                  "sequences": [{"name": "empty"}],
                                  "profiles": "junk",
                                  "active_profile": "missing"})
    assert back["folders"] == [] and back["sequences"] == []
    assert len(back["profiles"]) == 1                 # default injected
    assert back["active_profile"] == back["profiles"][0]["name"]
    assert session_doc_from_json(None)["session_version"] == 2


def test_session_doc_dedupes_profile_names():
    doc = _sample_doc()
    doc["profiles"] = [default_profile("P"), default_profile("P")]
    back = session_doc_from_json(doc)
    assert [p["name"] for p in back["profiles"]] == ["P", "P (2)"]


# --------------------------------------------------------------------------- #
# Legacy import
# --------------------------------------------------------------------------- #
def _legacy_v1_session():
    files = [r"C:\data\spears\a.tif", r"C:\data\spears\b.tif"]
    cfg = config_io.build_config(
        files=files, output_folder=r"C:\out",
        filters=[{"operation": "blur", "params": {"sigma": 1.5}}],
        persistence_percent=20.0, manifold="descending",
        connectivity=18, min_area=9, cores_per_slice=4, concurrent_slices=2)
    gui = {"version": 1, "folder": r"C:\data\spears",
           "subsequences": [{"name": "one", "files": files},
                            {"name": "two", "files": [r"C:\data\tomo\c.tif"]}],
           "persist_live": "15", "alpha": 0.7,
           "labels": {"version": 1, "n_classes": 2, "interactions": []}}
    return config_io.build_session(cfg, gui)


def test_legacy_session_import():
    notes = []
    doc = legacy_docs_to_session([("last_session.json", _legacy_v1_session())], notes)
    assert is_session_doc(doc)
    names = [f["name"] for f in doc["folders"]]
    assert "spears" in names and "tomo" in names        # one folder per parent dir
    assert len(doc["sequences"]) == 2
    assert doc["sequences"][0]["folder"] == "spears"
    assert doc["sequences"][0]["files"] == ["a.tif", "b.tif"]
    p = doc["profiles"][0]
    assert p["name"] == "imported"
    assert p["msc"]["manifold"] == "descending"
    assert p["msc"]["persistence_percent"] == 20.0
    assert p["filters"] and p["filters"][0]["operation"] == "blur"
    assert p["selection"]["connectivity"] == 18 and p["selection"]["min_area"] == 9
    assert doc["run"] == {"cores_per_slice": 4, "concurrent_slices": 2}
    assert doc["view"]["persist_live"] == "15" and doc["view"]["alpha"] == 0.7
    assert doc["labels"]["n_classes"] == 2


def test_legacy_multi_config_import():
    cfg1 = config_io.build_config(files=[r"C:\d1\a.tif"], output_folder="o",
                                  filters=[])
    cfg2 = config_io.build_config(files=[r"C:\d2\b.tif"], output_folder="o",
                                  filters=[])
    doc = legacy_docs_to_session([("config_0.json", cfg1),
                                  ("config_1.json", cfg2)])
    assert len(doc["sequences"]) == 2 and len(doc["folders"]) == 2
    assert doc["sequences"][1]["files"] == ["b.tif"]


def test_legacy_unreadable_docs():
    notes = []
    doc = legacy_docs_to_session([("bad.json", None)], notes)
    assert doc["sequences"] == [] and notes
