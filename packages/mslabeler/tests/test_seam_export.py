"""Seam export (msseg.labeler.seam_export): per-item JSON and the summary CSV.

    pytest packages/mslabeler/tests/test_seam_export.py
"""
import csv
import json

import numpy as np

from msseg.labeler import seam_export as se
from msseg.labeler.labeling import Placement
from msseg.labeler.seams import SeamGraph, SEAM_BOUNDARY, SEAM_INTERIOR


def blocks():
    lab = np.zeros((12, 12), np.int32)
    lab[:6, 6:] = 1
    lab[6:, :6] = 2
    lab[6:, 6:] = 3
    return lab


def test_item_stem():
    assert se.item_stem("data/s0.tiff") == "data_s0.tiff"
    assert se.item_stem("slides/a.svs@0#12000,4000,4096,4096") == "slides_a.svs_0_12000_4000_4096_4096"
    assert se.item_stem("///") == "item"


def test_rows_json_and_csv(tmp_path):
    g = SeamGraph.from_labels(blocks(), np, placement=Placement((100, 200), 2.0))
    cls = np.array([SEAM_BOUNDARY, SEAM_INTERIOR, 0, SEAM_BOUNDARY], np.uint8)
    p = np.array([0.9, 0.1, np.nan, 0.8])
    rows = se.seam_rows(g, cls, p, np)
    assert [r["class_name"] for r in rows] == ["boundary", "interior", "unknown", "boundary"]
    assert rows[2]["boundaryness"] is None and rows[0]["boundaryness"] == 0.9
    assert rows[0]["length"] == 6 and len(rows[0]["points"]) == 2      # a straight run
    # Points are in image coordinates through the placement.
    x, y = g.seam_points(0)[0]
    assert rows[0]["points"][0] == [100 + 2 * x, 200 + 2 * y]
    path = tmp_path / "seams_x.json"
    doc = se.write_seams_json(str(path), "data/x.tiff", g, cls, p, np)
    back = json.loads(path.read_text(encoding="utf-8"))
    assert back == doc and back["item"] == "data/x.tiff" and back["shape"] == [12, 12]
    assert back["placement"] == {"origin": [100.0, 200.0], "scale": 2.0}
    assert back["n_seams"] == 4 and back["classes"][2] == "boundary"
    srows = se.summary_rows("data/x.tiff", doc)
    csv_path = tmp_path / "seams_summary.csv"
    se.write_summary_csv(str(csv_path), srows)
    with open(csv_path, newline="", encoding="utf-8") as f:
        got = list(csv.reader(f))
    assert got[0] == list(se.SUMMARY_COLUMNS) and len(got) == 5
    assert got[1][:4] == ["data/x.tiff", "0", "0", "1"] and got[3][7] == ""
    # No classes / no boundaryness at all still exports.
    assert se.seam_rows(g, None, None, np)[0]["class"] == 0
