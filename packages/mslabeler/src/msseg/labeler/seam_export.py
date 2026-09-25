"""Exporting classified seams: one JSON per item with every seam's flanks,
class, boundaryness and polyline (image coordinates, collinear runs
compressed), plus one summary CSV over all items. What a later step will
rasterize with a width to train an image model."""
from __future__ import annotations

import csv
import json
import re

from .seam_path import compress_collinear
from .seams import SEAM_CLASSES


def item_stem(key: str) -> str:
    """A file-name-safe stem for an item key (``data/s0.tiff`` ->
    ``data_s0.tiff``)."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(key)).strip("_")
    return stem or "item"


def class_name(k, names=None) -> str:
    k = int(k)
    if names is not None:
        return str(names.get(k, "unknown" if k == 0 else f"class {k}"))
    return SEAM_CLASSES[k] if 0 <= k < len(SEAM_CLASSES) else str(k)


def seam_rows(graph, seam_class, boundaryness, np, names=None, predicted=None):
    """One dict per seam: ``id, a, b, j0, j1, length, class, class_name,
    boundaryness, points`` (points in image coordinates), plus
    ``predicted`` when the task's model has classified the item. `names`
    is the task's {class id: name}; None = the old two-class names."""
    cls = np.asarray(seam_class, np.int64) if seam_class is not None else np.zeros(graph.n_seams, np.int64)
    p = None if boundaryness is None else np.asarray(boundaryness, np.float64)
    lengths = graph.lengths(np)
    rows = []
    for i in range(graph.n_seams):
        pts = compress_collinear(graph.seam_points(i).tolist())
        rows.append({"id": i, "a": int(graph.a[i]), "b": int(graph.b[i]),
                     "j0": int(graph.j0[i]), "j1": int(graph.j1[i]),
                     "length": int(lengths[i]), "class": int(cls[i]),
                     "class_name": class_name(cls[i], names),
                     **({} if predicted is None else {"predicted": int(predicted[i])}),
                     "boundaryness": (None if p is None or not np.isfinite(p[i]) else round(float(p[i]), 6)),
                     "points": [[float(x), float(y)] for x, y in graph.to_image(pts)]})
    return rows


def item_doc(key, graph, seam_class, boundaryness, np, names=None, predicted=None):
    pl = graph.placement
    classes = (list(SEAM_CLASSES) if names is None
               else ["unknown"] + [class_name(k, names) for k in range(1, max(names) + 1)]
               if names else ["unknown"])
    return {"item": str(key), "shape": [int(graph.height), int(graph.width)],
            "placement": {"origin": [pl.ox, pl.oy], "scale": pl.scale},
            "classes": classes,
            "n_seams": int(graph.n_seams), "n_junctions": int(graph.n_junctions),
            "seams": seam_rows(graph, seam_class, boundaryness, np, names, predicted)}


def write_seams_json(path, key, graph, seam_class, boundaryness, np, names=None,
                     predicted=None):
    doc = item_doc(key, graph, seam_class, boundaryness, np, names, predicted)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return doc


SUMMARY_COLUMNS = ("item", "id", "a", "b", "length", "class", "class_name", "boundaryness")


def summary_rows(key, doc):
    return [(str(key), r["id"], r["a"], r["b"], r["length"], r["class"], r["class_name"],
             "" if r["boundaryness"] is None else r["boundaryness"]) for r in doc["seams"]]


def write_summary_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_COLUMNS)
        w.writerows(rows)


__all__ = ["item_stem", "class_name", "seam_rows", "item_doc", "write_seams_json",
           "SUMMARY_COLUMNS", "summary_rows", "write_summary_csv"]
