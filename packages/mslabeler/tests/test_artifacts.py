"""artifacts: provider references, the slots' document form, resolution, the
per-slot gate, a task's model read without activating it, and p(diff) from
another task's edge model."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from msseg.labeler import artifacts as af                       # noqa: E402
from msseg.labeler import derive                                 # noqa: E402
from msseg.labeler.session_doc import build_session_doc, session_doc_from_json  # noqa: E402
from msseg.labeler.task import Task                              # noqa: E402

NAMES = [f"mean_c{i}" for i in range(6)]


def test_reference_document_forms():
    t = af.ProviderRef.task("t_abc123")
    assert t.to_doc() == {"source": "task", "uid": "t_abc123"}
    assert af.ProviderRef.from_doc(t.to_doc()) == t
    lib = af.ProviderRef("library", "enc-7")
    assert lib.to_doc() == {"source": "library", "id": "enc-7"}
    assert af.ProviderRef.from_doc(lib.to_doc()) == lib
    for junk in (None, {}, {"source": "task"}, {"source": "cloud", "uid": "x"},
                 {"source": "task", "uid": ""}, "t_abc123"):
        assert af.ProviderRef.from_doc(junk) is None


def test_normalise_inputs_is_total():
    notes = []
    raw = {"labels": {"source": "task", "uid": "t_1", "boundary_class": "3"},
           "embedding": {"source": "library", "id": "enc"},
           "pdiff": {"source": "task"},                    # no uid
           "colour": {"source": "task", "uid": "t_1"}}     # no such slot
    out = af.normalise_inputs(raw, notes)
    assert out == {"labels": {"source": "task", "uid": "t_1", "boundary_class": 3},
                   "embedding": {"source": "library", "id": "enc"}}
    assert len(notes) == 2
    assert af.normalise_inputs(None) is None
    assert af.normalise_inputs({}) is None
    assert af.normalise_inputs("x", notes) is None and "not a mapping" in notes[-1]
    # Class 1 is "not a boundary": a derived boundary never lands there.
    assert af.boundary_class_of({"boundary_class": 1}) == af.DEFAULT_BOUNDARY_CLASS
    assert af.boundary_class_of({"boundary_class": "x"}) == af.DEFAULT_BOUNDARY_CLASS
    assert af.boundary_class_of(None) == af.DEFAULT_BOUNDARY_CLASS


def test_task_writes_inputs_only_when_set_and_round_trips():
    region = Task.new("glands", workflow="p1")
    walls = Task.new("walls", workflow="p1", kind="polyline")
    assert "inputs" not in walls.to_doc()
    walls.inputs = {"labels": {"source": "task", "uid": region.uid, "boundary_class": 2}}
    doc = walls.to_doc()
    assert list(doc)[-1] == "inputs" and doc["inputs"] == walls.inputs
    back = Task.from_doc(json.loads(json.dumps(doc)))
    assert back.inputs == walls.inputs
    assert walls.duplicate("walls 2").inputs == walls.inputs
    # A region task has no slots: a stray key is not read.
    rdoc = dict(region.to_doc(), inputs=walls.inputs)
    assert Task.from_doc(rdoc).inputs is None
    # Through the session document.
    sdoc = build_session_doc(app="labeler", folders=[], sequences=[], profiles=[{"name": "p1"}],
                             active_profile="p1", run={}, view={},
                             tasks=[region.to_doc(), walls.to_doc()], active_task=walls.uid)
    notes = []
    got = session_doc_from_json(json.loads(json.dumps(sdoc)), notes)
    assert notes == [] and got["tasks"][1]["inputs"] == walls.inputs
    assert "inputs" not in got["tasks"][0]


def test_resolve_and_capabilities_without_a_model():
    region = Task.new("glands", workflow="p1")
    walls = Task.new("walls", workflow="p1", kind="polyline")
    other = Task.new("more walls", workflow="p1", kind="polyline")
    tasks = [region, walls, other]
    assert af.resolve(None, tasks) is None
    p = af.resolve(af.ProviderRef.task(region.uid), tasks, consumer=walls.uid)
    assert isinstance(p, af.TaskProvider) and p.name == "glands"
    assert p.capabilities() == {"labels"}
    assert p.why_not("labels", NAMES) is None
    assert "no trained model" in p.why_not("embedding", NAMES)
    assert "no trained model" in p.why_not("pdiff", NAMES)
    # Gestures are geometry: the labels slot is read off the store by slide.
    region.store.add("squiggle", [(0, 0), (4, 4)], 1, "slide-a", 0, 0)
    region.store.add("squiggle", [(0, 0), (4, 4)], 2, "slide-b", 0, 0)
    assert [it.class_id for it in p.gestures_for("slide-a", None)] == [1]
    sig = p.signature("labels")
    region.store.add("squiggle", [(9, 9), (12, 12)], 2, "slide-a", 0, 0)
    assert p.signature("labels") != sig, "the provider's edits move the signature"
    # What never resolves says why.
    assert "own input" in af.resolve(af.ProviderRef.task(walls.uid), tasks,
                                     consumer=walls.uid).why_not("labels", NAMES)
    assert "deleted" in af.resolve(af.ProviderRef.task("t_gone00"), tasks).why_not("labels", None)
    assert "library" in af.resolve(af.ProviderRef("library", "x"), tasks).why_not("labels", None)
    assert "polyline task" in af.resolve(af.ProviderRef.task(other.uid), tasks,
                                         consumer=walls.uid).why_not("labels", None)


def test_requirement_gate():
    req = af.Requirement(tuple(NAMES), "L4")
    assert req.why_not(NAMES + ["extra"], "L4") is None
    assert req.why_not(None, "L4") is None, "an app that cannot say is not refused"
    why = req.why_not(NAMES[:1], "L4", "workflow 'p2'")
    assert "needs mean_c1, mean_c2, mean_c3, mean_c4 (+1 more)" in why and "'p2'" in why
    assert "measured at L4" in req.why_not(NAMES, "L0")


def _lattice(side=6, seed=0):
    rng = np.random.default_rng(seed)
    cells = np.arange(side * side)
    fids = cells * 3                                   # sparse label ids
    c = np.where(cells % side < side // 2, 1, 2)
    mu = np.where(c[:, None] == 1, -1.0, 1.0) * np.array([1, 0.8, 0.6, 0.4, 0.2, 0.1])
    X = mu + 0.35 * rng.standard_normal((len(cells), 6))
    a, b = [], []
    for i in cells:
        r, q = divmod(int(i), side)
        if q + 1 < side:
            a.append(fids[i]); b.append(fids[i + 1])
        if r + 1 < side:
            a.append(fids[i]); b.append(fids[i + side])
    arcs = {"a": np.asarray(a, np.int32), "b": np.asarray(b, np.int32),
            "saddle": np.full(len(a), 0.5, np.float32), "source": "msc"}
    return fids, X, c, arcs


def test_model_read_without_activation_and_pdiff(tmp_path):
    pytest.importorskip("sklearn")
    from msseg.labeler import bundle, edge_model as em, model_search as ms
    from msseg.labeler.table import FeatureTable
    fids, X, c, arcs = _lattice()
    est = ms.fit_estimator(ms.build_estimator(ms.ModelSpec(hidden=(16, 8), backend="sklearn",
                                                           max_iter=300), NAMES), X, c)
    edges = em.gather_edges([(fids, arcs, c, 0)])
    edge = em.fit_edge_model(est, X, c, edges, None, em.EdgeSpec(), NAMES)
    path = str(tmp_path / "glands.pkl")
    b = bundle.ModelBundle(model=est, names=NAMES, kind="dense FC", edge=edge.to_dict())
    b.save(path)
    region = Task.new("glands", workflow="p1")
    region.models.append(b.record_entry(path))
    region.model_pending = True
    walls = Task.new("walls", workflow="p1", kind="polyline")
    af.clear_stack_cache()
    p = af.resolve(af.ProviderRef.task(region.uid), [region, walls], consumer=walls.uid)
    assert p.capabilities() == {"labels", "embedding", "pdiff"}
    assert region.model.clf is None and region.model_pending, "the task is untouched"
    stack = p.stack()
    assert stack is af.load_model_stack(region), "read once, by path and mtime"
    assert stack.edge is not None and stack.names == NAMES
    net, names = p.base()
    assert net is stack.clf and names == NAMES
    assert p.requires("embedding") == af.Requirement(tuple(NAMES), None)
    assert p.why_not("pdiff", NAMES) is None
    assert "needs mean_c5" in p.why_not("pdiff", NAMES[:5])
    # p(diff) on the consumer's record: arcs in their own order, NaN where an
    # arc names a region without a row.
    table = FeatureTable(["feature_id"] + NAMES,
                         np.column_stack([fids.astype(float), X]))
    bad = dict(arcs)
    bad["a"] = np.append(arcs["a"], 999).astype(np.int32)
    bad["b"] = np.append(arcs["b"], 0).astype(np.int32)
    bad["saddle"] = np.append(arcs["saddle"], 0.5).astype(np.float32)
    pd = af.pdiff_for(stack, table, bad, None, np)
    assert pd.shape == (len(bad["a"]),) and np.isnan(pd[-1]) and not np.isnan(pd[:-1]).any()
    from msseg.labeler import magic_fill
    ia, ib, keep = magic_fill.index_arcs(arcs, fids, np)
    want = em.predict_pdiff(edge, est, X, ia, ib, arcs["saddle"][keep], None)
    assert np.allclose(pd[:-1], want)
    diff = c[ia] != c[ib]
    assert pd[:-1][diff].mean() > pd[:-1][~diff].mean()
    # A live stack wins over the pickle (trained this session, not saved).
    region.model.clf = est
    region.model.names = NAMES
    assert p.stack() is region.model and "pdiff" not in p.capabilities()
    assert "no edge model" in p.why_not("pdiff", NAMES)


def test_derived_boundary_class():
    """The labels slot's setting: a derived boundary lands in the chosen class,
    an explicit seam gesture still overwrites."""
    from msseg.labeler.seams import SeamGraph
    lab = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]], np.int32)
    g = SeamGraph.from_labels(lab, np)
    region_class = np.array([1, 2, 1, 1], np.uint8)       # region 1 differs
    two = derive.seam_labels(g, np, region_class)
    three = derive.seam_labels(g, np, region_class, boundary_class=3)
    assert (two.cls == 2).any()
    assert np.array_equal(three.cls == 3, two.cls == 2)
    assert np.array_equal(three.cls == 1, two.cls == 1)
