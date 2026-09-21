"""The seam model (msseg.labeler.seam_model): descriptors, fit, predict,
evaluate, pickle. Needs scikit-learn (the [classify] extra):

    pytest packages/mslabeler/tests/test_seam_model.py
"""
import numpy as np
import pytest

pytest.importorskip("sklearn")

from msseg.labeler import seam_model as sm            # noqa: E402
from msseg.labeler.bundle import ModelBundle           # noqa: E402
from msseg.labeler.seams import SeamGraph, SEAM_BOUNDARY, SEAM_INTERIOR   # noqa: E402
from msseg.labeler.table import FeatureTable           # noqa: E402


def lattice(rng, n=4, size=5, contrast=6.0):
    """An n x n block lattice whose left half is 'dark' and right half
    'bright': the seams between the halves are boundaries, the rest interior.
    Returns (labels, table, graph, y) with y the seam class per seam."""
    lab = np.zeros((n * size, n * size), np.int32)
    means = np.zeros(n * n)
    for r in range(n):
        for c in range(n):
            k = r * n + c
            lab[r * size:(r + 1) * size, c * size:(c + 1) * size] = k
            means[k] = (contrast if c >= n // 2 else 0.0) + rng.normal(0, 0.7)
    ids = np.arange(n * n)
    table = FeatureTable(["feature_id", "mean_base", "std_base", "ext_filtered", "ext_x", "ext_y"],
                         np.stack([ids, means, np.full(n * n, 1.0), means - 1.0,
                                   ids % n * size + 2.0, ids // n * size + 2.0], axis=1).astype(np.float64))
    g = SeamGraph.from_labels(lab, np)
    y = np.zeros(g.n_seams, np.int64)
    for i in range(g.n_seams):
        ca, cb = int(g.a[i]) % n, int(g.b[i]) % n
        y[i] = SEAM_BOUNDARY if (ca < n // 2) != (cb < n // 2) else SEAM_INTERIOR
    return lab, table, g, y


def test_descriptor_blocks_and_names():
    rng = np.random.default_rng(0)
    _lab, table, g, _y = lattice(rng)
    names = sm.region_feature_names(table)
    assert names == ["mean_base", "std_base", "ext_filtered"]
    F, fnames = sm.seam_features(g, table, names, np)
    assert F.shape == (g.n_seams, len(fnames)) and F.dtype == np.float32
    assert fnames[:3] == ["absdiff_mean_base", "absdiff_std_base", "absdiff_ext_filtered"]
    assert "prod_mean_base" in fnames and "barrier_depth" in fnames
    assert fnames[-5:] == list(sm.GEOMETRY_NAMES)
    assert np.isfinite(F).all()
    # length column = crack count; no saddles -> barrier depth 0, pdiff absent.
    assert (F[:, fnames.index("length")] == g.lengths(np)).all()
    assert (F[:, fnames.index("barrier_depth")] == 0).all()
    assert (F[:, fnames.index("pdiff_present")] == 0).all()
    # Subsets of the spec change the width and the names.
    F2, n2 = sm.seam_features(g, table, names, np, spec=sm.SeamSpec(features=("geometry",)))
    assert n2 == list(sm.GEOMETRY_NAMES) and F2.shape[1] == 5
    with pytest.raises(ValueError):
        sm.seam_features(g, table, names, np, spec=sm.SeamSpec(features=()))
    with pytest.raises(ValueError):
        sm.seam_features(g, table, names, np, spec=sm.SeamSpec(embed="net"))
    assert sm.embed_used(sm.SeamSpec(), None) == "features"
    spec = sm.SeamSpec.from_dict({"features": ["pair", "bogus"], "model": "mlp", "C": 2})
    assert spec.features == ("pair",) and spec.model == "mlp" and spec.C == 2.0
    assert sm.SeamSpec.from_dict(spec.to_dict()) == spec


def test_fit_predict_and_pickle_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    _lab, table, g, y = lattice(rng)
    names = sm.region_feature_names(table)
    F, fnames = sm.seam_features(g, table, names, np)
    with pytest.raises(ValueError):
        sm.fit_seam_model(F, np.where(y == SEAM_BOUNDARY, y, 0), None, fnames)   # one class
    model = sm.fit_seam_model(F, y, sm.SeamSpec(), fnames)
    assert model.n_seams == g.n_seams and model.n_boundary == int((y == SEAM_BOUNDARY).sum())
    p = sm.predict_boundaryness(model, F)
    assert p.shape == (g.n_seams,) and (p > 0).all() and (p < 1).all()
    assert p[y == SEAM_BOUNDARY].mean() > 0.8 and p[y == SEAM_INTERIOR].mean() < 0.2
    assert "logistic" in model.brief() and "features" in model.brief()
    with pytest.raises(ValueError):
        sm.predict_boundaryness(model, F[:, :3])
    # The pickle: the seam key only when set; a round trip predicts the same.
    b = ModelBundle(model=None, names=["x"], kind="dense FC", seam=model.to_dict())
    doc = b.to_doc()
    assert "seam" in doc and "seam" not in ModelBundle(model=None, names=["x"], kind="k").to_doc()
    path = tmp_path / "m.pkl"
    b.save(str(path))
    back = ModelBundle.load(str(path))
    m2 = sm.SeamModel.from_dict(back.seam)
    assert m2.feature_names == fnames and m2.names_hash == model.names_hash
    assert np.allclose(sm.predict_boundaryness(m2, F), p)
    assert back.record_entry(str(path))["seam"] is True
    assert "seam" not in ModelBundle(model=None, names=["x"], kind="k").record_entry(str(path))


def test_gather_and_evaluate_leave_items_out():
    rng = np.random.default_rng(2)
    items = []
    for k in range(4):
        _lab, table, g, y = lattice(rng)
        F, fnames = sm.seam_features(g, table, sm.region_feature_names(table), np)
        if k == 3:
            y = np.zeros_like(y)          # an unlabeled item contributes nothing
        items.append((f"item{k}", F, y, f"group{k}"))
    data = sm.gather_seams(items)
    assert data["F"].shape[0] == data["y"].shape[0] == len(data["group"]) == len(data["item"])
    assert set(data["group"][data["y"] > 0].tolist()) == {"group0", "group1", "group2"}
    progress = []
    rep = sm.evaluate_seams(data["F"], data["y"], data["group"], sm.SeamSpec(), seed=0,
                            progress_cb=lambda f, n: progress.append((f, n)))
    assert rep["cv_kind"] == "slices" and rep["n_folds"] == 3 and len(progress) == 3
    assert rep["mean"]["auc"] > 0.9 and rep["mean"]["bacc"] > 0.8
    assert rep["n"] == int((data["y"] > 0).sum())
    text = sm.summary(rep)
    assert "AUC" in text and "3-fold slices CV" in text
    assert sm.summary({"mean": {}}) == "no fold could be scored"
    # Stop event: no fold runs.
    import threading
    ev = threading.Event()
    ev.set()
    rep2 = sm.evaluate_seams(data["F"], data["y"], data["group"], stop_event=ev)
    assert rep2["stopped"] and rep2["folds"] == []
    assert sm.gather_seams([])["F"].shape == (0, 0)


def test_net_embedding_uses_the_base_hidden_layer():
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(3)
    _lab, table, g, _y = lattice(rng)
    names = sm.region_feature_names(table)
    X = np.stack([table.column(n) for n in names], axis=1)
    base = make_pipeline(StandardScaler(), MLPClassifier((8, 4), max_iter=300, random_state=0))
    base.fit(X, (X[:, 0] > 3).astype(int))
    assert sm.has_embedding(base) and sm.embed_used(sm.SeamSpec(), base) == "net"
    F, fnames = sm.seam_features(g, table, names, np, base_pipeline=base)
    assert fnames[:4] == ["absdiff_h0", "absdiff_h1", "absdiff_h2", "absdiff_h3"]
    assert F.shape[1] == 8 + 2 + 2 + 5
    F2, _ = sm.seam_features(g, table, names, np, base_pipeline=base,
                             spec=sm.SeamSpec(embed="features"))
    assert F2.shape[1] == 6 + 2 + 2 + 5
