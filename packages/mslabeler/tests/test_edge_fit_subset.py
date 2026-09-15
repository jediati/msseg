"""fit_edge_model embeds only the rows labeled pairs touch; the model it
fits is the one the full embedding gave."""
import numpy as np
import pytest

from msseg.labeler import edge_model


def _net(X, y):
    sklearn = pytest.importorskip("sklearn")
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    p = Pipeline([("scale", StandardScaler()),
                  ("net", MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=200, random_state=0))])
    return p.fit(X, y)


def test_subset_embedding_gives_the_same_pair_model():
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 6))
    cls = np.zeros(n, int)
    lab = rng.choice(n, 60, replace=False)
    cls[lab] = np.where(X[lab, 0] > 0, 1, 2)
    net = _net(X[lab], cls[lab])
    a = rng.integers(0, n, 3000); b = rng.integers(0, n, 3000)
    keep = a != b
    a, b = a[keep], b[keep]
    both = (cls[a] > 0) & (cls[b] > 0)
    edges = {"a": a, "b": b, "saddle": rng.random(len(a)), "length": np.full(len(a), np.nan),
             "slice": np.zeros(len(a), int), "both": both, "diff": both & (cls[a] != cls[b]),
             "n_rows": n}
    ext = rng.random(n)
    spec = edge_model.EdgeSpec()
    edge = edge_model.fit_edge_model(net, X, cls, edges, ext, spec, [f"f{i}" for i in range(6)])
    # the reference: pair features off the FULL embedding
    sel = np.nonzero(both)[0]
    emb = edge_model._layer(edge_model.embed(net, X), spec.layer)
    bar = edge_model.barrier(edges["saddle"][sel], ext[a[sel]], ext[b[sel]], n=len(sel))
    F = edge_model.pair_features(emb, a[sel], b[sel], bar, spec.features, contact=None)
    p_ref = edge.model.predict_proba(F)[:, 1]
    p_sub = edge_model.predict_pdiff(edge, net, X, a[sel], b[sel], edges["saddle"][sel], ext)
    assert edge.n_edges == int(both.sum()) and edge.width == emb.shape[1]
    assert np.allclose(np.clip(p_ref, 1e-4, 1 - 1e-4), p_sub, atol=1e-6)
