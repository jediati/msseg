"""edge_model: embedding, pair features, edges, the pair model, voting and the
leave-slices-out report -- on a synthetic lattice of regions."""
import os
import pickle
import sys
import threading

import numpy as np
import pytest

pytest.importorskip("sklearn")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from msseg.mscoupon import edge_model as em          # noqa: E402
from msseg.mscoupon import model_search as ms         # noqa: E402

NAMES = [f"mean_c{i}" for i in range(6)]


def lattice(n_slices=6, side=6, seed=0, unlabeled_every=7):
    """Per slice a side x side grid of regions with SPARSE label ids (3 * cell),
    class 1 on the left half and 2 on the right, 6 blob-ish features, and
    4-neighbour arcs with a fake saddle. Returns the stacked arrays and the
    gather_edges() input."""
    rng = np.random.default_rng(seed)
    X, cls, grp, ext, per_slice = [], [], [], [], []
    for k in range(n_slices):
        cells = np.arange(side * side)
        fids = cells * 3                                # sparse ids
        col = cells % side
        c = np.where(col < side // 2, 1, 2)
        mu = np.where(c[:, None] == 1, -1.0, 1.0) * np.array([1, 0.8, 0.6, 0.4, 0.2, 0.1])
        feats = mu + 0.35 * rng.standard_normal((len(cells), 6)) + 0.05 * k
        e = np.where(c == 1, 0.2, 0.8) + 0.05 * rng.standard_normal(len(cells))
        labeled = c.copy()
        labeled[cells % unlabeled_every == 3] = 0        # some unlabeled rows
        a, b, s = [], [], []
        for i in cells:
            r, q = divmod(int(i), side)
            if q + 1 < side:
                a.append(fids[i]); b.append(fids[i + 1]); s.append(0.5 + 0.1 * rng.standard_normal())
            if r + 1 < side:
                a.append(fids[i]); b.append(fids[i + side]); s.append(0.5 + 0.1 * rng.standard_normal())
        arcs = {"a": np.asarray(a, np.int32), "b": np.asarray(b, np.int32),
                "saddle": np.asarray(s, np.float32), "source": "msc"}
        X.append(feats); cls.append(labeled); grp.append(np.full(len(cells), k)); ext.append(e)
        per_slice.append((fids, arcs, labeled, k))
    X = np.concatenate(X); cls = np.concatenate(cls); grp = np.concatenate(grp); ext = np.concatenate(ext)
    return X, cls, grp, ext, per_slice


def region_net(X, cls, hidden=(16, 8), backend="sklearn"):
    lab = cls > 0
    est = ms.build_estimator(ms.ModelSpec(hidden=hidden, backend=backend, max_iter=300), NAMES)
    return ms.fit_estimator(est, X[lab], cls[lab])


def test_embed_matches_the_net_and_refuses_forests():
    X, cls, grp, ext, _ = lattice()
    est = region_net(X, cls)
    h1, h2 = em.embed(est, X)
    assert h1.shape == (len(X), 16) and h2.shape == (len(X), 8)
    assert (h1 >= 0).all() and (h2 >= 0).all()
    net = est.named_steps["dense"]
    logits = h2 @ net.coefs_[-1] + net.intercepts_[-1]
    # A binary sklearn MLP has ONE logistic output unit.
    idx = (logits[:, 0] > 0).astype(int) if logits.shape[1] == 1 else logits.argmax(1)
    assert np.array_equal(np.asarray(est.classes_)[idx], est.predict(X))
    assert em.hidden_widths(est) == [16, 8]
    assert len(em.net_hash(est)) == 40 and em.net_hash(est) == em.net_hash(est)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    forest = make_pipeline(StandardScaler(), RandomForestClassifier(10)).fit(X[cls > 0], cls[cls > 0])
    with pytest.raises(TypeError):
        em.embed(forest, X)
    with pytest.raises(TypeError):
        em.embed(RandomForestClassifier(5).fit(X[cls > 0], cls[cls > 0]), X)


def test_embed_torch_backend():
    pytest.importorskip("torch")
    X, cls, grp, ext, _ = lattice()
    est = region_net(X, cls, hidden=(8, 4), backend="torch")
    h1, h2 = em.embed(est, X)
    assert h1.shape == (len(X), 8) and h2.shape == (len(X), 4)
    assert em.hidden_widths(est) == [8, 4]
    # Manual forward through params_ reproduces the estimator's argmax.
    net = est.named_steps["dense"]
    W, b = net.params_[-1]
    logits = h2 @ np.asarray(W)[0] + np.asarray(b)[0]
    assert np.array_equal(np.asarray(est.classes_)[logits.argmax(1)], est.predict(X))


def test_pair_features_symmetric_and_barrier_width():
    emb = np.random.default_rng(1).random((10, 4)).astype(np.float32)
    a = np.array([0, 1, 2]); b = np.array([5, 6, 7])
    bar = em.barrier(np.array([0.5, np.nan, 0.2]), np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.1, 0.9]))
    assert bar.shape == (3, 2)
    assert np.isclose(bar[0, 0], 0.5 - 0.4) and bar[1, 0] == 0.0 and np.isclose(bar[1, 1], 0.1)
    assert em.barrier(None, None, None).shape == (0, 2)
    assert em.barrier(None, np.zeros(3), np.zeros(3)).shape == (3, 2)
    F = em.pair_features(emb, a, b, bar)
    G = em.pair_features(emb, b, a, bar)
    assert F.shape == (3, 4 + 4 + 2) and np.allclose(F, G)
    assert em.pair_features(emb, a, b, None, ("absdiff",)).shape == (3, 4)
    assert em.n_inputs(8) == 18 and em.n_inputs(8, ("absdiff", "prod")) == 16


def test_gather_edges_offsets_sparse_ids_and_masks():
    X, cls, grp, ext, per_slice = lattice(n_slices=2, side=3, unlabeled_every=4)
    edges = em.gather_edges(per_slice)
    assert edges["n_rows"] == 18
    assert (edges["a"] < 18).all() and (edges["b"] < 18).all() and (edges["a"] != edges["b"]).all()
    assert (edges["slice"][edges["a"] >= 9] == 1).all() and (edges["slice"][edges["a"] < 9] == 0).all()
    assert len(edges["a"]) == 2 * 12                     # 3x3 lattice: 12 arcs per slice
    assert np.isfinite(edges["saddle"]).all()
    assert edges["both"].dtype == bool and edges["diff"].sum() > 0
    assert (edges["diff"] <= edges["both"]).all()
    assert (cls[edges["a"][edges["both"]]] > 0).all()
    # An arc naming an id absent from the table is dropped; None arcs give none.
    fids, arcs, c, k = per_slice[0]
    bad = dict(arcs); bad["a"] = np.append(arcs["a"], 999); bad["b"] = np.append(arcs["b"], 0)
    bad["saddle"] = np.append(arcs["saddle"], 1.0)
    e2 = em.gather_edges([(fids, bad, c, 0), (fids, None, c, 1)])
    assert len(e2["a"]) == 12 and e2["n_rows"] == 18
    e3 = em.gather_edges([(fids, {"a": arcs["a"], "b": arcs["b"], "saddle": None}, c, 0)])
    assert np.isnan(e3["saddle"]).all()


def test_fit_predict_pdiff_and_roundtrip():
    X, cls, grp, ext, per_slice = lattice()
    edges = em.gather_edges(per_slice)
    train = grp < 4
    est = region_net(X, np.where(train, cls, 0))
    tr_edges = {k: (v[np.isin(edges["slice"], [0, 1, 2, 3])] if isinstance(v, np.ndarray) and len(v) == len(edges["a"]) else v)
                for k, v in edges.items()}
    model = em.fit_edge_model(est, X, cls, tr_edges, ext, em.EdgeSpec(), NAMES)
    assert model.n_in == 18 and model.width == 8 and model.used_saddle
    assert model.n_edges > 0 and 0 < model.n_diff < model.n_edges
    te = np.isin(edges["slice"], [4, 5]) & edges["both"]
    p = em.predict_pdiff(model, est, X, edges["a"][te], edges["b"][te], edges["saddle"][te], ext)
    assert p.shape == (int(te.sum()),) and (p > 0).all() and (p < 1).all()
    y = edges["diff"][te]
    assert p[y].mean() > 0.5 > p[~y].mean(), "boundary edges score higher"
    assert ((p >= 0.5) == y).mean() > 0.8
    # Pickle / dict round trip reproduces the probabilities.
    back = em.EdgeModel.from_dict(pickle.loads(pickle.dumps(model.to_dict())))
    p2 = em.predict_pdiff(back, est, X, edges["a"][te], edges["b"][te], edges["saddle"][te], ext)
    assert np.allclose(p, p2)
    assert back.spec == model.spec and back.names_hash == em.names_hash(NAMES)
    assert "edges" in model.brief() and "logistic" in model.describe([16, 8])
    # A different base net is refused; wrong feature width is refused.
    other = region_net(X, cls, hidden=(16, 8))
    with pytest.raises(ValueError):
        em.predict_pdiff(model, other, X, edges["a"][:3], edges["b"][:3])
    narrow = em.fit_edge_model(est, X, cls, tr_edges, ext, em.EdgeSpec(features=("absdiff",)), NAMES)
    assert narrow.n_in == 8
    # Single-kind edges refuse.
    one = dict(tr_edges); one["diff"] = np.zeros_like(tr_edges["diff"])
    with pytest.raises(ValueError):
        em.fit_edge_model(est, X, cls, one, ext, em.EdgeSpec(), NAMES)
    # The MLP pair model trains too.
    mlp = em.fit_edge_model(est, X, cls, tr_edges, ext, em.EdgeSpec(model="mlp"), NAMES)
    assert em.predict_pdiff(mlp, est, X, edges["a"][:5], edges["b"][:5]).shape == (5,)


def test_spec_round_trip_and_describe():
    s = em.EdgeSpec(layer=-2, features=("absdiff", "barrier"), model="mlp", C=0.5, lam=0.7, rounds=2)
    d = s.to_dict()
    assert d["features"] == ["absdiff", "barrier"]
    assert em.EdgeSpec.from_dict(d) == s
    assert em.EdgeSpec.from_dict({"model": "nope", "features": ["prod", "bogus"]}) == \
        em.EdgeSpec(model="logistic", features=("prod",))
    assert em.EdgeSpec.from_dict({}) == em.EdgeSpec()
    assert "previous (16-d)" in s.describe([16, 8]) and "MLP" in s.describe()
    assert "last (8-d)" in em.EdgeSpec().layer_text([16, 8])


def _vote_reference(P, classes, a, b, pdiff, lam, rounds):
    P = np.asarray(P, float); cur = np.asarray(classes)[P.argmax(1)]
    for _ in range(rounds):
        score = np.log(np.clip(P, 1e-6, 1)).copy()
        for e in range(len(a)):
            i, j, p = a[e], b[e], min(max(pdiff[e], 1e-4), 1 - 1e-4)
            for ci, c in enumerate(classes):
                score[i, ci] += lam * (np.log(1 - p) if cur[j] == c else np.log(p))
                score[j, ci] += lam * (np.log(1 - p) if cur[i] == c else np.log(p))
        new = np.asarray(classes)[score.argmax(1)]
        if np.array_equal(new, cur):
            break
        cur = new
    return cur


def test_vote_fixes_a_planted_flip_and_matches_a_reference():
    classes = np.array([1, 2])
    # Chain A - B - A: the middle weakly says 2, both neighbours confidently 1
    # and the edges say "same" -> it flips to 1.
    P = np.array([[0.95, 0.05], [0.45, 0.55], [0.95, 0.05]])
    a, b = np.array([0, 1]), np.array([1, 2])
    pd = np.array([0.05, 0.05])
    assert em.vote(P, classes, a, b, pd, 1.0, 3).tolist() == [1, 1, 1]
    # Edges that say "different" keep it.
    assert em.vote(P, classes, a, b, np.array([0.95, 0.95]), 1.0, 3).tolist() == [1, 2, 1]
    assert em.vote(P, classes, a, b, pd, 0.0, 3).tolist() == [1, 2, 1], "lam 0 = argmax"
    assert em.vote(P, classes, a, b, pd, 1.0, 0).tolist() == [1, 2, 1], "0 rounds = argmax"
    assert em.vote(P, classes, np.zeros(0, int), np.zeros(0, int), np.zeros(0), 1.0, 3).tolist() == [1, 2, 1]
    # Unscored rows (all-zero probabilities) do not break anything.
    P0 = np.vstack([P, [[0.0, 0.0]]])
    assert len(em.vote(P0, classes, a, b, pd, 1.0, 3)) == 4
    # A random graph agrees with the pure-Python reference.
    rng = np.random.default_rng(3)
    n = 40
    Pr = rng.random((n, 3)); Pr /= Pr.sum(1, keepdims=True)
    ar = rng.integers(0, n, 80); br = rng.integers(0, n, 80)
    keep = ar != br
    pr = rng.random(int(keep.sum()))
    cl = np.array([1, 2, 3])
    got = em.vote(Pr, cl, ar[keep], br[keep], pr, 0.7, 4)
    assert np.array_equal(got, _vote_reference(Pr, cl, ar[keep], br[keep], pr, 0.7, 4))


def test_evaluate_edges_report_and_stop():
    X, cls, grp, ext, per_slice = lattice()
    edges = em.gather_edges(per_slice)
    seen = []
    make = lambda: ms.build_estimator(ms.ModelSpec(hidden=(16, 8), backend="sklearn", max_iter=300), NAMES)
    rep = em.evaluate_edges(make, X, cls, grp, edges, ext, em.EdgeSpec(), seed=0,
                            progress_cb=lambda f, n: seen.append((f, n)))
    assert rep["cv_kind"] == "slices" and rep["n_folds"] == 5 and not rep["stopped"]
    assert [f for f, _n in seen] == [1, 2, 3, 4, 5]
    assert set(rep["edge"]) == set(em.EVAL_ROWS)
    for s in rep["edge"].values():
        assert s["n"] > 0 and 0 <= s["diff_recall"] <= 1 and np.isfinite(s["logloss"])
    assert rep["region"]["before"]["n"] == rep["region"]["after"]["n"] > 0
    assert len(rep["folds"]) + len(rep["skipped"]) == 5
    lines = em.report_lines(rep)
    assert lines[0].startswith("edge model") and any(l.startswith("learned") for l in lines)
    assert any(l.startswith("+ neighbour voting") for l in lines)
    # Stop after the first fold keeps what was scored.
    ev = threading.Event()
    rep2 = em.evaluate_edges(make, X, cls, grp, edges, ext, em.EdgeSpec(), seed=0,
                             progress_cb=lambda f, n: ev.set(), stop_event=ev)
    assert rep2["stopped"] and len(rep2["folds"]) + len(rep2["skipped"]) == 1
    # Slices whose labels are all one class give folds with one-kind training
    # edges or no scorable test edges: skipped, never fatal.
    cls1 = cls.copy(); cls1[grp >= 3] = np.where(cls[grp >= 3] > 0, 1, 0)
    e1 = em.gather_edges([(f, a, np.where(k >= 3, np.where(c > 0, 1, 0), c), k) for f, a, c, k in per_slice])
    rep3 = em.evaluate_edges(make, X, cls1, grp, e1, ext, em.EdgeSpec(), seed=0)
    assert isinstance(rep3["skipped"], list) and "edge" in rep3
