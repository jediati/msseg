"""Headless tests for the GPU dense classifier (msseg.mscoupon.torch_mlp) and
its wiring into the search (model_search backend="torch").

Skipped without torch; runs on CUDA when available, else CPU.

    pytest packages/mscoupon/tests/test_torch_mlp.py
"""
import pickle

import numpy as np
import pytest

pytest.importorskip("sklearn")
torch = pytest.importorskip("torch")

from msseg.mscoupon import model_search as ms
from msseg.mscoupon import torch_mlp as tm
from msseg.mscoupon.torch_mlp import TorchMLPClassifier


def blobs(n=300, d=8, seed=0, classes=3):
    rng = np.random.RandomState(seed)
    y = rng.randint(1, classes + 1, size=n)
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[:, :3] += 2.0 * np.stack([np.cos(y * 2.1), np.sin(y * 2.1), y / 2.0], 1)
    return X, y


def test_fit_predict_shapes_classes_and_pickle():
    X, y = blobs()
    clf = TorchMLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200).fit(X, y)
    p = clf.predict_proba(X)
    assert p.shape == (300, 3) and np.allclose(p.sum(1), 1.0) and p.dtype == np.float64
    assert list(clf.classes_) == [1, 2, 3]
    assert set(clf.predict(X)) <= {1, 2, 3}
    assert clf.score(X, y) > 0.9
    assert clf.n_iter_ <= 200 and len(clf.loss_curve_) == clf.n_iter_
    assert clf.device_ in ("cpu", "cuda")
    clf2 = pickle.loads(pickle.dumps(clf))
    assert np.allclose(clf2.predict_proba(X), p, atol=1e-6)
    # sklearn's clone/get_params contract (the pipeline relies on it).
    from sklearn.base import clone
    assert clone(clf).get_params()["hidden_layer_sizes"] == (32, 16)


def test_same_seed_same_weights():
    X, y = blobs()
    a = TorchMLPClassifier(hidden_layer_sizes=(16,), max_iter=50, random_state=3).fit(X, y)
    b = TorchMLPClassifier(hidden_layer_sizes=(16,), max_iter=50, random_state=3).fit(X, y)
    for (Wa, ba), (Wb, bb) in zip(a.params_, b.params_):
        assert np.array_equal(Wa, Wb) and np.array_equal(ba, bb)
    c = TorchMLPClassifier(hidden_layer_sizes=(16,), max_iter=50, random_state=4).fit(X, y)
    assert not np.array_equal(a.params_[0][0], c.params_[0][0])


def test_early_stopping_keeps_best_epoch_and_stops_early():
    X, y = blobs(n=400)
    clf = TorchMLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000,
                             early_stopping=True, n_iter_no_change=10).fit(X, y)
    assert clf.n_iter_ < 2000, "plateaued well before max_iter"
    assert 0 <= clf.best_epoch_ < clf.n_iter_
    assert clf.score(X, y) > 0.85


def test_minibatch_and_dropout_paths_train():
    X, y = blobs()
    clf = TorchMLPClassifier(hidden_layer_sizes=(32,), max_iter=60, batch_size=16,
                             dropout=0.3).fit(X, y)
    assert clf.score(X, y) > 0.8
    assert TorchMLPClassifier(hidden_layer_sizes=(8,), max_iter=5,
                              batch_size=10_000).fit(X, y).n_iter_ == 5


def test_sample_weight_shifts_probabilities():
    X, y = blobs(n=200, classes=2)
    base = TorchMLPClassifier(hidden_layer_sizes=(8,), max_iter=100).fit(X, y)
    w = np.where(y == 2, 20.0, 1.0)
    heavy = TorchMLPClassifier(hidden_layer_sizes=(8,), max_iter=100).fit(X, y, sample_weight=w)
    assert heavy.predict_proba(X)[:, 1].mean() > base.predict_proba(X)[:, 1].mean()


def test_stacked_folds_are_independent():
    """A fold's weights must not depend on which other fold shares the batch.
    Two stacked runs of the same shape draw the same init / shuffles /
    dropout masks for fold 0, so fold 0 of [A, B] and fold 0 of [A, C] must
    come out identical when B and C differ only in their rows."""
    X, y = blobs(n=360)
    pos = {c: i for i, c in enumerate(np.unique(y))}
    yi = np.asarray([pos[v] for v in y])
    A = (X[:120], yi[:120])
    B = (X[120:240], yi[120:240])
    C = (X[240:], yi[240:])
    for kw in (dict(batch_size="auto"), dict(batch_size=16, dropout=0.2)):
        ab = tm.train_stacked([A[0], B[0]], [A[1], B[1]], [None, None], n_classes=3,
                              hidden=(16,), max_iter=40, seed=0, device="cpu", **kw)
        ac = tm.train_stacked([A[0], C[0]], [A[1], C[1]], [None, None], n_classes=3,
                              hidden=(16,), max_iter=40, seed=0, device="cpu", **kw)
        for (W1, b1), (W2, b2) in zip(ab["params"], ac["params"]):
            assert np.allclose(W1[0], W2[0], atol=1e-6) and np.allclose(b1[0], b2[0], atol=1e-6)
            assert not np.allclose(W1[1], W2[1], atol=1e-6), "fold 1 did train on its own rows"
    # Padding rows carry no weight: a short fold next to a long one trains as
    # well as it would alone.
    long = (np.concatenate([X, X[:200]]), np.concatenate([yi, yi[:200]]))
    out = tm.train_stacked([long[0], B[0]], [long[1], B[1]], [None, None], n_classes=3,
                           hidden=(16,), max_iter=80, seed=0, device="cpu")
    p_b = tm.predict_stacked([(W[1:], b[1:]) for W, b in out["params"]], [B[0]], "cpu")[0]
    assert (p_b.argmax(1) == B[1]).mean() > 0.85


def test_cv_scores_per_fold_with_missing_class_and_ragged_folds():
    X, y = blobs(n=200)
    cv, _ = ms.make_cv(y, None, n_splits=4)
    folds = list(cv.split(X, y))
    # Drop class 3 from one fold's training rows: its test log-loss must still
    # be finite (the probability column exists, driven down).
    tr, te = folds[0]
    folds[0] = (tr[y[tr] != 3], te)
    scores = tm.cv_scores(X, y, folds, np.array([1, 2, 3]), hidden=(16,), max_iter=200,
                          early_stopping=True)
    assert len(scores) == 4
    assert all(np.isfinite(l) and 0.0 <= b <= 1.0 for l, b in scores)
    # The fold that never saw class 3 cannot score it (balanced accuracy is
    # capped at 2/3 there); the complete folds separate the blobs.
    assert scores[0][1] <= 2.0 / 3.0 + 1e-9
    assert all(b > 0.8 for _l, b in scores[1:]), scores


def test_model_search_torch_backend_end_to_end():
    from test_model_search import synth
    X, y, groups, names, schema = synth(240)
    assert ms.resolve_backend("auto") == "torch" and ms.resolve_backend("sklearn") == "sklearn"
    assert ms.backend_label("torch").startswith("torch on ")
    space = ms.SearchSpace(max_layers=2, units=(8, 64))
    res = ms.run_search(X, y, groups, names, schema, n_trials=6, seed=1, max_iter=150,
                        space=space, searcher="random", backend="torch")
    spec = res.spec
    assert spec.backend == "torch" and spec.n_trials == 6
    # The REQUEST is torch; a mini-batch winner below TORCH_MIN_BATCH trains
    # on sklearn (effective_backend), the describe() names what trained it.
    eff = ms.effective_backend(spec)
    want = "TorchMLPClassifier" if eff == "torch" else "MLPClassifier"
    assert type(res.estimator.named_steps["dense"]).__name__ == want
    assert spec.cv_score <= spec.baseline_score + 1e-12
    assert want in spec.describe()
    assert ("dropout" in spec.describe()) == (eff == "torch")
    # The baseline is full-batch, so it did train on torch.
    assert ms.effective_backend(res.baseline) == "torch"
    p = res.estimator.predict_proba(np.zeros((2, len(names))))
    assert p.shape == (2, 3)
    est2 = pickle.loads(pickle.dumps(res.estimator))
    assert np.allclose(est2.predict_proba(np.zeros((2, len(names)))), p, atol=1e-6)
    # Forcing sklearn on the same call builds MLPClassifier and never dropout.
    res_sk = ms.run_search(X, y, groups, names, schema, n_trials=2, seed=1, max_iter=100,
                           space=space, searcher="random", backend="sklearn",
                           importances=False)
    assert res_sk.spec.backend == "sklearn" and res_sk.spec.dropout == 0.0
    assert type(res_sk.estimator.named_steps["dense"]).__name__ == "MLPClassifier"
    # The dropout parameter only exists in a torch trial's Optuna params.
    if ms.have_optuna():
        import optuna
        st = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.RandomSampler(seed=0))
        grp = ms.feature_groups(names, schema)
        t = st.ask()
        ms.suggest_spec(t, grp, space, 0, 100, "torch")
        # ... when the draw trains on torch: a mini-batch below
        # TORCH_MIN_BATCH routes to sklearn, which has no dropout.
        assert (("dropout" in t.params)
                == ms._torch_trains("torch", ms._batch_value(t.params["batch"])))
        tt = optuna.trial.FixedTrial({"n_layers": 1, "units_0": 8, "alpha": 1e-4, "lr": 1e-3,
                                      "batch": "auto", "early_stopping": False,
                                      "dropout": 0.1,
                                      **{f"use_{g}": True for g in grp}})
        assert ms.suggest_spec(tt, grp, space, 0, 100, "torch").dropout == 0.1
        ts = optuna.trial.FixedTrial({"n_layers": 1, "units_0": 8, "alpha": 1e-4, "lr": 1e-3,
                                      "batch": "16", "early_stopping": False,
                                      **{f"use_{g}": True for g in grp}})
        small = ms.suggest_spec(ts, grp, space, 0, 100, "torch")
        assert small.dropout == 0.0 and ms.effective_backend(small) == "sklearn"
        t2 = st.ask()
        ms.suggest_spec(t2, grp, space, 0, 100, "sklearn")
        assert "dropout" not in t2.params
        assert "dropout" in ms.baseline_params(space, grp, "torch")
        assert "dropout" not in ms.baseline_params(space, grp, "sklearn")


def test_cv_scores_reports_heldout_loss_periodically_and_can_be_stopped():
    X, y = blobs(n=200)
    cv, _ = ms.make_cv(y, None, n_splits=3)
    folds = list(cv.split(X, y))
    seen = []
    tm.cv_scores(X, y, folds, np.array([1, 2, 3]), hidden=(16,), max_iter=100,
                 n_iter_no_change=1000, report_cb=lambda e, l: seen.append((e, l)),
                 report_every=20)
    assert [e for e, _l in seen] == [20, 40, 60, 80, 100]
    assert all(np.isfinite(l) and l > 0 for _e, l in seen)
    assert seen[-1][1] < seen[0][1], "held-out loss falls on separable blobs"

    class Stop(Exception):
        pass

    def cb(epoch, loss):
        if epoch >= 40:
            raise Stop()
    with pytest.raises(Stop):
        tm.cv_scores(X, y, folds, np.array([1, 2, 3]), hidden=(16,), max_iter=100,
                     n_iter_no_change=1000, report_cb=cb, report_every=20)
    # Without a callback nothing is reported and the result is unchanged.
    a = tm.cv_scores(X, y, folds, np.array([1, 2, 3]), hidden=(16,), max_iter=50)
    b = tm.cv_scores(X, y, folds, np.array([1, 2, 3]), hidden=(16,), max_iter=50,
                     report_cb=lambda e, l: None, report_every=10)
    assert np.allclose([s[0] for s in a], [s[0] for s in b])
