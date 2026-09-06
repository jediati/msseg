"""Headless tests for the dense-FC search (msseg.mscoupon.model_search).

No Tk, no compiled extension. scikit-learn is required; optuna is exercised
when installed and the random searcher is covered either way.

    pytest packages/mscoupon/tests/test_model_search.py
"""
import pickle

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from msseg.mscoupon import model_search as ms
from msseg.mscoupon.model_search import (ModelSpec, FeatureSubset, SearchSpace,
                                         build_estimator, feature_groups, make_cv,
                                         cv_evaluate, run_search)


# --------------------------------------------------------------------------- #
# Fixture: 12 channels x {mean, std} = 24 columns, three informative channels,
# six pseudo-slices, three classes; the class boundary is not axis-aligned.
# --------------------------------------------------------------------------- #
CHANNELS = ["base", "blur_s0.7", "blur_s1.5", "edges_s0.7", "edges_s1.5",
            "hessian_largest_s1.5", "hessian_smallest_s1.5", "gradmag_s3",
            "laplacian_s3", "blur_s3", "edges_s3", "hessian_largest_s3"]
INFORMATIVE = ("base", "edges_s1.5", "hessian_smallest_s1.5")


def _schema(names):
    out = []
    for n in names:
        red, chan = n.split("_", 1)
        out.append({"name": n, "channel": chan, "reduction": red})
    return out


def synth(n=300, seed=0):
    rng = np.random.RandomState(seed)
    names = [f"{r}_{c}" for c in CHANNELS for r in ("mean", "std")]
    y = rng.randint(1, 4, size=n)
    groups = rng.randint(0, 6, size=n)
    X = rng.normal(size=(n, len(names)))
    # Signal: class-dependent offsets on the informative channels, rotated so a
    # single feature never separates the classes alone.
    for c in INFORMATIVE:
        i = names.index(f"mean_{c}")
        j = names.index(f"std_{c}")
        X[:, i] += 1.6 * np.cos(y * 2.1) + 0.3 * groups / 6.0
        X[:, j] += 1.6 * np.sin(y * 2.1)
    return X, y, groups, names, _schema(names)


# --------------------------------------------------------------------------- #
# Spec + estimator
# --------------------------------------------------------------------------- #
def test_spec_round_trips_through_dict():
    s = ModelSpec(hidden=(96, 48, 16), alpha=3e-3, batch_size=32, early_stopping=True,
                  features=["mean_base", "std_base"], cv_score=0.4, cv_bacc=0.9)
    d = s.to_dict()
    assert d["hidden"] == [96, 48, 16] and d["batch_size"] == 32
    assert ModelSpec.from_dict(d) == s
    # Unknown keys (a newer writer) are ignored; "auto" batch survives.
    assert ModelSpec.from_dict({"hidden": [8], "batch_size": "auto", "future": 1}) == \
        ModelSpec(hidden=(8,))
    assert ModelSpec.from_dict({}).hidden == ms.BASELINE_HIDDEN


def test_feature_subset_keeps_named_columns_and_pickles():
    X, y, groups, names, schema = synth(60)
    keep = ["mean_base", "std_edges_s1.5"]
    fs = FeatureSubset(names, keep).fit(X)
    assert fs.transform(X).shape == (60, 2)
    assert np.array_equal(fs.transform(X)[:, 0], X[:, names.index("mean_base")])
    assert int(fs.get_support().sum()) == 2
    assert list(fs.get_feature_names_out()) == keep
    with pytest.raises(ValueError):
        FeatureSubset(names, ["nope"]).fit(X)
    with pytest.raises(ValueError):
        FeatureSubset(names, keep).fit(X[:, :5])
    fs2 = pickle.loads(pickle.dumps(fs))
    assert np.array_equal(fs2.transform(X), fs.transform(X))


def test_build_estimator_consumes_full_schema_and_pickles():
    X, y, groups, names, schema = synth(120)
    spec = ModelSpec(hidden=(16,), features=[f"mean_{c}" for c in INFORMATIVE], max_iter=200)
    est = ms.fit_estimator(build_estimator(spec, names), X, y)
    p = est.predict_proba(X)
    assert p.shape == (120, 3) and np.allclose(p.sum(1), 1.0)
    assert list(est.classes_) == [1, 2, 3]
    est2 = pickle.loads(pickle.dumps(est))
    assert np.allclose(est2.predict_proba(X), p)
    assert int(est.named_steps["select"].get_support().sum()) == 3


def test_describe_and_brief_render_from_the_spec():
    s = ModelSpec(hidden=(32, 8), features=["a"], cv_score=0.31, cv_bacc=0.875,
                  baseline_score=0.5, cv_kind="slices", n_folds=5, n_groups=6,
                  n_trials=40, searcher="optuna")
    text = s.describe(24)
    for piece in ("[32, 8]", "1/24 features", "leave-slices-out", "6 slices",
                  "0.310", "87.5%", "baseline", "0.500", "40 trials"):
        assert piece in text, (piece, text)
    assert s.brief(24) == "[32, 8] · 1/24 features · CV bacc 87.5%"
    assert "all 24 features" in ModelSpec().describe(24)


# --------------------------------------------------------------------------- #
# Groups + CV
# --------------------------------------------------------------------------- #
def test_feature_groups_by_channel_with_per_column_fallback():
    X, y, groups, names, schema = synth(10)
    g = feature_groups(names, schema)
    assert list(g) == CHANNELS
    assert g["base"] == ["mean_base", "std_base"]
    flat = feature_groups(names, [{"name": n, "channel": "", "reduction": ""} for n in names])
    assert list(flat) == names
    assert feature_groups(names, None) == flat


def test_make_cv_groups_by_slice_when_three_or_more():
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    X, y, groups, names, schema = synth(200)
    cv, kind = make_cv(y, groups)
    assert isinstance(cv, StratifiedGroupKFold) and kind == "slices"
    for tr, te in cv.split(X, y, groups):
        assert not set(groups[tr]) & set(groups[te]), "a slice never straddles folds"
    cv2, kind2 = make_cv(y, np.where(groups < 3, 0, 1))
    assert isinstance(cv2, StratifiedKFold) and kind2 == "stratified"
    cv3, _ = make_cv(y, None)
    assert isinstance(cv3, StratifiedKFold)
    # Folds are capped by the rarest class; a singleton class is refused.
    yy = np.array([1, 1, 1, 2, 2, 2, 3, 3])
    assert make_cv(yy, None)[0].get_n_splits() == 2
    with pytest.raises(ValueError):
        make_cv(np.array([1, 1, 2, 2, 3]), None)
    with pytest.raises(ValueError):
        make_cv(np.array([1, 1, 1, 1]), None)


def test_cv_evaluate_handles_a_fold_missing_a_class():
    X, y, groups, names, schema = synth(80)
    cv, _ = make_cv(y, None, n_splits=3)
    spec = ModelSpec(hidden=(8,), max_iter=100)
    loss, bacc = cv_evaluate(spec, X, y, cv, names)
    assert np.isfinite(loss) and 0.0 <= bacc <= 1.0
    # A widened proba never drops a class column: a 2-class estimator scored
    # against 3 classes still yields (n, 3) rows summing to one.
    est = ms.fit_estimator(build_estimator(spec, names), X[y != 3], y[y != 3])
    p = ms._full_proba(est, X[:5], np.array([1, 2, 3]))
    assert p.shape == (5, 3) and np.allclose(p.sum(1), 1.0) and p[:, 2].max() < 1e-6


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #
def _check_result(res, names):
    spec = res.spec
    assert spec.cv_score is not None and spec.baseline_score is not None
    assert spec.cv_score <= spec.baseline_score + 1e-12, "never worse than trial 0"
    assert spec.cv_kind == "slices" and spec.n_groups == 6
    assert res.estimator.predict_proba(np.zeros((2, len(names)))).shape == (2, 3)
    est2 = pickle.loads(pickle.dumps(res.estimator))
    assert np.allclose(est2.predict_proba(np.zeros((2, len(names)))),
                       res.estimator.predict_proba(np.zeros((2, len(names)))))
    if spec.features is not None:
        assert set(spec.features) <= set(names)
        # The informative channels' means survive the mask more often than not.
        kept = sum(f"mean_{c}" in spec.features for c in INFORMATIVE)
        assert kept >= 2, spec.features
    assert res.importances and res.importances[0][0] in names
    top = {n for n, _v in res.importances[:6]}
    assert any(f"mean_{c}" in top or f"std_{c}" in top for c in INFORMATIVE), top


def test_random_search_is_seeded_and_reports_progress():
    X, y, groups, names, schema = synth(240)
    seen = []
    space = SearchSpace(max_layers=2, units=(8, 64))
    res = run_search(X, y, groups, names, schema, n_trials=6, seed=1, max_iter=150,
                     space=space, searcher="random",
                     progress_cb=lambda d, n, b: seen.append((d, n, b is not None)))
    assert res.spec.searcher == "random" and res.n_trials == 6 and not res.stopped
    assert [d for d, _n, _b in seen] == [1, 2, 3, 4, 5, 6] and all(n == 6 for _d, n, _b in seen)
    _check_result(res, names)
    res2 = run_search(X, y, groups, names, schema, n_trials=6, seed=1, max_iter=150,
                      space=space, searcher="random", importances=False)
    assert res2.spec.to_dict() == res.spec.to_dict(), "same seed, same winner"


def test_stop_event_keeps_best_so_far():
    import threading
    X, y, groups, names, schema = synth(150)
    ev = threading.Event()

    def cb(done, n, best):
        if done >= 3:
            ev.set()
    res = run_search(X, y, groups, names, schema, n_trials=20, seed=0, max_iter=100,
                     space=SearchSpace(max_layers=1, units=(8, 16)), searcher="random",
                     progress_cb=cb, stop_event=ev, importances=False)
    assert res.stopped and res.n_trials == 3
    assert res.spec.n_trials == 3 and res.spec.cv_score is not None


def test_timeout_stops_the_search():
    X, y, groups, names, schema = synth(150)
    res = run_search(X, y, groups, names, schema, n_trials=50, seed=0, max_iter=100,
                     timeout_s=0.0, searcher="random", importances=False)
    assert res.stopped and res.n_trials == 1 and res.spec.hidden == ms.BASELINE_HIDDEN


@pytest.mark.skipif(not ms.have_optuna(), reason="optuna not installed")
def test_optuna_search_enqueues_the_baseline_and_improves():
    X, y, groups, names, schema = synth(240)
    space = SearchSpace(max_layers=2, units=(8, 64))
    res = run_search(X, y, groups, names, schema, n_trials=10, seed=3, max_iter=150,
                     space=space, searcher="optuna")
    assert res.spec.searcher == "optuna" and res.n_trials == 10
    _check_result(res, names)


def test_missing_optuna_falls_back_to_random(monkeypatch):
    monkeypatch.setattr(ms, "have_optuna", lambda: False)
    X, y, groups, names, schema = synth(120)
    res = run_search(X, y, groups, names, schema, n_trials=3, seed=0, max_iter=100,
                     space=SearchSpace(max_layers=1, units=(8, 16)), importances=False)
    assert res.spec.searcher == "random" and res.n_trials == 3


def test_search_refuses_unscorable_labels():
    X, y, groups, names, schema = synth(30)
    y = np.ones_like(y)
    with pytest.raises(ValueError):
        run_search(X, y, groups, names, schema, n_trials=2, searcher="random")


def test_effective_backend_routes_small_batches_to_sklearn():
    torch_spec = ModelSpec(backend="torch", batch_size=16)
    if ms.resolve_backend("torch") == "torch":
        assert ms.effective_backend(torch_spec) == "sklearn"
        assert ms.effective_backend(ModelSpec(backend="torch", batch_size="auto")) == "torch"
        assert ms.effective_backend(ModelSpec(backend="torch",
                                              batch_size=ms.TORCH_MIN_BATCH)) == "torch"
        assert ms.effective_backend(ModelSpec(backend="torch", batch_size=64)) == "sklearn"
    assert ms.effective_backend(ModelSpec(backend="sklearn", batch_size="auto")) == "sklearn"
    assert not ms._torch_trains("torch", 16) and ms._torch_trains("torch", "auto")
    assert ms._torch_trains("torch", 512) and not ms._torch_trains("sklearn", "auto")
    assert set(ms.BATCH_CHOICES) >= {"auto", "16", "256", "512"}


def test_search_budget_is_separate_from_the_refit_budget():
    X, y, groups, names, schema = synth(150)
    space = SearchSpace(max_layers=1, units=(8, 16))
    res = run_search(X, y, groups, names, schema, n_trials=3, seed=0, max_iter=60,
                     patience=3, refit_max_iter=777, space=space, searcher="random",
                     importances=False)
    assert res.spec.max_iter == 777, "the winner carries the refit cap"
    assert res.estimator.named_steps["dense"].max_iter == 777
    assert res.estimator.named_steps["dense"].n_iter_no_change == ms.N_ITER_NO_CHANGE
    # cv_evaluate's overrides never leak into the spec passed in.
    spec = ModelSpec(hidden=(8,), max_iter=500, backend="sklearn")
    cv, _ = make_cv(y, groups, n_splits=3)
    cv_evaluate(spec, X, y, cv, names, groups, max_iter=40, patience=2)
    assert spec.max_iter == 500
    est = build_estimator(spec, names, n_iter_no_change=4)
    assert est.named_steps["dense"].n_iter_no_change == 4
    assert build_estimator(spec, names).named_steps["dense"].n_iter_no_change == ms.N_ITER_NO_CHANGE


def test_parse_sizes_and_n_params():
    assert ms.parse_sizes("64-32, 32x16, 16 8; 4") == [(64, 32), (32, 16), (16, 8), (4,)]
    assert ms.parse_sizes("8-4\n4") == [(8, 4), (4,)]
    assert ms.size_text((16, 8)) == "16-8"
    assert ms.parse_sizes("8--") == [(8,)], "stray separators are ignored"
    for bad in ("", "a-b", "0", "4, ,x", "8-0"):
        with pytest.raises(ValueError):
            ms.parse_sizes(bad)
    assert ms.n_params((64, 32), 115, 3) == 116 * 64 + 65 * 32 + 33 * 3
    assert ms.n_params((), 5, 2) == 6 * 2


def test_fixed_hidden_pins_the_architecture():
    X, y, groups, names, schema = synth(150)
    grp = ms.feature_groups(names, schema)
    space = SearchSpace(fixed_hidden=(8, 4), max_layers=1, units=(64, 64))
    rng = np.random.RandomState(0)
    assert ms.random_spec(rng, grp, space, 0, 50).hidden == (8, 4)
    p = ms.baseline_params(space, grp)
    assert "n_layers" not in p and "units_0" not in p
    assert "n_layers" in ms.baseline_params(SearchSpace(), grp)
    if ms.have_optuna():
        import optuna
        st = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        t = st.ask()
        assert ms.suggest_spec(t, grp, space, 0, 50).hidden == (8, 4)
        assert "n_layers" not in t.params
    res = run_search(X, y, groups, names, schema, n_trials=3, seed=0, max_iter=60,
                     space=space, searcher="random", importances=False, backend="sklearn")
    assert res.spec.hidden == (8, 4) and res.baseline.hidden == (8, 4)
    assert res.spec.baseline_hidden == (8, 4)
    assert "(8, 4)" in res.spec.describe(len(names))
    d = res.spec.to_dict()
    assert d["baseline_hidden"] == [8, 4]
    assert ModelSpec.from_dict(d).baseline_hidden == (8, 4)
    assert ModelSpec.from_dict({"hidden": [8]}).baseline_hidden == ms.BASELINE_HIDDEN


def test_size_sweep_scores_each_rung_and_reports():
    X, y, groups, names, schema = synth(150)
    seen = []
    res = ms.run_size_sweep(X, y, groups, names, schema, sizes=[(16, 8), (4,)],
                            trials_per_size=2, seed=0, max_iter=60, patience=3,
                            refit_max_iter=90, searcher="random", backend="sklearn",
                            progress_cb=lambda *a: seen.append(a))
    assert [r.size for r in res.rows] == ["16-8", "4"]
    assert all(r.n_trials == 2 and r.spec.hidden == r.hidden for r in res.rows)
    assert all(r.spec.max_iter == 90 for r in res.rows)
    assert res.rows[0].n_params > res.rows[1].n_params
    assert res.rows[0].estimator.predict_proba(X[:2]).shape == (2, 3)
    assert [a[:2] for a in seen if a[-1] is not None] == [(0, 2), (1, 2)]
    assert any(a[-1] is None for a in seen), "per-trial progress too"
    b = res.best_index()
    assert res.relative_loss(b) == 0.0
    assert 0 <= res.smallest_within(1e9) == int(np.argmin([r.n_params for r in res.rows]))
    assert res.smallest_within(0.0) == b
    lines = res.report_lines()
    assert lines[0].startswith("size") and len(lines) == 4
    assert "best" in lines[1 + b]
    assert res.summary().startswith("best: ")
    d = res.to_dict()
    assert [r["hidden"] for r in d["rows"]] == [[16, 8], [4]] and d["report"] == lines
    assert ModelSpec.from_dict(d["rows"][0]["spec"]).hidden == (16, 8)
    # Breakdown is only ever a rung SMALLER than the best: at zero tolerance
    # the (16, 8) best makes (4,) the breakdown, a (4,) best has none.
    if res.rows[0].cv_score != res.rows[1].cv_score:
        assert res.breakdown_index(0.0) == (1 if b == 0 else None)
    assert res.breakdown_index(1e9) is None
    assert res.smaller_than_best() == ([1] if b == 0 else [])
    assert res.holds_below_breakdown(0.0) == []
    assert "breaks down at 4" in res.summary() if b == 0 else "no rung is smaller" in res.summary()


def test_size_sweep_stop_event_keeps_scored_rungs():
    import threading
    X, y, groups, names, schema = synth(120)
    ev = threading.Event()

    def cb(k, n, hidden, done, total, row):
        if row is not None:
            ev.set()
    res = ms.run_size_sweep(X, y, groups, names, schema, sizes=[(8,), (4,), (2,)],
                            trials_per_size=2, seed=0, max_iter=40, searcher="random",
                            backend="sklearn", progress_cb=cb, stop_event=ev)
    assert res.stopped and [r.size for r in res.rows] == ["8"]
    with pytest.raises(ValueError):
        ms.run_size_sweep(X, y, groups, names, schema, sizes=[(8,)], trials_per_size=1,
                          timeout_s=0.0, searcher="random", backend="sklearn",
                          stop_event=threading.Event())
