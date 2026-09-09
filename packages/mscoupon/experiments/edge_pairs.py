"""Edge experiment (see docs/mscoupon_edge_pairs.md):

Usage (from a source checkout, both namespace packages on the path):
  PYTHONPATH="packages/mscoupon/src;packages/msseg-viz/src" python packages/mscoupon/experiments/edge_pairs.py [out_dir]

Loads the labeler's last autosaved session, primes its slices headlessly, then: can a pair model on the region net's 16-d embedding tell
same-class from different-class edges of the region graph better than the
region net's own argmax, and does neighbour voting with it fix region errors?"""
import os, sys, time, json, warnings, tkinter as tk
import numpy as np
warnings.simplefilter("ignore")
from msseg.mscoupon import config_io, model_search as ms
from msseg.mscoupon.labeler import LabelerApp, _NON_FEATURE_FIELDS
from msseg.mscoupon.labeling import resolve_slice
from msseg.mscoupon import magic_fill

def say(*a):
    print("##", *a, flush=True)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, log_loss

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
root = tk.Tk(); root.withdraw()
app = LabelerApp(root, autosave=False)
doc = config_io.read_json_file(config_io.session_path(app="mscoupon-labeler"))
notes = app._apply_session_doc(doc, "edge experiment")
say("session notes:", notes)
app._run()
t0 = time.perf_counter()
while app.engine.run_active or app.engine.pending_work():
    root.update(); time.sleep(0.05)
    if time.perf_counter() - t0 > 1500:
        raise SystemExit("prime timeout")
say(f"primed {sum(len(p['pipes']) for p in app.primed)} slices in {time.perf_counter() - t0:.0f} s")
t0 = time.perf_counter()
slices = app._all_stat_slices("edge experiment")
say(f"materialized {len(slices)} slice records in {time.perf_counter() - t0:.0f} s")

# ---- regions: X (all), class (0 = unlabeled), slice id, and per-slice edges --
names = None; X, cls, grp = [], [], []
edges = []          # (slice k, row a, row b, saddle)
for k, (si, li, key, rec, table) in enumerate(slices):
    if names is None:
        names = [n for n in table.names if n not in _NON_FEATURE_FIELDS]
    mat = app._feature_matrix(table, names, np)
    fids = table.column("feature_id").astype(int)
    rc = resolve_slice(app.store.for_slice(key), rec["labels"], np)
    c = np.zeros(len(fids), int); ok = (fids >= 0) & (fids < len(rc)); c[ok] = rc[fids[ok]]
    base = len(np.concatenate(cls)) if cls else 0
    X.append(mat); cls.append(c); grp.append(np.full(len(c), k))
    row_of = {int(f): base + i for i, f in enumerate(fids)}
    arcs = rec.get("arcs")
    src = "msc"
    if arcs is None:                       # extension without region_arcs(): pixel adjacency
        arcs = magic_fill.arcs_from_labels(rec["labels"], np); src = "pixels"
    sad = arcs.get("saddle")
    sad = np.zeros(len(arcs["a"])) if sad is None else np.asarray(sad, float)
    n_e = 0
    if arcs is not None:
        for a, b, s in zip(arcs["a"].tolist(), arcs["b"].tolist(), sad.tolist()):
            ia, ib = row_of.get(int(a)), row_of.get(int(b))
            if ia is not None and ib is not None and ia != ib:
                edges.append((k, ia, ib, float(s))); n_e += 1
    say(f"  slice {k} {os.path.basename(key)}: {len(fids)} regions, {int((c > 0).sum())} labeled "
          f"({dict(zip(*np.unique(c[c > 0], return_counts=True)))}), {n_e} edges ({src})")
X = np.concatenate(X); cls = np.concatenate(cls); grp = np.concatenate(grp)
E = np.array([(k, a, b) for k, a, b, _s in edges], int); ES = np.array([s for *_r, s in edges], float)
both = (cls[E[:, 1]] > 0) & (cls[E[:, 2]] > 0)
diff = (cls[E[:, 1]] != cls[E[:, 2]]) & both
say(f"regions {len(cls)} ({int((cls > 0).sum())} labeled, {len(names)} features); edges {len(E)}, "
      f"both-labeled {int(both.sum())}, of which different-class {int(diff.sum())} "
      f"({diff.sum() / max(1, both.sum()):.1%})")
ext_col = names.index("ext_filtered") if "ext_filtered" in names else None

# ---- leave-slices-out folds on the labeled regions -------------------------
lab = np.nonzero(cls > 0)[0]
cv, kind = ms.make_cv(cls[lab], grp[lab])
say(f"cv: {kind}, {cv.get_n_splits()} folds")

def embed(est, X):
    z = est.named_steps["scale"].transform(est.named_steps["select"].transform(X))
    m = est.named_steps["dense"]
    hs = []
    h = z
    for W, b in zip(m.coefs_[:-1], m.intercepts_[:-1]):
        h = np.maximum(h @ W + b, 0.0); hs.append(h)
    return hs                                   # [h1 (16-d), h2 (8-d)]

def pair_feats(emb, a, b, extra=None):
    ea, eb = emb[a], emb[b]
    f = [np.abs(ea - eb), ea * eb]
    if extra is not None:
        f.append(extra)
    return np.concatenate(f, axis=1)

def barrier(a, b, s):
    if ext_col is None:
        return np.zeros((len(a), 1))
    ext = X[:, ext_col]
    return np.stack([s - np.maximum(ext[a], ext[b]), np.abs(ext[a] - ext[b])], 1)

results = {}   # name -> list of (y_true, p_diff) per fold
region_eval = {"before": [], "after_l1": [], "after_l05": [], "truth": []}
fold_reports = []
for f, (tr, te) in enumerate(cv.split(X[lab], cls[lab], grp[lab])):
    tr_rows, te_rows = lab[tr], lab[te]
    test_slices = sorted(set(grp[te_rows].tolist()))
    spec = ms.ModelSpec(hidden=(16, 8), backend="sklearn", max_iter=1000, seed=0)
    est = ms.fit_estimator(ms.build_estimator(spec, names), X[tr_rows], cls[tr_rows])
    P = est.predict_proba(X); classes = np.asarray(est.classes_)
    pred = classes[P.argmax(1)]
    h1, h2 = embed(est, X)
    in_test = np.isin(E[:, 0], test_slices)
    e_tr = np.nonzero(both & ~in_test)[0]
    e_te = np.nonzero(both & in_test)[0]
    y_tr, y_te = diff[e_tr].astype(int), diff[e_te].astype(int)
    a_tr, b_tr, a_te, b_te = E[e_tr, 1], E[e_tr, 2], E[e_te, 1], E[e_te, 2]
    bar_tr, bar_te = barrier(a_tr, b_tr, ES[e_tr]), barrier(a_te, b_te, ES[e_te])
    # baselines from the region net itself
    pd_argmax = (pred[a_te] != pred[b_te]).astype(float)
    pd_proba = 1.0 - (P[a_te] * P[b_te]).sum(1)
    results.setdefault("baseline: argmax differs", []).append((y_te, pd_argmax))
    results.setdefault("baseline: 1 - sum P_a P_b", []).append((y_te, pd_proba))

    def fit_lr(Ftr):
        m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0))
        return m.fit(Ftr, y_tr)

    def fit_mlp(Ftr):
        from sklearn.utils.class_weight import compute_sample_weight
        m = make_pipeline(StandardScaler(), MLPClassifier((32, 16), max_iter=1000, random_state=0))
        try:
            return m.fit(Ftr, y_tr, mlpclassifier__sample_weight=compute_sample_weight("balanced", y_tr))
        except TypeError:
            return m.fit(Ftr, y_tr)
    feats = {
        "emb16 |d|,prod (LR)": (fit_lr, lambda a, b, bar: pair_feats(h1, a, b)),
        "emb16 + barrier (LR)": (fit_lr, lambda a, b, bar: pair_feats(h1, a, b, bar)),
        "emb16 + barrier (MLP 32-16)": (fit_mlp, lambda a, b, bar: pair_feats(h1, a, b, bar)),
        "emb8 + barrier (LR)": (fit_lr, lambda a, b, bar: pair_feats(h2, a, b, bar)),
        "raw |d| + barrier (LR)": (fit_lr, lambda a, b, bar: np.concatenate([np.abs(X[a] - X[b]), bar], 1)),
        "barrier only (LR)": (fit_lr, lambda a, b, bar: bar),
    }
    edge_model = None
    for name, (fit, fx) in feats.items():
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 1:
            continue
        m = fit(fx(a_tr, b_tr, bar_tr))
        p = m.predict_proba(fx(a_te, b_te, bar_te))[:, 1]
        results.setdefault(name, []).append((y_te, p))
        if name == "emb8 + barrier (LR)":
            edge_model = m
    # ---- neighbour voting on the test slices with the emb16+barrier edge model
    e_all = np.nonzero(in_test)[0]
    a_all, b_all = E[e_all, 1], E[e_all, 2]
    pdiff = edge_model.predict_proba(pair_feats(h2, a_all, b_all, barrier(a_all, b_all, ES[e_all])))[:, 1]
    pdiff = np.clip(pdiff, 1e-4, 1 - 1e-4)
    logP = np.log(np.clip(P, 1e-6, 1))
    n_changed_l1 = 0
    for lam, key in ((1.0, "after_l1"), (0.5, "after_l05")):
        cur = pred.copy()
        for _it in range(3):
            score = logP.copy()
            for ci, c in enumerate(classes):
                contrib_a = np.where(cur[b_all] == c, np.log(1 - pdiff), np.log(pdiff))
                contrib_b = np.where(cur[a_all] == c, np.log(1 - pdiff), np.log(pdiff))
                np.add.at(score[:, ci], a_all, lam * contrib_a)
                np.add.at(score[:, ci], b_all, lam * contrib_b)
            cur = classes[score.argmax(1)]
        region_eval[key].append(cur[te_rows])
        if lam == 1.0:
            n_changed_l1 = int((cur != pred)[te_rows].sum())
    region_eval["before"].append(pred[te_rows]); region_eval["truth"].append(cls[te_rows])
    fold_reports.append(f"fold {f}: test slices {test_slices}, {len(te_rows)} labeled regions, "
                        f"{len(e_te)} labeled edges ({int(y_te.sum())} different); region bacc "
                        f"{balanced_accuracy_score(cls[te_rows], pred[te_rows]):.3f} -> "
                        f"{balanced_accuracy_score(cls[te_rows], region_eval['after_l1'][-1]):.3f} "
                        f"(lambda 1, {n_changed_l1} labeled regions flipped)")
say("\n".join(fold_reports))

def score(pairs):
    y = np.concatenate([p[0] for p in pairs]); p = np.concatenate([p[1] for p in pairs])
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    hard = (p >= 0.5).astype(int)
    tp = int(((hard == 1) & (y == 1)).sum()); fn = int(((hard == 0) & (y == 1)).sum()); fp = int(((hard == 1) & (y == 0)).sum())
    return dict(n=int(len(y)), bacc=float(balanced_accuracy_score(y, hard)),
                auc=float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
                logloss=float(log_loss(y, pc, labels=[0, 1])),
                diff_recall=tp / max(1, tp + fn), diff_precision=tp / max(1, tp + fp))

hdr = "edge model (pooled over folds, held-out slices)"
say(f"\n{hdr:44s} {'n':>6} {'bal.acc':>8} {'AUC':>6} {'logloss':>8} {'diff recall':>11} {'diff prec':>9}")
summary = {}
for name, pairs in results.items():
    s = score(pairs); summary[name] = s
    say(f"{name:44s} {s['n']:>6} {s['bacc']:>8.3f} {s['auc']:>6.3f} {s['logloss']:>8.3f} {s['diff_recall']:>11.1%} {s['diff_precision']:>9.1%}")
truth = np.concatenate(region_eval["truth"])
say("\nregion classification on held-out labeled regions (pooled):")
for key, label in (("before", "region net 16-8 alone"), ("after_l1", "+ neighbour voting, lambda 1"), ("after_l05", "+ neighbour voting, lambda 0.5")):
    pr = np.concatenate(region_eval[key])
    acc = float((pr == truth).mean()); bacc = float(balanced_accuracy_score(truth, pr))
    cm = {f"true{t}->pred{p}": int(((truth == t) & (pr == p)).sum()) for t in np.unique(truth) for p in np.unique(truth) if t != p}
    say(f"  {label:30s} acc {acc:.3f}  bal.acc {bacc:.3f}  errors {cm}")
    summary[f"region:{key}"] = {"acc": acc, "bacc": bacc, "errors": cm}
json.dump({"summary": summary, "folds": fold_reports, "n_regions": int(len(cls)), "n_edges": int(len(E)),
           "n_both": int(both.sum()), "n_diff": int(diff.sum())},
          open(os.path.join(OUT, "edge_experiment.json"), "w"), indent=2)
root.destroy()
