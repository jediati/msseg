"""Context ablation: which neighbourhood columns help the region classifier?

Usage (from a source checkout, the namespace packages on the path):
  PYTHONPATH="packages/mslabeler/src;packages/mscoupon/src;packages/msseg-viz/src" \
      python packages/mscoupon/experiments/context_ablation.py [out_dir] [--hidden 16-8] [--search N]

Loads the labeler's last autosaved session, primes its slices headlessly, then
for every ContextSpec variant builds the design matrix (``context.augment`` per
slice, the same call the labeler makes) and scores ONE fixed dense net by
leave-slices-out cross-validation (``model_search.cv_evaluate``: mean held-out
log-loss and balanced accuracy). Variants: no context; each kind alone; the
ring kinds together; every kind; each weighting under ring mean + contrast;
each source. ``--search N`` additionally runs Optimize's own search with every
context kind on and the per-group feature mask free, and reports which groups
the winner kept -- the mask is the experiment.

Writes ``context_ablation_<date>.json`` beside the script (or in out_dir).
"""
import os, sys, time, json, warnings, tkinter as tk
import numpy as np
warnings.simplefilter("ignore")
from msseg.mscoupon import config_io
from msseg.mscoupon.labeler import LabelerApp
from msseg.labeler import context as cx
from msseg.labeler import model_search as ms
from msseg.labeler import fields
from msseg.labeler.training import TrainingSetBuilder


def say(*a):
    print("##", *a, flush=True)


args = [a for a in sys.argv[1:]]
hidden = (16, 8)
n_search = 0
out_dir = None
i = 0
while i < len(args):
    if args[i] == "--hidden":
        hidden = ms.parse_sizes(args[i + 1])[0]; i += 2
    elif args[i] == "--search":
        n_search = int(args[i + 1]); i += 2
    else:
        out_dir = args[i]; i += 1
OUT = out_dir or os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

root = tk.Tk(); root.withdraw()
app = LabelerApp(root, autosave=False)
doc = config_io.read_json_file(config_io.session_path(app="mscoupon-labeler"))
notes = app._apply_session_doc(doc, "context ablation")
say("session notes:", notes)
app._run()
t0 = time.perf_counter()
while app.engine.run_active or app.engine.pending_work():
    root.update(); time.sleep(0.05)
    if time.perf_counter() - t0 > 1500:
        raise SystemExit("prime timeout")
say(f"primed {sum(len(p['pipes']) for p in app.primed)} slices in {time.perf_counter() - t0:.0f} s")
t0 = time.perf_counter()
slices = app._all_stat_slices("context ablation")
if slices is None:
    raise SystemExit("no slices: " + app.status_var.get())
say(f"materialized {len(slices)} slice records in {time.perf_counter() - t0:.0f} s")
conv = app.FIELDS if hasattr(app, "FIELDS") else fields.DEFAULT
builder = TrainingSetBuilder(conv)
schema0 = app._feature_schema_now() or []

# ---- the variants -----------------------------------------------------------
V = {}
V["none"] = cx.ContextSpec()
for k in cx.KINDS:
    V[k] = cx.ContextSpec(kinds=(k,))
V["ring (all)"] = cx.ContextSpec(kinds=cx.RING_KINDS)
V["every kind"] = cx.ContextSpec(kinds=cx.KINDS)
for w in cx.WEIGHTS:
    V[f"mean+contrast [{w}]"] = cx.ContextSpec(kinds=("ring_mean", "ring_contrast"), weights=(w,))
for s in cx.SOURCES:
    V[f"mean+contrast on {s}"] = cx.ContextSpec(kinds=("ring_mean", "ring_contrast"), source=s)


def design(spec, training=True, zero_labels=False):
    """(X, y, groups, names, augment seconds per slice) for one spec. With a
    labels block: `training` applies the seeded dropout; `zero_labels` hides
    every annotation (what a fresh, unannotated slice looks like)."""
    items, secs = [], []
    for si, li, key, rec, table in slices:
        t = time.perf_counter()
        arcs = app.regions.arcs(key, np)
        extra = None
        if spec.labels is not None:
            rc = builder.row_classes(app.store.for_slice(key), rec["labels"],
                                     table.column(conv.id_field), np, app.regions.label_layer(key))
            if zero_labels:
                rc = np.zeros_like(rc)
            rng = np.random.default_rng([spec.labels.seed, si, li]) if training else None
            extra = {"row_classes": rc, "rng": rng}
        aug = cx.augment(table, arcs, spec, np, conv=conv, labels=rec["labels"], extra=extra)
        secs.append(time.perf_counter() - t)
        items.append((key, rec, aug, app.catalogue.group_of(key), f"{si}:{li}"))
    X, y, g, names = builder.labeled_set(items, app.store, np,
                                         layer_of=lambda key, rec: app.regions.label_layer(key))
    return X, y, g, names, float(np.mean(secs)), float(np.max(secs))


def cv_transductive(spec_net, X_fit, X_score, y, g, names):
    """Leave-slices-out with a DIFFERENT design matrix for the held-out rows
    (labels hidden): fit on X_fit[train], score X_score[test]."""
    from sklearn.metrics import balanced_accuracy_score, log_loss
    cv, _kind = ms.make_cv(y, g)
    classes = np.unique(y)
    losses, baccs = [], []
    for tr, te in cv.split(X_fit, y, g):
        est = ms.fit_estimator(ms.build_estimator(spec_net, names), X_fit[tr], y[tr])
        P = ms._full_proba(est, X_score[te], classes)
        losses.append(log_loss(y[te], P, labels=classes))
        baccs.append(balanced_accuracy_score(y[te], classes[P.argmax(1)]))
    return float(np.mean(losses)), float(np.mean(baccs))


rows = []
spec_net = ms.ModelSpec(hidden=tuple(hidden), backend="sklearn", max_iter=1000, seed=0)
say(f"net {ms.size_text(hidden)}, leave-slices-out CV; {len(V)} variants")
for name, spec in V.items():
    X, y, g, names, t_mean, t_max = design(spec)
    cv, kind = ms.make_cv(y, g)
    t = time.perf_counter()
    loss, bacc = ms.cv_evaluate(spec_net, X, y, cv, names, groups=g)
    row = {"variant": name, "spec": spec.to_dict(), "n_features": int(X.shape[1]),
           "n_rows": int(len(y)), "cv": kind, "logloss": float(loss), "bacc": float(bacc),
           "augment_s_mean": t_mean, "augment_s_max": t_max,
           "fit_s": time.perf_counter() - t}
    rows.append(row)
    say(f"{name:28s} feats {X.shape[1]:4d}  log-loss {loss:.4f}  bal.acc {bacc:.1%}  "
        f"augment {t_mean * 1e3:6.0f} ms/slice (max {t_max * 1e3:.0f})  fit {row['fit_s']:.0f} s")

# Labels as context is transductive: score the held-out slices with their
# own annotations visible (the interactive loop) AND with none (a fresh slice).
for p in (0.3, 0.6):
    spec = cx.ContextSpec(labels=cx.LabelSpec(dropout=p, seed=0))
    X_fit, y, g, names, t_mean, t_max = design(spec, training=True)
    X_vis, _y, _g, _n, _a, _b = design(spec, training=False)
    X_zero, _y, _g, _n, _a, _b = design(spec, training=False, zero_labels=True)
    for label, X_score in (("labels visible", X_vis), ("labels hidden", X_zero)):
        t = time.perf_counter()
        loss, bacc = cv_transductive(spec_net, X_fit, X_score, y, g, names)
        name = f"labels p={p:g} ({label})"
        rows.append({"variant": name, "spec": spec.to_dict(), "n_features": int(X_fit.shape[1]),
                     "n_rows": int(len(y)), "cv": "leave-slices-out (transductive)",
                     "logloss": loss, "bacc": bacc, "augment_s_mean": t_mean,
                     "augment_s_max": t_max, "fit_s": time.perf_counter() - t})
        say(f"{name:28s} feats {X_fit.shape[1]:4d}  log-loss {loss:.4f}  bal.acc {bacc:.1%}")

base = rows[0]["logloss"]
say("---- vs no context (negative = better) ----")
for r in rows[1:]:
    say(f"{r['variant']:28s} {r['logloss'] - base:+.4f} log-loss  {r['bacc'] - rows[0]['bacc']:+.1%} bal.acc")

search = None
if n_search:
    spec = V["every kind"]
    X, y, g, names, _tm, _tx = design(spec)
    schema = list(schema0) + cx.schema_entries(spec, names, conv)
    say(f"search: {n_search} trials over {len(names)} features, mask over "
        f"{len(ms.feature_groups(names, schema))} groups")
    res = ms.run_search(X, y, g, names, schema, n_trials=n_search, seed=0,
                        max_iter=ms.SEARCH_MAX_ITER, patience=ms.SEARCH_PATIENCE,
                        refit_max_iter=1000, backend="sklearn",
                        progress_cb=lambda d, n, b: say(f"  trial {d}/{n}"
                                                        + (f" best {b.cv_score:.4f}" if b else "")))
    kept = res.spec.features if res.spec.features is not None else names
    groups = ms.feature_groups(names, schema)
    on = [gname for gname, cols in groups.items() if any(c in kept for c in cols)]
    off = [gname for gname in groups if gname not in on]
    say(f"winner: log-loss {res.spec.cv_score:.4f}, bal.acc {res.spec.cv_bacc:.1%}, "
        f"{res.spec.layers_text()}; groups kept: {on}; dropped: {off}")
    search = {"spec": res.spec.to_dict(), "groups_on": on, "groups_off": off,
              "n_trials": res.n_trials, "elapsed_s": res.elapsed_s}

stamp = time.strftime("%Y-%m-%d")
path = os.path.join(OUT, f"context_ablation_{stamp}.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump({"hidden": list(hidden), "n_slices": len(slices), "rows": rows, "search": search},
              f, indent=1)
say("wrote", path)
root.destroy()
