"""Foundation-model ablation: do pooled ViT tokens help the region or seam model?

Phase 0 of docs/design_foundation_features.md -- the go / no-go experiment.

Usage (from a source checkout; DINOv2 needs `pip install timm`, and the first
run downloads the weights into the Hugging Face cache):

  PYTHONPATH="packages/mslabeler/src;packages/mscoupon/src;packages/msseg-viz/src;packages/mspath/src"
      python packages/mspath/experiments/fm_ablation.py [out_dir]
          [--session PATH] [--task NAME|UID] [--model dinov2_vitb14_reg]... [--scale 1]...
          [--layers -1]... [--dims 64] [--tile 448] [--batch 8] [--weights CKPT]
          [--hidden 16-8] [--no-ring]
          [--fake]   (a random patch-local backbone, no download: a plumbing smoke test)

Loads the mspath labeler's last autosaved session (or --session), activates
the task (--task, default the active one), primes its items headlessly and
materialises every record exactly as Train does. For each extraction config
(model x scale x layers) it computes each item's token grid ONCE over the
item's pixels + halo -- cached on disk under <app data>/fm_cache, so a rerun
is free -- and fits a PCA basis over all items' tokens (unsupervised; it sees
the held-out slides' pixels but no labels). Then, by the task's kind:

REGION task -- every region pools the cells it covers (mean / purity /
interior / boundary + the reliability columns `tokens`, `purity_frac`) and ONE
fixed dense net is scored leave-slides-out on column subsets:

    stats                 the statistics row alone (the baseline)
    stats + fm mean       + area-weighted mean token (+ reliability)
    stats + fm all        + every pool
    fm only               every pool, no statistics
    stats + ring          ring mean/contrast over the statistics   (--no-ring skips)
    stats + fm + ring     the same over statistics + fm mean

POLYLINE task -- every seam pools the cells it RUNS THROUGH (each crack
weights its two flank pixels' cells; `seam_cells` = length in cell units),
and the task's own seam model (Model-tab seam spec, its estimator) is scored
leave-slides-out:

    seam                  the seam descriptor as the task builds it (baseline)
    seam + fm pair        the flank regions' fm mean columns enter the pair
                          terms (only when the pair block reads the table, not
                          an embedding input's net -- reported as `embed`)
    seam + fm crossing    + the seam's own crossing block
    seam + fm both        both
    fm crossing only      the crossing block alone

Every variant is reported overall AND by size in tokens (<1, 1-4, 4-16, >=16;
a region's area, a seam's length): a unit below one token gets a fraction of
one vector, so where the gain lands says whether a finer grid (--scale 2) or
an upsampler is needed. Also reported: extraction seconds per item, peak VRAM.

Writes ``fm_ablation_<date>.json`` beside the script (or in out_dir).
"""
import json
import os
import sys
import time
import tkinter as tk
import warnings
from dataclasses import replace

import numpy as np

warnings.simplefilter("ignore")
from msseg.labeler import context as cx
from msseg.labeler import fields
from msseg.labeler import model_search as ms
from msseg.labeler import seam_model as sm
from msseg.labeler import session_doc
from msseg.labeler.foundation import extract as fx
from msseg.labeler.foundation import pool as fp
from msseg.labeler.foundation.cache import TokenCache, cache_key
from msseg.labeler.foundation.project import fit_basis, project
from msseg.labeler.foundation.spec import POOLS, FoundationSpec, column_names, is_foundation_column
from msseg.labeler.training import TrainingSetBuilder
from msseg.mscoupon import config_io
from msseg.mspath.labeler import LabelerApp

BUCKETS = ((0, 1), (1, 4), (4, 16), (16, np.inf))


def say(*a):
    print("##", *a, flush=True)


# ---- arguments --------------------------------------------------------------
args = sys.argv[1:]
opt = {"model": [], "scale": [], "layers": []}
hidden, dims, tile, batch = (16, 8), 64, 448, 8
weights = session = out_dir = task_sel = None
ring = True
fake = False
i = 0
while i < len(args):
    a = args[i]
    if a in ("--model", "--scale", "--layers"):
        opt[a[2:]].append(args[i + 1]); i += 2
    elif a == "--hidden":
        hidden = ms.parse_sizes(args[i + 1])[0]; i += 2
    elif a == "--dims":
        dims = int(args[i + 1]); i += 2
    elif a == "--tile":
        tile = int(args[i + 1]); i += 2
    elif a == "--batch":
        batch = int(args[i + 1]); i += 2
    elif a == "--weights":
        weights = args[i + 1]; i += 2
    elif a == "--session":
        session = args[i + 1]; i += 2
    elif a == "--no-ring":
        ring = False; i += 1
    elif a == "--task":
        task_sel = args[i + 1]; i += 2
    elif a == "--fake":
        fake = True; i += 1
    else:
        out_dir = a; i += 1
models = opt["model"] or (["fake"] if fake else ["dinov2_vitb14_reg"])
scales = [int(s) for s in opt["scale"]] or [1]
layer_sets = [tuple(int(x) for x in s.split(",")) for s in opt["layers"]] or [(-1,)]
OUT = out_dir or os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# ---- session, prime, records ------------------------------------------------
root = tk.Tk(); root.withdraw()
app = LabelerApp(root, autosave=False)
doc = config_io.read_json_file(session or config_io.session_path(app=app.SESSION_APP))
say("session notes:", app._apply_session_doc(doc, "fm ablation"))
if task_sel is not None:
    pick = [t for t in app.tasks if task_sel in (t.uid, t.name)]
    if not pick:
        raise SystemExit(f"no task {task_sel!r}; tasks: "
                         + ", ".join(f"{t.name} ({t.uid}, {t.kind})" for t in app.tasks))
    if pick[0] is not app._task and not app._activate_task(pick[0]):
        raise SystemExit(f"could not activate task {pick[0].name}")
KIND = app._task_kind()
say(f"task {app._task.name} ({app._task.uid}, {KIND})")
app._run()
t0 = time.perf_counter()
root.update()
while getattr(app, "_run_active", False) or app.engine.pending_work():
    root.update(); time.sleep(0.05)
    if time.perf_counter() - t0 > 3600:
        raise SystemExit("prime timeout")
say(f"primed in {time.perf_counter() - t0:.0f} s")
t0 = time.perf_counter()
slices = app._all_stat_slices("fm ablation")
if not slices:
    raise SystemExit("no records: " + app.status_var.get())
say(f"materialised {len(slices)} item records in {time.perf_counter() - t0:.0f} s")
conv = getattr(app, "FIELDS", fields.DEFAULT)
builder = TrainingSetBuilder(conv)
cache = TokenCache(None if fake else os.path.join(session_doc.app_data_dir(app.SESSION_APP), "fm_cache"),
                   max_items=4)


class FakeBackbone:
    """--fake: a fixed random 14-px patch conv (no context, no download)."""
    patch, mean, std = 14, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)

    def __init__(self, dim=48):
        import torch
        g = torch.Generator().manual_seed(0)
        self.conv = torch.nn.Conv2d(3, dim, 14, stride=14).to(fx.default_device())
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(self.conv.weight.shape, generator=g) * 0.05)

    def __call__(self, x):
        return [self.conv(x.float())]


def halo_of(key):
    p = app.engine.primed.get(key)
    return int(p.halo) if p is not None else app._halo()


# ---- scoring ----------------------------------------------------------------
spec_net = ms.ModelSpec(hidden=tuple(hidden), backend="sklearn", max_iter=1000, seed=0)


def scores(y, P, classes, size, kind):
    """Overall + per-size-bucket (log-loss, balanced accuracy, n) of
    out-of-fold probabilities `P` (columns = `classes`)."""
    from sklearn.metrics import balanced_accuracy_score, log_loss
    P = np.clip(P, 1e-12, 1)
    P = P / P.sum(1, keepdims=True)

    def score(m):
        if m.sum() < 2 or len(np.unique(y[m])) < 2:
            return {"n": int(m.sum()), "logloss": None, "bacc": None}
        return {"n": int(m.sum()), "logloss": float(log_loss(y[m], P[m], labels=classes)),
                "bacc": float(balanced_accuracy_score(y[m], classes[P[m].argmax(1)]))}
    out = {"cv": kind, "all": score(np.ones(len(y), bool))}
    for lo, hi in BUCKETS:
        out[f"tok[{lo},{hi})"] = score((size >= lo) & (size < hi))
    return out


def oof_region(X, y, g, names, size):
    """The fixed dense net, leave-slides-out."""
    cv, kind = ms.make_cv(y, g)
    classes = np.unique(y)
    P = np.zeros((len(y), len(classes)))
    for tr, te in cv.split(X, y, g):
        est = ms.fit_estimator(ms.build_estimator(spec_net, names), X[tr], y[tr])
        P[te] = ms._full_proba(est, X[te], classes)
    return scores(y, P, classes, size, kind)


def oof_seam(F, y, g, size, seam_spec):
    """The task's seam model (its own estimator), leave-slides-out -- the
    split `seam_model.evaluate_seams` makes, kept out-of-fold per seam."""
    cv, kind = ms.make_cv(y, g)
    classes = np.unique(y)
    P = np.zeros((len(y), int(classes.max()) + 1))
    for tr, te in cv.split(F, y, g if kind == "slices" else None):
        est = sm._fit(sm._make_estimator(seam_spec), F[tr], y[tr], seam_spec)
        P[te] = sm._proba_columns(est, F[te], P.shape[1])
    return scores(y, P[:, classes], classes, size, kind)


def report(rows, cfg, vname, res):
    rows.append({"config": cfg, "variant": vname, **res})
    a = res["all"]
    say(f"{vname:20s} feats {res['n_features']:4d}  log-loss {a['logloss']:.4f}  "
        f"bal.acc {a['bacc']:.1%}  " + "  ".join(
            f"{b}:{res[b]['logloss']:.3f}(n={res[b]['n']})" if res[b]["logloss"] is not None
            else f"{b}:-(n={res[b]['n']})" for b in res if b.startswith("tok")))


def timed(fn, *a):
    t = time.perf_counter()
    res = fn(*a)
    res["fit_s"] = time.perf_counter() - t
    return res


def design(tables):
    items = [(key, rec, tab, app.catalogue.group_of(key), f"{si}:{li}")
             for (si, li, key, rec, _t), tab in zip(slices, tables)]
    try:
        X, y, g, names = builder.labeled_set(
            items, app.store, np, layer_of=lambda key, rec: app.regions.label_layer(key),
            gestures_of=app._gestures_for_key)
    except Exception as exc:              # TrainingProblem: nothing labelled
        raise SystemExit(f"task {app._task.name}: {exc}")
    return X, y, g, list(names)


def subset(X, names, keep):
    idx = [j for j, n in enumerate(names) if keep(n)]
    return X[:, idx], [names[j] for j in idx]


def ringed(tables):
    rs = cx.ContextSpec(kinds=("ring_mean", "ring_contrast"))
    return [cx.augment(t, app.regions.arcs(key, np), rs, np, conv=conv, labels=rec["labels"])
            for (_si, _li, key, rec, _t), t in zip(slices, tables)]


# ---- region task ------------------------------------------------------------
def fm_cols(table, rec, z, spec):
    """`table` + the spec's fm columns; `z` = (projected grid, stride, offset)."""
    grid, stride, offset = z
    return fp.columns(table, rec["labels"], grid, stride, spec, offset, conv.id_field)


def run_region(cfg, spec, zs, rows, baseline):
    mean_spec = replace(spec, pools=("mean",))
    full = [fm_cols(t, rec, zs[key], spec)
            for _si, _li, key, rec, t in slices]
    X, y, g, names = design(full)
    size = X[:, names.index(column_names(spec)[-2])]
    pre = f"fm{spec.hash8}__"
    rel = (pre + "tokens", pre + "purity_frac")
    designs = {k: subset(X, names, f) for k, f in {
        "stats": lambda n: not is_foundation_column(n),
        "stats + fm mean": lambda n: (not is_foundation_column(n)
                                      or n.startswith(pre + "mean_") or n in rel),
        "stats + fm all": lambda n: True,
        "fm only": lambda n: is_foundation_column(n),
    }.items()}
    if ring:
        if "stats + ring" not in baseline:
            designs["stats + ring"] = design(ringed([t for *_x, t in slices]))[0::3]
        mean_only = [fm_cols(t, rec, zs[key], mean_spec)
                     for _si, _li, key, rec, t in slices]
        Xr, yr, _gr, nr = design(ringed(mean_only))
        assert len(yr) == len(y)
        designs["stats + fm + ring"] = (Xr, nr)
    for vname, (Xv, nv) in designs.items():
        shared = vname in ("stats", "stats + ring")
        if shared and vname in baseline:
            res = baseline[vname]
        else:
            res = timed(oof_region, Xv, y, g, nv, size)
            res["n_features"] = len(nv)
            if shared:
                baseline[vname] = res
        report(rows, cfg, vname, res)


# ---- polyline task ----------------------------------------------------------
def run_polyline(cfg, spec, zs, rows, baseline):
    """Seams gather the cells they run through. The flank regions' fm columns
    reach the pair terms by appending them in `_context_table` -- the hook
    `_seam_items` already streams every record through -- so the descriptor
    is built by the task's own code in both modes."""
    mean_spec = replace(spec, pools=("mean",))
    orig = app._context_table
    mode = {"fm": False}

    def with_fm(key, rec, cspec, np_, training=False):
        table = orig(key, rec, cspec, np_, training)
        if not mode["fm"]:
            return table
        return fm_cols(table, rec, zs[key], mean_spec)

    app._context_table = with_fm
    try:
        built = {}
        for flag in (False, True):
            mode["fm"] = flag
            items = app._seam_items("fm ablation seams")
            if not items:
                raise SystemExit(f"task {app._task.name}: {app.status_var.get()}")
            built[flag] = items
    finally:
        app._context_table = orig
    seam_spec = app._seam_spec()
    embed = built[False][0][8]
    say(f"seam spec {seam_spec.describe()}; pair terms read: {embed}")
    cross = {it[0]: fp.pool_seams(it[4], *zs[it[0]]) for it in built[False]}

    def gathered(flag, with_cross, base=True):
        Fs, ys, gs = [], [], []
        for key, _si, _li, _rec, _g, cls, F, _n, _e in built[flag]:
            parts = (([np.asarray(F, np.float32)] if base else [])
                     + ([cross[key].astype(np.float32)] if with_cross else []))
            Fs.append(np.concatenate(parts, axis=1))
            ys.append(np.asarray(cls, np.int64).reshape(-1))
            gs.extend([app.catalogue.group_of(key)] * len(F))
        F, y, g = np.concatenate(Fs), np.concatenate(ys), np.asarray(gs, object)
        m = y > 0
        return F[m], y[m], g[m]

    F0, y0, _g = gathered(False, True)
    if not len(y0):
        raise SystemExit(f"task {app._task.name}: no labelled seams")
    size = F0[:, -1]                          # seam_cells: the length in cell units
    for vname, (flag, wc, base) in {"seam": (False, False, True),
                                    "seam + fm pair": (True, False, True),
                                    "seam + fm crossing": (False, True, True),
                                    "seam + fm both": (True, True, True),
                                    "fm crossing only": (False, True, False)}.items():
        if vname == "seam" and "seam" in baseline:
            report(rows, cfg, vname, baseline["seam"])
            continue
        if vname == "seam + fm pair" and embed == "net":
            say("seam + fm pair: skipped (the pair terms read the embedding input's net)")
            continue
        F, y, g = gathered(flag, wc, base)
        assert len(y) == len(y0)
        res = timed(oof_seam, F, y, g, size, seam_spec)
        res["n_features"] = int(F.shape[1])
        res["embed"] = embed
        if vname == "seam":
            baseline["seam"] = res
        report(rows, cfg, vname, res)


# ---- the configs ------------------------------------------------------------
rows, runs, baseline = [], [], {}
for model in models:
    for scale in scales:
        for layers in layer_sets:
            spec0 = FoundationSpec(model=model, weights=weights, layers=layers, tile=tile,
                                   scale=scale, dims=dims, pools=POOLS)
            cfg = f"{model} x{scale} L{','.join(map(str, layers))}"
            say(f"==== {cfg} (extract key {spec0.extract_key()})")
            bb = None
            secs, grids = [], []
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except ImportError:
                torch = None
            for si, li, key, rec, _table in slices:
                halo = halo_of(key)
                ck = cache_key(key, rec.get("level"), halo, spec0.extract_key())

                def compute():
                    global bb
                    if bb is None:
                        bb = FakeBackbone() if fake else fx.load_backbone(spec0)
                    arr, _geom, _read_s = app.engine.read_item(app._item_at(si, li), halo)
                    t = time.perf_counter()
                    out = fx.extract_grid(bb, arr, spec0, batch=batch)
                    secs.append(time.perf_counter() - t)
                    return out
                grids.append(cache.get_or_compute(ck, compute))
            vram = (torch.cuda.max_memory_allocated() / 2**30
                    if torch is not None and torch.cuda.is_available() else None)
            fx.release()
            say(f"grids: {len(secs)} computed ({np.mean(secs) if secs else 0:.2f} s/item), "
                f"{len(slices) - len(secs)} cached; D={grids[0][0].shape[-1]}"
                + (f"; peak VRAM {vram:.2f} GB" if vram is not None else ""))
            t = time.perf_counter()
            basis = fit_basis([g for g, _s in grids], dims)
            spec = spec0.with_basis(basis)
            # key -> (projected grid, stride, pixel offset = the halo)
            zs = {key: (project(grid, basis), stride, (halo_of(key),) * 2)
                  for (_si, _li, key, _rec, _t), (grid, stride) in zip(slices, grids)}
            say(f"basis {time.perf_counter() - t:.1f} s; columns {spec.hash8}")
            (run_polyline if KIND == "polyline" else run_region)(cfg, spec, zs, rows, baseline)
            runs.append({"config": cfg, "spec": spec.to_dict(include_basis=False),
                         "hash8": spec.hash8,
                         "extract_s_per_item": float(np.mean(secs)) if secs else None,
                         "n_computed": len(secs), "peak_vram_gb": vram,
                         "token_dim": int(grids[0][0].shape[-1])})

# ---- summary ----------------------------------------------------------------
ref = "seam" if KIND == "polyline" else "stats"
base = baseline[ref]["all"]
say(f"---- vs {ref} (negative log-loss = better) ----")
for r in rows:
    if r["variant"] == ref:
        continue
    a = r["all"]
    say(f"{r['config']:34s} {r['variant']:20s} {a['logloss'] - base['logloss']:+.4f} log-loss  "
        f"{a['bacc'] - base['bacc']:+.1%} bal.acc")
stamp = time.strftime("%Y-%m-%d")
path = os.path.join(OUT, f"fm_ablation_{stamp}.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump({"task": {"name": app._task.name, "uid": app._task.uid, "kind": KIND},
               "hidden": list(hidden), "dims": dims, "tile": tile, "n_items": len(slices),
               "buckets": [list(b) for b in BUCKETS], "runs": runs, "rows": rows}, f, indent=1,
              default=str)
say("wrote", path)
root.destroy()
