# Proposed dense FC optimization

Once the labeled training set is frozen, optimize the **dense FC** estimator
used by the mscoupon labeler (`StandardScaler` + `sklearn.neural_network.MLPClassifier`
with the current fixed `hidden_layer_sizes=(64, 32)`).

This note proposes:

1. a **primary comparison metric** (plus secondaries to break ties)
2. a **search space of variational parameters** specific to dense FC

It does **not** re-open feature engineering or switch model families.

---

## Setup assumptions

- Classes: `1 = bg`, `2 = fg` (class `0` unlabeled / ignored).
- Labels are scarce (~few hundred supervised rows) and can be imbalanced.
- Deployed use is a hard decision at a fixed threshold (today effectively
  `argmax` / `0.5` on the positive class), but probability quality still
  matters for reviewing borderline regions.
- Evaluation must be **out-of-sample**. Do **not** use train accuracy
  (what the labeler status line currently reports for dense FC).

Recommended protocol:

- **Stratified k-fold CV** on the frozen labeled set (`k = 5`, or
  leave-one-slice-out if multiple slices become available).
- Same feature schema as the active profile (whatever the labeler already
  feeds the pipeline).
- Fix `random_state` for folds and for each candidate so runs are
  comparable.

---

## A) Metric to compare models

### Primary metric: balanced accuracy

\[
\text{balanced accuracy}
  = \tfrac{1}{2}\bigl(\text{recall}_{\text{bg}} + \text{recall}_{\text{fg}}\bigr)
\]

Why this as the selection criterion:

- Matches the labeler’s actual product behavior: a **class decision**, not a
  ranking list.
- Treats bg and fg equally even when the labeled set is skewed (e.g.
  `labels_2` was ~2:1 bg:fg).
- Directly penalizes “always pick the majority class” models that look fine
  under plain accuracy.
- Is already the quantity that correlated with useful model ranking in the
  scratch comparison work.

**Select the candidate with the highest mean balanced accuracy across folds.**
Report mean ± std; prefer the simpler / cheaper candidate when means are
within ~1 percentage point.

### Secondary metrics (tie-breakers and diagnostics)

Use these to break near-ties and to catch failure modes the primary misses:

| Metric | Role |
|---|---|
| **ROC AUC** | Ranking / probability quality; threshold-independent. Prefer higher when balanced accuracy is tied. |
| **Macro F1** | Confirms both classes are precise *and* recalled. |
| **FG recall / FG precision** | Application-facing: voids you miss vs false voids you accept. |
| **Log-loss** (optional) | Calibrated probability quality; useful if you later soft-threshold or triage hard examples. |

Do **not** optimize on train accuracy or on a single held-out shuffle of a
small set without stratification.

### Decision rule (concrete)

For each hyperparameter candidate \(h\):

1. Run stratified 5-fold CV.
2. Record fold-wise balanced accuracy, ROC AUC, macro F1.
3. Rank by mean balanced accuracy.
4. If two candidates differ by \(< 0.01\) in mean balanced accuracy, prefer
   the one with higher mean ROC AUC; if still tied, prefer the smaller /
   faster architecture (fewer params, lower `max_iter` used).

Optional later step (after architecture is chosen): retune the decision
threshold on validation probabilities to maximize balanced accuracy or a
user cost (e.g. weight FG misses higher). That is **not** part of the
architecture search itself.

---

## B) Variational parameters for dense FC

Current baseline (labeler):

```python
make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=0),
)
```

Variational parameters below are the ones worth searching for this
estimator. Keep `StandardScaler` fixed (it belongs in the pipeline; do not
search “on/off”).

### 1. Architecture (highest priority)

| Parameter | What it is | Suggested search values |
|---|---|---|
| `hidden_layer_sizes` | Depth × width of the MLP | `(32,)`, `(64,)`, `(128,)`, `(64, 32)` *(baseline)*, `(128, 64)`, `(128, 32)`, `(256, 64)`, `(64, 64)`, `(128, 64, 32)`, `(64, 32, 16)` |
| `activation` | Nonlinearity | `"relu"` *(default)*, `"tanh"` |

Notes:

- With ~300 labeled rows, deeper than 3 hidden layers is unlikely to help.
- Prefer a **sparse grid**: one-layer vs two-layer vs three-layer, and a
  few widths at each depth, rather than every combination.
- `identity` / `logistic` are low priority; keep them out of the first pass.

### 2. Regularization / capacity control

| Parameter | What it is | Suggested search values |
|---|---|---|
| `alpha` | L2 penalty on weights | `1e-5`, `1e-4` *(default)*, `1e-3`, `1e-2`, `1e-1` |
| `early_stopping` | Stop on an internal validation split | `False` *(current)*, `True` |
| `validation_fraction` | Fraction held out when early stopping | `0.1`, `0.15`, `0.2` (only if early stopping on) |
| `n_iter_no_change` | Patience for early stopping / tol | `10`, `20`, `30` |

Small-data prior: stronger `alpha` and early stopping often beat simply
growing width.

### 3. Optimization dynamics

| Parameter | What it is | Suggested search values |
|---|---|---|
| `solver` | Optimizer | `"adam"` *(default)*, `"lbfgs"` |
| `learning_rate_init` | Adam step size | `1e-4`, `3e-4`, `1e-3` *(default)*, `3e-3` |
| `batch_size` | Minibatch size (adam only) | `"auto"`, `16`, `32`, `64` |
| `max_iter` | Epoch / iteration budget | `500`, `1000` *(current)*, `2000` |

Notes:

- For \(N \lesssim 500\), `"lbfgs"` is often strong and removes batch-size /
  learning-rate knobs; include it as a first-class competitor.
- If using adam + early stopping, `max_iter` is mostly an upper bound;
  still keep it high enough that early stopping, not the ceiling, decides.

### 4. Class imbalance (fit-time, not an MLP ctor knob)

`MLPClassifier` has no `class_weight=` argument (unlike the random forest).
Still treat balance as a variational choice:

| Parameter | What it is | Suggested values |
|---|---|---|
| sample weighting | Inverse-frequency `sample_weight` at `fit` | off *(current)*, on (`n / (2 n_c)` per class) |

This is important: the scratch PyTorch dense FC used class-weighted loss and
often beat or matched RF on balanced accuracy; the labeler’s current
sklearn dense FC does **not** weight classes.

### 5. Fixed / out of scope for v1

Leave these alone until the above search is done:

- Feature list / profile statistics schema (separate experiment).
- Including vs dropping `area` (already measured; weak).
- Switching to PyTorch / custom nets (only if sklearn MLP saturates).
- `random_state` (fix for reproducibility; do not search it).
- Dropping `StandardScaler`.

---

## Suggested search schedule

Cheap first pass (~20–40 candidates):

1. `solver ∈ {adam, lbfgs}`
2. `hidden_layer_sizes ∈ {(64,), (64, 32), (128, 64), (128, 64, 32)}`
3. `alpha ∈ {1e-4, 1e-3, 1e-2}`
4. sample weighting ∈ {off, on}
5. For adam only: `early_stopping=True`, `learning_rate_init ∈ {1e-3, 3e-4}`

Second pass (around the winner):

- Nudge width (±1 octave) and depth (±1 layer).
- Refine `alpha` on a log grid around the best value.
- If adam won: try `batch_size ∈ {16, 32}` and `validation_fraction ∈ {0.1, 0.2}`.

Stop when mean balanced accuracy gains fall below ~0.5–1 point, or when
extra capacity only improves train fit / worsens fold variance.

---

## What to record per candidate

Enough to re-run and to update the labeler default later:

- hyperparameter dict (including sample-weight on/off)
- fold-wise and mean ± std: balanced accuracy, ROC AUC, macro F1
- FG precision / recall
- wall-clock train time and classify time (optional but cheap)
- effective epochs / `n_iter_` when early stopping is on
- seed / fold definition

Promote the winner into `_make_model("dense FC", ...)` only after the
frozen-set CV comparison, and keep the previous `(64, 32)` config as a
baseline row in the same table.

---

## Short recommendation

| Item | Choice |
|---|---|
| **Compare models by** | mean stratified-CV **balanced accuracy** |
| **Break ties with** | ROC AUC, then model simplicity / speed |
| **Vary first** | `hidden_layer_sizes`, `solver`, `alpha`, sample weighting, adam early-stopping / `learning_rate_init` |
| **Leave fixed** | `StandardScaler`, feature schema, `random_state` |

That is enough to turn “one architecture” into a disciplined search without
confusing architecture choice with feature choice or threshold choice.
