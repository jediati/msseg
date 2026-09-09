# Edge pairs: can the region graph fix the region classifier?

*Experiment notes, 2026-09-05. Script:
[`packages/mscoupon/experiments/edge_pairs.py`](../packages/mscoupon/experiments/edge_pairs.py);
numbers: [`edge_pairs_2026-09-05.json`](../packages/mscoupon/experiments/edge_pairs_2026-09-05.json).*

## Question

The labeler classifies each living MSC region on its own from its statistics
row (the `dense (tuned)` kind; the size sweep put the best net at **16-8**,
see [mscoupon_labeler.md](mscoupon_labeler.md)). Regions are not independent:
they tile the slice and touch through saddles, and a void is surrounded by
metal. Two questions:

1. Take the trained net's hidden activations as an **embedding** of a region.
   For every edge of the region graph whose two ends are both labeled, can a
   *pair* model on the two embeddings predict whether the edge crosses classes
   (**same / different**) -- and can it do so better than the region net's own
   implied answer ("the two argmaxes differ")?
2. If so, do the edge predictions **fix region misclassifications** when the
   region graph is allowed to vote?

## Setup

* **Data**: the labeler's autosaved session
  (`%APPDATA%\mscoupon-labeler\last_session.json`): `tomo_sample_2_051_rec`,
  slices `recon_00005-00008` and `recon_00084-00088` (9 slices, 3232^2), the
  `default` profile (`edges -> blur` topology field, `normalize(gmm)` base
  chain, ascending, 3 % persistence, 22 measurement channels x 4 reductions +
  extremum = **115 features** after dropping positions), 418 gestures
  resolving to **5,977 labeled regions** (classes 1 and 2) out of 55,939.
  Slice `recon_00005` alone carries 5,266 of the labels.
* **Graph**: 148,009 edges; 14,892 join two labeled regions, **2,079 (14 %)**
  cross classes. The compiled extension in the tree at the time predated
  `Msc2DPipeline.region_arcs()`, so the graph was **pixel 4-adjacency**
  (`magic_fill.arcs_from_labels`) with **no saddle values** -- the "barrier"
  features below are only the two regions' extremum values and their
  difference. Rerun with a current build to get MSC arcs + saddle depths.
* **Protocol**: leave-slices-out, 5 folds (`model_search.make_cv`, grouped by
  slice). Per fold: fit the 16-8 region net (sklearn backend, balanced
  weights) on the training slices only; embed every region (`h1` = 16-d and
  `h2` = 8-d ReLU layers after FeatureSubset + StandardScaler); build pair
  features (`|e_a - e_b|`, `e_a * e_b`, barrier); train the pair models on
  the training slices' labeled edges; score on the test slices' labeled
  edges. Baselines come from the same fold's region net.
* **Voting**: on the test slices, three rounds of
  `score_i(c) = log P_i(c) + lambda * sum_j [log(1 - p_diff_ij) if class_j == c else log p_diff_ij]`
  over ALL edges (labeled or not), with `p_diff` from the 8-d pair model;
  scored on the labeled test regions.

## Results (pooled over held-out slices)

Edge task, the different-class edges being the ones that matter:

| pair model | bal. acc | AUC | log-loss | diff recall | diff precision |
|---|---|---|---|---|---|
| baseline: argmax differs | 0.958 | 0.958 | 0.363 | 93.5 % | 88.4 % |
| baseline: 1 - sum_c P_a(c) P_b(c) | 0.958 | 0.982 | 0.111 | 93.5 % | 88.4 % |
| 16-d embedding, logistic | 0.863 | 0.954 | 0.168 | 75.2 % | 82.5 % |
| 16-d embedding, MLP 32-16 | 0.839 | 0.974 | 0.222 | 69.5 % | 86.5 % |
| **8-d embedding, logistic** | **0.982** | **0.989** | **0.055** | **97.3 %** | **94.5 %** |
| raw 115-d abs-difference, logistic | 0.941 | 0.989 | 0.083 | 90.1 % | 89.1 % |
| barrier (extremum) only | 0.598 | 0.816 | 0.424 | 29.8 % | 32.3 % |

Region task, before / after voting with the 8-d pair model:

| | acc | bal. acc | true 1 -> pred 2 | true 2 -> pred 1 |
|---|---|---|---|---|
| region net 16-8 alone | 0.981 | 0.979 | 86 | 29 |
| + voting, lambda = 1 | **0.989** | **0.985** | **39** | 27 |
| + voting, lambda = 0.5 | 0.988 | 0.984 | 45 | 28 |

Per fold (lambda = 1): 0.937 -> 0.939, 0.996 -> 0.996, 0.906 -> 0.917,
0.987 -> 0.992 (the big slice), **0.795 -> 0.892** (the two smallest label
sets). Voting never hurt a fold and helped most where the net was weakest.

## What this says

* **Pairwise same/different is learnable beyond the region net.** The 8-d
  pair model catches 97 % of boundary edges at 94 % precision against 93 % /
  88 % for "argmax differs", with a fifth of the log-loss. The embedding is
  doing real work: the raw-feature pair model is worse than the 8-d one, and
  the 16-d layer is a worse pair space than the 8-d one (its ReLU features are
  less linearly comparable; a learned distance on it might fix that, but the
  8-d layer already works).
* **The graph fixes the isolated flips.** Errors fall from 115 to 66, almost
  all on the "metal predicted as void" side -- the isolated-region kind of
  mistake a neighbourhood vote is good at. The confusion the net makes on
  genuinely ambiguous regions (true 2 -> pred 1) barely moves.
* **Caveats.** Slice `recon_00005` holds 88 % of the labels, so the pooled
  numbers lean on it and the small folds are noisy; there were no saddle
  depths (pixel adjacency), so the barrier had no signal to give; the pair
  model trains on labeled-both edges, which are mostly interior edges.

## How to reproduce

```bash
# from a source checkout; uses the labeler's LAST AUTOSAVED session
PYTHONPATH="packages/mscoupon/src;packages/msseg-viz/src" python packages/mscoupon/experiments/edge_pairs.py [out_dir]
```

~65 s to prime nine slices (GPU gradient on), ~2 min for the five folds. It
prints per-slice counts, per-fold reports and the two tables above (lines are
prefixed `##` so they can be grepped out of MSCEER's own output) and writes
`edge_experiment.json` to `out_dir` (default: beside the script). To run on a
different session, save it in the labeler first (auto-save writes it every few
seconds), or point `config_io.session_path(app="mscoupon-labeler")` elsewhere.

Knobs inside the script: the region net spec (`ModelSpec(hidden=(16, 8), ...)`),
the pair feature sets (`feats` dict), which pair model drives the voting
(`edge_model = m` on `"emb8 + barrier (LR)"`), `lambda` and the number of
voting rounds.

## Re-measured with MSC arcs (2026-09-08)

With the refreshed extension the graph is MSCEER's saddle-joined living-region
pairs (99,981 edges; 10,726 labeled, 7.3 % crossing) and the barrier carries
the saddle depth. Pooled over the same folds:

| pair model | bal. acc | AUC | log-loss | diff recall | diff precision |
|---|---|---|---|---|---|
| baseline: argmax differs | 0.947 | 0.947 | 0.323 | 91.3 % | 79.6 % |
| baseline: 1 - sum P_a P_b | 0.947 | 0.973 | 0.100 | 91.3 % | 79.6 % |
| 16-d embedding + barrier, logistic | 0.910 | 0.963 | 0.126 | 85.0 % | 68.8 % |
| **8-d embedding + barrier, logistic** | **0.975** | 0.980 | **0.058** | **95.9 %** | **88.8 %** |
| raw 115-d abs-difference + barrier | 0.954 | 0.982 | 0.062 | 91.9 % | 86.6 % |
| barrier (saddle depth) only | 0.968 | **0.989** | 0.105 | 94.7 % | 87.4 % |

Voting with the 8-d pair model: region errors 115 -> 79 (acc 0.981 -> 0.987,
bal. acc 0.979 -> 0.983). The saddle depth alone is a strong boundary
signal (0.968), which the pixel-adjacency run could not see.

## Landed as

Paths 1-3 below are in the labeler (2026-09-08): `edge_model.py` (embedding,
pair features, `fit_edge_model` / `predict_pdiff`, `vote`, `evaluate_edges`),
the `dense (tuned) -> edges` / `custom FC -> edges` kinds with `freeze base`,
the `N` toggle, the Edge model panel with **Evaluate edges**, the `learned`
magic-fill metric and the `flipped by neighbours` / `boundary p(diff)`
coloring modes -- see [mscoupon_labeler.md](mscoupon_labeler.md), "Edge models".

## Paths forward

1. **Real arcs.** Rebuild the extension at a pin with `region_arcs()` (MSCEER
   `7cb4703`+, see CLAUDE.md "magic fill") and rerun: the edges become the
   MSC's saddle-joined pairs with saddle values, so `barrier = saddle -
   max(ext_a, ext_b)` is a real persistence-style depth. Expect the barrier
   row of the table to stop being noise, and the voting to sharpen.
2. **`learned` magic-fill metric.** `magic_fill.build_ladder` takes a metric
   name; the 8-d pair model's `p_diff` over the arcs is a learned
   dissimilarity that slots in beside `mean` / `cosine` / `proba`
   (`magic_fill.METRICS`, `EXTRA_METRICS` for a per-arc array handed in via
   `extra`). Needs the embedding + pair model to live on the labeler
   (`_clf` pipeline -> `embed()` as in the script), computed once per
   classification like `_pred`.
3. **"Refine with neighbours" in the labeler.** Apply the voting to
   `_pred[(si, li)]` after Classify (a button next to Classify, or a
   checkbox), with `lambda` as a control; show the flipped regions (a
   coloring mode like `P(class k)` / uncertainty) and let SHIFT-accept take
   them as labels. Measure on the confusion matrix, which already compares
   frozen predictions with live labels.
4. **Train the pair model properly.** Optimise it like the region net
   (leave-slices-out, log-loss on the different class), try a small
   symmetric net on `(|d|, prod)` of the 8-d layer, and a **learned distance
   on the 16-d layer** (a linear map before `|d|`), which may recover what
   the 16-d rows lost here. Balance: 14 % of labeled edges are boundaries.
5. **Joint training.** The natural end state is one model that sees a region
   and its neighbourhood (a tiny message-passing net over the region graph
   with the statistics rows as node features and saddle depth as edge
   feature). The voting result above is the cheap upper-bound check that it
   is worth building: the graph carries information the row does not.
6. **More labels off the big slice.** The weak folds are the thinly labeled
   slices; a few magic fills on `recon_00084-00088` would make the held-out
   numbers far less noisy than any modelling change.
