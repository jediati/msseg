# Design note: an offline, task-free region encoder for the labeler

Status: **harvest and train built** (2026-09-15); the probe and the labeler
exposure are still proposals. See "What exists" at the end.

## What it is for

The labeler's dense classifier is a 130-ish column statistics row pushed
through a `16-8` net into the classes of *one* task. Its 8-wide layer is a
good descriptor of exactly that task -- gland vs not gland -- and of nothing
else: whatever the row knows about stroma, necrosis, lumen or fat that the
task did not need has been squeezed out. The next session, with a different
class vocabulary, starts over.

This note proposes an **offline** process that learns a small, **task-free**
descriptor of a region from unlabeled slides -- an encoder over the region's
statistics row whose latent still separates the semantic kinds of tissue --
and exposes the **frozen encoder** to the labeler as ordinary columns. Every
labeling task then becomes a small head on the same latent, the magic fill and
ROI proposals get a stable similarity space, and a handful of labels goes
further than it does over 130 raw columns.

    mspath-embed harvest --tiff-folder ".../WSI" --process-profile ".../mincol_blur_edge.profile.json" --level 4 --out harvest_L4/
    mspath-embed train   harvest_L4/ --dim 8 --out encoders/wsi_L4.msenc
    mspath-embed probe   encoders/wsi_L4.msenc --session ".../WSI/session.json"

The harvest runs the labeler's own prime tile by tile and keeps rows and arcs
(no pixels), so it is minutes per slide; training is seconds to minutes on a
CPU; the probe reports whether the encoder is worth loading, and in
particular whether it **transfers across tasks**, which is the whole point.

## What a row encoder can and cannot do

The row is a hand-built descriptor. An encoder over it can only re-express
what the row already says; it cannot add information the row lacks. So:

* Two tissue kinds the raw row cannot separate stay inseparable in any
  latent of it. The probe's first line therefore fits a linear head on the
  **raw row** for every label vocabulary available; that is the ceiling.
* The reason to expect a good latent exists at all is that the `16-8` net
  already separates gland from stroma from the same row.
* The risk is not capacity but the **objective**. The row's variance is
  dominated by nuisance and redundancy: four sigmas times three planes times
  four reductions are heavily correlated, and `area`, `bbox_w/h` and the
  absolute intensity level swing over orders of magnitude. A plain
  autoencoder or PCA spends its dimensions on that, and tissue kind becomes
  a small residual. The design question is which **label-free signal** points
  the latent at tissue kind.

## The signal: spatial coherence

Tissue kinds are contiguous, and a region -- one basin of a blurred edge
field at 3 % persistence -- is far smaller than the structure it tiles, so
most arcs of the region graph join two regions of the *same* kind. The
evidence is already in the tree: neighbour voting cut held-out region errors
115 -> 66 (`docs/mscoupon_edge_pairs.md`), which only works when adjacency
mostly means agreement.

That gives a task-free objective. A region's latent must **predict its
neighbourhood**: for region `i`, a positive is a region reached by a short
random walk over the arcs (1-3 hops), negatives are regions drawn from
anywhere on the slide, and the loss is InfoNCE over the latents. It is
DeepWalk / node2vec over the region graph with the statistics row as the node
input, which is what makes it *inductive* -- the encoder maps a row it has
never seen, on a slide it has never seen, and needs no graph at inference.
Whatever varies inside a gland but not between gland and stroma is pushed out
of the latent, because it does not help tell a walk-neighbour from a random
region; whatever distinguishes kinds is kept, because it does.

Two more terms keep it honest:

* a light **reconstruction head** (linear, latent -> z-scored row), so the
  latent does not discard a whole channel the walk objective happens not to
  need -- the next task might;
* a **variance + covariance penalty** on the latent over the batch
  (VICReg-style), so every dimension keeps unit spread and the dimensions
  decorrelate: no dead dimensions, and columns a `StandardScaler` treats
  sensibly.

```
L = InfoNCE(z_i, z_walk(i); negatives = batch)  +  0.1 * MSE(row_hat, row)  +  0.05 * (var + cov)
```

**Input corruption** is the augmentation: at every step whole column
*groups* are dropped at the input (one sigma, one plane, the histogram
block -- `model_search.feature_groups` already names them) and single columns
are replaced by values drawn from the harvest's marginal (SCARF-style). The
latent has to encode what is shared across scales and planes rather than one
column's noise.

Positional columns (`FieldConventions.positional`) never enter, as
everywhere else. `area` and `bbox_w/h` do enter, log-transformed.

## Architecture

```
input   n_row (z-scored per column over the harvest; log1p on area / bbox)
enc     Linear 64, SiLU, Linear 32, SiLU, Linear d          d = 8 (or 16)
proj    Linear d -> 32, SiLU, Linear 32                     the InfoNCE head (discarded after training)
rec     Linear d -> n_row                                   the reconstruction head (discarded)
```

Roughly 12 k parameters. The projection head is the usual SimCLR trick: the
contrastive loss is taken on `proj(z)`, and `z` itself keeps the information
the loss would otherwise strip. AdamW 1e-3, batch 4096, ~50 epochs over a
few hundred thousand rows: a minute on a CPU, seconds on a GPU. `dim` is a
CLI knob; 8 matches the layer the labeler already uses, and the probe says
whether 16 buys anything.

### Baselines that must be beaten

* **PCA-8** (whitened) over the same z-scored rows. Trial zero. If PCA-8
  already passes the transfer test below, ship PCA-8.
* **A supervised multi-task trunk.** If several saved sessions carry
  different class vocabularies, one `n_row -> 16 -> 8` trunk with one softmax
  head per session's task yields an 8-wide layer forced to encode *every*
  distinction anyone has labeled. No new objective, and
  `edge_model.embed` already exposes hidden layers. It cannot represent a
  kind nobody has labeled, which is why the walk objective is still the
  proposal, but it is the strongest cheap comparison.

### What stays on the shelf

A **pixel encoder** (a conv net over a patch cut around the region's seeding
extremum, with the mask as a channel) is the only thing that can add
information the row lacks. It is deferred until the probe shows two kinds the
raw row cannot separate; the earlier draft of this note described it, and the
harvest below stores enough (item key, region id, extremum) to cut patches
later without re-priming.

## The harvest

`mspath-embed harvest` drives the labeler's own engine, so a harvested row is
exactly a row the labeler would show:

1. Read the profile; open each TIFF in the folder through
   `PyramidImageSource`; register it with a `SlideEngine`.
2. Tile the requested level into `--roi 4096` square `Item` rects (the
   level taken whole as an overview item when it fits) with the profile's
   `slide.halo`, rank the tiles by tissue content off a coarse level, skip
   those under `--min-tissue`, and `prime_item` the rest in that order
   (serial, as the engine insists; 2-9 s per tile, ~250 tiles at level 0 for
   the 4 Gpx test slide). The per-level persistence pin therefore resolves
   against the tile with the MOST tissue rather than whichever comes first
   -- a glass tile's tiny range would make every other tile under-simplified
   -- and is written to `harvest.json` and restored on resume;
   `--persistence-absolute` bypasses the pin altogether.
3. Per item, at the profile's persistence **and** at 0.5x and 2x of it (the
   encoder should be robust to the slider): `feature_table()` with the halo
   rows dropped as `ensure_record` drops them, and `region_arcs()` translated
   to the table's row indices.
4. Drop rows that describe slide background (mean OD below a floor): most of
   a slide is blank and teaches nothing; a random-walk positive into blank
   would also be a lie about coherence.
5. Write one shard per (tile, persistence): `rows float32[n, n_row]`,
   `arcs int32[m, 2]`, `feature_id`, `ext_x/ext_y`, the item key, the
   absolute persistence. A tile whose shards exist is skipped, so the harvest
   resumes; `train` can subsample by slide.

`harvest.json` records the profile and its hash, the level, the pinned
absolute persistence per level, the column names, and the per-column
z-score mean/std and the log-transformed set -- the input convention the
encoder bundle carries forward.

The coupon labeler's harvest is the same loop over slices instead of tiles: a
second small driver over the same shard writer.

## The encoder bundle

`wsi_L4.msenc` is a zip of:

* `weights.npz` -- the encoder's matrices (three linears), **numpy**, so
  inference needs no torch and a compiler-less collaborator loads it with the
  pure-Python wheel;
* `meta.json` -- `arch`, `dim`, the input convention (column names, z-score
  mean/std, log set), `scope` (`"L4"`), the profile hash, the latent
  whitening measured over the harvest, training settings, the probe report,
  and `hash`: SHA-1 of the weights.

The encoder is level-specific in the same sense the classifier is: a sigma is
in level pixels, so the level-4 row and the level-0 row are different
measurements with the same names. The bundle's scope goes through the
existing scope gate unchanged.

## Exposure to the labeler

A new headless framework module, `msseg.labeler.embedding`:

* `EncoderBundle.load(path)` -> `embed(X: float64[n, n_row]) -> float32[n, d]`
  over a matrix whose columns are the bundle's names (assembled by name, as
  `TrainingSetBuilder.feature_matrix` does), plus the metadata above.
* `columns(bundle, table)` -> a NEW `FeatureTable`: the record's rows plus `d`
  columns named `emb<hash8>__z00 .. z07`.

Two ways to wire it, both cheap; the first is the proposal:

1. **As columns.** `_stream_stat_slices` already yields the context-augmented
   table per record; the embedding is one more augmentation in that chain,
   applied *before* the context columns, so ring reductions of the latent
   (`ring_mean__emb…__z03`) fall out of the existing context code with no
   change. The name carries the encoder hash on purpose: the compatibility
   gate is a set comparison of names, so a model trained over one encoder's
   columns is refused under another with **no new gate logic**.
   `schema_entries` files the block as one channel so Optimize's per-group
   mask drops the whole latent in a single trial (the ablation for free), and
   an `--embedding-only` spec (`features` = the eight names) is the "small
   head on the frozen latent" configuration. The prefix never starts with
   `mean_` / `std_` / `hist`, so `FieldConventions` sees no phantom channel.
   The magic fill gets one more branch in `row_vectors` (metric `embedding`,
   cosine over the `emb` columns) and `propose.py` can score uncertainty in
   the same space.
2. **As a pipeline step.** Because the latent is a pure function of the row,
   the encoder can also sit as a frozen transformer between `FeatureSubset`
   and `StandardScaler` inside `build_estimator`, keeping the fingerprint
   equal to the raw row names. Nothing outside the estimator changes, but
   the magic fill and the context columns then never see the latent. Kept as
   an option; the columns route is what the tools want.

State follows the context precedent exactly: the model's own encoder (path
and hash) rides the pickle as `stack["embedding"]` and the session model
record; the Model tab's picked encoder (`view.embedding`) is the NEXT
model's. Both keys exist only when set, so a session without an encoder is
byte-identical to today's. Cost is negligible: three small matmuls per
record, recomputed on every commit like the context columns.

## The probe

`mspath-embed probe` answers "is this encoder worth loading" before anyone
opens the GUI. Given one or more session files with annotations, for each
label vocabulary it rebuilds the labeled rows (the same records, the same
`TrainingSetBuilder`) and reports, with `make_cv` leave-slides-out and the
edge model's balanced logistic as the head:

| design | measures |
|---|---|
| raw row, linear head | the ceiling: what the row can separate at all |
| raw row, the `16-8` net | today's labeler |
| PCA-8, linear head | trial zero |
| latent-8, linear head | the proposal |
| raw row ++ latent-8, the net | what the labeler will actually see |

plus the two numbers this note exists for:

* **Transfer.** Fit the head for task A (say gland vs not) on the frozen
  latent, then check task B (stroma vs background) is *still* linearly
  separable in the same latent, against the same test on the `16-8` layer
  of a net trained on task A alone. The gap is the task-specificity the
  encoder removes.
* **Label efficiency.** The learning curve of latent-8 vs raw row at 10, 20,
  50, 100 labeled regions per class. A frozen 8-d input is where few labels
  should pay off most.

And one picture: a 2-D UMAP of the latent over a harvested tile coloured by
the arc-neighbour agreement of a k-means over it, so a latent that clusters
along scale or intensity rather than tissue is visible at a glance.

## Layout

```
packages/mslabeler/src/msseg/labeler/embedding.py        bundle (numpy inference), columns, schema entries
packages/mslabeler/src/msseg/labeler/embedding_train/    shard format, walk sampler, model, objective, train loop, probe   ([torch] extra)
packages/mspath/src/msseg/mspath/embed.py                `mspath-embed`: the slide harvest driver over SlideEngine + tiles
packages/mscoupon/src/msseg/mscoupon/embed.py            `mscoupon-embed`: the slice harvest driver (later)
```

## Open questions

* **Walk length and the merge hierarchy.** One hop is the safest positive;
  three hops crosses a thin gland wall. The harvest at three persistences
  offers a second positive that costs nothing: a region and its surviving
  parent at the next persistence share an extremum. Whether that positive
  helps or blurs kinds is a probe run, not a guess.
* **Cross-slide genericity.** One slide's rows are one stain and one
  scanner. The encoder is only as generic as the folder it was trained on;
  the harvest should record per-slide row statistics so a probe on a new
  slide can flag drift before the gate lets a model through.
* **Where `i0` comes from** for the OD conversion in the profile's base
  chain: per tile today, which is wrong for a tile with no background. Take
  it from the overview once per slide and record it in the harvest.
* **Whether 8 is enough.** The layer the labeler uses is 8 wide for one
  task; a task-free latent may want 16. The probe's transfer test decides.

## What exists (2026-09-15)

* `msseg.labeler.embedding` -- `EncoderBundle`: the input convention, the
  dense layers, the latent whitening, `embed(X, names)` (numpy only),
  `columns(table)` -> the `emb<hash8>__z..` columns, `schema_entries()`,
  and the `.msenc` zip (`weights.npz` + `meta.json`, hash-checked on load).
* `msseg.labeler.embedding_train` -- `shards` (`HarvestWriter` /
  `HarvestReader`, resumable per-(tile, persistence) `.npz` shards, arcs as
  row indices that never cross a shard), `model` (the torch encoder with
  its projection and reconstruction heads; `pca_layer`), `train` (the walk
  InfoNCE + reconstruction + var/cov objective with group dropout and
  marginal corruption; 5 % of anchors held out and scored clean at the end),
  `cli` (the `train` sub-command).
* `msseg.mspath.embed` -- `mspath-embed harvest | train`. The harvest ranks
  tiles by tissue off a coarse level (`--min-tissue`, `--tissue-max`),
  refuses a level the reader does not have (never clamps: the level is the
  scope), records every tile at `--factors` of the profile's persistence,
  writes the per-level pin after the first tile and restores it on resume,
  and drops rows by an optional `--blank FIELD OP VALUE` rule.
* Tests: `packages/mslabeler/tests/test_embedding.py` (bundle round trip,
  shard format, walk sampler, PCA without torch, the walk objective
  separating three synthetic kinds) and `packages/mspath/tests/test_embed.py`
  (the helpers; a real harvest + resume + train on the selftest's synthetic
  slide when the extension imports).

First real run, `Subset1_Test_10.tiff` at level 4 with
`mincol_blur_edge.profile.json`: one 2940 x 4096 tile with tissue (the
second was glass), 55 k / 145 k / 13 k regions at factors 1 / 0.5 / 2, six
seconds to harvest; 213 k rows x 145 columns in 29 groups, 780 k arcs;
PCA-8 explains 85 % of the variance; the walk encoder (145 -> 64 -> 32 -> 8,
40 epochs, 31 s on the GPU) scores 0.59 held-out top-1 against 4096
candidates (chance 0.0002). Whether that latent transfers across tasks is
the probe's question, still open.
