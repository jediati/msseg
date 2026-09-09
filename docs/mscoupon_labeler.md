# mscoupon labeler (`mscoupon-labeler`)

The labeler is the viewer (`mscoupon-gui`, see [mscoupon_gui.md](mscoupon_gui.md))
with an annotation panel: the user paints **classes** onto the living MSC regions
of a slice, trains a per-region classifier on the statistics table, and exports
training sets. This page covers the annotation model and the drawing tools;
the engine, session file and config schema are the viewer's.

```bash
mscoupon-labeler                  # GUI
mscoupon-labeler --selftest       # Tk wiring check, no engine needed
# from a source checkout (both namespace packages on the path):
PYTHONPATH="packages/mscoupon/src;packages/msseg-viz/src" python -c "from msseg.mscoupon.labeler import main; import sys; sys.argv=['x','--selftest']; main()"
PYTHONPATH=packages/mscoupon/src pytest packages/mscoupon/tests/test_labeling.py packages/mscoupon/tests/test_magic_fill.py
```

## Layout

Three panes. The **left** pane is data navigation and processing *selection*:
the compute-profile dropdown, the session (folders, files, the sequence tree,
which divide the height through draggable sashes) and Run, pinned to the
bottom so it is always visible. Clicking a TIFF in the file list or an
unprimed row of a sequence previews it; the Image dropdown then computes the
chosen channel (filter chain, base chain, or a derived scale-space response)
on that preview, so a profile can be judged before any Run. A colour TIFF
previews as RGB, with `color` / `color_c<i>` entries beside the channels; its
chains start with a `color` card (see [mscoupon_gui.md](mscoupon_gui.md),
"Colour input"). The **right**
pane is annotation management (classes, tools, the
Magic rows, the per-class interaction lists, Save/Load annotations) and the
classifier (Train/Classify, the model strip, the confusion matrix, exports,
Save/Load classifier). The **center** is a notebook whose inactive tabs are
hidden:

| tab | holds |
|---|---|
| **Processing** | the profile tools (New/Dup/Rename/Delete, Save/Load profile) and the profile being edited, in two columns: filter chain + base channel, MSC parameters + statistics channels |
| **View** (default) | the slice canvas, hover readout, slice navigation, image/overlay/alpha controls and the persistence entry |
| **Model** | the classifier kind, a read-only description of its architecture, the **Edge model** panel (the `-> edges` kinds) and the **Optimize network** search (trials, time limit, seed, feature-subset toggle, progress line) |
| **Analysis** | **Predictions vs annotations** (the regions behind a confusion cell; double-click a row to go there) and the **Size sweep** report; a home for plots later |

Two link-labels say what is in effect: the Run section is headed by the
**selected workflow** as two compact chains --
`topo field: base→b(1.5)→e(0.7)→msc(asc, 10%)` (the field the MSC runs on,
then manifold and persistence, `mf` for merge forest) and
`stats: base→norm(gmm)→12ch×4` (the base chain the statistics are measured
on, then channels × reductions); hover for the code table -- and the classifier
section, above Train/Classify, by the **active model** (the kind Train will
build, or the trained model and its feature count); clicking either opens its
tab. The selected tab rides the session as
`view.center_tab` and is restored by name. The hotkeys are window-wide, so `Tab` still toggles the overlay from any
tab. The viewer (`mscoupon-gui`) keeps its two-pane layout.

## Annotations are gestures, not region ids

An annotation (`labeling.Interaction`) is one gesture in **image coordinates**
bound to one slice (`"folder/basename"`) and one class. Nothing stores a region
id: on every render the slice's gestures are rasterized against whatever
label raster is current (`touched_ids` → `resolve_slice`), later gestures
painting over earlier ones on any region both touch. That is what lets a
persistence change, a filter edit, or a switch between `msc` and
`merge_forest` simplification re-resolve the same annotations onto the new
decomposition. `annotations.json` (v2) holds the raw geometry; it also rides
the session autosave and is written beside every exported config.

Optional per-gesture **`meta`** (a JSON dict) carries provenance for display —
today only the magic fill writes it. Resolution never reads it, so a stale
`meta` can never change what a gesture paints; files without it are unchanged.

## Tools

| Tool | Gesture | Paints |
|---|---|---|
| squiggle | polyline (a click is a tap) | every region under the line |
| box | drag a rectangle | every region overlapping it |
| lasso | drag a closed polygon | every region under the filled polygon |
| magic | press, drag up/down, release | a similarity flood from the pressed region |
| blobber | as magic | the flood in the active class **and** its bounding regions in the ring class |
| *SHIFT + drag* | a box, any tool | **accepts** the classifier's predictions under it as `taps` |

Hotkeys: `1..4` arm a class, `0`/Escape disarm, `M` selects magic, `B` the
blobber, `Tab`
toggles the overlays, `Ctrl-Z`/`Ctrl-Y` undo/redo, `R` train + classify,
`C` classify. Middle/right drag always pans; a right-click opens the
annotation menu for the region under it.

### "Will be painted" preview

While any gesture is in flight the regions it would paint on release are
shown on a **transient canvas layer** in a brightened, opaque version of the
class color (the accept box shows each region in its *predicted* class color).
The layer ignores the overlay-alpha slider — the preview is the point while it
shows — and is dropped on release or Escape. Squiggles preview incrementally
(only the newest segment is rasterized); boxes and lassos recompute per move,
sampling with a stride once the slab exceeds ~1 Mpx so a full-image box on a
3232² slice stays around 10 ms. The commit is always the exact rasterization.
**Escape** now abandons the gesture in flight (any tool); with nothing in
flight it disarms the class as before.

### Magic fill

Press on a region (the **seed**) with a class armed. The fill grows over the
**living-region adjacency graph** at the current persistence: a region joins
when there is a path from the seed whose every hop is under the threshold.
Drag **up** to raise the threshold (more regions), **down** to lower it;
the canvas HUD shows `t`, how many regions are in, and their pixel count.
Release paints; Escape abandons.

*What is compared* (the **Magic:** row under the tool selector):

| metric | measures | needs |
|---|---|---|
| `mean` (default) | \|Δmean\| over the chosen channels, each z-scored by that column's spread on the slice | `mean_<channel>` |
| `bhattacharyya` | Gaussian overlap from mean and std per channel | `mean_`, `std_` |
| `histogram` | Hellinger distance between the regions' per-channel histograms (the concatenated bin fractions, scaled so the row sums to one) over the chosen channels that carry bins | `statistics.histogram` |
| `cosine` | `1 − cos` between the regions' whole statistics rows: every column except ids, positions and histogram bins, each z-scored over the slice. Ignores the channel list. | any statistics |
| `proba` | total variation (half the L1) between the classifier's class-probability vectors; a region the model never scored is at distance 1, so it joins last. Ignores the channel list. The press is refused until the slice has been classified at the current commit. | a Classify |
| `barrier` | saddle height above the seed's extremum: the persistence-style flood anchored at a point | MSC region arcs (`ext_filtered`) |

| mode | compares |
|---|---|
| `anchor` (default) | every candidate with the **seed** — no drift |
| `chain` | each region with the neighbour it grows from — follows gradients |

Channels are the names of the statistics spec (`base`, `blur_s1.5`, …),
comma-separated; unknown names are ignored, none valid falls back to `base`.

*How it is computed.* At the press, `magic_fill.build_ladder` turns the metric
into one weight per arc and runs a **priority flood** from the seed
(`growth_order`: a bottleneck/minimax Dijkstra whose ties are broken by the
most seed-like frontier region), giving every region the threshold at which
it joins — the **join ladder** — *and* the order in which the flood admits
them. The flood takes a **hop gain** `g` (the `hop×` entry, default 1.1,
range 1..2): the cost of a path is its bottleneck inflated by `g` per hop,
`max_i w_i · g^(hops after i)`, so an equally similar region far from the seed
costs more than a neighbour and the ladder is no longer flat across a
homogeneous plateau — the reason a few pixels of drag used to sweep half a
slice. `g = 1` is the pure bottleneck; the HUD shows `g1.1` and its `t` then
reads in inflated units. A drag tick is a rank on that order, i.e. a prefix of
it (`drag_to_rank`: linear near the start so single regions are reachable —
the `drag` entry sets the screen pixels per region, default 4 — quadratic
further out so a long drag sweeps a thousand-region ladder), so every pixel of
drag adds or removes one connected region. The prefix, not a
threshold-closed set, is deliberate: a bright outlier seed makes its first
neighbour's dissimilarity the bottleneck for most of the slice, so hundreds
of regions share one join value and a plain threshold jumps from one region
to half the slice at that rung; the HUD's `t` is the join value of the last
region admitted. The first threshold is the
one last released with the same metric/mode/channels/hop gain in this session, else a
natural break in the ladder (`initial_rank`: the largest relative gap within
its first 5 %).

*Adjacency.* The engine stores `rec["arcs"]` from
`Msc2DPipeline.region_arcs()` — MSCEER's `livingRegionArcs()`: living-region
pairs joined by a saddle, with the saddle value, in both `msc` and
`merge_forest` modes. An extension without it (an older wheel) makes the fill
derive 4-neighbour pixel adjacency from the label raster on the first press
(`arcs_from_labels`, ~0.3 s at 3232², cached on the record); `barrier` is then
unavailable and falls back to `mean`. In `msc` mode saddles with fewer than
two living extremum arcs are dropped by MSCEER, so a few regions can be
unreachable; the HUD's `k/n` shows how many are.

*What is stored.* The release commits **one `taps` interaction with a point
per grown region at its seeding extremum** (`ext_x/ext_y`, falling back to the
region's first pixel), plus `meta = {tool: "magic", seed, seed_id, threshold,
metric, mode, channels, hop_gain, n_regions, arcs}`. So the fill re-resolves after a
persistence change like every other gesture: on a coarser decomposition
several points collapse into one region; on a finer one only the sub-regions
containing a stored point stay painted. The annotation row reads
`#12 magic (37)` and hovering it shows the taps plus a dashed ring at the seed.
One undo step removes the whole fill.

### Blobber

The blobber is the magic fill plus a **ring**: the regions immediately
adjacent to the core (`magic_fill.ring_for_rank`, the arc graph's neighbours
of the prefix that are not in it) preview and commit in a second class. Click
on a void with class 2 armed and the void is class 2 with the material
bounding it in class 3; the drag grows the core and the ring follows it, and
once the core has taken every reachable region the ring is empty. The ring
class is the **`ring`** option in the Magic row: `next` (default) is the class
after the active one, wrapping past the last, or pick a fixed id; an id equal
to the active class falls back to `next`, and a two-class store (only class
1) refuses the press with a status line. The release commits two `taps`
interactions, `blobber ring (n)` first and `blobber core (n)` second, so on a
re-decomposition that merges a ring point's region into a core point's the
core wins; one undo step removes both.

## Edge models (the `-> edges` kinds)

The region net scores each living region from its statistics row alone.
Regions tile the slice and touch through saddles, and the edge-pairs
experiment ([mscoupon_edge_pairs.md](mscoupon_edge_pairs.md)) showed that a
pair model on the net's own hidden layer tells same-class from
different-class edges of the region graph better than the net's argmax, and
that letting the graph vote fixes the isolated flips. Two model kinds stack
that **edge model** on top of a dense base:

| kind | base | on top |
|---|---|---|
| `dense (tuned) -> edges` | the last Optimize / size-sweep winner | the edge model |
| `custom FC -> edges` | a dense net whose hidden sizes are typed on the Model tab (`base hidden`, default `16-8`) | the edge model |

(`custom FC` on its own is also a kind.) **Train (R)** on an edge kind fits
the base, then the pair model on every labeled edge -- the status line reports
both; with **freeze base** on, Train keeps the current base and refits only
the edges, so edge variants can be tried on one base. **Classify (C)** runs
the base, scores every arc of each slice's living-region graph with the pair
model (p(diff) = the probability the edge crosses classes), then runs
`rounds` of neighbour voting -- `score_i(c) = log P_i(c) + lambda * sum_j
[log(1 - p_ij) if class_j == c else log p_ij]` -- and the refined classes are
*the* prediction: the overlay, the confusion matrix, SHIFT-accept, the CSV and
the training set all see them. **N** flips between the edge kind and its base
(the cached predictions re-vote in milliseconds, no forward pass), so raw vs
refined is one keystroke and the confusion matrix is the before/after.

The **Edge model** panel on the Model tab holds the pair model's settings --
`layer` (last = the narrow layer, best in the experiment; previous = the wider
one), `model` (balanced logistic, or an MLP 32-16), the pair `features`
(`|d|`, `product`, and the saddle `barrier`: `saddle - max(ext_a, ext_b)` and
`|ext_a - ext_b|`, zeros on pixel adjacency), `C`, and the voting `lambda` and
`rounds` (these two apply at once to cached predictions; the rest wait for
Train) -- and **Evaluate edges**: a leave-slices-out report that refits the
base per fold and scores the pair model against the base's own answers
(`argmax differs`, `1 - sum P_a P_b`) on boundary recall / precision, plus
region errors before and after voting. It runs on the worker thread like
Optimize; Cancel stops it after the current fold.

Under Train/Classify the right panel shows the edge readout: how many pairs
the model was fit on, the share of boundaries, the held-out numbers once
evaluated, whether voting is on, and how many regions it flipped on this
slice. Two coloring modes come with an edge model: **flipped by neighbours**
(regions whose class voting changed) and **boundary p(diff)** (each region's
max p(diff) over its arcs). The magic fill gains the **`learned`** metric:
the pair model's p(diff) per arc as the flood's dissimilarity (refused until
the slice is classified with an edge model).

The edge model rides the classifier pickle (**v4**: `edge`, plus a `stack`
with the custom hidden sizes and the edge settings), the session's model
record (`edge: true`) and the session view (`neighbours`). A pickle whose
edge model was fit over other features loads its base and drops the edges.
Records without MSC arcs (an older extension) fall back to pixel adjacency,
where the barrier carries no saddle depth.

## Held-out vs fit check, and finding the errors

Two kinds of error count appear and they are not the same thing. The
confusion matrix on the right panel (rows = annotated class, columns =
predicted) compares the **current** model's predictions with the labels it
was **trained on** -- a fit check, so its off-diagonal counts are small.
**Evaluate edges** and **Optimize** report **held-out** numbers: every slice
is scored by models that never saw it (leave-slices-out folds), which is the
honest estimate and always higher. The report rows say `(held-out)`.

Clicking a confusion cell highlights its regions on the current slice and
lists them -- every slice, largest first, with the model's probabilities and
whether neighbour voting changed the class -- on the **Analysis** tab;
double-clicking the cell opens that tab. Double-clicking a row (or Enter)
opens the View tab on that slice, centred on the region's seeding extremum
(zooming in to 1:1 if further out), with the gestures touching it outlined
and the cell's highlight still on.

## Classifier and exports

Unchanged by the above: Train/Classify on the per-region statistics table
(positions excluded), SHIFT-accept to turn predictions into annotations, CSV
export (one row per living region, class 0 kept as negatives), and the
image training set (`train/` raw TIFFs + `labels/` per-pixel class masks,
annotations winning over predictions).

Colour statistics (`mean_color_c0`, `mean_dizenzo_largest_s1.5`, ...) and
histogram bins (`hist00_base`, ...) are ordinary columns of that table, so
they enter every model by name; a channel's bins stand or fall together with
its other reductions in the Optimize feature-subset search. Switching either
on changes the field set, so a model saved under the old profile is refused by
the compatibility gate ("profile adds: ...") until retrained -- by design, a
bin must mean the same thing the model learned it as.

## Optimize network (the `dense (tuned)` kind)

The plain `dense FC` kind is one fixed network, `StandardScaler ->
MLPClassifier((64, 32))`, fit on every labeled region with no held-out score.
**Optimize network** (the button on the Model tab, or `O`) searches the dense
space instead and installs the winner as the `dense (tuned)` model:

* **What is searched** (`model_search.py`, headless): depth (1-4 layers) and
  each layer's width (8-256), L2 `alpha`, learning rate, batch size, early
  stopping, and -- with *search feature subset* on -- which measurement
  **channels** the network sees (all reductions of a channel stand or fall
  together, so twelve channels are twelve booleans, not sixty). The pipeline
  still consumes the full profile schema: the subset is a `FeatureSubset` step
  *inside* the estimator, so the feature fingerprint, the profile-compatibility
  gate and the pickle/predict paths are those of every other kind.
* **How a candidate is scored**: mean held-out **log-loss** (balanced accuracy
  alongside) under cross-validation that **leaves whole slices out** whenever
  three or more slices carry labels (`StratifiedGroupKFold` by slice index;
  plain stratified folds below that, folds capped by the rarest class). Regions
  on one slice share the scan's intensity drift, so a split that mixes them
  reports how well the net memorised the slice, not how it transfers. Log-loss
  rather than accuracy because the class probabilities are what the magic-fill
  `proba` metric and the uncertainty coloring consume. Every fit uses balanced
  sample weights.
* **The searcher**: Optuna's TPE sampler with median pruning when `optuna` is
  installed (`pip install msseg-mscoupon[optimize]`), otherwise a seeded random
  search over the same space (the log says which). Trial 1 is always the
  un-tuned baseline, so the winner never loses to what Train would have built,
  and the readout states the gain over it.
* **Controls**: trials (default 300), a time limit in **minutes** (default
  480 = an overnight run; 0 = none; the session keeps it as `timeout_s`), the
  seed (sampler, folds and networks: same labels + seed = same winner), and the
  feature-subset toggle; they ride the session as `view.model_search`. The
  search runs on a worker thread -- labeling stays live, the canvas HUD shows
  `Optimizing k/n`, the progress line shows the best so far plus elapsed time,
  the per-trial average and the time left -- and **Cancel** stops after the
  current trial, keeping the best. When it finishes the winner is installed,
  **saved** as `models/tuned_<timestamp>.pkl` under the labeler's session
  folder (`%APPDATA%\mscoupon-labeler` on Windows) and recorded on the
  session, then classified -- so an unattended run is remembered: the session
  reloads its most recent recorded model on restore, and *Load classifier…*
  opens any of them. The log lists the permutation importances of the winner's
  columns.
* **Size sweep** (the Analysis tab): *how small can the
  network be?* Enter a ladder of architectures (`64-32, 32-16, 16-8, 8-4, 4`;
  layers joined by `-`, rungs by `,`) and the trials to spend per rung, then
  **Sweep sizes**. Each rung is a fixed-size search -- the architecture is
  pinned and alpha, learning rate, batch, early stopping, dropout and the
  feature subset are searched -- so a small net is judged at its own best
  settings rather than the big net's. The table fills in as rungs finish: size,
  parameter count, held-out log-loss and balanced accuracy, the loss relative to
  the best rung, trials, and the best settings; rungs more than 5 % worse than
  the best are red, the smallest rung within 5 % is shaded green, and the
  summary names the best, the smallest that still holds, and where the ladder
  breaks down. The best rung is installed, saved and classified when the sweep
  ends; **Install selected size** makes any other rung the model (its refit
  estimator, saved the same way). The rows, specs and table are also written as
  `models/sweep_<timestamp>.json`, and the log carries the table. The Optimize
  time limit caps the whole sweep; Cancel stops after the current trial and
  keeps the rungs already scored.
* **Trial budget**: a trial is a 5-fold CV of one candidate. Each fold trains
  for at most `SEARCH_MAX_ITER` = 300 epochs with an early-stop patience of 10
  (the winner is refit afterwards with the full 1000-epoch budget); a torch
  trial also reports its folds' mean held-out loss every 25 epochs so Optuna's
  median pruner can stop a hopeless candidate early. Mini-batch training is a
  Python loop of ~1.5 ms per optimizer step on either device, so candidates
  with a batch below `TORCH_MIN_BATCH` = 256 rows train on sklearn's CPU MLP
  (6 ms per epoch per fold) while full-batch and 256/512 candidates use the
  stacked torch loop; `spec.backend` records the request, the readout the
  backend that actually trained it. On ~6000 labeled regions a trial is then
  seconds rather than the minutes a 16-row batch cost on torch.
* **Backend** (`torch` / `sklearn` / `auto`): with PyTorch installed
  (`pip install msseg-mscoupon[torch]`, CUDA wheels from
  `download.pytorch.org/whl/cu128`) the net is `torch_mlp.TorchMLPClassifier`
  -- the same `hidden/alpha/lr/batch/early stopping` semantics plus
  **dropout**, which then joins the search -- and every fold of a trial trains
  as **one stacked batch** on the GPU (`torch_mlp.train_stacked`: weights are
  `(folds, in, out)` tensors, per-fold standardisation on the padded batch,
  per-fold early stopping keeping each fold's best epoch). `sklearn` forces
  `MLPClassifier` on the CPU; `auto` is torch when importable. The readout
  names the backend and device; a torch pickle needs torch to load but
  predicts on a CPU-only machine.
* **Afterwards**: the Model tab readout renders the winning spec (layers,
  alpha, learning rate, batch, early stopping, `k/n features`, CV log-loss and
  balanced accuracy against the baseline); the model hint and strip show the
  layers, feature count and CV balanced accuracy. **Train** on the
  `dense (tuned)` kind rebuilds that spec on the current labels without
  re-searching (before any search it builds the baseline). The spec rides the
  classifier pickle (v3, `spec`; v1/v2 pickles still load) and the session's
  model record, so a loaded tuned model is described and retrainable.
