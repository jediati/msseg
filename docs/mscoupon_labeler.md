# mscoupon labeler (`mscoupon-labeler`)

The labeler is the viewer (`mscoupon-gui`, see [mscoupon_gui.md](mscoupon_gui.md))
with an annotation panel. Both are bindings of the labeler framework
(`msseg.labeler`, see [labeler_framework.md](labeler_framework.md)):
`LabelerApp(AnnotationShell, MscouponApp)` adds only the coupon profile, the
persistence panel and the exports; the annotation model, tools and classifier
live in the framework. The user paints **classes** onto the living MSC regions
of a slice, trains a per-region classifier on the statistics table, and exports
training sets. This page covers the annotation model and the drawing tools;
the engine, session file and config schema are the viewer's.

```bash
mscoupon-labeler                  # GUI
mscoupon-labeler --selftest       # Tk wiring check, no engine needed
# from a source checkout (the three namespace packages on the path):
PYTHONPATH="packages/mslabeler/src;packages/mscoupon/src;packages/msseg-viz/src" python -c "from msseg.mscoupon.labeler import main; import sys; sys.argv=['x','--selftest']; main()"
pytest packages/mslabeler/tests packages/mscoupon/tests      # conftest.py puts the source trees on sys.path
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
"Colour input"). The **middle** column is the slice itself -- always on screen,
whatever you are editing -- and the **right** column is a notebook of the
things you edit.

| column / tab | holds |
|---|---|
| **left** | the workflow (compute-profile) picker, the **Tasks** list, the session browser (folders, files, sequences) and Run |
| **middle** | the slice canvas, hover readout, slice navigation, image/overlay/alpha controls and the persistence entry |
| **Processing** (default tab) | the profile tools (New/Dup/Rename/Delete, Save/Load profile) and the profile being edited, as one column of collapsible groups: colour input, filter chain, base channel, MSC parameters |
| **Features** | what every region is measured by: the base channel (the base chain, e.g. a normalize) and the statistics -- channels, reductions, seeding extremum and its sample radius, histograms -- i.e. the classifier's input columns (see "Features: an edit re-measures" below) |
| **Annotation** | classes, tools, the Magic rows, the per-class interaction lists, Save/Load annotations, and the classifier (Train/Classify, the model strip, the confusion matrix, exports, Save/Load classifier) |
| **Model** | the classifier kind, a read-only description of its architecture, the **Edge model** panel (the `-> edges` kinds) and the **Optimize network** search (trials, time limit, seed, feature-subset toggle, progress line) |
| **Analysis** | **Predictions vs annotations** (the regions behind a confusion cell; double-click a row to go there) and the **Size sweep** report; a home for plots later |

The three columns start at about **1:3:2**, and **only the middle one grows**:
widen the window and every new pixel goes to the picture. Drag either sash to
change the proportions -- they are remembered with the session as fractions of
the width, so they restore sanely into a differently sized window -- and press
**F9** to fold the tab column away entirely, F9 again to bring it back.

Each Processing (and Features) group folds away behind its title, and which
ones are shut is remembered too. That is what makes one column work: the filter chain alone
grows a card per stage, so all five groups open is about 850 px of panel and
all five shut is 135 px.

The canvas used to be a tab of its own, beside Processing. A filter chain is
judged by *looking* at what it produces, and that made the judging a round trip
-- edit, switch, squint, switch back, and no way to tell whether what was on
screen reflected the edit. So the picture is never the thing that is hidden.

## Tasks

A session holds several **tasks** -- named detectors such as *gland detector*
or *bubble & background* -- over the same folders and sequences, and exactly
one is active. A task owns what a detector is made of: its **workflow** (a
compute profile, referenced by name from the session's shared pool), its
**class vocabulary** (count, colours and names -- double-click a class title
to name it *gland* / *not gland*), its **annotations** with their undo
history, its **model stack** (the region classifier, the edge model, the
latent head, the seam model) with its predictions, its saved-model records,
and its Model-tab settings (kind, Optimize, neighbours, context). *Not gland*
is a statement about one detector's decision boundary, which is why the
gestures are per task rather than one ten-class vocabulary for everything.

The **Tasks** list sits above the session lists in the left column: name,
workflow, gesture count, model. Click a row to switch -- the workflow box
follows (it shows and rebinds the ACTIVE task's workflow; two tasks on one
workflow share its primed data, a different workflow drops it, as a profile
switch always has) and the class stack, the Model tab and the predictions are
the task's. **New** starts an empty task on the active workflow with the
current class count; **Dup** copies the vocabulary, workflow and Model-tab
settings and nothing else; **Rename…** and **Delete** (refused for the last
task; asks when the task has gestures or a model) do what they say. A switch
is refused while an Optimize / sweep / evaluate search runs, because its
result installs into the active task. A task's saved pickle loads when the
task is first activated, and its Optimize autosaves go to
`models/<task uid>/`.

The session document is **v3**: `tasks[]` + `active_task`, each task with its
`annotations`, `models` and `view`. A session written before tasks (v2)
restores as one task named after its active profile, with everything it had;
**New session…** keeps the tasks and empties their stores (their models too,
when the model option is kept). Renaming or deleting a profile follows
through to the tasks that point at it. Design and the stages that follow
(shared places, slide-bound gestures, cross-task masks and borrowed gestures):
[design_multi_model_tasks.md](design_multi_model_tasks.md).

### Editing a filter chain

Changing any parameter of either chain repaints the image about a quarter of a
second after you stop typing, with **no Run**: the chain is applied to the raw
slice on a worker thread (the window stays responsive even through a ~5 s GMM
normalize), the spinner says `Previewing <channel>`, and zoom and pan do not
move. Set **Image** to `filtered` to watch the topology field, `base` to watch
the base channel, or any derived statistics channel.

Only what is on screen is recomputed. Editing the topology chain while the
Image dropdown shows `base` costs nothing, and retyping a sigma you had before
comes back instantly from the cache. Derived statistics channels never depend
on the topology chain at all (a statistics source is the base raster or the
colour planes), so editing `filters` while one is shown is free too.

This works **after** a Run as well: the canvas switches to the live chain, the
region overlays come off -- they came from a different field, so drawing their
boundaries over this picture would be worse than drawing nothing -- and the
badge reads `Preview - filters changed, Run to re-prime`. Undo the edit and the
primed view comes straight back from the cache; Run and the overlays are about
the picture again.

### Features: an edit re-measures, it never re-primes

A prime is two computations: the **field** (the chains, the colour input, the
MSC, the base manifolds) and the **measurement** (the per-region rows). They
are keyed and cached apart. Editing anything on the Features tab settles for a
quarter second and then re-measures the slice on screen -- the canvas says
`Measuring` -- keeping its MSC, its regions and its arcs, and rebuilding only
the rows; other slices re-measure when something reads them (a Rerun, a Train,
navigating there). No Run. The same holds for a profile or task switch whose
profile differs only in its statistics (or selection): the primed stack is
kept and re-measured, where a different field still drops it, and for
**Create a profile from the model's statistics** after loading a classifier.
A Run reuses every sequence primed under the same field, re-measuring the ones
whose statistics moved. What a re-measure costs is a prime minus its MSC: on a
2048² item, 0.2 s for base-only and 0.9 s for twelve derived channels (the
channel bank is most of that), against the MSC's own 0.5-4 s. Predictions
still fall stale -- the rows changed -- and a model trained on other columns
is refused by the compat gate as before. The base chain (`base_filters`) is a
measurement too -- the MSC never reads it -- so it lives on the Features tab
and an edit re-measures: the slice file is re-read and the base raster
rebuilt, then the rows.

**What a switch costs.** A record's commit is the identity of the parameters
it was made under, and a few records per slice are kept
(`MSSEG_RECORDS_PER_ITEM`, default 4). Switching to a task on the same field
and back therefore finds the first task's records -- and its predictions --
again, with no re-measure. A task on another field still needs a Run in the
coupon labeler: one primed stack is kept at a time.

Three link-labels say what is in effect: the Run section is headed by the
**selected workflow** as two compact chains --
`topo field: base→b(1.5)→e(0.7)→msc(asc, 10%)` (the field the MSC runs on,
then manifold and persistence, `mf` for merge forest) and
`stats: base→norm(gmm)→12ch×4` (the base chain the statistics are measured
on, then channels × reductions); hover for the code table -- and the classifier
section, above Train/Classify, by the `stats:` line alone (opens **Features**)
and the **active model** (the kind Train will build, or the trained model and
its feature count; opens **Model**); clicking any of them opens its tab. The selected tab rides the session as
`view.center_tab` (with the sashes as `view.panes` and the folded Processing
groups as `view.proc_open`) and is restored by name, and so does the picked
model kind
(`view.model_kind`: a plain Train writes no pickle, so without it a restore
would land on the kind of the newest saved model). **New session…** (toolbar)
starts over with no data and no annotations; its dialog keeps the compute
profiles and the model selection (kind, the model in memory, edge and search
settings) unless unticked. The hotkeys are window-wide, so `Tab` still toggles
the overlay and `F9` still folds the tab column from any tab. The viewer
(`mscoupon-gui`) keeps its two-pane layout, with its parameter sections in the
left column as plain titled boxes -- and the same live filter preview.

## Annotations are gestures, not region ids

A gesture is keyed by the **slide** it was drawn on -- for a coupon the slice
file, which is its own slide; for `mspath` the slide id, never the ROI or the
overview it happened to be drawn on -- because it is a statement about tissue
at a location, and every item covering that location, at any level, should
see it (design note §8). In mspath that means a stroke on the overview is
there when you cut an ROI under it, a stroke in an ROI stays on the overview
when the ROI is removed and is back when the rect is re-cut, and the same
gestures serve a level study. Each new gesture records the level and zoom it
was drawn at (`meta.level` / `meta.scale` / `meta.px`); on its own level that
metadata is never read, off it a stroke keeps the width the user saw, a magic
fill / blob / accepted prediction resolves by its **outline** rather than by
seeds (which would land in different regions), and a trace resolves by a
corridor. An item showing gestures drawn two or more levels coarser gets a
`!` on its annot cell and one notice: a warning, not a refusal -- whether
*gland* at level 4 means *gland including lumen* at level 0 is the task's
call. Older sessions (gestures keyed by item) rebase on load.

An annotation (`labeling.Interaction`) is one gesture in **image coordinates**
bound to one item key and one class -- for the coupon labeler the key is the
slice's `"folder/basename"` (`adapters.SequenceCatalogue.key_of`). Nothing stores a region
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
| lasso | drag a closed polygon | every region under the filled polygon -- an **extent** (below); *Ctrl*-drag for a sample |
| magic | press, drag up/down, release | a similarity flood from the pressed region -- an **extent** while the `extent` box in the Magic rows is ticked |
| blobber | as magic | the flood in the active class **and** its bounding regions in the ring class; the core is always an **extent**, the ring a sample |
| *SHIFT + drag* | a box, any tool | **accepts** the classifier's predictions under it as `taps` (never an extent) |
| trace | click to anchor, move, click to extend, Enter / double-click | a livewire path along the **seams** (the boundaries between regions), labelled `boundary` -- see [seam_labeling.md](seam_labeling.md). Clicking the **first anchor again closes** the loop and commits it; a tap inside with a class armed then names the **enclosure** |
| scope | drag a rectangle | every seam fully inside it labelled `interior` (unless a trace says boundary) |

**Samples and extents.** A squiggle, a box or a tap is a *sample*: "these
regions are class k", nothing about their neighbours. An *extent* -- a
lasso, a released magic fill, a blob's core, an enclosure -- says "this set
IS the object": its outer seams are boundaries and its unlabelled neighbours
are not it. Nothing else changes about how they paint regions; the
difference is what the **seam and edge models learn** from them
(`derive.py`, [seam_labeling.md](seam_labeling.md) "Derived labels"): a
fill on a gland teaches the seam model the gland's edge with no trace drawn.
An extent only speaks about *unlabelled* neighbours, so filling twice on the
same gland, or patching a fill with a squiggle, never derives a boundary
inside it -- an instance boundary between two touching same-class objects is
a trace's job. Extents show ` ext` in the class panel's rows.

Hotkeys: `1..4` arm a class, `0`/Escape disarm, `M` selects magic, `B` the
blobber, `T` the trace and `S` the scope (seam tools; `E` toggles the seam
layer, Enter commits a trace, BackSpace drops its last leg), `Tab`
toggles the overlays, `Ctrl-Z`/`Ctrl-Y` undo/redo, `R` train + classify,
`C` classify, `F` flips the image between the original (base, or the colour
planes of an RGB slice) and the derived channel last shown (filtered until
one is picked). Middle/right drag always pans; a right-click opens the
annotation menu for the region under it.

Each image channel keeps its **own brightness window**: the Min/Max sliders
show and edit the channel on screen, a channel seen for the first time opens
at its raster's 1st/99th percentiles, and a window stays as set once moved.
The windows ride the session (`view.windows`).

### Removing data and annotations

A **right-click on a row of the sequence tree** (a sequence or one of its
slices; in `mspath-labeler` a slide, its overview or an ROI) offers *Go to*,
*Clear annotations… (n)* and *Remove …*. Both destructive entries ask first.
*Remove* takes the row out of the session together with what was computed for
it (the primed rasters, records and 3D assembly of a coupon slice; the primed
item of an ROI, or every item and the open pyramid of a slide) and with the
annotations on it -- except a slice that another sequence still holds, whose
key is the slice, so its annotations stay. The annotations come back with
**Ctrl-Z**, greyed until the data is added again; the data itself does not.
The *Remove* and *Clear all* buttons under the sequence list and the ROI
section's *Remove* go through the same path. Each **class panel's `clear`
button** deletes every annotation of that class on every slice, after asking,
as one undo step.

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
Release paints; Escape abandons. The **`extent`** box (second Magic row, on
by default, remembered with the session) makes the released set an extent
-- untick it to release a partial fill as a sample.

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
(`|d|`, `product`, the saddle `barrier`: `saddle - max(ext_a, ext_b)` and
`|ext_a - ext_b|`, zeros on pixel adjacency; and, off by default, `contact`:
`log(1 + shared boundary length)` measured on the label raster, so that with
`barrier` unticked the pair model reads no saddle value at all -- embeddings
plus geometry), `C`, and the voting `lambda` and
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
shows that slice, centred on the region's seeding extremum (zooming in to 1:1
if further out), with the gestures touching it outlined and the cell's
highlight still on. The tab does not change: the canvas is above the notebook,
so the list stays open beside the region it just sent you to.

## Classifier and exports

**The fast path is "retrain, classify what I'm looking at"** (2026-09-24):
`R` trains and classifies the item on screen, `C` classifies it, and
**Classify all** (the button) does every computed item. Model operations
never compute an item that is not computed yet -- in the whole-slide labeler
an ROI nobody has selected is skipped (and counted in the status line), and
an item is classified when it is selected. In the whole-slide labeler the
items are the ones the active **task** works (its enrolled places, greyed
rows are browsable only), and the Run section offers **Run task** and **Run
all tasks** (every task on the active workflow) -- see
[design_multi_model_tasks.md](design_multi_model_tasks.md) §5.

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

### Context features (the neighbourhood, as columns)

The row describes a region's own pixels, and sometimes they do not say
enough: a shallow dip inside metal and a real void can share a `mean_base`.
The **Context** panel on the Model tab appends columns computed from the
living-region graph (`msseg.labeler.context`), so the net sees the
neighbourhood *before* it decides -- unlike the edge model, which can only
reshuffle probabilities afterwards. Each checked **ring** kind adds one
column per source column:

| kind | column | what it is |
|---|---|---|
| `ring_mean` | `ring_mean__<col>` | weighted mean over the regions this one touches (its arc neighbours) |
| `ring_contrast` | `ring_contrast__<col>` | own value minus the ring mean |
| `ring_min` / `ring_max` | `ring_min__<col>` … | extremes over the ring |
| `ring_std` | `ring_std__<col>` | weighted spread over the ring |
| `hop2_mean` | `hop2_mean__<col>` | the weighted mean applied twice (the neighbours' neighbourhoods) |
| `slice_mean` / `slice_contrast` | `slice_mean__<col>` … | the item-wide mean, and own value minus it |

**weights** says how much each neighbour counts: `uniform` (once each; needs
only the arcs), `area` (by pixel count) or `contact` (by the shared boundary
length, measured on the label raster once per slice and cached on the arcs
as an optional `length`). More than one weighting gives a column set each
(`ring_mean[contact]__mean_base`). **source** picks which columns the
context is built over: every non-positional column, only `ext_*` (the
seeding extremum -- what the region looks like away from its boundary) or
only `mean_*`. A region with no neighbours keeps its own value (contrast 0).
No saddle value enters any of this: the saddle lives on the topology field
and only reports what the reduction to a scalar already made large.

The columns are ordinary columns from there on. Train fits over them, the
Optimize feature mask files each kind (and weighting) as its own group
(`use_ring_mean`, …) so the search can switch a rung off, the compatibility
gate expects them, and the pickle (`stack.context`), the session's model
record (`context`) and the session view (`context`, the *picked* spec for
the next Train) carry the spec. The model strip and the Model tab readout
show the spec in force (`ctx: ring(mean,contrast) [contact] on ext`) and
flag "context changed - Train to apply" when the panel differs from the
loaded model's. Predictions rebuild exactly the loaded model's columns, so
changing the panel never touches a classified slice. An empty spec is
byte-identical to before: same fingerprint, same pickle document.

**Latent ring head.** The raw rungs aggregate pixel statistics; the
feature-row -> 16 -> 8 net distils *what kind of material* a region is, and
that is the thing worth averaging over a ring. With **latent ring head**
on, Train fits the base net as usual, embeds every region with its hidden
layer (`layer`: last = the narrow one), averages the ring's embeddings
(`weight`: uniform / area / contact, or `latent` = a softmax over the
latent distance scaled by the slice's median arc distance, so neighbours
of the region's own kind count more) and, with **H0 shape** on, adds four
columns from the single-linkage filtration of the region plus its ring in
latent space: the largest merge, its ratio to the second, the region's own
attach distance (an outlier against its ring reads high) and the component
count at mean + z·std of the slice's arc distances (a ring with two kinds
of neighbour reads 2). Rows are bucketed by degree and each bucket runs one
vectorised Prim, so a slice costs milliseconds. A second net of the base's
own architecture (the tuned spec when there is one, its feature mask
widened by the new columns) is then fit on the row plus those columns with
balanced weights, and *its* probabilities are the prediction; the edge
model keeps embedding with the base. These columns come from the fitted
model, not the profile, so they are **not** in the fingerprint: the head
rides the pickle as `stack.latent` and is dropped on load when the base is
not the one it was fit on. A forest cannot host it (no embedding); the
status line says so.

**Labels as context.** With **labels as context** on, the row gains one
column per class -- the fraction of the ring annotated with it -- plus the
fraction annotated at all (`nbr_class__c1`, …, `nbr_class__any`, under
the same weightings as the ring kinds). A region's own label is never in
its ring. At training every labeled neighbour is hidden with probability
**dropout** (a seeded draw per slice), so the net learns to work with a
partly labeled ring; at prediction it sees everything drawn so far. That
makes the loop transductive: five taps re-predict the slice. Predictions
therefore depend on the store, and the labeler drops the cached ones the
moment an annotation changes (the overlay empties until Classify). A
held-out score with the slice's own labels visible is optimistic; the
ablation harness reports both "labels visible" and "labels hidden".

`packages/mscoupon/experiments/context_ablation.py` scores every kind,
weighting and source on the autosaved session by leave-slices-out log-loss
(and, with `--search N`, lets Optimize's mask pick).

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
  three or more slices carry labels (`StratifiedGroupKFold` by slice, the group
  `SequenceCatalogue.group_of` assigns -- the same grouping the edge model's
  evaluation uses; before 2026-09-09 the region model grouped by sequence;
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

## Profiling a slow pan (Ctrl+P)

The `[mscoupon] canvas.render …` line measures only the numpy/PIL composite --
the part of a frame that happens *inside* `SliceCanvas.render`. That is not
what a drag costs. Press **Ctrl+P** (or start the app with
`MSSEG_CANVAS_PROFILE=1`) to turn on the frame profiler
(`msseg/labeler/perf.py`), which accounts for the whole pan tick:

```
[perf] canvas #26 out=900x640 crop=246x175 scale=0.2732 src=in-memory items=1 lvl=0
[perf]   frame 73.8 ms: base.read 0.0 | base.window 0.2 | base.resize 2.9 | base.rgb 1.9
                      | ov.gather 6.4 (2x) | ov.lut 14.5 (2x) | ov.blend 38.4 (2x)
                      | photo 4.1 | canvas 3.5 | paint 1.3
[perf]   between frames 11.7 ms: view_cb 6.1 (106x) | annot.hover 4.3 (106x) | ...
[perf]   events=106 coalesced=105 latency(evt->pixels)=932 ms gap=933 ms (1.1 fps)
```

Read it in three parts:

* **`frame`** -- segments of one `render()`, in order. `base.*` is the source
  read + window + resample (a `src=pyramid` frame pays a file read and a PNG
  decode *per frame* in `base.read`; a zoomed-OUT frame pays `base.window` over
  the whole crop, which can be 12 Mpx). `ov.gather`/`ov.crop` fetches each
  region layer over the viewport, `ov.lut` looks up its colours, `ov.blend`
  composites -- all three are canvas-sized whatever the zoom, so they do *not*
  shrink when you zoom in. `photo` is `ImageTk.PhotoImage`, `canvas` the item
  bookkeeping, and `paint` the Tk window redraw, which `create_image` only
  queues and which profiling forces so it can be timed.
* **`between frames`** -- work paid *outside* any frame, with how many times it
  was paid. `view_cb` is `on_view_changed`, spent in `annot.persist`/
  `annot.hover` re-projecting annotation outlines; `tool.move` is a drawing
  tool's per-tick work and `hover` the value readout. These run once per
  **motion event**, and motion events outnumber painted frames.
* **the tail** -- `events` motion events since the last frame, `coalesced`
  repaints cancelled by the debounce before they ran, `deadline` repaints let
  through by the anti-starvation deadline, `latency(evt->pixels)` from the
  oldest unserved repaint request to the finished paint, `gap` between painted
  frames (the honest fps), and `annot_items` / `items`, how many Tk canvas
  items the paint has to walk.

Releasing the button prints a per-gesture summary (duration, frames, fps,
events, coalesced repaints, worst frame, and the per-frame average of every
segment), so a whole drag is one line to compare against another zoom level.

### What it found, and the two fixes that came out of it

The trace above is a real 2 fps drag at scale 0.27, and it says the paint was
never the problem (1.3 ms) and neither were the annotations (`annot.persist`
0.0). Two things were:

* **The repaint was starving on its own debounce.** `_schedule` re-armed a
  15 ms timer on every motion event, and events arrive every ~9 ms while the
  mouse moves, so the timer never expired: 179 events, 174 repaints cancelled,
  **5 frames in 1.85 s**, with 80 % of the drag spent rendering nothing. A
  repaint may now be coalesced but never postponed more than
  `SliceCanvas._MAX_DEFER_MS` (30 ms) past the first request that asked for it;
  past the deadline the armed job is left to fire instead of being re-armed.
* **The overlay composite was five full-size float temporaries per layer.**
  `a = ov[:,:,3:4]/255 * alpha; rgb = rgb*(1-a) + ov[:,:,:3]*a` allocated
  ~25 MB per overlay per frame and read the RGBA block twice with a stride.
  Every step of it depends only on the region id, so it now folds into the LUT
  (`_blend_luts`: premultiplied colour + `1-a`, both 3 wide so the gathers are
  contiguous and the multiply does not broadcast), leaving two gathers and an
  in-place multiply-add. It is exact -- the same float32 ops on the same
  values, once per LUT row instead of once per pixel -- so the composite stays
  bit-identical to the reference in `tests/test_sources.py`.

Together, at that zoom: the frame went 76 ms -> 40 ms and the drag 2.7 fps ->
~13 fps. Still open (it shows up zoomed *out*, not in): `base.window` clips the
full 12 Mpx crop before the resample throws most of it away.
