# Design note: many task-specific models over one set of slides

Status: **third draft; stages 1-5 built** (2026-09-24; first draft
2026-09-18). Stage 4 -- places, enrolment, a split Run and the on-demand fast
path -- is described where §5 and §12 now say "as built". Stage 5 landed in
two halves: the measurement split (the re-measure question, §6 / §13.1,
answered yes) and the layered cache keys (§6 "as built": a record's commit is
its identity, pipes live per field, the base chain is a measurement).
Stage 1 of §12 -- the task object with a stack, the Tasks list, session v3 --
stage 2 -- slide-bound gestures with a scale of intent, stroke width, extent
outlines and the trace corridor (§8) -- and stage 3 -- the derivation module
(§7.2), the extent flag and the enclosure gesture -- are implemented
(`msseg.labeler.task`, `extents`, `derive`, `AnnotationShell`; see "What
exists" at the end of §12). Stages 4-8 remain proposals. The second draft was a
critique-and-rewrite of the first (§1); the third adds §8, which reverses one
of the first draft's "load-bearing" claims -- that a gesture should bind to an
item -- after the observation that a squiggle saying *inside gland* is a
statement about tissue, not about the resolution it was drawn at.

Scope: `mspath-labeler` first, but every entity lives in the framework
(`msseg.labeler`), so `mscoupon-labeler` inherits it. Where a decision only
makes sense for slides, it is marked.

## 1. Weaknesses of the earlier drafts

From the first draft, fixed in the second:

1. **It modelled one kind of task.** The pickle already carries a *stack* --
   base region net, edge model over arcs, latent ring head, seam model over
   seams -- and the store already holds two *kinds* of gesture. A path-based
   all-tissue detector and a region-based gland detector score different
   **targets**. §4, §7.
2. **"Seam gestures -- they are annotations" was a category error.** Whether a
   shortest-path annotation is an input to a region model has a definite
   answer (open trace: no as a label, yes as a constraint; closed trace: yes,
   as an extent). §7.
3. **Borrowing assumed same-kind transfer.** The taps -> exterior-boundary
   conversion is the cross-kind case. §7.2.
4. **One workflow fingerprint bundled persistence in**, so two tasks differing
   only in persistence would have paid two primes. §6.
5. **It undersold the common case**: stroma, gland and an all-tissue path
   detector on one primed slice share every expensive layer. §9.
6. **Masks were deferred as a hazard** rather than designed. §8 -> now §9.
7. **Cross-task stacking was not addressed** (the seam model's `"net"` embed
   already depends on a base net). §9.
8. **The composite export ignored seams.** §11.

From the second draft, fixed in the third:

9. **Gestures bound to items.** Both drafts carried "the item key carries the
   level" over from `items.py` as a property that must survive, and applied it
   to gestures as well as to decompositions. It is right for records and
   models -- coarse and fine decompositions are not nested, and a model is
   valid at one level -- and wrong for gestures, whose geometry mspath
   *already* stores in slide coordinates. Binding them to an item made the
   same tissue invisible to an overlapping ROI, orphaned every gesture when
   an ROI was redrawn, and ruled out the level study that a task-level
   vocabulary makes natural. §8.

What all drafts keep: annotations are geometry and re-resolve (§3.1); places
belong to the slide and enrolment to the task (§5); the compat gate already
knows a model is workflow-bound (§3.3); a v2 session reads as one task (§12).

## 2. The ask

One session, one corpus of slides, and a **named set of detectors** -- "gland
detector", "stroma detector", "bubble & background", "all-tissue boundary" --
each with its own vocabulary, gestures, processing stack and estimators,
instead of one ten-class model. Detectors must run on the *same* primed
slice; one detector's output must be usable as a mask by another; a gesture
made for one detector should be reusable by another where meaningful; and
the same gestures should support asking *which resolution learns this task*.

The reason this is the right shape is in the data: *not gland* is a statement
about a decision boundary, not about tissue. One vocabulary forces a single
partition onto what is naturally a stack of independent, overlapping calls,
and forces every detector onto one persistence, one level and one statistics
bank.

## 3. Baseline (precisely)

Read from `session_doc.py`, `labeling.py`, `annotate.py`, `classifier.py`,
`bundle.py`, `edge_model.py`, `seam_model.py`, `seam_labeling.py`,
`mspath/items.py`, `mspath/labeler.py`, `mspath/engine.py`.

| Entity | Where | Cardinality | Bound to |
|---|---|---|---|
| folder / slide | `session.folders[]`, `session.sequences[]` (mspath: one slide each, `rois[]` on it) | many | -- |
| ROI | `sequence["rois"] = [{level,x,y,w,h}]` | many per slide | slide **and level** |
| item | `mspath.items.Item`; key `slides/a.svs@4`, `...@0#x,y,w,h` | derived | slide x level x rect |
| profile | `session.profiles[]` + `active_profile` | many, **one active** | -- |
| region gestures | `LabelStore.interactions` (squiggle / box / polygon / taps); points in **slide coordinates** (mspath) via `Placement(origin, scale)` | one store | `slice_key` = **item key** |
| seam gestures | `LabelStore.seams` (scope / trace), same uid space | same store | item key |
| class vocabulary | `LabelStore.n_classes` (2..5) + `colors`; no names | **one** | -- |
| region model | `_clf` (+ names / kind / spec / scope / context) | **one** | active profile, via the gate |
| edge model | `_edge_model`, stacked on `_clf` (`names_hash`) | 0..1 | the region model |
| latent head | `_context_model`, stacked on `_clf` | 0..1 | the region model |
| seam model | `_seam_model`; `embed="net"` stacked on `_clf` (`net_hash`), `"features"` free-standing | 0..1 | the region model, or nothing |
| model registry | `session.models[]` records | many | a path |
| prime | `SlideEngine._primed[item_key]`, LRU 3 | many | item key + *implicitly* the profile |
| record | `{commit, labels, stats, arcs, ...}` + lazily cached seams / contact lengths | per item per commit | prime + persistence |

Load-bearing properties that must survive:

1. **Annotations are geometry.** Region gestures rasterize against whatever
   labelling is current (`resolve_slice`: touched region ids per gesture, uid
   order); traces re-resolve against a new seam graph by **crack coverage**
   (`resolve_seams`, tau 0.5). A gesture is meaningful under a decomposition
   it was not drawn under -- which is what makes any cross-task and
   cross-level reuse possible.
2. **The item key carries the level, for decompositions.** "Same rect, other
   resolution" is a different *item*, with its own record and its own
   models; coarse and fine decompositions are not nested. This stays. What
   changes (§8) is that a *gesture* is no longer keyed by it.
3. **The gate already knows a model is workflow-bound.** Feature-name *set* +
   `scope` (`"L<level>"`) refuse a mismatch; stacked heads carry their
   upstream's hash and are dropped on mismatch. Missing is the *binding*, not
   the check.
4. **Arc labels are derived, not drawn.** `edge_model` builds same/different
   pairs from region labels on both flanks. There is no arc gesture. This is
   the template for every cross-kind derivation in §7.

## 4. Vocabulary

* **corpus** -- folders, slides, pyramids. Task-free.
* **frame** -- a slide's coordinate system: level-0 pixels from the level-0
  origin, plus the slide's physical scale (`mpp`) and, if ever needed, an
  affine to a canonical physical frame (§8.3). *New as a named concept.*
* **place** -- a rect in a slide's frame, or the whole slide. Task-free. *New:
  today's ROI record conflates it with a level.*
* **item** -- a place at a resolution: `slide@level[#rect]`. What *records and
  models* bind to. Unchanged.
* **workflow** -- today's *profile*, seen as three layers (§6): the **field**
  (chains, MSC mode, manifold, level), the **selection** (persistence) and
  the **measurement** (statistics channels, reductions, histograms, context).
* **target** -- what an estimator scores: a **region**, an **arc** (a region
  pair joined by a saddle) or a **seam** (a crack chain between two flanks).
* **gesture kind** -- **region gesture** (squiggle / box / polygon / taps) or
  **seam gesture** (scope / trace).
* **sample vs extent** -- a gesture's epistemic scope (§7.1). A *sample* says
  "these regions are class k". An *extent* says "this set IS the object".
* **scale of intent** -- the resolution a gesture was drawn at, recorded on
  it (§8.2). It is metadata, not a binding.
* **task** -- a named detector: a workflow reference, a class vocabulary, a
  gesture store, an enrolled place set, and one **stack** of heads.
* **layer** -- a typed per-item output a task publishes: region class
  probabilities, arc `pdiff`, seam boundaryness, an embedding, or a **mask**.
* **subscription** -- a task's declared use of another task's layer.
* **enrolment** -- the statement that a task works a given place.

Naming: the UI list can be titled *Models* or *Detectors*; the data-model
word is `task`, because "gland detector" outlives any trained pickle and
`models[]` already means "saved pickle records".

## 5. Places and enrolment

A place belongs to the slide; the work on it belongs to the task. **As
built (stage 4):** the ROI record grew into the place --
`{uid, level, x, y, w, h[, note][, origin]}` on the slide's list, `level`
being its default (cut) level -- and the task carries
`enrolled = {slide_id: {"overview": null, "<place uid>": level}}`: one entry
per place (the tree has one row per place), the **level on the enrolment**,
so two tasks can work one place at two levels (two item keys). `None` means
"every place at its own level" (the coupon, where a slice is its own place,
and nothing else once loaded); `{}` means nothing. The item a task works is
`place x its enrolled level` -- today's key grammar, unchanged. Drawing an
ROI from a view or proposing one creates the place (a rect already on the
slide is reused) and enrols it in the **active task only**, with
`origin = {task, reason, [score]}`. **The overview is never enrolled
automatically** -- not by adding a slide, a new task, a legacy load or a bulk
action: viewing the whole slide is *browsing* (no item current, nothing
primed or annotatable), working it is a choice (tree menu). A new task
enrols nothing. On the whole-slide view every place is outlined -- solid
with its level for the active task's, dashed grey for the rest. Removing a
place removes it for every task (the dialog names who worked it).

Rejected: task-owned ROIs (same rect becomes two objects; deleting a task
destroys a place someone else chose) and shared-and-auto-enrolled (every ROI
becomes an N-task prime commitment at 2-9 s and ~1 GB each).

Coupon: the item is a whole slice, so the place is the file; enrolment is
"which slices this task works".

## 6. The compute DAG, and what is shared at which layer

A prime is a chain of artifacts, each invalidated by a different part of the
workflow and each with its own cost (numbers from CLAUDE.md and the memory
notes; 3232² coupon slice / 4096² ROI).

```
L1  pixels          item read at level (+ halo)                 key: item, halo
L2  rasters         topology chain -> filtered (base chain: L4)   key: L1 + chains         ~0.5 s
L3  pipeline        MSC / merge forest, base manifolds,           key: L2 + msc mode,      ~4 s, ~1 GB   <- THE prime
                    native cancellation hierarchy                       manifold, accuracy
L4  measurement     base chain -> base; stats bank -> ChannelStats key: L3 + base chain    ~1 s
                    per base manifold                                   + stats spec
L5  record          select_persistence -> labels, living ids,     key: L3 + L4 + persistence   ms
                    arcs (livingRegionArcs), FeatureTable
L6  derived         seam graph, contact lengths, pixel adjacency, key: L5 (+ context spec) ms-100s ms
                    context columns
L7  inference       region proba, arc pdiff, seam boundaryness,   key: L5/L6 + model record hash
                    embeddings
L8  masks           region sets thresholded from L7 or L5 stats   key: L7 + rule
```

Field = L1-L3, measurement = L4, selection = L5. Two tasks share a link iff
their keys agree up to it:

* **Same field, different persistence**: one pipeline, two records,
  milliseconds. The *normal* way two tasks should differ.
* **Same field, different measurement**: **re-measured in place** (as built,
  2026-09-24). L4 was computed inside `Msc2DPipeline::build`, so a different
  stats spec was a rebuild; `build` is now the MSC phases followed by one
  `measure` step, and `Msc2DPipeline::remeasure` runs that step alone against
  the live base labelling, then re-selects at the current persistence. The
  labels and arcs are untouched and the rows equal a fresh build's. Measured
  on a 2048² level-0 ROI (CPU, merge forest): switching base-only -> twelve
  derived channels costs 0.90 s against a 1.47 s prime, and back costs 0.19 s
  against 0.91 s -- a re-measure is a prime minus its MSC, so the saving grows
  with the MSC (~4 s on a 3232² coupon slice, more on the MSC hierarchy).
* **Different field**: two pipelines, ~1 GB each, against an LRU of three.
  The genuinely expensive case; live side-by-side across fields is bounded by
  the LRU, and many tasks on many items across fields is batch.
* **Different level**: different items; nothing shared past L1. A level
  study (§8.4) is therefore one prime per level per place -- the cost is real
  and it is the point of the study.

**Cache keys are layered** (as built, 2026-09-24): live pipelines by
`(field, item)` -- mspath keeps one slot per field with its own pins, under
ONE live budget -- and records by `(item, prime, measurement, persistence)`
(mspath) / `(stack generation, parameter snapshot)` (coupon). The record key
names the PRIME, not a hash of the field: region ids are not stable from one
prime to the next, so a re-prime must never alias an older record. Derived
and inference artifacts needed no change at all: every cache already
compared `entry[0] == rec["commit"]`, so the commit became the record's
**identity** -- the interned id of its key (`msseg.labeler.record_keys`) --
and returning to earlier parameters finds the earlier records, predictions
included. A few records per item are kept (`MSSEG_RECORDS_PER_ITEM`, 4), and
records that differ only in their measurement share one label raster.
Switching tasks never drops a prime; it selects a different key.

**What a change costs** (the gesture columns reflect §8):

| Change | L3 pipeline | L5 record | region gestures | seam gestures | region model | arc / seam heads |
|---|---|---|---|---|---|---|
| persistence moved | kept | recomputed (ms) | re-resolve | re-resolve by coverage | kept | kept |
| statistics edited | **kept** (re-measured) | recomputed | re-resolve | re-resolve | **refused** (names) | dropped with it |
| chain / MSC mode edited | rebuilt | recomputed | re-resolve | re-resolve | refused | dropped |
| level changed | new item | new item | **kept**; applied with a scale warning; extents via outline | kept via corridor (§8.2); exact coverage at own level | refused (`scope`) | dropped |
| task switched (same field) | kept | found again if cached, else recomputed / re-measured | re-resolve | re-resolve | kept (its predictions valid again) | kept |
| task switched (other field) | kept in its own slot while the LRU allows; its records survive eviction | found again if cached | re-resolve | re-resolve | kept | kept |
| task renamed | -- | -- | -- | -- | -- | -- |

## 7. Targets: region-based and path-based, and what transfers

### 7.1 Sample vs extent

Every region gesture today means "the regions I touched are class k" and says
nothing about the regions I did *not* touch: a **sample**. A **boundary** is a
statement about the complement -- "the object ends here" -- and a sample
cannot yield one: the seams around three scribbled gland regions are where my
scribble stopped, not the gland's edge. Converting samples to boundary labels
would teach the seam model that every scribble edge is a class boundary.

So taps -> exterior boundary is valid **only for extent gestures**: a magic
fill accepted as complete, a blob (core + ring is already "this set and its
immediate complement"), a polygon drawn *around* an object, a closed trace.
Proposal: gestures carry an `extent` flag in `meta` (document form unchanged
for samples), set by the tools that produce closed sets -- blobber always,
magic fill and polygon by a modifier or toggle, SHIFT-accept taps never. An
extent additionally stores its **outline** (§8.2), because seeds do not
transfer across levels and an outline does. The flag is read by the
derivations below and by nothing else; resolution stays geometric.

### 7.2 The conversion matrix

Rows: what was drawn. Columns: the target a consumer wants labels for.
"Constraint" = a must-link / cannot-link relation between regions, usable by
the arc head and by voting, not by a per-row classifier.

| drawn \ wanted | region class | arc same/diff | seam boundary/interior |
|---|---|---|---|
| region **sample** (squiggle, box, taps) | itself | both flanks labelled -> same/diff (today's `edge_model`) | both flanks labelled -> boundary iff classes differ; **one flank labelled -> unknown**, never boundary |
| region **extent** (blob, closed polygon, accepted fill) | itself | as sample, plus inside/outside -> diff | inside/outside seam -> **boundary**; inside/inside -> interior. *This is taps -> exterior boundary.* |
| seam **scope** | nothing | every arc inside -> **same** | itself |
| seam **trace, open** | **nothing** -- it does not say which side is which | its arcs -> **diff** | itself |
| seam **trace, closed** | an *extent of unknown class*; one sample inside names it | inside/outside -> diff | itself |

Conclusions the matrix forces: **region extents are the richest primitive**
(lossless to every target -- make them easy to produce rather than enrich the
seam vocabulary); **an open trace is not a region label** and never will be,
but it *is* a label for a region task's arc head, which voting consumes;
**closed traces and scopes are extents of unknown class**, and "trace the
gland, tap it once" is worth a tool.

The matrix is implemented as pure **derivation functions** over `(gestures,
record)`: `region_labels`, `arc_labels`, `seam_labels`, each consuming both
gesture kinds. `resolve_slice`, `edge_model`'s pair gathering and
`resolve_seams` are three of the cells today; complete the table in one
module so no estimator reads gestures directly. Derived labels are **never
stored**: derivation runs at training time against the consumer's own record,
which is what lets one gesture serve two tasks, two persistences, or two
levels.

### 7.3 Region tasks and path tasks ~~are one task kind~~ -- answered: one kind each, joined by inputs

**Decided 2026-09-25** (built in six phases, `e462b31`..; the user-facing
account is [seam_labeling.md](seam_labeling.md)): the single-stack proposal
below was **not** taken. A task has a **kind**, `region` or `polyline`,
fixed at creation, and the kind decides what every tab, tool, hotkey, view
option and stage-strip box shows. What the proposal wanted from one stack --
a boundary detector that embeds with a gland net, derives boundaries from
gland extents, reads the gland edge model -- a polyline task gets by
**subscribing** to a region task through typed input slots (`artifacts.py`:
*region labels*, *region embedding*, *p(diff)*; §9's embedding and gestures
subscriptions, built), each slot a provider *reference* so a future model
artifact library fills them without a document change. Why two kinds:

* the review surface differs per target -- a class frame of seam gestures,
  a K x K matrix over seams counted by length, an error list that goes to a
  seam -- and one stack meant every panel asking "which head am I showing?"
  while the user is only ever training one of them;
* seam classes are **named per task** after all (class 1 = the "not a
  boundary" role, the rest are kinds of boundary: *gland wall*, *cut
  artefact*), which a shared store vocabulary cannot give a region head too;
* a polyline task's store vocabulary can then BE its seam vocabulary, so
  the whole class-panel machinery serves both kinds unchanged;
* a region pickle and a seam pickle are separate files (`ModelBundle` /
  `SeamBundle`), so neither gate has to reason about the other's head.

A region task keeps the livewire as an **outline** tool: a closed loop is
an extent of the armed class in one step (the "trace the gland, tap it once"
tool of §7.2, minus the tap); open traces and scopes exist only in polyline
tasks. The derivation matrix of §7.2 stands; its seam column is what a
polyline task's labels input computes against its own record.

The original proposal, for the record:

A task is a **stack** whose heads are chosen per target:

```jsonc
"heads": { "region": {"kind": "dense (tuned)", ...},          // optional
           "arc":    {"kind": "logistic", "on": "region"},     // optional, needs a region head
           "seam":   {"kind": "logistic", "embed": "net" | "features", ...} }   // optional
```

The gland detector: region + arc heads. The all-tissue boundary detector: a
seam head only, `embed="features"`, or `embed="net"` subscribed to another
task's region head (§9). Bubble & background: a region head whose positive
class is published as a mask. Region classes are named per task; seam classes
are the fixed pair `boundary` / `interior`. A seam-only task may still collect
region gestures (extents feed boundaries) under a single class *object*.
*(Superseded: see the decision above; seam classes are per task and a
polyline task collects no region gestures -- it reads a region task's.)*

## 8. The frame of a gesture: slide-bound, level-free

### 8.1 What is already true, and what the binding got wrong

mspath gestures are stored in **slide coordinates** -- level-0 pixels -- and
mapped to a raster index at resolution time by `Placement(origin, scale)`
(`mspath/labeler.py`: "what a gesture *stores* stays in slide coordinates").
The geometry is global. Only the **binding** is not: `Interaction.slice_key`
is the item key, `slide@level#rect`, so a gesture drawn on one ROI is
invisible to an overlapping ROI, to the overview beneath it, and to the same
rect worked at another level; redrawing an ROI's rect orphans everything on
it. `items.py` argued for the level in the key so that "labels drawn at one
resolution never silently reattach to another" -- correct for *labels* in the
sense of region ids and records, and mistakenly extended to gestures, which
are not region ids and reattach to a new decomposition by design (§3.1).

**Proposal: a gesture binds to a slide** (coupon: the slice, which is already
its own frame -- nothing changes there). `slice_key` becomes the slide id;
items **query** the store by rect (`for_rect(slide, rect)`, a bbox per
gesture, cheap). A user drawing *inside gland* and *outside gland* has said
something about tissue at a location; every item covering that location, at
any level, in any task that subscribes to those gestures, may use it.

### 8.2 The hazards, and what each needs

The binding was doing three jobs badly that now have to be done explicitly.

1. **A vocabulary has a scale.** *Gland* at level 4 includes the lumen; at
   level 0 the lumen is its own regions, and a coarse *inside gland* stroke
   applied fine labels lumen regions gland. Whether that is wrong depends on
   the task's vocabulary: for "gland including lumen" it is *right*, and it
   moves the difficulty to the model (a lumen-looking region surrounded by
   epithelium is gland -- which is what the ring / context columns are for).
   The design should not decide this for the user. Mitigation: record the
   **scale of intent** (`meta.level`, and the screen pixels per slide pixel
   at draw time) and warn -- a badge, not a refusal -- when a gesture is
   applied more than a configurable number of levels finer than it was drawn.
2. **A stroke has a width the user saw, not a width it stores.** A polyline
   is zero-width geometry; at level 0 a stroke drawn at level 4 touches a thin
   chain of tiny regions. As a *sample* that is fine (each touched region is
   genuinely under the stroke). But the user drew a ~3 px stroke at 1/16,
   i.e. a 48-pixel swath in slide units, and meant the swath. Mitigation:
   rasterize with a width equal to the draw-time width in slide pixels
   (from the recorded scale). At the drawing level this reproduces today's
   result; at finer levels it samples the swath rather than a hairline.
3. **Seeds do not transfer; outlines do.** A tap picks *the region under a
   point*, and a magic-fill extent is stored as one tap per region at its
   seeding extremum -- a representation chosen so the fill re-resolves under
   a persistence change on the *same field*. At another level those 40
   seeds pick 40 tiny regions and the extent dissolves. Mitigation: an
   **extent stores its outline polygon** in slide coordinates (a few hundred
   points for a 40-region blob), alongside its seeds; the outline is what
   crosses a level, the seeds are what re-resolve at the drawing level. This
   is the same requirement §7.1 arrived at from the boundary side.
4. **Crack coverage is per lattice.** A trace resolves by the ids of the
   cracks under it, and the level-0 lattice is sixteen times finer than the
   level-4 one; a level-0 seam meanders around a straight level-4 trace and
   tau 0.5 fails nearly everywhere. Mitigation: at a level other than the
   drawing level, resolve by **corridor** -- a seam is covered when >= tau of
   its cracks lie within `r` of the trace, `r` = one drawing-level pixel in
   slide units. At the drawing level exact crack ids remain, so today's
   results are byte-identical. Not built; a distance transform of the trace
   over the item's raster is enough.
5. **The same tissue twice.** The overview and an ROI both covering a gesture
   would both use it. A model is per level (`scope`), so a training set is
   per level, and only *overlapping ROIs at one level* duplicate rows --
   near-duplicates from two decompositions of the same pixels. Accept, or
   dedupe by place overlap; leave-slides-out CV (`SlideCatalogue.group_of`)
   already keeps both copies in the same fold.
6. **Bookkeeping.** Per-item annotation counts in the tree become rect
   queries; `_doomed_interactions` dooms gestures only when a *slide* goes
   (removing an ROI no longer deletes anything, which is the safer default);
   `rebind` binds by slide id and gets simpler.

### 8.3 Level-0 pixels or microns?

A slide's level-0 pixel grid is one physical frame among many, anchored at
its own (0, 0) with a scale the scanner chose. The question is which frame the
stored numbers are in.

**Level-0 pixels of the slide** (what exists):

* exact integers; what `items.py`, place rects, `Placement` and the pyramid
  already use, so nothing is converted twice;
* needs no metadata -- the coupon has no `mpp`, synthetic TIFFs often lack
  it, and a session mixing slides with and without it would silently mix
  units if microns were the storage form;
* a gesture only ever applies to its own slide, so a per-slide unit loses
  nothing *for gestures*.

**Microns** buy three things, and only the first concerns gestures:

* a **re-scan** of the same slide at a different `mpp` keeps the gestures --
  except that a re-scan is never pixel-aligned, so an affine (offset,
  rotation, scale) is needed regardless, and once there is an affine the
  unit of the source frame is incidental;
* **workflow parameters** that mean the same thing on a 0.25 and a 0.5 mpp
  scanner: a sigma in µm, a minimum area in µm², a ring radius. This is the
  real win, and it is a *workflow* concern (it belongs beside
  `design_filter_types.md`, as a `units` option on the parameters that are
  physically meaningful), not a gesture one;
* **derived features** comparable across slides: `area` at level 0 differs
  4x between those two scanners today, and a model trained on one is
  mis-scaled on the other in a way the gate cannot see. Also real; also a
  measurement concern (a `um` reduction option), not a gesture one.

**Recommendation: store gestures in level-0 pixels of their own slide, and
store the slide's frame once** -- `slide.frame = {mpp_x, mpp_y, affine?}` --
so microns are a derived view wherever they are wanted (HUD readouts, µm²
areas, physical sigmas). If a re-scan ever lands, a per-slide affine to a
canonical frame handles it, and the gestures stay integers on the slide they
were drawn on. Putting the physical unit where it pays -- parameters and
features -- keeps the gesture file exactly what it is today.

### 8.4 What this enables: the level study

With slide-bound gestures a level study is one task's vocabulary and store
against a **level sweep** -- the pattern `run_size_sweep` already has, with
the ladder over levels instead of hidden sizes: for each level, prime the
enrolled places at that level, derive labels from the *same* store against
that level's records, train the same stack, score leave-slides-out, and
report a table (level, rows, held-out log-loss, balanced accuracy, seconds
per prime). The `scope` gate still refuses applying one rung's model at
another rung, as it should; the study answers *which rung to work at*, and
a place found interesting at level 4 is already level-free, so the fine
ROIs it seeds cost nothing to declare.

## 9. Layers and subscriptions: masks, embeddings, borrowed gestures

Every task publishes typed **layers** per item; every inter-task dependency
is a **subscription** to a layer.

| subscription kind | source layer | consumer use | transfer across workflows / levels |
|---|---|---|---|
| **mask** | a task's region-class layer thresholded (`class in {..}`, `p >= t`) or a stats rule | drop rows outside the mask from training and prediction; dim them in display; exclude from proposals; optionally a `mask_frac__<task>` column | pixel-rasterize the source's region set at the source's record, then per-region **coverage fraction** on the consumer's record, threshold 0.5 -- gesture resolution's own machinery, and it crosses levels the same way |
| **context** | a task's region-class layer | ring / hop columns in the `nbr_class__*` style, from *predictions* | as mask, per class |
| **embedding** | a task's region head | the seam / arc head embeds with it (`embed="net"`) | requires the **same record**: an embedding is a function of the row, and rows are not comparable across decompositions -- refused otherwise, as `scope` refuses a level |
| **gestures** | a task's store, filtered by class and kind | derived through §7.2 as if drawn here, read-only, outside this task's undo | free -- gestures are geometry (§8) |

Rules: subscriptions form a **DAG** (checked on edit); resolution is **pull,
never push** (a source retraining reaches into no one); the consumer's model
record stores, per subscription, the source model record hash and the source
store `rev` it resolved against, so the UI can say *mask stale: bubble &
background retrained since* and a retrain is always explainable; masks are
**region sets with provenance**, pixel rasters only at the moment of
transfer; a subscription is part of the fingerprint when it changes the
training set or the row (gestures, context, mask-as-column), not when it only
restricts display.

## 10. Worked example: three detectors on one primed slice

```
workflow "H&E L0"      field: color(OD) -> blur 1.5 -> edges(dizenzo);  merge_forest, asc
                       measurement: base + blur/edges/hessian x 3 sigmas, 4 reductions
task bubbles+bg        H&E L0 @ 8 %;  region head; publishes mask = class "tissue"
task gland             H&E L0 @ 3 %;  region + arc heads; subscribes mask(bubbles+bg)
task stroma            H&E L0 @ 3 %;  region head; subscribes mask(bubbles+bg),
                                                   gestures(gland, class gland, as "not stroma")
task tissue boundary   H&E L0 @ 3 %;  seam head, embed=net <- gland.region;
                                      gestures(gland, extents only) -> boundaries
```

Computed once: L1-L4 (one read, one pipeline, one bank). Two records (8 % and
3 %), the second serving three tasks. One seam graph. The mask transfers
8 % -> 3 % by coverage once per (source model, consumer record). The boundary
task embeds with the gland net, legal because they share the 3 % record. Four
inference layers; display is four stacked layers over one raster.

Add a level study: `gland @ L2` is a sibling task (or a sweep rung) with the
same store and the same places; it costs one more prime per place and shares
nothing past L1 with `gland @ L0` -- which is the honest cost of asking the
question.

## 11. Displaying and exporting several tasks

N detectors do not produce a segmentation. Each publishes layers over its own
record; two may claim a region, or neither; a seam head publishes a boundary
set, not a region set. Live rendering is a **stack of layers** -- region
fills per task, boundary polylines per seam task, masks as dimming. A single
label image is *constructed* by an explicit **composite rule** on the
session -- an ordered `(task, class)` list, first match wins, plus background
-- applied to region layers only; boundary layers export as polylines
(`seams_<item>.json` exists). The ten-class model returns here, as a
reviewable export rule, not as the training problem.

## 12. Migration and staging

Session document v3; reading stays total. A v2 document reads as **one task**
named after the active profile, with whatever heads the pickle carries,
enrolled in every item that has a gesture, the registry as its history, no
subscriptions. Gesture rows keep their form; on read, an item-keyed
`slice_key` is rewritten to its slide id and the item's level recorded as the
gesture's scale of intent, so the byte-level baselines hold for coupon
stores (slice id == item id) and mspath stores upgrade in place, the way v1
basenames did.

Stages:

0. **No schema change.** Loading a model offers the profile built from its
   statistics as the normal path (`_profile_from_model` exists).
1. **The task object with a stack.** `tasks[]`, one active, owning workflow
   ref + vocabulary + store + heads. Left-panel task list. **The requested
   feature.**
2. **Slide-bound gestures** (§8.1-8.2): slide id as `slice_key`, `for_rect`,
   scale of intent in `meta`, width-aware rasterization, extents with
   outlines, corridor resolution for traces off their level. Independently
   valuable: overlapping ROIs and the overview start sharing gestures.
3. **Derivation module** (§7.2) and the closed-trace-plus-tap tool.
4. **Places and enrolment** (§5).
5. **Layered cache keys** (§6); the re-measure question answered here.
6. **Level sweep** (§8.4) -- cheap once 2 and 5 exist.
7. **Layers and subscriptions** (§9): masks and gestures first, then context,
   then embeddings.
8. **Multi-task overlay and composite export** (§11).

**What exists (stage 1, 2026-09-21).** `msseg/labeler/task.py` (`Task`,
`ModelStack`, `TaskCaches`, `TASK_VIEW_KEYS`, `new_uid`); class names on
`LabelStore` (`names` / `set_name`, written only when set); session document
v3 (`tasks[]` + `active_task`, tasks-less form unchanged, a v2 document
reads as one task); `AnnotationShell` keeps every downstream attribute name
as a property over the active task, with `_activate_task`, `_bind_workflow`,
a `_switch_profile` override that stamps the task's workflow, profile
rename/delete propagation, lazy pickle reload on activation, per-task
`models/<uid>/` autosaves, a New session that keeps the tasks; the Tasks
`Treeview` + New / Dup / Rename… / Delete in the left column, the profile
box titled "0. Workflow", double-click-to-name on a class title. Decisions
taken on the way: task identity is uid + name (§13.3 closed); the four
Model-tab keys left the window `view` for the task's; the rev-keyed caches
are cleared on a switch rather than discriminated. Tests: `test_task.py`,
`test_class_names.py`, `test_session_doc_tasks.py`, task sections in both
labeler selftests. Not done, by design: places/enrolment, slide-bound
gestures, layered cache keys (a cross-workflow switch still drops primes),
subscriptions, multi-task display.

**What exists (stage 2, 2026-09-21).** Gestures are keyed by the slide
(`ItemCatalogue.binding_of` / `rebase`, `LabelStore.for_item` /
`gesture_extent`, the shell's `_gestures_for_key` / `_gesture_on_item`
replacing every item-key and hint-equality selection; hints name the
slide's first row); an ROI's removal dooms nothing, the slide's dooms all;
older item-keyed stores rebase on load with the item's level / scale
recorded. `_draw_meta` records the scale of intent (`level`, `scale`,
`px`; mspath only). Width-aware strokes (`stroke_mask`, PIL; always for new
gestures, > 1.5 raster px). Extents (`extents.py`): the outline of a magic
fill / blob / accepted prediction as closed loops via the seam graph of the
0/1 mask, Euler-chained, even-odd filled on a 2x centre grid; consulted off
the drawing level only. Trace corridor (`seam_labeling.corridor_coverage`,
radius = the coarser of the two pixels; exact ids on the own lattice).
`_accept_predictions` fixed to honour the placement. Coarse-gesture mark
(`!`, `COARSE_LEVELS = 2`) and a once-per-item notice. Decisions taken on
the way: width applies always for new gestures (§8.2.2), mpp deferred
(§8.3, still recommended as a per-slide frame later), `meta` keys are read
only to choose HOW geometry applies off its level (§8.1's rule made exact),
the seam-graph-of-the-mask outline rather than a contour tracer (§13.3
closed: polygon loops, holes as loops). Tests: `test_slide_binding.py`,
`test_catalogue_binding.py`, `test_stroke_width.py`, `test_extents.py`,
`test_seam_corridor.py`; the mspath selftest's tree-rows block rewritten
(no orphan concept). One-way compatibility: a store written by stage 2
(slide keys) loaded by stage-1 code binds nothing in mspath (kept greyed).

**What exists (stage 3, 2026-09-21).** `msseg/labeler/derive.py` is the one
label derivation: `region_labels` (= `resolve_slice`), `seam_labels` and
`arc_labels` implement the §7.2 matrix in three passes -- samples, extents,
explicit seam gestures -- with `is_extent`, `extent_sets`, `enclosed_ids`,
`is_closed`, `labels_for_item`. Consumers: `panels/seams._seam_cache_for`
(derived labels reach the overlay, the readout `(n derived)` and the seam
model), `training.edge_set(arc_labels_of=)` via `classifier._arc_labels_for`.
Producers of the flag: the blob core always, the magic fill by an `extent`
checkbox (default on, in `view.magic`), the lasso unless Ctrl-dragged;
SHIFT-accept never. The enclosure gesture: a trace closed on its first anchor
auto-commits and parks; a tap inside with a class armed stores the enclosure
as a fill-shaped extent (`_commit_enclosure` -> `_commit_blob`, meta
`enclosure` + `trace` uid); outside starts a new trace; Esc leaves the
boundary. Three amendments to the matrix as written above, taken during the
design pass: (1) an extent speaks only about UNLABELLED neighbours (a
same-class sample next to it stays interior / same), so fill-then-patch
never derives a boundary inside an object and the seam and arc targets agree
-- the instance boundary is a trace's job (`EXTENT_EDGE_VS_SAME_CLASS`);
(2) only the blob's core is an extent, its ring is a sample; (3) the
enclosure is stored as `taps` + outline, not as a `polygon` gesture, because
a polygon fill on a crack-aligned loop leaks one pixel outward while the
even-odd fill of `extents.extent_mask` is exact. §13.2 is closed (blob
always, fill by checkbox, lasso by Ctrl). Tests: `test_derive.py`,
`test_edge_hook.py`; coupon selftest sections for the flag, the derived
seams and the enclosure flow.

**What exists (stage 4, 2026-09-24).** Places and enrolment as §5 now reads:
`msseg/mspath/places.py` (uids, rect lookup, enrol / unenrol, legacy
materialisation, drops), `Task.enrolled` + `session_doc.normalise_enrolled`
(written only when not None; `{}` written), the viewer hooks
`_place_enrolled` / `_place_level` behind `_enumerate_items` -- the one choke
point, so Train / Classify / exports / Run follow the active task -- and a
**browse** state for rows the task does not work (`slice_var = -1`,
`_current()` None, so no gesture can land on the wrong item). The Run splits
into **Run task** and **Run all tasks** (the union over tasks on the active
workflow; tasks on another workflow wait for stage 5). **The fast path**:
model operations never prime (`_stream_ready`, uncomputed items are skipped
and counted), `C` / `R` classify the item on screen, *Classify all* does the
rest, and an item is classified when it is selected (`CLASSIFY_ON_ARRIVAL`,
mspath). The per-task cache signature is `(keys, rows)`, stamped on leaving
a task too, pruning only vanished keys. Legacy note: a v2 / pre-stage-4
task reads as "works its places" -- never the overview -- with a load note;
stage-3 code reading a stage-4 document drops the uids and ignores
`enrolled`, so every task would see everything again. Tests:
`test_enrolment_doc.py`, `test_places.py`; the mspath labeler selftest's
enrolment block. Pre-existing limit made easier to hit: `_feature_scope`
still names only the overview level.

**What exists (stage 5, first half: the measurement split, 2026-09-24).**
C++: `Msc2DPipeline::build` = the MSC phases + one `measure_leaves` step
(relevance, GPU or CPU accumulation, extremum; byte-identical output, same
phase order), and `remeasure(base, filtered, cfg, color)` runs that step
alone on a built pipeline -- shape-checked, then `select_persistence` at the
current persistence, its phases in `build_timings()` (no `msc`). pybind:
`pipe.remeasure(params_json, base, filtered, color=None)`. Two fingerprints
over the params document (`msseg/mscoupon/fingerprints.py`; `session.
field_fingerprint` / `measure_fingerprint`): the **field** is the chains, the
colour input minus its plane count, and the MSC keys minus the result-free
ones (cores, builder, GPU flags) and minus the sample radius, with the
persistence percentage reduced to the cancellation cap `max(10 %, pct)`;
the **measurement** is the `statistics` block, the sample radius and the
plane count. Engines keep both stamps: the coupon `ComputeEngine` stamps
`measured[]` per primed slice and re-measures lazily inside `_slice_result`
from the rasters it already holds; `SlideEngine` stamps each `Primed` with
`field` / `measured` / the chains it was primed with, re-measures in
`ensure_record` (and on the worker for navigation, `remeasure_only`), and
re-reads the item through those chains -- never the panel's, since an
un-Run chain edit is a preview. A Run keeps what the field can reuse (the
coupon by the field fingerprint per sequence, mspath by `keep_field`, pins
kept when anything was); a profile switch -- hence a task switch and
`_profile_from_model` -- resets only when the field differs
(`_keep_compute_for` / `_after_profile_kept` / `_remeasure_current` in the
shell). The labeler moved the statistics (and the ext sample radius, which is
a measurement) to a **Features** tab, re-measures the item on screen after a
settled edit, and heads the classifier section with a `stats:` link to it.
Tests: `test_msc2d_remeasure` (C++), `test_gpu_stats_parity.py`
(remeasure == fresh prime, CPU and GPU), `test_engine_remeasure.py`,
`test_slide_engine.py` (re-measure keeps labels, `keep_field`, the chains
the pipe was primed with, `remeasure_only` never primes), the fingerprint
test in `test_session_model.py`; the mspath labeler selftest re-measures a
real item and keeps it across a statistics-only profile switch. Still open
for stage 5: records keyed by `(field, measurement, persistence, item)` so
two tasks on different statistics keep BOTH rows live (today the last one
measured wins, and a switch re-measures), `base_filters` on the measurement
side (needs the raw slice kept; a base-chain edit still re-primes), and
`RegionProvider.commit` becoming a record key.

**What exists (stage 5, second half: layered keys, 2026-09-24).**
`msseg/labeler/record_keys.py`: `Interner` (key -> stable int) and
`RecordCache` (a few records per item, LRU, plus the decomposition shared by
records that differ only in their measurement). `rec["commit"]` is the
interned id of what the record is a function of; `protocols.py` says so and
`RegionProvider.commit` is now an informational generation. mspath
`SlideEngine`: field slots (`primed` / `level_range` / `persistence_abs` are
properties over the ACTIVE slot, so no caller changed), `use_field` (never
drops another field; the old `keep_field` is its alias), one global LRU over
every slot, pins per field, `set_params` + `record_id` + a record cache that
outlives a released pipe, and `start_run` jobs `(item, profile)` that prime
into that profile's slot -- the labeler's **Run all tasks** now primes every
task's items under its own workflow. The coupon `ComputeEngine`: `_ByCommit`
maps (a few records per slice, two assemblies per sequence) read as the
current record, `commit_selection(snapshot)` interns `(stack generation,
_selection_snapshot())`, so going back to earlier parameters returns their
commit; the coupon keeps ONE field live (a cross-field switch still drops the
stack -- a decision, for memory). The base chain moved to the measurement
side: the fingerprints, both engines' re-measure (the coupon re-reads the
slice file and rebuilds `base` + `normalizers`; mspath rebuilds `Primed.base`),
the Features tab (`_FEATURE_SECTIONS = ("stats", "base")`), and one settle
path -- `AnnotationShell._preview_edit_settled` asks the app's
`_measurement_moved()` and re-measures the item on screen for a base or
statistics edit, leaving a field edit a preview. Tests: `test_record_keys.py`,
`test_slide_engine.py` (identity, bound, fields and pins, jobs, base chain),
`test_engine_remeasure.py` (A -> B -> A, bound, base chain), the fingerprint
test; the mspath labeler selftest switches task A -> B -> A and finds A's
record and prediction with no re-measure, and runs Run all tasks over a
third field; the coupon labeler selftest checks that a profile round trip
returns its commit.

Stage 1 touched `session_doc.py`, the new `msseg/labeler/task.py`,
`labeling.py`, `annotate.py`, `shell.py`, `panels/classpanel.py`,
`mspath/labeler.py` (`_profile_from_model`). Stage 2 touched `labeling.py`
(binding, bbox, width), `seam_labeling.py` (corridor), `annotate.py`
(`_slice_key`, counts, dooming), `mspath/labeler.py`, and the magic-fill /
blobber commit path (outlines) -- plus `protocols.py`: `ItemCatalogue` grew
`binding_of` and `rebase` (and `index_of` resolves a slide key), the one
protocol change before stage 5, which changes `RegionProvider.commit` to a
record key; stage 8 needs several providers' layers live at once.

## 13. Open questions

1. ~~**Can `Msc2DPipeline` be re-measured in place?**~~ Closed 2026-09-24:
   yes (`Msc2DPipeline::remeasure`, §6). Measurement is a cheap axis -- a
   prime minus its MSC -- though not as cheap as persistence: the channel
   bank is most of it (0.6 s of 0.9 s for twelve channels on 2048²).
2. ~~**Extent by tool or by modifier?**~~ Closed in stage 3: the blob's core
   is always an extent (its ring never), the magic fill by an `extent`
   checkbox (default on), the lasso unless Ctrl-dragged, an accepted
   prediction never. Open in its place: whether an extent's edge against a
   same-class *labelled* neighbour should ever read as an instance boundary
   (`derive.EXTENT_EDGE_VS_SAME_CLASS`, off; a trace does it today).
3. **Outline representation for extents**: polygon (compact, may
   self-intersect after a re-resolution) vs run-length mask in slide
   coordinates (exact, larger). Polygon first.
4. **Scale warning threshold**: how many levels finer than drawn before the
   badge; and whether a task declares a level *range* its vocabulary is
   valid over.
5. **Corridor radius for traces off-level**: one drawing-level pixel, or the
   recorded stroke width.
6. ~~**Does a seam-only task collect region gestures**, or only subscribe to a
   region task's extents?~~ Closed 2026-09-25: it subscribes (§7.3: a
   polyline task's *region labels* input; its store holds seam gestures only).
7. **Level per task** (a coarse pass and fine ROIs are two tasks or two
   sweep rungs) -- confirm acceptable.
8. **Mask threshold semantics** per subscription (any-overlap vs majority).
9. ~~**Task uid vs name** for cross-references.~~ Closed: uid (an input
   slot stores `{"source": "task", "uid"}`; a rename keeps it working).
10. **Where gestures live on disk** with ten tasks.
11. **Model history retention.**
12. **Physical units on workflow parameters** (`sigma` in µm, `area` in µm²)
    -- a separate note under `design_filter_types.md`, referenced from §8.3.
