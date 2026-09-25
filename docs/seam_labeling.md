# Seam labeling: polyline tasks

The labeler labels **regions**, and -- in a **polyline task** -- the
**boundaries between them**. A task's kind is fixed when it is made (Tasks >
New > *Region task* / *Polyline task*): a region task classifies regions, a
polyline task classifies the polylines between regions (**seams**) into
classes of its own ("not a boundary", "gland wall", "cut artefact", ...).
The user draws **traces** (livewire shortest paths that snap to the
polylines) and **scopes** (boxes inside which every polyline is labelled) in
an armed class, a multiclass **seam model** learns the classes from the
descriptors of the two flanking regions, and the classified polylines are
reviewed (confusion matrix, error list, click-to-see) and exported. A
polyline task can also read a region task's work -- its annotations, its
net, its edge model -- through **input slots**. Everything lives in the
framework (`msseg.labeler`), so `mscoupon-labeler` and `mspath-labeler` both
have it.

The region side is the standard: every review affordance a region task has
(class frames with counts, click-to-see, drag-to-relabel, a K x K fit check
with highlight, a held-out report, an error list that goes to the place) a
polyline task has over seams.

## Vocabulary

| term | meaning |
|---|---|
| **region** | a living basin, one id in the int32 label raster (-1 = background) |
| **descriptor** | a region's statistics row in the `FeatureTable` (its feature vector) |
| **arc** | the adjacency pair `(a, b, saddle)` of two regions (`record["arcs"]`, unchanged) |
| **corner** | a pixel-corner lattice point at integer image coordinates; pixel `(ix, iy)` covers `[ix, ix+1) x [iy, iy+1)` |
| **crack** | the unit lattice step between two corners that separates two differently labelled pixels; its id is label-independent (`2c` for the step from corner `c = iy*(w+1)+ix` to `(ix+1, iy)`, `2c+1` to `(ix, iy+1)`) |
| **seam** | a maximal chain of cracks between the same two regions, junction to junction; a **loop** when it closes on itself (an island) |
| **junction** | a corner where three or more regions meet, or where a seam ends: a degree-1 corner (against background or the raster edge) or a degree-2 corner whose two cracks separate different flank pairs |
| **seam graph** | junctions + seams -- the polyline graph between regions |
| **flanks** | the two regions `(a < b)` a seam separates |
| **seam descriptor** | the per-seam feature vector (pair terms on the flank descriptors or their embedding, saddle barrier, edge-model p(diff), geometry) |
| **seam class** | a polyline task's own vocabulary: `0` unknown, `1` the "not a boundary" role (renameable), `2..` kinds of boundary. A new task has `1 not a boundary`, `2 boundary` (`task.POLYLINE_DEFAULT_CLASSES`) |
| **boundaryness** | `1 - P(class 1)` per seam: the probability that a seam is *some* boundary, whatever the kind |
| **toll** | the cost per crack of walking a seam: `eps + (1 - affinity)` |
| **trace** | a drawn shortest path along seams (a stored gesture, in the armed class) |
| **scope** | a box inside which every seam is labelled (in the armed class; a trace inside it wins) |
| **input** | a slot a polyline task fills from another task: region labels, region embedding, p(diff) (`artifacts`) |

"Edge" is deliberately not used for any of these: it already means the
`edges` derived statistics channel (say *edge response*) and the region-pair
**edge model** (`edge_model.py`, `pdiff`), both unchanged.

## Region tasks and polyline tasks

`Task.kind` is `"region"` or `"polyline"`, written in every task of the
session document (v4) and fixed from creation (converting would orphan every
gesture). *Dup* keeps the kind; the Tasks list shows it (▦ / ⌇). The kind
decides what the five right-hand tabs show -- Processing is the workflow
and is the same for both:

| tab | region task | polyline task |
|---|---|---|
| Features | base channel + statistics | base channel + statistics (the flank rows) + **6. Seam descriptor** (which blocks the seam model reads) + **7. Inputs** (slots) |
| Annotation | classes, region tools (squiggle, box, lasso, magic, blobber, **outline**), class frames, ML Region Classifier | classes, polyline tools (**trace**, **scope**), the trace row (toll, channels), class frames of seam gestures, **ML Seam Classifier** |
| Model | region kind, context, edge model, Optimize | seam classifier (logistic / MLP, C, embed, layer) + **Evaluate** with a held-out report table |
| Analysis | error list per confusion cell, size sweep | seam error list per confusion cell (double-click = go to the seam) |

`AnnotationShell._apply_task_kind` packs one variant and forgets the other
on every task switch. The tools are gated by kind (`_REGION_TOOLS` /
`_POLYLINE_TOOLS`; a `tool_var` trace coerces a tool the kind does not offer
back, which also covers a restored session view), and so are the keys:
digits arm a class and **R** / **C** train / classify the ACTIVE task's model
in both; **M** / **B** only in region tasks; **T** / **S** / **E** and, with
a trace in flight, **Enter** / **BackSpace** only in polyline tasks.

**A polyline task's store vocabulary IS its seam vocabulary.** Its
`LabelStore`'s `n_classes` / `colors` / `names` describe seam classes, its
gestures live in `store.seams` and `store.interactions` stays empty. That is
what lets the class panel machinery serve both kinds: the class frames list
seam gestures (`#uid trace (n seams, toll)`, `#uid scope (n seams)`), the
titles count `gestures · seams`, a row click goes to the gesture and draws
its polyline (`_draw_seam_geometry`, re-projected on pan / zoom), a row drag
relabels (`store.set_class` covers seams), the context menu and *clear
class* work, and canvas hover / right-click find seam gestures
(`_seam_gesture_at`, the last match in application order -- the one whose
class the seam shows). `add_seam` accepts classes `1 .. n_classes - 1`.

**The outline tool (region tasks).** The livewire also serves region tasks,
as a way to draw an enclosure: with a class armed, trace around the object
and click the first anchor again -- the loop closes and becomes ONE extent
of that class (`_commit_outline` -> `_commit_blob`, `meta = {"tool":
"outline", "extent": True, "n_regions", "toll", "anchors"}`: the same
`taps` + outline a magic fill writes, so it re-resolves and undoes like a
fill). No seam gesture is stored -- open traces and scopes exist only in
polyline tasks. An outline that does not close stays in flight ("An outline
closes on its first anchor"). The inside test is the even-odd fill of the
loop's cracks (`extents.extent_mask`), exact on the crack lattice where a
polygon fill would leak a pixel outward; a loop needs at least two
intermediate anchors, since a return leg after a single leg retraces it
(the search is symmetric) and encloses nothing.

## Where the polylines come from: the crack graph

The seams are traced from the **label raster**, not from MSCEER's arc
geometry. MSCEER can build a ridge polyline graph (`computePolylineGraph`),
but it needs `build_arc_geometry` on at prime time, does not exist in the
default `merge_forest` mode, lands on half-pixel cell coordinates, and still
needs every polyline attributed to its two flanking basins. The crack graph
is exactly the boundary of the regions the user clicks, in every mode, with
no re-prime -- and its corners sit on a segmentation-independent lattice,
which is what lets a stored trace be matched against a *different*
decomposition of the same raster (below).

`msseg::extract_seam_graph` (`libs/core/msseg/graph/seam_graph.cpp`, no
MSCEER, exposed as `mscoupon_py.seam_graph(labels)`) enumerates the cracks
(a horizontal lattice edge exists where the pixel above differs from the
pixel below, both >= 0; vertical likewise), counts the degree of every
corner, flags the junctions, chains cracks from every junction in a fixed
direction order (+x, +y, -x, -y) until the next junction, and sweeps the
leftover cracks into loops. The output is **canonical** -- an open seam runs
from the smaller junction id to the larger (a seam from a junction back to
itself runs toward its smaller second corner), a loop starts at its smallest
`(y, x)` corner and leaves it in +x, seams are sorted by
`(a, b, j0, j1, first corner, second corner, length)`, junction ids are ranks
in raster order -- so the pure numpy reference
`msseg.labeler.seams.seams_from_labels` reproduces it array for array.
`packages/mscoupon/tests/test_seam_graph.py` is the parity test; the
reference is the fallback when the extension lacks the symbol (~1-2 s on
3232², against milliseconds for the C++). The seven arrays are
`a, b, j0, j1` (`j0 == j1 == -1` marks a loop), `offsets`, `points (P, 2)`
(a loop repeats its first corner) and `junction_xy (J, 2)`.

The framework reaches it through `RegionProvider.seams(key, np)`: the
coupon and mspath providers build a `seams.SeamGraph` once per record
(commit-keyed, cached on the record like the pixel-adjacency arcs); the
mspath one carries the item's `Placement`, so corner points convert to and
from slide coordinates.

## Annotations are geometry, and re-resolve by crack coverage

Seam gestures are `labeling.Interaction`s in image coordinates like every
other gesture, keyed by the SLIDE and seen by every item covering them
(docs/design_multi_model_tasks.md §8). On the lattice a gesture was drawn on
`meta` is never read by resolution; off it -- a trace drawn on the level-4
overview resolved against a level-0 ROI, whose lattice is sixteen times finer
-- `meta["scale"]` (the drawing item's slide px per raster px, recorded by
the app's `_draw_meta`) switches a trace from exact crack ids to a
**corridor**: a seam is covered when at least `SEAM_COVER_TAU` of its corner
points lie within one pixel of the coarser lattice of the trace's polyline
(`seam_labeling.corridor_coverage`, `tests/test_seam_corridor.py`). A scope
needs nothing: bbox containment is scale-invariant. They live in a
**separate list**, `LabelStore.seams`, and serialize under a `"seams"` key
with `"version": 3` -- written, and the version raised, **only when there
are any**, so a store without seams is byte-identical to v2
(`tests/test_compat_docs.py`) and an older reader, which rasterizes any
unknown tool as a squiggle, never paints a trace onto regions. The store's
`add_seam` / `for_slice_seams` / `remove` / `remove_many` / `rebind` cover
both lists; uids are shared, so undo (whole-store snapshots), row removal
and the sequence tree's "Clear annotations" take seam gestures along.

`seam_labeling.resolve_seams(gestures, graph) -> uint8[S]`:

* a **scope** labels every seam whose corners all lie inside its box with
  its class;
* a **trace** is turned into crack ids (`seams.cracks_of_polyline`; stored
  points are corners with collinear runs compressed, so a segment expands
  into its unit steps) and labels every seam at least `SEAM_COVER_TAU`
  (0.5) of whose cracks lie under it with its class;
* scopes apply first (by uid), then traces (by uid), each overwriting -- so
  a scope widened *after* a trace does not erase it. This is the one
  deliberate departure from `resolve_slice`'s pure uid order.

After a persistence change the graph is rebuilt and the same gestures
re-resolve: a seam that merged into a longer one keeps its label while it is
still half covered, and one that vanished simply has no successor
(`tests/test_seam_labeling.py::test_traces_re_resolve_after_a_merge`).


## Derived labels: region work speaks to the seams

Seam gestures are not the only source of seam labels. `derive.seam_labels`
(`msseg.labeler.derive`, the one label derivation for regions, arcs and
seams -- design note §7.2) runs three passes over an item's seam graph:

1. **samples** -- a seam whose two flanks both carry a region class is a
   boundary when the classes differ and class 1 when they agree;
2. **extents** -- a lasso, a released magic fill, a blob's core, an outline
   (`meta["extent"]`; older fills with an outline count too, blob rings
   never, accepted predictions never): a seam with one flank in the extent
   and the other flank *unlabelled* is a boundary. Only unlabelled
   neighbours: a same-class sample next to an extent stays class 1 (and
   the arc "same"), so filling a gland twice, or patching a fill with a
   squiggle, never derives a boundary inside it. The instance boundary
   between two touching same-class objects is a trace's job
   (`derive.EXTENT_EDGE_VS_SAME_CLASS` keeps the other reading one constant
   away);
3. the **explicit seam gestures**, scopes then traces, overwrite.

Where the region gestures come from depends on the task:

* in a **region task** they are its own, and the derivation feeds the arc
  target of its edge model (`derive.arc_labels`, through the edge set's
  `arc_labels_of` hook: both endpoints labelled -> same / diff, extent |
  unlabelled -> different);
* in a **polyline task** they come from its **region labels input** -- a
  region task's gestures on the same slide, re-resolved against THIS task's
  decomposition (gestures are geometry, so the two tasks may run different
  workflows). The derived boundary lands in the slot's *boundary class*
  (default 2; `seam_labels(boundary_class=)`). Derived labels are shown and
  trained on like drawn ones (the readout adds `(n derived)`), never stored.

Gestures drawn much coarser than the item are left out of the derivation (a
swath's edges are not boundaries).

## Inputs: what a polyline task reads from region work

The strongest evidence about a boundary is often region work that already
exists. A polyline task declares **input slots** (Features > **7. Inputs**),
each filled by a *provider*:

| slot | what it gives the polyline task | today's provider |
|---|---|---|
| region labels | derived seam labels (above), into the chosen boundary class | a region task's annotations |
| region embedding | the seam descriptor's `pair` terms in the net's hidden layer (`embed` = net / auto) | a region task's trained net |
| p(diff) | the seam descriptor's `edges` term and the `edges` toll | a region task's edge model |

`msseg.labeler.artifacts` is written for a **model artifact library** that
does not exist yet: a slot stores a *reference*, `{"source": "task", "uid":
...}` today and `{"source": "library", "id": ...}` later, and
`artifacts.resolve` turns it into a `Provider` with capabilities
(`gestures_for`, `base`, `stack`) and **requirements** (`Requirement`: the
feature columns its model reads and the scope they were measured at -- the
compatibility gate a loaded model passes, as a subset test). A reference
that does not resolve (a deleted task, the unbuilt library, the task
itself) resolves to a `MissingProvider` that says why. The offline region
encoder (`.msenc`, `embedding.EncoderBundle`) is the first library provider
the interface fits.

A task provider (`TaskProvider`) is read **without activating it**: its
store is plain data, and its model is the live stack when it was trained or
loaded this session, else its newest saved region pickle read by the pure
`load_model_stack` into a stack of its own (cached by path and mtime; the
provider's own stack and `model_pending` are untouched, since loading that
gates against the active workflow). `artifacts.pdiff_for` runs the
provider's net + edge model on the consumer's record, per arc in the
record's order -- what `_predict_slice` puts in `aux["pdiff"]` for its own
task -- cached per `(commit, provider signature)`.

Every slot shows its verdict under its picker: `3 region gestures; derived
boundaries -> class 2 'gland wall'`, or `cannot use: needs mean_blur_s1.5,
mean_blur_s3 (+2 more); workflow 'wide' does not measure it`, `glands has no
edge model (Model tab, an '-> edges' kind)`, `glands's model reads context
columns (not usable as an input yet)`. The stage strip grows an **inputs**
box feeding the model -- green when every filled slot can be used, orange
(with the reasons as its tip) when one cannot; a click opens Features. The
seam model stamps the inputs it was trained with (`trained_inputs`, the
slots' provider signatures, which include the labels provider's store rev),
so the model box turns stale when the provider's annotations or model, or a
slot, change. Changing a slot drops the seam predictions (their descriptor
moved) and re-derives.

`Task.inputs` is `{slot: reference[, "boundary_class": k]}`, written in the
task's document only when set (`artifacts.normalise_inputs` is the total
reader) and copied by Dup; a region task has none.

## The tools

Both live in a polyline task's Annotation tab (Tool row: `trace` **T**,
`scope` **S**) and paint the **armed** class. `tools.TraceController` owns
both, and the region tasks' outline; `DrawController` dispatches to it
before the armed-region-class test.

**Trace** is a livewire (`seam_path.Livewire`); it needs a class armed (a
digit, or a click on a class title). A press snaps to the nearest
crack within 8 raster pixels (`seams.nearest_seam_point`, re-derived from the
label window under the pointer) and anchors there: the anchor is a virtual
node joined to its seam's two junctions at the partial-length cost, and ONE
single-source Dijkstra (heapq, the `magic_fill` pattern) runs over the
junction graph. Every mouse move then shows the cheapest path from the anchor
to the seam point under the pointer -- a predecessor walk, not a search --
with the HUD reading the toll, the anchors, the accumulated cost, the seams
on the path and the hover leg's cost. A click freezes that path as a leg and
re-anchors; **Enter** or a double-click commits the legs as ONE `trace`
gesture (`meta`: toll, anchors, seams, cost, scoped); **BackSpace** drops the
last leg (the first anchor too, which abandons); **Escape** abandons. An
anchor inside a scope confines the search to that scope's seams. A same-seam
target takes the direct run, or the cheaper way round a loop; a detour via
the junctions can beat the direct run on an expensive seam.

**Scope** is a box drag previewing, on the transient layer, the seams that
lie fully inside it; the release commits a `scope` gesture.

**Scope** is a box drag previewing, on the transient layer, the seams that
lie fully inside it; the release commits a `scope` gesture in the armed
class (class 1, "not a boundary", is the usual choice: a scope says
"everything in here that nobody traced is not a boundary").

### Tolls

`seam_path.seam_affinity(graph, toll, ...)` gives an affinity in [0, 1] per
seam (1 = certainly a boundary); `toll = 0.05 + (1 - affinity)`; a leg
costs `sum(length x toll)`.

| toll | affinity | needs |
|---|---|---|
| `geometric` | 0 everywhere: shortest in cracks | nothing |
| `feature` | z-scored mean dissimilarity of the flanks over the channels (`magic_fill`'s `mean` metric), scaled by its 95th percentile over the slice | the statistics table |
| `bhattacharyya` | the flanks' per-channel Gaussian overlap, likewise | the table |
| `barrier` | the arc's saddle depth `|saddle - max(ext_a, ext_b)|`, likewise | MSC region arcs with saddles |
| `edges` | an edge model's p(diff) for the flank pair | the p(diff) input (Features > Inputs) |
| `model` | the seam model's boundaryness | a trained seam model |

A toll whose input is missing refuses the press with the reason on the HUD
and the status line (the magic fill's "Classify first" idiom); a seam whose
pair has no arc (MSC mode drops some saddles) gets affinity 0 under the arc
tolls.

## Rendering and the view options

In a polyline task the region overlays fade to context and the seam layers
ride on top (`panels/view.py::_seg_overlays` -> `_seam_layers`):
`seams.seam_pixel_raster` paints the seam index on BOTH flank pixels of
every crack -- a 2-pixel line that survives the canvas's nearest-neighbour
zoom-out -- and LUTs colour it:

* the **predictions** (*Show classification*), coloured by the seam colour
  mode: `class` (the predicted class in the task's colours),
  `boundaryness`, `uncertainty`, or `P(class k)` for each class, on the
  labeler's ramp (`seam_scalar_lut`);
* the **labels** (*Show GT*): resolved classes, drawn and derived
  (`seam_class_lut`, class 0 transparent);
* the **highlight** of a selected confusion cell's seams.

Everything goes through `_region_overlay`, so mspath places it on the slide
like every region layer; **E** toggles the seam layers. The raster is
rebuilt per commit and the classes per `(commit, store.rev[, labels input])`
(`_seam_raster_for` / `_seam_cache_for`). Committed traces and scopes draw
as Tk items in the persistent annotation layer and on row hover; the trace
in flight, its hover leg and its anchors are the `"trace"`-tagged items the
controller re-projects on every pan/zoom. A region task draws no seam layer.

## The seam model

`seam_model.py` mirrors `edge_model.py`. The **seam descriptor**
(`seam_features`) concatenates, per the `SeamSpec` (Features > Seam
descriptor for the blocks, Model > Seam classifier for the rest; the spec
rides the task's view as `seam_spec`):

* `pair`: `|d|` and the product of the two flanks' region descriptors -- the
  z-scored statistics rows (`embed = "features"`), or the flanks' embedding
  in a region net's hidden layer (`"net"`, through `edge_model.embed`; in a
  polyline task that net is the **embedding input's**; `"auto"` picks it
  when available and the net's inputs are plain table columns);
* `barrier`: `edge_model.barrier` -- the saddle depth and the extremum gap;
* `edges`: p(diff) for the pair (the **p(diff) input's** edge model), plus a
  present flag;
* `geometry`: crack length, its log, the bbox aspect, the tortuosity
  (length / chord) and whether the seam is a loop.

**Train (R)** (ML Seam Classifier) gathers every primed item's descriptors
and resolved classes (`SeamModelMixin._seam_items`, over
`_stream_stat_slices`), fits a balanced **multinomial** logistic regression
(or a (32, 16) MLP) behind a `StandardScaler` on the labelled seams (at
least two classes), stamps what it was trained on (`trained_rev`,
`trained_measure`, `trained_inputs` -- the stage strip's staleness) and
scores every seam of every item. `_seam_pred[key] = (commit, boundaryness,
proba (S, n_classes))`, columns by class id; `predicted_class` is the
argmax. **Classify all (C)** re-scores. The boundaryness is what the
livewire's `model` toll and the boundaryness colouring read.

**Review**, as for regions:

* the **K x K fit check** in the ML Seam Classifier (truth = resolved seam
  classes, prediction = argmax), counted by *seams* or by *crack length*
  (`_seam_confusion_counts`); a click on a cell highlights its seams on the
  item on screen (`_seam_confusion_hits`), a double-click opens the
  **Analysis** list of that cell's seams on every classified item -- item,
  seam, flanks, annotated, predicted, length, P(annotated), P(predicted) --
  and a double-click on a row goes there (`_goto_seam`: centre on the
  seam's midpoint, draw it, the `_goto_region` pattern);
* **Evaluate** (Model tab, on the Optimize worker/pump; Cancel stops after
  the current fold) runs leave-items-out CV (`model_search.make_cv` over
  `catalogue.group_of`) and fills a table per fold plus the mean -- seams,
  balanced accuracy, boundaryness AUC, log-loss -- and a per-class table of
  held-out recall / precision.

**Its own pickle.** A polyline task's model is a `bundle.SeamBundle`
(`{"app", "version": 1, "task_kind": "polyline", "seam", "statistics",
"classes": {k: name}[, "scope"]}`), saved and loaded from the ML Seam
Classifier; the task's model record carries `"task_kind": "polyline"`, and a
task switch reloads it lazily like a region model. A region pickle carries
no seam head any more, and each loader refuses the other's file.

## Export

**Export seams…** (`SeamModelMixin._export_seams_to`) writes, for every
primed item, `seams_<item>.json` -- `{item, shape, placement, classes,
n_seams, n_junctions, seams: [{id, a, b, j0, j1, length, class, class_name,
predicted, boundaryness, points}]}` with the task's class names, the points
in image coordinates and collinear runs compressed -- and one
`seams_summary.csv` over all items. That is the input a later step
rasterizes with a width to train an image model.

## Sessions

The session document is **v4**: every task declares its `kind`, a polyline
task may carry `inputs`, and its `view` may carry `seam_spec`. A labeler
document that does not declare kinds (anything older) **fails to load**
with "written by an older labeler -- start a new session"
(`session_doc.labeler_refusal`); there is no migration. The viewers'
tasks-less document still reads as one region task. The window view keeps
`view.seams` (toll, channels, show, colouring) and `view.tool`, which falls
back to the kind's first tool when the active task does not offer it. Seam
gestures serialize under the store's `"seams"` key (`"version": 3` only when
present).

## Tests

`packages/mslabeler/tests`: `test_seams.py` (crack ids, the junction rule,
the canonical form, random rasters, snapping, the pixel raster, LUTs,
placement), `test_seam_labeling.py` (the store's seam list, JSON v3, scope
containment, coverage, precedence, re-resolution after merges, placement),
`test_seam_path.py` (tolls, the livewire's paths, reroutes, scope
restriction, legs, loops, collinear compression), `test_seam_model.py`
(descriptors, multiclass fit / predict, three classes, the `SeamBundle`,
leave-items-out evaluation with per-class scores, the net embedding; needs
scikit-learn), `test_seam_export.py`, `test_task_kind.py` (kinds, the v4
document, the older-document refusal, polyline vocabularies),
`test_artifacts.py` (references, slots, resolution, the gate, a model read
without activation, p(diff) from another task, the derived boundary class)
and `test_compat_docs.py` (v2 byte-identity, the v3 fixture).
`packages/mscoupon/tests/test_seam_graph.py` certifies the C++ against the
numpy reference; `libs/core/tests/core_smoke.cpp` section 7 checks the C++
on its own. Both labeler selftests drive a polyline task headlessly (the
coupon one end to end: tools, class frames, the model, the matrix, the
tabs, the pickle, the inputs).
