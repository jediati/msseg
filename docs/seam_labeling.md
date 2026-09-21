# Seam labeling: boundaries as first-class annotations

The labeler labels **regions**. Boundaries used to be reachable only
indirectly: the `-> edges` model scores region *pairs* and is trained from
region labels, so it never sees a boundary annotation. Seam labeling makes
the polylines between regions annotatable in their own right: the user draws
**traces** (livewire shortest paths that snap to the polylines) that become
`boundary`, and **scopes** (boxes inside which every polyline is labelled,
`interior` by default), a **seam model** learns boundary vs interior from the
descriptors of the two flanking regions, and the classified polylines are
exported. The tools live in the framework (`msseg.labeler`), so
`mscoupon-labeler` and `mspath-labeler` both have them.

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
| **seam descriptor** | the per-seam feature vector (pair terms on the flank descriptors, saddle barrier, edge-model p(diff), geometry) |
| **boundaryness** | the seam model's p(boundary) per seam |
| **toll** | the cost per crack of walking a seam: `eps + (1 - affinity)` |
| **trace** | a drawn shortest path along seams (a stored gesture, class boundary by default) |
| **scope** | a box inside which every seam is labelled (interior unless a trace says otherwise) |
| **seam class** | `0 unknown`, `1 interior`, `2 boundary` (`seams.SEAM_CLASSES`) |

"Edge" is deliberately not used for any of these: it already means the
`edges` derived statistics channel (say *edge response*) and the region-pair
**edge model** (`edge_model.py`, `pdiff`), both unchanged.

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
other gesture (`meta` is never read by resolution). They live in a
**separate list**, `LabelStore.seams`, and serialize under a `"seams"` key
with `"version": 3` -- written, and the version raised, **only when there
are any**, so a store without seams is byte-identical to v2
(`tests/test_compat_docs.py`) and an older reader, which rasterizes any
unknown tool as a squiggle, never paints a trace onto regions. The store's
`add_seam` / `for_slice_seams` / `remove` / `remove_many` / `rebind` cover
both lists; uids are shared, so undo (whole-store snapshots), row removal
and the sequence tree's "Clear annotations" take seam gestures along.

`seam_labeling.resolve_seams(gestures, graph) -> uint8[S]`:

* a **scope** labels every seam whose corners all lie inside its box;
* a **trace** is turned into crack ids (`seams.cracks_of_polyline`; stored
  points are corners with collinear runs compressed, so a segment expands
  into its unit steps) and labels every seam at least `SEAM_COVER_TAU`
  (0.5) of whose cracks lie under it;
* scopes apply first (by uid), then traces (by uid), each overwriting -- so
  a scope widened *after* a trace does not erase it. This is the one
  deliberate departure from `resolve_slice`'s pure uid order.

After a persistence change the graph is rebuilt and the same gestures
re-resolve: a seam that merged into a longer one keeps its label while it is
still half covered, and one that vanished simply has no successor
(`tests/test_seam_labeling.py::test_traces_re_resolve_after_a_merge`).

## The tools

Both sit in the **Seams** cluster of the annotation pane (between the class
stack and the classifier) and in the Tool row of the store: `trace` (key
**T**) and `scope` (key **S**), painting the seam class picked beside them
(boundary / interior). `tools.TraceController` owns both; `DrawController`
dispatches to it before the armed-region-class test, since seam gestures
carry their own class.

**Trace** is a livewire (`seam_path.Livewire`). A press snaps to the nearest
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
| `edges` | the `-> edges` model's p(diff) for the flank pair | a `-> edges` model classified at this commit |
| `model` | the seam model's boundaryness | a trained seam model |

A toll whose input is missing refuses the press with the reason on the HUD
and the status line (the magic fill's "Classify first" idiom); a seam whose
pair has no arc (MSC mode drops some saddles) gets affinity 0 under the arc
tolls.

## Rendering

The seam layer rides the class stack (`panels/view.py::_seg_overlays`, after
the ground truth): `seams.seam_pixel_raster` paints the seam index on BOTH
flank pixels of every crack -- a 2-pixel line that survives the canvas's
nearest-neighbour zoom-out -- and a LUT colours it by resolved class
(`seam_class_lut`, class 0 transparent) or, with `colour: boundaryness` and a
seam model scored at this commit, by boundaryness on the labeler's ramp
(`seam_scalar_lut`). It goes through `_region_overlay`, so mspath places it
on the slide like every region layer; **E** toggles it. The raster is
rebuilt per commit and the classes per `(commit, store.rev)`
(`SeamPanelMixin._seam_raster_for` / `_seam_cache_for`), and the layer is
skipped entirely while nothing is labelled. Committed traces and scopes draw
as Tk items in the persistent annotation layer and on row hover; the trace in
flight, its hover leg and its anchors are the `"trace"`-tagged items the
controller re-projects on every pan/zoom.

## The seam model

`seam_model.py` mirrors `edge_model.py`. The **seam descriptor**
(`seam_features`) concatenates, per the `SeamSpec`:

* `pair`: `|d|` and the product of the two flanks' region descriptors -- the
  z-scored statistics rows (`embed = "features"`), or the base region net's
  last hidden layer when a dense classifier is trained (`"net"`, through
  `edge_model.embed`; `"auto"` picks it when available and the net's inputs
  are plain table columns);
* `barrier`: `edge_model.barrier` -- the saddle depth and the extremum gap;
* `edges`: the edge model's p(diff) for the pair, plus a present flag;
* `geometry`: crack length, its log, the bbox aspect, the tortuosity
  (length / chord) and whether the seam is a loop.

**Train seams** gathers every primed item's descriptors and resolved classes
(`SeamModelMixin._seam_items`, over `_stream_stat_slices`), fits a balanced
logistic regression (or a small MLP) behind a `StandardScaler` on the
labelled seams (both classes are needed: draw a scope and a trace), installs
it as `_seam_model`, scores every seam of every item (`_seam_pred`, per
commit) -- which is what the `model` toll and the boundaryness colouring read
-- and reports. **Evaluate** runs leave-items-out CV (`model_search.make_cv`
over `catalogue.group_of`) on the Optimize worker/pump (`seam_progress` /
`seam_done` events, Cancel stops after the current fold) and reports held-out
log-loss, balanced accuracy, AUC and boundary recall / precision. The model
rides the classifier pickle under its own `seam` key (`ModelBundle.seam`,
written only when set, so a seam-less pickle is unchanged), is restored with
it unless its pair block embedded with another base net, and the session's
`models[]` entry carries `"seam": true`. The readout under the buttons shows
the current item's gesture and seam counts and the model's brief. Saving the
classifier is what persists the seam model; a plain Train seams writes no
pickle.

## Export

**Export…** (`SeamModelMixin._export_seams_to`) writes, for every primed
item, `seams_<item>.json` -- `{item, shape, placement, classes, n_seams,
n_junctions, seams: [{id, a, b, j0, j1, length, class, class_name,
boundaryness, points}]}` with the points in image coordinates and collinear
runs compressed -- and one `seams_summary.csv` over all items
(`item, id, a, b, length, class, class_name, boundaryness`). That is the
input a later step rasterizes with a width to train an image model.

## Sessions

`view.seams` carries the toll, its channels, the show toggle, the colouring
and the picked class (`SeamPanelMixin._seams_view_state` /
`_apply_seams_view`); `view.tool` may be `trace` / `scope`. New session drops
the seam gestures (the annotations document goes back to v2) and keeps the
seam model with the model selection.

## Tests

`packages/mslabeler/tests`: `test_seams.py` (crack ids, the junction rule,
the canonical form, random rasters, snapping, the pixel raster, LUTs,
placement), `test_seam_labeling.py` (the store's seam list, JSON v3, scope
containment, coverage, precedence, re-resolution after merges, placement),
`test_seam_path.py` (tolls, the livewire's paths, reroutes, scope
restriction, legs, loops, collinear compression), `test_seam_model.py`
(descriptors, fit / predict / pickle, leave-items-out evaluation, the net
embedding; needs scikit-learn), `test_seam_export.py`, and
`test_compat_docs.py` (v2 byte-identity, the v3 fixture).
`packages/mscoupon/tests/test_seam_graph.py` certifies the C++ against the
numpy reference; `libs/core/tests/core_smoke.cpp` section 7 checks the C++ on
its own. Both labeler selftests drive the tools headlessly.
