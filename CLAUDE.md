# MSSeg — agent orientation

MSSeg is a Morse-Smale segmentation platform: a **portable core** library plus
thin **frontends**. It takes a floating-point volume, transforms it (FeatureJ /
diffg filters), computes a discrete gradient + Morse-Smale complex (MSCEER /
GInt), simplifies by persistence, and segments via graph-walking into a label
volume. (The old 2D pipeline became the `mscoupon` instance.)

It is a **monorepo of independently pip-installable distributions**: one group
can `pip install msseg-mscoupon`, another `pip install msseg-cellseg`, without
either build knowing about the other. Shared code is shared by the mechanism
appropriate to its kind (see below).

## Layout

```
libs/
  core/               portable C++ core (namespace msseg::): msseg_core + msseg_io
                      filter · compute (msc2d/msc3d) · graph · segment · workflow · io
                      guarded (if(NOT TARGET msseg_core)) so every package add_subdirectory's it
  render/             shared OpenGL 3D renderer lib (msrender) — OPTIONAL, desktop/Windows only
apps/
  msviewer/           generic core-level OpenGL debug viewer (links libs/render); native, not pip
cmake/                Dependencies · AddInstance (add_msseg_instance) · VendoredGL
ext/win64/            vendored GL headers/binaries for the viewer (VendoredGL uses CMAKE_SOURCE_DIR/ext)
packages/             one independently pip-installable distribution each:
  mscoupon/           "msseg-mscoupon": 2D TIFF-slice pipeline (lib + cli + pybind) + `mscoupon` CLI
  cellseg/            "msseg-cellseg":  3D fluorescent-membrane cell seg (lib + cli + pybind) + `cellseg` GUI
  msworkflow/         "msseg-workflow": generic JSON workflow runner (cli + pybind)
  msseg-viz/          "msseg-viz": pure-Python shared viewers (palette, merge-tree icicle) — universal wheel
  mslabeler/          "msseg-labeler": pure-Python labeler framework (viewer/annotation shells, canvas,
                      annotation store, region-graph tools, classifier training) — universal wheel
  msseg-meta/         "msseg": umbrella that depends on all of the above
CMakeLists.txt        dev "build & test everything" root (add_subdirectory libs/core + each package)
```

### How sharing works (two kinds of shared code)

| Shared code | Mechanism | In wheels? | Portable? |
|---|---|---|---|
| C++ **core** (`libs/core`) | source-shared: each package `add_subdirectory(../../libs/core)`, static-linked | yes (into each pyd/CLI) | yes |
| C++ **render** (`libs/render`) | source-shared, gated by `MSSEG_BUILD_VIEWER` (Windows-x64, OFF in wheels) | **no** | no (desktop) |
| Python **viz** (`packages/msseg-viz`) | its own pure-Python distribution; instance packages depend on it | it *is* a wheel | yes (universal) |
| Python **labeler framework** (`packages/mslabeler`) | its own pure-Python distribution (`msseg.labeler`); the labeler packages subclass its shells and implement its four protocols -- see [docs/labeler_framework.md](docs/labeler_framework.md) | it *is* a wheel | yes (universal) |

**Namespace:** every distribution ships `src/msseg/<name>/` with **no** top-level
`msseg/__init__.py` — `msseg` is a PEP 420 namespace, so `msseg.mscoupon`,
`msseg.cellseg`, `msseg.viz`, `msseg.workflow` coexist without any inter-package
dependency (except the viewer packages depending on `msseg-viz`).

Dependencies (diffg, MSCEER/GInt, TinyTIFF, nlohmann_json, pybind11) are pinned
via FetchContent (`cmake/Dependencies.cmake`), with a `MSSEG_DEPS_DIR` local
override for offline/HPC. Local checkouts to read as references:
`../MSCEER` (GInt + `msc_2d_lib`), `../../libraries/FeatureJ/diffg`.

## Build / test

**Dev build (everything at once).** Run inside a VS dev env
(`VsDevCmd.bat -arch=x64`) so `cl.exe`/`ninja` are found:

```bash
cmake --preset windows-msvc -DCMAKE_MAKE_PROGRAM=C:/Users/jediati/bin/ninja.exe
cmake --build --preset windows-msvc
ctest --preset windows-msvc            # core_smoke + mscoupon_tests + cellseg_tests
```

Add `-DMSSEG_BUILD_PYTHON=ON` for the pybind modules; `-DMSSEG_BUILD_VIEWER=ON`
needs the `ext/win64` GL binaries present. Presets `linux-gcc` / `hpc` build the
portable parts off Windows (viewer excluded).

**Per-package install (what a collaborator does).** Each package builds on its
own via scikit-build-core (`add_subdirectory`'ing `libs/core`):

```bash
pip install ./packages/msseg-viz ./packages/cellseg   # local dep first
pip install ./packages/mslabeler                       # the labeler framework (mscoupon depends on it)
pip install ./packages/mscoupon                        # -> `mscoupon` CLI
```

`import msseg.cellseg` / `import msseg.mscoupon` then work; the `cellseg` /
`mscoupon` console commands are the entry points. For the Linux/HPC recipe see
**[docs/dane_hpc_build.md](docs/dane_hpc_build.md)**.

## The one hard rule: the GInt firewall

MSCEER's `gi_*.h` (and `msc_2d_lib.h`) are C++11-era and compile **only** in
`libs/core/msseg/compute/msc3d.cpp` / `msc2d.cpp`. `GInt`/`msc_2d_lib` link
PRIVATE to `msseg_core`. Everything else crosses the boundary through the
plain-data `MscGraph` and the `Msc3D` API — never `#include "gi_*.h"` elsewhere.

## Extending MSSeg

To add a segmentation strategy, a new instance/package, or a core/filter stage,
read **[docs/adding_instances.md](docs/adding_instances.md)** — the
`add_msseg_instance` contract, the Python binding pattern, and the GInt gotchas
(include order, `INDEX_TYPE`/`INT_TYPE` are global macros, volume layout). A new
frontend is a new `packages/<name>/` with its own `pyproject.toml` +
`CMakeLists.txt` (mirror `packages/mscoupon`).

## Status

Restructured (this branch, `refactor/split-packages`) from the single `msseg`
wheel into per-package distributions: `libs/{core,render}`,
`packages/{mscoupon,cellseg,msworkflow,msseg-viz,msseg-meta}`, generic
`apps/msviewer`. Prior milestones: M1 (restructure + parity), M3 (3D MSC core +
`core_smoke`), M4 (python + wheels), M5 (generic runner), M6 (Windows viewer).
Pending: M2 (Linux/HPC parity). Note: the `cellseg` Python smoke test has a
pre-existing drift from the current cellseg output (the C++ `cellseg_tests` is
authoritative and passes).

**mscoupon cross-slice matching** (on by default, `--no-matching` /
`matching.enabled=false` to disable): after per-slice segmentation, a serial
in-order stage links kept 2D features into 3D features by 26-neighbor
connectivity between consecutive slices (union-find over `(slice, id)` nodes,
`libs`→`packages/mscoupon/lib/mscoupon/matcher.cpp`). Per-slice masks/CSVs are
unchanged; two derived files are written at the end — `feature_map.csv`
(`slice_index, segment_id → global_id`) and `global_segments.csv` (aggregated
master table, sorted by voxel count descending). The per-slice size threshold
still gates which features participate.

**mscoupon 2-point normalization** (`base_filters[]`, off by default): a coupon
stack's absolute intensity drifts slice to slice, so a raw threshold that is
right at the start of a scan is wrong by the end. Two landmarks per slice — low
(air/void) and high (metal/solid) — let a threshold be written once as a
normalized number: **`0.7` means `0.3*low + 0.7*high`**, resolved per slice.
Three measures produce the landmarks, all in portable C++ with pybind wrappers
(`fit_gmm` / `measure_histogram` / `measure_regions`): a 2-component Gaussian
mixture (`lib/mscoupon/gmm.cpp`), histogram peak finding (`histogram_peaks.cpp`),
and two hand-picked rectangles (`region_measure.cpp`). **`omit_value`** governs
the no-data mask everywhere: the sentinel to drop, defaulting to **0** (and to
**none** for regions, whose rectangles are chosen physical areas), with `null`
meaning "keep every pixel". It is a *value* rather than a flag because a stack
may pad with any constant -- one set pads with 43 -- and dropping the wrong value
leaves that plateau in the fit as a spurious population. The older boolean
`omit_zeros` is still honoured (true -> 0, false -> none); an explicit
`omit_value` wins. Comparison happens in the raster's own dtype, so a float32
image is matched against the sentinel rounded the way it was stored.

Normalization is modelled as a **filter stage on a channel**, not as a transform
on each threshold: `base_filters` preprocesses the base channel (the raster
statistics and pixel filters are read from) while `filters` builds the topology
field, both derived from the raw slice. Rewriting the channel once means every
downstream statistic is already normalized — so `std` scales correctly without a
per-field location/spread table, and per-slice sums merge correctly into 3D
features with no change to `matcher.cpp`. The map is affine and order-preserving,
so the MSC is provably unchanged (`test_normalize_preserves_msc_labels`), and no
pixel *value* is a sentinel anywhere (background is label `-1`), so shifting
zeros off zero is safe. `persistence_percent` is invariant under the map and
keeps its meaning. An empty `base_filters` reproduces the previous output
byte-for-byte. Python-side library: `src/msseg/mscoupon/normalize/` (the single
home for the mask/subsample/trim/percentile/peak helpers the ~13 one-off
`measure_*`/`calculate_*`/`plot_*` scripts used to each carry a copy of);
scikit-learn is now only a **test** dependency (`tests/gmm_parity.py`).

**mscoupon interactive viewer** (`mscoupon-gui`, Tkinter — see
[docs/mscoupon_gui.md](docs/mscoupon_gui.md)): browse TIFF sequences into
subsequences, chain filters, set persistence + manifold, prime each subsequence,
then live-drag a persistence slider and filter 3D features by statistics, and
export a `config.json` the CLI reproduces. Backed by a **two-phase statistics
pipeline** in the portable core: `msseg::Msc2DPipeline`
(`libs/core/msseg/compute/msc2d.cpp`) runs the MSC once, keeps the MSCEER engine
alive, and caches the base 2-manifold decomposition + per-manifold statistics, so
`select_persistence` re-thresholds cheaply via MSCEER's **native** cancellation
hierarchy (`setPersistence` + `ascending/descending2Manifolds` remap each base
extremum to its living representative — adjacent-basin merges, so every living
feature stays connected). It is the **authoritative** segmentation for BOTH the
GUI and the CLI (the batch pipeline uses `Msc2DPipeline` too), so an exported
config reproduces the viewer output. Filter chain (`filters[]`, incl. diffg
morphology `erode/dilate/open/close`) and the feature-query chain
(`feature_filters[]`, evaluated by the single-source `mscoupon::row_passes`) are
honored by the CLI; the GUI's on-the-fly 3D assembly
(`src/msseg/mscoupon/assembly.py`) mirrors the matcher's connectivity.

**Labeler framework** (`packages/mslabeler`, `msseg.labeler`, 2026-09-09, see
[docs/labeler_framework.md](docs/labeler_framework.md)): the viewer and the
labeler are bindings of a pure-Python framework. `ViewerShell` (window,
session browser, profiles, navigation, work-queue pump, session document) and
`AnnotationShell` (annotation store + undo, drawing tools, three columns,
classifier lifecycle in `classifier.py`, UI clusters as `panels/*` mixins) are
cooperative base classes: `MscouponApp(ViewerShell)` and
`LabelerApp(AnnotationShell, MscouponApp)`. Data and compute reach the
framework only through four protocols -- `ItemCatalogue` (items under opaque
string keys; coupon: slices as `"folder/basename"`), `RegionProvider` (records
`{commit, labels, stats, arcs}` on demand; coupon: `adapters.EngineRegionProvider`
over `ComputeEngine`), `ImageSource` (base pixels by level + region; the canvas
draws a tiled pyramid without holding it) and `LabelLayer` (region ids by
crop) -- plus `FieldConventions` for the statistics table's column names.
Headless pieces: `labeling`, `magic_fill`, `training.TrainingSetBuilder`,
`bundle.ModelBundle` (pickle v4 writer, v1-v4 reader, the compat gate),
`model_search`, `edge_model`, `torch_mlp`, `session_doc`. The old
`msseg.mscoupon.<module>` paths are shims that re-export the framework modules
wholesale -- classifier pickles name `msseg.mscoupon.model_search.FeatureSubset`
and `msseg.mscoupon.torch_mlp.TorchMLPClassifier`, so the shims must stay. One
behaviour change rode along: cross-validation groups are now one per slice for
both the region model and the edge model (`SequenceCatalogue.group_of`); the
region model used to group by sequence, which for a single sequence silently
fell back to plain stratified folds. Both selftests, the pytest suites and the
byte-level baselines (annotations.json round trip, canvas composite, classifier
pickles) were held identical through the extraction.

**Labeler tasks** (2026-09-21, `msseg.labeler.task`, stage 1 of
[docs/design_multi_model_tasks.md](docs/design_multi_model_tasks.md)): a
session holds several named detectors ("gland detector", "bubble &
background") over one corpus, exactly one active. A `Task` owns a
**workflow** (a profile NAME in the session's shared pool -- profiles stay
session-level so two tasks on one workflow share its primes), the class
vocabulary (count, colours, and now **names**, `LabelStore.names` /
`set_name`, written into `classes[].name` only when set), the `LabelStore`
with its undo history, the `ModelStack` (region clf + names/kind/spec/scope/
context, edge, latent, search winner, seam) with its prediction caches, the
saved-model records and the Model-tab view (`TASK_VIEW_KEYS`: `model_kind`,
`model_search`, `neighbours`, `context`). Identity is a `uid` (`t_` + 6 hex)
apart from the display name. **Every attribute the mixins, tools and
selftests read -- `store`, `models`, `_clf*`, `_pred`, `_seam_model`,
`_undo_stack`, `_models_dir`, ... -- is a property over `self._task`**
(`annotate.py`), so ~150 call sites are unchanged and a switch is one
assignment. `_activate_task` stashes the outgoing Model-tab view, clears the
`store.rev`-keyed caches (`_class_luts`, seam caches, `_ctx_cache`), switches
to the task's workflow via `_switch_profile` (no-op when shared), loads its
newest pickle **lazily** (the load gates against the ACTIVE workflow),
pushes its view and repaints; refused while a search worker runs (its finish
installs into the active task). `_switch_profile` stamps the active task's
workflow; profile rename/delete propagate; `_profile_from_model` binds via
`_bind_workflow`. Session document **v3**: `tasks[]` + `active_task`, no
top-level `annotations`/`models`; the tasks-less form is unchanged (the
viewers write it) and reads as ONE task named after the active profile with
the four view keys moved out of `view`; the reader returns
`annotations`/`models` as the active task's for older callers and now keeps
the model record's `context` key (a pre-existing drop). New session keeps the
tasks with emptied stores. UI: a Tasks `Treeview` (name / workflow / annot /
model) above the session lists with New / Dup / Rename… / Delete; the
profile box is titled "0. Workflow" (`PROFILE_SECTION_TITLE`); double-click
a class title to name it. Accepted for stage 1: a cross-workflow switch drops
primes; row removal dooms only the active task's gestures (others rebind
greyed on activation); `MAX_CLASSES` still bounds every vocabulary.
Tests: `test_task.py`, `test_class_names.py`, `test_session_doc_tasks.py`,
plus task sections in both labeler selftests.

**Slide-bound gestures** (2026-09-21, stage 2 of the same design note, §8):
a gesture is a statement about tissue at a location, so it is keyed by the
**slide** (`Interaction.slice_key` = `items.slide_id` in mspath; the coupon
slice is its own slide, unchanged), NOT by the item (`slide@level#rect`) it
was drawn on; every item covering the location -- an ROI, the overview, the
same rect at another level -- queries it by rect (`LabelStore.for_item`,
`gesture_extent`; `ItemCatalogue.binding_of(key) -> (slide, rect)`). The
shell's `_gestures_for_key` / `_gestures_for` / `_gesture_on_item` replace
every `for_slice(item key)` and `(it.si, it.li) == current` test (view,
class panel, seams panel, training/classifier, exports); the `(si, li)`
hints of a gesture name the slide's first row. Removing an ROI dooms no
gesture (they stay visible on the overview, back on a re-cut); removing the
slide dooms all. An older store's item keys **rebase** on load
(`rebind(rebase=catalogue.rebase)`: key -> slide, the item's level/scale
recorded in `meta`). Every new gesture records its **scale of intent**
(`_draw_meta`: `meta.level`, `meta.scale` = slide px per raster px,
`meta.px` = slide px per screen px; the coupon records nothing, so its
documents are unchanged) and the meta rule is amended: on the drawing level
geometry alone decides; off it, `px` keeps a stroke the width the user saw
(`stroke_mask`, PIL, applied always for new gestures when > 1.5 raster px),
`outline` (an extent's closed loops, `extents.py`: the seam graph of the
0/1 mask, Euler-chained, even-odd fill on a 2x centre grid) resolves a
magic fill / blob / SHIFT-accept by outline instead of seeds, and `scale`
switches a trace to a corridor (`seam_labeling.corridor_coverage`, r = the
coarser of the two pixels; exact crack ids on its own lattice). Also fixed:
`_accept_predictions` now goes through `_region_placement()`. A `!` on the
tree's annot cell and one notice per item flag gestures drawn >= 2 levels
coarser. mpp / physical units deferred. Tests: `test_slide_binding.py`,
`test_catalogue_binding.py`, `test_stroke_width.py`, `test_extents.py`,
`test_seam_corridor.py`; the mspath selftest's tree-rows block rewritten.

**Derived labels, extents and the enclosure** (2026-09-21, stage 3 of the
design note, §7): `msseg.labeler.derive` is the ONE label derivation --
gestures -> region classes (`resolve_slice`, unchanged), arc same/diff
(`arc_labels`) and seam boundary/interior (`seam_labels`), the §7.2 matrix
in three passes: samples (both flanks labelled -> boundary iff the classes
differ), **extents** (a lasso, a released magic fill, a blob's CORE, an
enclosure -- `meta["extent"]`, `derive.is_extent`; older fills with an
outline count, blob rings and accepted predictions never): one flank in the
extent and the other flank UNLABELLED -> boundary / different -- only
unlabelled neighbours, so filling a gland twice or patching a fill never
derives a boundary inside it and the seam and arc targets agree
(`EXTENT_EDGE_VS_SAME_CLASS = None` holds the instance-boundary reading);
then the explicit seam gestures overwrite (scopes -> same/interior, boundary
traces -> different/boundary). Consumers: `panels/seams._seam_cache_for`
(entry gains `[5] = derived`; the overlay and readout show derived labels,
the seam model trains on them) and `training.edge_set(arc_labels_of=)`
(`classifier._arc_labels_for`; `_classes_rc_sets_for` memoizes the touched
sets beside the row classes) -- gestures drawn `COARSE_LEVELS` coarser are
left out. Producers: blob core `extent=True`; magic fill by the `extent`
checkbox (Magic row 2, `magic_extent_var`, `view.magic.extent`, default
`_DEFAULT_MAGIC_EXTENT = True`); lasso `{"extent": True}` unless **Ctrl**
at press (`DrawController._sample`; Windows/X11 only). **Enclosure**: a
trace closed by clicking its first anchor commits at once and parks
(`TraceController._pending`; `derive.enclosed_ids` via `extent_mask`, exact
on the crack lattice where a polygon fill leaks); the next press inside with
a class armed calls `_commit_enclosure` -> `_commit_blob` with
`{"tool": "enclosure", "extent": True, "trace": uid}` (seeds + outline, one
undo step, row `enclosure (n) ext`); outside -> a new trace; Esc drops it. A
return leg after one leg retraces (Dijkstra is symmetric): two intermediate
anchors at least. `_commit_seam` now returns the Interaction. Tests:
`test_derive.py`, `test_edge_hook.py`; coupon selftest sections (extent
meta, checkbox + view, Ctrl-lasso, derived seams after a blob, the closed
trace / outside / arm-a-class / name / Esc flow; its overlay counts grew by
the derived seams layer).

**Places, enrolment and the fast path** (2026-09-24, stage 4 of the design
note, §5): an mspath ROI record is now a **place** (`{uid, level, x, y, w,
h[, note][, origin]}` on the slide; `level` = its cut level; `_clean_rois`
keeps uid/note/origin; `msseg.mspath.places` does the dict work), and a
task's **enrolment** says what it works: `Task.enrolled = {slide:
{"overview": None, "<uid>": level}}` (the level lives ON the enrolment, so
two tasks can work one place at two levels; None = every place at its own
level, the coupon's and a legacy task's reading, materialised on load in
mspath -- never including the overview; `{}` = nothing, what a new task and
the first task get; written by `Task.to_doc` only when not None,
`session_doc.normalise_enrolled`). The viewer hooks `_place_enrolled` /
`_place_level` sit behind `_enumerate_items`, the single choke point, so
flat_slices / catalogue / Train / Classify / exports / Run are the ACTIVE
task's (`ViewerShell.ENROLMENT`, mspath labeler True). **The overview is
never enrolled automatically**; adding a slide works nothing. A row the task
does not work is greyed (`_row_tags`) and **browsed** (`MsPathApp._browse`:
`slice_var = -1`, `_current()` None -- no gesture, prime or prediction can
land) and every place is outlined on the whole-slide view (solid + level =
the active task's, dashed grey = others'; `_redraw_place_outlines`, tag
`places`). Add-from-view / propose reuse a place with the same rect and
enrol it in the active task with `origin`; tree menu: Enrol / Unenrol, Work
at L<n> (refused, never shrunk, when degenerate or over budget), Note…,
Enrol every place (not the overview); removing a place removes it for every
task and forgets its keys at every level. `_activate_task` calls
`_enrolment_changed` (no prime on a switch) and keeps per-task caches by a
`(keys, rows)` signature stamped on leaving too. Run is **Run task** / **Run
all tasks** (`_prime_items`; the union over tasks on the active workflow).
**Fast path** ("retrain, classify what I'm looking at"): streams never prime
(`_stream_ready`), `C` / `R` classify the current item
(`_classify_current`), the button is *Classify all*, and mspath classifies
an item on arrival (`CLASSIFY_ON_ARRIVAL`). Tests: `test_enrolment_doc.py`,
`test_places.py`, the mspath labeler selftest's enrolment block.

**mscoupon labeler magic fill + gesture previews** (`mscoupon-labeler`, see
[docs/mscoupon_labeler.md](docs/mscoupon_labeler.md)): every drawing gesture now
previews the regions it WILL paint on a transient canvas layer (brightened class
color, ignores the alpha slider; `SliceCanvas.set_transient`), and a `magic` tool
grows a similarity flood from the pressed region over the **living-region
adjacency graph** -- `Msc2DPipeline::region_arcs()` wraps MSCEER's
`livingRegionArcs()` (pin `7cb4703`; pairs joined by a saddle, with its value,
in both `msc` and `merge_forest` modes), translated into the compact label-id
space; older extensions fall back to 4-neighbour pixel adjacency. All
seed-dependent work happens once per press as a *join ladder*
(`magic_fill.build_ladder`: metric -> row vectors -> arc weights -> priority flood
with a geometric hop gain, `hop×` default 1.1, g=1 being the pure bottleneck),
so a drag tick is a rank on the ladder -- a prefix of the flood's discovery order
(`growth_order`), NOT a threshold-closed set, because an outlier seed makes its
gateway neighbour's dissimilarity the bottleneck for most of the slice and a
threshold then jumps from one region to half of them -- and the threshold reads
in the data's own units on the HUD. Metrics: z-scored `mean` (default), `bhattacharyya`, saddle
`barrier`, `cosine` (the whole statistics row minus positions, z-scored) and
`proba` (total variation between classifier class probabilities; the press is
refused until the slice is classified); modes `anchor` (vs the seed) / `chain`
(vs the neighbour); the `drag` entry sets screen px per region. The fill
commits as ONE `taps` interaction with a point per region at its seeding
extremum plus a `meta` provenance dict, so it re-resolves after a persistence
change through the unchanged geometric path. The **blobber** (key B) is the same
fill plus a ring -- the core's immediate neighbours (`ring_for_rank`) in a second
class (`next` after the active one, or a fixed id) -- committed as two taps
interactions, ring first so the core wins on a merge. Escape abandons any
gesture in flight.

**mscoupon labeler layout** (2026-09-18: three columns, **data | picture |
tabs**, about **1:3:2**). Left is data navigation + processing *selection*
(profile dropdown, session, Run); the middle is the slice (`self.right`, still
a plain frame, so every viewer-area builder packs into it unchanged); the right
is a `ttk.Notebook` of everything you EDIT -- **Processing**, **Annotation**
(the class stack, the drawing tools and the classifier, which used to be the
window's third pane), **Model** and **Analysis**. **Only the middle pane
carries a weight**, so window growth goes entirely to the picture and the two
side columns keep the width they were given; the proportions are the SASHES,
placed explicitly from `_LABELER_PANES` and remembered as fractions in
`view.panes` (a weight would drift them). `F9` folds the tab column away.
The profile and the picture were sibling tabs until 2026-09-17, which made
judging a filter chain a round trip (edit, switch, squint, switch back); the
chain IS judged by looking, so the picture is never the thing that is hidden.
`_goto_region` no longer switches tabs, so the Analysis list stays open beside
the region it sent you to. **Processing is ONE column of collapsible groups**
(`widgets.Collapsible`, built through the `_group(parent, text, key)` hook --
a plain `ttk.LabelFrame` in the viewer's left column, a folding group in the
labeler's column; which groups are shut rides `view.proc_open`): two columns
needed a tab as wide as the window, where a column that folds needs only the
group being edited -- all five open is 844 px, all five shut is 135 px.
`MscouponApp` grew three
layout hooks (`_build_center`, `_profile_tools_parent`, `_processing_parent`)
whose defaults reproduce the viewer's tree exactly; the labeler overrides them
rather than forking `_build_left`. The session view keeps the selected tab
(`view.center_tab`) and the picked model kind (`view.model_kind`, restored after
the pickle reload, which would otherwise impose the saved model's kind). **New
session…** (`_new_session`, headless-callable with a `keep` dict; hooks
`_new_session_options` / `_new_session_doc` / `_after_new_session`) empties
folders, sequences, results and (labeler) annotations, keeping profiles and,
in the labeler, the model selection with the in-memory model stashed across the
apply; the old session is auto-saved first. Two more hooks (`_build_left_shell`, `_left_section_parent`,
`_session_group`) let the labeler drop the left scroll frame for a vertical
`ttk.PanedWindow` over the session's three lists with Run packed first at the
bottom. **Previews compute channels**: `_preview_channel` runs the base chain /
filter chain / `stat_channel_images` on the raw preview array (memoised per
path, channel and params), so the Image dropdown shows any derived field
before a Run; an unprimed sequence-tree row previews like a file-list click.
**A parameter edit repaints, without priming** (2026-09-17): a filter field used
to commit into `card["params"]` and notify nothing, so the canvas only caught
up at the next unrelated event. Every commit now calls
`ViewerShell._notify_profile_edit`, which settles for `_PREVIEW_SETTLE_MS`
(250 ms -- a sigma typed `1`, `.`, `5` is ONE edit) and then hands
`_launch_preview` the shown channel; a 400 ms `_preview_poll` over a
`_chain_fingerprint` is the backstop, because the chain cards are two
near-identical implementations (coupon and mspath) rebuilt from scratch on
every operation change, so a commit path that forgets to report itself must
degrade to a delay rather than to a dead control. The compute is the SAME calls
a run makes, moved off the Tk thread: `engine.preview_raster` is Tk-free and
cache-free, run by `msseg.labeler.preview.PreviewWorker` (queue + daemon thread
+ `root.after` pump + supersede token, modelled on the Optimize search) with a
`sync=True` path for the selftests; `_preview_channel` keeps its signature and
runs the same function inline for the callers that need a raster in hand. What
is *not* recomputed is the point: a channel's cache key IS its dependency set,
so editing `filters` while the dropdown shows `base` is one tuple comparison,
and retyping the old sigma is a cache hit. The preview stays live **after** a
prime -- `_paint_live` records a `_preview_override` and takes the region
overlays OFF (they are from a different field, so drawing them over this raster
would not be slightly stale but wrong) with a `Preview - filters changed, Run to
re-prime` badge that a Run or an undo clears. Zoom and pan survive, because
`set_base` never touches the viewport and the path key is held constant.
A derived statistics channel cannot depend on `filters` at all: a statistics
source is validated to be base or colour, and `build_stat_channels` reads the
filtered raster only for a channel whose *kind* is `filtered`, so a derived
preview skips the topology chain entirely (`engine._spec_reads_filtered`).
mspath gets the same loop, over `SlideEngine.read_item` -- the read is what
costs seconds at a deep level, so the array is cached per (item, level, halo)
and a chain edit does not invalidate it -- with the result placed by
`PlacedImageSource` at the item's own origin and scale, halo trimmed.
Two link-labels -- heading the Run section and the classifier section
respectively -- show the active workflow rendered by
`session.profile_summary` (two lines: `topo field: base→b(1.5)→e(0.7)→msc(asc, 10%)`
and `stats: base→norm(gmm)→12ch×4`, polled every 0.7 s from the panel) and the active model -- each a
click away from its tab.

**mscoupon labeler "Optimize network"** (`dense (tuned)` kind, key `O`, see
[docs/mscoupon_labeler.md](docs/mscoupon_labeler.md)): the dense FC classifier
was one fixed `(64, 32)` net fit with no held-out score. `model_search.py`
(headless, pytest-covered) describes a net as a plain-data `ModelSpec`, builds
its pipeline (`FeatureSubset` -> `StandardScaler` -> `MLPClassifier`, balanced
sample weights), scores it by **leave-slices-out** CV (`StratifiedGroupKFold` by
slice when >= 3 slices carry labels; mean held-out log-loss, since the
probabilities feed magic-fill `proba` and the uncertainty coloring) and searches
depth, widths, alpha, learning rate, batch, early stopping and a **per-channel
feature mask** -- Optuna TPE + median pruning when installed (`[optimize]`
extra), else seeded random search over the same space; trial 1 is always the
baseline so the winner never loses to it. The labeler runs it on a worker
thread drained by a `root.after` pump (Cancel keeps the best so far), installs
the winner via the `_install_model` tail Train also uses, and classifies. The
subset lives INSIDE the estimator, so the feature fingerprint, compat gate and
pickle/predict paths are untouched; the spec rides the pickle (v3, `spec`),
the session model record, and `view.model_search` carries the search settings.
**GPU backend** (`torch_mlp.py`, `spec.backend` = `torch`/`sklearn`, `auto` =
torch when importable, `[torch]` extra): `TorchMLPClassifier` is an
`MLPClassifier` drop-in (plus dropout, searched) and `train_stacked` trains
every CV fold of a trial as ONE batched computation -- `(F, in, out)` weights
via `baddbmm`, per-fold standardisation/early stopping/best-epoch snapshots
vectorised with `torch.where`, one host sync per epoch -- so a trial is one
training loop, not five; labels encode over the global class list so a fold
missing a class still yields its probability column. **Trial cost**: a
mini-batch step is ~1.5 ms of per-op overhead on either device, so specs with
`batch_size` < `TORCH_MIN_BATCH` (256) train on sklearn (`effective_backend`;
`spec.backend` keeps the request), trials run under a separate budget
(`SEARCH_MAX_ITER` 300 / `SEARCH_PATIENCE` 10; the winner is refit with the
full 1000 via `refit_max_iter`), and a torch trial reports its held-out loss
every `REPORT_EVERY` epochs through `train_stacked(monitor=...)` so Optuna can
prune. Defaults are overnight-sized (300 trials, 480 min, the entry in minutes,
the session in seconds) and `_finish_search` pickles the winner to
`<app_data_dir>/models/tuned_<stamp>.pkl` and records it, so a restore reloads
it. **Size sweep** (`run_size_sweep`, lower half of the Model tab): one
fixed-architecture search per rung of a ladder (`SearchSpace.fixed_hidden`
pins the size, trial 1 is that size at baseline settings, `spec.baseline_hidden`
records it), reported as a Treeview (params, CV log-loss, bal. acc, loss vs the
best rung, best settings) with `SweepResult.summary()` naming the best rung, the
smallest within `SWEEP_TOLERANCE` (5 %) and the breakdown rung; the best rung is
installed + saved, any rung is installable from the table, and the sweep is
written as `models/sweep_<stamp>.json`.
**Edge-pairs experiment** (2026-09-05, [docs/mscoupon_edge_pairs.md](docs/mscoupon_edge_pairs.md),
script `packages/mscoupon/experiments/edge_pairs.py`): a logistic pair model on the
16-8 net's 8-d layer tells same/different-class edges of the region graph better
than the net's own argmax (97 % / 94 % boundary recall / precision vs 93 % / 88 %),
and three rounds of neighbour voting cut held-out region errors 115 -> 66. Paths
forward are listed in the doc (real MSC arcs + saddles, a `learned` magic-fill
metric, a refine-with-neighbours action, joint training). **Landed** (2026-09-08)
as `edge_model.py` + the `-> edges` model kinds: `_clf` stays the BASE
pipeline, `_edge_model` (a balanced logistic on the base net's last hidden
layer: `|d|`, product, saddle barrier) sits on top; Train fits base then edges
(`freeze base` keeps the base); `_predict_slice` stores a 4-tuple `(commit,
final, proba, aux)` with per-arc `pdiff` aligned to the RECORD's arcs and
`raw`, voting runs in label space (`_vote_entry`), `N` flips kind <-> base
and re-votes from the cache (`_revote_all`), pickle v4 carries `edge` +
`stack`, the `learned` magic metric reads `aux["pdiff"]`, coloring modes
`flipped by neighbours` / `boundary p(diff)`, and `Evaluate edges` reports
leave-slices-out through the Optimize worker/pump; report rows are marked
held-out (the confusion matrix is a fit check on the training labels, not
held-out). An **Analysis** center tab holds the size sweep and the
predictions-vs-annotations list (`_confusion_cell_rows` / `_fill_error_list`: a
confusion cell's regions across slices; double-click -> `_goto_region`,
which uses `SliceCanvas.center_on`). The in-tree pyd was
refreshed from the build tree so records carry MSC arcs with saddles.

**Labeler context features** (2026-09-14, `msseg.labeler.context`, the
**Context** panel on the Model tab, [docs/mscoupon_labeler.md](docs/mscoupon_labeler.md)
"Context features"): input-side neighbourhood context for the region
classifier, as ORDINARY columns appended to the statistics row -- ring
reductions over a region's arc neighbours (`ring_mean/min/max/std`,
`ring_contrast` = own minus ring mean), a 2-hop mean, and per-item
`slice_mean/contrast` -- over a chosen source (all columns, `ext_*` only,
`mean_*` only) and an OPTIONAL neighbour weighting (`uniform` default,
`area`, `contact` = shared boundary length from the label raster, derived
lazily and cached on the arcs dict as `length`). Columns are named
`<kind>[<weight>]__<col>` (prefixes never collide with `mean_`/`std_`/`hist`),
`schema_entries` files each kind as its own Optimize mask group, and
`augment` returns a NEW table (`rec["stats"]` untouched, so the magic fill
and readouts see the raw one). No saddle value is read anywhere. Plumbing:
`_stream_stat_slices(action, spec)` yields the augmented table; the model's
own spec (`_clf_context`, pickle `stack["context"]`, model record `context`)
is what predictions and the compat gate rebuild (`_expected_names_for`),
the Model tab's picked spec (`view.context`) is the NEXT model's; both keys
exist only when non-empty, so an empty spec is byte-identical to before.
Two more rungs ride the same spec: the edge model's opt-in `contact` pair
feature (`log1p` shared boundary length; with `barrier` off the pair model
is saddle-free; `edge_model.DEFAULT_FEATURES` keeps the old three), and the
**latent ring head** (`ContextSpec.latent` = `LatentSpec`): after the base
is fit, `context.latent_columns` averages the ring's hidden-layer embeddings
(uniform/area/contact/`latent` softmax) and `ring_h0` adds the H0 shape of
region + ring in latent space (largest merge, ratio, own attach, component
count; degree-bucketed vectorised Prim), and a second net of the base's own
spec is fit on row ++ those columns and makes the prediction
(`_context_model`, pickle `stack["latent"]`, dropped on load unless
`names_hash`/`net_hash` match; NOT in the fingerprint). **Labels as context**
(`ContextSpec.labels` = `LabelSpec(dropout, seed)`, columns
`nbr_class__c<k>` + `nbr_class__any`, IN the fingerprint): the ring's
annotated-class fractions, own label excluded, training-time dropout via a
seeded rng from `_context_table(..., training=True)`; predictions depend on
the store, so `_predict_slice` and the `_rebuild_class_panels` override in
`AnnotationShell` (`_labels_changed`) clear `_pred` when `store.rev` moves.
`experiments/context_ablation.py` scores every variant (labels visible vs
hidden on the held-out slices).

**mscoupon extremum statistics** (`ext_x`, `ext_y`, `ext_base`, `ext_filtered`):
the per-slice selection chain can also ask about a region's **seeding critical
point** — the minimum for ascending manifolds, the maximum for descending — not
just aggregates over its basin. This separates a real void (whose well bottom is
genuinely dark) from a shallow dip inside metal with a similar `mean_base`. A base
manifold is one extremum's basin and every other pixel flows to it, so the seed is
just the pixel attaining `filt_min` (asc) / `filt_max` (dsc) — derived from the
labeling rather than `criticalPoints()`, which keeps it free of MSCEER node-id
semantics (serial vs partitioned) and always lands on a real pixel (a maximum is a
2-cell, so its native position is a half-pixel). A merged feature inherits the
**surviving** extremum, a direct `leaf_stats[living_id]` lookup rather than an
accumulation — which is why `ext_filtered` can sit above the merged region's
`min_filtered`: persistence, not depth, decides which minimum survives. `ext_base`
samples the BASE channel (i.e. post-`base_filters`, so it reads in normalized
units) at that pixel; `msc.extremum_sample_radius` (default 0) switches it to the
mean over the `(2r+1)²` window. The fields are queryable, offered in the GUI
dropdown, and appended to `{stem}_segments.csv`. `feature_filters[].field` is now
validated against the `feature_row` schema (`mscoupon::is_feature_field`) —
previously a typo silently excluded every feature.

**mscoupon statistics are a spec, not a fixed list** (`statistics` config block,
`msseg::StatsSpec`). The per-feature row is rebuilt on every persistence change
and marshalled across pybind, so field count drives GUI slider latency — what a
workflow does not ask for is not accumulated and does not become a field:

```jsonc
"statistics": {
  "channels": [ "base",                     // "filtered" opts its aggregates back in
                { "kind": "blur",    "sigmas": [0.7, 1.5, 3.0] },
                { "kind": "edges",   "sigmas": [0.7, 1.5, 3.0] },
                { "kind": "hessian", "sigmas": [0.7, 1.5, 3.0] } ],
  "reductions": ["mean", "min", "max", "std"],
  "extremum": true, "extremum_sample_radius": 0,
  "per_slice": { "quantities": ["area", "bbox_w", "bbox_h"],
                 "reductions": ["mean", "min", "max", "std"] }
}
```

Omitting it gives base-only aggregates plus the extremum. The **filtered
aggregates now default OFF** — `mean/min/max/std_filtered` had no reader anywhere
in the tree, and a config still naming one fails validation with the available
names listed (leaf `filt_min`/`filt_max` are still computed regardless: they
*define* the seeding extremum). `mscoupon.feature_fields(params_json)` returns the
schema, and the GUI dropdown is generated from it, so there is no hand-kept mirror
to drift — the previous `QUERY_FIELDS` literal survives only as a headless
fallback.

**A measurement channel is a scale-space response, not just base-or-filtered.**
A bare string names one of the two rasters the pipeline already builds; an object
names a **derived** channel (`blur`/`edges`/`gradmag`/`laplacian`/`hessian`),
whose `sigmas` list is a cross-product — so the block above is twelve derived
channels (`hessian` yields `hessian_largest_s1.5` *and* `hessian_smallest_s1.5`
per sigma) plus `base`, giving fields like `mean_blur_s0.7`,
`max_hessian_smallest_s3`, `ext_edges_s1.5`. Sigma renders with `%g`, so `3.0`
becomes `s3`. The point is discrimination the intensity alone cannot give: a real
void and a shallow dip inside metal can share a `mean_base` but not a curvature
signature across scales. This per-feature vector is meant to become the design
matrix for a later material/air model fit, which is why the row is columnar.

Derived channels are **measure-only** — `filters` is still the sole topology
field, and the seeding extremum is still located on it — and they are computed on
the **base** raster (post-`base_filters`), so a normalized workflow measures them
in normalized units. They resolve once per run
(`msseg::resolve_stat_channels`, `libs/core/msseg/workflow/stat_channels.cpp`) and
are materialized in ONE `diffg::apply_filter_bank` traversal that shares its
separable passes: ~1 s/slice for twelve channels on 3232², against ~4 s for the
MSC. `Msc2DPipeline::build` takes an optional caller-owned bank so the CLI's
connected-component stage measures the same rasters without a second traversal,
while the GUI passes none and frees them as soon as the per-manifold cells exist.

**Per-channel aggregates live in a flat side table, not in the stat structs.**
`msseg::ChannelStats` (`libs/core/msseg/compute/channel_stats.hpp`) is a
region-major `n_regions × n_channels` array of `{sum, sumsq, min, max}` plus the
per-channel value at the seeding extremum — the `PerSliceAcc` shape from
`matcher.cpp`, one dimension up. `Msc2DFeatureStat`, `CcNodeStat` and
`GlobalFeatureStat` keep only what is *not* a measurement (bbox, the extremum's
position, and `filt_min`/`filt_max`, which locate it whether or not `filtered` is
measured). Channels are addressed by **slot**, resolved once per run; nothing
hashes a string inside a pixel loop. Adding a statistic used to mean editing four
parallel structs, two hand-synced CSV column sequences and a numpy mirror — the
`relevance_base` change touched 17 files; twelve channels × four reductions is not
reachable that way.

**The per-feature row is columnar** (`mscoupon::FeatureTable`): the schema once,
then `n_rows × n_fields` doubles, row-major. `feature_schema()` returns
`{name, channel, reduction}` per column, so the GUI builds two-level
`[channel][reduction]` pickers without parsing `mean_blur_s0.7` — and without
mistaking `min_x` for the `min` reduction of a channel `x`. `pipe.feature_table()`
crosses pybind as `(names, (n, f) float64)` instead of a dict per feature, and
`compile_queries` resolves field names to column indices once per slice rather
than per feature. `global_segments.csv`'s header AND values now come from that one
projection, so they cannot drift; an empty derived list reproduces the previous
header byte-for-byte (`ext_base` still precedes `ext_filtered`). The per-slice
`{stem}_segments.csv` keeps its own spec-blind `SegmentStat` rescan — unchanged,
as `relevance_base` left it.

**3D features now carry a seeding extremum** (`ext_x/y/z` and `ext_filtered` on
`GlobalFeatureStat`, plus every channel sampled there in its `ChannelStats`). Each per-slice CC node derives its own
from its pixels — argmin (asc) / argmax (dsc) of the filtered field, the same rule
as 2D — rather than inheriting the MSC feature's: a CC node has no MSC identity
left (the mask is a union of kept features) and the pixel trim runs first, so an
inherited extremum could name a trimmed-away pixel. The cross-slice merge
**carries the whole tuple** from the most extreme constituent slice rather than
reducing each field independently, which would pair a position from one slice with
a value from another. Reductions over `field` stay voxel-pooled in 3D (a mean of
per-slice means weights by slice count, not area); `per_slice` reductions instead
run *across* slices, which is the only way `area`/`bbox_w/h` mean anything under
mean/min/max/std. `assembly.py` mirrors all of it so the GUI's 3D assembly agrees
with the CLI.

**mscoupon colour input** (RGB/RGBA TIFFs, CLI and GUI alike): a TIFF's samples
load as planar float32 planes (`msseg::read_tiff_planes` -> `InputSlice`;
TinyTIFF decodes chunky and planar files one sample at a time; 4 samples read
as RGBA and 2 as gray+alpha, the alpha dropped unless `input.color.alpha` is
`keep` -- TinyTIFF exposes no ExtraSamples, so alpha is inferred from the
count). Each chain reduces the planes to its scalar through a leading `color`
stage (`libs/core/msseg/filter/color_stage.cpp`): `pick`, `luminance`
(Rec.709), `weighted`, `mean`/`max`/`min`, `hsv`, `optical_density` (`i0` =
`"max"` | number | per-plane; an optional unit `stain` vector, `weights` or
`channels` projection; default the total OD), and diffg's multi-channel
`chgradmag`, `dizenzo` and `structure`. It is the ONLY stage that consumes
planes, so it is valid at index 0 only, and each chain picks its own -- the
topology field can run on Di Zenzo edge strength while statistics read OD. A
multi-plane input whose chain has none gets `input.color.default_method`
(luminance). One plane and no colour stage is the scalar chain exactly
(byte-identical outputs, gated on two grayscale slices with normalize +
derived channels + matching). The GUI loads through the same reader
(`common.load_slice`, Pillow fallback for compressed files), renders a planar
base as RGB, offers `color` / `color_c<i>` in the Image dropdown, and a `color`
card at the head of a chain with per-method rows; the profile carries
`input.color` and the params JSON gets `input.color.channels` from the slice on
screen. **Colour statistics sources**: `statistics.channels[]` takes the bare
string `"color"` (the raw planes, `color_c0..`) and `source: "color"` on a
derived request -- per-channel kinds then emit one response per plane
(`blur_c0_s1.5`, `hessian_largest_c1_s1.5`, plane-major to match diffg's bank
order) and the cross-channel kinds `chgradmag` / `dizenzo` reduce over the
planes (they require that source). The plane count is a declared config fact
(`input.color.channels`, 3 when a colour source is named but nothing is
declared) so the schema resolves with no raster in hand; every loaded slice is
checked against it. `build_stat_channels` runs a second bank traversal over the
planes; colour-sourced specs stay on the CPU statistics path (diffg's GPU bank
is single-channel). Per-channel kinds on colour multiply the derived bank by C,
so the GUI keeps a slice's planes in the file's integer dtype.

**mscoupon histograms** (`statistics.histogram`, the first vector statistic):
`{"bins": 16, "channels": ["base"], "ranges": {"*": [0, 1]}}` adds K
equal-width bins over a FIXED range per channel (a name or `"*"`), opt-in per
channel because the counts live per base manifold (a 3232² slice can have ~1e6).
The range is fixed by config rather than measured per slice on purpose: the
bins must add across slices (the matcher and `assembly.py` merge bin-wise with
`np.add.at`) and a bin must mean the same thing on every slice a classifier is
trained over; out-of-range values clamp into the end bins so the counts always
sum to the area, NaN is skipped. Counts are a uint32 side table of
`ChannelStats` (add / saturating merge / append / clear), projected as
`hist<kk>_<channel>` (reduction `hist<kk>`, zero-padded) AFTER the extremum
block as bin fractions, so every previous column order is a prefix and
`global_segments.csv` picks them up through the schema. The GUI panel's
"measure" fills the `"*"` range from the slice on screen; the summary reads
`+16h×2`. Downstream: the classifier's per-channel feature groups keep a
channel's bins together (`model_search.feature_groups`), the magic fill gains a
`histogram` metric (Hellinger over the concatenated bin fractions) and `cosine`
leaves the bins out, and switching histograms on widens the field set so saved
models are invalidated by the compat gate, as designed. Histogram specs stay on
the CPU statistics path. Gradient-orientation histograms are designed, not
built: `docs/design_orientation_histograms.md`.

**Seam labeling** (2026-09-15, [docs/seam_labeling.md](docs/seam_labeling.md)):
boundaries between regions are first-class annotations. Vocabulary: a
**seam** is a junction-to-junction chain of **cracks** (unit steps between
pixel **corners** that separate two differently labelled pixels) between the
same two **flanks**; a **junction** is a corner where >= 3 regions meet or a
seam ends; a region's statistics row is its **descriptor**; "edge" keeps its
existing meanings (the `edges` channel, the region-pair edge model). Seams are
traced from the label raster's crack graph -- `msseg::extract_seam_graph`
(`libs/core/msseg/graph/seam_graph.cpp`, no MSCEER; `mscoupon_py.seam_graph`)
with a canonical output the numpy reference `msseg.labeler.seams` reproduces
array for array (`packages/mscoupon/tests/test_seam_graph.py`), reached
through `RegionProvider.seams(key, np)` -- NOT from MSCEER arc geometry: the
crack lattice is segmentation-independent, so a stored **trace** re-resolves
against a new decomposition by crack coverage (`seam_labeling.resolve_seams`,
tau 0.5; scopes first, then traces, by uid). Tools (`tools.TraceController`,
keys T / S / E / Enter / BackSpace): a **scope** box labels every seam inside
it interior; a **trace** is a livewire (`seam_path.Livewire`: virtual anchor
on any seam point, ONE Dijkstra over the junction graph per anchor, hover =
predecessor walk) whose **toll** is `geometric` / `feature` / `bhattacharyya`
/ `barrier` / `edges` (edge-model pdiff) / `model`. Seam gestures live in
`LabelStore.seams` and serialize under `"seams"` with `"version": 3` only
when present (a seam-less store is byte-identical to v2). The **seam model**
(`seam_model.py`, `seam_classifier.SeamModelMixin`: pair terms on the flank
descriptors or the base net's embedding + saddle barrier + pdiff + geometry,
balanced logistic, leave-items-out Evaluate on the Optimize pump) scores
every seam's **boundaryness** (the `model` toll, the boundaryness colouring)
and rides the classifier pickle under `ModelBundle.seam`; **Export** writes
`seams_<item>.json` + `seams_summary.csv`. The overlay paints both flank
pixels of every crack through `_region_overlay` (mspath places it).

**The MSC and the statistics are cached apart** (2026-09-24, first half of
stage 5 of the multi-model design note): `Msc2DPipeline::build` is the MSC
phases then ONE measure step (`measure_leaves`), and
`Msc2DPipeline::remeasure(base, filtered, cfg, color)` / pybind
`pipe.remeasure(params_json, base, filtered, color=None)` re-runs only that
step on a built pipe -- labels, arcs and persistence kept, rows equal to a
fresh build's (`test_msc2d_remeasure`, `test_gpu_stats_parity.py`). Two
fingerprints over the params document (`msseg/mscoupon/fingerprints.py`,
`session.field_fingerprint` / `measure_fingerprint`): **field** = chains +
colour input (minus plane count) + MSC keys (minus cores / builder / GPU
flags / sample radius; the percentage only as the cap `max(10, pct)`);
**measurement** = `statistics` + sample radius + plane count. The coupon
engine stamps `measured[]` per primed slice and re-measures lazily in
`_slice_result`; `SlideEngine` stamps `Primed.field` / `.measured` /
`.chains` and re-measures in `ensure_record` (navigation: on the worker,
`start_run(remeasure_only=True)`), re-reading through the chains the pipe
was PRIMED with (an un-Run edit is a preview). A Run keeps what the field can
reuse (coupon per sequence, mspath `keep_field`); a profile / task switch
resets only when the field differs (shell `_keep_compute_for`). The labeler's
statistics and ext sample radius moved to a **Features** tab; a settled edit
re-measures the item on screen. 2048² ROI: base-only -> 12 channels 0.90 s vs
a 1.47 s prime.

**Layered cache keys** (2026-09-24, second half of stage 5): a record's
`commit` is its **identity** -- the interned id (`msseg.labeler.record_keys`:
`Interner`, `RecordCache`) of what it is a function of: mspath `(item, prime
id, measurement, persistence)`, coupon `(stack_gen, _selection_snapshot())`.
Never a parameter hash alone (region ids differ per prime). Every framework
cache already compared `entry[0] == rec["commit"]`, so nothing downstream
changed, and a task switched back to finds its records AND predictions.
Records are kept a few per item (`MSSEG_RECORDS_PER_ITEM`, 4; mspath shares
one label raster across measurements; the coupon's `_ByCommit` maps read as
the current record and keep 2 assemblies). mspath pipes live per **field
slot** (`SlideEngine.use_field`; `primed` / `level_range` /
`persistence_abs` are properties over the active slot; one LRU across all;
pins per field) and `start_run` takes `(item, profile)` jobs, so **Run all
tasks** primes every task under its own workflow. The coupon keeps one field
live. **The base chain is a measurement** (fingerprints, both re-measures --
the coupon re-reads the slice file -- and the Features tab); a settled edit
goes through `AnnotationShell._preview_edit_settled` -> the app's
`_measurement_moved()` -> `_remeasure_current`, a field edit stays a preview.

**Max-area simplification rule** (2026-09-24, MSCEER pin `898fd95`,
`ComputeOptions::rules`): `msc.max_region_area` (pixels; null / <= 0 = off)
vetoes every cancellation / forest merge that would grow a region of the
profile's `manifold` past the cap, at every persistence (a base basin larger
than the cap is not split); `msc.max_region_parallel` (default true) keeps the
builder parallel -- best effort: the banded forest still equals serial, the
partitioned MSC holds the cap but may pick other merges; false forces a serial
build. Both modes, CLI + pybind + both GUIs' MSC panel (`build_max_area_row`
in the coupon `app.py`, reused by mspath); emitted into the params only when
on, so capless profiles prime (and fingerprint) as before; both keys are
FIELD keys. Summary: `msc(asc, 10%, mf, ≤5000px)` (`*` = serial). Tests:
`test_msc2d_max_region_area` (C++), `test_max_region_area.py`.

**Stage strip** (2026-09-24, labelers only, `msseg.labeler.panels.stages`):
the top-left of the canvas shows `[msc]--[stats]--+--[classified]` with
`[model]` joining below, for the current task x item, COMPUTED on every
`_update_busy` / navigation / store edit / model install / prediction and the
0.7 s hint poll -- never set by the code that changes state. The app judges
msc and stats (`_stage_field` / `_stage_measure`, from the engines' field and
measure fingerprints, record ids, running keys; mspath adds `cached` for a
released pipe with a kept record and `error` from `item_error`), the
framework judges model (compat gate cached per measure key;
`ModelStack.trained_rev` / `trained_measure` stamped by `_install_model`,
None for a loaded pickle) and classified (`pred[key][0] == rec["commit"]`);
`propagate` makes what is downstream of stale / busy / error stale.
`SliceCanvas.set_stages` draws it (a press on a box is claimed before tools
and pan; hover shows the tip), a click opens the box's tab
(msc Processing, stats Features, model Model, classified Annotation), and
`_compute_badge(text, stage=)` spins a box instead of the HUD
(`STAGE_STRIP` makes the apps' `_update_busy` leave stage states to the strip;
the viewers are unchanged). Tests: `test_stages.py`, the canvas strip test in
`test_sources.py`, strip blocks in both labeler selftests.

**Region encoder, offline** (2026-09-15, [docs/design_region_autoencoder.md](docs/design_region_autoencoder.md)):
a task-free latent of the statistics ROW, so the labeler's head is not the
only thing that ever compresses it. `mspath-embed harvest --tiff-folder DIR
--process-profile P.json --out H/ [--level 4]` primes slides tile by tile
through `SlideEngine.prime_item` (tiles ranked by tissue, the per-level
persistence pin written after the first tile and restored on resume, every
tile recorded at `--factors 1,0.5,2` of the profile's persistence) into
resumable `.npz` shards of rows + arcs; `mspath-embed train H/ --out E.msenc`
(`msseg.labeler.embedding_train`) fits `n_row -> 64 -> 32 -> d` by InfoNCE
whose positive is a random walk over the region arcs (spatial coherence as
the label-free signal), plus a reconstruction head and a VICReg var/cov
penalty, with schema-group dropout and marginal corruption as augmentation;
`--arch pca` is trial zero. The `.msenc` bundle (`msseg.labeler.embedding.
EncoderBundle`) applies with numpy alone and exposes the latent as
`emb<hash8>__z..` columns so the name-set compat gate needs no new logic.
NOT yet wired into the labeler, and the transfer probe is unbuilt.
