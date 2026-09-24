# The labeler framework (`msseg-labeler`, `msseg.labeler`)

`packages/mslabeler` is the pure-Python framework the MSSeg labelers are built
on. `mscoupon-gui` and `mscoupon-labeler` are its first bindings; a
whole-slide pathology labeler (gigapixel RGB, tiled TIFF pyramids, compute on
demand per ROI, annotations rendered to per-pixel masks) is the derivative it
was extracted for. Nothing here knows what a slice, a coupon or an MSC is: the
data and the compute reach the framework through four protocols, and every
domain decision is a hook with a documented default.

```bash
pip install ./packages/msseg-viz ./packages/mslabeler ./packages/mscoupon
# from a source checkout (all three namespace packages on the path):
PYTHONPATH="packages/mslabeler/src;packages/mscoupon/src;packages/msseg-viz/src"
pytest packages/mslabeler/tests packages/mscoupon/tests
```

The wheel is universal (hatchling, `src/msseg/labeler/`, no top-level
`msseg/__init__.py`); `numpy` and `pillow` are its only hard dependencies.
Extras: `classify` (scikit-learn), `optimize` (+ optuna), `torch`, `pyramid`
(large-image). `msseg-mscoupon` depends on it and forwards those extras.

## Module map

| module | holds | Tk? |
|---|---|---|
| `protocols.py` | `ItemKey`, `ItemCatalogue`, `RegionProvider` (+ `RegionRecord`), `ImageSource`, `LabelLayer`, `FeatureTableLike` | no |
| `fields.py` | `FieldConventions` -- the statistics table's column names (`DEFAULT` = the coupon schema) | no |
| `labeling.py` | `LabelStore` / `Interaction`: gestures in image coordinates, rasterized against a label raster on demand; the LUT builders | no |
| `magic_fill.py` | the region-graph flood (join ladder, hop gain, ring), metrics over a `FeatureTableLike` | no |
| `seams.py` | the seam graph -- the crack polylines between regions: `SeamGraph`, the numpy reference of `msseg::extract_seam_graph`, snapping, the pixel raster, LUTs (see [seam_labeling.md](seam_labeling.md)) | no |
| `seam_labeling.py` | scopes and traces resolved against a seam graph by crack coverage (exact ids on their own lattice, a corridor off it) | no |
| `derive.py` | **the one label derivation**: gestures -> region classes, arc same/diff, seam boundary/interior (the §7.2 matrix of the design note: samples, then extents' unlabelled neighbours, then the explicit seam gestures); `is_extent`, `enclosed_ids` | no |
| `extents.py` | a region set's outline as closed loops (the seam graph of the 0/1 mask) and the even-odd fill -- how a fill / blob / enclosure crosses a level | no |
| `seam_path.py` | tolls and the livewire (one Dijkstra over the junction graph per anchor) | no |
| `seam_model.py`, `seam_export.py` | the seam model (boundary vs interior from a seam descriptor) and the classified-seam export | no |
| `training.py` | `TrainingSetBuilder`: annotations + tables -> `(X, y, groups, names)` and the edge model's arrays | no |
| `context.py` | `ContextSpec` + `augment`: neighbourhood columns (ring / 2-hop / per-item reductions over the region graph, uniform / area / contact weighting, contact lengths from the label raster) appended to a table | no |
| `bundle.py` | `ModelBundle`: the classifier pickle (v4 writer, v1-v4 reader) and `compat_message` | no |
| `model_search.py`, `edge_model.py`, `torch_mlp.py` | the dense-net search, the pair model + voting, the GPU MLP | no |
| `table.py` | `FeatureTable` (the columnar per-region table) | no |
| `session_doc.py` | the session document (folders, sequences, profiles, view) and the session-file I/O | no |
| `sources.py` | `ArrayImageSource`, `ArrayLabelLayer` (+ `PyramidImageSource` re-exported) | no |
| `pyramid.py` | `PyramidImageSource` over a tiled pyramid: the OpenSlide / large_image / tifffile backends, the tile LRU | no |
| `canvas.py` | `SliceCanvas`: zoom/pan, an `ImageSource` base, `LabelLayer` overlays through LUTs, the transient preview layer, the HUD | yes |
| `widgets.py` | `ScrollFrame`, `jump_scale`, `scrolled_listbox`, tooltips | yes |
| `shell.py` | `ViewerShell`: window, session browser, profiles, navigation, pump, session flow | yes |
| `annotate.py` | `AnnotationShell`: the labeler layer (store, tools, hotkeys, the three columns and their tabs, session additions) | yes |
| `classifier.py` | `ClassifierMixin`: Train / Classify / Optimize / size sweep / edge evaluation | yes |
| `seam_classifier.py` | `SeamModelMixin`: Train seams / Evaluate / Export, the seam model on the classifier pickle | yes |
| `panels/` | `HintsMixin`, `ModelPanelMixin`, `AnalysisPanelMixin`, `ViewControlsMixin`, `ClassPanelMixin`, `SeamPanelMixin` (the Seams cluster, its caches and overlay) | yes |
| `tools.py` | `DrawController`, `MagicFillController`, `TraceController` (trace + scope), `_extremum_points` | yes |
| `defaults.py` | tunables and vocabularies (alphas, coloring modes, model kinds, search budgets, tool options) | -- |

`import msseg.labeler` imports only the protocols; the headless modules never
import tkinter or scikit-learn at module level, so training and bundle code
runs in a worker process or on a machine without Tk.

## Composition

```
class MyViewer(ViewerShell):            # the data + compute binding (mscoupon: MscouponApp)
class MyLabeler(AnnotationShell, MyViewer)   # the labeler on top (mscoupon: LabelerApp)
MRO: MyLabeler -> AnnotationShell -> [HintsMixin .. ClassifierMixin] -> MyViewer -> ViewerShell
```

The shells are cooperative base classes: every override calls `super()`, so
`_view_state`, `_session_doc`, `_apply_session_doc`, `_refresh_render`,
`_seg_overlays`, `_on_hover`, `_handle_event`, `_goto_slice`, `_build_left`
and the New-session hooks compose along the chain. The UI clusters are mixins
rather than delegation objects: state (Tk variables, the store, the caches,
`_clf*`) stays on the app, so a derivative reaches it as `self.<name>` and a
selftest can too.

`ViewerShell.__init__` order is load-bearing: data model -> `_init_compute()`
-> catalogue + region provider -> `_init_variables()` -> the shell's variables
-> toolbar -> paned window -> `_build_left_shell()` -> `_build_center()` ->
`_build_left()` -> `_build_right()` -> `_after_layout()` -> the chain
fingerprint + preview poll -> status bar -> initial folder -> auto-save.
`AnnotationShell.__init__` sets the labeler's state first (the base constructor
calls the overridden builders), then chains. Note that `_build_left()` builds
the parameter cards *before* `_build_right()` makes the viewer, which is why
every edit-driven path guards on `self.viewer is None`.

**A compute parameter was edited.** The filter chain is judged by looking at
what it produces, so a field commit has to repaint -- without priming. Each
commit calls `_notify_profile_edit()`, which debounces `_PREVIEW_SETTLE_MS`
and then calls the app's `_launch_preview()`; `_preview_poll()` re-snapshots
`_chain_fingerprint()` every `_PREVIEW_POLL_MS` as a backstop, because the
chain cards are two near-identical implementations rebuilt from scratch on
every operation change and a commit path that forgets to report itself must
degrade to a delay rather than to a dead control (the same reasoning that made
the workflow hint a poll). `_apply_profile_to_ui_quietly` mutes the signal
while a profile or session lands on the widgets, which would otherwise fire a
burst of chain recomputes for a chain nobody touched.

`preview.PreviewWorker` runs the compute: a daemon thread, a `queue.Queue` the
Tk thread drains on a `root.after` pump, a `threading.Event` the callable
checks between stages, and one live submission at a time (a newer submit
supersedes the older, whose result is dropped rather than painted). It takes a
callable and knows nothing about filters. `submit(..., sync=True)` runs inline
and pumps once, which is how the selftests drive it. It never touches a cache:
the callable returns arrays and the pump stores them, so the raster LRUs stay
single-threaded (they are bare `OrderedDict`s whose reads reorder them).

## The four seams (`protocols.py`)

| protocol | what the framework asks | coupon implementation |
|---|---|---|
| `ItemCatalogue` | `keys()`, `key_of(*index)`, `index_of(key)`, `label(key)`, `group_of(key)`, `tree()` | `adapters.SequenceCatalogue`: items are slices, key = `"folder/basename"` (what `annotations.json` has always stored), address = `(si, li)`, group = the flat slice index (leave-slices-out CV) |
| `RegionProvider` | `commit`, `keys()`, `record(key)`, `ensure_record(key)`, `request(key)`, `pending()`, `poll()`, `arcs(key, np)`, `seams(key, np)`, `label_layer(key)` | `adapters.EngineRegionProvider` over `ComputeEngine`: cached per-slice records, synchronous compute through `ComputeEngine.ensure_slice`, MSC arcs or a cached pixel-adjacency fallback, the seam graph (C++ `seam_graph` or the numpy reference) cached on the record and, in mspath, placed on the slide |
| `ImageSource` | `levels`, `channels`, `level_shape`, `level_scale`, `best_level(scale)`, `value_range()`, `read_region(level, x, y, w, h)` | `sources.ArrayImageSource` (one level, native floats) and `pyramid.PyramidImageSource` (a tiled pyramid: the file's own levels and dtype, colour kept, reads assembled from a byte-budgeted tile LRU) |
| `LabelLayer` | `shape`, `n_ids`, `rev`, `crop(level, x, y, w, h)`, `id_at(x, y)`, `full()` | `sources.ArrayLabelLayer` over the record's int32 raster |

A `RegionRecord` is `{commit, labels (int32 raster, -1 = background), stats
(FeatureTableLike), arcs ({"a","b","saddle"|None,"source"} or None), kept, cc}`.
`commit` is the parameter generation: every cache (class LUTs, predictions,
pixel adjacency, the seam graph and its raster) keys on it, and a record
whose commit is not the provider's is stale.

`FieldConventions` names the table's columns: the id column, the positional
set (never features), the mean/std prefixes, the histogram-bin pattern, the
extremum's position and value columns. The magic fill, `TrainingSetBuilder`,
`_extremum_points` and the shells take it as `conv` / `self.FIELDS`.

## The shell hooks

`ViewerShell` (identity: `SESSION_APP`, `APP_TITLE`, `WINDOW_TITLE`,
`LOG_PREFIX`, `SESSION_IO`):

| hook | default | coupon |
|---|---|---|
| `_default_profile(name)` | `{"name": name}` | `session.default_profile` |
| `_make_catalogue()` / `_make_region_provider()` | required | the adapters |
| `_init_compute()` / `_init_variables()` / `_after_layout()` | no-op | cards + engine / the coupon Tk vars / channel picker |
| `_build_processing_sections()` / `_build_run_section()` | none / a Run button | sections 1b-5 / cores + Run |
| `_build_segmentation_controls(row)` / `_build_live_panel(parent)` | none / slice nav + window + alpha | seg radios + mask / persistence, queries, pixels, connectivity |
| `_bind_preview()` / `_preview_file(path)` / `_list_files(folder)` | none / none / TIFFs | the preview machinery |
| `_launch_preview()` | no-op | recompute and repaint the shown channel from the edited chain, off the Tk thread (coupon: `_preview_plan` + `preview_raster`; mspath: `SlideEngine.read_item` + a `PlacedImageSource`) |
| `_group(parent, text, key)` | a `ttk.LabelFrame`, packed | the labeler returns a `widgets.Collapsible` body instead -- its Processing tab is one tall column, where a group you are not editing is only in the way. Both return the frame the rows pack into, so the apps' `_build_processing_sections` reads the same either way. |
| `_DEFAULT_PANES` | `()` | the labeler's `1:3:2`. Fractions of `self.paned`'s width, one per sash, placed by `_schedule_panes` once the window has a width and reported by `_pane_fractions`. Only ONE pane carries a weight (the viewer area), so a resize goes entirely to the picture and the sashes hold the proportions. |
| `_slice_msc_mark` / `_slice_nav_text` / `_enumerate_items` | "" / the key / provider keys | primed marks / `folder/base` / primed slices |
| `ITEM_NOUN` / `_row_kind(si, li)` / `_row_description(si, li)` / `_remove_target(si, li)` / `_row_owns_key(si, li, key)` / `_goto_row(si, li)` | "slice" / sequence-or-item / a phrase / the row itself / the catalogue's keys / navigate-or-preview | the defaults (mspath: "item", slide/overview/ROI, the overview redirects to its slide, a slide row owns every key on the slide, go-to also fits the item) |
| `_original_channel()` | `"base"` -- what the `F` swap flips back to; `_window_for(channel)` (call after the source is on the canvas) hands out each channel's brightness window, first measured by `windowing.source_window` | `"color"` when the slice has planes / mspath `"slide"` |
| `_remove_item_at(si, li)` / `_remove_sequence_at(si)` / `_after_rows_removed(cur_key, pos)` | edit `subsequences` / rebuild + retree + renavigate | also `engine.drop_slice` / `drop_sequence` (mspath: `forget` / `forget_slide`, the ROI hint, a cleared canvas) |
| `_reset_compute()` / `_settle_controls()` / `_update_busy()` / `_refresh_render()` / `_handle_compute_event(ev)` | minimal | the engine, Rerun/Run state, HUD, the render, primed/assembly events |
| `_run_settings()` / `_apply_run_settings` / `_apply_view_state` | `{}` / none / none | cores + concurrency / the coupon view keys |
| `_session_doc_from_json` / `_import_legacy_docs` / `_profile_to_file_doc` / `_profile_from_file_doc` / `_profile_from_ui` / `_apply_profile_to_ui` | passthrough | `session.*` |
| `_session_doc_kwargs()` / `PROFILE_SECTION_TITLE` / `_left_section_parent("tasks")` | `{}` / `"0. Compute profile"` / `self.left` | the annotation shell passes its `tasks` + `active_task` (the document becomes v3), titles the box `"0. Workflow"` and packs the Tasks list above the session lists |
| `_draw_meta(si, li)` / `_annotation_mark(si, li, count)` / `_sequence_annotation_count(si, counts)` | None / `str(count)` / `sum(counts)` | mspath records `{"level", "scale", "px"}` on every new gesture (its scale of intent); the labeler appends `!` when an item shows gestures drawn `COARSE_LEVELS` (2) or more levels coarser and counts distinct gestures per sequence (a gesture inside an ROI is seen by the ROI and the overview) |
| `ENROLMENT` / `_row_tags(si, li)` / `_enrolment_changed()` / `_normalize_enrolment(task)` | False / `()` / no-op / no-op | mspath: a task works only what it enrols (`Task.enrolled`); rows it does not are greyed (`UNENROLLED_TAG`) and *browsed* (no item current); the switch rebuilds navigation without priming; a legacy task materialises to its places. **`catalogue.keys()` is therefore per task** in an enrolment app. |
| `_stream_ready(key)` / `CLASSIFY_ON_ARRIVAL` / `_classify_current()` | True / False / -- | the fast path: model operations never prime (mspath skips uncomputed items and says how many); `C` / `R` classify the item on screen, *Classify all* the rest; mspath classifies an item when it is selected |
| `_prime_items(scope)` (mspath) | every listed item | the labeler's **Run task** (active task's items) / **Run all tasks** (union over tasks on the active workflow) |
| `ItemCatalogue.binding_of(key)` / `rebase(key)` | (protocol) | coupon `(key, None)` / None; mspath `(slide, rect)` / `(slide, level, scale)` for an item key -- and `index_of` resolves a bare slide key to the slide's first row |

**Tasks** (`task.py`, `AnnotationShell`): a session holds several named
detectors and one is active. Every attribute the mixins, the tools and the
selftests read -- `store`, `models`, `_clf*`, `_edge_model`, `_context_model`,
`_seam_model`, `_search_spec`, `_pred`, `_seam_pred`, `_cm_cell`, the undo
stacks, `_models_dir` -- is a **property over `self._task`**, so a task switch
is one assignment and nothing downstream knows there is more than one.
`_activate_task(task)` stashes the outgoing task's Model-tab view
(`_task_view_from_ui`: kind, search, neighbours, context), clears the
rev-keyed caches (`_class_luts`, the seam caches, `_ctx_cache`), switches to
the task's workflow through `_switch_profile` (a no-op when shared, so the
primes survive), loads its newest saved pickle **lazily** (the load runs the
compatibility gate against the ACTIVE workflow, which is why it cannot happen
at restore time for an inactive task), pushes its view and repaints; it is
refused while an Optimize / sweep / evaluate worker runs, because the finish
installs into the active task. `_switch_profile` is overridden to stamp the
active task's `workflow`; `_profile_rename` / `_profile_delete` propagate to
every task; the bindings' `_profile_from_model` should append the profile and
call `_bind_workflow(name)` rather than set `active_profile_idx`.
`_task_new(name)` / `_task_duplicate(name)` / `_task_rename(name)` /
`_task_delete(confirm)` are headless-callable with their arguments given.

**Row removal** is one path for the sequence tree's right-click menu
(*Go to* / *Clear annotations…* / *Remove …*), the Remove / Clear all
buttons and headless callers: `_remove_rows_guarded(rows)` refuses while the
engine is busy, asks with `_remove_rows_message(rows)`, then
`_remove_rows(rows)` removes items before sequences from the highest index
down and settles through `_after_rows_removed`. The labeler layers the
annotations on it -- `_doomed_interactions(rows)` are the gestures whose item
no longer exists afterwards (a slice held by a second sequence keeps its; a
slide row also owns the keys of ROIs cut away earlier), they leave as one undo
step, and the store rebinds through `catalogue.index_of` so keys that are not
`folder/basename` bind too. Per-row `_clear_row_annotations_guarded` and the
class panels' `clear` button (`_clear_class_guarded`) delete gestures without
touching the data.

`AnnotationShell` adds: `FIELDS`, `_workflow_summary(profile)`,
`_workflow_hint_tooltip()`, `_set_region_layer_visible(on)`,
`_region_layer_visible()`, `_on_regions_toggle()`, `_stats_brief(stats)`,
`_expected_feature_names()` (None skips the compatibility gate),
`_feature_schema_now()`, `_profile_from_model(path, statistics)`. The tools
reach the app only through `viewer`, `regions`, `catalogue`, `store`,
`status_var`, the tool/magic Tk variables, `_pred`, `_current()`,
`_current_key()`, `_commit_interaction`, `_commit_blob`,
`_accept_predictions`, `_begin/_preview_regions/_end_preview`,
`_class_color_hex`, `_unfocus_entries`, `_refresh_render`, `_push_history`.

## Contracts that must not break

* **annotations.json v2** (`LabelStore.to_json`): raw gesture geometry,
  keyed by the SLIDE (coupon: the folder-qualified slice file, which is its
  own slide; mspath: `items.slide_id`); v1 bare-basename keys migrate on
  `rebind()` when unambiguous, and item keys of a store written before
  gestures were slide-bound (`slide@level#rect`) rebase to the slide through
  the catalogue's `rebase`, recording the item's level / scale as the
  gesture's scale of intent. `tests/test_compat_docs.py` pins the
  byte-identical round trip; `tests/test_slide_binding.py` the rebase.
* **Gestures are the slide's; items query them** (`LabelStore.for_item`,
  the shell's `_gestures_for_key` / `_gesture_on_item`): an item sees the
  gestures whose extent (`gesture_extent`: points, outline, half the stroke
  width) meets its rect (`ItemCatalogue.binding_of`). Never select by
  `it.slice_key == item key` or by the `(si, li)` hints -- the hints name
  the slide's first row.
* **`meta` and resolution.** On the level a gesture was drawn at the
  geometry alone decides what it paints (a re-decomposition cannot be
  steered by stale metadata). Off that level, `meta["px"]` (slide px per
  screen px at draw time) keeps a stroke the width the user saw
  (`stroke_mask`, when it exceeds 1.5 raster px -- also on the drawing
  level for gestures that carry it), `meta["outline"]` (an extent's closed
  loops, `extents.py`) resolves a magic fill / blob / accepted prediction
  by its outline instead of its seeds, and `meta["scale"]` switches a trace
  from exact crack ids to a corridor (`seam_labeling.corridor_coverage`).
  Gestures without these keys -- every coupon gesture, every older one --
  behave exactly as before. `_draw_meta` is where an app records them.
* **Classifier pickle** (`ModelBundle`): v4 written, v1-v4 read by feature
  detection. The old module paths `msseg.mscoupon.model_search.FeatureSubset`
  and `msseg.mscoupon.torch_mlp.TorchMLPClassifier` resolve through the shims
  in `packages/mscoupon` -- never delete those shim modules.
* **Session document v2 / v3** (`session_doc`): profiles are opaque to the
  framework; the app's reader/default are injected. A document with
  `tasks[]` + `active_task` is **v3** (`SESSION_DOC_VERSION_TASKS`); each
  task entry carries `uid`, `name`, `workflow` (a profile name),
  `annotations` (the store document), `models` (the saved-model records)
  and `view` (the four `TASK_VIEW_KEYS`: `model_kind`, `model_search`,
  `neighbours`, `context`, which are NOT in the window `view` any more).
  Without `tasks` the document is the v2 one it always was, byte for byte --
  the viewers keep writing it -- and the reader turns it into ONE task named
  after the active profile, moving the four keys out of `view`. The reader
  always returns `tasks` + `active_task`, and `annotations` / `models` as
  the active task's for older callers; a task whose workflow names no
  profile is repointed to the active one with a note
  (`tests/test_session_doc_tasks.py`).
* **Class names** (`LabelStore.names`, `set_name`): display only -- ids stay
  the wire format of every raster, LUT and probability column. `to_json`
  writes `"name"` inside a `classes[]` entry only when one is set, so an
  unnamed store's document is unchanged (`tests/test_class_names.py`).
* **`np` as a parameter**: `labeling`, `magic_fill`, `training` and the tools
  take numpy as an argument instead of importing it, so the pure-function
  tests and headless callers control the import. Keep the convention.
* Cross-validation groups come from `catalogue.group_of(key)` -- one group per
  item for both the region model and the edge model (leave-items-out).
* **Context columns are ordinary columns.** `context.augment` returns a NEW
  table (`rec["stats"]` is never mutated; the magic fill and every readout
  keep reading the raw one), and the classifier's stream yields it in place
  of the record's table. The compatibility gate compares
  `expected + context.column_names(model's spec, expected)` -- the spec the
  model was fit under rides with it (`_clf_context`, the pickle's
  `stack["context"]`, the model record's `context`), separately from the
  Model tab's picked spec (`view.context`), which describes the NEXT model.
  Both keys exist only when there is a spec, so a context-free pickle and
  session are the documents they always were. The latent-ring head
  (`context.LatentContextModel`) is `stack["latent"]`, restored only when
  its `names_hash` / `net_hash` match the pickle's base; its columns are
  never part of the fingerprint. The labels block (`LabelSpec`) IS in the
  fingerprint and makes predictions depend on the store: `AnnotationShell`
  overrides `_rebuild_class_panels` (the "labels changed" signal) to call
  `_labels_changed`, which drops cached predictions when `store.rev` moved.
* The arcs dict (`{"a", "b", "saddle", "source"}`) may carry an OPTIONAL
  `length` array (shared boundary length per arc, `context.ensure_contact`),
  derived from the label raster on demand and cached in place; nothing
  requires it, and no new feature reads a saddle value.

* **annotations.json v3**: the `seams` list (scope / trace gestures) is
  written, and the version raised to 3, only when there are any; a store
  without seams is byte-identical to v2 and an older reader ignores the key
  (`tests/test_compat_docs.py`, `tests/data/annotations_v3.json`).
* **`ModelBundle.seam`**: the seam model's key is written only when one was
  fit, so a seam-less pickle is the v4/v5 document it always was.

## Tests

`packages/mslabeler/tests` (pure Python, `conftest.py` puts the three source
trees on `sys.path`): `test_labeling`, `test_magic_fill`, `test_training`,
`test_context`, `test_bundle`, `test_model_search`, `test_edge_model`, `test_torch_mlp`,
`test_compat_docs`, `test_sources` (the canvas composite is compared
pixel-for-pixel with a reference implementation). `packages/mscoupon/tests`
keeps the coupon tests plus `test_shims` (shim identity and the old pickle
paths). The two `--selftest`s (`mscoupon-gui`, `mscoupon-labeler`) are the
integration tests of the shells.

## Building a derivative (guidance, not code)

A whole-slide labeler would implement the seams like this:

* **`ImageSource`** over the slide's tiled pyramid -- `pyramid.PyramidImageSource`
  already is this: `levels` = the pyramid depth, `read_region` assembles a
  window at a level from cached tiles, `best_level` picks the level nearest the
  canvas zoom. Nothing full-resolution is ever materialised. `level_scale` is
  the file's true downsample, not `2**k`: levels floor their dimensions, so a
  deep level of a 90 000-row slide is 515.6x. A rect running off a level is
  zero-filled rather than clipped, which is what lets a halo'd ROI read across
  an edge.
* **`ItemCatalogue`** over slides and their ROIs: an item is one ROI (key e.g.
  `"slide.svs#roi3"`, stable across sessions because it is what
  `annotations.json` stores); `group_of` returns the slide, so cross-validation
  leaves whole slides out.
* **`RegionProvider`** computing on demand: `record(key)` returns the cached
  ROI decomposition at the current commit, `ensure_record` runs the
  segmentation for that ROI only (on the ROI's pixels, at the working level),
  `request` schedules it off-thread and `poll()` reports completion the way
  `ComputeEngine.poll()` does; `label_layer(key)` serves the ROI's ids placed at
  the ROI's offset. Region ids stay ROI-local ints (the LUT pattern sizes
  arrays by `labels.max()+1`).
* **`LabelLayer`** for ROIs that are themselves tiled: `crop(level, ...)`
  assembles from tiles and `full()` returns None -- the canvas then resizes
  the crop nearest-neighbour. Gesture rasterization (`labeling.touched_ids`)
  still expects a full raster today; a bbox-limited variant reading
  `layer.crop(0, bbox)` is the next seam to add.
* **`FieldConventions`** for whatever the ROI statistics table calls its
  columns (e.g. optical-density channels instead of `mean_base`).
* **Annotations -> pixels**: `labeling.resolve_slice` gives region -> class;
  the per-pixel mask is `class_of[labels]` over the ROI, written tile by tile.
  `LabelerApp._write_training_set` is the coupon version to mirror.

Then `class SlideViewer(ViewerShell)` with the hooks above, and
`class SlideLabeler(AnnotationShell, SlideViewer)` with `SESSION_APP`,
`FIELDS` and the annotation hooks; a `[project.scripts]` entry makes it a
command.
