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
| `training.py` | `TrainingSetBuilder`: annotations + tables -> `(X, y, groups, names)` and the edge model's arrays | no |
| `bundle.py` | `ModelBundle`: the classifier pickle (v4 writer, v1-v4 reader) and `compat_message` | no |
| `model_search.py`, `edge_model.py`, `torch_mlp.py` | the dense-net search, the pair model + voting, the GPU MLP | no |
| `table.py` | `FeatureTable` (the columnar per-region table) | no |
| `session_doc.py` | the session document (folders, sequences, profiles, view) and the session-file I/O | no |
| `sources.py` | `ArrayImageSource`, `PyramidImageSource`, `ArrayLabelLayer` | no |
| `canvas.py` | `SliceCanvas`: zoom/pan, an `ImageSource` base, `LabelLayer` overlays through LUTs, the transient preview layer, the HUD | yes |
| `widgets.py` | `ScrollFrame`, `jump_scale`, `scrolled_listbox`, tooltips | yes |
| `shell.py` | `ViewerShell`: window, session browser, profiles, navigation, pump, session flow | yes |
| `annotate.py` | `AnnotationShell`: the labeler layer (store, tools, hotkeys, three panes, session additions) | yes |
| `classifier.py` | `ClassifierMixin`: Train / Classify / Optimize / size sweep / edge evaluation | yes |
| `panels/` | `HintsMixin`, `ModelPanelMixin`, `AnalysisPanelMixin`, `ViewControlsMixin`, `ClassPanelMixin` | yes |
| `tools.py` | `DrawController`, `MagicFillController`, `_extremum_points` | yes |
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
`_build_left()` -> `_build_right()` -> `_after_layout()` -> status bar ->
initial folder -> auto-save. `AnnotationShell.__init__` sets the labeler's
state first (the base constructor calls the overridden builders), then chains.

## The four seams (`protocols.py`)

| protocol | what the framework asks | coupon implementation |
|---|---|---|
| `ItemCatalogue` | `keys()`, `key_of(*index)`, `index_of(key)`, `label(key)`, `group_of(key)`, `tree()` | `adapters.SequenceCatalogue`: items are slices, key = `"folder/basename"` (what `annotations.json` has always stored), address = `(si, li)`, group = the flat slice index (leave-slices-out CV) |
| `RegionProvider` | `commit`, `keys()`, `record(key)`, `ensure_record(key)`, `request(key)`, `pending()`, `poll()`, `arcs(key, np)`, `label_layer(key)` | `adapters.EngineRegionProvider` over `ComputeEngine`: cached per-slice records, synchronous compute through `ComputeEngine.ensure_slice`, MSC arcs or a cached pixel-adjacency fallback |
| `ImageSource` | `levels`, `channels`, `level_shape`, `level_scale`, `best_level(scale)`, `value_range()`, `read_region(level, x, y, w, h)` | `sources.ArrayImageSource` (one level, native floats) and `sources.PyramidImageSource` (large_image, level k = 1/2^k, display-scaled uint8) |
| `LabelLayer` | `shape`, `n_ids`, `rev`, `crop(level, x, y, w, h)`, `id_at(x, y)`, `full()` | `sources.ArrayLabelLayer` over the record's int32 raster |

A `RegionRecord` is `{commit, labels (int32 raster, -1 = background), stats
(FeatureTableLike), arcs ({"a","b","saddle"|None,"source"} or None), kept, cc}`.
`commit` is the parameter generation: every cache (class LUTs, predictions,
pixel adjacency) keys on it, and a record whose commit is not the provider's
is stale.

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
| `_slice_msc_mark` / `_slice_nav_text` / `_enumerate_items` | "" / the key / provider keys | primed marks / `folder/base` / primed slices |
| `_reset_compute()` / `_settle_controls()` / `_update_busy()` / `_refresh_render()` / `_handle_compute_event(ev)` | minimal | the engine, Rerun/Run state, HUD, the render, primed/assembly events |
| `_run_settings()` / `_apply_run_settings` / `_apply_view_state` | `{}` / none / none | cores + concurrency / the coupon view keys |
| `_session_doc_from_json` / `_import_legacy_docs` / `_profile_to_file_doc` / `_profile_from_file_doc` / `_profile_from_ui` / `_apply_profile_to_ui` | passthrough | `session.*` |

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

* **annotations.json v2** (`LabelStore.to_json`): raw gesture geometry per
  item key; v1 bare-basename keys migrate on `rebind()` when unambiguous.
  `tests/test_compat_docs.py` pins the byte-identical round trip.
* **Classifier pickle** (`ModelBundle`): v4 written, v1-v4 read by feature
  detection. The old module paths `msseg.mscoupon.model_search.FeatureSubset`
  and `msseg.mscoupon.torch_mlp.TorchMLPClassifier` resolve through the shims
  in `packages/mscoupon` -- never delete those shim modules.
* **Session document v2** (`session_doc`): profiles are opaque to the
  framework; the app's reader/default are injected.
* **`np` as a parameter**: `labeling`, `magic_fill`, `training` and the tools
  take numpy as an argument instead of importing it, so the pure-function
  tests and headless callers control the import. Keep the convention.
* Cross-validation groups come from `catalogue.group_of(key)` -- one group per
  item for both the region model and the edge model (leave-items-out).

## Tests

`packages/mslabeler/tests` (pure Python, `conftest.py` puts the three source
trees on `sys.path`): `test_labeling`, `test_magic_fill`, `test_training`,
`test_bundle`, `test_model_search`, `test_edge_model`, `test_torch_mlp`,
`test_compat_docs`, `test_sources` (the canvas composite is compared
pixel-for-pixel with a reference implementation). `packages/mscoupon/tests`
keeps the coupon tests plus `test_shims` (shim identity and the old pickle
paths). The two `--selftest`s (`mscoupon-gui`, `mscoupon-labeler`) are the
integration tests of the shells.

## Building a derivative (guidance, not code)

A whole-slide labeler would implement the seams like this:

* **`ImageSource`** over the slide's tiled pyramid (`large_image`, OpenSlide,
  tifffile's zarr store): `levels` = the pyramid depth, `read_region` reads a
  tile-aligned window at a level, `best_level` picks the level nearest the
  canvas zoom. The canvas already draws such a source; nothing full-resolution
  is ever materialised.
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
