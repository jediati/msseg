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
  msseg-meta/         "msseg": umbrella that depends on all of the above
CMakeLists.txt        dev "build & test everything" root (add_subdirectory libs/core + each package)
```

### How sharing works (two kinds of shared code)

| Shared code | Mechanism | In wheels? | Portable? |
|---|---|---|---|
| C++ **core** (`libs/core`) | source-shared: each package `add_subdirectory(../../libs/core)`, static-linked | yes (into each pyd/CLI) | yes |
| C++ **render** (`libs/render`) | source-shared, gated by `MSSEG_BUILD_VIEWER` (Windows-x64, OFF in wheels) | **no** | no (desktop) |
| Python **viz** (`packages/msseg-viz`) | its own pure-Python distribution; instance packages depend on it | it *is* a wheel | yes (universal) |

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
  "channels": ["base"],                     // "filtered" opts its aggregates back in
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

**3D features now carry a seeding extremum** (`ext_x/y/z`, `ext_base`,
`ext_filtered` on `GlobalFeatureStat`). Each per-slice CC node derives its own
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
