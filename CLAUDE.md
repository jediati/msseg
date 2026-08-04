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

**mscoupon interactive viewer** (`mscoupon-gui`, Tkinter — see
[docs/mscoupon_gui.md](docs/mscoupon_gui.md)): browse TIFF sequences into
subsequences, chain filters, set persistence + manifold, prime each subsequence,
then live-drag a persistence slider and filter 3D features by statistics, and
export a `config.json` the CLI reproduces. Backed by a **merge/statistics tree**
in the portable core: `msseg::Msc2DPipeline` (`libs/core/msseg/compute/msc2d.cpp`)
runs the MSC once and caches the base 2-manifold decomposition + a merge tree
(`libs/core/msseg/graph/merge_tree.{hpp,cpp}`, promoted from cellseg, generalized
asc/desc) + per-manifold statistics, so `select_persistence` re-thresholds cheaply.
The merge tree is the **authoritative** segmentation for BOTH the GUI and the CLI
(the batch pipeline uses `Msc2DPipeline` too), so an exported config reproduces
the viewer output; it intentionally diverges from GInt's native cancellation above
persistence 0 (branch decomposition vs pairwise cancel). Filter chain
(`filters[]`, incl. diffg morphology `erode/dilate/open/close`) and the
feature-query chain (`feature_filters[]`, evaluated by the single-source
`mscoupon::row_passes`) are honored by the CLI; the GUI's on-the-fly 3D assembly
(`src/msseg/mscoupon/assembly.py`) mirrors the matcher's connectivity.
