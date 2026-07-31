# Adding a new instance or workflow to MSSeg

This guide is for agents (and humans) extending MSSeg with new segmentation
work. It assumes you have read the top-level layout in [`CLAUDE.md`](../CLAUDE.md).

MSSeg is a Morse-Smale segmentation platform split into a **portable core**
(`msseg_core` + `msseg_io`, in `libs/core/`) and thin **frontends** that consume
it, each shipped as its own independently pip-installable distribution under
`packages/`:

- **instances** — verified, (mostly) hardcoded pipelines shipped as a
  `lib + cli + python` trio (e.g. `mscoupon`, the 2D TIFF-slice pipeline).
- **generic runner** — `msworkflow`, which runs an arbitrary JSON-described
  workflow over the core stages.

There are three common tasks, in rough order of frequency:

1. [Add a segmentation strategy](#1-add-a-segmentation-strategy) — the usual
   "iterate on graph-walking / segmentation" work.
2. [Add a new instance](#2-add-a-new-instance) — a new verified pipeline with
   its own CLI + Python module, as a new package.
3. [Add a core stage / filter op](#3-extend-a-core-stage) — new filter
   operations, workflow params, etc.

---

## Architecture in one screen

```
libs/core/msseg/                 namespace msseg   (portable: Win/Linux/HPC)
  volume/types.hpp               Volume = diffg::Image<float>, LabelVolume = diffg::Image<int32_t>
  filter/filter_stage.hpp        apply_filter(Volume, FilterParams) -> Volume     (diffg; 2D & 3D)
  compute/msc2d.hpp              compute_msc2d_labels(Volume, Msc2DParams)         (msc_2d_lib facade)
  compute/msc3d.hpp              class Msc3D  (PIMPL over the GInt 3D stack)
  graph/msc_graph.hpp            MscNode / MscArc / MscGraph  (plain data, the contract)
  graph/graph_ops.hpp           reusable walking primitives over MscGraph
  segment/strategy.hpp          SegmentationStrategy (the iteration seam)
  segment/registry.hpp          make_strategy(name)  -> strategy factory
  workflow/params.hpp           FilterParams, Msc2DParams
  workflow/pipeline.hpp         WorkflowParams, Pipeline::run, parse_workflow(json)
  io/raw_io.hpp                  raw float32 <-> Volume (in msseg_core)
  io/tiff_io.hpp                 TIFF <-> buffers (in msseg_io; isolates TinyTIFF)
libs/core/tests/core_smoke.cpp   portable core smoke test (wired from the root CMakeLists)

packages/<name>/                 an independently pip-installable distribution ("msseg-<name>")
  pyproject.toml                 scikit-build-core; name, deps, [project.scripts], wheel.packages=["src/msseg"]
  CMakeLists.txt                 standalone-or-root (see below); calls add_msseg_instance(<name> ...)
  lib/<name>/*.{hpp,cpp}         namespace <name>; headers included as "<name>/..."; the workflow logic
  cli/main.cpp                   thin native CLI entry point
  python/<name>_py.cpp           pybind11 module -> installed as msseg/<name>/<name>_py
  src/msseg/<name>/__init__.py   re-exports the compiled module (PEP 420 namespace: no msseg/__init__.py)
  tests/<name>_tests.cpp         C++ unit test (built under MSSEG_BUILD_TESTS)

libs/render/                     shared OpenGL renderer (msrender) — optional, desktop/Windows only
apps/msviewer/                   generic core-level OpenGL debug viewer (links libs/render); not pip
```

**Hard rule — the GInt firewall:** MSCEER's `gi_*.h` headers are C++11-era and
are compiled **only** in `libs/core/msseg/compute/msc3d.cpp` (and `msc2d.cpp`,
which includes `msc_2d_lib.h`). `GInt` / `msc_2d_lib` link **PRIVATE** to
`msseg_core`. Never `#include "gi_*.h"` or `msc_2d_lib.h` from any other file
— cross the boundary through the plain-data `MscGraph` and the `Msc3D` API
instead. This keeps MSCEER out of every other TU and out of the Python ABI.

---

## 1. Add a segmentation strategy

This is the primary extension point for iterating on segmentation. A strategy
turns a (simplified) MS complex + the filtered volume into a `LabelVolume`.

**Interface** (`libs/core/msseg/segment/strategy.hpp`):

```cpp
class SegmentationStrategy {
 public:
  virtual LabelVolume segment(const MscGraph& graph, Msc3D& msc,
                              const Volume& filtered,
                              const SegmentationParams& params) = 0;
};
```

- `graph` — plain-data snapshot (nodes/arcs/adjacency) at the current
  persistence. Walk it with `graph/graph_ops.hpp` helpers (`nodes_of_index`,
  `reachable`, …) or your own traversal.
- `msc` — the live `Msc3D`; call `msc.basin_labels(ascending)`,
  `msc.fill_manifold(node_id, ascending, out)`, `msc.select_persistence(p)`,
  `msc.snapshot()` for heavier geometry. `Msc3D` is where GInt lives.
- `params.extra` — an opaque `nlohmann::json` blob for your strategy's own
  knobs (this is the tuning seam; nothing else needs to change to add options).

**Steps:**

1. Add a class in `libs/core/msseg/segment/registry.cpp` (or a new file added to
   `libs/core/CMakeLists.txt`'s `msseg_core` sources) subclassing
   `SegmentationStrategy`. See `BasinLabelStrategy` there as a worked example —
   it reads `params.extra.value("manifold", "ascending")` and delegates to
   `msc.basin_labels(...)`.
2. Register it by name in `make_strategy()` in the same file:
   ```cpp
   if (name == "my_strategy") return std::make_unique<MyStrategy>();
   ```
3. Use it: set `segmentation.strategy = "my_strategy"` in a workflow JSON, or
   `SegmentationParams{"my_strategy", {...}}` in C++.

**Voxel-labeling recipe** (if you need per-voxel labels from manifolds): mirror
`Msc3D::basin_labels` — `fillGeometry(nodeId, cells, ascending)` gives manifold
cells; keep the dimension-0 cells and map them to voxels via
`mesh->VertexNumberFromCellID(cell)`; remap base-node ids through
`GatherNodes(nodeId, constituents, ascending)` to fold the simplification
hierarchy in. This all lives inside `msc3d.cpp` because it touches GInt; expose
a new `Msc3D` method rather than reaching into GInt from a strategy.

Add/extend an assertion in `libs/core/tests/core_smoke.cpp` for the new behavior.

---

## 2. Add a new instance

An instance is a verified pipeline with its own CLI and Python module, shipped as
its own package. The `add_msseg_instance` helper (`cmake/AddInstance.cmake`)
emits the `lib + cli + pybind` trio. **Copy `packages/mscoupon/` as the template.**

**Directory layout:**

```
packages/<name>/
  pyproject.toml
  CMakeLists.txt
  lib/<name>/...            headers included as "<name>/..."; the workflow logic
  cli/main.cpp              parses args, calls into lib/
  python/<name>_py.cpp      PYBIND11_MODULE(<name>_py, m) { ... }
  src/msseg/<name>/__init__.py   re-exports the compiled module
  tests/<name>_tests.cpp    optional C++ test
```

**CMakeLists.txt** — the standalone-or-root pattern. The `if(NOT TARGET
msseg_core)` block bootstraps the shared deps + `libs/core` for a standalone
`pip install`; from the dev root it is skipped because `msseg_core` already
exists:

```cmake
cmake_minimum_required(VERSION 3.21)
if(NOT TARGET msseg_core)
  project(<name> LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  option(MSSEG_BUILD_PYTHON "" OFF)
  option(MSSEG_BUILD_TESTS "" OFF)
  get_filename_component(MSSEG_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
  list(APPEND CMAKE_MODULE_PATH "${MSSEG_REPO_ROOT}/cmake")
  find_package(OpenMP QUIET)
  include(Dependencies)
  add_subdirectory(${MSSEG_REPO_ROOT}/libs/core ${CMAKE_BINARY_DIR}/_msseg_core)
endif()
include(AddInstance)
add_msseg_instance(<name>
  PYTHON            # omit if no python module
  NEEDS_IO          # omit if you don't need msseg_io (TIFF)
  LIB_SOURCES lib/<name>/foo.cpp lib/<name>/bar.cpp
)
```

`add_msseg_instance` creates `<name>_lib` (STATIC, links `msseg_core`
[+`msseg_io`], exposes `lib/` as an include dir), `<name>` (exe from
`cli/main.cpp`), and — when `PYTHON` and `MSSEG_BUILD_PYTHON` — `<name>_py`
(pybind module installed into the wheel as `msseg/<name>/`).

**pyproject.toml** (scikit-build-core, src-layout):

```toml
[build-system]
requires = ["scikit-build-core>=0.9", "pybind11>=2.13"]
build-backend = "scikit_build_core.build"
[project]
name = "msseg-<name>"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = ["numpy", ...]
[project.scripts]
<name> = "msseg.<name>.cli:main"      # or analysis:main, etc.
[tool.scikit-build]
wheel.packages = ["src/msseg"]        # PEP 420 namespace: ships msseg/<name>/ only
install.components = ["msseg_python"]
[tool.scikit-build.cmake.define]
MSSEG_BUILD_PYTHON = "ON"
MSSEG_BUILD_VIEWER = "OFF"
MSSEG_BUILD_TESTS = "OFF"
```

**src/msseg/<name>/__init__.py** re-exports the compiled extension so
`from msseg import <name>` works:

```python
from .<name>_py import *  # or: from .<name>_py import run, version, ...
```

**Register it in the dev root `CMakeLists.txt`** (build-everything) next to the
others:

```cmake
add_subdirectory(packages/mscoupon)
add_subdirectory(packages/<name>)      # add this
```

There is **no** top-level `msseg/__init__.py` to edit — the `msseg` package is a
PEP 420 namespace, so each distribution's `src/msseg/<name>/` stands alone.

**CLI convention** — `cli/main.cpp` should be a thin `try/catch` that calls a
`run_cli(argc, argv)` in the lib (see `packages/mscoupon/cli/main.cpp` and
`lib/mscoupon/mscoupon_cli.cpp`).

**What goes in the lib vs the core:** single-volume stage logic that is reusable
belongs in `msseg_core`; instance-specific orchestration (batch scheduling,
config schema, file naming, per-domain heuristics) belongs in the instance lib.
`mscoupon` is the reference: its `lib/mscoupon/filter.cpp` and `msc_stage.cpp`
are *thin adapters* that convert the instance's `Image2D` to `diffg::Image` and
delegate to `msseg::apply_filter` / `msseg::compute_msc2d_labels`.

**Shared Python (viewers, etc.)** goes in the pure-Python `msseg-viz` package
(`packages/msseg-viz/src/msseg/viz/`), which instance packages depend on and
build on — never duplicated per instance.

---

## 3. Extend a core stage

**New filter operation:** add a branch in `apply_filter`
(`libs/core/msseg/filter/filter_stage.cpp`) dispatching on `filter.operation`,
using a diffg call. diffg ops are dimension-general, so 2D (depth 1) and 3D both
work. Read op params from the `nlohmann::json filter.params` with the `get_*`
helpers.

**New workflow knob:** extend the relevant struct in
`libs/core/msseg/workflow/params.hpp` or `pipeline.hpp` (`FilterParams`,
`Msc2DParams`, `Msc3DParams`, `SimplifyParams`, `SegmentationParams`), then
parse it in `parse_workflow()` (`pipeline.cpp`) so the generic runner and JSON
configs pick it up.

**New Msc3D capability:** add a method to `class Msc3D`
(`libs/core/msseg/compute/msc3d.hpp` + `.cpp`). This is the only place allowed to
touch GInt. Model calling sequences on the local MSCEER checkout:
`code/MSCEER/steepest/steepest.cxx` (in-process gradient),
`code/MSCEER/extractmsc/extractmsc.h` (canonical 3D typedef stack), and
`code/MSCEER/msc_2d_lib/msc_2d_lib.cxx` (the manifold-labeling recipe).

---

## Python bindings pattern

Follow MSCEER's `msc_py` style and the existing modules
(`packages/msworkflow/python/msworkflow_py.cpp`,
`packages/mscoupon/python/mscoupon_py.cpp`):

- Arrays cross as `py::array_t<float|int32>`, C-order, shape `(depth, height,
  width)` for volumes / `(height, width)` for slices — this matches diffg's
  row-major, x-fastest layout, so you can `memcpy` between numpy and
  `diffg::Image` with no transpose.
- Release the GIL around heavy C++: `{ py::gil_scoped_release rel; ... }`.
- `install(TARGETS <name>_py LIBRARY DESTINATION msseg/<name> COMPONENT
  msseg_python)` — the `COMPONENT` is what keeps FetchContent deps' own install
  output out of the wheel (`add_msseg_instance` already does this for you).

---

## Build, run, and test

**Dev build (everything).** Configure/build/test use CMake presets
(`CMakePresets.json`). On Windows, run inside a VS dev environment so
`cl.exe`/`ninja` are on `PATH` (`VsDevCmd.bat -arch=x64`).

```bash
cmake --preset windows-msvc -DCMAKE_MAKE_PROGRAM=C:/Users/jediati/bin/ninja.exe
cmake --build --preset windows-msvc
ctest --preset windows-msvc            # core_smoke + mscoupon_tests + cellseg_tests
```

- Python modules: add `-DMSSEG_BUILD_PYTHON=ON` at configure time.
- Per-package wheel: `pip wheel ./packages/<name> --no-deps -w <out>` (inside
  VsDevCmd). Each wheel contains only `msseg/<name>/` + its `.pyd`.
- Linux/HPC: use the `linux-gcc` / `hpc` presets; for offline HPC set
  `MSSEG_DEPS_DIR` and use `pip install --no-build-isolation ./packages/<name>`.

**Every new instance/strategy should add a smoke test** — a `<name>_tests.cpp`
wired in the package's `CMakeLists.txt` under `MSSEG_BUILD_TESTS`, and/or an
assertion in `libs/core/tests/core_smoke.cpp`; for behavior changes to
`mscoupon`, verify output parity against a known-good run.

---

## Conventions & gotchas (learned the hard way)

- **GInt firewall** — see the hard rule above. `GInt`/`msc_2d_lib` are PRIVATE
  deps of `msseg_core`.
- **GInt include order** — in `msc3d.cpp`, include `gi_basic_types.h` +
  `gi_timing.h` **before** the labeling/robins headers; they use `ThreadedTimer`
  without including it themselves.
- **`INDEX_TYPE` / `INT_TYPE` are global macros** (from `gi_basic_types.h`), not
  members of namespace `GInt`. Use them unqualified; `GInt::INT_TYPE` is a
  compile error (`GInt::int`).
- **Volume layout** — `diffg::Image` is row-major, x fastest:
  `(z*height + y)*width + x`. 2D = `depth == 1`. `RegularGrid3D` uses the same
  index order, so a `Volume`'s buffer maps to GInt with `X=width, Y=height,
  Z=depth` and no transpose.
- **Namespaces** — core is `msseg`; each instance uses its own namespace
  (`mscoupon`). Core includes read `"msseg/..."`; instance includes read
  `"<name>/..."`.
- **Namespace package** — no distribution ships a top-level `msseg/__init__.py`;
  `msseg` is a PEP 420 namespace so `msseg.mscoupon` / `msseg.cellseg` /
  `msseg.viz` coexist across separately-installed wheels.
- **Viewer is Windows-x64 only and optional** (`MSSEG_BUILD_VIEWER`, gated on
  `WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8`, forced OFF in wheels). The reusable
  engine is `libs/render` (msrender); never let it become a dependency of the
  core/instance/python build — those must stay portable.
- **Line endings** — the repo picks up CRLF on Windows; the `LF will be replaced
  by CRLF` git warnings are benign.
```
