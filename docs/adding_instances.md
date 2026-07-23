# Adding a new instance or workflow to MSSeg

This guide is for agents (and humans) extending MSSeg with new segmentation
work. It assumes you have read the top-level layout in [`CLAUDE.md`](../CLAUDE.md).

MSSeg is a Morse-Smale segmentation platform split into a **portable core**
(`msseg_core` + `msseg_io`) and thin **frontends** that consume it:

- **instances** — verified, (mostly) hardcoded pipelines shipped as a
  `lib + cli + python` trio (e.g. `mscoupon`, the 2D TIFF-slice pipeline).
- **generic runner** — `msworkflow`, which runs an arbitrary JSON-described
  workflow over the core stages.

There are three common tasks, in rough order of frequency:

1. [Add a segmentation strategy](#1-add-a-segmentation-strategy) — the usual
   "iterate on graph-walking / segmentation" work.
2. [Add a new instance](#2-add-a-new-instance) — a new verified pipeline with
   its own CLI + Python module.
3. [Add a core stage / filter op](#3-extend-a-core-stage) — new filter
   operations, workflow params, etc.

---

## Architecture in one screen

```
core/msseg/                      namespace msseg   (portable: Win/Linux/HPC)
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

instances/<name>/                namespace <name>  (one dir per verified instance)
  CMakeLists.txt                 add_msseg_instance(<name> [PYTHON] [NEEDS_IO] LIB_SOURCES ...)
  lib/<name>/*.{hpp,cpp}         the workflow definition + orchestration
  cli/main.cpp                   thin CLI entry point
  python/<name>_py.cpp           pybind11 module -> installed as msseg.<name>

generic/msworkflow/              JSON-driven runner (cli + python) over core
apps/msviewer/                   Windows-only OpenGL debug viewer (optional)
```

**Hard rule — the GInt firewall:** MSCEER's `gi_*.h` headers are C++11-era and
are compiled **only** in `core/msseg/compute/msc3d.cpp` (and `msc2d.cpp`, which
includes `msc_2d_lib.h`). `GInt` / `msc_2d_lib` link **PRIVATE** to
`msseg_core`. Never `#include "gi_*.h"` or `msc_2d_lib.h` from any other file
— cross the boundary through the plain-data `MscGraph` and the `Msc3D` API
instead. This keeps MSCEER out of every other TU and out of the Python ABI.

---

## 1. Add a segmentation strategy

This is the primary extension point for iterating on segmentation. A strategy
turns a (simplified) MS complex + the filtered volume into a `LabelVolume`.

**Interface** (`core/msseg/segment/strategy.hpp`):

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

1. Add a class in `core/msseg/segment/registry.cpp` (or a new file added to
   `core/CMakeLists.txt`'s `msseg_core` sources) subclassing
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

Add/extend an assertion in `tests/core_smoke.cpp` for the new behavior.

---

## 2. Add a new instance

An instance is a verified pipeline with its own CLI and Python module. The
`add_msseg_instance` helper (`cmake/AddInstance.cmake`) emits the whole trio.

**Directory layout** (copy `instances/mscoupon/` as a template):

```
instances/<name>/
  CMakeLists.txt
  lib/<name>/...            headers included as "<name>/..."; the workflow logic
  cli/main.cpp              parses args, calls into lib/
  python/<name>_py.cpp      PYBIND11_MODULE(<name>_py, m) { ... }
```

**CMakeLists.txt:**

```cmake
add_msseg_instance(<name>
  PYTHON            # omit if no python module
  NEEDS_IO          # omit if you don't need msseg_io (TIFF)
  LIB_SOURCES
    lib/<name>/foo.cpp
    lib/<name>/bar.cpp
)
```

This creates: `<name>_lib` (STATIC, links `msseg_core` [+`msseg_io`], exposes
`lib/` as an include dir), `<name>` (exe from `cli/main.cpp`), and — when
`PYTHON` and `MSSEG_BUILD_PYTHON` — `<name>_py` (pybind module installed into
the wheel as `msseg/<name>/`).

**Register it** in the root `CMakeLists.txt` next to the existing instance:

```cmake
if(MSSEG_BUILD_INSTANCES)
  add_subdirectory(instances/mscoupon)
  add_subdirectory(instances/<name>)      # add this
endif()
```

**Wire the Python submodule** into `python/msseg/__init__.py`:

```python
try:
    from .<name> import <name>_py as <name>
except Exception:
    <name> = None
```

**CLI convention** — `cli/main.cpp` should be a thin `try/catch` that calls a
`run_cli(argc, argv)` in the lib (see `instances/mscoupon/cli/main.cpp` and
`lib/mscoupon/mscoupon_cli.cpp`).

**What goes in the lib vs the core:** single-volume stage logic that is reusable
belongs in `msseg_core`; instance-specific orchestration (batch scheduling,
config schema, file naming, per-domain heuristics) belongs in the instance lib.
`mscoupon` is the reference: its `lib/mscoupon/filter.cpp` and `msc_stage.cpp`
are *thin adapters* that convert the instance's `Image2D` to `diffg::Image` and
delegate to `msseg::apply_filter` / `msseg::compute_msc2d_labels`.

---

## 3. Extend a core stage

**New filter operation:** add a branch in `apply_filter`
(`core/msseg/filter/filter_stage.cpp`) dispatching on `filter.operation`, using
a diffg call. diffg ops are dimension-general, so 2D (depth 1) and 3D both work.
Read op params from the `nlohmann::json filter.params` with the `get_*` helpers.

**New workflow knob:** extend the relevant struct in
`core/msseg/workflow/params.hpp` or `pipeline.hpp` (`FilterParams`,
`Msc2DParams`, `Msc3DParams`, `SimplifyParams`, `SegmentationParams`), then
parse it in `parse_workflow()` (`pipeline.cpp`) so the generic runner and JSON
configs pick it up.

**New Msc3D capability:** add a method to `class Msc3D`
(`core/msseg/compute/msc3d.hpp` + `.cpp`). This is the only place allowed to
touch GInt. Model calling sequences on the local MSCEER checkout:
`code/MSCEER/steepest/steepest.cxx` (in-process gradient),
`code/MSCEER/extractmsc/extractmsc.h` (canonical 3D typedef stack), and
`code/MSCEER/msc_2d_lib/msc_2d_lib.cxx` (the manifold-labeling recipe).

---

## Python bindings pattern

Follow MSCEER's `msc_py` style and the existing modules
(`generic/msworkflow/python/msworkflow_py.cpp`,
`instances/mscoupon/python/mscoupon_py.cpp`):

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

Configure/build/test use CMake presets (`CMakePresets.json`). On Windows, run
inside a VS dev environment so `cl.exe`/`ninja` are on `PATH`
(`VsDevCmd.bat -arch=x64`).

```bash
# configure (first time, or after CMake changes)
cmake --preset windows-msvc -DCMAKE_MAKE_PROGRAM=C:/Users/jediati/bin/ninja.exe
# build everything
cmake --build --preset windows-msvc
# run the test suite (mscoupon_tests + core_smoke)
ctest --preset windows-msvc
```

- Python modules: add `-DMSSEG_BUILD_PYTHON=ON` at configure time (pulls
  pybind11, finds Python).
- Wheel: `pip wheel . --no-deps -w <out>` (run inside VsDevCmd). The wheel must
  contain only `msseg/` + the `.pyd` modules.
- Linux/HPC: use the `linux-gcc` / `hpc` presets. For offline HPC, set
  `MSSEG_DEPS_DIR` to a dir of local dependency checkouts and use the `hpc`
  preset (`FETCHCONTENT_FULLY_DISCONNECTED=ON`); `pip install --no-build-isolation`.

**Every new instance/strategy should add a smoke test** (extend
`tests/core_smoke.cpp`, or add a `<name>_tests.cpp` wired in the root
`CMakeLists.txt` tests section) and, for behavior changes to `mscoupon`, verify
output parity against a known-good run.

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
- **Viewer is Windows-x64 only and optional** (`MSSEG_BUILD_VIEWER`, gated on
  `WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8`). Never let it become a dependency of
  the core/instance/python build; those must stay portable.
- **Line endings** — the repo picks up CRLF on Windows; the `LF will be replaced
  by CRLF` git warnings are benign.
```
