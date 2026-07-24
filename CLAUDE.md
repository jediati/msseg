# MSSeg — agent orientation

MSSeg is a Morse-Smale segmentation platform: a **portable core** library plus
thin **frontends**. It takes a floating-point volume, transforms it (FeatureJ /
diffg filters), computes a discrete gradient + Morse-Smale complex (MSCEER /
GInt), simplifies by persistence, and segments via graph-walking into a label
volume. (The old 2D pipeline became the `mscoupon` instance.)

## Layout

```
core/msseg/         portable core (namespace msseg): msseg_core + msseg_io
                    filter · compute (msc2d/msc3d) · graph · segment · workflow · io
instances/mscoupon/ verified 2D TIFF-slice pipeline (lib + cli + python)
instances/cellseg/  3D fluorescent-membrane cell segmentation (lib + cli + python):
                    two-phase heavy-lift + threshold-tuning (merge tree / masks)
generic/msworkflow/ JSON-driven workflow runner (cli + python) over the core
apps/msviewer/      Windows-only OpenGL debug viewer (optional; M6, in progress)
python/msseg/       pip package; per-frontend pybind modules install as submodules
cmake/              Dependencies · AddInstance (instance helper) · VendoredGL
```

Dependencies (diffg, MSCEER/GInt, TinyTIFF, nlohmann_json, pybind11) are pinned
via FetchContent (`cmake/Dependencies.cmake`), with a `MSSEG_DEPS_DIR` local
override for offline/HPC. Local checkouts to read as references:
`../MSCEER` (GInt + `msc_2d_lib`), `../../libraries/FeatureJ/diffg`,
`../../from_old/code/MSCapps` (viewer sources).

## Build / test (Windows)

Run inside a VS dev env (`VsDevCmd.bat -arch=x64`) so `cl.exe`/`ninja` are found.

```bash
cmake --preset windows-msvc -DCMAKE_MAKE_PROGRAM=C:/Users/jediati/bin/ninja.exe
cmake --build --preset windows-msvc
ctest --preset windows-msvc            # mscoupon_tests + core_smoke
```

Add `-DMSSEG_BUILD_PYTHON=ON` for the pybind modules. Presets `linux-gcc` /
`hpc` build the portable parts off Windows (viewer excluded). For a
step-by-step Linux/HPC recipe (modules, venv, `pip install`, cellseg smoke
test), see **[docs/dane_hpc_build.md](docs/dane_hpc_build.md)**.

## The one hard rule: the GInt firewall

MSCEER's `gi_*.h` (and `msc_2d_lib.h`) are C++11-era and compile **only** in
`core/msseg/compute/msc3d.cpp` / `msc2d.cpp`. `GInt`/`msc_2d_lib` link PRIVATE
to `msseg_core`. Everything else crosses the boundary through the plain-data
`MscGraph` and the `Msc3D` API — never `#include "gi_*.h"` elsewhere.

## Extending MSSeg

To add a segmentation strategy, a new instance, or a core/filter stage, read
**[docs/adding_instances.md](docs/adding_instances.md)** — it has the
step-by-step recipes, the `add_msseg_instance` contract, the Python binding
pattern, and the GInt gotchas (include order, `INDEX_TYPE`/`INT_TYPE` are global
macros, volume layout).

## Status

Done: M1 (restructure + parity), M3 (3D MSC core + `core_smoke`), M4 (python +
wheel), M5 (generic runner), M6 (Windows viewer). Pending: M2 (Linux/HPC
parity). Work happens on branch `refactor/msseg-scaffold`.
