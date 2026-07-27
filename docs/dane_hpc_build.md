# Building MSSeg on Dane (LLNL)

Recipe for building and installing the MSSeg Python package into a virtual
environment on **Dane**, an LLNL CTS/TOSS Linux cluster, and running the
`cellseg` (3D fluorescent-membrane cell segmentation) Python smoke test.

The package is installed with `pip` via the existing
[scikit-build-core](https://scikit-build-core.readthedocs.io/) backend
(`pyproject.toml`), which compiles the C++ core and the pybind11 extension
modules and pulls the pinned dependencies (diffg, MSCEER, TinyTIFF,
nlohmann_json, pybind11) with CMake `FetchContent`.

> These instructions assume the build node has outbound git/network access
> through the LLNL HTTPS proxy (the default). If your environment blocks it, see
> [Offline fallback](#offline-fallback) at the end.

## 1. Grab an interactive debug node

Do the build and smoke test on a compute node rather than a shared login node.
Request a single node in the debug partition (`pdebug`) with an interactive
shell; `salloc` drops you onto the allocated node:

```bash
salloc -N 1 -p pdebug -t 2:00:00        # add -A <bank> if your default bank isn't set
```

Everything below runs inside that allocation. When you're done, `exit` releases
the node. (Debug-partition time limits are short — a couple of hours is plenty
for the build plus the smoke test.)

## 2. Load modules

MSSeg's core is **C++20** (`CMakeLists.txt` sets `CMAKE_CXX_STANDARD 20`), so the
default TOSS system GCC is too old — load a newer toolchain. You need:

| Tool   | Minimum        | Why                                   |
|--------|----------------|---------------------------------------|
| GCC    | 10+            | C++20 support                         |
| CMake  | 3.21           | required by the build                 |
| Ninja  | any recent     | the CMake generator used by the build |
| Python | 3.9            | `requires-python = ">=3.9"`           |
| Git    | any            | `FetchContent` clones dependencies    |

Exact module names vary; discover them with `module avail` and load, e.g.:

```bash
module avail gcc cmake python ninja git     # find the exact names/versions
module load gcc/<ver>                        # GCC 10+ (C++20)
module load cmake                            # >= 3.21
module load python/<ver>                     # >= 3.9
module load ninja                            # or ensure `ninja` is on PATH
module load git

# Point CMake at the module-provided compilers.
export CC=gcc CXX=g++

# Sanity check.
g++ --version && cmake --version && ninja --version && python --version
```

If the build later fails to reach github.com, make sure the proxy variables are
set in your shell (commonly provided by an LLNL module or your `~/.bashrc`):

```bash
echo "$https_proxy $http_proxy"      # should be non-empty
```

## 3. Clone the repository

```bash
git clone <MSSeg repo URL> ~/MSSeg
cd ~/MSSeg
git checkout refactor/msseg-scaffold
```

## 4. Create and activate a virtual environment

```bash
python -m venv ~/venvs/msseg
source ~/venvs/msseg/bin/activate
python -m pip install --upgrade pip
```

## 5. Install MSSeg

From the repository root (with the venv active):

```bash
pip install numpy            # runtime dependency (also declared in pyproject.toml)
pip install -v .             # builds the C++ core + pybind modules, installs the msseg wheel
```

The `-v` flag streams the CMake/compile output so you can watch `FetchContent`
clone the dependencies and the `cellseg_py`, `mscoupon_py`, and `msworkflow_py`
extensions build. The build uses `scikit-build-core`, which already forces
`MSSEG_BUILD_PYTHON=ON` and disables the (Windows-only) viewer and the C++ tests.

Verify the extension loaded:

```bash
python -c "from msseg import cellseg; print(cellseg.version())"   # -> 0.1.0
```

If it prints `0.1.0`, the compiled `cellseg` extension is installed and importable.

## 6. Run the cellseg smoke test

The smoke test needs no external data — it synthesizes a small fluorescent-shell
volume in memory and drives the full two-phase pipeline:

```bash
python python/tests/smoke_cellseg.py
```

Expected output (values approximate):

```
[PASS] heavy_lift: value_range=... heavy_persistence=...
[PASS] merge_tree: roots=1 leaves=3
[PASS] segment: foreground=... membrane=... nonbg_ascending=... distinct_ids=...
cellseg smoke test OK
```

Exit code `0` means success. The same file is collectable by pytest
(`pytest python/tests/smoke_cellseg.py`) if you prefer.

## Troubleshooting

- **`from msseg import cellseg` returns `None`** — the extension did not build.
  The package imports each compiled submodule lazily and falls back to `None` on
  failure. Re-run `pip install -v .` and inspect the log for the `cellseg_py`
  target and any compile errors.
- **C++20 / `-std=c++20` errors** — the GCC in your environment is too old; load
  a newer `gcc` module and re-export `CC`/`CXX`.
- **MSCEER / GInt errors under GCC** (`constexpr operator[]` redeclaration in
  `gi_vectors.h`, or `memcpy`/`memset` "not declared" from
  `gi_regular_grid_trilinear_function.h` / `gi_discrete_gradient_labeling.h`) —
  MSCEER is C++11-era code that only ever built on MSVC. `cmake/Dependencies.cmake`
  fixes this automatically by bumping the `GInt` / `msc_2d_lib` targets to
  `CXX_STANDARD 17` and force-including `<cstring>` (the `-include` is `PUBLIC`
  so it also reaches `msc3d.cpp` / `msc2d.cpp`, which include the MSCEER headers
  directly). Guarded by `if(NOT MSVC)` so the Windows build is untouched. If the
  MSCEER pin is bumped and a *new* missing-header error appears in the same class,
  extend that block (another force-include) rather than patching `_deps` — those
  are regenerated by `FetchContent` on a clean build.
- **`ninja: command not found` or "generator Ninja not found"** — load/PATH a
  Ninja binary before installing.
- **`FetchContent` hangs or fails to clone** — the HTTPS proxy is not set;
  export `https_proxy`/`http_proxy` (see step 1) and retry.
- **No allocation** — if `salloc` can't get a node, the build and the tiny smoke
  test also run fine on a login node; just avoid running heavy segmentation
  workloads there.

## Offline fallback

If the build environment cannot reach github.com, pre-stage local checkouts of
the dependencies and build offline. Clone each dependency into a mirror
directory using the subdir names CMake expects (see `cmake/Dependencies.cmake`):

```bash
export MSSEG_DEPS_DIR=~/msseg-deps
mkdir -p "$MSSEG_DEPS_DIR"
git clone https://github.com/jediati/diffg.git       "$MSSEG_DEPS_DIR/diffg"
git clone https://github.com/sci-visus/MSCEER.git    "$MSSEG_DEPS_DIR/msceer"
git clone https://github.com/jkriege2/TinyTIFF.git   "$MSSEG_DEPS_DIR/tinytiff"
git clone https://github.com/nlohmann/json.git       "$MSSEG_DEPS_DIR/json"
git clone https://github.com/pybind/pybind11.git     "$MSSEG_DEPS_DIR/pybind11"
# (check out the pinned revisions from cmake/Dependencies.cmake for reproducibility)
```

Then install without build isolation so the build sees `MSSEG_DEPS_DIR` and the
`hpc` CMake preset's `FETCHCONTENT_FULLY_DISCONNECTED=ON`:

```bash
pip install scikit-build-core pybind11 numpy      # build deps into the venv
CC=gcc CXX=g++ MSSEG_DEPS_DIR=~/msseg-deps \
  pip install -v --no-build-isolation .
```

The `hpc` configure preset (`CMakePresets.json`) exists for the same purpose when
building the C++ side directly with CMake (`cmake --preset hpc`), reading
`$CC`, `$CXX`, and `$MSSEG_DEPS_DIR` from the environment.
