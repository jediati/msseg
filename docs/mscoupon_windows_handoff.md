# Handing mscoupon to a Windows machine with no compiler

The recipient needs **no C++ compiler, no CUDA toolkit, and no CMake**. They get
two prebuilt wheels and `pip install` them. Everything below assumes the
collaborator is on **Windows x64 with CPython 3.11**.

A wheel with a compiled extension is locked to one interpreter version
(`cp311` = CPython 3.11) and one platform (`win_amd64`). If they move to 3.12 or
3.13, rebuild with that interpreter — nothing else changes.

## Where the wheels live

Published as GitHub Release assets on the (public) `jediati/msseg` repo, so the
collaborator needs no GitHub account and no manual download:

**<https://github.com/jediati/msseg/releases/tag/mscoupon-win-0.1.0>**

GitHub Packages is not an option — it has no Python/PyPI registry — and
committing wheels into the tree would bloat it, so Releases is the mechanism.

Cutting a new one (`gh` authenticates over HTTPS; `--target` needs a **full**
SHA or a branch name, an abbreviated SHA is rejected as
`target_commitish is invalid`):

```bash
gh release create <tag> --target $(git rev-parse HEAD) --title "..." --notes-file notes.md dist-ship/*.whl
```

## What you send

Two files — or, with the release above, just the install line:

| File | What it is |
|---|---|
| `msseg_mscoupon-0.1.0-cp311-cp311-win_amd64.whl` | the CLI, GUI, labeler + the compiled `mscoupon_py` extension (GPU-enabled) |
| `msseg_viz-0.1.0-py3-none-any.whl` | the shared palette / icicle package; pure Python, not on PyPI, so it has to travel with the other wheel |

Every other dependency (numpy, scipy, pandas, tifffile, bokeh, matplotlib,
large-image, scikit-learn) comes from PyPI at install time.

## What has to be on their machine

1. **CPython 3.11, 64-bit** — the python.org installer is fine, and it includes
   Tkinter, which the GUI and labeler need.
2. **Microsoft Visual C++ 2015–2022 Redistributable (x64)** —
   [aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe).
   This is a redistributable runtime, **not** a compiler. The extension links
   `MSVCP140.dll` and `VCOMP140.DLL` (the OpenMP runtime), and Python ships
   neither — it only carries `vcruntime140.dll`. Most machines already have the
   redistributable from some other application; if `import msseg.mscoupon` dies
   with a DLL load error, this is why.
3. **An NVIDIA driver** — only if they want the GPU path. No CUDA toolkit: the
   CUDA runtime is statically linked into the extension and the driver API is
   resolved lazily, so the wheel imports and runs on machines with no NVIDIA
   hardware at all.

## Install

Straight from the release, nothing to download by hand (the PEP 508
`name @ url` form is what lets an extra ride along on a URL — a bare
`<url>[classify]` does not parse):

```bash
pip install "msseg-viz @ https://github.com/jediati/msseg/releases/download/mscoupon-win-0.1.0/msseg_viz-0.1.0-py3-none-any.whl" "msseg-mscoupon[classify] @ https://github.com/jediati/msseg/releases/download/mscoupon-win-0.1.0/msseg_mscoupon-0.1.0-cp311-cp311-win_amd64.whl"
```

Or from local files:

```bash
pip install msseg_viz-0.1.0-py3-none-any.whl "msseg_mscoupon-0.1.0-cp311-cp311-win_amd64.whl[classify]"
```

Both wheels are version `0.1.0`; a later build that reuses that version needs
`pip install --force-reinstall` to take effect.

The `[classify]` extra pulls scikit-learn, which powers the labeler's
Train/Classify buttons. Drop it and everything else still works; the import is
lazy and the status bar says what is missing.

## Run

The wheel installs three console entry points, so there is no `PYTHONPATH` to
set and no working directory to be in:

```bash
mscoupon-labeler
```

`mscoupon-gui` is the sequence browser/viewer, and `mscoupon` is the batch CLI
(`mscoupon --help`).

## The GPU path

GPU is a **runtime opt-in**, not a separate build: `use_gpu_gradient` in the
config (the GUI exposes it as a checkbox) selects the CUDA discrete gradient,
and `use_gpu_stats` follows it unless set explicitly. With no CUDA device the
code falls back to the CPU loop, so the same wheel serves both cases.

Kill switches for an A/B run, no config edit needed:

```bash
set MSSEG_GPU_STATS=0
```

The shipped wheel carries SASS for `sm_75, 80, 86, 89, 90, 100, 120, 121` —
Turing through Blackwell, consumer and datacenter. CUDA 13 dropped Maxwell,
Pascal and Volta, so a **pre-Turing GPU (GTX 10-series and older) is not
covered**; that needs a rebuild against a CUDA 12.x toolkit with the
corresponding `CMAKE_CUDA_ARCHITECTURES`.

GPU and CPU gradients agree to ~99.99% of pixels — the same margin two CPU runs
agree to. Note that raw label **ids are not stable between runs** (they are
assigned in thread-completion order), so compare segmentations by partition,
not by comparing label arrays elementwise.

## Rebuilding the wheel

Run inside a VS dev environment so `cl.exe` is on `PATH`. `MSSEG_GPU` stays
**OFF** by default in `pyproject.toml` on purpose — that keeps
`pip install ./packages/mscoupon` working on a machine with no CUDA toolkit —
so the GPU build asks for it explicitly:

```bash
python -m pip wheel --no-deps -w dist-ship --config-settings=cmake.define.MSSEG_GPU=ON --config-settings="cmake.define.CMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe" --config-settings="cmake.define.CMAKE_CUDA_ARCHITECTURES=75-real;80-real;86-real;89-real;90-real;100-real;120-real;121-real;120-virtual" ./packages/mscoupon
```

Two details that bite:

- **Use the v13.0 `nvcc` explicitly.** The `nvcc` on `PATH` is 12.1, which
  cannot target `sm_120`.
- **Always pass `CMAKE_CUDA_ARCHITECTURES`.** MSCEER defaults to
  `native`, which bakes in *only the build machine's* GPU — a wheel built
  without this flag runs on your laptop and nowhere else.

Build the companion wheel with any interpreter, since it is pure Python:

```bash
python -m pip wheel --no-deps -w dist ./packages/msseg-viz
```
