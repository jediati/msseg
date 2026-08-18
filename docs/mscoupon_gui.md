# mscoupon interactive viewer (`mscoupon-gui`)

A Tkinter desktop tool for exploring the 2D Morse–Smale slice-segmentation
pipeline: set parameters, prime a test TIFF sequence, and interactively re-threshold
and filter features while viewing the result — then export a `config.json` that the
C++ CLI reproduces on a larger machine.

## Install & run

```bash
pip install ./packages/msseg-viz ./packages/mscoupon   # local dep first
mscoupon-gui [folder_or_first_tiff]
mscoupon-gui --selftest                                 # headless logic check
```

Dependencies (declared in `packages/mscoupon/pyproject.toml`): `numpy`, `scipy`,
`pillow`, `matplotlib`, `msseg-viz`, and `large-image[tiff]` (pyramidal base
rendering; optional — the canvas falls back to in-memory display without it).

## Layout

**Left panel — build the workflow**
1. **Sequences** — browse a folder; ctrl-select contiguous runs and
   *Make subsequence from selection*. Each subsequence is processed as its own 3D
   stack (e.g. `asdf_0011..0013` + `asdf_0025..0026` → two subsequences).
2. **Filter chain** — pick a filter type to populate its parameter widgets; a
   trailing "none" card lets you append more (chain e.g. morphology → edges).
   Types: `blur, derivative, laplacian, zero_crossings, hessian_eigenvalues,
   structure_eigenvalues, edges, erode, dilate, open, close, label_components`.
3. **MSC parameters** — max persistence %, ascending/descending 2-manifold,
   optional accurate gradient (slower/more memory), per-slice min area gate.
4. **Run with selected** — discards prior runs; per slice, runs the filter chain
   and primes the MSC base decomposition + statistics tree (threaded).
5. **Export config.json** — writes one config per subsequence (an explicit
   `input.files` list) so `mscoupon --config config_N.json` reproduces the output.

**Right panel — view & refine (one slice at a time)**
- **Renderer** — grayscale base + toggleable overlay channels (filtered field,
  segmentation, mask), with min/max brightness/contrast and an overlay-alpha
  slider; zoom (wheel) and pan (drag).
- **Slice slider (top)** — a single slider linearized over *all* subsequences'
  slices (shown when > 1 TIFF); crossing a boundary switches the active stack.
- **Persistence %** — live re-threshold via native cancellation (cheap; no MSC
  recompute).
- **Per-slice selection** — an extendable `field / op / value` chain (`area`,
  `mean_base`, `min_base`, `max_base`, `std_base`, `bbox_w/h`,
  `min_x/max_x/min_y/max_y`, `ext_x`, `ext_y`, `ext_base`, `ext_filtered`;
  ops `lt le gt ge eq between`). The dropdown is generated from the C++ schema
  (`mscoupon.feature_fields()`), so it always offers exactly the fields the
  `statistics` block computes — including the filtered-channel aggregates when
  they are switched on, and nothing when a reduction is switched off)
  gating the 2D MSC regions before the 3D assembly. The `ext_*` fields describe
  the region's **seeding critical point** — the minimum for ascending manifolds,
  the maximum for descending — so `ext_base < 0.3` rejects a basin whose well
  bottom is not actually dark, which `mean_base` alone cannot distinguish.
- **ext sample radius** (`msc.extremum_sample_radius`) — `0` reads `ext_base` at
  the single critical pixel; `r > 0` averages the `(2r+1)²` window around it,
  trading exactness for noise robustness.
- **3D connectivity** (6/18/26) — cross-slice linking for the on-the-fly 3D
  assembly.
- **Show merge tree** — the voxel-count icicle for the current slice.

## How it works (engine)

The interactivity comes from a two-phase C++ core facade,
`msseg::Msc2DPipeline` (`libs/core/msseg/compute/msc2d.cpp`), mirroring cellseg's
`Msc3D`/`CellPipeline`:

- **Prime** (`prime_slice`) runs the MSC once, keeps the MSCEER engine alive, and
  caches the finest 2-manifold labels plus per-manifold statistics on both the
  base image and the filtered field — including each manifold's **seeding
  extremum** (the pixel attaining its filtered min/max) and the base channel
  sampled there.
- **Re-threshold** (`select_persistence`) uses MSCEER's **native cancellation**
  hierarchy (`setPersistence` + `ascending/descending2Manifolds` remap each base
  extremum to its living representative) and rolls the cached statistics up — no
  gradient or base-manifold recompute, so dragging the slider is cheap.

`Msc2DPipeline` is **authoritative** for segmentation, used by both the GUI and the
CLI batch pipeline, so an exported config reproduces the viewer's per-slice output.

3D features are assembled in Python on-the-fly
(`src/msseg/mscoupon/assembly.py`, union-find over `(slice, feature)` linked by an
N-neighbour cross-slice stencil), mirroring the C++ `SliceMatcher` connectivity so
the viewer and CLI 3D groupings agree. The feature-query chain is evaluated by the
single-source C++ evaluator (`mscoupon::row_passes`, exposed as
`evaluate_queries`), shared by 2D features, 3D features, and the CLI.

## Config schema (extends the CLI's `AppConfig`)

```jsonc
{
  "input":  { "folder": "...", "files": ["slice_0000.tiff", ...] },
  "output": { "folder": "..." },
  "filters": [ { "operation": "blur", "params": { "sigma": 1.0 } }, ... ],
  "msc":    { "persistence_percent": 10.0, "manifold": "ascending",
              "accurate_ascending": false, "accurate_descending": false,
              "extremum_sample_radius": 0 },
  "segments": { "min_area": 25 },
  "feature_filters": [ { "field": "area", "op": "ge", "value": 50 },
                       { "field": "ext_base", "op": "lt", "value": 0.3 } ],
  "assembly": { "connectivity": 26 },
  "matching": { "enabled": true }
}
```

A singular legacy `"filter"` object is still accepted (read as a one-element chain).

## Logging (verbose by design)

Launch `mscoupon-gui` (or the `mscoupon` CLI) **from a terminal** to see the full
log. Two streams, both intentionally verbose:

- **MSCEER's own stdout** — the discrete-gradient / MSC / cancellation output
  (critical-point counts, cancellation values, timings). This is *not* squelched:
  it tells you what topological structure was actually in the data and how the
  persistence parameter relates to it.
- **`[mscoupon] ...` stage logs** — per-run and per-slice summaries:
  - GUI (during Run): image `shape/min/max/mean`; each filter step's operation +
    params + output `min/max`; `value_range` + region count after priming.
  - GUI (on a persistence/query change): persistence %, 2D features total + size-
    gated, 3D feature count, and how many pass the feature-query chain.
  - CLI: a startup config summary, then one line per slice
    (`image[min,max] filtered[min,max] regions=N kept=M`).

Per-arc **geometry is never built** (only MSC connectivity + the cancellation
hierarchy), so priming stays fast on data with long separatrices
(`ComputeOptions.buildArcGeometry=false`; `Msc2D::arcGeometry()` returns
connectivity with empty polylines).

## Caveat: TIFF decoding

The GUI reads TIFFs via Pillow / `large_image` (handles compressed formats), but
the C++ CLI reads via TinyTIFF, which cannot decode Deflate-compressed TIFFs. A
sequence viewable in the GUI may therefore need re-saving as uncompressed for the
CLI run.
