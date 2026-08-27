# mscoupon interactive viewer (`mscoupon-gui`)

A Tkinter desktop tool for exploring the 2D Morse–Smale slice-segmentation
pipeline: set parameters, prime a test TIFF sequence, and interactively re-threshold
and filter features while viewing the result — then export a `config.json` that the
C++ CLI reproduces on a larger machine.

## Install & run

```bash
pip install ./packages/msseg-viz ./packages/mscoupon   # local dep first
mscoupon-gui [folder_or_first_tiff]
mscoupon-gui --selftest        # headless logic check (incl. the config-load
                               # round-trip; never touches your saved session)
```

Dependencies (declared in `packages/mscoupon/pyproject.toml`): `numpy`, `scipy`,
`pillow`, `matplotlib`, `msseg-viz`, and `large-image[tiff]` (pyramidal base
rendering; optional — the canvas falls back to in-memory display without it).

## Layout

**Toolbar — load & auto-save**
- **Load config.json…** — refills the widgets from a config, so a workflow does
  not have to be re-entered by hand. It restores the parameters *and* the input
  side (`input.folder` → the browsed folder, `input.files` → a subsequence). The
  dialog is multi-select: hand it the `config_0.json … config_N.json` that
  *Export* wrote and all N subsequences come back, with the parameters taken from
  the first file.
- **Restore last** — reloads the auto-saved session (see **Session file** below).
- **auto-save** (on by default) — records the session every few seconds when
  something has changed, and again on window close. Unchecking stops it.

All three are **best-effort**: a folder that has moved, a TIFF that is gone, an
unknown filter operation, a mistyped parameter, or a `feature_filters` field the
current statistics spec no longer offers is skipped and reported in the status bar
(in full in the terminal log). Loading never raises and never half-applies — if no
file can be read, nothing changes. Loading discards any primed stack, since the
parameters that produced it have just been replaced; click *Run* again.

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
- **Renderer** — a grayscale background channel + toggleable overlays
  (segmentation, mask), with min/max brightness/contrast and an overlay-alpha
  slider; zoom (wheel) and pan (drag). The **Image** dropdown offers every
  measurement channel, not just `base` and `filtered`, so a threshold on
  `max_edges_s0.7` can be looked at on the raster it is thresholding. A derived
  channel is computed for the displayed slice on demand and memoised — holding
  a twelve-channel stack for every primed slice would dominate memory.
- **Slice slider (top)** — a single slider linearized over *all* subsequences'
  slices (shown when > 1 TIFF); crossing a boundary switches the active stack.
- **Persistence %** — live re-threshold via native cancellation (cheap; no MSC
  recompute).
- **Per-slice selection** — an extendable `channel / reduction / op / value`
  chain gating the 2D MSC regions before the 3D assembly (ops
  `lt le gt ge eq between`). Both dropdowns are generated from the C++ schema
  (`mscoupon.feature_schema()`) against the **live** statistics block, so they
  offer exactly the fields the CLI will accept — and nothing when a channel or a
  reduction is switched off. Picking the channel first and the reduction second
  is what keeps a twelve-channel stack (~60 fields) usable; the split is
  structural rather than parsed, so `min_x` is a geometry field and not the `min`
  reduction of a channel called `x`. Geometry (`area`, `bbox_w/h`,
  `min_x/max_x/min_y/max_y`, `ext_x`, `ext_y`) lives under a `geometry`
  pseudo-channel. `relevance_base` — the experimental shifted base contrast
  `(max_base-min_base) / (min_base + relevance_ceiling-relevance_floor)` — sits
  on the base channel. The `ext_*` fields describe
  the region's **seeding critical point** — the minimum for ascending manifolds,
  the maximum for descending — so `ext_base < 0.3` rejects a basin whose well
  bottom is not actually dark, which `mean_base` alone cannot distinguish.
- **ext sample radius** (`msc.extremum_sample_radius`) — `0` reads `ext_base` at
  the single critical pixel; `r > 0` averages the `(2r+1)²` window around it,
  trading exactness for noise robustness.
- **3D connectivity** (6/18/26) — cross-slice linking for the on-the-fly 3D
  assembly.
- **Statistics channels** (left panel, section 5) — which rasters a feature is
  measured on. `base` and `filtered` are the two the pipeline already builds;
  each derived kind (`blur`, `edges`, `gradmag`, `laplacian`, `hessian`) takes a
  **sigma list**, and the cross-product is the channel set — so
  `blur/edges/hessian × {0.7, 1.5, 3.0}` is twelve channels (hessian yields
  largest and smallest per sigma) from three lines. Reduction checkboxes
  (`mean/min/max/std`) apply to every channel, and the readout under the panel
  shows the resulting channel and field counts, which are what set the width of
  every per-feature row.

  Derived channels are **measure-only**: `filters` is still the sole topology
  field the MSC runs on, and the seeding extremum is still located on it. They
  are computed on the **base** raster, i.e. after `base_filters`, so a normalized
  workflow's scale-space responses are in normalized units too. Measured on a
  3232² stack, twelve derived channels add roughly a second per slice — they
  collapse into a single diffg filter-bank traversal that shares its separable
  passes rather than running one full pass per filter.

## How it works (engine)

The interactivity comes from a two-phase C++ core facade,
`msseg::Msc2DPipeline` (`libs/core/msseg/compute/msc2d.cpp`), mirroring cellseg's
`Msc3D`/`CellPipeline`:

- **Prime** (`prime_slice`) runs the MSC once, keeps the MSCEER engine alive, and
  caches the finest 2-manifold labels plus per-manifold statistics over every
  measurement channel — including each manifold's **seeding extremum** (the pixel
  attaining its filtered min/max) and every channel sampled there. The channel
  rasters themselves are released once the per-manifold cells are accumulated;
  only the cells are kept, at 24 B per manifold per channel. Priming also computes the slice's base-channel relevance
  floor/ceiling (absolute extrema by default, or configured percentiles).
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
  // base channel chain (statistics + pixel thresholds read from its output),
  // typically a single `normalize` stage
  "base_filters": [ { "operation": "normalize", "params": { "method": "gmm" } } ],
  "msc":    { "persistence_percent": 10.0, "manifold": "ascending",
              "accurate_ascending": false, "accurate_descending": false,
              "extremum_sample_radius": 0 },
  "segments": { "min_area": 25 },
  // Which channels a feature is measured on, and with which reductions. Omitting
  // the block gives the default spec (base aggregates + extremum + relevance),
  // and the viewer writes one only when it differs from that -- so a workflow
  // that never opened the panel exports exactly what it did before.
  // A bare string is one of the two rasters the pipeline already builds; an
  // object is a derived scale-space channel measured on `base`. `sigmas` is a
  // cross-product, and `hessian` yields two channels per sigma
  // (`hessian_largest_s1.5`, `hessian_smallest_s1.5`).
  "statistics": {
    "channels": [ "base",
                  { "kind": "blur",    "sigmas": [0.7, 1.5, 3.0] },
                  { "kind": "edges",   "sigmas": [0.7, 1.5, 3.0] },
                  { "kind": "hessian", "sigmas": [0.7, 1.5, 3.0],
                    "sort_by_absolute_value": true } ],
    "reductions": ["mean", "min", "max"],
    "extremum": true,
    "relevance": { "enabled": true,
                   "low_percentile": 1.0, "high_percentile": 99.0 }
  },
  "feature_filters": [ { "field": "area", "op": "ge", "value": 50 },
                       { "field": "max_edges_s0.7", "op": "lt", "value": 0.02 },
                       { "field": "relevance_base", "op": "gt", "value": 0.2 } ],
  // per-pixel trim applied before connected components
  "pixel_filters": [ { "channel": "filtered", "mode": "omit",
                       "op": "lt", "value": 0.1 } ],
  "assembly": { "connectivity": 26 },
  "matching": { "enabled": true }
}
```

A singular legacy `"filter"` object is still accepted (read as a one-element chain).

A config carrying only `msc.persistence_absolute` (or the legacy `msc.persistence`)
loads fine, but the viewer has no absolute-persistence widget — a percent cannot be
derived from an absolute threshold without the slice value range, which is unknown
until *Run*. The *Max persistence %* entry keeps its current value and the status
bar says so.

## Session file

`auto-save` writes to `%APPDATA%\mscoupon\last_session.json` on Windows, and
`$XDG_CONFIG_HOME/mscoupon/last_session.json` (or `~/.config/mscoupon/…`)
elsewhere. *Restore last* reads it back.

The file **is** a config — a valid `AppConfig` for the first subsequence — plus one
extra top-level `"_gui"` key carrying what `AppConfig` cannot express: every
subsequence with its name, the browsed folder, and the view state. The C++ parser
reads named keys only, so the extra one is ignored and
`mscoupon --config last_session.json` runs normally. Exported configs never carry
`"_gui"`.

## Logging (verbose by design)

Launch `mscoupon-gui` (or the `mscoupon` CLI) **from a terminal** to see the full
log. Two streams, both intentionally verbose:

- **MSCEER's own stdout** — the discrete-gradient / MSC / cancellation output
  (critical-point counts, cancellation values, timings). This is *not* squelched:
  it tells you what topological structure was actually in the data and how the
  persistence parameter relates to it.
- **`[mscoupon] ...` stage logs** — per-run and per-slice summaries:
  - GUI (during Run): image `shape/min/max/mean`; each filter step's operation +
    params + output `min/max`; `value_range`, relevance floor/ceiling, and region
    count after priming.
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
