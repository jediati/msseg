# Design note: gradient-orientation histograms (deferred)

Per-region intensity histograms landed as `statistics.histogram` (see
`CLAUDE.md`, "mscoupon histograms"): K bins over a fixed range per channel,
kept in a uint32 side table of `msseg::ChannelStats`, merged bin-wise, and
projected as `hist<kk>_<channel>` columns after the extremum block. This note
records how a **gradient-orientation histogram** (a HOG-like descriptor per
region) would ride the same machinery, and why it was not built in the same
pass.

## What it is

For a channel `x` at sigma `s`, the orientation `theta = atan2(dy, dx)` in
`[0, 2pi)` (or `[0, pi)` if sign is folded), binned circularly into K bins and
**weighted by the gradient magnitude** `sqrt(dx^2 + dy^2)`, so flat pixels do not
vote. The result is a K-vector per region that is invariant to intensity
scale and reads the texture's dominant direction and anisotropy.

## Seams it passes through

1. **Bank request.** The components come from diffg's `GradientComponents`
   kind (`grad_s<σ>_x`, `_y`). `resolve_stat_channels` needs a new derived kind
   (say `orient`) that expands to K *virtual* channels or, better, to ONE
   channel carrying a histogram and no aggregates.
2. **Accumulation.** `ChannelStats::add` bins a *value*; orientation needs a
   *pair* of rasters (dx, dy) and a *weight* per pixel. Add a second table
   alongside `hist_`: `orient_` of `n_regions × n_orient × K` float32 (weighted
   sums, not counts) with its own `orient_slots_` naming the (dx, dy) slot
   pair. The circular bin is `floor(theta / (2pi/K)) mod K`; the fold-to-pi
   variant is a flag on the request.
3. **Merge.** Weighted sums add bin-wise like counts, so `merge_region`,
   `append` and the matcher need only the extra table; `assembly.py` mirrors
   it with `np.add.at` on an `(n, K)` block under `chan_col(name, "orient")`.
4. **Projection.** Columns `orient<kk>_<channel>` after the histogram block,
   normalized by the region's total weight (not the area) so the vector is a
   distribution; a region with no gradient at all reports zeros.
5. **ML.** The columns carry `channel` in the schema, so the feature groups
   keep them together with the channel's other reductions. A magic-fill
   metric would compare them with a *circular* distance (e.g. the minimum over
   cyclic shifts of Hellinger) if rotation invariance is wanted, or plain
   Hellinger if not.

## Why it was deferred

* It needs a per-pixel *weight* path in the accumulator, which the intensity
  histogram does not; that is a new hot-loop shape, not a parameter.
* The GPU statistics path has no histogram reduce at all yet; adding two
  kinds of CPU-only vector statistics at once doubles the surface the GPU
  path silently lacks.
* Colour input makes the orientation question richer (Di Zenzo gives a
  colour gradient *direction* too, which diffg does not expose -- it emits
  eigenvalues only), and the right choice between per-plane and colour
  orientation deserves a look at real histology first.
