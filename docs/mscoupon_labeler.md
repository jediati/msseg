# mscoupon labeler (`mscoupon-labeler`)

The labeler is the viewer (`mscoupon-gui`, see [mscoupon_gui.md](mscoupon_gui.md))
with an annotation panel: the user paints **classes** onto the living MSC regions
of a slice, trains a per-region classifier on the statistics table, and exports
training sets. This page covers the annotation model and the drawing tools;
the engine, session file and config schema are the viewer's.

```bash
mscoupon-labeler                  # GUI
mscoupon-labeler --selftest       # Tk wiring check, no engine needed
# from a source checkout (both namespace packages on the path):
PYTHONPATH="packages/mscoupon/src;packages/msseg-viz/src" python -c "from msseg.mscoupon.labeler import main; import sys; sys.argv=['x','--selftest']; main()"
PYTHONPATH=packages/mscoupon/src pytest packages/mscoupon/tests/test_labeling.py packages/mscoupon/tests/test_magic_fill.py
```

## Annotations are gestures, not region ids

An annotation (`labeling.Interaction`) is one gesture in **image coordinates**
bound to one slice (`"folder/basename"`) and one class. Nothing stores a region
id: on every render the slice's gestures are rasterized against whatever
label raster is current (`touched_ids` → `resolve_slice`), later gestures
painting over earlier ones on any region both touch. That is what lets a
persistence change, a filter edit, or a switch between `msc` and
`merge_forest` simplification re-resolve the same annotations onto the new
decomposition. `annotations.json` (v2) holds the raw geometry; it also rides
the session autosave and is written beside every exported config.

Optional per-gesture **`meta`** (a JSON dict) carries provenance for display —
today only the magic fill writes it. Resolution never reads it, so a stale
`meta` can never change what a gesture paints; files without it are unchanged.

## Tools

| Tool | Gesture | Paints |
|---|---|---|
| squiggle | polyline (a click is a tap) | every region under the line |
| box | drag a rectangle | every region overlapping it |
| lasso | drag a closed polygon | every region under the filled polygon |
| magic | press, drag up/down, release | a similarity flood from the pressed region |
| *SHIFT + drag* | a box, any tool | **accepts** the classifier's predictions under it as `taps` |

Hotkeys: `1..4` arm a class, `0`/Escape disarm, `M` selects magic, `Tab`
toggles the overlays, `Ctrl-Z`/`Ctrl-Y` undo/redo, `R` train + classify,
`C` classify. Middle/right drag always pans; a right-click opens the
annotation menu for the region under it.

### "Will be painted" preview

While any gesture is in flight the regions it would paint on release are
shown on a **transient canvas layer** in a brightened, opaque version of the
class color (the accept box shows each region in its *predicted* class color).
The layer ignores the overlay-alpha slider — the preview is the point while it
shows — and is dropped on release or Escape. Squiggles preview incrementally
(only the newest segment is rasterized); boxes and lassos recompute per move,
sampling with a stride once the slab exceeds ~1 Mpx so a full-image box on a
3232² slice stays around 10 ms. The commit is always the exact rasterization.
**Escape** now abandons the gesture in flight (any tool); with nothing in
flight it disarms the class as before.

### Magic fill

Press on a region (the **seed**) with a class armed. The fill grows over the
**living-region adjacency graph** at the current persistence: a region joins
when there is a path from the seed whose every hop is under the threshold.
Drag **up** to raise the threshold (more regions), **down** to lower it;
the canvas HUD shows `t`, how many regions are in, and their pixel count.
Release paints; Escape abandons.

*What is compared* (the **Magic:** row under the tool selector):

| metric | measures | needs |
|---|---|---|
| `mean` (default) | \|Δmean\| over the chosen channels, each z-scored by that column's spread on the slice | `mean_<channel>` |
| `bhattacharyya` | Gaussian overlap from mean and std per channel | `mean_`, `std_` |
| `barrier` | saddle height above the seed's extremum: the persistence-style flood anchored at a point | MSC region arcs (`ext_filtered`) |

| mode | compares |
|---|---|
| `anchor` (default) | every candidate with the **seed** — no drift |
| `chain` | each region with the neighbour it grows from — follows gradients |

Channels are the names of the statistics spec (`base`, `blur_s1.5`, …),
comma-separated; unknown names are ignored, none valid falls back to `base`.

*How it is computed.* At the press, `magic_fill.build_ladder` turns the metric
into one weight per arc and runs a **priority flood** from the seed
(`growth_order`: a bottleneck/minimax Dijkstra whose ties are broken by the
most seed-like frontier region), giving every region the threshold at which
it joins — the **join ladder** — *and* the order in which the flood admits
them. A drag tick is a rank on that order, i.e. a prefix of it
(`drag_to_rank`: linear near the start so single regions are reachable,
quadratic further out so a long drag sweeps a thousand-region ladder), so
every pixel of drag adds or removes one connected region. The prefix, not a
threshold-closed set, is deliberate: a bright outlier seed makes its first
neighbour's dissimilarity the bottleneck for most of the slice, so hundreds
of regions share one join value and a plain threshold jumps from one region
to half the slice at that rung; the HUD's `t` is the join value of the last
region admitted. The first threshold is the
one last released with the same metric/mode/channels in this session, else a
natural break in the ladder (`initial_rank`: the largest relative gap within
its first 5 %).

*Adjacency.* The engine stores `rec["arcs"]` from
`Msc2DPipeline.region_arcs()` — MSCEER's `livingRegionArcs()`: living-region
pairs joined by a saddle, with the saddle value, in both `msc` and
`merge_forest` modes. An extension without it (an older wheel) makes the fill
derive 4-neighbour pixel adjacency from the label raster on the first press
(`arcs_from_labels`, ~0.3 s at 3232², cached on the record); `barrier` is then
unavailable and falls back to `mean`. In `msc` mode saddles with fewer than
two living extremum arcs are dropped by MSCEER, so a few regions can be
unreachable; the HUD's `k/n` shows how many are.

*What is stored.* The release commits **one `taps` interaction with a point
per grown region at its seeding extremum** (`ext_x/ext_y`, falling back to the
region's first pixel), plus `meta = {tool: "magic", seed, seed_id, threshold,
metric, mode, channels, n_regions, arcs}`. So the fill re-resolves after a
persistence change like every other gesture: on a coarser decomposition
several points collapse into one region; on a finer one only the sub-regions
containing a stored point stay painted. The annotation row reads
`#12 magic (37)` and hovering it shows the taps plus a dashed ring at the seed.
One undo step removes the whole fill.

## Classifier and exports

Unchanged by the above: Train/Classify on the per-region statistics table
(positions excluded), SHIFT-accept to turn predictions into annotations, CSV
export (one row per living region, class 0 kept as negatives), and the
image training set (`train/` raw TIFFs + `labels/` per-pixel class masks,
annotations winning over predictions).
