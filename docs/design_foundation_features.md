# Foundation-model region features

Status (2026-10-13): **phase 0 built** (`msseg.labeler.foundation` + the
experiment `packages/mspath/experiments/fm_ablation.py`), first run below;
**phase 1 landed** (the labeler, opt-in, both task kinds). The go/no-go
is answered by use (2026-10-13): in the user's own testing the foundation
columns perform as well as or better than the hand-made feature combinations.

## Why

The region classifier sees only the MSC statistics row: intensity,
scale-space and histogram aggregates, plus ring context. What the row cannot
describe is tissue **architecture**: gland layout, nuclear crowding, stroma
weave. A ViT pretrained on images (DINOv2/v3 today; pathology models are a
later option) encodes exactly that, and it keeps one contextual token per
patch. We pool those tokens per MSC region and append the result as ordinary
columns. The MSC regions, the seams and the statistics row are all kept.

## Tiles, patches, tokens

- A ViT reads a **tile** (`spec.tile`, default 448 px). The tile sets the
  context (attention spans the tile and nothing beyond) and the cost (one
  forward pass per tile, roughly quadratic in its token count).
- The model splits the tile into **patches** of 14 px (DINOv2) or 16 px
  (DINOv3). The patch is the resolution: it is fixed by the trained
  patch-embedding weights, and **one token per patch survives every block**.
- The CLS and register tokens are dropped. The tile is only how the model is
  fed; the output we keep is a grid of tokens.
- `extract_grid` tiles an item with overlap (`overlap_tokens`, default 2). A
  tile *owns* the tokens nearest its centre: an owned-range boundary sits
  mid-way through the overlap, so a kept token is never at an interior tile
  edge, where it would lack context on one side.
- The result is ONE `(Gh, Gw, D)` grid over the item + halo. Cell `(i, j)`
  covers input pixels `[i*s, (i+1)*s)`, with `s = patch / scale`.
- `scale: 2` upsamples the image before the model, so a patch covers half
  the pixels. That buys resolution at the price of showing the model an
  apparent magnification it was not trained at.
- Intermediate blocks (`layers`) are concatenated. Earlier blocks are more
  local (less leakage from neighbours); later blocks are more semantic.

## Regions gather tokens (`pool.py`)

Each label pixel falls in one cell. `A[r, t]`, the pixels of region `r` in
cell `t`, is built sparse from one `np.unique` over `region*T + cell`. So a
3-pixel region and a 30 000-pixel one both reduce to a weighted mean over the
cells they touch: no padding and no MIL bags. The pools:

| pool | weight per cell | meaning |
|---|---|---|
| `mean` | pixels | the area-weighted mean token |
| `purity` | pixels × fraction of the cell covered | a cell straddling a boundary counts less |
| `interior` | cells covered wholly | falls back to `mean` when the region has none |
| `boundary` | cells shared with another region | falls back to `mean` when the region has none |

Two reliability columns ride along:
- `tokens`: the area in cell units; below 1 the region is a fraction of one
  token.
- `purity_frac`: the mean coverage of the cells the region touches.

Cell capacity is counted from the raster, so a fractional stride and the
cells overhanging the item edge are exact.

The pooler's offset is the **halo**. The grid is computed over the halo-padded
read, so tokens near the item edge keep their context, while the record's
labels are halo-cropped.

## Seams gather the cells they run through (polyline tasks)

A polyline task's unit is a **seam**, not a region, so it gets its own pool
(`pool_seams`). A crack separates two pixels, and it adds half its weight to
each flank pixel's cell. A seam is then the crack-weighted mean over every
cell it runs through. A seam lying on a cell edge reads both cells equally;
one inside a cell column reads that column. `seam_cells`, the length in cell
units, is the seam block's reliability column and its size bucket.

Region fm columns also reach a seam through the seam model's **pair terms**
(`|a - b|`, `a * b` over the flank descriptors). This happens whenever the
pair block reads the table rather than an embedding input's net. The
experiment gets both by appending fm columns in `_context_table`, the hook
`_seam_items` already streams through, so the descriptor is still built by
the task's own code.

## Naming, projection, compatibility

- The grid is reduced to `dims` (64) by a whitened PCA (`project.py`), fitted
  once and **frozen onto the spec** (`Basis`).
- Columns are named `fm<hash8>__<pool>_<kk>`, `fm<hash8>__tokens` and
  `fm<hash8>__purity_frac`. `hash8` covers the extract settings plus the
  basis digest, so a model trained over another backbone's or basis's
  columns is refused by the existing name-set compat gate. This is the
  `EncoderBundle` trick (`embedding.py`).
- `extract_key()` excludes `dims`, `pools` and the basis. A grid cache keys
  on it, so a refit or a new pool choice reuses every grid.
- The `fm` prefix collides with no context kind, so `context.source_columns`
  treats the fm columns as base columns. Ring/hop context over them comes for
  free when they are appended before `context.augment`.
- Each pool is ONE Optimize mask group (`schema_entries`). The reliability
  columns are their own group.

## Cache and determinism (`cache.py`)

- A grid is a function of the pixels and the extract settings only, not of
  persistence, the statistics spec or the projection. It survives every
  re-prime, re-measure and refit.
- It is expensive (a GPU pass), and fp16 attention is not bit-reproducible.
  So it is computed once and kept: an LRU in memory, plus
  `<app data>/fm_cache/<sha1>.npz` on disk (fp16).
- A recompute could move predictions slightly, which is why the disk cache is
  the source of truth, not a convenience.

## Hardware (RTX 5070 Laptop, 8 GB, sm_120)

- Torch needs the cu128 wheels (already the case for `torch_mlp`). Attention
  goes through torch SDPA; xformers / flash-attn wheels are not assumed for
  sm_120.
- Frozen inference fits easily up to ViT-L (and ViT-g at small batches).
  Fine-tuning is out of scope: full fine-tuning does not fit, and LoRA is not
  worth it yet.
- `extract_grid` halves its batch on OOM. `release()` returns VRAM to the
  driver for the diffg bank and `torch_mlp`.
- Storage, not compute, is the limit for whole slides: about 400 KB of raw
  tokens per 224² tile at D=1024. Hence the cache covers enrolled items only.

## Phase 0: the experiment

```
PYTHONPATH="packages/mslabeler/src;packages/mscoupon/src;packages/msseg-viz/src;packages/mspath/src" \
    python packages/mspath/experiments/fm_ablation.py [out] --model dinov2_vitb14_reg --scale 1 --layers -1
```

- It loads the mspath labeler's last session, activates `--task` (default
  the active one), primes, and materialises the records as Train does. It
  then extracts and caches the grids and fits the basis.
- A **polyline** task scores its own seam model leave-slides-out on: seam /
  + fm pair / + fm crossing / both / crossing only.
- A **region** task pools and scores one fixed dense net leave-slides-out on
  these column subsets:
  - stats
  - stats + fm mean
  - stats + fm all
  - fm only
  - stats + ring
  - stats + fm + ring
- Scores are reported overall and **by region size in tokens** (<1, 1–4,
  4–16, ≥16), with extraction s/item and peak VRAM.
- `--fake` swaps in a random patch-local backbone for a plumbing smoke test
  with no download.
- **Go** if there is a clear held-out log-loss gain overall, or in a size
  bucket that matters, at an acceptable time per item.
- Caveat: the PCA basis is fitted over every item's tokens, held-out slides
  included. That is unsupervised and label-free, but transductive.

### First run (2026-10-13, DINOv2 ViT-B/14-reg, one mspath session)

| task | baseline log-loss | best with fm | bal. acc |
|---|---|---|---|
| region, 1 item, 529 labelled regions | stats 0.765 | stats + fm mean 0.039 | 85.5% -> 98.0% |
| polyline, 2 items, 1840 labelled seams | seam 0.866 | seam + fm crossing 0.182 | 83.9% -> 94.1% |

**Not a go decision yet.** With fewer than 3 labelled slides `make_cv` falls
back to random folds. A DINOv2 token sees ~448 px of context, so a held-out
region's neighbours sit in the training fold with near-identical tokens,
which inflates exactly the fm variants. The gain appears in every size
bucket, which is encouraging. Rerun on >= 3 labelled slides.

Other findings from this run:
- `scale 2` did not help, and costs 6-7 s/item and up to 1.9 GB.
- Ring context adds nothing on top of the fm columns.
- For seams, the crossing pool and the flank pair terms overlap.

At `scale 1` a 2048² item takes 0.6 s and 0.6 GB.

**Install trap:** `pip install timm` pulls the newest torchvision, which
pulls a CPU-only torch from PyPI over the cu128 build. Install
torch/torchvision from `--index-url https://download.pytorch.org/whl/cu128`
together (torch 2.11.0 <-> torchvision 0.26.0).

## Phase 1: the labeler (landed 2026-10-13)

Built before the go/no-go rerun on >= 3 slides, at the user's request. It
is opt-in: with the group off, every document, pickle and row is what it
was before.

**One group, two homes.** The Features tab's "8. Foundation features"
(`panels/foundation.py`, `FoundationMixin`) describes the active task's NEXT
model: model, layers, scale, dims, region pools, and *Refit projection*.
- A **region** task carries the spec inside its `ContextSpec`
  (`ContextSpec.foundation`). So the pickle (`stack["context"]`, written
  `to_dict(full=True)` with the basis), the model record and the session
  view (`to_dict()`: the spec + `hash8`, never the basis), the compat gate
  (`_expected_names_for` -> `context.all_column_names`) and Optimize's
  feature mask (`context.schema_entries`, one group per pool) all see it
  with no new keys.
- A **polyline** task carries it inside its `SeamSpec`
  (`SeamSpec.foundation`). That covers the task view's `seam_spec` and the
  `SeamBundle` (`SeamModel.to_dict` writes the spec in full).
- Only the active kind's spec is pushed onto the shared group
  (`_apply_context_spec` / `_apply_seam_spec`).

**Rows.**
- `classifier._context_table` appends the region pools BEFORE
  `context.augment`. The context kinds never reduce over `fm` columns
  (`source_columns`): 4 x 130 ring columns, for no measured gain.
- `seam_classifier._seam_items`, the one choke point for seam
  train / classify / evaluate, appends the region pools to the table (the
  pair terms read them when they read the table, not an embedding input's
  net), then the crossing block (`fm<h>__seam_<kk>`, `seam_cells`).

**The projection is fitted at Train.**
- `_training_set` -> `_resolve_context_spec`, and `_seam_items`, fit over
  every ready item's grid when the spec has no basis.
- The fitted spec becomes the group's (`_fm_fitted`). The model keeps it,
  and loading a model restores it, so Classify rebuilds the same names.
- Optimize and the size sweep now read the Model-tab spec AFTER
  `_training_set` for that reason.
- *Refit projection* forgets the basis: a new feature set, which the gate
  refuses old models on.

**Pixels.**
- An OPTIONAL provider pair, `item_pixels(key, rec, np) -> (array, halo)`
  and `pixels_halo(key, rec)` (`protocols.py`).
- mspath's `SlideRegionProvider` reads `SlideEngine.read_item` with the
  record's halo, so the model sees what the MSC saw.
- The coupon has no `item_pixels` yet and refuses with a message.
- Grids come from `TokenCache` at `<app data>/fm_cache`; projected grids are
  memoised per item.

**Fixed on the way.** `_save_classifier_to` never wrote the model's scope.
Every mspath (level-scoped) region pickle came back "trained at no scope"
and Classify refused it. It now passes `scope=_clf_scope` (scope-less
pickles are unchanged).

**Not built yet:**
- A worker thread: extraction runs on the Tk thread under the compute badge,
  like record streaming. The first Train loads the backbone (about 7 s),
  then about 0.6 s per uncached 2048² item.
- An `fm` stage-strip box.
- The coupon's `item_pixels`.
- The core / CLI / 3D matcher are not planned: a `ChannelStats` cell is a
  fixed 24 B for every slot, and the CLI cannot run torch.

Tests: `test_foundation_spec.py` (spec plumbing) and the mspath labeler
selftest's foundation block (fake backbone: pixels + halo, fitted basis,
names, gate, classify, view, pickle round trip, cache reuse, polyline seam
block).
