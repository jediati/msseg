# Region embeddings as one selectable thing -- design note

2026-10-15. Status: **built** (the same day), phases 1-6 below; see §6 for
what landed differently from the proposal.

Several tools read "a vector per region" and compare two of them: the
magic fill and the blobber, the seam livewire, the edge model, the seam
model's pair block, the latent ring head and the polyline task's embedding
input. Each one picks its vectors, normalizes them and measures distance in
its own code. This note lists those places, proposes one module that
**enumerates, resolves and caches** region embeddings for every consumer, and
lays out a path to an **Inspect** panel (hover a region, recolour the picture
by similarity, look at the embedding space).

## 1. Where embeddings are read today

| Consumer | What it reads | Layer / normalization | Distance / pair op | Where |
|---|---|---|---|---|
| Magic fill, blobber (`mean`, `bhattacharyya`, `histogram`, `cosine`, `proba`) | table columns (by channel list), or the whole row minus positions and histogram bins, or `_pred` probabilities | z-scored **per slice** | euclidean / bhattacharyya / hellinger / cosine / tv | `magic_fill.row_vectors`, `pairwise` |
| Magic fill `learned` | edge-model p(diff) per arc | -- | arc weight | `tools.py:443` |
| Seam livewire tolls `feature`, `bhattacharyya` | `magic_fill.row_vectors` again | per slice | same | `seam_path.py:115` |
| Edge model (`-> edges` kinds) | base net hidden layer `spec.layer` (default -1) | the pipeline's training-set scaler | `|d|`, product | `edge_model.embed`, `_layer`, `pair_features` |
| Seam model pair block | `features` = rows z-scored per slice, **histograms included**; `net` = `edge_model.embed(...)[spec.layer]` | per slice / scaler | `|d|`, product | `seam_model.zscore_rows`, `seam_features` |
| Latent ring head | base net layer `LatentSpec.layer` | scaler | ring means, H0 in latent space | `context.embedding_of` |
| Context columns `source` | `all` / `ext` / `mean` column families | raw | ring reductions | `context.py:62` |
| Polyline **embedding** input slot | another task's net, a hidden layer picked by index | scaler | (feeds the seam pair block) | `artifacts.TaskProvider.base`, `layer_of` |
| Foundation features | `fm<hash8>__*` columns: already an embedding, stored as a column family | whitened PCA | (classifier input) | `foundation/pool.py` |
| Region encoder | `emb<hash8>__z*` columns (not wired in yet) | encoder's own | -- | `embedding.EncoderBundle.columns` |
| Coloring modes | `_pred` probabilities, `aux["pdiff"]` | -- | scalar per region | `panels/view.py:_region_scalar` |
| Edge-pairs experiment | `embed(est, X)` | scaler | pair logistic | `experiments/edge_pairs.py` |

### What is duplicated or inconsistent

1. **Three normalizations of "the raw row"**, each with its own column rule.
   `cosine` drops positions and histogram bins. `seam_model.zscore_rows`
   drops positions only, so it keeps the bins. The pipeline's scaler applies
   the model's `FeatureSubset` and training-set statistics. Per-slice
   z-scoring is fine within one slice, but it is **wrong across slices**: the
   same region gets different vectors on two slices. That matters as soon as
   an anchor from slice A recolours slice B.
2. **Three layer conventions.** `edge_model._layer` and `context.embedding_of`
   take a Python index (`-1` = last) and **silently fall back to the last
   layer** on an IndexError. `seam_model` takes the index modulo the depth and
   names columns `h<k>_*`. `artifacts.layer_of` takes a non-negative index
   and **refuses** an out-of-range one. A retrain to a shallower net changes
   what the first two read without a word; only the third says so.
3. **No caching across consumers.** `_predict_slice` embeds every row inside
   `predict_pdiff`. The latent head embeds the same rows again, the seam
   model again, and a provider slot again. Nothing keys the embedding by
   `(record commit, net)`.
4. **The magic fill cannot use the model.** The tool that most needs "these
   look alike" has no net-layer metric, even though `edge_model.embed` sits
   right next to it. `proba` is the only model-derived region metric, and it
   is the most compressed signal the net has.
5. **Column-family embeddings are invisible as such.** `fm<hash>__*`,
   `emb<hash>__*`, `hist*` and the context kinds are each a coherent vector
   block, but every consumer sees one flat row. You cannot say "cosine on the
   foundation block alone".
6. **Distances are spread over three modules.** `magic_fill.pairwise` has
   five kinds, while `edge_model.pair_features` and `seam_features` each
   rebuild `|d|` and product.

## 2. The proposal: `msseg.labeler.represent`

The name avoids `embedding.py`, which already holds the encoder bundle. The
module is pure numpy, holds no Tk code, and is pytest-covered like
`magic_fill`.

### 2.1 A reference: what to embed

```python
@dataclass(frozen=True)
class EmbeddingRef:
    source: str            # "columns" | "net" | "head" | "proba" | "pdiff"?
    provider: Optional[ProviderRef] = None   # None = the active task; else artifacts' refs
    block: str = "row"     # columns: "row" | "channels" | "family:<prefix>" | "names"
    channels: Tuple[str, ...] = ()           # block == "channels"
    names: Tuple[str, ...] = ()              # block == "names" (explicit)
    layer: Optional[int] = None              # net/head: 0 = the net's INPUT (subset+scaler),
                                             # k >= 1 = hidden layer h<k>; None = last hidden
    norm: str = "auto"     # "slice" (z per item) | "model" (the scaler) | "l2" | "none"
    def to_dict(self) / from_dict(d)          # rides view keys, specs and gesture meta
    def key(self) -> tuple                    # hashable, for caches
```

`layer` uses **one convention**: non-negative, 0 = what the net sees, hidden
layers from 1. That matches the `h<k>` column names, and `None` means "last".
`from_dict` maps the legacy forms: `EdgeSpec.layer = -1` → `None`,
`-2` → `depth - 1`, and the slot entry's 0-based `layer` → `layer + 1`.
Resolving a layer that does not exist **refuses with a reason**, as
`artifacts.layer_of` does today. No silent fallback.

`layer 0` is new and cheap. It is the model's own input after
`FeatureSubset` and `StandardScaler`, which makes it the right "raw features"
vector for **cross-slice** comparison. It also exists for a forest, whose
pipeline still standardizes.

### 2.2 Options: what can be embedded right now

```python
@dataclass(frozen=True)
class EmbeddingOption:
    ref: EmbeddingRef
    label: str             # "h2 · 16-d (region net)", "foundation fm3a1c…  (32-d)", "row (42-d)"
    dim: Optional[int]
    group: str             # "statistics" | "model" | "other tasks" | "learned columns"
    why_not: Optional[str] # greyed in the picker, with the tooltip

def options(table_names, stack, providers=(), conv=DEFAULT) -> List[EmbeddingOption]
```

This is the "look at the current model and populate" part. It is built from
things that already exist:

* **Statistics.** `row` (positions and histograms excluded, the current
  `cosine` rule). `channels` (the `mean`/`bhattacharyya` pair) and
  `histogram`. Each **column family** gets a block, found by a prefix
  registry that `FieldConventions` gains: `fm<hash8>__`, `emb<hash8>__`,
  `hist`, and the context kinds `ring_*` / `hop2_*` / `slice_*` /
  `nbr_class__*`. Every one of these spellings is already a reserved prefix,
  so the registry only collects them.
* **The active model.** `edge_model.hidden_widths(stack.clf)` gives
  `h1..hN`, plus `layer 0`. When there is a latent head
  (`_context_model`), its own net's hidden layers are offered under
  `source="head"`; its input is `X ++ latent`, built as `predict_latent`
  builds it. `proba` comes when `_pred` is current.
* **Other tasks.** `artifacts.TaskProvider` already reads another task's
  stack without activating it. Each provider adds its layers under "other
  tasks", with `Requirement.why_not` as the option's `why_not`, so the gate
  is the existing one.
* **Later.** The model library (`ProviderRef.library`) and a loaded
  `.msenc` encoder. Each is one more provider kind, and the module does not
  change.

A model introspection shim, `layers_of(estimator) -> [(index, width,
name)]`, takes over from the `hasattr(coefs_) / hasattr(params_)` switch in
`edge_model._hidden_layers`. That switch is the single place to teach about
a new net kind (a torch module with named layers, a CNN head).

### 2.3 Resolving: vectors for an item

```python
@dataclass
class Embedding:
    Z: np.ndarray          # float32[n_rows, d], row-aligned with the table
    ids: np.ndarray        # the table's feature_id column
    names: List[str]       # column names ("h2_00".., "fm..__mean_03"..), for readouts / pair blocks
    signature: tuple       # what Z is a function of
    default_distance: str  # "cosine" for net layers / fm, "euclidean" for z-rows, "tv" for proba

def resolve(ref, *, table, arcs, labels, stack, np, row_classes=None, conv=DEFAULT) -> Embedding
```

`resolve` reuses `artifacts.input_table` (context columns for a net that
reads them), `TrainingSetBuilder.feature_matrix` and `edge_model.embed`.
Nothing new is computed. It just happens in one place.

The **signature** is `(rec["commit"], ref.key(), net_hash(clf) or
names_hash, ctx.key(), store.rev if the net reads labels-as-context)`. That
is the `TaskProvider.signature` recipe with `net_hash` in place of `id()`,
so a reloaded identical pickle still hits the cache. The cache is
`record_keys.RecordCache`-shaped: a few entries per item, keyed by
signature. The first consumer pays the forward pass, and every later one
(prediction, edge model, magic fill, Inspect) gets it free.

### 2.4 Distances and pair operations

`magic_fill.pairwise` moves to `represent.distance(Z, I, J, kind)`, and
`magic_fill` re-exports it. Two kinds are added: `dot` (the "product
similarity" you asked for) and `l2n` (euclidean on L2-normalized rows).
`represent.pair_block(Z, a, b, ops=("absdiff", "prod"))` replaces the two
hand-built blocks in `edge_model.pair_features` and
`seam_model.seam_features`. Barrier, contact and pdiff stay where they are,
because they are arc terms rather than embedding terms.

## 3. Migration, consumer by consumer

Each step stands alone. Each one keeps pickles, session documents and gesture
meta readable.

| # | Consumer | Change | Compat |
|---|---|---|---|
| 1 | `represent` module + tests | `EmbeddingRef`, `options`, `resolve`, `distance`, cache; parity tests that today's vectors come out byte-equal | none |
| 2 | Magic fill / blobber | the metric row becomes **embedding + distance**; the old metric names are presets (`mean` = channels/slice/euclidean, `cosine` = row/slice/cosine...). **New: any net layer, any family, with cosine/dot.** | `view.magic.metric` still read; gesture meta gains `embedding` only when not a preset |
| 3 | Seam livewire tolls | `feature`/`bhattacharyya` go through the same picker | `toll` names kept |
| 4 | Edge model | `fit_edge_model` / `predict_pdiff` take an `Embedding`; `_predict_slice` resolves once and passes it (`emb=` already exists) | `EdgeSpec.layer` read via `from_dict`; `net_hash` gate unchanged |
| 5 | Latent head, seam model `net`, provider slot | `embedding_of`, `seam_features`, `TaskProvider.base` resolve through `represent`; the three layer conventions collapse into one | pickled `LatentSpec.layer` / `SeamSpec.layer` / slot `layer` mapped in `from_dict`; the silent fallback becomes a logged refusal |
| 6 | Context `source` | `all/ext/mean` become column-family blocks; `fm` / `emb` become possible sources | `SOURCES` kept as aliases |

Step 2 is where the payoff is. "Flood while the region net's h2 stays within
cosine 0.1 of the seed" is a better similarity than z-scored means once a
model exists. It is the inverse of what `proba` does: the representation
before the classifier throws information away.

## 4. The Inspect panel

### Phase A: similarity as a coloring mode (small, no new panel)

* `_region_modes` gains **`similarity`** whenever an embedding resolves. The
  picker (embedding + distance) sits on the overlay row next to the region
  mode, with `view.inspect = {embedding, distance}`.
* **Anchor.** The canvas hover callback (`canvas.on_hover`, already wired
  for the class panel) sets the anchor region. A click **pins** it, and
  Shift-click adds to the pinned set (similarity = max over the set, or
  similarity to the set's mean). A third anchor kind is "class k centroid":
  the mean embedding of the regions labelled k on this item, or across all
  items.
* `_region_scalar` returns `sim(Z[anchor], Z)` per region, and the existing
  ramp renders it. One `n×d` mat-vec per hover is cheap: at 1e5 regions and
  d = 64 it takes about 1 ms. Throttle it to the existing hover span.
* **Cross-item.** A pinned anchor keeps its **vector** rather than its id.
  Navigating to another slice or ROI recolours that item by similarity to
  the pinned vector. This needs a cross-item-stable embedding (`layer >= 0`,
  `fm`, `emb`, `proba`), so `norm="slice"` options are greyed there with
  the reason.

### Phase B: the Inspect tab (right-hand notebook, next to Analysis)

* **Neighbours list.** The top-k nearest regions to the anchor, across the
  enrolled items, as a Treeview: item, region, distance, annotated class,
  predicted class. Double-click goes to `_goto_region`, the same as the
  error list (`panels/analysis.py`). This answers "where else does the model
  think this looks the same", and it finds label errors.
* **Scatter.** A 2-D projection of the embedding over the current item or a
  sample across items. PCA by default; UMAP when importable, as an optional
  extra, the way Optuna is. Points are coloured by annotation, prediction,
  uncertainty or similarity. Linked brushing: hovering the canvas highlights
  the point, and a lasso on the scatter selects regions on the canvas, which
  can be **committed as a taps gesture** like a magic fill. Embed matplotlib
  via `FigureCanvasTkAgg`, which cellseg already uses, or draw on a plain Tk
  canvas for speed with ~1e5 points.
* **Unit view.** `unit h2[5]` as a scalar coloring mode: one hidden unit's
  activation per region, to see what a unit responds to.

### Phase C: diagnostics over options

For every option in `options()`, report the leave-items-out kNN /
nearest-centroid accuracy against the annotations (the helper exists in
`test_embedding.py`), plus the silhouette per class. That gives a table that
says which layer, family or foundation block separates the classes best. It
also doubles as the **transfer probe** the region-encoder note lists as
unbuilt.

## 5. Open questions

* **Does `norm="auto"` mean "model" whenever a model exists?** That would
  change what `mean` / `cosine` magic fills do on a trained task. The
  proposal keeps the presets on `slice` and makes `model` explicit.
* **Seam-level embeddings** (the seam descriptor, `pool_seams`) are the same
  idea for seams. `represent` should take the object kind (`region` |
  `seam`) as a parameter from the start, even if only regions are wired.
* **Hover on very large mspath overviews.** The mat-vec is cheap, but the
  LUT repaint over a 1e6-region overview may not be. Measure it before
  promising live hover there, with pin-to-recolour as the fallback.

## 6. As built (2026-10-15)

**`msseg.labeler.represent`** holds `EmbeddingRef` (`to_dict` writes only what
differs from the default), `legacy_layer` / `resolve_layer` (one convention:
0 = the net's scaled input, k = `h<k>`, None = last; a missing layer is a
`ValueError` naming it), `layers_of` / `depth_of`, the column-family registry
`families` / `family_of` (it lives here rather than on `FieldConventions`,
because naming the foundation / encoder / context prefixes there would make
`fields.py` import those modules), `options`, `resolve`, `signature_of`,
`EmbeddingCache` (a `RecordCache` keyed by signature; `proba` is never
cached), `distance` (`pairwise` plus `l2n`), `similarity` / `to_unit` /
`top_k`, `pair_block`, `project_2d` (seeded, sign-fixed PCA) and
`separability` (leave-items-out kNN + nearest centroid in cosine geometry,
silhouette). `edge_model.embed_layers` is the one forward pass (`embed` =
its hidden part). In `options` the LAST hidden layer is offered as `layer
None`, so a pick survives a retrain that changes the depth and it shares a
cache entry with the edge model's default.

**Consumers moved.** `edge_model._layer`, `context.embedding_of` and
`seam_model.seam_features` resolve layers through `represent` and REFUSE a
missing layer (previously: silent last-layer fallback / modulo). The pair
terms of the edge and seam models go through `pair_block` (column order
unchanged). `_predict_slice` resolves the edge model's layer through the
app's `EmbeddingCache` (`ClassifierMixin._embedding`) and passes `emb=`, so
the magic fill and Inspect reuse that forward pass.

**Magic fill.** The metric combo lists the named metrics, a separator, then
every usable `options()` entry; picking one floods over those vectors
(`build_ladder(..., vectors=(X, kind))`, metric `"embedding"`) by the
distance beside it (`cosine` / `euclidean` / `l2n`). `view.magic` and the
gesture meta gain `embedding` + `distance` only for an embedding fill.

**Inspect** (`panels/inspect.py`, `InspectMixin`, the sixth tab; the TAB is the
pin tool -- open, a canvas click pins and the colouring switches to
`similarity`; closed, the drawing tool and the previous colouring are back;
key I toggles the tab. Its sections run embedding, scatter, evaluation, then
anchors and the nearest-regions list at the bottom):
the `similarity` and `embedding unit` region modes (no prediction needed),
pins (click, Shift adds, Escape clears; a pin keeps its vector, the task and
the net's signature, so a retrain or another embedding retires it), live
hover when nothing is pinned (the similarity LUT is swapped in place on the
canvas's overlay list -- one mat-vec + one LUT per hover tick), a class
centroid pin, the nearest-regions list (this item or every computed item for
a cross-item embedding; double-click goes there, "Label" commits one taps
gesture per item, a sample), the PCA scatter (PIL raster, up to 60k points;
coloured by annotation / prediction / similarity; canvas hover rings the
point, scatter hover outlines the region, a box-drag selects and "Label
selection" commits), and "Evaluate embeddings" (gathered on the Tk thread,
scored on a worker, best first; double-click inspects that embedding). The
view keeps `view.inspect` (`embedding`, `similarity`, `reduce`, `live`);
pins are not saved. `_region_at` is the placement-correct region lookup (the
provider's `LabelLayer.id_at`); the class panel's hover uses it too, which
fixes its image-coordinate indexing on a placed (mspath) item.

**Not done / different:**
- the seam livewire's `feature` / `bhattacharyya` tolls still call
  `row_vectors` directly (the same vectors `resolve` returns) and got no
  embedding picker;
- the context `source` (`all` / `ext` / `mean`) is unchanged;
- `artifacts.layer_of` keeps its 0-based slot convention (it already
  refused), so the session documents are unchanged;
- the scatter is PCA only (no UMAP) and selects by box, not lasso;
- in a polyline task the similarity colouring is not drawn (its overlay path
  draws the seams); the Inspect tab's list / scatter / evaluation still work;
- labelling regions on several items from the list is one undo step per item.

Tests: `test_represent.py` (parity with the per-tool vectors, layers,
options, cache, similarity, PCA, separability, refusals), the vectors path
in `test_magic_fill.py`, the coupon labeler selftest's embedding-fill and
Inspect blocks, and the mspath labeler selftest's placed-item block.
