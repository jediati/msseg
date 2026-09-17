# Design note: filter-chain input/output types

The filter chain has exactly one carrier type — `diffg::Image<float>`: one
plane, continuous — from stage 0 onward. Every stage that naturally produces
something else reduces in place, through a *parameter of an unrelated
operation*. `hessian_eigenvalues` computes two eigenvalue planes and returns one
via `component`. `edges` computes a magnitude and a mask and returns one via
`output`. `label_components` widens int32 ids to float, so a following `blur` is
legal and meaningless. And a multi-plane input whose chain has no leading
`color` stage gets one **synthesized at the front**, with no trace in the config,
the UI, or the log.

diffg does not have this restriction. `apply_filter_bank` and
`structure_eigenvalues` take a `MultiImageView` and are natively multi-channel:
per-channel kinds replicate per input plane at `channel_in_request =
input_channel * k + slot`. The *measurement* path already exploits this —
`statistics.channels` with `source: "color"` emits `blur_c0_s1.5`,
`blur_c1_s1.5`, … The *transform* path does not. **That asymmetry is the
subject of this note**: the same library, reached two ways, is multi-channel on
the way that measures and single-channel on the way that transforms.

The concrete motivation is mspath. Its default topology chain is
`optical_density → blur 1.5 → edges 1.0`, and the total OD is dominated by
hematoxylin — so nuclei are far more intense than the pinkish eosin tissue whose
architecture the regions are supposed to enclose. Nuclei carve their own basins
and their rings dominate the decomposition. The chain has no way to say "the
eosin contrast, damped by the hematoxylin", because it cannot carry two planes
between two stages.

This note was written before any of it was built. Stages 1 and 2 of §7 have
since landed and are marked there; the rest stands as proposal. §7 says what
each step collides with, and records the two things the build found that the
design had not.

## 1. The coercion inventory

Eleven sites. Each is a place where a type changes with no record.

| # | Site | Coercion |
|---|---|---|
| 1 | `color_stage.cpp:323` `plan_color_chain` | C planes → 1, synthesized, invisible, `default_color_method` |
| 2 | `filter_stage.cpp:198` | every stage after index 0 is scalar by construction |
| 3 | `tiff_io.cpp:117` `read_tiff_float32` | `read_planes_impl(...).scalar()` — planes 1..C−1 dropped |
| 4 | `filter_stage.cpp:94` `hessian_eigenvalues` | 2–3 eigenvalue planes → 1 via `component` |
| 5 | `filter_stage.cpp:105` `structure_eigenvalues` | same, via `component` |
| 6 | `filter_stage.cpp:116` `edges` | {magnitude, mask} → 1 via `output` |
| 7 | `filter_stage.cpp:87` `zero_crossings` | boolean → float 0/1 |
| 8 | `filter_stage.cpp:154` `label_components` | int32 ids → float |
| 9 | `color_stage.cpp:238` `optical_density` | naturally C→C, always projected to 1 |
| 10 | `color_stage.cpp:213` `hsv` | computes all three, returns one |
| 11 | `color_stage.cpp:147` `dizenzo` / `structure` | both eigenvalues computed, one returned via `eigen` |

They are not one problem. **1, 3–6 and 9–11 are lossy**: work is done and thrown
away, and the thing thrown away is often the thing a workflow wants. Twelve
derived statistics channels exist precisely because a single scalar cannot
separate a void from a shallow dip — and then the transform chain reduces to a
single scalar anyway.

**7 and 8 are kind-laundering**: nothing is lost, but the result is no longer the
type it claims to be. A label raster and an intensity raster are both
`Image<float>`, and `blur` accepts either. This is the more alarming of the two
categories and the *less* urgent one, because **no current consumer
distinguishes** — there is no code today that would behave differently if it
knew. A `FieldKind` tag would be a type system with no theorems. Deferred in §7,
and deliberately not the headline.

Found while writing this: **mspath's declared default colour method never takes
effect.** `SlideEngine.prime_item` reads `profile.get("color_method",
"luminance")` (`mspath/engine.py:203`, `:209`) and `_preview_chain_key` the same
(`mspath/app.py:1128`), but nothing writes a top-level `color_method` —
`profile_params_json` writes it under `input.color.default_method`
(`session.py:198`), which is what the coupon engine reads
(`mscoupon/engine.py:345`). mspath's default conversion is therefore always
luminance, masked only because mspath always writes an explicit `color` card at
index 0. Delete that card and the chain silently converts by luminance instead
of the configured method. This is site 1 with the volume turned up: a silent
default that is also the *wrong* silent default, and no test caught it because
no test deletes the card. **Fixed 2026-09-17**: `mspath.engine.default_color_method`
reads the canonical key through the coupon session reader, and
`_profile_from_ui` carries the `input` block through the UI round trip so the
three readers agree. The value is still `luminance` until a profile says
otherwise — mspath has no colour-input section — but it is no longer ignored.

## 2. The model: `StageIO` and `plan_chain`

The proposal is not a new carrier struct. It is a **declared signature per
stage** and a **plan** computed from it:

```cpp
struct StageIO {
  enum class In  { Any, ScalarOnly, PlanesOnly };
  enum class Out { Same, One, KPerPlane, K };
  In in; Out out; int k = 1;
  bool liftable;      // a 1->1 stage MAY run component-wise over C planes
};
StageIO stage_io(const FilterParams&, std::size_t in_channels);

struct StageRecord { std::size_t in_c, out_c; bool lifted, synthesized; };
std::vector<StageRecord> plan_chain(const std::vector<FilterParams>&,
                                    std::size_t channels,
                                    const std::string& default_method);
```

What `plan_chain` has to earn is the same thing `resolve_stat_channels` earns by
being diffg-free: **it is computable with no raster.** Config validation runs
long before a slice is loaded; the GUI draws cards before a Run; the runner needs
the same answer at execution. One function, three callers, no drift. That is the
lesson `feature_fields(params_json)` already taught when it replaced the
hand-kept `QUERY_FIELDS` mirror.

**The four-runner problem is the most important structural fact in this note.**
There are four independent implementations of chain semantics:

- `filter_stage.cpp:185` — core, `plan_color_chain` + fold
- `filter.cpp:46` `run_scalar_chain` — mscoupon; batches consecutive core ops
  into one core call and applies `normalize` in place between flushes
- `mscoupon/engine.py:278` `_leading_color` / `:305` `_apply_base_chain`
- `mscoupon/engine.py:137` `preview_raster`

They agree today because the semantics are small enough to re-derive. Any
extension not consumed by all four re-forks them, and the fork is invisible until
a config behaves differently under the CLI than under the GUI. A `plan_chain`
that only core consumes is worse than no `plan_chain`.

### The synthesized colour stage stays where it is

`plan_color_chain` synthesizes at the **front**, so `[blur, edges]` on RGB means
`edges(blur(luminance(rgb)))`. Component-wise chaining would instead *lift* each
1→1 stage over the planes and reduce at the end. That is the more natural
semantics and it is **not** what this note proposes, because the two differ for
every non-linear stage — Gaussians commute with a linear reduction, morphology,
`edges` and `hessian` do not — and the configs that would change are exactly
those that never asked for anything.

The decision: **the stage stays at the front; what changes is that it stops
being invisible.** `plan_chain` materializes it into the plan, and the GUI draws
it as an `(auto)` card the user can see, and pin, and only then move. Zero
behaviour change, every existing config byte-identical, and the silence gone.
Lifting becomes a thing a user *does*, not a thing that happens to them.

## 3. `adapt` — the uniform adaptor

One operation, mode-dispatched, positionally free:

| mode | params | arity |
|---|---|---|
| `select` | `channels: [i…]` | C→k |
| `project` | `matrix: [[…C…] × k]`, or `preset` | C→k |
| `reduce` | `how: mean\|sum\|max\|min\|norm\|first`, `weights` | C→1 |
| `broadcast` | `channels: k` | 1→k |
| `cast` | `to`, `threshold`, `invert` | kind change — deferred with `FieldKind` |

Every existing `color` method decomposes onto it: `pick` is `select`,
`mean`/`max`/`min` are `reduce`, `luminance` and `weighted` are `project` with a
1×3 row. That is an **explanation, not a migration**.

### `color` is a frozen alias

`color` keeps its own code path and its own rule. "`color` must be the FIRST
stage" (`color_stage.cpp:316`, `filter_stage.cpp:58`, `config.cpp:63`) becomes a
rule about *the literal operation name* — a legacy spelling with legacy placement
— while `adapt` is the spelling that is positionally free.

This is worth more than it looks. Under a design where `color` itself became
positionally free, `[blur, color{mean}]` on planar input would stop raising, and
two tests asserting exactly that (`test_color_input.py:135`,
`mscoupon_tests.cpp:1847`) would have to be edited. Freezing the alias means
**no existing config
and no existing assertion changes at all** — "existing configs keep working"
becomes literally rather than approximately true. The cost is one deprecated
spelling documented forever, which is cheap.

Reuse: the GUI already renders params-dependent card rows through
`filter_param_schema(operation, params)` (`config_io.py:98`), which dispatches
`COLOR_METHOD_PARAMS` on `params["method"]`. `adapt` dispatches on
`params["mode"]` through that same function. No new mechanism, one new table.

## 4. Colour→colour stages

- **`stain_deconvolution`** — OD planes → stain concentrations, `c = pinv(M)·od`,
  where the columns of `M` are unit OD vectors. Presets `he` / `hdab` / `hed`;
  when only two stains are known the third column is Ruifrok's complement, the
  normalized cross product of the first two. Output planes named `hematoxylin`,
  `eosin`, `residual`. Nothing like it exists in the tree: `hematoxylin`,
  `eosin` and `deconvol` have zero hits.
- **`optical_density` gains `output: "planes"`** (C→C). Its existing
  `stain`/`weights`/`channels` projection stays, as sugar for a trailing
  `adapt{project}`.
- `hsv` → 3 planes; `dizenzo` / `structure` / `hessian` → all eigenvalues, with
  `component` / `eigen` as sugar. **Low priority**: these are conveniences, the
  existing parameters work, and nothing is blocked on them.

## 5. Arithmetic — and why registers are not needed

Unary, per-plane, C→C: `scale{a,b}`, `abs`, `clamp{lo,hi}`, `pow{p}`, `log1p`,
`invert{max}`, `sigmoid{center,slope}`, `rescale{from,to,clamp}`. diffg has **no
arithmetic at all** — no add, subtract, multiply or scale anywhere in its
headers — so these are pure MSSeg pixel loops. No dependency bump, no pin change.

The binary case is where the design nearly went wrong. The obvious shape is a
register machine: `branch{name, filters:[…]}` to run a sub-chain and stash it,
`combine{with, op}` to mix it back. That is what you reach for if you want
`E − λ·H`. It is not needed:

> With `stain_deconvolution` producing **c = pinv(M)·od**, `E − λ·H` is
> `(row_E − λ·row_H) · c`. Because `pinv(M)` is itself a matrix, that row
> composes with it: **`E − λ·H` is a single 1×C matrix applied directly to the
> OD planes.** `adapt{project, matrix: [[…]]}` is the entire feature — no
> registers, no sub-chains, and not even a deconvolution stage in the chain.

Checked numerically rather than asserted. With the Ruifrok H&E matrix and
λ = 0.7, deconvolving and then taking `E − 0.7·H` agrees with the single row
`[-2.113, 1.229, -0.641]` applied to the OD planes to `8.9e-16` — float noise.
The row is worth reading: it is *not* "a bit of eosin minus a bit of
hematoxylin", it is dominated by a large negative red coefficient, which is what
makes it different from anything reachable by tuning a `weighted` stage by hand.

`branch` / `combine` are therefore **deferred**, and each reason is a real cost:

- They are the only proposed feature that **breaks `run_scalar_chain`'s
  batch-and-flush invariant**. A register created inside one
  `apply_filter_chain` batch dies at its return, so
  `[branch, normalize, combine]` works in the core runner and in the pybind path
  but throws "unknown register" in the CLI — a **CLI/extension divergence on the
  same config JSON**, which is the exact class of bug the shared-parser note at
  `mscoupon_py.cpp:103` exists to prevent.
- They have **no home in the Python mirrors**, which apply one stage at a time
  through `filter_slice` (`engine.py:305`, `:419`). A register table there is a
  fifth reimplementation of chain semantics.
- They introduce a **name namespace** into a format that has none, which then
  has to survive session round-trip, the cards, and validation.
- They make `plan_chain` **non-linear**: a stage record stops being a function of
  its predecessor, and the "computable with no raster" property gets harder to
  state.

**The trigger condition for revisiting**: a chain needing a *spatial* operator
applied **unequally** across planes — `E − λ·blur(H)`, or `H · mask(E > t)`. A
projection cannot express those, because the two planes get different treatment
before mixing.

And, so nobody assumes otherwise: **component-wise lifting does not subsume
registers.** Lifting `blur` over the planes and then projecting gives
`blur(E) − λ·blur(H)`, not `E − λ·blur(H)`. Lifting handles only the case where
every plane gets the *same* spatial treatment.

## 6. Nuclei

The two chains already do different jobs — `filters` builds the topology field,
`base_filters` builds the channel statistics are measured from. That split is
the whole answer to "suppress nuclei but keep measuring them": **suppress in
`filters` only, and leave `base_filters` on the full OD**, so size, density and
shape survive as features. Nothing below touches the statistics chain except
§6.4, which adds to it.

1. **Today, zero code.** The `optical_density` stage already takes a `stain`
   vector; Ruifrok eosin is ≈ `[0.07, 0.99, 0.11]`. This is the baseline to
   beat, and it is one config line. It is also genuinely partial: the H and E OD
   unit vectors have cosine similarity **0.77**, and a bare eosin projection
   correlates only **0.64** with the deconvolved eosin concentration. Projecting
   onto E does not remove H; it merely stops asking for it.
2. **`stain_deconvolution` → `adapt{select: [1]}`.** The pseudo-inverse removes
   H's leakage into E rather than down-weighting it — the difference between
   0.64 and 1.0 above. This is what earns the stage its place even though §5
   shows the *arithmetic* can bypass it: the presets are legible, and the
   intermediate planes are measurable (§6.4).
3. **`adapt{project, matrix: [[…]]}`** for `E − λ·H`, λ ≈ 0.5–1.0 — one row
   applied to the OD planes, per §5. Suppression rather than removal: a nucleus
   still occupies space and its boundary is real, and what is wanted is that it
   stop being the *dominant* boundary, not that it vanish.
4. **Nuclei as statistics columns.** `base_filters` ends with C planes,
   `resolve_stat_channels` resolves them, and `mean_hematoxylin` becomes a real
   feature. This is the part that turns nucleus density into classifier input
   rather than only removing it from the topology — and it is by far the most
   invasive of the four (§7, Stage 3).

A fifth route was considered and set aside: **scale-selective morphology**. A
nucleus is a compact extremum at a known physical size, and `open` (OD space,
where nuclei are maxima) or `close` (intensity space, minima) at a radius above
the nuclear radius removes extrema smaller than the structuring element and
leaves larger structure alone. Both ops exist today, so it is testable with no
code. It is recorded here because it costs nothing to try, not because it is
planned.

## 7. Seams, hazards, build order

Each of these is a real collision, found against the code, and each needs a
recorded decision before the step that hits it.

**`normalize` on C planes is undefined and must refuse.** `measure_two_point`
returns **one** `TwoPoint`, and `normalizers_out` is indexed *positionally by
stage* in two readers (`pipeline.cpp:372`, `app.py:843`). A GMM is a
2-population fit: shared across hematoxylin and eosin it is meaningless, and
per-plane it is C× the EM *and* destroys the cross-plane comparability that a
following `project` consumes — normalizing before a projection changes what the
projection means. Decision: `StageIO{in: ScalarOnly}`, rejected by `plan_chain`
with a message naming `adapt{reduce}` / `adapt{select}` as the fix. Refusing
costs nothing and forecloses nothing; per-plane lifting can arrive later behind
an explicit `per_plane: true` if anyone needs it.

**`Image2D` is single-plane** (`types.hpp:15`), and `from_diffg` reads
`input.size()` floats into a flat vector with no channel count — so a C-plane
result becomes a C×h-tall image with wrong `width`/`height`. Silent corruption,
not a throw.

**The pybind result copies were a latent heap overflow** — `filter_slice` and
`filter_chain` allocated `FloatArray out({h, w})` and memcpied
`filtered.size() * sizeof(float)` into it, with the extent taken from the
*input* and nothing forcing the result to match. **Fixed 2026-09-17**: a shared
`to_2d_array` helper checks the count and throws a named error instead, and
`segment_slice` / `pipeline_labels` got the same guard. Stage 2 therefore meets
a diagnosable error there rather than a silent overflow — but the binding still
cannot *return* planes, which is Stage 2's actual work. `prime_slice` (`:280`)
and `stat_channel_images` (`:249`) call `to_image(base, …)`, which throws on
`ndim != 2` — noisy, therefore fine.

**`base_c<i>` naming and position.** The column-order prefix property cannot
break, because no existing config can produce a C>1 base; every ordering choice
for C>1 is free. The invariant to pin is narrower and sharper: *when
`base_filters` yields one plane, the name must be the literal `"base"`, never
`"base_c0"`, and the whole list must be identical to today's.* Three collisions:

- A base-sourced per-plane derived name decorated `_c<i>` **collides with the
  colour source's** (`stat_channels.cpp:134`) and trips the duplicate-name
  check. Use `_b<i>`. Pin it before a CSV ships — it is not reversible after.
- `relevance_base` and `ext_base` are keyed by the string `"base"`, and
  `pipeline.cpp:310` writes **`0.0f` into every row** when the column lookup
  returns −1. Silent zeros in a shipped CSV column, not an error.
- The default histogrammed channel is the literal `"base"`
  (`stat_channels.cpp:150`), so a C-plane base turns a previously valid
  `{"bins": 16}` block into a validation failure.

Position: immediately after the `filtered` block and before the `color_c<i>`
block (`stat_channels.cpp:69`–`:80`), keeping the two raw-plane blocks adjacent.

**GPU statistics would go silently wrong.** `msc2d.cpp:524` routes *every*
base-sourced derived channel to bank 0 — `c.source == "color" ? 1 + c.input_channel
: 0` — with no term for base planes, and the slot guard `planes_needed + 2 >
kMaxSlots` (`:498`) has none either. `blur_b0_s1` and `blur_b2_s1` would land on
consecutive slots of the same bank over `d_base`: wrong pixels, consistent slot
count, no assertion fired. Cheap first cut: `decline("a multi-plane base")`, one
line, which ships Stage 3 without touching the compile-gated `.cu` that most
developers do not build.

**A multi-plane base is not a statistics-layer change.** Six sites in
`pipeline.cpp` assume a scalar base — including `compute_segment_table` (the
segment CSV's own intensity columns), `label_selected_components` (the pixel-trim
chain) and `segment_slice_pipeline` — plus `base_relevance_floor` / `ceiling`
(`msc_stage.cpp:62`), which are percentiles *of the base raster* and undefined
for C planes without a choice.

**The `(auto)` card is derived state, never stored.** `_on_filter_op_change`
(`app.py:915`, `mspath/app.py:333`) wipes `params` and rebuilds every card from
scratch on any operation change anywhere in the chain, so the auto card must be
re-synthesized from the plan on every rebuild. Carry its marker as a *card-level
sibling key* (`{"operation": …, "params": …, "auto": True}`), which
`filters_to_json` drops for free; a marker inside `params` would be exported and
change config bytes. Note also that **there is no reorder UI today** — "pin and
move" is net-new machinery in two files, which is why the table below defers it.

**`ops = FILTER_OPERATIONS if idx == 0 else [… != "color"]`** appears verbatim
at `app.py:789` and `mspath/app.py:243`. Put the op list behind a `config_io`
function before touching either, or the plan lookup is the same edit twice — the
two card implementations are near-identical by copy, and that is precisely the
duplication `_preview_poll` already exists to survive.

### Build order

Each stage is independently shippable and testable.

| Stage | Content | Unblocks |
|---|---|---|
| 0 | this note | — |
| 1 | `StageIO` + `plan_chain`, **no behaviour change**; `plan_color_chain` becomes a wrapper returning identical results; both card UIs replace the `idx == 0` test with a plan lookup; `(auto)` card display-only | the silence |
| 2 | **landed 2026-09-17**: `plane_stages.cpp` (`adapt{select,project}`, `stain_deconvolution`), `optical_density{output:"planes"}`, `apply_filter_chain_planes`, the pybind result path returning `(C, h, w)`, the mscoupon chain split (plane prefix in core, scalar tail here so `normalize` still measures in Python), the plan-driven Python mirror, and the GUI schema rows | **stain deconvolution and `E − λ·H`** |
| 3 | `base_c<i>` in `resolve_stat_channels`; a third bank traversal; a plane base through `Msc2DPipeline::build`; the six `pipeline.cpp` sites; GPU declines | **`mean_hematoxylin` as a feature** |
| 4+ | `FieldKind` + `cast`; component-wise lifting; pin-and-move UI; `adapt{reduce, broadcast}`; the rest of the unary arithmetic; `branch`/`combine`; all-component `hsv`/`dizenzo`/`structure`/`hessian` | deferrable indefinitely |

**What Stage 2 settled.** The composition claim of §5 is no longer an argument:
`test_plane_stages` reads `inv(M)` off a basis image and checks that
`E − λ·H` computed by deconvolution equals one `adapt{project}` row on the OD
planes, and `test_plane_carrying_chain` checks that OD planes plus that row
reproduce `optical_density`'s own `stain` projection. Both hold to float noise
through the real API, so registers stay deferred on evidence rather than on
algebra done in a scratch file.

Three things the build found that this note had not, all of the same shape — a
rule written when `color` was the only plane-consuming stage:

* `plan_chain` synthesized its conversion whenever a chain did not *start with
  `color`*, so a leading `stain_deconvolution` got one plane instead of three.
  The test is now "unless the head already consumes the stack".
* The Python mirror was wrong for every plane-stage chain — it invented a colour
  head and reported each stage as 1→1 — **while its parity suite passed**,
  because the cases predated the stages. That is the argument for extending a
  parity suite in the same change that extends what it mirrors.
* `filters_to_json` keeps only the params a colour *method* declares, so
  `optical_density`'s new `output: "planes"` was dropped on export and the next
  stage met one plane instead of three. A mode that is not a schema row does not
  survive a round trip. The default is still dropped, so existing configs stay
  byte-identical.

The first two were caught by tests written alongside the code; the third only by
running the whole path end to end, which is the argument for doing that too.

Stated plainly, because it is the useful conclusion: **Stage 1 and Stage 2 are
the only prerequisites for the nuclei work.** `Field`, `FieldKind`, lifting and
registers are not. The proposal that started as a type system for the filter
chain reduces, for the problem that motivated it, to *one adaptor operation and
one new colour stage*.

## 8. Byte-compat gates

The contract the build order must not break:

- `mscoupon_tests.cpp` `test_color_chain_rules_and_identity` (l.1810) —
  grayscale byte-identity between the scalar and planar overloads, synthesized
  default == explicit `mean`, `apply_filter` refuses a colour stage on a scalar.
- `mscoupon_tests.cpp` `test_color_stat_channels` (l.1954) — the exact slot
  name/order list and plane-major `slot_in_request`.
- `test_color_input.py::test_extension_measures_colour_channels` (:177) — the
  resolved name list and `imgs.shape == (8, 24, 30)`.
- `test_preview_raster.py` :42, :58 — the three-way agreement between
  `preview_raster`, `filter_chain` and `_apply_base_chain`. :122 asserts the
  error *string* `"must be base or color"`.
- `test_normalize.py` :299, :406, :227 — the card round-trip is a fixpoint, so
  the `(auto)` card must not reach JSON.
- `test_histogram_stats.py::test_extension_projects_histogram_columns_last`
  (:86) — the prefix property.
- Both `--selftest` paths: coupon `app.py:2681` (`values[:4] == ["color",
  "color_c0", "color_c1", "color_c2"]`) and `:2910` (`after == before` config
  re-export); mspath `selftest.py:254`–`:309` (the chain-fingerprint cycle — a
  plan-derived auto card that is not stable under a `json.dumps` round trip
  breaks `_preview_is_stale`).

And the two that a naive design would have broken, recorded here because
**decision §3 — `color` as a frozen alias — is why they do not**:
`test_color_input.py:135` and `mscoupon_tests.cpp:1847`, both asserting that
a `color` stage after index 0 raises.

One test should legitimately *gain* a case rather than change:
`test_gpu_stats_parity.py:21`–`:31` wants a multi-plane-base spec in its
parametrize list once Stage 3 lands, so the bank-map hazard above cannot come
back silently.
