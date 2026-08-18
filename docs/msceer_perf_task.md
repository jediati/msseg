# MSCEER task: make `msc_2d_lib` re-thresholding ~15x cheaper

Repo: `MSCEER` (local checkout `C:\Users\jediati\Desktop\JEDIATI\code\MSCEER`,
currently clean at the pinned commit `0c1b90b` — "make msc_2d_lib.cxx respect
build arc geometry flags"). All line numbers below are from that commit.

Single file: **`msc_2d_lib/msc_2d_lib.cxx`**. Two functions, near-duplicates of
each other: `Msc2D::ascending2Manifolds()` (line 227) and
`Msc2D::descending2Manifolds()` (line 341). Both fixes apply to **both**.

## Why

These are the interactive path in a downstream viewer: dragging a persistence
slider calls `setPersistence()` + `ascending2Manifolds()` once per slice. On a
3232x3232 slice (10.4 Mpx, ~1100 living features, `BuilderMode::Partitioned`)
that pair costs **875-915 ms**, and profiling shows almost none of it is the
hierarchy walk — that part is correctly cheap. It is two full-image passes
bolted around it, one of which is dead.

Measured on that slice, per call:

| phase | ms |
|---|---|
| `setPersistence` + living-node walk + `GatherNodes` | small |
| dead diagnostic loop (fix 1) | **~390** |
| `LabelImage` alloc (42 MB) + 10.4M `remap.find()` (fix 2) | **~500** |

Target: **~50-80 ms**.

## Fix 1 — the diagnostic loop runs unconditionally (~390 ms)

`ascending2Manifolds()` lines 299-308 (and the identical block at 413-422 in
`descending2Manifolds()`):

```cpp
    size_t base_unlabeled = 0;
    std::unordered_set<int> base_unique_ids;
    for (size_t i = 0; i < m_impl->baseLabelingAsc2.size(); ++i) {
        const int base = m_impl->baseLabelingAsc2[i];
        if (base < 0) {
            base_unlabeled++;
        } else {
            base_unique_ids.insert(base);
        }
    }
```

That is one `std::unordered_set<int>::insert` **per pixel**, 10.4M of them. Both
`base_unlabeled` and `base_unique_ids` are used in exactly one place: the
`printf` inside `if (shouldEmitLabelDiagnostics())` at line 325 (line 439 for
descending). `shouldEmitLabelDiagnostics()` (line 37) reads the
`MSC2D_LABEL_DIAGNOSTICS` env var, so in every normal run this loop is pure
waste.

Guard the loop behind the same condition. Verified locally: this alone takes the
pair from 875-915 ms to **492-520 ms**, with byte-identical `LabelImage` output.

## Fix 2 — per-pixel hash probe into `remap` (~500 ms)

Lines 269-296 build `std::unordered_map<int, int> remap` (base identity ->
living node id), then lines 315-324 walk all 10.4M pixels doing
`remap.find(base)` — a hash probe per pixel.

**The trap, which you must handle:** `baseLabelingAsc2` stores a *different id
space* depending on builder mode.

- **Serial** (line 262, the `else` branch): `static_cast<int>(nid)` — a **node
  id**. Small range (number of critical points).
- **Partitioned** (lines 253-254): `static_cast<int>(base_cells[ci])` — a
  **topological cell index**, because `GetAscendingLineageCells()` returns cells.
  Range is up to `grid->NumElements()`, ~41.7M for a 3232^2 image.

`remap`'s keys match whichever is in use.

I tried the naive dense LUT (`std::vector<int>` indexed by the raw value) and it
made things **worse** — 492 ms -> 634 ms — because in partitioned mode it
allocates and fills a ~167 MB table on every call. Do not do that.

The right shape: **the number of distinct base identities is small in both modes**
(one per base minimum, thousands). Only the *range* explodes. So compact the ids
once, then index densely:

1. When `baseLabelingAsc2` is first built (the `if (...empty())` block at 233-265
   — cached, so this is one-time), assign each distinct raw base identity a
   compact `0..M-1` id and store **compact ids** in `baseLabelingAsc2`. Keep the
   raw values in a side `std::vector<INDEX_TYPE>` if anything else needs them.
2. Per call, when building `remap`, translate each constituent's raw id to its
   compact id (a hash lookup over *thousands* of entries, not millions) and write
   into a `std::vector<int> remapDense(M, -1)`.
3. The per-pixel loop becomes one array read:
   `const int c = baseLabelingAsc2[i]; out.labels[i] = (c >= 0) ? remapDense[c] : -1;`

Preserve `remap_miss` semantics (count pixels with `base >= 0` that have no
mapping) so the diagnostic output is unchanged when enabled.

## Fix 3 (optional, smaller) — reuse the output buffer

`LabelImage out; out.labels.assign(mX*mY, -1);` (lines 310-313 / 424-427)
allocates ~42 MB per call. If the API allows, keep a member buffer and refill it,
or add an overload that writes into a caller-provided span. Only worth doing
after 1 and 2.

## Hard constraint

**The returned `LabelImage` must be bit-identical to today's** for both builder
modes and both manifold directions — same living node ids, same `-1` background,
same dimensions. These are compaction and dead-code changes only; no change to
which features survive, their ids, or their extents.

## Verification

- `msc_2d_lib/msc_2d_lib_smoke.cxx` and `msc_2d_partitioned_smoke.cxx` must pass.
- Exercise **both** builder modes and **both** directions — the id-space trap
  above only shows up in partitioned, and the descending path is a separate copy
  of the same code.
- Regression-check equality directly: capture `ascending2Manifolds().labels` at
  a few persistences before and after, and assert element-wise equality.
- Confirm `MSC2D_LABEL_DIAGNOSTICS=1` still prints the same numbers.
- Downstream: MSSeg's `ctest` (core_smoke + mscoupon_tests + cellseg_tests) must
  stay green, and `MSSEG_TIME_MSC=1` prints the per-phase split of the calling
  code so the win is directly visible as the `msceer` line.

## After pushing

MSSeg pins MSCEER by commit in `cmake/Dependencies.cmake` (line ~50,
`GIT_TAG 0c1b90b5708988da56cc560f14edfd2bf3be6cc6`). Send me the new SHA and I
will bump the pin and re-measure.

## Note on the consumer (no change needed there)

MSSeg already compacts whatever ids it receives into its own dense `0..M-1` space
at build time (`libs/core/msseg/compute/msc2d.cpp`, `nid_to_compact`), which is
why its own two 10.4M-pixel passes cost ~70 ms combined versus MSCEER's ~500 ms —
same work, compact space instead of hashing raw ids. That stays as is; it is
independent of this fix.
