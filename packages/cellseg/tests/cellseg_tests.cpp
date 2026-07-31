// Smoke test for the cellseg instance (two-phase 3D cell segmentation).
//
// Synthesizes a few hollow fluorescent shells, runs the heavy lift + a Phase-B
// segmentation entirely in memory, and checks structural invariants of the
// merge tree and the seg8 / ids output volumes. No external data required.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <unordered_set>

#include "cellseg/cell_pipeline.hpp"
#include "cellseg/config.hpp"
#include "cellseg/heavy_lift.hpp"
#include "cellseg/segment.hpp"
#include "diffg/image.hpp"
#include "msseg/volume/types.hpp"

namespace {

int g_failures = 0;
void check(bool cond, const char* what) {
  std::printf(cond ? "  ok:   %s\n" : "  FAIL: %s\n", what);
  if (!cond) ++g_failures;
}

// Three bright hollow shells (fluorescent membranes) on a dark background. The
// first two overlap (12 apart, radius 8) so they share a genuine cell-cell
// membrane -- a separating ridge with real maxima that survives persistence
// simplification -- while the third is isolated. (Isolated shells alone produce
// no cell-cell membrane once the branch relabel folds boundary fragments, so a
// touching pair is what exercises the membrane walk-up.)
msseg::Volume make_shells(int n) {
  msseg::Volume v(diffg::Dimensions{static_cast<std::size_t>(n), static_cast<std::size_t>(n),
                                    static_cast<std::size_t>(n)});
  const struct {
    float cx, cy, cz;
  } centers[] = {{16, 24, 24}, {28, 24, 24}, {36, 12, 12}};
  const float radius = 8.0f, thick = 1.5f;
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        float s = 0.0f;
        for (const auto& c : centers) {
          const float r = std::sqrt((x - c.cx) * (x - c.cx) + (y - c.cy) * (y - c.cy) +
                                    (z - c.cz) * (z - c.cz));
          s += std::exp(-((r - radius) * (r - radius)) / (2.0f * thick * thick));
        }
        v.data()[(static_cast<std::size_t>(z) * n + y) * n + x] = s;
      }
  return v;
}

}  // namespace

// Count living minima (merge-tree leaves) at the heavy persistence.
int living_minima(cellseg::CellPipeline& pipe) {
  pipe.select_persistence(pipe.heavy_persistence());
  int leaves = 0;
  for (const auto& node : pipe.view().tree.nodes)
    if (node.is_leaf) ++leaves;
  return leaves;
}

int main() {
  const int N = 48;
  msseg::Volume vol = make_shells(N);

  // --- Path A: classic boundary rule (minima_ignore_boundary = false) -------
  // A rich, well-conditioned complex: exercises the full Phase-B machinery
  // (merge tree -> cut -> separating saddles -> membrane walk-up -> seg8/ids).
  cellseg::HeavyLiftConfig hc;
  hc.blur_sigma = 1.5f;
  hc.persistence_percent = 5.0f;
  hc.minima_ignore_boundary = false;

  cellseg::CellPipeline pipe(cellseg::run_heavy_lift(vol, hc));
  std::printf("[cellseg_tests] value_range=%.4f heavy_persistence=%.4f\n", pipe.value_range(),
              pipe.heavy_persistence());
  check(pipe.value_range() > 0.0f, "value range is positive");

  const int classic_leaves = living_minima(pipe);
  const cellseg::LivingView& view = pipe.view();
  int mergers = 0;
  for (const auto& node : view.tree.nodes)
    if (!node.is_leaf) ++mergers;
  std::printf("  merge tree (classic): %zu roots, %d leaves, %d mergers\n", view.tree.roots.size(),
              classic_leaves, mergers);
  check(classic_leaves >= 3, "merge tree has at least one leaf per shell");
  check(!view.tree.roots.empty(), "merge tree has at least one root");

  // Cut at 0 (fully separate the basins); intensity threshold picks the shells.
  const cellseg::SegmentResult res = pipe.segment(/*cut_threshold=*/0.0f, /*background_threshold=*/0.4f);
  check(res.seg8.size() == vol.size(), "seg8 matches input dimensions");
  check(res.ids.size() == vol.size(), "ids matches input dimensions");

  std::size_t membrane = 0, foreground = 0, nonbg_asc = 0;
  for (std::size_t i = 0; i < res.seg8.size(); ++i) {
    const std::int32_t b = res.seg8.data()[i];
    if (b & 2) ++membrane;
    if (b & 8) ++foreground;
    if (b & 1) ++nonbg_asc;
  }
  std::printf("  seg8: cleaned-membrane=%zu foreground=%zu nonbg-ascending=%zu\n", membrane,
              foreground, nonbg_asc);
  check(foreground > 0, "intensity foreground (bit 8) is non-empty");
  check(membrane > 0, "cleaned membrane (bit 2) is non-empty");
  check(nonbg_asc > 0, "non-background ascending regions (bit 1) are non-empty");

  std::unordered_set<std::int32_t> ids;
  for (std::size_t i = 0; i < res.ids.size(); ++i) ids.insert(res.ids.data()[i]);
  std::printf("  ids: %zu distinct labels\n", ids.size());
  check(ids.size() >= 2, "id volume has multiple distinct regions");

  // --- Path B: Option 3 (minima_ignore_boundary = true) ---------------------
  // Relaxing the boundary gate for min->1-saddle arcs must let boundary-only
  // minima cancel by persistence, collapsing the fragmented background into
  // strictly fewer living minima than the classic rule. (This is the whole
  // point of the flag; membrane reconstruction on this degenerate tiny
  // complete-shell volume is not asserted -- it needs a real, richer complex.)
  cellseg::HeavyLiftConfig hc_ib = hc;
  hc_ib.minima_ignore_boundary = true;
  cellseg::CellPipeline pipe_ib(cellseg::run_heavy_lift(vol, hc_ib));
  const int ignore_leaves = living_minima(pipe_ib);
  std::printf("  living minima: classic=%d, ignore-boundary=%d\n", classic_leaves, ignore_leaves);
  check(ignore_leaves >= 3, "ignore-boundary keeps at least one true minimum per shell");
  check(ignore_leaves < classic_leaves,
        "ignore-boundary cancels boundary-only minima (fewer living minima than classic)");
  // The Phase-B pipeline still runs and separates the shells from background.
  const cellseg::SegmentResult res_ib = pipe_ib.segment(/*cut_threshold=*/0.0f, /*background_threshold=*/0.4f);
  std::unordered_set<std::int32_t> ids_ib;
  for (std::size_t i = 0; i < res_ib.ids.size(); ++i) ids_ib.insert(res_ib.ids.data()[i]);
  check(ids_ib.size() >= 2, "ignore-boundary id volume still has multiple regions");

  std::printf("[cellseg_tests] %s\n", g_failures == 0 ? "PASSED" : "FAILED");
  return g_failures == 0 ? 0 : 1;
}
