// End-to-end smoke test for the 3D Morse-Smale core path (M3).
//
// Builds a small synthetic volume, runs the full msseg_core sequence
// (build -> compute -> select_persistence -> snapshot -> basin_labels) and
// checks structural invariants. No external data required.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <vector>

#include "diffg/image.hpp"
#include "msseg/compute/msc3d.hpp"
#include "msseg/graph/msc_graph.hpp"
#include "msseg/io/raw_io.hpp"
#include "msseg/segment/registry.hpp"
#include "msseg/volume/types.hpp"

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
  if (!cond) {
    std::printf("  FAIL: %s\n", what);
    ++g_failures;
  } else {
    std::printf("  ok:   %s\n", what);
  }
}

msseg::Volume make_blobs(int n) {
  msseg::Volume v(diffg::Dimensions{static_cast<std::size_t>(n), static_cast<std::size_t>(n),
                                    static_cast<std::size_t>(n)});
  const struct {
    float cx, cy, cz, a;
  } blobs[] = {{6, 6, 6, 1.0f}, {17, 7, 10, 0.8f}, {9, 18, 15, 1.2f}, {18, 18, 5, 0.7f}};
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        float s = 0.0f;
        for (const auto& b : blobs) {
          const float d2 = (x - b.cx) * (x - b.cx) + (y - b.cy) * (y - b.cy) + (z - b.cz) * (z - b.cz);
          s += b.a * std::exp(-d2 / 20.0f);
        }
        v.data()[(static_cast<std::size_t>(z) * n + y) * n + x] = s;
      }
  return v;
}

}  // namespace

int main() {
  const int N = 24;
  const msseg::Volume vol = make_blobs(N);

  msseg::Msc3D msc;
  msseg::Msc3DParams params;  // RobinsNoalloc gradient, default
  msc.build(vol, params);
  msc.compute(params);

  std::printf("[core_smoke] MSC built on %dx%dx%d volume\n", N, N, N);

  // 1. Non-empty complex at base persistence.
  msc.select_persistence(0.0f);
  const msseg::MscGraph g = msc.snapshot();
  std::printf("  nodes=%zu arcs=%zu\n", g.nodes.size(), g.arcs.size());
  check(!g.nodes.empty(), "complex has living nodes");
  check(!g.arcs.empty(), "complex has living arcs");

  // 2. Every arc endpoint indexes a living node.
  bool endpoints_ok = true;
  for (const auto& a : g.arcs) {
    if (a.lower < 0 || a.lower >= static_cast<msseg::NodeId>(g.nodes.size()) || a.upper < 0 ||
        a.upper >= static_cast<msseg::NodeId>(g.nodes.size())) {
      endpoints_ok = false;
      break;
    }
  }
  check(endpoints_ok, "all arc endpoints are living nodes");

  // 3. Living critical-point count is non-increasing as persistence rises.
  const int c0 = msc.living_node_count();
  msc.select_persistence(0.25f);
  const int c1 = msc.living_node_count();
  msc.select_persistence(1.0f);
  const int c2 = msc.living_node_count();
  std::printf("  living CPs: p=0 -> %d, p=0.25 -> %d, p=1.0 -> %d\n", c0, c1, c2);
  check(c1 <= c0 && c2 <= c1, "living CP count is monotonic non-increasing in persistence");
  check(c2 >= 1, "at least one critical point survives");

  // 4. Basin labeling covers (nearly) all voxels.
  msc.select_persistence(0.1f);
  const msseg::LabelVolume labels = msc.basin_labels(/*ascending=*/true);
  std::size_t labeled = 0;
  for (std::size_t i = 0; i < labels.size(); ++i)
    if (labels.data()[i] != msseg::kBackgroundLabel) ++labeled;
  const double frac = static_cast<double>(labeled) / static_cast<double>(labels.size());
  std::printf("  basin coverage: %zu / %zu (%.3f)\n", labeled, labels.size(), frac);
  check(labels.size() == vol.size(), "label volume matches input dimensions");
  check(frac >= 0.95, "basin labeling covers >= 95%% of voxels");

  // 5. The "basin" strategy is registered and runs through the seam.
  auto strategy = msseg::make_strategy("basin");
  const msseg::LabelVolume via_strategy = strategy->segment(g, msc, vol, msseg::SegmentationParams{});
  check(via_strategy.size() == vol.size(), "basin strategy returns a full-size label volume");

  // 6. raw_io round-trips a volume.
  {
    const auto tmp = std::filesystem::temp_directory_path() / "msseg_core_smoke.raw";
    msseg::write_raw_volume(tmp, vol);
    const msseg::Volume rt = msseg::read_raw_volume(tmp);
    bool same = rt.size() == vol.size();
    for (std::size_t i = 0; same && i < vol.size(); ++i) same = rt.data()[i] == vol.data()[i];
    check(same, "raw_io write/read round-trips exactly");
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
  }

  std::printf("[core_smoke] %s\n", g_failures == 0 ? "PASSED" : "FAILED");
  return g_failures == 0 ? 0 : 1;
}
