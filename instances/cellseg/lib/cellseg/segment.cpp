#include "cellseg/segment.hpp"

#include <cstddef>
#include <unordered_map>
#include <unordered_set>

namespace cellseg {
namespace {

// Distinct downward minima (snapshot NodeIds) of a 1-saddle.
std::vector<msseg::NodeId> downward_minima(const msseg::MscGraph& g, msseg::NodeId saddle) {
  std::vector<msseg::NodeId> mins;
  for (const msseg::NodeId aid : g.adjacency[static_cast<std::size_t>(saddle)]) {
    const auto& a = g.arcs[static_cast<std::size_t>(aid)];
    if (a.upper == saddle && g.nodes[static_cast<std::size_t>(a.lower)].index_dim == 0) {
      bool seen = false;
      for (auto m : mins)
        if (m == a.lower) { seen = true; break; }
      if (!seen) mins.push_back(a.lower);
    }
  }
  return mins;
}

// Upward neighbors of `node` at `target_dim` (arcs where node is the lower end).
void upward_of_dim(const msseg::MscGraph& g, msseg::NodeId node, int target_dim,
                   std::vector<msseg::NodeId>& out) {
  for (const msseg::NodeId aid : g.adjacency[static_cast<std::size_t>(node)]) {
    const auto& a = g.arcs[static_cast<std::size_t>(aid)];
    if (a.lower == node && g.nodes[static_cast<std::size_t>(a.upper)].index_dim == target_dim)
      out.push_back(a.upper);
  }
}

}  // namespace

SegmentResult run_segmentation(msseg::Msc3D& msc, const msseg::Volume& filtered,
                               const LivingView& view, float cut_threshold,
                               float background_threshold) {
  const msseg::MscGraph& g = view.graph;
  const auto dims = filtered.dims();
  const std::size_t n = filtered.size();

  // --- Step 9: cut the merge tree -> per-minimum region ids ----------------
  std::vector<std::int64_t> region_voxels;
  const std::unordered_map<msseg::NodeId, int> region_of_min =
      cut_regions(view.tree, cut_threshold, &region_voxels);

  // --- Step 10: background region = largest cut region by voxel count ------
  int background_region = -1;
  std::int64_t best = -1;
  for (std::size_t r = 0; r < region_voxels.size(); ++r) {
    if (region_voxels[r] > best) {
      best = region_voxels[r];
      background_region = static_cast<int>(r);
    }
  }

  // Per-voxel ascending region id (-1 where no living minimum owns the voxel).
  std::vector<int> asc_region(n, -1);
  const std::int32_t* asc = view.asc_labels.data();
  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t lab = asc[i];
    if (lab <= 0) continue;
    const auto it = region_of_min.find(static_cast<msseg::NodeId>(lab - 1));
    if (it != region_of_min.end()) asc_region[i] = it->second;
  }

  // --- Step 10/11: separating 1-saddles -> living 2-saddles -> maxima -------
  std::unordered_set<msseg::NodeId> selected_maxima;
  for (const auto& node : g.nodes) {
    if (node.index_dim != 1) continue;
    const std::vector<msseg::NodeId> mins = downward_minima(g, node.id);
    if (mins.size() < 2) continue;
    const auto r0 = region_of_min.find(mins[0]);
    bool separating = false;
    for (std::size_t k = 1; k < mins.size() && !separating; ++k) {
      const auto rk = region_of_min.find(mins[k]);
      if (r0 != region_of_min.end() && rk != region_of_min.end() && r0->second != rk->second)
        separating = true;  // bridges two different cut regions (cell-cell or inside-outside)
    }
    if (!separating) continue;

    std::vector<msseg::NodeId> two_saddles;
    upward_of_dim(g, node.id, 2, two_saddles);
    for (const msseg::NodeId t : two_saddles) {
      std::vector<msseg::NodeId> maxima;
      upward_of_dim(g, t, 3, maxima);
      for (const msseg::NodeId mx : maxima) selected_maxima.insert(mx);
    }
  }

  // --- Step 11: membrane-regions mask = descending basins of those maxima ---
  const msseg::LabelVolume dsc_labels = msc.living_labels(/*ascending=*/false);
  const std::int32_t* dsc = dsc_labels.data();

  std::vector<std::uint8_t> membrane_region(n, 0);  // descending 3m of a boundary max
  std::vector<std::uint8_t> fluor(n, 0);            // intensity foreground (step 12)
  std::vector<std::uint8_t> cleaned(n, 0);          // step 13
  const float* fdata = filtered.data();
  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t dl = dsc[i];
    if (dl > 0 && selected_maxima.count(static_cast<msseg::NodeId>(dl - 1))) membrane_region[i] = 1;
    if (fdata[i] >= background_threshold) fluor[i] = 1;
    cleaned[i] = (membrane_region[i] && fluor[i]) ? 1 : 0;
  }

  // --- Step 14: 8-bit bit-flag segmentation volume -------------------------
  SegmentResult result{msseg::LabelVolume(dims), msseg::LabelVolume(dims)};
  std::int32_t* seg8 = result.seg8.data();
  for (std::size_t i = 0; i < n; ++i) {
    std::int32_t bits = 0;
    if (asc_region[i] >= 0 && asc_region[i] != background_region) bits |= 1;  // non-bg ascending 3m
    if (cleaned[i]) bits |= 2;                                                // cleaned membrane
    if (membrane_region[i]) bits |= 4;                                        // membrane region (dsc 3m)
    if (fluor[i]) bits |= 8;                                                  // foreground (not background)
    seg8[i] = bits;
  }

  // --- Step 15: cell-id volume ---------------------------------------------
  // Start from the ascending cut regions; each ascending 3m carries its region.
  std::int32_t* ids = result.ids.data();
  for (std::size_t i = 0; i < n; ++i)
    ids[i] = static_cast<std::int32_t>(asc_region[i] >= 0 ? asc_region[i] : background_region);

  // For each selected descending 3m, its most-likely cell = the non-background
  // ascending region with the largest overlap inside its cleaned-membrane part.
  std::unordered_map<msseg::NodeId, std::unordered_map<int, std::int64_t>> overlap;
  for (std::size_t i = 0; i < n; ++i) {
    if (!cleaned[i]) continue;
    const std::int32_t dl = dsc[i];
    if (dl <= 0) continue;
    const msseg::NodeId mx = static_cast<msseg::NodeId>(dl - 1);
    if (!selected_maxima.count(mx)) continue;
    const int r = asc_region[i];
    if (r < 0 || r == background_region) continue;
    ++overlap[mx][r];
  }
  std::unordered_map<msseg::NodeId, int> best_region;
  for (const auto& [mx, hist] : overlap) {
    int arg = -1;
    std::int64_t bestc = -1;
    for (const auto& [r, c] : hist)
      if (c > bestc) { bestc = c; arg = r; }
    if (arg >= 0) best_region[mx] = arg;
  }
  // Paint only background-labeled voxels inside the cleaned-membrane part.
  for (std::size_t i = 0; i < n; ++i) {
    if (!cleaned[i]) continue;
    const std::int32_t dl = dsc[i];
    if (dl <= 0) continue;
    const msseg::NodeId mx = static_cast<msseg::NodeId>(dl - 1);
    const auto it = best_region.find(mx);
    if (it == best_region.end()) continue;
    if (ids[i] == background_region) ids[i] = static_cast<std::int32_t>(it->second);
  }

  return result;
}

}  // namespace cellseg
