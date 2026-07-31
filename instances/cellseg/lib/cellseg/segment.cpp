#include "cellseg/segment.hpp"

#include <cstddef>
#include <functional>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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

// Downward neighbors of `node` at `target_dim` (arcs where node is the upper end).
void downward_of_dim(const msseg::MscGraph& g, msseg::NodeId node, int target_dim,
                     std::vector<msseg::NodeId>& out) {
  for (const msseg::NodeId aid : g.adjacency[static_cast<std::size_t>(node)]) {
    const auto& a = g.arcs[static_cast<std::size_t>(aid)];
    if (a.upper == node && g.nodes[static_cast<std::size_t>(a.lower)].index_dim == target_dim)
      out.push_back(a.lower);
  }
}

inline float node_value(const msseg::MscGraph& g, msseg::NodeId id) {
  return g.nodes[static_cast<std::size_t>(id)].value;
}

// A frontier hop for the Prim-style label growth (min-heap by weight).
struct Hop {
  float weight;          // value[to_max] - value[2saddle]; smaller = stronger ridge
  msseg::NodeId to_max;
  int cell;
};
struct HopGreater {
  bool operator()(const Hop& a, const Hop& b) const { return a.weight > b.weight; }
};

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
      cut_regions(view.tree, cut_threshold, view.persistence, &region_voxels);

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

  // --- Step 12 (moved up): fluorescent-membrane / foreground mask -----------
  // Needed by the step-11 seed-label overlap; a fixed intensity threshold.
  const float* fdata = filtered.data();
  std::vector<std::uint8_t> fluor(n, 0);
  for (std::size_t i = 0; i < n; ++i)
    if (fdata[i] >= background_threshold) fluor[i] = 1;

  // Descending (maxima) manifold labels: label = living maximum NodeId + 1.
  const msseg::LabelVolume dsc_labels = msc.living_labels(/*ascending=*/false);
  const std::int32_t* dsc = dsc_labels.data();

  // --- Step 11: touched boundary maxima (barrier-respecting walk-up) -------
  // From each *separating* 1-saddle walk 1->2->3, only through 2-saddles that
  // do not cross the cut (value >= cut_threshold).
  std::unordered_set<msseg::NodeId> touched_maxima;
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
      if (node_value(g, t) < cut_threshold) continue;  // never cross the cut
      std::vector<msseg::NodeId> maxima;
      upward_of_dim(g, t, 3, maxima);
      for (const msseg::NodeId mx : maxima) touched_maxima.insert(mx);
    }
  }

  // Seed-label each touched max: the non-background ascending cut-region with
  // the largest foreground overlap inside that max's descending manifold.
  std::unordered_map<msseg::NodeId, std::unordered_map<int, std::int64_t>> seed_overlap;
  for (std::size_t i = 0; i < n; ++i) {
    if (!fluor[i]) continue;
    const std::int32_t dl = dsc[i];
    if (dl <= 0) continue;
    const msseg::NodeId mx = static_cast<msseg::NodeId>(dl - 1);
    if (!touched_maxima.count(mx)) continue;
    const int r = asc_region[i];
    if (r < 0 || r == background_region) continue;
    ++seed_overlap[mx][r];
  }
  std::unordered_map<msseg::NodeId, int> labeled;  // max NodeId -> cell id
  for (const auto& [mx, hist] : seed_overlap) {
    int arg = -1;
    std::int64_t bestc = -1;
    for (const auto& [r, c] : hist)
      if (c > bestc) { bestc = c; arg = r; }
    if (arg >= 0) labeled[mx] = arg;
  }

  // --- Step 11.5: extend labels along max->2saddle->max ridges (Prim) ------
  std::priority_queue<Hop, std::vector<Hop>, HopGreater> frontier;
  auto push_hops = [&](msseg::NodeId from_max, int cell) {
    std::vector<msseg::NodeId> tsads;
    downward_of_dim(g, from_max, 2, tsads);
    for (const msseg::NodeId t : tsads) {
      if (node_value(g, t) < cut_threshold) continue;  // never cross the cut
      std::vector<msseg::NodeId> maxs;
      upward_of_dim(g, t, 3, maxs);
      for (const msseg::NodeId m2 : maxs) {
        if (m2 == from_max || labeled.count(m2)) continue;
        frontier.push(Hop{node_value(g, m2) - node_value(g, t), m2, cell});
      }
    }
  };
  {
    const std::vector<std::pair<msseg::NodeId, int>> seeds(labeled.begin(), labeled.end());
    for (const auto& [mx, cell] : seeds) push_hops(mx, cell);
  }
  while (!frontier.empty()) {
    const Hop h = frontier.top();
    frontier.pop();
    if (labeled.count(h.to_max)) continue;  // assigned via a stronger (earlier) hop
    labeled[h.to_max] = h.cell;
    push_hops(h.to_max, h.cell);
  }

  // --- Steps 11/13: membrane-regions + cleaned membrane --------------------
  std::vector<std::uint8_t> membrane_region(n, 0);  // descending 3m of a labeled max
  std::vector<std::uint8_t> cleaned(n, 0);          // membrane_region & fluorescent
  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t dl = dsc[i];
    if (dl > 0 && labeled.count(static_cast<msseg::NodeId>(dl - 1))) membrane_region[i] = 1;
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

  // --- Step 15: cell-id volume (grow ascending regions through membrane) ----
  // FIRST label every voxel by its ascending cut region.
  std::int32_t* ids = result.ids.data();
  for (std::size_t i = 0; i < n; ++i)
    ids[i] = static_cast<std::int32_t>(asc_region[i] >= 0 ? asc_region[i] : background_region);

  // A "cleaned membrane region" is one maximum's descending manifold restricted
  // to foreground (fluorescent) voxels. Overlap each with the non-background
  // ascending regions.
  std::unordered_map<msseg::NodeId, std::unordered_map<int, std::int64_t>> mem_overlap;
  for (std::size_t i = 0; i < n; ++i) {
    if (!fluor[i]) continue;
    const std::int32_t dl = dsc[i];
    if (dl <= 0) continue;
    const msseg::NodeId M = static_cast<msseg::NodeId>(dl - 1);
    mem_overlap[M];  // register this membrane region (even if no non-bg overlap)
    const int r = asc_region[i];
    if (r >= 0 && r != background_region) ++mem_overlap[M][r];
  }

  // Seed: each membrane region that overlaps a non-background ascending region
  // takes the id of the maximum overlap.
  std::unordered_map<msseg::NodeId, int> cell_of_max;
  for (const auto& [M, hist] : mem_overlap) {
    int arg = -1;
    std::int64_t bestc = -1;
    for (const auto& [r, c] : hist)
      if (c > bestc) { bestc = c; arg = r; }
    if (arg >= 0) cell_of_max[M] = arg;
  }

  // Non-overlapping membrane regions: shortest path over max -> 2-saddle -> max
  // (skipping cut arcs, i.e. 2-saddles below the cut) to the nearest already-
  // labeled maximum, whose id they inherit (multi-source Dijkstra, edge weight =
  // value drop from the next max down to the connecting 2-saddle).
  {
    std::unordered_map<msseg::NodeId, float> dist;
    using DN = std::pair<float, msseg::NodeId>;
    std::priority_queue<DN, std::vector<DN>, std::greater<DN>> pq;
    for (const auto& [M, cell] : cell_of_max) {
      dist[M] = 0.0f;
      pq.push({0.0f, M});
    }
    while (!pq.empty()) {
      const auto [d, M] = pq.top();
      pq.pop();
      const auto di = dist.find(M);
      if (di == dist.end() || d > di->second) continue;
      const int cell = cell_of_max[M];
      std::vector<msseg::NodeId> tsads;
      downward_of_dim(g, M, 2, tsads);
      for (const msseg::NodeId t : tsads) {
        if (node_value(g, t) < cut_threshold) continue;  // never cross the cut
        std::vector<msseg::NodeId> maxs;
        upward_of_dim(g, t, 3, maxs);
        for (const msseg::NodeId M2 : maxs) {
          if (M2 == M) continue;
          const float nd = d + (node_value(g, M2) - node_value(g, t));
          const auto it2 = dist.find(M2);
          if (it2 == dist.end() || nd < it2->second) {
            dist[M2] = nd;
            cell_of_max[M2] = cell;  // inherit the source region's id
            pq.push({nd, M2});
          }
        }
      }
    }
  }

  // Grow each cell across its cleaned membrane region.
  for (std::size_t i = 0; i < n; ++i) {
    if (!fluor[i]) continue;
    const std::int32_t dl = dsc[i];
    if (dl <= 0) continue;
    const auto it = cell_of_max.find(static_cast<msseg::NodeId>(dl - 1));
    if (it != cell_of_max.end()) ids[i] = static_cast<std::int32_t>(it->second);
  }

  return result;
}

}  // namespace cellseg
