#include "mscoupon/matcher.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <unordered_set>

namespace mscoupon {
namespace {

// The per-slice value of a named quantity for one component. These are the
// quantities whose spread ACROSS slices says something about a 3D feature's
// shape -- unlike the intensity field, whose reductions stay voxel-pooled.
bool per_slice_value(const std::string& quantity, const CcNodeStat& p, double& out) {
  if (quantity == "area") {
    out = static_cast<double>(p.area);
  } else if (quantity == "bbox_w") {
    out = static_cast<double>(p.max_x - p.min_x + 1);
  } else if (quantity == "bbox_h") {
    out = static_cast<double>(p.max_y - p.min_y + 1);
  } else {
    return false;
  }
  return true;
}

}  // namespace

// Running per-slice reductions, one accumulator per (global id, quantity). Kept
// as sums/min/max/sumsq rather than the samples themselves so memory stays O(gids)
// instead of O(nodes).
class PerSliceAcc {
 public:
  PerSliceAcc(std::size_t n_gids, const std::vector<std::string>& quantities)
      : quantities_(quantities), n_(n_gids) {
    if (quantities_.empty()) return;
    cells_.assign(n_ * quantities_.size(), Cell{});
  }

  void add(int gid, const CcNodeStat& p) {
    if (quantities_.empty() || p.area == 0) return;
    for (std::size_t q = 0; q < quantities_.size(); ++q) {
      double v = 0.0;
      if (!per_slice_value(quantities_[q], p, v)) continue;
      Cell& c = cells_[static_cast<std::size_t>(gid) * quantities_.size() + q];
      c.n += 1;
      c.sum += v;
      c.sumsq += v * v;
      c.lo = std::min(c.lo, v);
      c.hi = std::max(c.hi, v);
    }
  }

  void finish(int gid, const StatisticsConfig& cfg, std::map<std::string, double>& out) const {
    if (quantities_.empty() || !cfg.any_per_slice()) return;
    for (std::size_t q = 0; q < quantities_.size(); ++q) {
      const Cell& c = cells_[static_cast<std::size_t>(gid) * quantities_.size() + q];
      if (c.n == 0) continue;
      const std::string& name = quantities_[q];
      const double mean = c.sum / static_cast<double>(c.n);
      if (cfg.ps_mean) out[name + "_mean"] = mean;
      if (cfg.ps_min) out[name + "_min"] = c.lo;
      if (cfg.ps_max) out[name + "_max"] = c.hi;
      if (cfg.ps_std) {
        const double var = c.sumsq / static_cast<double>(c.n) - mean * mean;
        out[name + "_std"] = var > 0.0 ? std::sqrt(var) : 0.0;
      }
    }
  }

 private:
  struct Cell {
    std::int64_t n = 0;
    double sum = 0.0, sumsq = 0.0;
    double lo = std::numeric_limits<double>::max();
    double hi = std::numeric_limits<double>::lowest();
  };
  std::vector<std::string> quantities_;
  std::size_t n_ = 0;
  std::vector<Cell> cells_;
};

int SliceMatcher::find(int x) {
  while (parent_[x] != x) {
    parent_[x] = parent_[parent_[x]];  // path halving
    x = parent_[x];
  }
  return x;
}

void SliceMatcher::unite_first_seen(int a, int b) {
  a = find(a);
  b = find(b);
  if (a == b) return;
  parent_[std::max(a, b)] = std::min(a, b);  // lower index (first-seen) wins
}

void SliceMatcher::add_slice(const std::vector<int>& cc_labels, int width, int height,
                             const std::vector<CcNodeStat>& node_stats, int slice_index,
                             int connectivity) {
  const int n = static_cast<int>(node_stats.size());
  const int node_base = static_cast<int>(parent_.size());
  for (int c = 0; c < n; ++c) {
    parent_.push_back(node_base + c);
    node_stat_.push_back(node_stats[static_cast<std::size_t>(c)]);
    node_slice_.push_back(slice_index);
  }

  // Cross-slice linking to the previously added slice by the connectivity stencil
  // (6 -> same (x,y) only; 18 -> + 4-neighbours; 26 -> full 3x3).
  const bool edges = (connectivity != 6);
  const bool diag = (connectivity == 26);
  if (prev_node_base_ >= 0 && !slices_.empty() && prev_width_ == width &&
      prev_height_ == height &&
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height) == cc_labels.size()) {
    const std::vector<int>& prev_cc = slices_.back().cc;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int b = cc_labels[static_cast<std::size_t>(y) * width + x];
        if (b < 0) continue;
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (!edges && (dx != 0 || dy != 0)) continue;        // 6: only (0,0)
            if (edges && !diag && dx != 0 && dy != 0) continue;  // 18: no corners
            const int nx = x + dx, ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
            const int a = prev_cc[static_cast<std::size_t>(ny) * width + nx];
            if (a < 0) continue;
            unite_first_seen(node_base + b, prev_node_base_ + a);
          }
        }
      }
    }
  }

  slices_.push_back(SliceRaster{slice_index, width, height, node_base, n, cc_labels});
  prev_width_ = width;
  prev_height_ = height;
  prev_node_base_ = node_base;
}

void SliceMatcher::finalize(std::vector<FeatureMapRow>& map_out,
                            std::vector<GlobalFeatureStat>& table_out,
                            std::vector<GlobalLabelRaster>& rasters_out) {
  rasters_out.clear();
  finalize(map_out, table_out, [&rasters_out](GlobalLabelRaster&& r) {
    rasters_out.push_back(std::move(r));
  });
}

void SliceMatcher::finalize(std::vector<FeatureMapRow>& map_out,
                            std::vector<GlobalFeatureStat>& table_out,
                            const GlobalRasterSink& emit) {
  map_out.clear();
  table_out.clear();
  const int N = static_cast<int>(parent_.size());
  if (N == 0) return;

  // Global ids in FIRST-SEEN (appearance) order: scan nodes 0..N-1 (slice-major),
  // give each newly-seen root the next id. Node 0 is the earliest, so ids come out
  // deterministically in appearance order and match the GUI's connected_components
  // first-seen renumbering.
  std::vector<int> gid_of_root(N, -1);
  std::vector<int> gid_of_node(N, -1);
  int next_gid = 0;
  for (int node = 0; node < N; ++node) {
    const int root = find(node);
    if (gid_of_root[root] < 0) gid_of_root[root] = next_gid++;
    gid_of_node[node] = gid_of_root[root];
  }

  // Aggregate node stats into their global feature (mergeable).
  std::vector<CcNodeStat> acc(static_cast<std::size_t>(next_gid));
  std::vector<int> min_z(next_gid, std::numeric_limits<int>::max());
  std::vector<int> max_z(next_gid, std::numeric_limits<int>::lowest());
  // Distinct slices per global feature. Nodes are numbered slice-major (add_slice
  // appends a contiguous block per slice), so walking 0..N-1 visits slices in
  // ascending order and a change in z is a new slice for that gid. That replaces
  // one unordered_set per global id -- on a deep stack there can be ~1e6 of them,
  // and the set headers alone outweigh the counts they hold.
  std::vector<int> nslices(static_cast<std::size_t>(next_gid), 0);
  std::vector<int> last_z(static_cast<std::size_t>(next_gid), std::numeric_limits<int>::min());
  std::vector<char> init(static_cast<std::size_t>(next_gid), 0);
  // The winning slice component's extremum, carried as a tuple so the reported
  // (x, y, z) always names one real voxel.
  std::vector<int> ext_z(static_cast<std::size_t>(next_gid), -1);
  std::vector<const CcNodeStat*> ext_node(static_cast<std::size_t>(next_gid), nullptr);
  // Per-slice quantity samples, for the reductions across slices.
  PerSliceAcc per_slice(static_cast<std::size_t>(next_gid), stats_.per_slice_quantities);

  for (int node = 0; node < N; ++node) {
    const int gid = gid_of_node[node];
    const CcNodeStat& p = node_stat_[static_cast<std::size_t>(node)];
    CcNodeStat& a = acc[static_cast<std::size_t>(gid)];
    if (!init[static_cast<std::size_t>(gid)]) {
      a = p;
      init[static_cast<std::size_t>(gid)] = 1;
    } else {
      a.area += p.area;
      a.base_sum += p.base_sum;
      a.base_sumsq += p.base_sumsq;
      a.filt_sum += p.filt_sum;
      a.filt_sumsq += p.filt_sumsq;
      a.base_min = std::min(a.base_min, p.base_min);
      a.base_max = std::max(a.base_max, p.base_max);
      a.filt_min = std::min(a.filt_min, p.filt_min);
      a.filt_max = std::max(a.filt_max, p.filt_max);
      a.min_x = std::min(a.min_x, p.min_x);
      a.max_x = std::max(a.max_x, p.max_x);
      a.min_y = std::min(a.min_y, p.min_y);
      a.max_y = std::max(a.max_y, p.max_y);
    }
    const int z = node_slice_[static_cast<std::size_t>(node)];
    min_z[gid] = std::min(min_z[gid], z);
    max_z[gid] = std::max(max_z[gid], z);
    if (last_z[static_cast<std::size_t>(gid)] != z) {
      last_z[static_cast<std::size_t>(gid)] = z;
      ++nslices[static_cast<std::size_t>(gid)];
    }

    // Seeding extremum: keep the component whose ext_filtered is most extreme in
    // the manifold's direction. This is the 3D analogue of the 2D rule, and it is
    // a selection rather than a per-field reduction -- reducing ext_x and
    // ext_base independently would report a position from one slice and a value
    // from another.
    if (stats_.spec.extremum && p.area > 0) {
      const CcNodeStat*& best = ext_node[static_cast<std::size_t>(gid)];
      const bool better = best == nullptr ||
                          (ascending_ ? p.ext_filtered < best->ext_filtered
                                      : p.ext_filtered > best->ext_filtered);
      if (better) {
        best = &p;
        ext_z[static_cast<std::size_t>(gid)] = z;
      }
    }
    per_slice.add(gid, p);
  }

  const auto stddev = [](double sum, double sumsq, std::size_t area) -> float {
    if (area == 0) return 0.0f;
    const double m = sum / static_cast<double>(area);
    const double var = sumsq / static_cast<double>(area) - m * m;
    return var > 0.0 ? static_cast<float>(std::sqrt(var)) : 0.0f;
  };

  // Master table, in global-id (appearance) order so gid == row index.
  table_out.reserve(static_cast<std::size_t>(next_gid));
  for (int gid = 0; gid < next_gid; ++gid) {
    const CcNodeStat& a = acc[static_cast<std::size_t>(gid)];
    const double area = a.area > 0 ? static_cast<double>(a.area) : 1.0;
    GlobalFeatureStat s;
    s.global_id = gid;
    s.voxel_count = a.area;
    s.num_slices = nslices[static_cast<std::size_t>(gid)];
    s.first_slice = min_z[gid];
    s.last_slice = max_z[gid];
    s.min_x = a.min_x; s.min_y = a.min_y; s.min_z = min_z[gid];
    s.max_x = a.max_x; s.max_y = a.max_y; s.max_z = max_z[gid];
    s.mean_base = static_cast<float>(a.base_sum / area);
    s.min_base = a.base_min; s.max_base = a.base_max;
    s.std_base = stddev(a.base_sum, a.base_sumsq, a.area);
    s.mean_filtered = static_cast<float>(a.filt_sum / area);
    s.min_filtered = a.filt_min; s.max_filtered = a.filt_max;
    s.std_filtered = stddev(a.filt_sum, a.filt_sumsq, a.area);
    if (const CcNodeStat* e = ext_node[static_cast<std::size_t>(gid)]) {
      s.ext_x = e->ext_x;
      s.ext_y = e->ext_y;
      s.ext_z = ext_z[static_cast<std::size_t>(gid)];
      s.ext_base = e->ext_base;
      s.ext_filtered = e->ext_filtered;
    }
    per_slice.finish(gid, stats_, s.per_slice);
    table_out.push_back(s);
  }

  // Per-slice -> global map (segment_id == per-slice CC id) and relabeled rasters.
  //
  // The relabel is done IN PLACE and the buffer is then moved to the sink, so no
  // second raster is ever allocated and each slice's memory is released as soon
  // as the sink is done with it. Rewriting cc ids to global ids in place is safe
  // because every pixel is read exactly once -- nothing later re-reads a pixel
  // and mistakes an already-written global id for a local one.
  //
  // Why this cannot be streamed during add_slice: a later slice can bridge two
  // components that were disjoint in earlier slices, retroactively merging their
  // ids, so no slice's final numbering is known until every slice is in.
  for (auto& sr : slices_) {
    for (int c = 0; c < sr.n; ++c) {
      map_out.push_back(FeatureMapRow{sr.slice_index, c, gid_of_node[sr.node_base + c]});
    }
    for (int& v : sr.cc) {
      if (v >= 0) v = gid_of_node[sr.node_base + v];
    }
    GlobalLabelRaster r;
    r.slice_index = sr.slice_index;
    r.width = sr.width;
    r.height = sr.height;
    r.data = std::move(sr.cc);
    emit(std::move(r));
    // Drop whatever the sink left behind; a writing sink leaves nothing.
    sr.cc.clear();
    sr.cc.shrink_to_fit();
  }
}

}  // namespace mscoupon
