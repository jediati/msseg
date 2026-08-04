#include "msseg/compute/msc2d.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

#include "msc_2d_lib.h"

#include "msseg/graph/merge_tree.hpp"
#include "msseg/graph/msc_graph.hpp"

namespace msseg {
namespace {

// The linked msc_2d_lib may or may not expose a ComputeOptions/BuilderMode
// surface (it depends on the pinned MSCEER revision). Detect it at compile
// time so we can honor compute_algorithm/accurate flags when available and
// fall back to the legacy 5-arg compute() otherwise.
template <typename MscType, typename = void>
struct HasComputeOptions : std::false_type {};

template <typename MscType>
struct HasComputeOptions<MscType, std::void_t<typename MscType::ComputeOptions>> : std::true_type {};

template <typename MscType>
void compute_with_algorithm(MscType& msc, const float* pixels, int rows, int cols, const Msc2DParams& cfg) {
  if constexpr (HasComputeOptions<MscType>::value) {
    typename MscType::ComputeOptions options;
    options.accurateAsc = cfg.accurate_ascending;
    options.accurateDsc = cfg.accurate_descending;
    // Pass the caller's per-dim arc-geometry choice straight through; the core
    // imposes no policy (the instance decides -- see Msc2DParams).
    options.buildArcGeometry[0] = cfg.build_arc_geometry[0];
    options.buildArcGeometry[1] = cfg.build_arc_geometry[1];
    options.buildArcGeometry[2] = cfg.build_arc_geometry[2];
    if (cfg.requested_parallelism > 0) {
      options.requestedParallelism = cfg.requested_parallelism;
    }
    if (cfg.compute_algorithm == "partitioned") {
      options.builderMode = MscType::BuilderMode::Partitioned;
    } else {
      options.builderMode = MscType::BuilderMode::Serial;
    }
    msc.compute(pixels, rows, cols, options);
  } else {
    if (cfg.compute_algorithm == "partitioned") {
      throw std::runtime_error(
          "Configured msc.compute_algorithm='partitioned' but linked msc_2d_lib does not expose "
          "ComputeOptions/BuilderMode.");
    }
    msc.compute(pixels, rows, cols, cfg.accurate_ascending, cfg.accurate_descending);
  }
}

}  // namespace

std::vector<int> compute_msc2d_labels(const diffg::Image<float>& filtered, const Msc2DParams& cfg) {
  const int width = static_cast<int>(filtered.dims().width);
  const int height = static_cast<int>(filtered.dims().height);
  if (width <= 0 || height <= 0) {
    throw std::runtime_error("Invalid image dimensions for MSC.");
  }
  if (filtered.dims().depth != 1) {
    throw std::runtime_error("compute_msc2d_labels requires a 2D image (depth == 1).");
  }

  GInt::Msc2D::Msc2D msc;
  compute_with_algorithm(msc, filtered.data(), height, width, cfg);

  float persistence_absolute = 0.0f;
  if (cfg.persistence_absolute.has_value()) {
    persistence_absolute = *cfg.persistence_absolute;
  } else if (cfg.persistence_percent.has_value()) {
    float min_v = std::numeric_limits<float>::max();
    float max_v = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < filtered.size(); ++i) {
      const float v = filtered.data()[i];
      min_v = std::min(min_v, v);
      max_v = std::max(max_v, v);
    }
    const float range = max_v - min_v;
    persistence_absolute = range * (*cfg.persistence_percent / 100.0f);
  } else {
    throw std::runtime_error("MSC persistence not configured.");
  }
  msc.setPersistence(persistence_absolute);

  if (cfg.manifold == "ascending") {
    return msc.ascending2Manifolds().labels;
  }
  if (cfg.manifold == "descending") {
    return msc.descending2Manifolds().labels;
  }
  throw std::runtime_error("Invalid msc.manifold value. Use 'ascending' or 'descending'.");
}

// --------------------------------------------------------------------------- //
// Msc2DPipeline: two-phase 2D facade (build once, re-threshold cheaply).
// --------------------------------------------------------------------------- //
namespace {

void accumulate(Msc2DFeatureStat& dst, const Msc2DFeatureStat& src) {
  if (src.area == 0) return;
  if (dst.area == 0) {
    const NodeId keep = dst.feature_id;
    dst = src;
    dst.feature_id = keep;
    return;
  }
  dst.area += src.area;
  dst.base_sum += src.base_sum;
  dst.base_sumsq += src.base_sumsq;
  dst.base_min = std::min(dst.base_min, src.base_min);
  dst.base_max = std::max(dst.base_max, src.base_max);
  dst.filt_sum += src.filt_sum;
  dst.filt_sumsq += src.filt_sumsq;
  dst.filt_min = std::min(dst.filt_min, src.filt_min);
  dst.filt_max = std::max(dst.filt_max, src.filt_max);
  dst.min_x = std::min(dst.min_x, src.min_x);
  dst.min_y = std::min(dst.min_y, src.min_y);
  dst.max_x = std::max(dst.max_x, src.max_x);
  dst.max_y = std::max(dst.max_y, src.max_y);
}

}  // namespace

struct Msc2DPipeline::Impl {
  int width = 0;
  int height = 0;
  float value_range = 0.0f;
  bool ascending = true;
  float current_persistence = 0.0f;

  MscGraph graph;
  MergeTree tree;
  // Per-base-extremum leaf statistics, indexed by compact NodeId (entries at
  // non-extremum node ids stay area==0 and are ignored).
  std::vector<Msc2DFeatureStat> leaf_stats;
  // Compact extremum NodeId per pixel (row-major), -1 where unlabeled.
  std::vector<int> base_labels;

  // Derived at the current persistence:
  std::vector<int> labels;                       // surviving feature id per pixel
  std::vector<Msc2DFeatureStat> features;        // aggregated per surviving feature
};

Msc2DPipeline::Msc2DPipeline() : impl_(std::make_unique<Impl>()) {}
Msc2DPipeline::~Msc2DPipeline() = default;
Msc2DPipeline::Msc2DPipeline(Msc2DPipeline&&) noexcept = default;
Msc2DPipeline& Msc2DPipeline::operator=(Msc2DPipeline&&) noexcept = default;

void Msc2DPipeline::build(const diffg::Image<float>& base, const diffg::Image<float>& filtered,
                          const Msc2DParams& cfg) {
  const int width = static_cast<int>(filtered.dims().width);
  const int height = static_cast<int>(filtered.dims().height);
  if (width <= 0 || height <= 0) throw std::runtime_error("Invalid image dimensions for MSC.");
  if (filtered.dims().depth != 1) throw std::runtime_error("Msc2DPipeline requires depth == 1.");
  if (base.dims().width != filtered.dims().width || base.dims().height != filtered.dims().height) {
    throw std::runtime_error("Msc2DPipeline: base and filtered images must share dimensions.");
  }
  const bool ascending = (cfg.manifold != "descending");
  const int extremum_dim = ascending ? 0 : 2;

  impl_->width = width;
  impl_->height = height;
  impl_->ascending = ascending;

  // Value range of the filtered field (for percent->absolute persistence).
  float min_v = std::numeric_limits<float>::max();
  float max_v = std::numeric_limits<float>::lowest();
  for (std::size_t i = 0; i < filtered.size(); ++i) {
    min_v = std::min(min_v, filtered.data()[i]);
    max_v = std::max(max_v, filtered.data()[i]);
  }
  impl_->value_range = max_v - min_v;

  // Heavy: compute the MSC once, then view it at persistence -1 (all living) so
  // criticalPoints/arcGeometry/2-manifolds give the finest base complex.
  GInt::Msc2D::Msc2D msc;
  compute_with_algorithm(msc, filtered.data(), height, width, cfg);
  msc.setPersistence(-1.0f);

  const std::vector<GInt::Msc2D::CriticalPoint> cps = msc.criticalPoints();
  const std::vector<GInt::Msc2D::ArcGeometry> arcs = msc.arcGeometry();
  const GInt::Msc2D::LabelImage base_img =
      ascending ? msc.ascending2Manifolds() : msc.descending2Manifolds();

  // Compact NodeId space over all critical points, in criticalPoints() order.
  std::unordered_map<int, NodeId> raw_to_compact;
  raw_to_compact.reserve(cps.size() * 2);
  MscGraph& graph = impl_->graph;
  graph.nodes.clear();
  graph.arcs.clear();
  graph.adjacency.clear();
  graph.nodes.reserve(cps.size());
  for (const auto& cp : cps) {
    const NodeId cid = static_cast<NodeId>(graph.nodes.size());
    raw_to_compact[cp.id] = cid;
    MscNode n;
    n.id = cid;
    n.cell_index = cp.id;   // stash the raw msc_2d_lib node id
    n.index_dim = cp.dim;
    n.value = cp.value;
    n.pos = {cp.x, cp.y, 0.0f};
    graph.nodes.push_back(n);
  }
  graph.adjacency.assign(graph.nodes.size(), {});
  graph.arcs.reserve(arcs.size());
  for (const auto& ag : arcs) {
    const auto lo = raw_to_compact.find(ag.lowerNodeId);
    const auto hi = raw_to_compact.find(ag.upperNodeId);
    if (lo == raw_to_compact.end() || hi == raw_to_compact.end()) continue;
    MscArc a;
    a.id = static_cast<NodeId>(graph.arcs.size());
    a.lower = lo->second;
    a.upper = hi->second;
    a.index_dim = ag.dim;
    graph.adjacency[static_cast<std::size_t>(a.lower)].push_back(a.id);
    graph.adjacency[static_cast<std::size_t>(a.upper)].push_back(a.id);
    graph.arcs.push_back(std::move(a));
  }

  // Base labels: remap raw extremum node ids -> compact ids.
  impl_->base_labels.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), -1);
  const std::size_t npix = impl_->base_labels.size();
  for (std::size_t i = 0; i < npix && i < base_img.labels.size(); ++i) {
    const int raw = base_img.labels[i];
    if (raw < 0) continue;
    const auto it = raw_to_compact.find(raw);
    if (it != raw_to_compact.end()) impl_->base_labels[i] = static_cast<int>(it->second);
  }

  // Per-base-extremum leaf statistics (one pass over pixels, both images).
  std::vector<Msc2DFeatureStat>& leaf = impl_->leaf_stats;
  leaf.assign(graph.nodes.size(), Msc2DFeatureStat{});
  for (std::size_t c = 0; c < leaf.size(); ++c) {
    leaf[c].feature_id = static_cast<NodeId>(c);
    leaf[c].base_min = std::numeric_limits<float>::max();
    leaf[c].base_max = std::numeric_limits<float>::lowest();
    leaf[c].filt_min = std::numeric_limits<float>::max();
    leaf[c].filt_max = std::numeric_limits<float>::lowest();
    leaf[c].min_x = width;
    leaf[c].min_y = height;
    leaf[c].max_x = -1;
    leaf[c].max_y = -1;
  }
  for (std::size_t i = 0; i < npix; ++i) {
    const int c = impl_->base_labels[i];
    if (c < 0) continue;
    Msc2DFeatureStat& s = leaf[static_cast<std::size_t>(c)];
    const float b = base.data()[i];
    const float f = filtered.data()[i];
    const int x = static_cast<int>(i % static_cast<std::size_t>(width));
    const int y = static_cast<int>(i / static_cast<std::size_t>(width));
    s.area += 1;
    s.base_sum += b;
    s.base_sumsq += static_cast<double>(b) * b;
    s.base_min = std::min(s.base_min, b);
    s.base_max = std::max(s.base_max, b);
    s.filt_sum += f;
    s.filt_sumsq += static_cast<double>(f) * f;
    s.filt_min = std::min(s.filt_min, f);
    s.filt_max = std::max(s.filt_max, f);
    s.min_x = std::min(s.min_x, x);
    s.min_y = std::min(s.min_y, y);
    s.max_x = std::max(s.max_x, x);
    s.max_y = std::max(s.max_y, y);
  }

  // Merge tree over the base complex, with per-extremum voxel counts.
  std::vector<std::int64_t> counts(graph.nodes.size(), 0);
  for (std::size_t c = 0; c < leaf.size(); ++c) counts[c] = leaf[c].area;
  impl_->tree = build_merge_tree(graph, counts, ascending, extremum_dim, /*saddle_dim=*/1);

  // Initial persistence from cfg.
  float persistence = 0.0f;
  if (cfg.persistence_absolute.has_value()) {
    persistence = *cfg.persistence_absolute;
  } else if (cfg.persistence_percent.has_value()) {
    persistence = impl_->value_range * (*cfg.persistence_percent / 100.0f);
  }
  select_persistence(persistence);
}

int Msc2DPipeline::width() const { return impl_->width; }
int Msc2DPipeline::height() const { return impl_->height; }
float Msc2DPipeline::value_range() const { return impl_->value_range; }
float Msc2DPipeline::current_persistence() const { return impl_->current_persistence; }

void Msc2DPipeline::select_persistence(float persistence_absolute) {
  impl_->current_persistence = persistence_absolute;
  const float inf = std::numeric_limits<float>::infinity();
  // Persistence-only cut: no value barrier (ascending -> -inf, descending -> +inf).
  const float cut = impl_->ascending ? -inf : inf;
  const std::unordered_map<NodeId, NodeId> survivors =
      branch_survivors(impl_->tree, cut, persistence_absolute);

  // Remap base labels (compact extremum id) -> surviving feature id.
  impl_->labels.assign(impl_->base_labels.size(), -1);
  std::unordered_map<NodeId, Msc2DFeatureStat> agg;
  for (std::size_t i = 0; i < impl_->base_labels.size(); ++i) {
    const int c = impl_->base_labels[i];
    if (c < 0) continue;
    const auto it = survivors.find(static_cast<NodeId>(c));
    const NodeId sv = (it != survivors.end()) ? it->second : static_cast<NodeId>(c);
    impl_->labels[i] = static_cast<int>(sv);
  }
  // Aggregate leaf statistics up to the surviving features.
  for (const auto& node : impl_->tree.nodes) {
    if (!node.is_leaf) continue;
    const NodeId ext = node.msc_node;
    const auto it = survivors.find(ext);
    const NodeId sv = (it != survivors.end()) ? it->second : ext;
    Msc2DFeatureStat& dst = agg[sv];
    dst.feature_id = sv;
    accumulate(dst, impl_->leaf_stats[static_cast<std::size_t>(ext)]);
  }
  impl_->features.clear();
  impl_->features.reserve(agg.size());
  for (auto& [id, s] : agg) impl_->features.push_back(s);
}

const std::vector<int>& Msc2DPipeline::labels() const { return impl_->labels; }
std::vector<Msc2DFeatureStat> Msc2DPipeline::feature_stats() const { return impl_->features; }
std::string Msc2DPipeline::merge_tree_json() const { return merge_tree_to_json(impl_->tree); }

}  // namespace msseg
