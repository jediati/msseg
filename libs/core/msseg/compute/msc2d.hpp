#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "diffg/image.hpp"
#include "msseg/graph/msc_graph.hpp"
#include "msseg/workflow/params.hpp"

namespace msseg {

// Compute the 2D Morse-Smale complex over a filtered slice, simplify to the
// configured persistence, and return the ascending/descending 2-manifold
// label image (row-major, one label per pixel). Requires depth == 1.
std::vector<int> compute_msc2d_labels(const diffg::Image<float>& filtered, const Msc2DParams& cfg);

// Aggregated statistics for one 2D feature (a surviving 2-manifold at the current
// persistence). Sums/sumsq are mergeable so features combine cheaply up the merge
// tree; means/std are derived (mean = sum/area). `min/max_x/y` is the bounding box.
struct Msc2DFeatureStat {
  NodeId feature_id = -1;   // surviving extremum NodeId (compact) == label value
  std::int64_t area = 0;
  double base_sum = 0.0, base_sumsq = 0.0;   // on the original (unfiltered) image
  float base_min = 0.0f, base_max = 0.0f;
  double filt_sum = 0.0, filt_sumsq = 0.0;   // on the filtered (topology) field
  float filt_min = 0.0f, filt_max = 0.0f;
  int min_x = 0, min_y = 0, max_x = 0, max_y = 0;
};

// Two-phase 2D pipeline mirroring cellseg's Msc3D/CellPipeline. build() runs the
// heavy MSC compute once and caches the base (finest) 2-manifold decomposition, a
// merge tree that mirrors the manifold merger, and per-base-manifold statistics
// on both the base image and the filtered field. select_persistence() then
// re-thresholds cheaply (a merge-tree branch cut + statistics aggregation) so a
// GUI can drag a persistence slider without recomputing the MSC. The merge tree +
// base labeling are the authoritative segmentation for both the viewer and CLI.
//
// Implemented in msc2d.cpp (the GInt firewall TU); this is a PIMPL facade so the
// MSCEER types never leak into the header.
class Msc2DPipeline {
 public:
  Msc2DPipeline();
  ~Msc2DPipeline();
  Msc2DPipeline(Msc2DPipeline&&) noexcept;
  Msc2DPipeline& operator=(Msc2DPipeline&&) noexcept;
  Msc2DPipeline(const Msc2DPipeline&) = delete;
  Msc2DPipeline& operator=(const Msc2DPipeline&) = delete;

  // Heavy lift: compute the MSC over `filtered`, build the base decomposition +
  // merge tree + per-manifold stats (base stats over `base`, i.e. the original
  // image; topology stats over `filtered`). `base` and `filtered` must share
  // dimensions and be 2D (depth == 1). The initial persistence follows `cfg`.
  void build(const diffg::Image<float>& base, const diffg::Image<float>& filtered,
             const Msc2DParams& cfg);

  int width() const;
  int height() const;
  // Value range (max - min) of the filtered field, for percent->absolute.
  float value_range() const;
  float current_persistence() const;

  // Re-threshold at an absolute persistence: remap the base labels to surviving
  // features and re-aggregate statistics. Cheap relative to build().
  void select_persistence(float persistence_absolute);

  // Feature id per pixel (row-major) at the current persistence: the surviving
  // extremum NodeId, or -1 where the base was unlabeled.
  const std::vector<int>& labels() const;
  // Per-surviving-feature aggregated statistics at the current persistence.
  std::vector<Msc2DFeatureStat> feature_stats() const;
  // The merge tree (flat JSON) mirroring the manifold merger.
  std::string merge_tree_json() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace msseg
