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

// Aggregated statistics for one 2D feature (a living 2-manifold at the current
// persistence). Sums/sumsq are mergeable so per-base-manifold stats combine cheaply
// when rolled up to living features; means/std are derived (mean = sum/area).
// `min/max_x/y` is the bounding box.
struct Msc2DFeatureStat {
  NodeId feature_id = -1;   // living extremum compact id == label value
  std::int64_t area = 0;
  double base_sum = 0.0, base_sumsq = 0.0;   // on the original (unfiltered) image
  float base_min = 0.0f, base_max = 0.0f;
  double filt_sum = 0.0, filt_sumsq = 0.0;   // on the filtered (topology) field
  float filt_min = 0.0f, filt_max = 0.0f;
  int min_x = 0, min_y = 0, max_x = 0, max_y = 0;

  // The feature's SEEDING extremum: the minimum for ascending manifolds, the
  // maximum for descending, always a real pixel of the feature. Unlike
  // everything above these are NOT mergeable -- a merged feature inherits the
  // SURVIVING extremum, so select_persistence() stamps them from the surviving
  // base manifold instead of accumulating them. That is why `ext_filtered` can
  // sit above `filt_min`: persistence, not depth, decides which minimum
  // survives, so a deep low-persistence basin can merge into a shallower one.
  // `ext_base` samples the BASE channel there -- a single pixel, or the mean
  // over a window (see Msc2DParams::extremum_sample_radius).
  float ext_x = -1.0f, ext_y = -1.0f;
  float ext_base = 0.0f, ext_filtered = 0.0f;
};

// Two-phase 2D pipeline. build() runs the heavy MSC compute once, keeps the MSCEER
// engine alive, and caches the base (finest) 2-manifold decomposition plus
// per-base-manifold statistics on both the base image and the filtered field.
// select_persistence() then re-thresholds cheaply via MSCEER's NATIVE cancellation
// hierarchy (setPersistence + ascending/descending2Manifolds remap each base
// minimum to its living representative) and rolls up the cached base stats -- so a
// GUI can drag a persistence slider without recomputing the MSC, and every living
// feature stays spatially connected (adjacent-basin cancellation). The cancellation
// hierarchy is capped at the configured max persistence at build time.
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

  // Heavy lift: compute the MSC over `filtered`, cache the base decomposition +
  // per-manifold stats (base stats over `base`, i.e. the original image; topology
  // stats over `filtered`). `base` and `filtered` must share dimensions and be 2D
  // (depth == 1). The initial persistence follows `cfg`, which also caps the
  // cancellation hierarchy (the max selectable persistence).
  void build(const diffg::Image<float>& base, const diffg::Image<float>& filtered,
             const Msc2DParams& cfg);

  int width() const;
  int height() const;
  // Value range (max - min) of the filtered field, for percent->absolute.
  float value_range() const;
  float current_persistence() const;

  // Re-threshold at an absolute persistence via native cancellation: remap base
  // labels to their living representatives and roll up cached stats. Cheap relative
  // to build() (no gradient/base recompute). Clamped to the build-time cap.
  void select_persistence(float persistence_absolute);

  // Feature id per pixel (row-major) at the current persistence: the living
  // extremum compact id, or -1 where the base was unlabeled.
  const std::vector<int>& labels() const;
  // Per-living-feature aggregated statistics at the current persistence.
  std::vector<Msc2DFeatureStat> feature_stats() const;
  // The statistics build() was asked for. Fields the spec excludes are left at
  // zero in the structs above, so a consumer turning stats into named rows must
  // consult this rather than emitting every member.
  const StatsSpec& stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace msseg
