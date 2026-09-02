#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "diffg/image.hpp"
#include "msseg/compute/channel_stats.hpp"
#include "msseg/graph/msc_graph.hpp"
#include "msseg/workflow/params.hpp"
#include "msseg/workflow/stat_channels.hpp"

namespace msseg {

struct StatChannelBank;

// Compute the 2D Morse-Smale complex over a filtered slice, simplify to the
// configured persistence, and return the ascending/descending 2-manifold
// label image (row-major, one label per pixel). Requires depth == 1.
std::vector<int> compute_msc2d_labels(const diffg::Image<float>& filtered, const Msc2DParams& cfg);

// Geometry and extremum machinery for one 2D feature (a living 2-manifold at the
// current persistence). `min/max_x/y` is the bounding box.
//
// The per-channel AGGREGATES do not live here -- they live in a parallel flat
// ChannelStats table owned by the pipeline, indexed by the same feature order.
// That is what lets a workflow ask for a twelve-channel scale-space stack
// without this struct growing a member per channel: see
// Msc2DPipeline::feature_channels().
//
// What remains here is what is NOT a measurement: the bounding box, the slice
// metadata `relevance` needs, and the extremum machinery. `filt_min`/`filt_max`
// stay because they LOCATE the seeding critical point -- they are computed
// whenever the extremum is wanted, whether or not "filtered" is a measurement
// channel at all.
struct Msc2DFeatureStat {
  NodeId feature_id = -1;   // living extremum compact id == label value
  std::int64_t area = 0;
  // Extent of the filtered (topology) field over the feature. Extremum
  // machinery, not a statistic -- the `filtered` measurement channel is what a
  // workflow queries as min_filtered/max_filtered.
  float filt_min = 0.0f, filt_max = 0.0f;
  // Slice-level finite base-channel percentiles used by `relevance_base`.
  // These are metadata shared by every feature in a slice, not mergeable stats.
  float base_relevance_floor = 0.0f, base_relevance_ceiling = 0.0f;
  int min_x = 0, min_y = 0, max_x = 0, max_y = 0;

  // The feature's SEEDING extremum: the minimum for ascending manifolds, the
  // maximum for descending, always a real pixel of the feature. Unlike
  // everything above these are NOT mergeable -- a merged feature inherits the
  // SURVIVING extremum, so select_persistence() stamps them from the surviving
  // base manifold instead of accumulating them. That is why `ext_filtered` can
  // sit above `filt_min`: persistence, not depth, decides which minimum
  // survives, so a deep low-persistence basin can merge into a shallower one.
  // Every measurement channel is sampled there too, as ChannelStats::ext().
  float ext_x = -1.0f, ext_y = -1.0f;
  float ext_filtered = 0.0f;
};

// One adjacency between two living features at the current persistence, in the
// compact id space of labels()/feature_id. a < b, one entry per unordered pair,
// sorted by (a, b). saddle_value is the most extreme saddle joining the pair
// (lowest for ascending/minima, highest for descending/maxima); count is how
// many base arcs collapsed onto the pair.
struct Msc2DRegionArc { int a = -1; int b = -1; float saddle_value = 0.0f; int count = 0; };

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
  // per-manifold statistics over every measurement channel `cfg.stats` asks for.
  // `base` and `filtered` must share dimensions and be 2D (depth == 1). The
  // initial persistence follows `cfg`, which also caps the cancellation
  // hierarchy (the max selectable persistence).
  //
  // The two arguments carry ROLES that the measurement channel set does not:
  // `filtered` is the topology field the MSC runs on and the field whose extent
  // locates each feature's seeding critical point; `base` is the raster the
  // derived scale-space channels are computed FROM and the one `relevance` is
  // measured on. Either may additionally be a measurement channel, but a derived
  // channel is measure-only and can be neither.
  //
  // `bank`, when non-null, is a caller-owned channel bank to measure against
  // instead of building one here. It must have been resolved from the same
  // spec. The CLI passes one because the connected-component stage measures the
  // very same channels a moment later, and the scale-space stack should be
  // traversed once per slice; the GUI passes nullptr so the rasters are freed
  // as soon as the per-manifold cells are accumulated, rather than held for
  // every primed slice.
  void build(const diffg::Image<float>& base, const diffg::Image<float>& filtered,
             const Msc2DParams& cfg, const StatChannelBank* bank = nullptr);

  int width() const;
  int height() const;
  // Value range (max - min) of the filtered field, for percent->absolute.
  float value_range() const;
  float current_persistence() const;
  float base_relevance_floor() const;
  float base_relevance_ceiling() const;

  // Re-threshold at an absolute persistence via native cancellation: remap base
  // labels to their living representatives and roll up cached stats. Cheap relative
  // to build() (no gradient/base recompute). Clamped to the build-time cap.
  void select_persistence(float persistence_absolute);

  // Feature id per pixel (row-major) at the current persistence: the living
  // extremum compact id, or -1 where the base was unlabeled.
  const std::vector<int>& labels() const;
  // Living-region adjacency at the current persistence (which living features
  // touch which, through a saddle). Empty when the linked msc_2d_lib predates
  // livingRegionArcs(). Computed lazily on first call after each
  // select_persistence() and cached until the next one.
  const std::vector<Msc2DRegionArc>& region_arcs();
  // Per-living-feature geometry + extremum at the current persistence.
  std::vector<Msc2DFeatureStat> feature_stats() const;
  // The resolved measurement channels, in slot order. This is the schema for
  // feature_channels(): slot k of every feature is channels()[k].
  const std::vector<ResolvedStatChannel>& channels() const;
  // Per-living-feature per-channel aggregates at the current persistence,
  // indexed in lockstep with feature_stats().
  const ChannelStats& feature_channels() const;
  // The statistics build() was asked for. Fields the spec excludes are left at
  // zero in the structs above, so a consumer turning stats into named rows must
  // consult this rather than emitting every member.
  const StatsSpec& stats() const;

  // Free this pipeline's GPU residue (MSCEER's device label context, ~2-3
  // label images of VRAM) while keeping every host-side result. A stack viewer
  // holding one primed pipeline per slice calls this on the slices it leaves;
  // the next select_persistence() on this pipeline lazily re-uploads once and
  // is GPU-painted again. No-op on CPU builds/pins. build() also calls it on
  // completion, so priming a long sequence never accumulates contexts.
  void release_gpu();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace msseg
