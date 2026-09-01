#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace msseg {

// Parameters for the filter/transform stage. `operation` selects a diffg
// filter; `params` carries operation-specific keys (sigma, thresholds, ...).
struct FilterParams {
  std::string operation = "none";
  nlohmann::json params = nlohmann::json::object();
};

// Which per-feature reductions to compute. Statistics are the pipeline's hot
// path -- the per-feature row is rebuilt on every persistence change and
// marshalled across the pybind boundary -- so what a workflow does not ask for
// is not accumulated, and does not become a field.
//
// `Sample` is the seeding extremum: the minimum of an ascending manifold, the
// maximum of a descending one. It is the one reduction that is not an aggregate
// over the feature's pixels but a single distinguished pixel of it.
enum class Reduction { Mean, Min, Max, Std, Sample };

// One requested measurement channel, before expansion.
//
// `base` and `filtered` are the two rasters the pipeline already builds. Every
// other kind is a DERIVED channel: a Gaussian-derivative response computed off
// the BASE raster (i.e. after `base_filters`, so it reads in normalized units
// whenever normalization is on). Derived channels are measure-only -- they are
// never the topology field the MSC runs on, and never define the seeding
// extremum; see the role note on Msc2DPipeline::build.
//
// `sigmas` is a cross-product: one request with three sigmas expands to three
// channels (six for `hessian`, which yields two eigenvalues in 2D). Naming the
// scales in one entry is how a scale-space stack is actually specified, and it
// is also what lets the whole set collapse into a single diffg filter-bank
// traversal that shares its separable passes.
struct StatChannelRequest {
  // "base" | "filtered" | "blur" | "edges" | "gradmag" | "laplacian" | "hessian"
  std::string kind = "base";
  // Empty for base/filtered (they take no scale); required otherwise.
  std::vector<double> sigmas;
  // `hessian` only. With true (diffg's default) the eigenvalues are ordered by
  // magnitude, so "largest"/"smallest" mean largest/smallest |lambda|.
  bool sort_by_absolute_value = true;
  // Optional name prefix, replacing the generated one (e.g. "blur" -> "sharp").
  std::string name;
};

// One channel after expansion: a stable name plus how to produce it. Resolved
// once per run, so the hot loops index channels by SLOT and never by name.
struct ResolvedStatChannel {
  std::string name;           // "base", "filtered", "blur_s0.7", "hess_largest_s3"
  std::string kind;           // the request kind it came from
  double sigma = 0.0;         // 0 for base/filtered
  int slot_in_request = 0;    // hessian: 0 = largest, 1 = smallest
  bool sort_by_absolute_value = true;
};

// Aggregate reductions on the channels a workflow actually reads.
//
// The filtered channel defaults OFF: its aggregates (mean/min/max/std of the
// topology field) have no reader anywhere in the tree, while costing a
// sum + sum-of-squares per pixel and 24 B per manifold. Note this is only about
// the *aggregates* -- the filtered channel's per-feature min/max still gets
// computed regardless when `extremum` is on, because those are what define the
// seeding critical point.
struct StatsSpec {
  bool base_channel = true;
  bool filtered_channel = false;
  // Derived measurement channels, in config order. Empty (the default) is
  // exactly the pre-existing two-channel behaviour.
  std::vector<StatChannelRequest> derived;
  bool mean = true, min = true, max = true, std = true;
  bool extremum = true;
  // Experimental merge-tree-inspired contrast on the base channel. The slice
  // range used to shift values positive is selected by these percentiles:
  // 0/100 means the absolute finite min/max; 1/99 gives a robust range.
  bool relevance = true;
  double relevance_low_percentile = 0.0;
  double relevance_high_percentile = 100.0;
  // Radius (in pixels) of the square window used to sample the BASE channel at
  // the seeding extremum. 0 => the single critical pixel; r > 0 => the mean over
  // the (2r+1)^2 window, clamped at the image border, trading the exact critical
  // value for noise robustness.
  int extremum_sample_radius = 0;

  bool any_aggregate() const { return mean || min || max || std; }
  bool needs_sums() const { return (mean || std) && (base_channel || filtered_channel); }
  // The filtered channel's min/max are load-bearing for the extremum even when
  // its aggregates are switched off.
  bool needs_filtered_extent() const { return extremum || (filtered_channel && (min || max)); }
};

// Parameters for the 2D Morse-Smale compute + simplification + 2-manifold
// labeling (backed by MSCEER's GInt::Msc2D facade).
struct Msc2DParams {
  std::optional<float> persistence_absolute;
  std::optional<float> persistence_percent = 10.0f;
  std::string compute_algorithm = "serial";  // "serial" | "partitioned"
  // How the persistence simplification is represented.
  //   "msc"          -- the Morse-Smale complex + its cancellation hierarchy.
  //   "merge_forest" -- the extremum network: extrema from the flow-terminal
  //                     labeling, saddles read off the same maps, and one
  //                     union-find pass recording a forest that answers every
  //                     threshold. Much cheaper to build (no MSC, no hierarchy,
  //                     no partition reconcile) but NOT bit-identical: it
  //                     measures ~99.7-99.9% pixel agreement with "msc".
  // Requires an msc_2d_lib new enough to expose ComputeOptions::simplification;
  // older pins warn and use the MSC. Env kill-switch: MSSEG_SIMPLIFICATION=msc.
  std::string simplification = "msc";  // "msc" | "merge_forest"
  bool accurate_ascending = true;
  bool accurate_descending = true;
  std::string manifold = "ascending";  // "ascending" | "descending"
  // Partition/thread count for the discrete gradient and (in "partitioned"
  // mode) the parallel MSC/hierarchy build. 0 => leave the msc_2d_lib default.
  int requested_parallelism = 0;
  // Compute the base discrete gradient on the GPU (MSCEER's fused CUDA
  // kernel, bit-identical to the CPU pairing). Requires msc_2d_lib built with
  // GPU_DGRAD_ENABLED (MSSeg: -DMSSEG_GPU=ON) and a CUDA device; otherwise
  // MSCEER falls back to the CPU path with a notice.
  bool use_gpu_gradient = false;
  // Accumulate the per-region statistics on the GPU (label CSR + segmented
  // reduces over device-resident diffg channels). Unset => follows
  // use_gpu_gradient. Requires -DMSSEG_GPU=ON and a CUDA device; falls back to
  // the CPU loop per call otherwise. Env kill-switch: MSSEG_GPU_STATS=0.
  std::optional<bool> use_gpu_stats;
  // Per-arc-dimension geometry construction, passed through to the MSC compute.
  // The core does not impose a policy here -- each instance decides (building arc
  // geometry is costly, so it defaults off; turn on the dims a downstream
  // approach actually consumes).
  std::array<bool, 3> build_arc_geometry = {{false, false, false}};
  // Which per-feature statistics to accumulate. `stats.extremum_sample_radius`
  // supersedes the legacy `extremum_sample_radius` below when a workflow sets a
  // `statistics` block.
  StatsSpec stats;
  // Deprecated alias for `stats.extremum_sample_radius`, kept so existing
  // `msc.extremum_sample_radius` configs and the GUI's params JSON keep working.
  int extremum_sample_radius = 0;
};

}  // namespace msseg
