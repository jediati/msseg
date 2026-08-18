#pragma once

#include <array>
#include <optional>
#include <string>

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
  bool mean = true, min = true, max = true, std = true;
  bool extremum = true;
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
  bool accurate_ascending = true;
  bool accurate_descending = true;
  std::string manifold = "ascending";  // "ascending" | "descending"
  // Partition/thread count for the discrete gradient and (in "partitioned"
  // mode) the parallel MSC/hierarchy build. 0 => leave the msc_2d_lib default.
  int requested_parallelism = 0;
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
