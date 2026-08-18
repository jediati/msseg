#pragma once

// Two-point normalization: map a pair of measured intensity landmarks onto
// [0, 1] so a single threshold transfers across a whole stack.
//
// WHY THIS IS A FILTER. Thresholds could instead be converted at every
// comparison site, but that requires classifying each statistic as
// location-like (mean/min/max, affine) or scale-like (std, scale-only), and it
// breaks down for a 3D feature spanning slices with different landmarks. Making
// normalization a filter stage sidesteps both: statistics are computed on the
// channel's end result, so they come out already normalized, sums merge across
// slices correctly, and std falls out right on its own because the offset
// cancels (std((v-lo)/s) == std(v)/s).
//
// WHAT IT DOES NOT BREAK. The map is affine with a positive scale, so it is
// order-preserving: the discrete gradient, the MSC and the merge hierarchy are
// unchanged. Nothing in the pipeline treats an exact-zero pixel VALUE as
// no-data (background is carried by label -1), so shifting zeros off zero is
// safe. `omit_zeros` remains a property of the fit, not of the transform.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "mscoupon/types.hpp"

namespace mscoupon {

// The two landmarks, in the source channel's own units.
struct TwoPoint {
  double low = 0.0;
  double high = 1.0;

  bool valid() const { return high > low; }
  double scale() const { return high - low; }

  // Normalized -> raw. `t = 0.7` means 0.3*low + 0.7*high.
  double to_raw(double t) const { return low + t * (high - low); }
  // Raw -> normalized.
  double to_norm(double v) const { return (v - low) / (high - low); }
};

// How the two landmarks are measured.
enum class NormalizeMethod {
  Gmm,        // fit a 2-component mixture; landmarks from the component stats
  Histogram,  // smooth the histogram; landmarks from the two strongest maxima
  Regions,    // two caller-specified rectangles
  Manual,     // literal low/high values
};

struct NormalizeConfig {
  NormalizeMethod method = NormalizeMethod::Gmm;

  // Which measure outputs become the landmarks. Defaults are per-method:
  //   gmm        mu_1 / mu_2      (also: hard_mean_N, median_N, mode_N, sigma_N)
  //   histogram  peak_low / peak_high
  //   regions    air_p50 / metal_p50
  std::string low_from;
  std::string high_from;

  // Manual landmarks; also the fallback when a fit fails.
  std::optional<double> manual_low;
  std::optional<double> manual_high;

  // Method-specific options, parsed from the same JSON the measures accept.
  nlohmann::json params = nlohmann::json::object();

  // Clamp the normalized output to [0, 1]. Off by default: values outside the
  // landmarks are meaningful, and clipping would destroy the topology.
  bool clamp = false;
};

// Parse a `normalize` filter op's params block.
NormalizeConfig parse_normalize_config(const nlohmann::json& params);

// Measure the two landmarks of `image` under `cfg`.
// Throws std::runtime_error if the measure fails or yields high <= low and no
// manual fallback was supplied.
TwoPoint measure_two_point(const Image2D& image, const NormalizeConfig& cfg);

// Apply the affine map in place. No-op when `tp` is invalid.
void apply_two_point(Image2D& image, const TwoPoint& tp, bool clamp = false);

}  // namespace mscoupon
