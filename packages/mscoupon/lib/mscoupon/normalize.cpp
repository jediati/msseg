#include "mscoupon/normalize.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

#include "mscoupon/measure_config.hpp"

namespace mscoupon {
namespace {

// Named outputs of a GMM fit, so `low_from`/`high_from` can pick any of them
// ("mu_1" is the fitted mean of the low component, "median_2" the empirical
// median of the pixels hard-assigned to the high component, ...).
std::unordered_map<std::string, double> gmm_landmarks(const GmmResult& r) {
  std::unordered_map<std::string, double> out;
  for (std::size_t i = 0; i < r.components.size(); ++i) {
    const std::string n = std::to_string(i + 1);  // 1-based, as in the CSVs
    const GmmComponent& c = r.components[i];
    out["mu_" + n] = c.mean;
    out["sigma_" + n] = c.sigma;
    out["weight_" + n] = c.weight;
    out["hard_mean_" + n] = c.hard_mean;
    out["median_" + n] = c.median;
    out["mode_" + n] = c.mode;
  }
  return out;
}

std::unordered_map<std::string, double> histogram_landmarks(const HistogramResult& r) {
  std::unordered_map<std::string, double> out{
      {"peak_low", r.peak_low}, {"peak_high", r.peak_high},
      {"peak_1", r.peak_low},   {"peak_2", r.peak_high},
      {"min", r.min_value},     {"max", r.max_value},
      {"hist_lo", r.hist_lo},   {"hist_hi", r.hist_hi},
  };
  const std::vector<std::string>& names = default_percentile_names();
  for (std::size_t i = 0; i < names.size() && i < r.percentiles.size(); ++i) {
    out[names[i]] = r.percentiles[i];
  }
  return out;
}

void add_region_landmarks(std::unordered_map<std::string, double>& out, const std::string& prefix,
                          const RegionStats& s) {
  out[prefix + "_mean"] = s.mean;
  out[prefix + "_min"] = s.min_value;
  out[prefix + "_max"] = s.max_value;
  out[prefix + "_std"] = s.std_dev;
  const std::vector<std::string>& names = default_percentile_names();
  for (std::size_t i = 0; i < names.size() && i < s.percentiles.size(); ++i) {
    out[prefix + "_" + names[i]] = s.percentiles[i];
    // "air_median" is friendlier than "air_p50_0" and is the default landmark.
    if (names[i] == "p50_0") out[prefix + "_median"] = s.percentiles[i];
  }
}

double pick(const std::unordered_map<std::string, double>& table, const std::string& key,
            const char* what) {
  const auto it = table.find(key);
  if (it == table.end())
    throw std::runtime_error("normalize: unknown " + std::string(what) + " '" + key + "'");
  return it->second;
}

// Run the configured measure and pull the two named landmarks out of it. May
// throw; measure_two_point() is the wrapper that applies the fallback policy.
TwoPoint measure_landmarks(const Image2D& image, const NormalizeConfig& cfg) {
  TwoPoint tp;

  switch (cfg.method) {
    case NormalizeMethod::Manual: {
      tp.low = *cfg.manual_low;
      tp.high = *cfg.manual_high;
      break;
    }
    case NormalizeMethod::Gmm: {
      GmmOptions opts = parse_gmm_options(cfg.params);
      // The hard-assignment statistics are only computed on request, so ask for
      // them whenever a landmark names one.
      if (cfg.low_from.rfind("mu_", 0) != 0 || cfg.high_from.rfind("mu_", 0) != 0)
        opts.compute_hard_stats = true;
      const auto table = gmm_landmarks(fit_gmm(image, opts));
      tp.low = pick(table, cfg.low_from, "gmm landmark");
      tp.high = pick(table, cfg.high_from, "gmm landmark");
      break;
    }
    case NormalizeMethod::Histogram: {
      const HistogramOptions opts = parse_histogram_options(cfg.params);
      const auto table = histogram_landmarks(measure_histogram(image, opts));
      tp.low = pick(table, cfg.low_from, "histogram landmark");
      tp.high = pick(table, cfg.high_from, "histogram landmark");
      break;
    }
    case NormalizeMethod::Regions: {
      if (!cfg.params.contains("air") || !cfg.params.contains("metal"))
        throw std::runtime_error("normalize: method 'regions' requires 'air' and 'metal' rects");
      const RegionOptions opts = parse_region_options(cfg.params);
      std::unordered_map<std::string, double> table;
      add_region_landmarks(table, "air",
                           measure_region(image, parse_rect_json(cfg.params.at("air")), opts));
      add_region_landmarks(table, "metal",
                           measure_region(image, parse_rect_json(cfg.params.at("metal")), opts));
      tp.low = pick(table, cfg.low_from, "region landmark");
      tp.high = pick(table, cfg.high_from, "region landmark");
      break;
    }
  }
  return tp;
}

}  // namespace

NormalizeConfig parse_normalize_config(const nlohmann::json& params) {
  NormalizeConfig cfg;

  const auto method = params.value("method", std::string("gmm"));
  if (method == "gmm") {
    cfg.method = NormalizeMethod::Gmm;
    cfg.low_from = "mu_1";
    cfg.high_from = "mu_2";
  } else if (method == "histogram") {
    cfg.method = NormalizeMethod::Histogram;
    cfg.low_from = "peak_low";
    cfg.high_from = "peak_high";
  } else if (method == "regions") {
    cfg.method = NormalizeMethod::Regions;
    cfg.low_from = "air_median";
    cfg.high_from = "metal_median";
  } else if (method == "manual") {
    cfg.method = NormalizeMethod::Manual;
  } else {
    throw std::runtime_error(
        "normalize: method must be 'gmm', 'histogram', 'regions' or 'manual'");
  }

  cfg.low_from = params.value("low_from", cfg.low_from);
  cfg.high_from = params.value("high_from", cfg.high_from);
  if (params.contains("low") && !params.at("low").is_null())
    cfg.manual_low = params.at("low").get<double>();
  if (params.contains("high") && !params.at("high").is_null())
    cfg.manual_high = params.at("high").get<double>();
  cfg.clamp = params.value("clamp", cfg.clamp);
  cfg.params = params;

  if (cfg.method == NormalizeMethod::Manual && !(cfg.manual_low && cfg.manual_high))
    throw std::runtime_error("normalize: method 'manual' requires both 'low' and 'high'");

  return cfg;
}

TwoPoint measure_two_point(const Image2D& image, const NormalizeConfig& cfg) {
  // A slice can legitimately defeat a measure -- an all-background slice at the
  // end of a stack has no two populations to find, and a collapsed fit yields
  // high <= low. Both fall back to the caller's pair when one was supplied, so
  // a single bad slice does not abort the stack.
  const auto fallback = [&](const std::string& why) -> TwoPoint {
    if (cfg.manual_low && cfg.manual_high) {
      const TwoPoint tp{*cfg.manual_low, *cfg.manual_high};
      if (tp.valid()) return tp;
    }
    throw std::runtime_error("normalize: " + why +
                             " (supply 'low' and 'high' for a fallback pair)");
  };

  TwoPoint tp;
  try {
    tp = measure_landmarks(image, cfg);
  } catch (const std::exception& e) {
    return fallback(std::string("measure failed: ") + e.what());
  }
  if (!tp.valid()) return fallback("measured landmarks are degenerate (high <= low)");
  return tp;
}

void apply_two_point(Image2D& image, const TwoPoint& tp, bool clamp) {
  if (!tp.valid()) return;

  // Precompute the reciprocal: one multiply per pixel instead of a divide.
  const auto offset = static_cast<float>(tp.low);
  const auto inv_scale = static_cast<float>(1.0 / tp.scale());
  for (float& v : image.pixels) {
    v = (v - offset) * inv_scale;
    if (clamp) v = std::clamp(v, 0.0f, 1.0f);
  }
}

}  // namespace mscoupon
