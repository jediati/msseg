#include "mscoupon/measure_config.hpp"

#include <optional>
#include <stdexcept>
#include <string>

namespace mscoupon {
namespace {

// Return the named block if present, else treat the object itself as the block.
const nlohmann::json& block(const nlohmann::json& cfg, const char* key) {
  return cfg.contains(key) ? cfg.at(key) : cfg;
}

// Read the no-data policy: `omit_value` (a number, or null to keep every pixel)
// with the older boolean `omit_zeros` still honoured. A stack's padding is not
// always 0 -- some reconstructions pad with an arbitrary constant -- so the value
// is configurable, but a config written before that was true keeps working:
// omit_zeros=true means "the sentinel is 0", false means "keep everything".
// An explicit omit_value wins over omit_zeros.
void parse_omit_policy(const nlohmann::json& node, std::optional<double>& omit_value) {
  if (node.contains("omit_zeros") && !node.at("omit_zeros").is_null()) {
    omit_value = node.at("omit_zeros").get<bool>() ? std::optional<double>(0.0)
                                                    : std::optional<double>();
  }
  if (node.contains("omit_value")) {
    const auto& v = node.at("omit_value");
    omit_value = v.is_null() ? std::optional<double>() : std::optional<double>(v.get<double>());
  }
}

}  // namespace

GmmOptions parse_gmm_options(const nlohmann::json& cfg) {
  const nlohmann::json& g = block(cfg, "gmm");

  // `measure` is the default because it is the preset built for measuring
  // landmarks on raw reconstruction intensities, which sit near 1e-4. It carries
  // reg_covar=1e-12; `two_gaussian` keeps sklearn's 1e-6 default, which on such
  // data is the same size as the whole distribution's variance -- every
  // component's variance floors at reg_covar, the two Gaussians become
  // identical, and EM collapses onto the global mean in a handful of iterations
  // (silently: it reports converged=true). This default also matches the Python
  // library's (normalize/gmm.py PRESET_MEASURE), so the GUI, the CLI and the
  // scripts agree unless a caller asks for `two_gaussian` explicitly.
  const auto preset = g.value("preset", std::string("measure"));
  GmmOptions o;
  if (preset == "two_gaussian") {
    o = gmm_options_two_gaussian();
  } else if (preset == "measure") {
    o = gmm_options_measure();
  } else {
    throw std::runtime_error("gmm: preset must be 'two_gaussian' or 'measure'");
  }

  o.n_components = g.value("n_components", o.n_components);
  o.downsample_factor = g.value("downsample_factor", o.downsample_factor);
  parse_omit_policy(g, o.omit_value);
  o.omit_nonfinite = g.value("omit_nonfinite", o.omit_nonfinite);
  o.trim_percent = g.value("trim_percent", o.trim_percent);
  if (g.contains("init")) {
    const auto init = g["init"].get<std::string>();
    if (init == "kmeans") {
      o.init = GmmInit::KMeans;
    } else if (init == "quantile") {
      o.init = GmmInit::Quantile;
    } else {
      throw std::runtime_error("gmm: init must be 'kmeans' or 'quantile'");
    }
  }
  o.n_init = g.value("n_init", o.n_init);
  o.max_iter = g.value("max_iter", o.max_iter);
  o.tol = g.value("tol", o.tol);
  o.reg_covar = g.value("reg_covar", o.reg_covar);
  o.seed = g.value("seed", o.seed);
  o.compute_hard_stats = g.value("compute_hard_stats", o.compute_hard_stats);
  o.mode_bins = g.value("mode_bins", o.mode_bins);
  return o;
}

HistogramOptions parse_histogram_options(const nlohmann::json& cfg) {
  const nlohmann::json& h = block(cfg, "histogram");

  HistogramOptions o;
  o.downsample_factor = h.value("downsample_factor", o.downsample_factor);
  parse_omit_policy(h, o.omit_value);
  o.omit_nonfinite = h.value("omit_nonfinite", o.omit_nonfinite);
  o.bins = h.value("bins", o.bins);
  o.smooth_width = h.value("smooth_width", o.smooth_width);
  o.peak_window = h.value("peak_window", o.peak_window);
  o.min_peak_distance = h.value("min_peak_distance", o.min_peak_distance);
  o.hist_lo_percentile = h.value("hist_lo_percentile", o.hist_lo_percentile);
  o.hist_hi_percentile = h.value("hist_hi_percentile", o.hist_hi_percentile);
  o.seed = h.value("seed", o.seed);
  return o;
}

RegionOptions parse_region_options(const nlohmann::json& cfg) {
  // Unlike the other two, these options are read from the TOP level: the
  // "regions" block is the name -> rectangle map, so looking inside it would
  // silently ignore the no-data policy (and mistake it for a malformed rectangle).
  RegionOptions o;
  parse_omit_policy(cfg, o.omit_value);
  o.omit_nonfinite = cfg.value("omit_nonfinite", o.omit_nonfinite);
  return o;
}

Rect parse_rect_json(const nlohmann::json& node) {
  if (!node.is_object() || !node.contains("rows") || !node.contains("cols"))
    throw std::runtime_error("region: expected an object with 'rows' and 'cols'");
  return parse_rect(node.at("rows").get<std::string>(), node.at("cols").get<std::string>());
}

}  // namespace mscoupon
