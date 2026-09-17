#include "msseg/filter/color_stage.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "diffg/filter_bank.hpp"
#include "diffg/structure.hpp"
#include "msseg/filter/chain_plan.hpp"

namespace msseg {
namespace {

const char* const kMethods[] = {"pick", "luminance", "weighted", "mean", "max", "min", "hsv",
                                "optical_density", "chgradmag", "dizenzo", "structure"};

std::string method_of(const FilterParams& stage) { return stage.params.value("method", "luminance"); }

double get_double(const nlohmann::json& params, const char* key, double default_value) {
  if (!params.contains(key)) return default_value;
  return params.at(key).get<double>();
}

int get_int(const nlohmann::json& params, const char* key, int default_value) {
  if (!params.contains(key)) return default_value;
  return params.at(key).get<int>();
}

bool has(const nlohmann::json& params, const char* key) {
  return params.contains(key) && !params.at(key).is_null();
}

std::vector<double> get_doubles(const nlohmann::json& params, const char* key) {
  if (!has(params, key)) return {};
  const auto& v = params.at(key);
  if (!v.is_array()) throw std::runtime_error(std::string("color: '") + key + "' must be an array of numbers.");
  return v.get<std::vector<double>>();
}

bool fail(std::string* why, const std::string& message) {
  if (why != nullptr) *why = message;
  return false;
}

std::string method_list() {
  std::string out;
  for (const auto& m : color_methods()) {
    if (!out.empty()) out += ", ";
    out += m;
  }
  return out;
}

bool wants_largest(const nlohmann::json& params) { return params.value("eigen", "largest") == "largest"; }

diffg::ExecutionOptions exec_of(const FilterParams& stage) {
  diffg::ExecutionOptions exec{};
  exec.threads = std::max(1, get_int(stage.params, "threads", 1));
  return exec;
}

}  // namespace

const std::vector<std::string>& color_methods() {
  static const std::vector<std::string> methods(std::begin(kMethods), std::end(kMethods));
  return methods;
}

bool is_color_method(const std::string& method) {
  return std::find(std::begin(kMethods), std::end(kMethods), method) != std::end(kMethods);
}

bool color_stage_accepts(const FilterParams& stage, std::size_t channels, std::string* why) {
  const std::string method = method_of(stage);
  if (!is_color_method(method)) {
    return fail(why, "color: unknown method '" + method + "'. Available: " + method_list() + ".");
  }
  if (channels == 0) return fail(why, "color: the input has no planes.");
  const auto& p = stage.params;
  const std::string n_planes = std::to_string(channels) + " plane(s)";
  if (method == "luminance" || method == "hsv") {
    if (channels < 3) {
      return fail(why, "color: '" + method + "' needs at least 3 planes (RGB); the input has " + n_planes + ".");
    }
    if (method == "hsv") {
      const std::string c = p.value("component", "value");
      if (c != "hue" && c != "saturation" && c != "value") {
        return fail(why, "color: hsv.component must be hue|saturation|value (got '" + c + "').");
      }
    }
  } else if (method == "pick") {
    const int c = get_int(p, "channel", 0);
    if (c < 0 || static_cast<std::size_t>(c) >= channels) {
      return fail(why, "color: pick.channel " + std::to_string(c) + " is out of range for " + n_planes + ".");
    }
  } else if (method == "weighted") {
    const auto w = get_doubles(p, "weights");
    if (w.size() != channels) {
      return fail(why, "color: weighted.weights has " + std::to_string(w.size()) + " entries; the input has " +
                           n_planes + ".");
    }
  } else if (method == "optical_density") {
    int selectors = 0;
    if (has(p, "stain")) {
      ++selectors;
      const auto s = get_doubles(p, "stain");
      if (s.size() != channels) {
        return fail(why, "color: optical_density.stain has " + std::to_string(s.size()) +
                             " entries; the input has " + n_planes + ".");
      }
      double norm = 0.0;
      for (double v : s) norm += v * v;
      if (!(norm > 0.0)) return fail(why, "color: optical_density.stain must not be the zero vector.");
    }
    if (has(p, "weights")) {
      ++selectors;
      if (get_doubles(p, "weights").size() != channels) {
        return fail(why, "color: optical_density.weights must have one entry per plane.");
      }
    }
    if (has(p, "channels")) {
      ++selectors;
      for (const double c : get_doubles(p, "channels")) {
        if (c < 0 || static_cast<std::size_t>(c) >= channels) {
          return fail(why, "color: optical_density.channels names a plane out of range.");
        }
      }
    }
    if (selectors > 1) {
      return fail(why, "color: optical_density takes at most one of stain / weights / channels.");
    }
    if (has(p, "i0")) {
      const auto& i0 = p.at("i0");
      if (i0.is_array() && i0.size() != channels) {
        return fail(why, "color: optical_density.i0 as an array needs one entry per plane.");
      }
      if (i0.is_string() && i0.get<std::string>() != "max") {
        return fail(why, "color: optical_density.i0 must be \"max\", a number, or a per-plane array.");
      }
      if (!i0.is_array() && !i0.is_string() && !i0.is_number()) {
        return fail(why, "color: optical_density.i0 must be \"max\", a number, or a per-plane array.");
      }
    }
  } else if (method == "dizenzo" || method == "structure") {
    const std::string e = p.value("eigen", "largest");
    if (e != "largest" && e != "smallest") {
      return fail(why, "color: " + method + ".eigen must be largest|smallest (got '" + e + "').");
    }
  }
  return true;
}

diffg::Image<float> apply_color_stage(diffg::MultiImageView<const float> planes,
                                      const FilterParams& stage) {
  std::string why;
  if (!color_stage_accepts(stage, planes.channels(), &why)) throw std::runtime_error(why);

  const std::string method = method_of(stage);
  const auto& p = stage.params;
  const std::size_t C = planes.channels();
  const std::size_t n = planes.channel_stride();
  std::vector<const float*> in(C);
  for (std::size_t c = 0; c < C; ++c) in[c] = planes.channel_data(c);

  diffg::Image<float> out(planes.dims(), planes.spacing());
  float* o = out.data();

  if (method == "pick") {
    const std::size_t c = static_cast<std::size_t>(get_int(p, "channel", 0));
    std::copy(in[c], in[c] + n, o);
    return out;
  }

  if (method == "luminance") {
    const float wr = 0.2126f, wg = 0.7152f, wb = 0.0722f;
    for (std::size_t i = 0; i < n; ++i) o[i] = wr * in[0][i] + wg * in[1][i] + wb * in[2][i];
    return out;
  }

  if (method == "weighted") {
    const auto w = get_doubles(p, "weights");
    for (std::size_t i = 0; i < n; ++i) {
      double acc = 0.0;
      for (std::size_t c = 0; c < C; ++c) acc += w[c] * in[c][i];
      o[i] = static_cast<float>(acc);
    }
    return out;
  }

  if (method == "mean") {
    const double inv = 1.0 / static_cast<double>(C);
    for (std::size_t i = 0; i < n; ++i) {
      double acc = 0.0;
      for (std::size_t c = 0; c < C; ++c) acc += in[c][i];
      o[i] = static_cast<float>(acc * inv);
    }
    return out;
  }

  if (method == "max" || method == "min") {
    const bool want_max = method == "max";
    for (std::size_t i = 0; i < n; ++i) {
      float v = in[0][i];
      for (std::size_t c = 1; c < C; ++c) v = want_max ? std::max(v, in[c][i]) : std::min(v, in[c][i]);
      o[i] = v;
    }
    return out;
  }

  if (method == "hsv") {
    const std::string component = p.value("component", "value");
    for (std::size_t i = 0; i < n; ++i) {
      const float r = in[0][i], g = in[1][i], b = in[2][i];
      const float mx = std::max(r, std::max(g, b));
      const float mn = std::min(r, std::min(g, b));
      const float d = mx - mn;
      if (component == "value") {
        o[i] = mx;
      } else if (component == "saturation") {
        o[i] = mx > 0.0f ? d / mx : 0.0f;
      } else {
        float h = 0.0f;
        if (d > 0.0f) {
          if (mx == r) h = 60.0f * std::fmod((g - b) / d, 6.0f);
          else if (mx == g) h = 60.0f * ((b - r) / d + 2.0f);
          else h = 60.0f * ((r - g) / d + 4.0f);
          if (h < 0.0f) h += 360.0f;
        }
        o[i] = h;
      }
    }
    return out;
  }

  if (method == "optical_density") {
    const double eps = get_double(p, "eps", 1e-3);
    // The white reference per plane: the slice's own per-plane maximum, one
    // constant, or one constant per plane.
    std::vector<double> i0(C, 0.0);
    if (!has(p, "i0") || p.at("i0").is_string()) {
      for (std::size_t c = 0; c < C; ++c) {
        float mx = std::numeric_limits<float>::lowest();
        for (std::size_t i = 0; i < n; ++i) mx = std::max(mx, in[c][i]);
        if (!(mx > 0.0f)) {
          throw std::runtime_error("color: optical_density i0=\"max\" found no positive value in plane " +
                                   std::to_string(c) + "; set i0 explicitly.");
        }
        i0[c] = mx;
      }
    } else if (p.at("i0").is_array()) {
      i0 = get_doubles(p, "i0");
    } else {
      std::fill(i0.begin(), i0.end(), p.at("i0").get<double>());
    }
    for (std::size_t c = 0; c < C; ++c) {
      if (!(i0[c] > 0.0)) throw std::runtime_error("color: optical_density.i0 must be > 0.");
    }
    // The projection: a unit stain vector, explicit weights, a plane subset, or
    // (the default) the total OD over every plane.
    std::vector<double> w(C, 1.0);
    if (has(p, "stain")) {
      w = get_doubles(p, "stain");
      double norm = 0.0;
      for (double v : w) norm += v * v;
      norm = std::sqrt(norm);
      for (double& v : w) v /= norm;
    } else if (has(p, "weights")) {
      w = get_doubles(p, "weights");
    } else if (has(p, "channels")) {
      std::fill(w.begin(), w.end(), 0.0);
      for (const double c : get_doubles(p, "channels")) w[static_cast<std::size_t>(c)] = 1.0;
    }
    const double inv_ln10 = 1.0 / std::log(10.0);
    for (std::size_t i = 0; i < n; ++i) {
      double acc = 0.0;
      for (std::size_t c = 0; c < C; ++c) {
        if (w[c] == 0.0) continue;
        const double v = std::max(static_cast<double>(in[c][i]), eps);
        acc += w[c] * (-std::log(v / i0[c]) * inv_ln10);
      }
      o[i] = static_cast<float>(acc);
    }
    return out;
  }

  if (method == "chgradmag" || method == "dizenzo") {
    const double sigma = get_double(p, "sigma", 1.0);
    diffg::FilterBankOptions options;
    options.execution = exec_of(stage);
    const std::vector<diffg::FilterRequest> requests{method == "chgradmag"
                                                         ? diffg::channel_gradient_magnitude_filter(sigma)
                                                         : diffg::dizenzo_filter(sigma)};
    auto result = diffg::apply_filter_bank(planes, requests, diffg::OutputShape::SeparateImages, options);
    if (result.images.empty()) throw std::runtime_error("color: the filter bank returned no channel.");
    if (method == "chgradmag") return std::move(result.images.front());
    // Di Zenzo eigenvalues come out descending: front is the colour edge
    // strength, back the smallest.
    return std::move(wants_largest(p) ? result.images.front() : result.images.back());
  }

  if (method == "structure") {
    const double smoothing = get_double(p, "smoothing_sigma", 1.0);
    const double integration = get_double(p, "integration_sigma", 2.0);
    auto result = diffg::structure_eigenvalues(planes, smoothing, integration, exec_of(stage));
    return std::move(wants_largest(p) ? result.largest : result.smallest);
  }

  throw std::runtime_error("color: unhandled method '" + method + "'.");
}

bool color_stage_keeps_planes(const FilterParams& stage) {
  return method_of(stage) == "optical_density" && stage.params.value("output", "scalar") == "planes";
}

diffg::MultiImage<float> apply_color_stage_multi(diffg::MultiImageView<const float> planes,
                                                 const FilterParams& stage) {
  if (!color_stage_keeps_planes(stage)) {
    diffg::Image<float> one = apply_color_stage(planes, stage);
    diffg::MultiImage<float> out(one.dims(), 1, one.spacing());
    std::copy(one.data(), one.data() + one.size(), out.channel_data(0));
    return out;
  }
  std::string why;
  if (!color_stage_accepts(stage, planes.channels(), &why)) throw std::runtime_error(why);

  const auto& p = stage.params;
  const std::size_t C = planes.channels();
  const std::size_t n = planes.channel_stride();
  const double eps = get_double(p, "eps", 1e-3);

  // The same white reference the projecting path uses; only the projection is
  // skipped, so `output: "planes"` followed by an adapt{project} of the stain
  // row reproduces `stain` exactly.
  std::vector<double> i0(C, 0.0);
  if (!has(p, "i0") || p.at("i0").is_string()) {
    for (std::size_t c = 0; c < C; ++c) {
      float mx = std::numeric_limits<float>::lowest();
      for (std::size_t i = 0; i < n; ++i) mx = std::max(mx, planes.channel_data(c)[i]);
      if (!(mx > 0.0f)) {
        throw std::runtime_error("color: optical_density i0=\"max\" found no positive value in plane " +
                                 std::to_string(c) + "; set i0 explicitly.");
      }
      i0[c] = mx;
    }
  } else if (p.at("i0").is_array()) {
    i0 = get_doubles(p, "i0");
  } else {
    std::fill(i0.begin(), i0.end(), p.at("i0").get<double>());
  }
  for (std::size_t c = 0; c < C; ++c) {
    if (!(i0[c] > 0.0)) throw std::runtime_error("color: optical_density.i0 must be > 0.");
  }

  const double inv_ln10 = 1.0 / std::log(10.0);
  diffg::MultiImage<float> out(planes.dims(), C, planes.spacing());
  for (std::size_t c = 0; c < C; ++c) {
    const float* in = planes.channel_data(c);
    float* o = out.channel_data(c);
    for (std::size_t i = 0; i < n; ++i) {
      o[i] = static_cast<float>(-std::log(std::max(static_cast<double>(in[i]), eps) / i0[c]) * inv_ln10);
    }
  }
  return out;
}

// The leading-colour view of a full ChainPlan. Kept because the three runners
// only ever needed those two facts, and narrowing here means plan_chain can
// grow without touching them.
ColorChainPlan plan_color_chain(const std::vector<FilterParams>& chain, std::size_t channels,
                                const std::string& default_method) {
  const ChainPlan plan = plan_chain(chain, channels, default_method);
  ColorChainPlan out;
  if (plan.color_stage >= 0) {
    const StageRecord& rec = plan.stages[static_cast<std::size_t>(plan.color_stage)];
    out.color = rec.stage;
    // A synthesized stage is not IN the caller's chain, so every config stage is
    // still ahead; an explicit one is the caller's index 0, so the scalar part
    // starts at 1.
    out.first_scalar_stage = rec.synthesized() ? 0 : 1;
  }
  return out;
}

}  // namespace msseg
