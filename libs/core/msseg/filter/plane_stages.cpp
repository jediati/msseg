#include "msseg/filter/plane_stages.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "msseg/filter/color_stage.hpp"

namespace msseg {
namespace {

const char* const kAdaptModes[] = {"select", "project", "reduce"};
const char* const kReduceHow[] = {"mean", "sum", "max", "min", "norm", "first"};
const char* const kPromoted[] = {"hsv", "optical_density", "chgradmag", "dizenzo", "structure"};
const char* const kStainPresets[] = {"he", "hdab", "hed"};

// Ruifrok & Johnston's unit OD vectors, the same numbers scikit-image carries as
// rgb_from_hed. Normalized on use, so these stay as published.
constexpr double kH[3] = {0.65, 0.70, 0.29};
constexpr double kE[3] = {0.07, 0.99, 0.11};
constexpr double kDAB[3] = {0.27, 0.57, 0.78};

bool fail(std::string* why, const std::string& message) {
  if (why != nullptr) *why = message;
  return false;
}

int fail_int(std::string* why, const std::string& message) {
  fail(why, message);
  return -1;
}

std::string mode_of(const FilterParams& stage) { return stage.params.value("mode", "select"); }

bool has(const nlohmann::json& p, const char* key) {
  return p.contains(key) && !p.at(key).is_null();
}

std::vector<double> get_doubles(const nlohmann::json& p, const char* key) {
  if (!has(p, key)) return {};
  const auto& v = p.at(key);
  if (!v.is_array()) throw std::runtime_error(std::string("adapt: '") + key + "' must be an array.");
  return v.get<std::vector<double>>();
}

// `matrix` as k rows of C, flattened row-major. Rejects a ragged matrix here so
// the planner and the applier agree on k.
bool matrix_rows(const nlohmann::json& p, std::size_t channels, std::vector<double>* out,
                 std::size_t* rows, std::string* why) {
  const auto& m = p.at("matrix");
  if (!m.is_array() || m.empty()) return fail(why, "adapt: project.matrix must be a non-empty array of rows.");
  out->clear();
  // A flat array of numbers is ONE row. Projecting to a single plane is the
  // common case by a wide margin -- it is what every stain contrast is -- and a
  // GUI entry holding "a, b, c" should not have to spell it [[a, b, c]].
  if (m.front().is_number()) {
    if (m.size() != channels) {
      return fail(why, "adapt: project.matrix has " + std::to_string(m.size()) +
                           " entries; the input has " + std::to_string(channels) + " plane(s).");
    }
    for (const auto& v : m) {
      if (!v.is_number()) return fail(why, "adapt: project.matrix must hold numbers.");
      out->push_back(v.get<double>());
    }
    *rows = 1;
    return true;
  }
  for (const auto& row : m) {
    if (!row.is_array()) return fail(why, "adapt: project.matrix must be an array of ROWS.");
    if (row.size() != channels) {
      return fail(why, "adapt: project.matrix row has " + std::to_string(row.size()) +
                           " entries; the input has " + std::to_string(channels) + " plane(s).");
    }
    for (const auto& v : row) {
      if (!v.is_number()) return fail(why, "adapt: project.matrix must hold numbers.");
      out->push_back(v.get<double>());
    }
  }
  *rows = m.size();
  return true;
}

void normalize3(double* v) {
  const double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (!(n > 0.0)) throw std::runtime_error("stain_deconvolution: a stain vector is zero.");
  v[0] /= n;
  v[1] /= n;
  v[2] /= n;
}

// Invert a 3x3 held column-major (column j at m[3*j .. 3*j+2]). The stain matrix
// is small and fixed, so a cofactor inverse is exact enough and has no
// dependency; a singular matrix means two stains were given as parallel.
std::vector<double> invert3(const std::vector<double>& m) {
  const auto a = [&](int r, int c) { return m[static_cast<std::size_t>(3 * c + r)]; };
  const double det = a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) -
                     a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) +
                     a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
  if (std::abs(det) < 1e-12) {
    throw std::runtime_error("stain_deconvolution: the stain matrix is singular (parallel stains?).");
  }
  const double inv_det = 1.0 / det;
  std::vector<double> out(9);
  // out is row-major here: row i of the inverse maps OD -> concentration i.
  out[0] = (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) * inv_det;
  out[1] = (a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2)) * inv_det;
  out[2] = (a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)) * inv_det;
  out[3] = (a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2)) * inv_det;
  out[4] = (a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0)) * inv_det;
  out[5] = (a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2)) * inv_det;
  out[6] = (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0)) * inv_det;
  out[7] = (a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1)) * inv_det;
  out[8] = (a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)) * inv_det;
  return out;
}

}  // namespace

bool is_plane_operation(const std::string& operation) {
  return operation == kAdaptOperation || operation == kStainOperation ||
         is_promoted_color_operation(operation);
}

bool is_promoted_color_operation(const std::string& operation) {
  return std::find(std::begin(kPromoted), std::end(kPromoted), operation) != std::end(kPromoted);
}

const std::vector<std::string>& promoted_color_operations() {
  static const std::vector<std::string> v(std::begin(kPromoted), std::end(kPromoted));
  return v;
}

FilterParams as_color_stage(const FilterParams& stage) {
  FilterParams out;
  out.operation = kColorOperation;
  out.params = stage.params;
  out.params["method"] = stage.operation;
  return out;
}

const std::vector<std::string>& adapt_modes() {
  static const std::vector<std::string> v(std::begin(kAdaptModes), std::end(kAdaptModes));
  return v;
}

const std::vector<std::string>& stain_presets() {
  static const std::vector<std::string> v(std::begin(kStainPresets), std::end(kStainPresets));
  return v;
}

std::vector<double> stain_matrix(const std::string& preset, std::string* why) {
  const double* s0 = kH;
  const double* s1 = nullptr;
  const double* s2 = nullptr;
  if (preset == "he") {
    s1 = kE;
  } else if (preset == "hdab") {
    s1 = kDAB;
  } else if (preset == "hed") {
    s1 = kE;
    s2 = kDAB;
  } else {
    fail(why, "stain_deconvolution: unknown preset '" + preset + "'. Available: he, hdab, hed.");
    return {};
  }
  std::vector<double> m(9);
  for (int i = 0; i < 3; ++i) {
    m[static_cast<std::size_t>(i)] = s0[i];
    m[static_cast<std::size_t>(3 + i)] = s1[i];
  }
  normalize3(m.data());
  normalize3(m.data() + 3);
  if (s2 != nullptr) {
    for (int i = 0; i < 3; ++i) m[static_cast<std::size_t>(6 + i)] = s2[i];
    normalize3(m.data() + 6);
  } else {
    // Ruifrok's complement: what neither stain explains gets its own axis rather
    // than leaking into the two that matter.
    const double* a = m.data();
    const double* b = m.data() + 3;
    m[6] = a[1] * b[2] - a[2] * b[1];
    m[7] = a[2] * b[0] - a[0] * b[2];
    m[8] = a[0] * b[1] - a[1] * b[0];
    normalize3(m.data() + 6);
  }
  return m;
}

int plane_stage_output_channels(const FilterParams& stage, std::size_t in_channels,
                                std::string* why) {
  if (in_channels == 0) return fail_int(why, "the stage was handed no input planes.");
  if (stage.operation == kAdaptOperation) {
    const std::string mode = mode_of(stage);
    const auto& p = stage.params;
    if (mode == "select") {
      const auto ch = get_doubles(p, "channels");
      if (ch.empty()) return fail_int(why, "adapt: select.channels must name at least one plane.");
      for (const double c : ch) {
        if (c < 0 || static_cast<std::size_t>(c) >= in_channels) {
          return fail_int(why, "adapt: select.channels names plane " + std::to_string(static_cast<int>(c)) +
                                   ", out of range for " + std::to_string(in_channels) + " plane(s).");
        }
      }
      return static_cast<int>(ch.size());
    }
    if (mode == "project") {
      if (has(p, "preset")) {
        const std::string preset = p.value("preset", "");
        if (preset != "luminance") {
          return fail_int(why, "adapt: project.preset must be 'luminance' (got '" + preset + "').");
        }
        if (in_channels < 3) {
          return fail_int(why, "adapt: project.preset 'luminance' needs at least 3 planes.");
        }
        return 1;
      }
      if (!has(p, "matrix")) return fail_int(why, "adapt: project needs a matrix or a preset.");
      std::vector<double> flat;
      std::size_t rows = 0;
      if (!matrix_rows(p, in_channels, &flat, &rows, why)) return -1;
      return static_cast<int>(rows);
    }
    if (mode == "reduce") {
      const std::string how = p.value("how", "mean");
      if (std::find(std::begin(kReduceHow), std::end(kReduceHow), how) == std::end(kReduceHow)) {
        return fail_int(why, "adapt: reduce.how must be mean|sum|max|min|norm|first (got '" + how + "').");
      }
      if (how == "mean" && has(p, "weights") && get_doubles(p, "weights").size() != in_channels) {
        return fail_int(why, "adapt: reduce.weights needs one entry per plane.");
      }
      return 1;
    }
    return fail_int(why, "adapt: unknown mode '" + mode + "'. Available: select, project, reduce.");
  }
  if (stage.operation == kStainOperation) {
    if (in_channels != 3) {
      return fail_int(why, "stain_deconvolution: needs exactly 3 planes (RGB); the input has " +
                               std::to_string(in_channels) + ".");
    }
    std::string local;
    if (stain_matrix(stage.params.value("preset", "he"), &local).empty()) return fail_int(why, local);
    return 3;
  }
  if (is_promoted_color_operation(stage.operation)) {
    const FilterParams as_color = as_color_stage(stage);
    if (!color_stage_accepts(as_color, in_channels, why)) return -1;
    // `optical_density` can keep its planes; every other promoted method is a
    // reduction to one.
    return color_stage_keeps_planes(as_color) ? static_cast<int>(in_channels) : 1;
  }
  return fail_int(why, "'" + stage.operation + "' is not a plane stage.");
}

diffg::MultiImage<float> apply_plane_stage(diffg::MultiImageView<const float> planes,
                                           const FilterParams& stage) {
  std::string why;
  const int out_c = plane_stage_output_channels(stage, planes.channels(), &why);
  if (out_c < 0) throw std::runtime_error(why);

  const std::size_t C = planes.channels();
  const std::size_t n = planes.channel_stride();
  const std::size_t K = static_cast<std::size_t>(out_c);
  std::vector<const float*> in(C);
  for (std::size_t c = 0; c < C; ++c) in[c] = planes.channel_data(c);
  diffg::MultiImage<float> out(planes.dims(), K, planes.spacing());

  if (is_promoted_color_operation(stage.operation)) {
    // One implementation per method: the promoted operation is the colour stage
    // with its method filled in, run on whatever stack reaches it.
    return apply_color_stage_multi(planes, as_color_stage(stage));
  }

  if (stage.operation == kAdaptOperation) {
    const std::string mode = mode_of(stage);
    if (mode == "reduce") {
      const std::string how = stage.params.value("how", "mean");
      const auto w = get_doubles(stage.params, "weights");
      float* o = out.channel_data(0);
      for (std::size_t i = 0; i < n; ++i) {
        if (how == "max" || how == "min") {
          float v = in[0][i];
          for (std::size_t c = 1; c < C; ++c) v = how == "max" ? std::max(v, in[c][i]) : std::min(v, in[c][i]);
          o[i] = v;
        } else if (how == "first") {
          o[i] = in[0][i];
        } else if (how == "norm") {
          double acc = 0.0;
          for (std::size_t c = 0; c < C; ++c) acc += static_cast<double>(in[c][i]) * in[c][i];
          o[i] = static_cast<float>(std::sqrt(acc));
        } else {  // mean (optionally weighted) or sum
          double acc = 0.0;
          for (std::size_t c = 0; c < C; ++c) {
            acc += (w.empty() ? 1.0 : w[c]) * static_cast<double>(in[c][i]);
          }
          o[i] = static_cast<float>(how == "sum" ? acc : acc / static_cast<double>(C));
        }
      }
      return out;
    }
    if (mode == "select") {
      const auto ch = get_doubles(stage.params, "channels");
      for (std::size_t k = 0; k < K; ++k) {
        const std::size_t src = static_cast<std::size_t>(ch[k]);
        std::copy(in[src], in[src] + n, out.channel_data(k));
      }
      return out;
    }
    // project: rows of the matrix, or the Rec.709 row.
    std::vector<double> flat;
    if (has(stage.params, "preset")) {
      flat = {0.2126, 0.7152, 0.0722};
      flat.resize(C, 0.0);
    } else {
      std::size_t rows = 0;
      matrix_rows(stage.params, C, &flat, &rows, &why);
    }
    for (std::size_t k = 0; k < K; ++k) {
      float* o = out.channel_data(k);
      const double* row = flat.data() + k * C;
      for (std::size_t i = 0; i < n; ++i) {
        double acc = 0.0;
        for (std::size_t c = 0; c < C; ++c) acc += row[c] * in[c][i];
        o[i] = static_cast<float>(acc);
      }
    }
    return out;
  }

  // stain_deconvolution: optical density, then c = inv(M) * od.
  const auto& p = stage.params;
  const std::vector<double> m = stain_matrix(p.value("preset", "he"), &why);
  const std::vector<double> minv = invert3(m);
  const double eps = p.contains("eps") ? p.at("eps").get<double>() : 1e-3;
  const bool do_od = p.value("od", true);

  std::vector<double> i0(C, 1.0);
  if (do_od) {
    if (!has(p, "i0") || p.at("i0").is_string()) {
      for (std::size_t c = 0; c < C; ++c) {
        float mx = std::numeric_limits<float>::lowest();
        for (std::size_t i = 0; i < n; ++i) mx = std::max(mx, in[c][i]);
        if (!(mx > 0.0f)) {
          throw std::runtime_error("stain_deconvolution: i0=\"max\" found no positive value in plane " +
                                   std::to_string(c) + "; set i0 explicitly.");
        }
        i0[c] = mx;
      }
    } else if (p.at("i0").is_array()) {
      i0 = p.at("i0").get<std::vector<double>>();
      if (i0.size() != C) throw std::runtime_error("stain_deconvolution: i0 needs one entry per plane.");
    } else {
      std::fill(i0.begin(), i0.end(), p.at("i0").get<double>());
    }
    for (std::size_t c = 0; c < C; ++c) {
      if (!(i0[c] > 0.0)) throw std::runtime_error("stain_deconvolution: i0 must be > 0.");
    }
  }

  const double inv_ln10 = 1.0 / std::log(10.0);
  float* o0 = out.channel_data(0);
  float* o1 = out.channel_data(1);
  float* o2 = out.channel_data(2);
  for (std::size_t i = 0; i < n; ++i) {
    double od[3];
    for (std::size_t c = 0; c < 3; ++c) {
      if (do_od) {
        const double v = std::max(static_cast<double>(in[c][i]), eps);
        od[c] = -std::log(v / i0[c]) * inv_ln10;
      } else {
        od[c] = in[c][i];
      }
    }
    o0[i] = static_cast<float>(minv[0] * od[0] + minv[1] * od[1] + minv[2] * od[2]);
    o1[i] = static_cast<float>(minv[3] * od[0] + minv[4] * od[1] + minv[5] * od[2]);
    o2[i] = static_cast<float>(minv[6] * od[0] + minv[7] * od[1] + minv[8] * od[2]);
  }
  return out;
}

}  // namespace msseg
