#include "mscoupon/cc_stage.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace mscoupon {
namespace {

// Base-channel value at the seeding extremum: the single pixel for radius 0, or
// the mean over the clamped (2r+1)^2 window. Mirrors the core's sampler so the
// 3D extremum reads the same way the 2D one does.
float sample_base(const Image2D& base, int width, int height, int px, int py, int radius) {
  const auto at = [&](int x, int y) {
    return base.pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                       static_cast<std::size_t>(x)];
  };
  if (radius <= 0) return at(px, py);
  const int x0 = std::max(0, px - radius), x1 = std::min(width - 1, px + radius);
  const int y0 = std::max(0, py - radius), y1 = std::min(height - 1, py + radius);
  double sum = 0.0;
  int n = 0;
  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) { sum += at(x, y); ++n; }
  }
  return n > 0 ? static_cast<float>(sum / n) : at(px, py);
}

}  // namespace

bool pixel_predicate(float value, const std::string& op, double threshold) {
  const double v = static_cast<double>(value);
  if (op == "lt") return v < threshold;
  if (op == "le") return v <= threshold;
  if (op == "gt") return v > threshold;
  if (op == "ge") return v >= threshold;
  return false;
}

int label_selected_components(const std::vector<int>& labels, int width, int height,
                              const Image2D& base, const Image2D& filtered,
                              const std::unordered_set<int>& keep_ids,
                              const std::vector<PixelFilter>& pixel_filters,
                              int connectivity, bool ascending,
                              const msseg::StatsSpec& stats,
                              std::vector<int>& cc_out,
                              std::vector<CcNodeStat>& stats_out) {
  const std::size_t n_pix = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

  // 1. keep mask = selected region AND passes every pixel-filter rule.
  std::vector<uint8_t> keep(n_pix, 0);
  for (std::size_t i = 0; i < n_pix; ++i) {
    if (labels[i] < 0 || keep_ids.count(labels[i]) == 0) continue;
    bool ok = true;
    for (const auto& r : pixel_filters) {
      const float val = (r.channel == "filtered") ? filtered.pixels[i] : base.pixels[i];
      const bool passed = pixel_predicate(val, r.op, r.value);
      if (r.mode == "keep" ? !passed : passed) { ok = false; break; }
    }
    keep[i] = ok ? 1 : 0;
  }

  // 2. in-plane connected components (iterative flood fill). 4-conn for 6, else 8.
  const bool diag = (connectivity != 6);
  cc_out.assign(n_pix, -1);
  std::vector<int> stack;
  int n_comp = 0;
  for (std::size_t seed = 0; seed < n_pix; ++seed) {
    if (!keep[seed] || cc_out[seed] >= 0) continue;
    const int comp = n_comp++;
    stack.clear();
    stack.push_back(static_cast<int>(seed));
    cc_out[seed] = comp;
    while (!stack.empty()) {
      const int q = stack.back();
      stack.pop_back();
      const int x = q % width;
      const int y = q / width;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0) continue;
          if (!diag && dx != 0 && dy != 0) continue;
          const int nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const int np = ny * width + nx;
          if (keep[np] && cc_out[np] < 0) {
            cc_out[np] = comp;
            stack.push_back(np);
          }
        }
      }
    }
  }

  // 3. per-component statistics from the actual pixel sets.
  const msseg::StatsSpec& spec = stats;
  const bool want_base_sums = spec.base_channel && (spec.mean || spec.std);
  const bool want_filt_sums = spec.filtered_channel && (spec.mean || spec.std);
  const bool want_filt_extent = spec.needs_filtered_extent();

  stats_out.assign(static_cast<std::size_t>(n_comp), CcNodeStat{});
  for (auto& s : stats_out) {
    s.base_min = std::numeric_limits<float>::max();
    s.base_max = std::numeric_limits<float>::lowest();
    if (want_filt_extent) {
      s.filt_min = std::numeric_limits<float>::max();
      s.filt_max = std::numeric_limits<float>::lowest();
    }
    s.min_x = width; s.min_y = height; s.max_x = -1; s.max_y = -1;
  }
  // Pixel index attaining the seeding extremum, per component.
  std::vector<std::size_t> arg_ext(static_cast<std::size_t>(n_comp), 0);
  for (std::size_t i = 0; i < n_pix; ++i) {
    const int c = cc_out[i];
    if (c < 0) continue;
    CcNodeStat& s = stats_out[static_cast<std::size_t>(c)];
    const float b = base.pixels[i];
    const int x = static_cast<int>(i) % width;
    const int y = static_cast<int>(i) / width;
    s.area += 1;
    if (want_base_sums) { s.base_sum += b; s.base_sumsq += static_cast<double>(b) * b; }
    s.base_min = std::min(s.base_min, b); s.base_max = std::max(s.base_max, b);
    if (want_filt_sums || want_filt_extent) {
      const float f = filtered.pixels[i];
      if (want_filt_sums) { s.filt_sum += f; s.filt_sumsq += static_cast<double>(f) * f; }
      if (want_filt_extent) {
        if (ascending) {
          if (f < s.filt_min) { s.filt_min = f; arg_ext[static_cast<std::size_t>(c)] = i; }
          s.filt_max = std::max(s.filt_max, f);
        } else {
          if (f > s.filt_max) { s.filt_max = f; arg_ext[static_cast<std::size_t>(c)] = i; }
          s.filt_min = std::min(s.filt_min, f);
        }
      }
    }
    s.min_x = std::min(s.min_x, x); s.max_x = std::max(s.max_x, x);
    s.min_y = std::min(s.min_y, y); s.max_y = std::max(s.max_y, y);
  }

  if (spec.extremum) {
    for (std::size_t c = 0; c < static_cast<std::size_t>(n_comp); ++c) {
      CcNodeStat& s = stats_out[c];
      if (s.area == 0) continue;
      const std::size_t i = arg_ext[c];
      const int x = static_cast<int>(i) % width;
      const int y = static_cast<int>(i) / width;
      s.ext_x = static_cast<float>(x);
      s.ext_y = static_cast<float>(y);
      s.ext_filtered = ascending ? s.filt_min : s.filt_max;
      s.ext_base = sample_base(base, width, height, x, y, spec.extremum_sample_radius);
    }
  }
  return n_comp;
}

}  // namespace mscoupon
