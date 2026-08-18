#include "mscoupon/measure_util.hpp"

#include <algorithm>

namespace mscoupon {

double percentile_linear(std::vector<double>& v, double q) {
  const std::size_t n = v.size();
  if (n == 0) return 0.0;
  if (n == 1) return v[0];

  const double h = static_cast<double>(n - 1) * (q / 100.0);
  std::size_t lo = static_cast<std::size_t>(std::floor(h));
  std::size_t hi = static_cast<std::size_t>(std::ceil(h));
  if (hi >= n) hi = n - 1;
  if (lo > hi) lo = hi;

  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(lo), v.end());
  const double a = v[lo];
  double b = a;
  if (hi != lo) {
    // Everything at or past lo+1 is >= a after the partition above, so the
    // hi-th order statistic can be found within that suffix.
    std::nth_element(v.begin() + static_cast<std::ptrdiff_t>(lo) + 1,
                     v.begin() + static_cast<std::ptrdiff_t>(hi), v.end());
    b = v[hi];
  }
  return a + (b - a) * (h - static_cast<double>(lo));
}

double parabolic_offset(double y1, double y2, double y3) {
  const double denom = y1 - 2.0 * y2 + y3;
  if (denom == 0.0) return 0.0;
  return std::clamp(0.5 * (y1 - y3) / denom, -1.0, 1.0);
}

void subsample_inplace(std::vector<double>& v, int factor, std::mt19937_64& rng) {
  if (factor <= 1 || v.empty()) return;

  std::size_t n_sample = std::max<std::size_t>(10, v.size() / static_cast<std::size_t>(factor));
  n_sample = std::min(n_sample, v.size());
  if (n_sample >= v.size()) return;

  for (std::size_t i = 0; i < n_sample; ++i) {
    std::uniform_int_distribution<std::size_t> pick(i, v.size() - 1);
    std::swap(v[i], v[pick(rng)]);
  }
  v.resize(n_sample);
}

}  // namespace mscoupon
