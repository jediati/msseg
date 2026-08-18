#include "mscoupon/region_measure.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace mscoupon {
namespace {

// Parse one "START:END" edge pair.
std::pair<int, int> parse_range(const std::string& text) {
  const std::size_t colon = text.find(':');
  if (colon == std::string::npos || colon == 0 || colon + 1 == text.size())
    throw std::runtime_error("region: expected START:END, got '" + text + "'");

  try {
    std::size_t used_lo = 0;
    std::size_t used_hi = 0;
    const int lo = std::stoi(text.substr(0, colon), &used_lo);
    const std::string tail = text.substr(colon + 1);
    const int hi = std::stoi(tail, &used_hi);
    if (used_lo != colon || used_hi != tail.size())
      throw std::runtime_error("region: expected integer START:END, got '" + text + "'");
    return {lo, hi};
  } catch (const std::invalid_argument&) {
    throw std::runtime_error("region: expected integer START:END, got '" + text + "'");
  } catch (const std::out_of_range&) {
    throw std::runtime_error("region: START:END out of range in '" + text + "'");
  }
}

}  // namespace

Rect parse_rect(const std::string& rows, const std::string& cols) {
  const auto [row0, row1] = parse_range(rows);
  const auto [col0, col1] = parse_range(cols);
  return Rect{row0, row1, col0, col1};
}

RegionStats measure_values(std::vector<double>& values) {
  if (values.empty()) throw std::runtime_error("region: contains no usable pixels");

  RegionStats out;
  out.n_pixels = static_cast<std::int64_t>(values.size());

  const auto mm = std::minmax_element(values.begin(), values.end());
  out.min_value = *mm.first;
  out.max_value = *mm.second;

  const auto n = static_cast<double>(values.size());
  double sum = 0.0;
  for (const double v : values) sum += v;
  out.mean = sum / n;

  // Population standard deviation, to match numpy's default ddof=0.
  double acc = 0.0;
  for (const double v : values) {
    const double d = v - out.mean;
    acc += d * d;
  }
  out.std_dev = std::sqrt(std::max(acc / n, 0.0));

  const std::vector<double>& ladder = default_percentiles();
  out.percentiles.reserve(ladder.size());
  for (const double q : ladder) out.percentiles.push_back(percentile_linear(values, q));

  return out;
}

RegionStats measure_region(const Image2D& image, const Rect& rect, const RegionOptions& opts) {
  return measure_region(image.pixels.data(), image.width, image.height, rect, opts);
}

}  // namespace mscoupon
