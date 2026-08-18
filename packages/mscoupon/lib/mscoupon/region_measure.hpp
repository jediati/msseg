#pragma once

// Fixed-rectangle intensity measurement: a C++ port of measure_2_regions.py.
//
// The manual counterpart to the GMM and histogram measures -- instead of
// inferring the two populations, the caller names two rectangles known to sit
// in air and in metal, and each is reduced to basic statistics plus the shared
// percentile ladder.
//
// NOTE the no-data policy differs from the other two measures on purpose. These
// are explicitly chosen physical regions, so every pixel inside them is RETAINED
// by default (measure_2_regions.py: "Exact zero values are INCLUDED by
// default"). omit_value therefore defaults to nullopt here and to 0 elsewhere.

#include <cstddef>
#include <optional>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "mscoupon/measure_util.hpp"
#include "mscoupon/types.hpp"

namespace mscoupon {

// A half-open pixel rectangle: rows [row0, row1) x columns [col0, col1).
// Matches the numpy slice image[row0:row1, col0:col1] -- rows are Y, cols are X.
struct Rect {
  int row0 = 0;
  int row1 = 0;
  int col0 = 0;
  int col1 = 0;
};

struct RegionOptions {
  // Retain everything inside the rectangle by default; see the note above.
  std::optional<double> omit_value;
  bool omit_nonfinite = true;
};

struct RegionStats {
  std::int64_t n_pixels = 0;
  double min_value = 0.0;
  double max_value = 0.0;
  double mean = 0.0;
  double std_dev = 0.0;
  // Parallel to default_percentiles().
  std::vector<double> percentiles;
};

// Parse "START:END" into a Rect edge pair. Throws std::runtime_error on a
// malformed or empty range, matching the Python's argparse rejection.
Rect parse_rect(const std::string& rows, const std::string& cols);

// Reduce an already-collected sample. Throws when empty.
RegionStats measure_values(std::vector<double>& values);

// Extract `rect` from a raw single-channel buffer and measure it. Throws if the
// rectangle falls outside the image.
template <typename T>
RegionStats measure_region(const T* pixels, int width, int height, const Rect& rect,
                           const RegionOptions& opts) {
  if (rect.row1 <= rect.row0 || rect.col1 <= rect.col0)
    throw std::runtime_error("region: END must be greater than START");
  if (rect.row0 < 0 || rect.col0 < 0)
    throw std::runtime_error("region: coordinates must be >= 0");
  if (rect.row1 > height)
    throw std::runtime_error("region: row range exceeds image height");
  if (rect.col1 > width)
    throw std::runtime_error("region: column range exceeds image width");

  std::vector<double> values;
  values.reserve(static_cast<std::size_t>(rect.row1 - rect.row0) *
                 static_cast<std::size_t>(rect.col1 - rect.col0));
  for (int y = rect.row0; y < rect.row1; ++y) {
    const T* row = pixels + static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
    std::vector<double> kept = collect_valid_pixels(row + rect.col0,
                                                    static_cast<std::size_t>(rect.col1 - rect.col0),
                                                    opts.omit_value, opts.omit_nonfinite);
    values.insert(values.end(), kept.begin(), kept.end());
  }
  return measure_values(values);
}

// Convenience overload for the package's slice type.
RegionStats measure_region(const Image2D& image, const Rect& rect, const RegionOptions& opts);

}  // namespace mscoupon
