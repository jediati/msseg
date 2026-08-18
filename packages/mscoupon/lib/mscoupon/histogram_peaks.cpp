#include "mscoupon/histogram_peaks.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace mscoupon {
namespace {

// Centred moving average with edge padding, matching numpy's
// np.convolve(np.pad(counts, w//2, mode="edge"), ones(w)/w, mode="valid").
std::vector<double> moving_average(const std::vector<double>& counts, int width) {
  if (width <= 1) return counts;
  if (width % 2 == 0) ++width;  // prefer an odd window so it stays centred

  const auto n = static_cast<std::ptrdiff_t>(counts.size());
  const auto half = static_cast<std::ptrdiff_t>(width / 2);
  std::vector<double> out(counts.size(), 0.0);

  for (std::ptrdiff_t i = 0; i < n; ++i) {
    double acc = 0.0;
    for (std::ptrdiff_t k = -half; k <= half; ++k) {
      const std::ptrdiff_t j = std::clamp(i + k, std::ptrdiff_t{0}, n - 1);  // edge padding
      acc += counts[static_cast<std::size_t>(j)];
    }
    out[static_cast<std::size_t>(i)] = acc / static_cast<double>(width);
  }
  return out;
}

// Indices that are maxima within +/- radius bins. A plateau is recorded once,
// at its highest point, rather than as a run of neighbouring peaks.
std::vector<std::size_t> local_maximum_indices(const std::vector<double>& y, int radius) {
  radius = std::max(1, radius);
  const auto r = static_cast<std::size_t>(radius);
  std::vector<std::size_t> candidates;
  if (y.size() <= 2 * r) return candidates;

  for (std::size_t i = r; i < y.size() - r; ++i) {
    const double window_max = *std::max_element(y.begin() + static_cast<std::ptrdiff_t>(i - r),
                                                y.begin() + static_cast<std::ptrdiff_t>(i + r + 1));
    if (y[i] < window_max) continue;

    if (!candidates.empty() && i - candidates.back() <= r) {
      if (y[i] > y[candidates.back()]) candidates.back() = i;
    } else {
      candidates.push_back(i);
    }
  }
  return candidates;
}

}  // namespace

void find_two_peaks(const std::vector<double>& counts, const std::vector<double>& centers,
                    const HistogramOptions& opts, HistogramResult& out) {
  if (counts.size() < 2 || counts.size() != centers.size())
    throw std::runtime_error("histogram: counts and centers must match and hold at least 2 bins");

  const std::vector<double> smooth = moving_average(counts, opts.smooth_width);
  std::vector<std::size_t> candidates = local_maximum_indices(smooth, opts.peak_window);

  // Rank local maxima by smoothed height, then greedily take the two that are
  // at least min_peak_distance apart.
  std::sort(candidates.begin(), candidates.end(),
            [&smooth](std::size_t a, std::size_t b) { return smooth[a] > smooth[b]; });

  const auto min_distance = static_cast<std::size_t>(std::max(0, opts.min_peak_distance));
  std::vector<std::size_t> selected;
  for (const std::size_t i : candidates) {
    const bool far_enough = std::all_of(selected.begin(), selected.end(), [&](std::size_t j) {
      return (i > j ? i - j : j - i) >= min_distance;
    });
    if (far_enough) selected.push_back(i);
    if (selected.size() == 2) break;
  }

  // Fallback when the strict local-max test found fewer than two: take the
  // global maximum, suppress its neighbourhood, then take the next strongest.
  if (selected.size() < 2) {
    const auto first =
        static_cast<std::size_t>(std::distance(smooth.begin(), std::max_element(smooth.begin(), smooth.end())));
    std::vector<double> suppressed = smooth;
    const std::size_t lo = first > min_distance ? first - min_distance : 0;
    const std::size_t hi = std::min(suppressed.size(), first + min_distance + 1);
    for (std::size_t i = lo; i < hi; ++i) suppressed[i] = -std::numeric_limits<double>::infinity();
    const auto second = static_cast<std::size_t>(
        std::distance(suppressed.begin(), std::max_element(suppressed.begin(), suppressed.end())));
    selected = {first, second};
  }

  // Order by intensity, not by height.
  std::sort(selected.begin(), selected.end());
  const std::size_t i1 = selected[0];
  const std::size_t i2 = selected[1];

  const double bin_width = centers[1] - centers[0];
  const auto refine = [&](std::size_t i) {
    if (i == 0 || i + 1 >= smooth.size()) return centers[i];
    return centers[i] + parabolic_offset(smooth[i - 1], smooth[i], smooth[i + 1]) * bin_width;
  };

  out.peak_low = refine(i1);
  out.peak_high = refine(i2);
  out.peak_low_bin = static_cast<int>(i1);
  out.peak_high_bin = static_cast<int>(i2);
  out.peak_low_height = smooth[i1];
  out.peak_high_height = smooth[i2];
}

HistogramResult measure_histogram_1d(std::vector<double>& values, const HistogramOptions& opts) {
  if (opts.bins < 2) throw std::runtime_error("histogram: bins must be >= 2");
  if (!(opts.hist_lo_percentile < opts.hist_hi_percentile))
    throw std::runtime_error("histogram: hist_lo_percentile must be < hist_hi_percentile");

  HistogramResult result;
  result.n_valid = static_cast<std::int64_t>(values.size());
  if (result.n_valid < 10)
    throw std::runtime_error("histogram: not enough valid nonzero pixels (need at least 10)");

  std::mt19937_64 rng(opts.seed);
  subsample_inplace(values, opts.downsample_factor, rng);
  result.n_sampled = static_cast<std::int64_t>(values.size());

  const auto mm = std::minmax_element(values.begin(), values.end());
  result.min_value = *mm.first;
  result.max_value = *mm.second;

  const std::vector<double>& ladder = default_percentiles();
  result.percentiles.reserve(ladder.size());
  for (const double q : ladder) result.percentiles.push_back(percentile_linear(values, q));

  result.hist_lo = percentile_linear(values, opts.hist_lo_percentile);
  result.hist_hi = percentile_linear(values, opts.hist_hi_percentile);
  if (!(result.hist_hi > result.hist_lo))
    throw std::runtime_error("histogram: degenerate percentile range; cannot build a histogram");

  const auto bins = static_cast<std::size_t>(opts.bins);
  const double width = (result.hist_hi - result.hist_lo) / static_cast<double>(bins);
  std::vector<double> counts(bins, 0.0);
  for (const double x : values) {
    // np.histogram with an explicit range drops values outside it; the top edge
    // is inclusive and lands in the last bin.
    if (x < result.hist_lo || x > result.hist_hi) continue;
    auto b = static_cast<std::size_t>((x - result.hist_lo) / width);
    if (b >= bins) b = bins - 1;
    counts[b] += 1.0;
  }

  std::vector<double> centers(bins, 0.0);
  for (std::size_t b = 0; b < bins; ++b) {
    centers[b] = result.hist_lo + (static_cast<double>(b) + 0.5) * width;
  }

  find_two_peaks(counts, centers, opts, result);
  return result;
}

HistogramResult measure_histogram(const Image2D& image, const HistogramOptions& opts) {
  return measure_histogram(image.pixels.data(), image.pixels.size(), opts);
}

}  // namespace mscoupon
