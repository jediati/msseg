#pragma once

// Histogram-based two-population measurement: a dependency-free C++ port of
// measure_im.py.
//
//   mask no-data + non-finite -> random 1/N subsample -> percentiles ->
//   histogram over [p_lo, p_hi] -> moving-average smoothing -> the two
//   strongest separated local maxima -> parabolic sub-bin refinement.
//
// Peaks come back ordered by intensity, not by height: peak_low is the
// low-intensity population (air/void), peak_high the high one (metal/solid).
//
// The local-maximum test is deliberately stricter than "higher than both
// neighbours": a candidate must dominate a window of +/- peak_window bins,
// which suppresses the shoulders of a broad peak.
//
// KNOWN LIMITATION, inherited from measure_im.py and kept for parity: a
// candidate must have peak_window bins on each side, so a population sitting
// within peak_window bins of the histogram's edge can never be selected. That
// happens when one population is so dominant that hist_lo_percentile lands
// inside it -- e.g. a mostly-background slice. Symptom: both reported peaks sit
// in the flat region between the real populations. Widen the percentile range
// or shrink peak_window if you hit it.

#include <cstdint>
#include <vector>

#include "mscoupon/measure_util.hpp"
#include "mscoupon/types.hpp"

namespace mscoupon {

struct HistogramOptions {
  // Random 1/N subsample of the valid pixels, without replacement. 1 = use all.
  int downsample_factor = 1;
  // No-data sentinel dropped before the histogram is built; std::nullopt keeps
  // every pixel. See GmmOptions::omit_value.
  std::optional<double> omit_value = 0.0;
  // Drop NaN/Inf. Only meaningful for floating-point inputs.
  bool omit_nonfinite = true;
  int bins = 1024;
  // Width, in bins, of the centred moving average. Forced odd.
  int smooth_width = 11;
  // Radius, in bins, over which a candidate must be the maximum.
  int peak_window = 32;
  // Minimum separation, in bins, between the two selected maxima.
  int min_peak_distance = 64;
  // The histogram spans these two percentiles of the sample. measure_im.py names
  // these p5/p95 in its locals but actually reads p1_0/p99_0; the percentiles,
  // not the names, are what is reproduced here.
  double hist_lo_percentile = 1.0;
  double hist_hi_percentile = 99.0;
  std::uint64_t seed = 0;
};

struct HistogramResult {
  // Ordered by intensity: low population first.
  double peak_low = 0.0;
  double peak_high = 0.0;
  int peak_low_bin = 0;
  int peak_high_bin = 0;
  double peak_low_height = 0.0;
  double peak_high_height = 0.0;

  // Histogram support (the two percentile cut points).
  double hist_lo = 0.0;
  double hist_hi = 0.0;

  std::int64_t n_total = 0;    // every pixel in the input
  // Pixels equal to the no-data sentinel (omit_value), counted before masking;
  // exact zeros when no sentinel is configured.
  std::int64_t n_zero = 0;
  std::int64_t n_valid = 0;    // surviving the no-data / non-finite mask
  std::int64_t n_sampled = 0;  // surviving the random subsample

  double min_value = 0.0;
  double max_value = 0.0;
  // Parallel to default_percentiles().
  std::vector<double> percentiles;
};

// Locate the two strongest separated maxima of an already-built histogram.
// `counts` and `centers` must be the same length. Exposed for testing.
void find_two_peaks(const std::vector<double>& counts, const std::vector<double>& centers,
                    const HistogramOptions& opts, HistogramResult& out);

// Measure already-masked values. `values` is consumed (reordered and shrunk).
// Throws std::runtime_error where the Python raises ValueError.
HistogramResult measure_histogram_1d(std::vector<double>& values, const HistogramOptions& opts);

// Mask, subsample and measure in one call.
template <typename T>
HistogramResult measure_histogram(const T* pixels, std::size_t count,
                                  const HistogramOptions& opts) {
  std::int64_t n_zero = 0;
  const T sentinel = static_cast<T>(opts.omit_value.value_or(0.0));
  for (std::size_t i = 0; i < count; ++i) {
    if (pixels[i] == sentinel) ++n_zero;
  }
  std::vector<double> values =
      collect_valid_pixels(pixels, count, opts.omit_value, opts.omit_nonfinite);
  HistogramResult r = measure_histogram_1d(values, opts);
  r.n_total = static_cast<std::int64_t>(count);
  r.n_zero = n_zero;
  return r;
}

// Convenience overload for the package's slice type.
HistogramResult measure_histogram(const Image2D& image, const HistogramOptions& opts);

}  // namespace mscoupon
