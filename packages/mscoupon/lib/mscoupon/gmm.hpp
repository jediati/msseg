#pragma once

// One-dimensional Gaussian mixture fitting, as a dependency-free C++ port of the
// two sklearn-based analysis scripts under src/msseg/mscoupon/:
//
//   calculate_2_gaussian_mixture.py
//       mask zeros + non-finite -> random 1/N subsample -> GaussianMixture(
//       n_components=2, covariance_type="full", n_init=3, max_iter=300,
//       tol=1e-6, reg_covar=1e-6) -> components sorted by increasing mean.
//       Reproduced by gmm_options_two_gaussian().
//
//   measure_gmm.py
//       the same, plus a symmetric percentile trim, means_init at the 25th/75th
//       percentiles, reg_covar=1e-12, tol=1e-8, max_iter=500, and per-component
//       hard-assignment statistics (mean, median, histogram mode).
//       Reproduced by gmm_options_measure().
//
// In one dimension sklearn's "full" covariance degenerates to a scalar variance,
// so the whole algorithm -- k-means++ seeding, Lloyd's, EM, the n_init restart
// loop -- needs no linear algebra and no third-party library.
//
// PARITY. This matches sklearn *algorithmically*, not bit-for-bit: the same EM
// update equations, the same reg_covar handling, the same convergence test on
// the mean per-sample log-likelihood, the same best-lower-bound selection across
// restarts, and the same component ordering. Exact reproduction would require
// porting NumPy's PCG64 (used by rng.choice) and sklearn's MT19937 RandomState
// (used by k-means++), which is not a goal here. On real data the fitted
// mu/sigma/weight agree with the Python to several significant digits.
//
// RNG SCOPE. The Python scripts thread a single RNG through a whole folder, so
// each slice draws a different subsample. Here the seed is per call: two calls
// with the same options and the same pixels give the same answer. Callers that
// want the folder behaviour should vary GmmOptions::seed per slice.

#include <cmath>
#include <cstddef>
#include <optional>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "mscoupon/measure_util.hpp"
#include "mscoupon/types.hpp"

namespace mscoupon {

// How component means are seeded before EM runs.
enum class GmmInit {
  // k-means++ seeding + Lloyd's, then one M-step off the hard assignment.
  // Equivalent to sklearn init_params="kmeans".
  KMeans,
  // As KMeans, but the resulting means are then overridden with evenly spaced
  // sample quantiles (for K=2: the 25th and 75th percentiles), leaving the
  // k-means weights and variances in place. This is exactly what sklearn does
  // when means_init is supplied, and is what measure_gmm.py asks for.
  Quantile,
};

struct GmmOptions {
  int n_components = 2;
  // Random 1/N subsample of the valid pixels, without replacement. 1 = use all.
  int downsample_factor = 1;
  // The stack's no-data sentinel, dropped before fitting. Usually 0 (the
  // background of a reconstructed slice), but a stack may pad with any constant
  // -- leaving that plateau in would fit it as a spurious population.
  // std::nullopt keeps every pixel.
  std::optional<double> omit_value = 0.0;
  // Drop NaN/Inf. Only meaningful for floating-point inputs.
  bool omit_nonfinite = true;
  // Symmetric percentile trim applied after subsampling, in [0, 50).
  // 0.5 keeps the 0.5th through 99.5th percentiles.
  double trim_percent = 0.0;
  GmmInit init = GmmInit::KMeans;
  // Random restarts; the run with the highest log-likelihood wins.
  int n_init = 3;
  int max_iter = 300;
  // Convergence threshold on the change in mean per-sample log-likelihood.
  double tol = 1e-6;
  // Added to every variance in every M-step (sklearn reg_covar).
  double reg_covar = 1e-6;
  std::uint64_t seed = 0;
  // Compute per-component hard-assignment mean/median/mode.
  bool compute_hard_stats = false;
  // Histogram bins used for the hard-assignment mode estimate.
  int mode_bins = 512;
};

struct GmmComponent {
  double mean = 0.0;
  double sigma = 0.0;
  double weight = 0.0;
  // Populated only when GmmOptions::compute_hard_stats is set.
  std::int64_t n_hard = 0;
  double hard_mean = 0.0;
  double median = 0.0;
  double mode = 0.0;
};

struct GmmResult {
  // Sorted by increasing mean, so component 0 is the low-intensity population.
  std::vector<GmmComponent> components;
  std::int64_t n_valid = 0;    // pixels surviving the zero / non-finite mask
  std::int64_t n_sampled = 0;  // pixels surviving the random subsample
  std::int64_t n_fit = 0;      // pixels surviving the percentile trim (fitted on)
  double trim_lo = 0.0;        // trim cut points; min/max when trimming is off
  double trim_hi = 0.0;
  double log_likelihood = 0.0;  // mean per-sample log-likelihood (sklearn lower_bound_)
  int n_iter = 0;
  bool converged = false;
};

// Fit a mixture to already-masked values. `values` is consumed: it is reordered
// by the subsample and shrunk by the trim, so callers that still need the data
// must pass a copy. Throws std::runtime_error where the Python raises ValueError
// (fewer than 10 usable values, or an empty component under hard assignment).
GmmResult fit_gmm_1d(std::vector<double>& values, const GmmOptions& opts);

// collect_valid_pixels() -- the zero / non-finite mask -- is shared with the
// other two measures and lives in measure_util.hpp.

// Mask, subsample, trim and fit in one call.
template <typename T>
GmmResult fit_gmm(const T* pixels, std::size_t count, const GmmOptions& opts) {
  std::vector<double> values =
      collect_valid_pixels(pixels, count, opts.omit_value, opts.omit_nonfinite);
  return fit_gmm_1d(values, opts);
}

// Convenience overload for the package's slice type.
GmmResult fit_gmm(const Image2D& image, const GmmOptions& opts);

// Presets matching the two scripts, so callers do not re-derive the constants.
GmmOptions gmm_options_two_gaussian();  // calculate_2_gaussian_mixture.py
GmmOptions gmm_options_measure();       // measure_gmm.py

}  // namespace mscoupon
