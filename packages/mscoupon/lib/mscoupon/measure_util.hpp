#pragma once

// Numeric helpers shared by the three intensity measures (gmm.hpp,
// histogram_peaks.hpp, region_measure.hpp).
//
// All three measures start from the same two steps -- mask the no-data pixels,
// then optionally take a random 1/N subsample -- and two of them refine a
// histogram peak to sub-bin precision the same way. Keeping one copy here means
// the C++ measures cannot drift from each other, and matches the single
// get_valid_pixels / random_downsample / parabolic refinement that the Python
// side collapses to.

#include <cmath>
#include <cstddef>
#include <optional>
#include <cstdint>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace mscoupon {

// The percentile ladder reported by measure_im.py and measure_2_regions.py.
inline const std::vector<double>& default_percentiles() {
  static const std::vector<double> kLadder = {0.01, 0.1, 1.0,  5.0,  25.0, 50.0,
                                              75.0, 95.0, 99.0, 99.9, 99.99};
  return kLadder;
}

// Column names for the ladder above, in the same order. These are the Python
// spelling -- str(p).replace(".", "_") -- so whole numbers keep one decimal
// ("p1_0", not "p1"). Listed rather than formatted so the C++ and the CSVs
// cannot drift over float repr.
inline const std::vector<std::string>& default_percentile_names() {
  static const std::vector<std::string> kNames = {"p0_01", "p0_1",  "p1_0",  "p5_0",
                                                  "p25_0", "p50_0", "p75_0", "p95_0",
                                                  "p99_0", "p99_9", "p99_99"};
  return kNames;
}

// numpy.percentile with the default "linear" interpolation. Reorders `v`, which
// is harmless here: every consumer treats the sample as an unordered bag.
double percentile_linear(std::vector<double>& v, double q);

// Sub-bin peak refinement: fit a parabola through the peak bin and its two
// neighbours and return the offset from the peak bin centre, in bins. Returns 0
// when the parabola is degenerate; the result is clamped to [-1, 1] so a flat
// or noisy triple cannot throw the peak into a neighbouring bin.
double parabolic_offset(double y1, double y2, double y3);

// Random 1/factor subsample without replacement (partial Fisher-Yates), in
// place. Keeps at least 10 values. factor <= 1 leaves `v` untouched.
void subsample_inplace(std::vector<double>& v, int factor, std::mt19937_64& rng);

// Apply the no-data / non-finite mask to a raw single-channel pixel buffer.
// Templated so callers can stay dtype-general; works for float, double, and the
// integer TIFF types. Integer inputs skip the finite test, mirroring the
// np.issubdtype(x.dtype, np.floating) branch in the Python.
//
// `omit_value` is the stack's no-data sentinel, not necessarily zero: a
// reconstruction may pad with any constant (43, say), and dropping the wrong
// value leaves that plateau in the fit as a spurious population. std::nullopt
// keeps every pixel. The comparison is done in the pixel type so a float32
// raster is matched against the sentinel rounded the same way it was stored,
// rather than against a double that never compares equal.
template <typename T>
std::vector<double> collect_valid_pixels(const T* pixels, std::size_t count,
                                         std::optional<double> omit_value,
                                         bool omit_nonfinite) {
  std::vector<double> out;
  out.reserve(count);
  const bool has_omit = omit_value.has_value();
  const T omit = has_omit ? static_cast<T>(*omit_value) : T(0);
  for (std::size_t i = 0; i < count; ++i) {
    const T v = pixels[i];
    if (has_omit && v == omit) continue;
    if constexpr (std::is_floating_point_v<T>) {
      if (omit_nonfinite && !std::isfinite(v)) continue;
    }
    out.push_back(static_cast<double>(v));
  }
  return out;
}

}  // namespace mscoupon
