#include "msseg/compute/msc2d.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "msc_2d_lib.h"

namespace msseg {
namespace {

// The linked msc_2d_lib may or may not expose a ComputeOptions/BuilderMode
// surface (it depends on the pinned MSCEER revision). Detect it at compile
// time so we can honor compute_algorithm/accurate flags when available and
// fall back to the legacy 5-arg compute() otherwise.
template <typename MscType, typename = void>
struct HasComputeOptions : std::false_type {};

template <typename MscType>
struct HasComputeOptions<MscType, std::void_t<typename MscType::ComputeOptions>> : std::true_type {};

template <typename MscType>
void compute_with_algorithm(MscType& msc, const float* pixels, int rows, int cols, const Msc2DParams& cfg) {
  if constexpr (HasComputeOptions<MscType>::value) {
    typename MscType::ComputeOptions options;
    options.accurateAsc = cfg.accurate_ascending;
    options.accurateDsc = cfg.accurate_descending;
    if (cfg.requested_parallelism > 0) {
      options.requestedParallelism = cfg.requested_parallelism;
    }
    if (cfg.compute_algorithm == "partitioned") {
      options.builderMode = MscType::BuilderMode::Partitioned;
    } else {
      options.builderMode = MscType::BuilderMode::Serial;
    }
    msc.compute(pixels, rows, cols, options);
  } else {
    if (cfg.compute_algorithm == "partitioned") {
      throw std::runtime_error(
          "Configured msc.compute_algorithm='partitioned' but linked msc_2d_lib does not expose "
          "ComputeOptions/BuilderMode.");
    }
    msc.compute(pixels, rows, cols, cfg.accurate_ascending, cfg.accurate_descending);
  }
}

}  // namespace

std::vector<int> compute_msc2d_labels(const diffg::Image<float>& filtered, const Msc2DParams& cfg) {
  const int width = static_cast<int>(filtered.dims().width);
  const int height = static_cast<int>(filtered.dims().height);
  if (width <= 0 || height <= 0) {
    throw std::runtime_error("Invalid image dimensions for MSC.");
  }
  if (filtered.dims().depth != 1) {
    throw std::runtime_error("compute_msc2d_labels requires a 2D image (depth == 1).");
  }

  GInt::Msc2D::Msc2D msc;
  compute_with_algorithm(msc, filtered.data(), height, width, cfg);

  float persistence_absolute = 0.0f;
  if (cfg.persistence_absolute.has_value()) {
    persistence_absolute = *cfg.persistence_absolute;
  } else if (cfg.persistence_percent.has_value()) {
    float min_v = std::numeric_limits<float>::max();
    float max_v = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < filtered.size(); ++i) {
      const float v = filtered.data()[i];
      min_v = std::min(min_v, v);
      max_v = std::max(max_v, v);
    }
    const float range = max_v - min_v;
    persistence_absolute = range * (*cfg.persistence_percent / 100.0f);
  } else {
    throw std::runtime_error("MSC persistence not configured.");
  }
  msc.setPersistence(persistence_absolute);

  if (cfg.manifold == "ascending") {
    return msc.ascending2Manifolds().labels;
  }
  if (cfg.manifold == "descending") {
    return msc.descending2Manifolds().labels;
  }
  throw std::runtime_error("Invalid msc.manifold value. Use 'ascending' or 'descending'.");
}

}  // namespace msseg
