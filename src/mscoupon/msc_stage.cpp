#include "mscoupon/msc_stage.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "msc_2d_lib.h"

namespace mscoupon {

std::vector<int> compute_msc_labels(const Image2D& filtered_image, const MscConfig& cfg) {
  if (filtered_image.width <= 0 || filtered_image.height <= 0) {
    throw std::runtime_error("Invalid image dimensions for MSC.");
  }

  GInt::Msc2D::Msc2D msc;
  msc.compute(filtered_image.pixels.data(), filtered_image.height, filtered_image.width, cfg.accurate_ascending,
              cfg.accurate_descending);

  float persistence_absolute = 0.0f;
  if (cfg.persistence_absolute.has_value()) {
    persistence_absolute = *cfg.persistence_absolute;
  } else if (cfg.persistence_percent.has_value()) {
    float min_v = std::numeric_limits<float>::max();
    float max_v = std::numeric_limits<float>::lowest();
    for (float v : filtered_image.pixels) {
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

}  // namespace mscoupon
