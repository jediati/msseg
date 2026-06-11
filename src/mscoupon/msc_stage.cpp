#include "mscoupon/msc_stage.hpp"

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
  msc.setPersistence(cfg.persistence);

  if (cfg.manifold == "ascending") {
    return msc.ascending2Manifolds().labels;
  }
  if (cfg.manifold == "descending") {
    return msc.descending2Manifolds().labels;
  }
  throw std::runtime_error("Invalid msc.manifold value. Use 'ascending' or 'descending'.");
}

}  // namespace mscoupon
