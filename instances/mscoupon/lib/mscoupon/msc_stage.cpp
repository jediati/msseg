#include "mscoupon/msc_stage.hpp"

#include <algorithm>
#include <cstddef>

#include "diffg/image.hpp"
#include "msseg/compute/msc2d.hpp"

namespace mscoupon {

// Thin adapter over the core 2D MSC stage. Maps this instance's MscConfig onto
// msseg::Msc2DParams and its Image2D onto a diffg image (depth == 1).
std::vector<int> compute_msc_labels(const Image2D& filtered_image, const MscConfig& cfg) {
  diffg::Image<float> filtered(diffg::Dimensions{
      static_cast<std::size_t>(filtered_image.width), static_cast<std::size_t>(filtered_image.height), 1});
  std::copy(filtered_image.pixels.begin(), filtered_image.pixels.end(), filtered.data());

  msseg::Msc2DParams params;
  params.persistence_absolute = cfg.persistence_absolute;
  params.persistence_percent = cfg.persistence_percent;
  params.compute_algorithm = cfg.compute_algorithm;
  params.accurate_ascending = cfg.accurate_ascending;
  params.accurate_descending = cfg.accurate_descending;
  params.manifold = cfg.manifold;
  params.requested_parallelism = cfg.requested_parallelism;

  return msseg::compute_msc2d_labels(filtered, params);
}

}  // namespace mscoupon
