#pragma once

#include <vector>

#include "mscoupon/config.hpp"
#include "mscoupon/types.hpp"

namespace mscoupon {

std::vector<int> compute_msc_labels(const Image2D& filtered_image, const MscConfig& cfg);

}  // namespace mscoupon
