#pragma once

#include <vector>

#include "mscoupon/config.hpp"
#include "mscoupon/types.hpp"

namespace mscoupon {

Image2D apply_filter(const Image2D& image, const FilterConfig& filter);

// Apply an ordered chain of filters (output -> input). An empty chain returns a
// copy of the input.
Image2D apply_filter_chain(const Image2D& image, const std::vector<FilterConfig>& filters);

}  // namespace mscoupon
