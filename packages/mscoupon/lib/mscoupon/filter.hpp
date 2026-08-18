#pragma once

#include <vector>

#include "mscoupon/config.hpp"
#include "mscoupon/normalize.hpp"
#include "mscoupon/types.hpp"

namespace mscoupon {

// The one filter operation this package implements itself; everything else is
// delegated to the core filter stage. See normalize.hpp for why two-point
// normalization is modelled as a filter rather than a threshold transform.
inline constexpr const char* kNormalizeOperation = "normalize";

Image2D apply_filter(const Image2D& image, const FilterConfig& filter);

// Apply an ordered chain of filters (output -> input). An empty chain returns a
// copy of the input.
//
// When `normalizers_out` is non-null, the landmarks measured by each
// `normalize` stage are appended to it, in chain order, so callers can report
// or pin them without re-measuring.
Image2D apply_filter_chain(const Image2D& image, const std::vector<FilterConfig>& filters,
                           std::vector<TwoPoint>* normalizers_out = nullptr);

}  // namespace mscoupon
