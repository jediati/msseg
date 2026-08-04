#pragma once

#include <vector>

#include "diffg/image.hpp"
#include "msseg/workflow/params.hpp"

namespace msseg {

// Apply a diffg filter/transform to a volume. Works for 2D (depth == 1) and
// 3D inputs alike, since every diffg operation is dimension-general.
// Returns the transformed volume; "none"/empty operation returns a copy.
diffg::Image<float> apply_filter(const diffg::Image<float>& input, const FilterParams& filter);

// Apply an ordered chain of filters, feeding each stage's output into the next.
// An empty chain returns a copy of the input.
diffg::Image<float> apply_filter_chain(const diffg::Image<float>& input,
                                       const std::vector<FilterParams>& filters);

}  // namespace msseg
