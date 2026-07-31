#pragma once

#include <vector>

#include "diffg/image.hpp"
#include "msseg/workflow/params.hpp"

namespace msseg {

// Compute the 2D Morse-Smale complex over a filtered slice, simplify to the
// configured persistence, and return the ascending/descending 2-manifold
// label image (row-major, one label per pixel). Requires depth == 1.
std::vector<int> compute_msc2d_labels(const diffg::Image<float>& filtered, const Msc2DParams& cfg);

}  // namespace msseg
