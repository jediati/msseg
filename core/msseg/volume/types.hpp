#pragma once

#include <cstddef>
#include <cstdint>

#include "diffg/image.hpp"

namespace msseg {

// The core currency for volumetric data. diffg::Image is 3D-capable
// (Dimensions{width, height, depth}); a 2D slice is simply depth == 1.
using Volume = diffg::Image<float>;
using LabelVolume = diffg::Image<std::int32_t>;

struct Dims3 {
  std::size_t width = 0;
  std::size_t height = 0;
  std::size_t depth = 1;
};

// Reserved label for "not a feature" voxels in a segmentation.
inline constexpr std::int32_t kBackgroundLabel = 0;

}  // namespace msseg
