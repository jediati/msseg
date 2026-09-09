#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "diffg/image.hpp"
#include "diffg/multi_image.hpp"

namespace msseg {

// How a multi-sample TIFF becomes the planes the filter chains consume.
struct ColorInputPolicy {
  enum class Alpha { Drop, Keep };
  // A 4-sample file is read as RGBA and a 2-sample file as gray+alpha (TinyTIFF
  // exposes no ExtraSamples tag, so alpha is inferred from the count). `Drop`
  // discards that last plane: differentiating an alpha mask would make every
  // alpha edge a colour edge.
  Alpha alpha = Alpha::Drop;
  // The colour->scalar method a chain gets when the input has more than one
  // plane and the chain's first stage is not an explicit `color` stage.
  std::string default_method = "luminance";
};

// One loaded slice: C planar float32 planes (depth 1), channel-slowest, the
// layout diffg's MultiImage uses. A grayscale file is one plane; a consumer that
// only knows scalars reads `scalar()`, which is that plane copied out.
struct InputSlice {
  diffg::MultiImage<float> planes;
  std::uint16_t file_samples = 0;   // SamplesPerPixel as stored in the file
  bool alpha_dropped = false;

  std::size_t channels() const { return planes.channels(); }
  std::size_t width() const { return planes.dims().width; }
  std::size_t height() const { return planes.dims().height; }
  bool is_scalar() const { return planes.channels() == 1; }
  diffg::MultiImageView<const float> view() const { return planes.view(); }

  // Copy of plane 0 -- the byte-identical grayscale path.
  diffg::Image<float> scalar() const { return plane(0); }

  diffg::Image<float> plane(std::size_t c) const {
    diffg::Image<float> out(planes.dims(), planes.spacing());
    std::memcpy(out.data(), planes.channel_data(c), out.size() * sizeof(float));
    return out;
  }

  // Wrap a scalar image as a one-plane slice.
  static InputSlice from_image(const diffg::Image<float>& image) {
    InputSlice slice;
    slice.planes = diffg::MultiImage<float>(image.dims(), 1, image.spacing());
    std::memcpy(slice.planes.channel_data(0), image.data(), image.size() * sizeof(float));
    slice.file_samples = 1;
    return slice;
  }
};

}  // namespace msseg
