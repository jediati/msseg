#pragma once

#include <filesystem>

#include "msseg/volume/types.hpp"

namespace msseg {

// Read a raw float32 volume. Dimensions come from `dims` (row-major, x fastest
// then y then z). The file must contain exactly width*height*depth floats.
Volume read_raw_volume(const std::filesystem::path& path, const Dims3& dims);

// Read a raw float32 volume, taking dimensions from a sidecar text file
// (`<path>.dat` by convention) containing "width height depth".
Volume read_raw_volume(const std::filesystem::path& path);

// Write a raw float32 volume and a "<path>.dat" sidecar with its dimensions.
void write_raw_volume(const std::filesystem::path& path, const Volume& volume);

// Write a label volume as raw int32 plus a "<path>.dat" dimension sidecar.
void write_raw_labels(const std::filesystem::path& path, const LabelVolume& labels);

}  // namespace msseg
