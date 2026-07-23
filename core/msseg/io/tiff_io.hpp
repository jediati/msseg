#pragma once

#include <cstdint>
#include <filesystem>

#include "diffg/image.hpp"

namespace msseg {

// Single-slice TIFF I/O (part of the msseg_io target, which isolates the
// TinyTIFF dependency from the portable core). Inputs must be single-sample
// float32; a slice loads as a diffg::Image with depth == 1.
diffg::Image<float> read_tiff_float32(const std::filesystem::path& path);

void write_tiff_float32(const std::filesystem::path& path, int width, int height, const float* data);
void write_tiff_mask_u8(const std::filesystem::path& path, int width, int height, const std::uint8_t* data);
void write_tiff_int32(const std::filesystem::path& path, int width, int height, const std::int32_t* data);

}  // namespace msseg
