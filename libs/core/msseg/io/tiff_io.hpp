#pragma once

#include <cstdint>
#include <filesystem>

#include "diffg/image.hpp"
#include "msseg/workflow/input_slice.hpp"

namespace msseg {

// Single-slice TIFF I/O (part of the msseg_io target, which isolates the
// TinyTIFF dependency from the portable core). The input must be single-sample
// (grayscale); 8/16/32-bit integer (signed or unsigned) samples are converted to
// float, and 32-bit float is read as-is. A slice loads as a diffg::Image with
// depth == 1. Everything downstream (filters, MSC) operates on float32.
diffg::Image<float> read_tiff_float32(const std::filesystem::path& path);

// Every sample of a TIFF as planar float32 planes: a grayscale file is one
// plane, RGB three, RGBA four minus the alpha the policy drops. Same dtype
// handling as read_tiff_float32; both chunky and planar files decode (TinyTIFF
// extracts one sample at a time either way, re-scanning the strips of a chunky
// file once per sample).
InputSlice read_tiff_planes(const std::filesystem::path& path, const ColorInputPolicy& policy = {});

void write_tiff_float32(const std::filesystem::path& path, int width, int height, const float* data);
void write_tiff_mask_u8(const std::filesystem::path& path, int width, int height, const std::uint8_t* data);
void write_tiff_int32(const std::filesystem::path& path, int width, int height, const std::int32_t* data);

}  // namespace msseg
