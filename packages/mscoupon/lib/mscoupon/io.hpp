#pragma once

#include <filesystem>
#include <vector>

#include "mscoupon/types.hpp"

namespace mscoupon {

Image2D read_tiff_float32(const std::filesystem::path& path);
void write_tiff_mask_u8(const std::filesystem::path& path, const Mask2D& mask);
void write_tiff_float32(const std::filesystem::path& path, const Image2D& image);
void write_tiff_int32(const std::filesystem::path& path, int width, int height, const std::vector<int>& labels);
void write_segment_table_csv(const std::filesystem::path& path, const std::vector<SegmentStat>& stats);
void write_feature_map_csv(const std::filesystem::path& path, const std::vector<FeatureMapRow>& rows);
void write_global_table_csv(const std::filesystem::path& path, const std::vector<GlobalFeatureStat>& stats);

}  // namespace mscoupon
