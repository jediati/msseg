#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace mscoupon {

struct Image2D {
  int width = 0;
  int height = 0;
  std::vector<float> pixels;
};

struct Mask2D {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> pixels;
};

struct SegmentStat {
  int segment_id = -1;
  int slice_index = -1;
  std::size_t area = 0;
  float min_value = 0.0f;
  float max_value = 0.0f;
  float mean_value = 0.0f;
  int min_x = 0;
  int min_y = 0;
  int max_x = 0;
  int max_y = 0;
};

struct SegmentTable {
  std::vector<SegmentStat> rows;
};

struct StageTiming {
  double read_ms = 0.0;
  double filter_ms = 0.0;
  double msc_ms = 0.0;
  double stats_ms = 0.0;
  double select_ms = 0.0;
  double write_ms = 0.0;
  double total_ms = 0.0;
  double out_time_ms = 0.0;
};

struct SliceOutput {
  std::filesystem::path input_path;
  std::filesystem::path mask_output_path;
  std::filesystem::path table_output_path;
  int slice_index = 0;
  StageTiming timing;
};

}  // namespace mscoupon
