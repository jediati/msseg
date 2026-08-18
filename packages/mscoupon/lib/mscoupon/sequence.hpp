#pragma once

#include <filesystem>
#include <vector>

#include "mscoupon/config.hpp"

namespace mscoupon {

struct SliceJob {
  std::filesystem::path input_path;
  std::filesystem::path mask_output_path;
  std::filesystem::path table_output_path;
  std::filesystem::path filter_output_path;
  std::filesystem::path label_output_path;
  std::filesystem::path cc_label_output_path;      // per-slice in-plane CC id raster
  std::filesystem::path global_label_output_path;  // per-slice GLOBAL id raster
  int slice_index = 0;
};

std::vector<SliceJob> build_sequence(const AppConfig& cfg);

}  // namespace mscoupon
