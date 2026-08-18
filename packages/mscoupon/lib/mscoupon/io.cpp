#include "mscoupon/io.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>

#include "msseg/io/tiff_io.hpp"

namespace mscoupon {

// TIFF I/O is delegated to the reusable msseg_io layer; this instance only
// adapts between its Image2D/Mask2D batch types and the core buffers, and owns
// the segment-table CSV (which is instance-specific).

Image2D read_tiff_float32(const std::filesystem::path& path) {
  const diffg::Image<float> img = msseg::read_tiff_float32(path);
  Image2D out;
  out.width = static_cast<int>(img.dims().width);
  out.height = static_cast<int>(img.dims().height);
  out.pixels.assign(img.data(), img.data() + img.size());
  return out;
}

void write_tiff_mask_u8(const std::filesystem::path& path, const Mask2D& mask) {
  msseg::write_tiff_mask_u8(path, mask.width, mask.height, mask.pixels.data());
}

void write_tiff_float32(const std::filesystem::path& path, const Image2D& image) {
  msseg::write_tiff_float32(path, image.width, image.height, image.pixels.data());
}

void write_tiff_int32(const std::filesystem::path& path, int width, int height, const std::vector<int>& labels) {
  msseg::write_tiff_int32(path, width, height, reinterpret_cast<const std::int32_t*>(labels.data()));
}

void write_segment_table_csv(const std::filesystem::path& path, const std::vector<SegmentStat>& stats) {
  std::ofstream out(path);
  if (!out.good()) {
    throw std::runtime_error("Failed to write table: " + path.string());
  }
  out << "segment_id,slice_index,area,min_value,max_value,mean_value,min_x,min_y,max_x,max_y,"
         "ext_x,ext_y,ext_base,ext_filtered\n";
  for (const auto& row : stats) {
    out << row.segment_id << "," << row.slice_index << "," << row.area << "," << row.min_value << "," << row.max_value
        << "," << row.mean_value << "," << row.min_x << "," << row.min_y << "," << row.max_x << "," << row.max_y
        << "," << row.ext_x << "," << row.ext_y << "," << row.ext_base << "," << row.ext_filtered << "\n";
  }
}

void write_feature_map_csv(const std::filesystem::path& path, const std::vector<FeatureMapRow>& rows) {
  std::ofstream out(path);
  if (!out.good()) {
    throw std::runtime_error("Failed to write feature map: " + path.string());
  }
  out << "slice_index,segment_id,global_id\n";
  for (const auto& row : rows) {
    out << row.slice_index << "," << row.segment_id << "," << row.global_id << "\n";
  }
}

void write_global_table_csv(const std::filesystem::path& path,
                            const std::vector<GlobalFeatureStat>& stats,
                            const StatisticsConfig& cfg) {
  std::ofstream out(path);
  if (!out.good()) {
    throw std::runtime_error("Failed to write global table: " + path.string());
  }
  // Columns follow the active statistics spec, so a run that does not compute a
  // channel does not emit a column of zeros for it. The per-slice column names
  // come from the first row -- every row carries the same keys.
  const msseg::StatsSpec& spec = cfg.spec;
  std::vector<std::string> ps_names;
  if (!stats.empty()) {
    for (const auto& [k, v] : stats.front().per_slice) ps_names.push_back(k);
  }

  out << "global_id,voxel_count,num_slices,first_slice,last_slice,"
         "min_x,min_y,min_z,max_x,max_y,max_z";
  const auto channel_header = [&](const char* suffix) {
    if (spec.mean) out << ",mean_" << suffix;
    if (spec.min) out << ",min_" << suffix;
    if (spec.max) out << ",max_" << suffix;
    if (spec.std) out << ",std_" << suffix;
  };
  if (spec.base_channel) channel_header("base");
  if (spec.filtered_channel) channel_header("filtered");
  if (spec.extremum) out << ",ext_x,ext_y,ext_z,ext_base,ext_filtered";
  for (const auto& n : ps_names) out << "," << n;
  out << "\n";

  for (const auto& row : stats) {
    out << row.global_id << "," << row.voxel_count << "," << row.num_slices << "," << row.first_slice << ","
        << row.last_slice << "," << row.min_x << "," << row.min_y << "," << row.min_z << "," << row.max_x << ","
        << row.max_y << "," << row.max_z;
    if (spec.base_channel) {
      if (spec.mean) out << "," << row.mean_base;
      if (spec.min) out << "," << row.min_base;
      if (spec.max) out << "," << row.max_base;
      if (spec.std) out << "," << row.std_base;
    }
    if (spec.filtered_channel) {
      if (spec.mean) out << "," << row.mean_filtered;
      if (spec.min) out << "," << row.min_filtered;
      if (spec.max) out << "," << row.max_filtered;
      if (spec.std) out << "," << row.std_filtered;
    }
    if (spec.extremum) {
      out << "," << row.ext_x << "," << row.ext_y << "," << row.ext_z << "," << row.ext_base
          << "," << row.ext_filtered;
    }
    for (const auto& n : ps_names) {
      const auto it = row.per_slice.find(n);
      out << "," << (it == row.per_slice.end() ? 0.0 : it->second);
    }
    out << "\n";
  }
}

}  // namespace mscoupon
