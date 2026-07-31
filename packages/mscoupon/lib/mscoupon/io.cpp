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
  out << "segment_id,slice_index,area,min_value,max_value,mean_value,min_x,min_y,max_x,max_y\n";
  for (const auto& row : stats) {
    out << row.segment_id << "," << row.slice_index << "," << row.area << "," << row.min_value << "," << row.max_value
        << "," << row.mean_value << "," << row.min_x << "," << row.min_y << "," << row.max_x << "," << row.max_y << "\n";
  }
}

}  // namespace mscoupon
