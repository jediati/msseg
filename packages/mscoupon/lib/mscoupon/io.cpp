#include "mscoupon/io.hpp"

#include "mscoupon/query.hpp"

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
                            const GlobalFeatureTable& table,
                            const StatisticsConfig& cfg) {
  std::ofstream out(path);
  if (!out.good()) {
    throw std::runtime_error("Failed to write global table: " + path.string());
  }
  // Header AND values come from one projection, so they cannot drift: a run that
  // does not compute a channel emits no column for it, and a twelve-channel
  // scale-space stack needs no edit here at all. The per-slice reductions are the
  // one tail the channel schema does not name -- their keys come from the first
  // row, and every row carries the same keys.
  const mscoupon::FeatureTable projected =
      global_feature_table(table.rows, table.channels, table.schema, cfg.spec);

  std::vector<std::string> ps_names;
  if (!table.rows.empty()) {
    for (const auto& kv : table.rows.front().per_slice) ps_names.push_back(kv.first);
  }

  for (std::size_t c = 0; c < projected.fields.size(); ++c) {
    out << (c == 0 ? "" : ",") << projected.fields[c].name;
  }
  for (const auto& n : ps_names) out << "," << n;
  out << "\n";

  for (std::size_t r = 0; r < projected.n_rows; ++r) {
    for (std::size_t c = 0; c < projected.fields.size(); ++c) {
      out << (c == 0 ? "" : ",") << projected.at(r, c);
    }
    const auto& per_slice = table.rows[r].per_slice;
    for (const auto& n : ps_names) {
      const auto it = per_slice.find(n);
      out << "," << (it == per_slice.end() ? 0.0 : it->second);
    }
    out << "\n";
  }
}

}  // namespace mscoupon
