#include "mscoupon/stats.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace mscoupon {
namespace {

struct Accum {
  std::size_t area = 0;
  double sum = 0.0;
  float min_value = std::numeric_limits<float>::max();
  float max_value = std::numeric_limits<float>::lowest();
  int min_x = std::numeric_limits<int>::max();
  int min_y = std::numeric_limits<int>::max();
  int max_x = std::numeric_limits<int>::lowest();
  int max_y = std::numeric_limits<int>::lowest();
};

bool contains(const std::vector<int>& values, int needle) {
  return std::find(values.begin(), values.end(), needle) != values.end();
}

}  // namespace

SegmentTable compute_segment_table(const Image2D& source, const std::vector<int>& labels, int slice_index) {
  if (source.width <= 0 || source.height <= 0) throw std::runtime_error("Invalid image dimensions in stats stage.");
  if (labels.size() != source.pixels.size()) throw std::runtime_error("Label size does not match image size.");

  std::unordered_map<int, Accum> map;
  for (int y = 0; y < source.height; ++y) {
    for (int x = 0; x < source.width; ++x) {
      const std::size_t idx = static_cast<std::size_t>(y) * static_cast<std::size_t>(source.width) + static_cast<std::size_t>(x);
      const int id = labels[idx];
      auto& acc = map[id];
      ++acc.area;
      const float value = source.pixels[idx];
      acc.sum += value;
      acc.min_value = std::min(acc.min_value, value);
      acc.max_value = std::max(acc.max_value, value);
      acc.min_x = std::min(acc.min_x, x);
      acc.min_y = std::min(acc.min_y, y);
      acc.max_x = std::max(acc.max_x, x);
      acc.max_y = std::max(acc.max_y, y);
    }
  }

  SegmentTable table;
  table.rows.reserve(map.size());
  for (const auto& [id, acc] : map) {
    SegmentStat row;
    row.segment_id = id;
    row.slice_index = slice_index;
    row.area = acc.area;
    row.min_value = acc.min_value;
    row.max_value = acc.max_value;
    row.mean_value = static_cast<float>(acc.sum / static_cast<double>(acc.area));
    row.min_x = acc.min_x;
    row.min_y = acc.min_y;
    row.max_x = acc.max_x;
    row.max_y = acc.max_y;
    table.rows.push_back(row);
  }
  std::sort(table.rows.begin(), table.rows.end(),
            [](const SegmentStat& a, const SegmentStat& b) { return a.segment_id < b.segment_id; });
  return table;
}

std::unordered_set<int> select_segment_ids(const SegmentTable& table, const SegmentKeepConfig& keep) {
  std::unordered_set<int> keep_ids;
  for (const auto& row : table.rows) {
    bool selected = true;
    if (keep.min_area.has_value() && row.area < *keep.min_area) selected = false;
    if (keep.max_area.has_value() && row.area > *keep.max_area) selected = false;
    if (keep.min_value.has_value() && row.min_value < *keep.min_value) selected = false;
    if (keep.max_value.has_value() && row.max_value > *keep.max_value) selected = false;
    if (keep.min_mean.has_value() && row.mean_value < *keep.min_mean) selected = false;
    if (keep.max_mean.has_value() && row.mean_value > *keep.max_mean) selected = false;
    if (!keep.allow_ids.empty() && !contains(keep.allow_ids, row.segment_id)) selected = false;
    if (!keep.deny_ids.empty() && contains(keep.deny_ids, row.segment_id)) selected = false;
    if (selected) keep_ids.insert(row.segment_id);
  }
  return keep_ids;
}

Mask2D build_mask(const std::vector<int>& labels, int width, int height, const std::unordered_set<int>& keep_ids) {
  Mask2D mask;
  mask.width = width;
  mask.height = height;
  mask.pixels.resize(labels.size(), 0);
  for (std::size_t i = 0; i < labels.size(); ++i) {
    mask.pixels[i] = keep_ids.count(labels[i]) > 0 ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
  }
  return mask;
}

}  // namespace mscoupon
