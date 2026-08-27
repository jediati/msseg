#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "msseg/compute/channel_stats.hpp"
#include "msseg/workflow/stat_channels.hpp"

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
  // The MSC region's seeding critical point (minimum for ascending manifolds,
  // maximum for descending), copied from msseg::Msc2DFeatureStat: its position
  // plus the base and filtered channels sampled there. Left at the defaults for
  // the background row (segment_id == -1), which has no critical point.
  float ext_x = -1.0f;
  float ext_y = -1.0f;
  float ext_base = 0.0f;
  float ext_filtered = 0.0f;
};

struct SegmentTable {
  std::vector<SegmentStat> rows;
};

// One kept 2D segment's membership in a cross-slice 3D feature: the per-slice
// (slice_index, segment_id) local identity mapped to its resolved global id.
struct FeatureMapRow {
  int slice_index = -1;
  int segment_id = -1;
  int global_id = -1;
};

// A cross-slice 3D feature: the union of per-slice connected components linked by
// `assembly.connectivity` across consecutive slices, with aggregated base+filtered
// statistics (means/std derived from mergeable sums; min/max/bbox unioned).
struct GlobalFeatureStat {
  int global_id = -1;
  std::size_t voxel_count = 0;
  int num_slices = 0;
  int first_slice = 0;
  int last_slice = 0;
  int min_x = 0;
  int min_y = 0;
  int min_z = 0;
  int max_x = 0;
  int max_y = 0;
  int max_z = 0;
  // The per-channel aggregates live in a parallel flat msseg::ChannelStats table
  // indexed by row, not as members here -- see GlobalFeatureTable. Reductions
  // over a channel stay VOXEL-pooled in 3D: a mean of per-slice means would
  // weight by slice count rather than by area. `per_slice` below is the one that
  // reduces across slices.
  float relevance_base = 0.0f;

  // The 3D feature's seeding extremum: the constituent slice component whose
  // `ext_filtered` is most extreme (lowest for ascending, highest for descending).
  // Merging carries the whole tuple rather than reducing each field
  // independently, so (ext_x, ext_y, ext_z) always names one real voxel and every
  // channel's sample is the value actually read at that voxel.
  float ext_x = -1.0f, ext_y = -1.0f;
  int ext_z = -1;
  float ext_filtered = 0.0f;

  // Reductions over the feature's PER-SLICE values, keyed "<quantity>_<reduction>"
  // (e.g. "area_mean", "bbox_w_max"). Unlike the field statistics above -- which
  // pool voxels, because a mean of per-slice means would weight by slice count
  // rather than area -- these describe how the feature's footprint varies from
  // slice to slice, which is the only way area/bbox mean anything under
  // mean/min/max/std. Empty unless `statistics.per_slice` asks for them.
  std::map<std::string, double> per_slice;
};

// The aggregated 3D master table: rows plus the flat per-channel plane they are
// measured on. `channels` is `rows.size() * schema.size()` cells, row-major, and
// `schema` names slot k -- the same (rows, ChannelStats, schema) triple the 2D
// side passes around.
struct GlobalFeatureTable {
  std::vector<GlobalFeatureStat> rows;
  msseg::ChannelStats channels;
  std::vector<msseg::ResolvedStatChannel> schema;
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
