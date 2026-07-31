#pragma once

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mscoupon/types.hpp"

namespace mscoupon {

// Accumulates cross-slice feature identity as a union-find over
// (slice_index, segment_id) nodes. Slices are added strictly in ascending slice
// order; a kept feature in slice N is linked to a kept feature in the previously
// added slice when they are 26-neighbor connected (a 3x3 stencil into the prior
// slice — in-plane connectivity is already handled by the 2D MSC component).
//
// Per-slice output (masks / per-slice CSVs) is never modified. finalize()
// resolves the union-find into deterministic global ids and produces the
// per-slice->global map plus an aggregated master table.
class SliceMatcher {
 public:
  // Add one slice's kept features and link them to the previously added slice.
  //   labels    per-pixel local-id image, row-major, size width*height
  //   keep_ids  local ids that passed the per-slice size threshold
  //   rows      per-segment statistics (only kept ids are consumed)
  void add_slice(const std::vector<int>& labels, int width, int height,
                 const std::unordered_set<int>& keep_ids, const std::vector<SegmentStat>& rows,
                 int slice_index);

  // Resolve the union-find, assign global ids in (slice_index, segment_id)
  // appearance order, and fill the per-slice->global map plus the aggregated
  // master table (sorted by voxel_count descending, tie -> global_id ascending).
  void finalize(std::vector<FeatureMapRow>& map_out, std::vector<GlobalFeatureStat>& table_out);

 private:
  struct NodePayload {
    int slice_index = 0;
    int segment_id = 0;
    std::size_t area = 0;
    double sum_value = 0.0;
    float min_value = 0.0f;
    float max_value = 0.0f;
    int min_x = 0;
    int min_y = 0;
    int max_x = 0;
    int max_y = 0;
  };

  static std::int64_t key_of(int slice_index, int segment_id) {
    return (static_cast<std::int64_t>(slice_index) << 32) |
           static_cast<std::int64_t>(static_cast<std::uint32_t>(segment_id));
  }

  int find(int x);
  void unite(int a, int b);
  int node_for(int slice_index, int segment_id) const;  // -1 if absent

  std::unordered_map<std::int64_t, int> index_;  // key(slice,id) -> node index
  std::vector<int> parent_;
  std::vector<int> rank_;
  std::vector<NodePayload> payload_;

  std::vector<int> prev_labels_;
  int prev_width_ = 0;
  int prev_height_ = 0;
  int prev_slice_index_ = -1;
  std::unordered_set<int> prev_keep_;
};

}  // namespace mscoupon
