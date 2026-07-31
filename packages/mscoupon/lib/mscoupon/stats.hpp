#pragma once

#include <unordered_set>
#include <vector>

#include "mscoupon/config.hpp"
#include "mscoupon/types.hpp"

namespace mscoupon {

SegmentTable compute_segment_table(const Image2D& source, const std::vector<int>& labels, int slice_index);
std::unordered_set<int> select_segment_ids(const SegmentTable& table, const SegmentKeepConfig& keep);
Mask2D build_mask(const std::vector<int>& labels, int width, int height, const std::unordered_set<int>& keep_ids);

}  // namespace mscoupon
