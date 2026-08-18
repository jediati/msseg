#pragma once

#include <vector>

#include "mscoupon/config.hpp"
#include "mscoupon/types.hpp"
#include "msseg/compute/msc2d.hpp"

namespace mscoupon {

std::vector<int> compute_msc_labels(const Image2D& filtered_image, const MscConfig& cfg);

// Per-slice segmentation via the merge-tree authoritative Msc2DPipeline (shared
// by the CLI and the GUI). Returns the feature-id-per-pixel labels at the
// configured persistence plus the per-feature statistics (on both `original` and
// `filtered`). This is the CLI's segmentation path so exported configs reproduce
// the viewer output.
struct SliceSegmentation {
  std::vector<int> labels;
  std::vector<msseg::Msc2DFeatureStat> features;
};

SliceSegmentation segment_slice_pipeline(const Image2D& original, const Image2D& filtered,
                                         const MscConfig& cfg, const msseg::StatsSpec& stats);

}  // namespace mscoupon
