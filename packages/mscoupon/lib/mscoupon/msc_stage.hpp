#pragma once

#include <vector>

#include "mscoupon/config.hpp"
#include "mscoupon/types.hpp"
#include "msseg/compute/msc2d.hpp"
#include "msseg/filter/filter_stage.hpp"

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
  // Per-feature per-channel aggregates, in lockstep with `features`; `channels`
  // is the slot schema for both.
  msseg::ChannelStats feature_channels;
  std::vector<msseg::ResolvedStatChannel> channels;
  float base_relevance_floor = 0.0f;
  float base_relevance_ceiling = 0.0f;
};

// `base`/`filtered` are the slice's two rasters already in diffg form, and
// `bank` is its measurement channels -- both built once by the caller and reused
// by the connected-component stage, so neither the conversion nor the
// scale-space traversal happens twice per slice.
SliceSegmentation segment_slice_pipeline(const diffg::Image<float>& base,
                                         const diffg::Image<float>& filtered,
                                         const MscConfig& cfg, const msseg::StatsSpec& stats,
                                         const msseg::StatChannelBank& bank);

}  // namespace mscoupon
