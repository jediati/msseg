#pragma once

#include <cstdint>
#include <vector>

#include "cellseg/merge_tree.hpp"
#include "msseg/compute/msc3d.hpp"
#include "msseg/graph/msc_graph.hpp"
#include "msseg/volume/types.hpp"

namespace cellseg {

// The "living view" of the complex at a chosen persistence, cached so Phase-B
// thresholds can be re-applied cheaply.
struct LivingView {
  msseg::MscGraph graph;
  msseg::LabelVolume asc_labels;         // ascending living labels (min NodeId + 1)
  std::vector<std::int64_t> min_counts;  // ascending voxel count per snapshot NodeId
  MergeTree tree;
};

struct SegmentResult {
  msseg::LabelVolume seg8;  // int32 carrying 8-bit flags (1|2|4|8) per voxel
  msseg::LabelVolume ids;   // int32 cell ids
};

// Steps 10-15: tree cut -> bridging/background analysis -> membrane regions ->
// masks -> the seg8 (bit-flag) and ids (cell-id) volumes. `msc` must still be
// at the same persistence/snapshot that produced `view`.
SegmentResult run_segmentation(msseg::Msc3D& msc, const msseg::Volume& filtered,
                               const LivingView& view, float cut_threshold,
                               float background_threshold);

}  // namespace cellseg
