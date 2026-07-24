#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "msseg/graph/msc_graph.hpp"

namespace cellseg {

// One node of the minima -> 1-saddle merge tree. Leaves are minima (value =
// minimum's function value); mergers are 1-saddles (value = saddle's function
// value). voxel_count is the summed ascending-basin voxel count of the subtree.
struct MergeNode {
  int id = -1;
  bool is_leaf = true;
  float value = 0.0f;
  std::int64_t voxel_count = 0;
  msseg::NodeId msc_node = -1;   // snapshot NodeId of the minimum or 1-saddle
  std::vector<int> children;     // indices into MergeTree::nodes (mergers only)
};

struct MergeTree {
  std::vector<MergeNode> nodes;
  std::vector<int> roots;        // top-level node indices (>1 if disconnected)
  std::unordered_map<msseg::NodeId, int> leaf_of_min;  // min NodeId -> node index
};

// Build the merge tree over the living complex: seed every minimum as its own
// set, then add 1-saddles in ascending value order, merging the two disjoint
// minima sets each connects. `min_counts` is indexed by snapshot NodeId (from
// Msc3D::living_labels).
MergeTree build_merge_tree(const msseg::MscGraph& graph,
                           const std::vector<std::int64_t>& min_counts);

// Serialize as a nested JSON tree: {"roots": [ {id,type,value,voxel_count,
// node_id,children:[...]}, ... ]}. Easy to load in matplotlib / the viewer.
std::string merge_tree_to_json(const MergeTree& tree);

// Apply a cut: every merger whose saddle value exceeds `cut_threshold` is cut,
// splitting its children into separate regions. Returns min NodeId -> region id
// (contiguous 0..K-1). Also fills `region_voxels` (summed voxel count per
// region) when non-null.
std::unordered_map<msseg::NodeId, int> cut_regions(
    const MergeTree& tree, float cut_threshold,
    std::vector<std::int64_t>* region_voxels = nullptr);

}  // namespace cellseg
