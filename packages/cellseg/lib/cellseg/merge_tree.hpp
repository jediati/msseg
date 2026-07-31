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

// Branch decomposition + persistence/cut relabel. Each living minimum-leaf owns
// a branch that dies at the merger where a deeper sibling first appears (its
// leaf-to-merge persistence = that merger's value minus the leaf value). A
// branch stays separate from the branch it dies into only when its death is a
// *barrier*: death value > `cut_threshold` AND persistence >=
// `select_persistence`. Otherwise it is relabeled into (merged with) the deeper
// branch. Returns, per living minimum NodeId, the NodeId of the surviving
// branch's leaf minimum (the deepest minimum of the feature it belongs to).
// Pass cut = -inf to relabel by persistence only (the "asc tree" simplification
// that finishes the persistence cancellation the GInt arc-count cap skips).
std::unordered_map<msseg::NodeId, msseg::NodeId> branch_survivors(
    const MergeTree& tree, float cut_threshold, float select_persistence);

// Relabel-then-cut: collapse every branch below `select_persistence` into its
// parent (branch_survivors), then number the surviving branches into regions.
// Returns min NodeId -> region id (contiguous 0..K-1, pre-order first-reference
// order). Also fills `region_voxels` (summed voxel count per region) when
// non-null. With select = -inf this reduces to the plain merge-tree cut.
std::unordered_map<msseg::NodeId, int> cut_regions(
    const MergeTree& tree, float cut_threshold, float select_persistence,
    std::vector<std::int64_t>* region_voxels = nullptr);

// Post-cut absorption: every minimum whose VALUE is above the cut is merged into
// an adjacent CELL -- a non-background cut region (branch_survivors + heaviest =
// background) whose deepest minimum lies below the cut. Adjacency is the LIVING
// min->1-saddle network (`graph`), NOT the merge-tree branch structure, so a
// min's cell can differ from the branch it merges into. Implemented as a
// marker-based watershed flood: seed from the below-cut minima of cell regions
// and grow across 1-saddles in increasing saddle value, so each above-cut min
// joins the cell reached via the lowest connecting saddle (nesting handled by
// the priority queue). Background is not a marker: an above-cut min joins the
// lowest reachable NON-background cell, and only keeps its cut region when no
// cell path exists. Returns min NodeId -> the deepest-min NodeId of its final
// cell (so the per-minimum palette colors it like the tree).
std::unordered_map<msseg::NodeId, msseg::NodeId> absorb_above_cut_minima(
    const msseg::MscGraph& graph, const MergeTree& tree, float cut_threshold,
    float select_persistence);

}  // namespace cellseg
