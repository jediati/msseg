#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "msseg/graph/msc_graph.hpp"

namespace msseg {

// One node of an extremum -> saddle merge tree. Leaves are extrema (value = the
// extremum's function value); mergers are saddles (value = the saddle's value).
// For an ASCENDING tree the extrema are minima (index_dim 0) and the tree grows
// with increasing value; for a DESCENDING tree they are maxima (index_dim == the
// top cell dimension) and it grows with decreasing value. voxel_count is the
// summed basin voxel count of the subtree.
struct MergeNode {
  int id = -1;
  bool is_leaf = true;
  float value = 0.0f;
  std::int64_t voxel_count = 0;
  NodeId msc_node = -1;   // snapshot NodeId of the extremum or saddle
  std::vector<int> children;     // indices into MergeTree::nodes (mergers only)
};

struct MergeTree {
  std::vector<MergeNode> nodes;
  std::vector<int> roots;        // top-level node indices (>1 if disconnected)
  std::unordered_map<NodeId, int> leaf_of_min;  // extremum NodeId -> node index
  bool ascending = true;         // orientation (drives the relabel comparisons)
};

// Build the merge tree over the living complex: seed every extremum (index_dim
// == extremum_dim) as its own set, then add saddles (index_dim == saddle_dim) in
// value order (ascending value for an ascending tree, descending for a
// descending one), merging the disjoint extremum sets each connects.
// `extremum_counts` is indexed by snapshot NodeId (basin voxel count).
// Defaults reproduce the ascending minima->1-saddle tree (the cellseg 3D case).
MergeTree build_merge_tree(const MscGraph& graph,
                           const std::vector<std::int64_t>& extremum_counts,
                           bool ascending = true,
                           int extremum_dim = 0,
                           int saddle_dim = 1);

// Serialize as the FLAT JSON format {"nodes":[{id,type,value,voxel_count,
// node_id,children:[...]}], "roots":[...], "ascending":bool}. Flat (not nested)
// so a million-deep caterpillar tree can't overflow the parser.
std::string merge_tree_to_json(const MergeTree& tree);

// Branch decomposition + persistence/cut relabel. Each extremum-leaf owns a
// branch that dies at the merger where a deeper sibling first appears (its
// leaf-to-merge persistence = |merger value - leaf value|). A branch stays
// separate from the branch it dies into only when its death is a *barrier*:
// death is beyond `cut_threshold` (in the surviving direction) AND persistence
// >= `select_persistence`. Otherwise it is relabeled into (merged with) the
// deeper branch. Returns, per living extremum NodeId, the NodeId of the
// surviving branch's leaf extremum. Pass cut = -inf (ascending) / +inf
// (descending) to relabel by persistence only.
std::unordered_map<NodeId, NodeId> branch_survivors(
    const MergeTree& tree, float cut_threshold, float select_persistence);

// Relabel-then-cut: collapse every branch below `select_persistence` into its
// parent (branch_survivors), then number the surviving branches into regions.
// Returns extremum NodeId -> region id (contiguous 0..K-1, pre-order first-
// reference order). Also fills `region_voxels` (summed voxel count per region)
// when non-null.
std::unordered_map<NodeId, int> cut_regions(
    const MergeTree& tree, float cut_threshold, float select_persistence,
    std::vector<std::int64_t>* region_voxels = nullptr);

// Post-cut absorption (ascending / minima oriented; used by cellseg "cells").
// Every minimum whose VALUE is above the cut is merged into an adjacent CELL -- a
// non-background cut region whose deepest minimum lies below the cut -- via a
// marker-based watershed over the living min->1-saddle network. Returns min
// NodeId -> the deepest-min NodeId of its final cell.
std::unordered_map<NodeId, NodeId> absorb_above_cut_minima(
    const MscGraph& graph, const MergeTree& tree, float cut_threshold,
    float select_persistence);

}  // namespace msseg
