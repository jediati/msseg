#pragma once

// The merge-tree data structure + operations were promoted to the portable core
// (msseg::, libs/core/msseg/graph/merge_tree.hpp) so the 2D mscoupon pipeline can
// share them. cellseg keeps using the unqualified names via these re-exports;
// its callers (cell_pipeline.cpp, segment.cpp) are unchanged. The ascending
// (minima -> 1-saddle) defaults reproduce the original 3D behavior exactly.

#include "msseg/graph/merge_tree.hpp"

namespace cellseg {

using msseg::MergeNode;
using msseg::MergeTree;
using msseg::build_merge_tree;
using msseg::merge_tree_to_json;
using msseg::branch_survivors;
using msseg::cut_regions;
using msseg::absorb_above_cut_minima;

}  // namespace cellseg
