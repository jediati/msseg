#pragma once

#include <string>

#include "cellseg/heavy_lift.hpp"
#include "cellseg/segment.hpp"

namespace cellseg {

// Stateful Phase-B facade over a primed CellState. Owns the heavy-lift result
// and caches the "living view" (snapshot + ascending labels + merge tree) at
// the currently selected persistence so thresholds can be re-applied cheaply.
// This is the single implementation shared by the CLI, the Python binding, and
// (via a config JSON) the viewer.
class CellPipeline {
 public:
  explicit CellPipeline(CellState state);

  float value_range() const { return state_.value_range; }
  float heavy_persistence() const { return state_.heavy_persistence; }
  float current_persistence() const { return current_persistence_; }

  // The filtered (blurred) volume the topology was computed on.
  const msseg::Volume& filtered() const { return state_.filtered; }

  // Re-select the persistence: re-snapshot, recompute ascending living labels +
  // voxel counts, and rebuild the merge tree. Cheap relative to the heavy lift.
  void select_persistence(float persistence);

  const LivingView& view();               // ensures a view exists first
  std::string merge_tree_json();          // nested JSON of the merge tree
  SegmentResult segment(float cut_threshold, float background_threshold);

  // Living ascending-manifold labels (minimum NodeId + 1 per voxel) at the
  // current persistence.
  const msseg::LabelVolume& ascending_labels();
  // Branch-decomposition ("asc tree") segmentation: ascending labels relabeled
  // by branch, collapsing every branch below the current persistence into its
  // parent. Each voxel carries its surviving branch's deepest-minimum NodeId + 1
  // (0 = background), so the per-minimum palette colors it like the tree.
  msseg::LabelVolume ascending_tree_labels();

  // "cells" segmentation (post-cut absorption): the relabel-then-cut regions
  // (at cut_threshold) with every above-cut minimum region-grown into the lowest
  // adjacent non-background cell over the living min->1-saddle network. Each
  // voxel carries its final cell's deepest-minimum NodeId + 1 (shared palette).
  // Cut-dependent -- refetch when the cut changes.
  msseg::LabelVolume cell_labels(float cut_threshold);
  // Absolute cancellation persistence per living node (aligned to view().graph
  // NodeIds); NaN where a node is never cancelled.
  std::vector<float> node_cancellation_persistence();

 private:
  void ensure_view();

  CellState state_;
  LivingView view_;
  float current_persistence_ = 0.0f;
  bool have_view_ = false;
};

}  // namespace cellseg
