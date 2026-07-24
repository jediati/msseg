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

 private:
  void ensure_view();

  CellState state_;
  LivingView view_;
  float current_persistence_ = 0.0f;
  bool have_view_ = false;
};

}  // namespace cellseg
