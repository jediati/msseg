#include "cellseg/cell_pipeline.hpp"

#include <utility>

#include "cellseg/merge_tree.hpp"

namespace cellseg {

CellPipeline::CellPipeline(CellState state) : state_(std::move(state)) {}

void CellPipeline::select_persistence(float persistence) {
  state_.msc.select_persistence(persistence);
  view_.graph = state_.msc.snapshot();
  view_.asc_labels = state_.msc.living_labels(/*ascending=*/true, &view_.min_counts);
  view_.tree = build_merge_tree(view_.graph, view_.min_counts);
  current_persistence_ = persistence;
  have_view_ = true;
}

void CellPipeline::ensure_view() {
  if (!have_view_) select_persistence(state_.heavy_persistence);
}

const LivingView& CellPipeline::view() {
  ensure_view();
  return view_;
}

std::string CellPipeline::merge_tree_json() {
  ensure_view();
  return merge_tree_to_json(view_.tree);
}

SegmentResult CellPipeline::segment(float cut_threshold, float background_threshold) {
  ensure_view();
  return run_segmentation(state_.msc, state_.filtered, view_, cut_threshold, background_threshold);
}

}  // namespace cellseg
