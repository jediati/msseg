#include "cellseg/cell_pipeline.hpp"

#include <limits>
#include <utility>

#include "cellseg/merge_tree.hpp"

namespace cellseg {

CellPipeline::CellPipeline(CellState state) : state_(std::move(state)) {}

void CellPipeline::select_persistence(float persistence) {
  state_.msc.select_persistence(persistence);
  view_.graph = state_.msc.snapshot();
  view_.asc_labels = state_.msc.living_labels(/*ascending=*/true, &view_.min_counts);
  view_.tree = build_merge_tree(view_.graph, view_.min_counts);
  view_.persistence = persistence;
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

const msseg::LabelVolume& CellPipeline::ascending_labels() {
  ensure_view();
  return view_.asc_labels;
}

msseg::LabelVolume CellPipeline::ascending_tree_labels() {
  ensure_view();
  // Branch decomposition relabeled by persistence only (no cut): fold every
  // branch below the current select persistence into its parent -- the
  // simplification the GInt arc-count cap skips. Each voxel carries its
  // surviving branch's deepest-minimum NodeId (+1), so the same per-minimum
  // palette colors it identically to the tree.
  const float ninf = -std::numeric_limits<float>::infinity();
  const std::unordered_map<msseg::NodeId, msseg::NodeId> survivors =
      branch_survivors(view_.tree, ninf, current_persistence_);

  const msseg::LabelVolume& asc = view_.asc_labels;
  msseg::LabelVolume out(asc.dims());
  const std::int32_t* src = asc.data();
  std::int32_t* dst = out.data();
  for (std::size_t i = 0; i < asc.size(); ++i) {
    const std::int32_t lab = src[i];
    if (lab <= 0) {
      dst[i] = 0;
      continue;
    }
    const msseg::NodeId mn = static_cast<msseg::NodeId>(lab - 1);
    const auto it = survivors.find(mn);
    const msseg::NodeId sv = (it != survivors.end()) ? it->second : mn;
    dst[i] = static_cast<std::int32_t>(sv) + 1;
  }
  return out;
}

std::vector<float> CellPipeline::node_cancellation_persistence() {
  ensure_view();
  return state_.msc.node_cancellation_persistence();
}

}  // namespace cellseg
