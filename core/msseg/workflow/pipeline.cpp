#include "msseg/workflow/pipeline.hpp"

#include "msseg/segment/registry.hpp"

namespace msseg {

WorkflowResult Pipeline::run(const Volume& input, const WorkflowParams& params) {
  WorkflowResult result;

  // Stage 1: filter/transform (live).
  result.filtered = apply_filter(input, params.filter);

  // Stage 2-3: discrete gradient + MSC + simplification. Implemented in M3.
  Msc3D msc;
  msc.build(result.filtered, params.msc);
  msc.compute(params.msc);
  if (params.simplify.persistence_absolute.has_value()) {
    msc.select_persistence(*params.simplify.persistence_absolute);
  }
  result.graph = msc.snapshot();

  // Stage 4: graph-walking segmentation.
  auto strategy = make_strategy(params.segmentation.strategy_name);
  result.labels = strategy->segment(result.graph, msc, result.filtered, params.segmentation);

  return result;
}

}  // namespace msseg
