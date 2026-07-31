#include "msseg/workflow/pipeline.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>

#include "msseg/segment/registry.hpp"

namespace msseg {
namespace {

float value_range(const Volume& v) {
  float lo = std::numeric_limits<float>::max();
  float hi = std::numeric_limits<float>::lowest();
  for (std::size_t i = 0; i < v.size(); ++i) {
    lo = std::min(lo, v.data()[i]);
    hi = std::max(hi, v.data()[i]);
  }
  return (v.size() == 0) ? 0.0f : (hi - lo);
}

}  // namespace

WorkflowResult Pipeline::run(const Volume& input, const WorkflowParams& params) {
  WorkflowResult result;

  // Stage 1: filter/transform.
  result.filtered = apply_filter(input, params.filter);

  // Stages 2-3: discrete gradient + MSC + simplification hierarchy.
  Msc3D msc;
  msc.build(result.filtered, params.msc);
  msc.compute(params.msc);

  // Resolve the persistence threshold (absolute wins; else percent-of-range).
  float persistence = 0.0f;
  if (params.simplify.persistence_absolute.has_value()) {
    persistence = *params.simplify.persistence_absolute;
  } else if (params.simplify.persistence_percent.has_value()) {
    persistence = value_range(result.filtered) * (*params.simplify.persistence_percent / 100.0f);
  }
  msc.select_persistence(persistence);
  result.graph = msc.snapshot();

  // Stage 4: graph-walking segmentation.
  auto strategy = make_strategy(params.segmentation.strategy_name);
  result.labels = strategy->segment(result.graph, msc, result.filtered, params.segmentation);

  return result;
}

WorkflowParams parse_workflow(const nlohmann::json& spec) {
  WorkflowParams p;

  if (spec.contains("filter")) {
    const auto& f = spec.at("filter");
    p.filter.operation = f.value("operation", p.filter.operation);
    if (f.contains("params")) p.filter.params = f.at("params");
  }

  if (spec.contains("gradient")) {
    const auto& g = spec.at("gradient");
    const std::string mode = g.value("mode", "robins_noalloc");
    if (mode == "grad_file") {
      p.msc.gradient_mode = Msc3DParams::GradientMode::GradFile;
      p.msc.grad_file_path = g.value("path", "");
    } else {
      p.msc.gradient_mode = Msc3DParams::GradientMode::RobinsNoalloc;
    }
    p.msc.build_arcs = g.value("build_arcs", p.msc.build_arcs);
    p.msc.build_arc_geometry = g.value("build_arc_geometry", p.msc.build_arc_geometry);
  }

  if (spec.contains("simplify")) {
    const auto& s = spec.at("simplify");
    p.simplify.persistence_absolute.reset();
    p.simplify.persistence_percent.reset();
    if (s.contains("persistence_absolute")) {
      p.simplify.persistence_absolute = s.at("persistence_absolute").get<float>();
    } else if (s.contains("persistence_percent")) {
      p.simplify.persistence_percent = s.at("persistence_percent").get<float>();
    }
  }

  if (spec.contains("segmentation")) {
    const auto& sg = spec.at("segmentation");
    p.segmentation.strategy_name = sg.value("strategy", p.segmentation.strategy_name);
    p.segmentation.extra = sg;  // strategy-specific keys (e.g. "manifold")
  }

  return p;
}

}  // namespace msseg
