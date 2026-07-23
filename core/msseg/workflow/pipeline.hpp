#pragma once

#include <optional>

#include <nlohmann/json.hpp>

#include "msseg/compute/msc3d.hpp"
#include "msseg/filter/filter_stage.hpp"
#include "msseg/graph/msc_graph.hpp"
#include "msseg/segment/strategy.hpp"
#include "msseg/volume/types.hpp"
#include "msseg/workflow/params.hpp"

namespace msseg {

struct SimplifyParams {
  std::optional<float> persistence_absolute;
  std::optional<float> persistence_percent = 10.0f;
};

// Full 3D workflow specification: transform -> gradient/MSC -> simplify ->
// graph-walking segmentation.
struct WorkflowParams {
  FilterParams filter;
  Msc3DParams msc;
  SimplifyParams simplify;
  SegmentationParams segmentation;
};

// Intermediates are kept alive so the viewer can display each stage.
struct WorkflowResult {
  Volume filtered;
  MscGraph graph;
  LabelVolume labels;
};

class Pipeline {
 public:
  // Run the full workflow over a volume: filter -> 3D discrete gradient / MSC
  // -> simplify (persistence) -> graph-walking segmentation.
  static WorkflowResult run(const Volume& input, const WorkflowParams& params);
};

// Parse a JSON workflow description into WorkflowParams. Schema:
//   {
//     "filter":       {"operation": "...", "params": {...}},
//     "gradient":     {"mode": "robins_noalloc" | "grad_file", "path": "..."},
//     "simplify":     {"persistence_absolute": x}  or  {"persistence_percent": p},
//     "segmentation": {"strategy": "basin" | "null", ...strategy-specific...}
//   }
// All sections are optional; omitted sections take their struct defaults.
WorkflowParams parse_workflow(const nlohmann::json& spec);

}  // namespace msseg
