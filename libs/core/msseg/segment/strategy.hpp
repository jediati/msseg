#pragma once

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "msseg/compute/msc3d.hpp"
#include "msseg/graph/msc_graph.hpp"
#include "msseg/volume/types.hpp"

namespace msseg {

// Parameters for the segmentation stage. `strategy_name` selects a registered
// strategy; `extra` is an opaque per-strategy blob -- this is the iteration
// seam for graph-walking / local-segmentation experiments.
struct SegmentationParams {
  std::string strategy_name = "null";
  nlohmann::json extra = nlohmann::json::object();
};

// A segmentation strategy turns a (simplified) MS complex + the filtered
// volume into a label volume (kBackgroundLabel for non-feature voxels). New
// graph-walking approaches are added as new subclasses -- no plumbing changes.
class SegmentationStrategy {
 public:
  virtual ~SegmentationStrategy() = default;
  virtual LabelVolume segment(const MscGraph& graph, Msc3D& msc, const Volume& filtered,
                              const SegmentationParams& params) = 0;
};

}  // namespace msseg
