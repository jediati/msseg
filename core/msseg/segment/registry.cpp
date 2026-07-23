#include "msseg/segment/registry.hpp"

#include <stdexcept>
#include <string>

#include "msseg/volume/types.hpp"

namespace msseg {
namespace {

// Fallback strategy: label every voxel as background. Useful as a baseline and
// for smoke-testing the pipeline plumbing.
class NullStrategy : public SegmentationStrategy {
 public:
  LabelVolume segment(const MscGraph&, Msc3D&, const Volume& filtered, const SegmentationParams&) override {
    LabelVolume out(filtered.dims());
    for (std::size_t i = 0; i < out.size(); ++i) out.data()[i] = kBackgroundLabel;
    return out;
  }
};

// Per-extremum basins: label each voxel by the ascending manifold of a minimum
// (default) or the descending manifold of a maximum, at the MSC's currently
// selected persistence. The first real graph-walking strategies will build on
// top of / alongside this.
class BasinLabelStrategy : public SegmentationStrategy {
 public:
  LabelVolume segment(const MscGraph&, Msc3D& msc, const Volume&, const SegmentationParams& params) override {
    const std::string manifold = params.extra.value("manifold", "ascending");
    const bool ascending = (manifold != "descending");
    return msc.basin_labels(ascending);
  }
};

}  // namespace

std::unique_ptr<SegmentationStrategy> make_strategy(const std::string& name) {
  if (name == "null") {
    return std::make_unique<NullStrategy>();
  }
  if (name == "basin") {
    return std::make_unique<BasinLabelStrategy>();
  }
  throw std::runtime_error("Unknown segmentation strategy: " + name);
}

}  // namespace msseg
