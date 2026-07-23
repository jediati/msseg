#include "msseg/segment/registry.hpp"

#include <stdexcept>

#include "msseg/volume/types.hpp"

namespace msseg {
namespace {

// Fallback strategy: label every voxel as background. Useful as a baseline and
// for smoke-testing the pipeline plumbing before real strategies land.
class NullStrategy : public SegmentationStrategy {
 public:
  LabelVolume segment(const MscGraph&, Msc3D&, const Volume& filtered, const SegmentationParams&) override {
    LabelVolume out(filtered.dims());
    for (std::size_t i = 0; i < out.size(); ++i) out.data()[i] = kBackgroundLabel;
    return out;
  }
};

}  // namespace

std::unique_ptr<SegmentationStrategy> make_strategy(const std::string& name) {
  if (name == "null") {
    return std::make_unique<NullStrategy>();
  }
  // "basin" and future graph-walking strategies are registered in M3+.
  throw std::runtime_error("Unknown segmentation strategy: " + name);
}

}  // namespace msseg
