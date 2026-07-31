#pragma once

#include <memory>
#include <string>

#include "msseg/segment/strategy.hpp"

namespace msseg {

// Look up and construct a registered segmentation strategy by name. Throws if
// the name is unknown. Built-in strategies: "null" (all background) and
// "basin" (per-minimum ascending/descending manifold basins; implemented in M3).
std::unique_ptr<SegmentationStrategy> make_strategy(const std::string& name);

}  // namespace msseg
