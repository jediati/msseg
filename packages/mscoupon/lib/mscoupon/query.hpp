#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "mscoupon/config.hpp"       // FeatureQuery
#include "msseg/compute/msc2d.hpp"   // msseg::Msc2DFeatureStat

namespace mscoupon {

// Single-source feature-selection query evaluation, shared by the CLI and the
// GUI (via pybind). Queries are evaluated against a field-name -> value "row" so
// the same chain applies to 2D features and to Python-assembled 3D features.

// True iff `value` satisfies the predicate (op: lt/le/gt/ge/eq/between; between
// uses [value, value2]).
bool eval_predicate(double value, const FeatureQuery& query);

// True iff the row satisfies every query (AND-chain). A query naming a field the
// row does not contain fails (the feature is excluded), surfacing typos rather
// than silently keeping everything.
bool row_passes(const std::unordered_map<std::string, double>& row,
                const std::vector<FeatureQuery>& queries);

// Standard derived fields for a 2D feature statistic: area, mean_base,
// mean_filtered, min_base, max_base, std_base, min_filtered, max_filtered,
// std_filtered, bbox_w, bbox_h, feature_id.
std::unordered_map<std::string, double> feature_row(const msseg::Msc2DFeatureStat& s);

}  // namespace mscoupon
