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

// Derived fields for a 2D feature statistic, as selected by `spec`.
//
// Always present: feature_id, area, bbox_w, bbox_h, min_x, max_x, min_y, max_y.
// Per enabled channel C ("base" / "filtered"), one field per enabled reduction:
// mean_C, min_C, max_C, std_C. With `spec.extremum`: the seeding critical point
// ext_x, ext_y, ext_base, ext_filtered.
//
// The row is rebuilt per feature on every persistence change and marshalled
// across the pybind boundary, so the spec is what keeps that cost proportional
// to what a workflow actually queries.
std::unordered_map<std::string, double> feature_row(const msseg::Msc2DFeatureStat& s,
                                                    const msseg::StatsSpec& spec);

// The field names feature_row() produces under `spec`, sorted. Drives config
// validation and the GUI's field dropdown, so there is no hand-kept mirror to
// drift out of sync.
std::vector<std::string> feature_fields(const msseg::StatsSpec& spec);

// True iff `field` is one of the names feature_row() produces under `spec`.
// Config validation uses this so a typo -- or a field the spec switched off --
// is rejected up front rather than silently excluding every feature (row_passes
// fails closed on an unknown field).
bool is_feature_field(const std::string& field, const msseg::StatsSpec& spec);

}  // namespace mscoupon
