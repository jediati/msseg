#pragma once

// JSON -> options parsing for the three intensity measures.
//
// Lives in the library rather than in the pybind layer so the CLI (via the
// `normalize` filter op) and the Python bindings parse exactly the same keys.
// Each parser accepts either its named block or a bare options object, so
// {"gmm": {...}} and {...} are equivalent.

#include <nlohmann/json.hpp>

#include "mscoupon/gmm.hpp"
#include "mscoupon/histogram_peaks.hpp"
#include "mscoupon/region_measure.hpp"

namespace mscoupon {

// Accepts an optional "preset" ("two_gaussian" | "measure") to seed the
// defaults from one of the two analysis scripts; every other key then overrides
// individual fields.
GmmOptions parse_gmm_options(const nlohmann::json& cfg);

HistogramOptions parse_histogram_options(const nlohmann::json& cfg);

RegionOptions parse_region_options(const nlohmann::json& cfg);

// Parse {"rows": "250:350", "cols": "740:840"}.
Rect parse_rect_json(const nlohmann::json& node);

}  // namespace mscoupon
