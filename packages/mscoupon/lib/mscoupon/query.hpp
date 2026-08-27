#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "mscoupon/config.hpp"          // FeatureQuery
#include "mscoupon/types.hpp"           // GlobalFeatureStat
#include "msseg/compute/channel_stats.hpp"
#include "msseg/compute/msc2d.hpp"      // msseg::Msc2DFeatureStat
#include "msseg/workflow/stat_channels.hpp"

namespace mscoupon {

// Single-source feature-selection query evaluation and the per-feature
// statistics schema, shared by the CLI and the GUI (via pybind).
//
// The schema is COLUMNAR: names once, values as a flat row-major array. A
// twelve-channel scale-space stack is ~50 fields, and the table is rebuilt for
// every feature on every persistence change and marshalled to Python, so a
// per-feature string-keyed map would spend most of its time allocating strings.
// Names are resolved to column indices once (compile_queries) and everything
// downstream indexes.

// True iff `value` satisfies the predicate (op: lt/le/gt/ge/eq/between; between
// uses [value, value2]).
bool eval_predicate(double value, const FeatureQuery& query);

// True iff the row satisfies every query (AND-chain). A query naming a field the
// row does not contain fails (the feature is excluded), surfacing typos rather
// than silently keeping everything.
bool row_passes(const std::unordered_map<std::string, double>& row,
                const std::vector<FeatureQuery>& queries);

// Experimental shifted relevance. `f_m`/`f_s` are the feature's base min/max;
// floor/ceiling select the slice range used to shift both values.
double relevance_base_value(double f_m, double f_s, double floor, double ceiling);

// One column of the per-feature statistics table.
//
// `channel` and `reduction` are what the GUI builds its two-level
// [channel][reduction] pickers from, so nothing has to re-parse a name like
// "mean_blur_s0.7" -- and "min_x" is not mistaken for reduction "min" on a
// channel "x". Geometry columns carry an empty `channel`.
struct FeatureField {
  std::string name;        // "mean_base", "max_hess_largest_s1.5", "area"
  std::string channel;     // "base", "blur_s0.7", ... or "" for geometry
  std::string reduction;   // "mean" | "min" | "max" | "std" | "ext" | "relevance" | ""
};

// The columns produced under `spec`, in TABLE order: geometry first, then each
// resolved channel's enabled reductions in channel order, then the extremum
// block. That order is the CSV header order and the feature-table column order;
// there is exactly one ordering in the system.
std::vector<FeatureField> feature_schema(const msseg::StatsSpec& spec);

// The columns of the 3D master table: 3D identity + extent, then the same
// channel and extremum blocks feature_schema() uses, so a 2D field and its 3D
// counterpart always carry the same name. `per_slice` reductions are appended by
// the writer, since config names them rather than the channel set.
std::vector<FeatureField> global_feature_schema(const msseg::StatsSpec& spec);

// Just the names of feature_schema(spec), same order. Drives config validation
// and the GUI dropdown, so there is no hand-kept mirror to drift.
std::vector<std::string> feature_fields(const msseg::StatsSpec& spec);

// True iff `field` is one of the names feature_schema() produces under `spec`.
// Config validation uses this so a typo -- or a field the spec switched off --
// is rejected up front rather than silently excluding every feature (row_passes
// fails closed on an unknown field).
bool is_feature_field(const std::string& field, const msseg::StatsSpec& spec);

// A block of per-feature statistics: the schema once, then `n_rows` rows of
// `fields.size()` doubles, row-major.
struct FeatureTable {
  std::vector<FeatureField> fields;
  std::size_t n_rows = 0;
  std::vector<double> values;

  std::size_t n_cols() const { return fields.size(); }
  double at(std::size_t row, std::size_t col) const { return values[row * fields.size() + col]; }
  const double* row_data(std::size_t row) const { return values.data() + row * fields.size(); }
  // Column index of `name`, or -1. Linear -- for one-off lookups only, never in
  // a loop over rows; use compile_queries for that.
  int column(const std::string& name) const;
};

// Project a slice's living 2D features into the table. `channels` and `stats`
// come straight off the Msc2DPipeline; `stats.channels()` is the slot schema.
FeatureTable feature_table(const std::vector<msseg::Msc2DFeatureStat>& features,
                           const msseg::ChannelStats& channels,
                           const std::vector<msseg::ResolvedStatChannel>& channel_schema,
                           const msseg::StatsSpec& spec);

// One feature's row as a name -> value map. A convenience for callers that are
// not in a hot loop (tests, single-feature readouts); the CLI, the CSVs and the
// pybind boundary all use the columnar form directly.
std::unordered_map<std::string, double> feature_row(const FeatureTable& table, std::size_t row);

// Project the matcher's 3D master table into the columnar form, using
// global_feature_schema(). `relevance_base` is read from the row rather than
// recomputed: the slice-level floor/ceiling union lives in the matcher.
FeatureTable global_feature_table(const std::vector<GlobalFeatureStat>& rows,
                                  const msseg::ChannelStats& channels,
                                  const std::vector<msseg::ResolvedStatChannel>& channel_schema,
                                  const msseg::StatsSpec& spec);

// A query chain with each field name already resolved to a column of a
// particular table. A query naming a column the table does not have resolves to
// -1 and fails closed, exactly as the map-based row_passes does.
struct CompiledQueries {
  std::vector<int> columns;               // parallel to `queries`
  std::vector<FeatureQuery> queries;
  bool empty() const { return queries.empty(); }
};
CompiledQueries compile_queries(const FeatureTable& table,
                                const std::vector<FeatureQuery>& queries);

// True iff row `row` of `table` satisfies every compiled query.
bool row_passes(const FeatureTable& table, std::size_t row, const CompiledQueries& compiled);

}  // namespace mscoupon
