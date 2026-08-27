#include "mscoupon/query.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mscoupon {
namespace {

// The aggregate reductions, in the order they appear in a channel's column
// block. One place decides this, so the schema, the table and the CSV header
// cannot disagree about it.
struct ReductionSlot {
  const char* name;
  bool msseg::StatsSpec::*enabled;
};
const ReductionSlot kReductions[] = {
    {"mean", &msseg::StatsSpec::mean},
    {"min", &msseg::StatsSpec::min},
    {"max", &msseg::StatsSpec::max},
    {"std", &msseg::StatsSpec::std},
};

double mean_of(const msseg::ChannelAccum& a, double area) {
  return area > 0 ? a.sum / area : 0.0;
}

double std_of(const msseg::ChannelAccum& a, double area) {
  if (area <= 0) return 0.0;
  const double m = a.sum / area;
  const double var = a.sumsq / area - m * m;
  return var > 0.0 ? std::sqrt(var) : 0.0;
}

double reduction_value(const char* reduction, const msseg::ChannelAccum& a, double area) {
  if (reduction[0] == 'm' && reduction[1] == 'e') return mean_of(a, area);   // mean
  if (reduction[0] == 'm' && reduction[1] == 'i') return a.min;             // min
  if (reduction[0] == 'm' && reduction[1] == 'a') return a.max;             // max
  return std_of(a, area);                                                   // std
}

}  // namespace

bool eval_predicate(double value, const FeatureQuery& query) {
  const double t = query.value;
  if (query.op == "lt") return value < t;
  if (query.op == "le") return value <= t;
  if (query.op == "gt") return value > t;
  if (query.op == "ge") return value >= t;
  if (query.op == "eq") return value == t;
  if (query.op == "between") return value >= t && value <= query.value2;
  return false;  // unknown op: config validation rejects these, so be conservative
}

bool row_passes(const std::unordered_map<std::string, double>& row,
                const std::vector<FeatureQuery>& queries) {
  for (const auto& q : queries) {
    const auto it = row.find(q.field);
    if (it == row.end()) return false;  // unknown field -> exclude (surfaces typos)
    if (!eval_predicate(it->second, q)) return false;
  }
  return true;
}

double relevance_base_value(double f_m, double f_s, double floor, double ceiling) {
  const double numerator = f_s - f_m;
  const double m_shift = f_m + (ceiling - floor);
  if (m_shift == 0.0) {
    return numerator == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
  }
  return numerator / m_shift;
}

namespace {

// The channel block: for each resolved channel, one column per enabled
// reduction. Shared by the 2D and 3D schemas so the two cannot disagree about
// which columns exist or in what order.
void append_channel_columns(std::vector<FeatureField>& out,
                            const std::vector<msseg::ResolvedStatChannel>& channels,
                            const msseg::StatsSpec& spec) {
  for (const auto& c : channels) {
    for (const auto& r : kReductions) {
      if (!(spec.*(r.enabled))) continue;
      out.push_back(FeatureField{std::string(r.name) + "_" + c.name, c.name, r.name});
    }
    // `relevance` is defined on the base channel only -- it is a contrast
    // against slice-level base percentiles, which no derived channel has.
    if (c.name == "base" && spec.relevance) {
      out.push_back(FeatureField{"relevance_base", "base", "relevance"});
    }
  }
}

// The extremum block: the seed's position, its value on the topology field, and
// its value on every measurement channel. `with_z` adds the slice index, which
// only a 3D feature has.
void append_ext_columns(std::vector<FeatureField>& out,
                        const std::vector<msseg::ResolvedStatChannel>& channels,
                        const msseg::StatsSpec& spec, bool with_z) {
  if (!spec.extremum) return;
  // The seed's POSITION is geometry, not a reduction of any channel: there is no
  // "ext of channel x". Leaving the reduction empty is what keeps a GUI picker
  // from offering a (geometry, ext) pair that names no field.
  out.push_back(FeatureField{"ext_x", "", ""});
  out.push_back(FeatureField{"ext_y", "", ""});
  if (with_z) out.push_back(FeatureField{"ext_z", "", ""});
  // One column per measurement channel, then the topology field. `ext_filtered`
  // exists whether or not `filtered` is a measurement channel -- it is what
  // LOCATES the seed -- so it is emitted once, here, and the loop skips it to
  // avoid producing the same name twice. Emitting it LAST is what keeps a
  // base-only spec's column order (..., ext_base, ext_filtered) byte-identical
  // to what this table looked like before derived channels existed.
  for (const auto& c : channels) {
    if (c.name == "filtered") continue;
    out.push_back(FeatureField{"ext_" + c.name, c.name, "ext"});
  }
  out.push_back(FeatureField{"ext_filtered", "filtered", "ext"});
}

}  // namespace

std::vector<FeatureField> feature_schema(const msseg::StatsSpec& spec) {
  const auto channels = msseg::resolve_stat_channels(spec);
  std::vector<FeatureField> out;
  // Geometry: always present, independent of any channel.
  for (const char* n : {"feature_id", "area", "bbox_w", "bbox_h", "min_x", "max_x", "min_y",
                        "max_y"}) {
    out.push_back(FeatureField{n, "", ""});
  }
  append_channel_columns(out, channels, spec);
  append_ext_columns(out, channels, spec, /*with_z=*/false);
  return out;
}

std::vector<FeatureField> global_feature_schema(const msseg::StatsSpec& spec) {
  const auto channels = msseg::resolve_stat_channels(spec);
  std::vector<FeatureField> out;
  // 3D identity and extent. `per_slice` reductions are appended by the writer,
  // since they are named by config rather than by the channel set.
  for (const char* n : {"global_id", "voxel_count", "num_slices", "first_slice", "last_slice",
                        "min_x", "min_y", "min_z", "max_x", "max_y", "max_z"}) {
    out.push_back(FeatureField{n, "", ""});
  }
  append_channel_columns(out, channels, spec);
  append_ext_columns(out, channels, spec, /*with_z=*/true);
  return out;
}

std::vector<std::string> feature_fields(const msseg::StatsSpec& spec) {
  const auto schema = feature_schema(spec);
  std::vector<std::string> names;
  names.reserve(schema.size());
  for (const auto& f : schema) names.push_back(f.name);
  return names;
}

bool is_feature_field(const std::string& field, const msseg::StatsSpec& spec) {
  for (const auto& f : feature_schema(spec)) {
    if (f.name == field) return true;
  }
  return false;
}

int FeatureTable::column(const std::string& name) const {
  for (std::size_t i = 0; i < fields.size(); ++i) {
    if (fields[i].name == name) return static_cast<int>(i);
  }
  return -1;
}

FeatureTable feature_table(const std::vector<msseg::Msc2DFeatureStat>& features,
                           const msseg::ChannelStats& channels,
                           const std::vector<msseg::ResolvedStatChannel>& channel_schema,
                           const msseg::StatsSpec& spec) {
  FeatureTable table;
  table.fields = feature_schema(spec);
  table.n_rows = features.size();
  const std::size_t n_cols = table.fields.size();
  table.values.assign(table.n_rows * n_cols, 0.0);

  for (std::size_t r = 0; r < features.size(); ++r) {
    const msseg::Msc2DFeatureStat& s = features[r];
    const double area = static_cast<double>(s.area);
    double* row = table.values.data() + r * n_cols;
    std::size_t col = 0;

    row[col++] = static_cast<double>(s.feature_id);
    row[col++] = area;
    row[col++] = static_cast<double>(s.max_x - s.min_x + 1);
    row[col++] = static_cast<double>(s.max_y - s.min_y + 1);
    row[col++] = static_cast<double>(s.min_x);
    row[col++] = static_cast<double>(s.max_x);
    row[col++] = static_cast<double>(s.min_y);
    row[col++] = static_cast<double>(s.max_y);

    for (std::size_t k = 0; k < channel_schema.size(); ++k) {
      const msseg::ChannelAccum& a = channels.cell(r, k);
      for (const auto& red : kReductions) {
        if (!(spec.*(red.enabled))) continue;
        row[col++] = reduction_value(red.name, a, area);
      }
      if (channel_schema[k].name == "base" && spec.relevance) {
        row[col++] = relevance_base_value(a.min, a.max, s.base_relevance_floor,
                                          s.base_relevance_ceiling);
      }
    }

    if (spec.extremum) {
      row[col++] = s.ext_x;
      row[col++] = s.ext_y;
      for (std::size_t k = 0; k < channel_schema.size(); ++k) {
        if (channel_schema[k].name == "filtered") continue;
        row[col++] = channels.ext(r, k);
      }
      row[col++] = s.ext_filtered;
    }
  }
  return table;
}

FeatureTable global_feature_table(const std::vector<GlobalFeatureStat>& rows,
                                  const msseg::ChannelStats& channels,
                                  const std::vector<msseg::ResolvedStatChannel>& channel_schema,
                                  const msseg::StatsSpec& spec) {
  FeatureTable table;
  table.fields = global_feature_schema(spec);
  table.n_rows = rows.size();
  const std::size_t n_cols = table.fields.size();
  table.values.assign(table.n_rows * n_cols, 0.0);

  for (std::size_t r = 0; r < rows.size(); ++r) {
    const GlobalFeatureStat& s = rows[r];
    // Reductions over a channel are VOXEL-pooled, so the divisor is the 3D
    // feature's voxel count, not its slice count.
    const double area = static_cast<double>(s.voxel_count);
    double* row = table.values.data() + r * n_cols;
    std::size_t col = 0;

    row[col++] = static_cast<double>(s.global_id);
    row[col++] = static_cast<double>(s.voxel_count);
    row[col++] = static_cast<double>(s.num_slices);
    row[col++] = static_cast<double>(s.first_slice);
    row[col++] = static_cast<double>(s.last_slice);
    row[col++] = static_cast<double>(s.min_x);
    row[col++] = static_cast<double>(s.min_y);
    row[col++] = static_cast<double>(s.min_z);
    row[col++] = static_cast<double>(s.max_x);
    row[col++] = static_cast<double>(s.max_y);
    row[col++] = static_cast<double>(s.max_z);

    for (std::size_t k = 0; k < channel_schema.size(); ++k) {
      const msseg::ChannelAccum& a = channels.cell(r, k);
      for (const auto& red : kReductions) {
        if (!(spec.*(red.enabled))) continue;
        row[col++] = reduction_value(red.name, a, area);
      }
      // relevance_base is precomputed by the matcher, which is where the
      // slice-level floor/ceiling union lives.
      if (channel_schema[k].name == "base" && spec.relevance) row[col++] = s.relevance_base;
    }

    if (spec.extremum) {
      row[col++] = s.ext_x;
      row[col++] = s.ext_y;
      row[col++] = static_cast<double>(s.ext_z);
      for (std::size_t k = 0; k < channel_schema.size(); ++k) {
        if (channel_schema[k].name == "filtered") continue;
        row[col++] = channels.ext(r, k);
      }
      row[col++] = s.ext_filtered;
    }
  }
  return table;
}

std::unordered_map<std::string, double> feature_row(const FeatureTable& table, std::size_t row) {
  std::unordered_map<std::string, double> out;
  out.reserve(table.fields.size());
  for (std::size_t c = 0; c < table.fields.size(); ++c) {
    out.emplace(table.fields[c].name, table.at(row, c));
  }
  return out;
}

CompiledQueries compile_queries(const FeatureTable& table,
                                const std::vector<FeatureQuery>& queries) {
  CompiledQueries out;
  out.queries = queries;
  out.columns.reserve(queries.size());
  for (const auto& q : queries) out.columns.push_back(table.column(q.field));
  return out;
}

bool row_passes(const FeatureTable& table, std::size_t row, const CompiledQueries& compiled) {
  for (std::size_t i = 0; i < compiled.queries.size(); ++i) {
    const int col = compiled.columns[i];
    if (col < 0) return false;  // unknown field -> exclude (surfaces typos)
    if (!eval_predicate(table.at(row, static_cast<std::size_t>(col)), compiled.queries[i])) {
      return false;
    }
  }
  return true;
}

}  // namespace mscoupon
