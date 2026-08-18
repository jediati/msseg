#include "mscoupon/query.hpp"

#include <algorithm>
#include <cmath>

namespace mscoupon {

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

std::unordered_map<std::string, double> feature_row(const msseg::Msc2DFeatureStat& s,
                                                    const msseg::StatsSpec& spec) {
  const double area = static_cast<double>(s.area);
  const auto mean = [&](double sum) { return area > 0 ? sum / area : 0.0; };
  const auto stddev = [&](double sum, double sumsq) {
    if (area <= 0) return 0.0;
    const double m = sum / area;
    const double var = sumsq / area - m * m;
    return var > 0.0 ? std::sqrt(var) : 0.0;
  };

  std::unordered_map<std::string, double> row{
      {"feature_id", static_cast<double>(s.feature_id)},
      {"area", area},
      {"bbox_w", static_cast<double>(s.max_x - s.min_x + 1)},
      {"bbox_h", static_cast<double>(s.max_y - s.min_y + 1)},
      // Raw bbox corners (used by the Python 3D assembly to build 3D bounding
      // boxes; also queryable if a user wants position constraints).
      {"min_x", static_cast<double>(s.min_x)},
      {"max_x", static_cast<double>(s.max_x)},
      {"min_y", static_cast<double>(s.min_y)},
      {"max_y", static_cast<double>(s.max_y)},
  };

  const auto add_channel = [&](const char* suffix, double sum, double sumsq, float lo, float hi) {
    if (spec.mean) row["mean_" + std::string(suffix)] = mean(sum);
    if (spec.min) row["min_" + std::string(suffix)] = lo;
    if (spec.max) row["max_" + std::string(suffix)] = hi;
    if (spec.std) row["std_" + std::string(suffix)] = stddev(sum, sumsq);
  };
  if (spec.base_channel) add_channel("base", s.base_sum, s.base_sumsq, s.base_min, s.base_max);
  if (spec.filtered_channel)
    add_channel("filtered", s.filt_sum, s.filt_sumsq, s.filt_min, s.filt_max);

  // The seeding critical point (minimum for ascending manifolds, maximum for
  // descending) and the two channels sampled there. `ext_base` is what separates
  // a real void -- whose well bottom is genuinely dark -- from a shallow dip
  // inside metal that happens to have a similar mean.
  if (spec.extremum) {
    row["ext_x"] = s.ext_x;
    row["ext_y"] = s.ext_y;
    row["ext_base"] = s.ext_base;
    row["ext_filtered"] = s.ext_filtered;
  }
  return row;
}

std::vector<std::string> feature_fields(const msseg::StatsSpec& spec) {
  // Derived from feature_row itself so there is no second list to drift.
  const auto schema = feature_row(msseg::Msc2DFeatureStat{}, spec);
  std::vector<std::string> names;
  names.reserve(schema.size());
  for (const auto& [k, v] : schema) names.push_back(k);
  std::sort(names.begin(), names.end());
  return names;
}

bool is_feature_field(const std::string& field, const msseg::StatsSpec& spec) {
  return feature_row(msseg::Msc2DFeatureStat{}, spec).count(field) != 0;
}

}  // namespace mscoupon
