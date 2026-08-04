#include "mscoupon/query.hpp"

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

std::unordered_map<std::string, double> feature_row(const msseg::Msc2DFeatureStat& s) {
  const double area = static_cast<double>(s.area);
  const auto mean = [&](double sum) { return area > 0 ? sum / area : 0.0; };
  const auto stddev = [&](double sum, double sumsq) {
    if (area <= 0) return 0.0;
    const double m = sum / area;
    const double var = sumsq / area - m * m;
    return var > 0.0 ? std::sqrt(var) : 0.0;
  };
  return {
      {"feature_id", static_cast<double>(s.feature_id)},
      {"area", area},
      {"mean_base", mean(s.base_sum)},
      {"mean_filtered", mean(s.filt_sum)},
      {"min_base", s.base_min},
      {"max_base", s.base_max},
      {"std_base", stddev(s.base_sum, s.base_sumsq)},
      {"min_filtered", s.filt_min},
      {"max_filtered", s.filt_max},
      {"std_filtered", stddev(s.filt_sum, s.filt_sumsq)},
      {"bbox_w", static_cast<double>(s.max_x - s.min_x + 1)},
      {"bbox_h", static_cast<double>(s.max_y - s.min_y + 1)},
      // Raw bbox corners (used by the Python 3D assembly to build 3D bounding
      // boxes; also queryable if a user wants position constraints).
      {"min_x", static_cast<double>(s.min_x)},
      {"max_x", static_cast<double>(s.max_x)},
      {"min_y", static_cast<double>(s.min_y)},
      {"max_y", static_cast<double>(s.max_y)},
  };
}

}  // namespace mscoupon
