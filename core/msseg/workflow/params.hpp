#pragma once

#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace msseg {

// Parameters for the filter/transform stage. `operation` selects a diffg
// filter; `params` carries operation-specific keys (sigma, thresholds, ...).
struct FilterParams {
  std::string operation = "none";
  nlohmann::json params = nlohmann::json::object();
};

// Parameters for the 2D Morse-Smale compute + simplification + 2-manifold
// labeling (backed by MSCEER's GInt::Msc2D facade).
struct Msc2DParams {
  std::optional<float> persistence_absolute;
  std::optional<float> persistence_percent = 10.0f;
  std::string compute_algorithm = "serial";  // "serial" | "partitioned"
  bool accurate_ascending = true;
  bool accurate_descending = true;
  std::string manifold = "ascending";  // "ascending" | "descending"
  // Partition/thread count for the discrete gradient and (in "partitioned"
  // mode) the parallel MSC/hierarchy build. 0 => leave the msc_2d_lib default.
  int requested_parallelism = 0;
};

}  // namespace msseg
