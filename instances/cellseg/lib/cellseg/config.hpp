#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace cellseg {

// --- Phase A (heavy lift) ---------------------------------------------------
struct HeavyLiftConfig {
  std::filesystem::path input_path;  // "<name>_XxYxZ.raw" (x fastest, float32 LE)
  // Optional explicit dims; used when the filename does not encode _XxYxZ or a
  // "<path>.dat" sidecar is absent.
  std::optional<std::size_t> dim_x;
  std::optional<std::size_t> dim_y;
  std::optional<std::size_t> dim_z;

  // Filter: Gaussian blur only (median is done upstream in Python for now).
  float blur_sigma = 2.0f;

  // Gradient: MSCEER OnDemandAccurate; both flags false == steepest descent.
  bool accurate_ascending_3m = false;
  bool accurate_descending_3m = false;
  float presimp_threshold = 0.0f;
  float integration_error = 0.01f;
  float gradient_threshold = 0.0f;
  int integration_max_iter = 1000;

  // Simplification persistence: absolute wins; else percent-of-range (5% def).
  std::optional<float> persistence_absolute;
  std::optional<float> persistence_percent = 5.0f;
};

// --- Phase B (threshold tuning) --------------------------------------------
struct PhaseBConfig {
  // Persistence selection for the living MSC view (step 7). If unset, the
  // heavy-lift persistence is reused.
  std::optional<float> persistence_selection;
  // Merge-tree cut: mergers whose 1-saddle value exceeds this are cut (step 9).
  float cut_threshold = 0.0f;
  // Intensity threshold on the filtered volume for the fluorescent-membrane /
  // background mask (step 12).
  float background_threshold = 0.0f;
};

struct OutputConfig {
  std::filesystem::path seg8_path = "cellseg_seg8.raw";        // 8-bit bit-flags
  std::filesystem::path ids_path = "cellseg_ids.raw";          // int32 cell ids
  std::filesystem::path merge_tree_path = "cellseg_merge_tree.json";
};

struct AppConfig {
  HeavyLiftConfig heavy;
  PhaseBConfig phase_b;
  OutputConfig output;
  bool dry_run = false;
};

struct CliOptions {
  std::optional<std::filesystem::path> config_path;
  std::optional<std::filesystem::path> input_override;
  bool dry_run = false;
};

CliOptions parse_cli(int argc, char** argv);
AppConfig load_config(const CliOptions& cli);
void validate_config(const AppConfig& cfg);

// Parse "<name>_XxYxZ.raw" -> (X, Y, Z). Returns false if the pattern is absent.
bool parse_dims_from_filename(const std::filesystem::path& path, std::size_t& x,
                              std::size_t& y, std::size_t& z);

// Build the config from a parsed JSON object (also used by the Python binding).
AppConfig config_from_json(const nlohmann::json& root);

}  // namespace cellseg
