#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace mscoupon {

struct InputConfig {
  std::filesystem::path folder;
  std::vector<std::string> extensions{".tif", ".tiff"};
  std::string match;
  bool use_regex = false;
  bool natural_sort = true;
  std::size_t start = 0;
  std::optional<std::size_t> count;
  std::size_t stride = 1;
  // Optional explicit, ordered file list. When non-empty it takes precedence
  // over folder scanning + match/start/stride (a GUI-exported subsequence is a
  // concrete list). Paths may be absolute or relative to `folder`.
  std::vector<std::string> files;
};

struct OutputConfig {
  std::filesystem::path folder;
  std::string mask_template = "{stem}_mask.tiff";
  std::string table_template = "{stem}_segments.csv";
  bool overwrite = true;
};

struct FilterConfig {
  std::string operation = "none";
  nlohmann::json params = nlohmann::json::object();
};

// One predicate in the feature-selection query chain. Predicates are ANDed:
// a feature is kept only if it satisfies every query. `field` names a per-feature
// statistic (area, mean_base, mean_filtered, min_base, max_base, std_base,
// bbox_w, bbox_h, ...); `op` is one of lt/le/gt/ge/eq/between. `value2` is only
// used by `between` (kept iff value <= stat <= value2).
struct FeatureQuery {
  std::string field;
  std::string op = "gt";
  double value = 0.0;
  double value2 = 0.0;
};

struct MscConfig {
  std::optional<float> persistence_absolute;
  std::optional<float> persistence_percent = 10.0f;
  std::string compute_algorithm = "serial";
  bool accurate_ascending = true;
  bool accurate_descending = true;
  std::string manifold = "ascending";
  // Discrete-gradient thread count and (in "partitioned" mode) MSC/hierarchy
  // partition count. 0 => leave the msc_2d_lib default (8).
  int requested_parallelism = 0;
};

struct SegmentKeepConfig {
  std::optional<std::size_t> min_area;
  std::optional<std::size_t> max_area;
  std::optional<float> min_value;
  std::optional<float> max_value;
  std::optional<float> min_mean;
  std::optional<float> max_mean;
  std::vector<int> allow_ids;
  std::vector<int> deny_ids;
};

struct ExecutionConfig {
  int total_threads = 0;
  int threads_per_slice = 4;
  int concurrent_slices = 0;
  int read_threads = 1;
  int write_threads = 1;
  std::size_t max_slices_at_a_time = 4;
  std::size_t read_queue_capacity = 4;
  std::size_t write_queue_capacity = 4;
};

struct MatchingConfig {
  // Cross-slice feature matching: link kept 2D features into 3D features by
  // 26-neighbor connectivity between consecutive slices. Per-slice masks/CSVs are
  // unchanged; the cross-slice identity is emitted as two derived files at the end
  // of the run (a per-slice->global map and an aggregated master table).
  bool enabled = true;
  std::string map_template = "feature_map.csv";
  std::string global_table_template = "global_segments.csv";
};

// 3D assembly of kept per-slice features into 3D features. `connectivity` is the
// voxel neighborhood (6/18/26); the CLI matcher and the GUI's scipy labeling
// share this value so they produce the same grouping.
struct AssemblyConfig {
  int connectivity = 26;
};

struct TimingConfig {
  bool write_json = true;
  bool write_csv = false;
  std::filesystem::path output_path = "timing_report.json";
};

struct DebugOutputConfig {
  bool write_filter_tiff = false;
  bool write_label_tiff = false;
  std::string filter_template = "{stem}_filter.tiff";
  std::string label_template = "{stem}_labels_i32.tiff";
};

struct AppConfig {
  InputConfig input;
  OutputConfig output;
  // Ordered filter chain (applied output->input). Populated from a `filters`
  // array, or from a singular legacy `filter` object as a one-element chain.
  std::vector<FilterConfig> filters;
  FilterConfig filter;  // legacy single-filter mirror (== filters.front()).
  MscConfig msc;
  SegmentKeepConfig segments;
  // Feature-selection query chain (ANDed). Generalizes `segments`; empty means
  // no query filtering beyond the legacy `segments` keep config.
  std::vector<FeatureQuery> feature_filters;
  ExecutionConfig execution;
  MatchingConfig matching;
  AssemblyConfig assembly;
  TimingConfig timing;
  DebugOutputConfig debug_output;
  bool dry_run = false;
};

struct CliOptions {
  std::filesystem::path config_path;
  std::optional<std::filesystem::path> input_folder_override;
  std::optional<std::filesystem::path> output_folder_override;
  std::optional<std::string> match_override;
  std::optional<std::size_t> start_override;
  std::optional<std::size_t> count_override;
  std::optional<std::size_t> stride_override;
  std::optional<int> worker_override;
  std::optional<int> parallelism_override;
  bool dump_filter_tiff = false;
  bool dump_label_tiff = false;
  bool disable_matching = false;
  bool dry_run = false;
};

CliOptions parse_cli(int argc, char** argv);
AppConfig load_config(const CliOptions& cli);
void validate_config(const AppConfig& cfg);

}  // namespace mscoupon
