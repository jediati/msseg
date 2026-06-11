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

struct MscConfig {
  float persistence = 0.0f;
  bool accurate_ascending = true;
  bool accurate_descending = true;
  std::string manifold = "ascending";
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

struct TimingConfig {
  bool write_json = true;
  bool write_csv = false;
  std::filesystem::path output_path = "timing_report.json";
};

struct AppConfig {
  InputConfig input;
  OutputConfig output;
  FilterConfig filter;
  MscConfig msc;
  SegmentKeepConfig segments;
  ExecutionConfig execution;
  TimingConfig timing;
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
  bool dry_run = false;
};

CliOptions parse_cli(int argc, char** argv);
AppConfig load_config(const CliOptions& cli);
void validate_config(const AppConfig& cfg);

}  // namespace mscoupon
