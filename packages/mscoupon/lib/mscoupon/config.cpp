#include "mscoupon/config.hpp"

#include <fstream>
#include <stdexcept>
#include <string_view>

namespace mscoupon {
namespace {

template <typename T>
void set_if_present(const nlohmann::json& object, const char* key, T& out) {
  if (object.contains(key) && !object.at(key).is_null()) {
    out = object.at(key).get<T>();
  }
}

std::filesystem::path require_path(const nlohmann::json& object, const char* key) {
  if (!object.contains(key)) {
    throw std::runtime_error(std::string("Missing required key: ") + key);
  }
  return object.at(key).get<std::string>();
}

void parse_input(const nlohmann::json& root, InputConfig& input) {
  const auto& in = root.at("input");
  input.folder = require_path(in, "folder");
  set_if_present(in, "extensions", input.extensions);
  set_if_present(in, "match", input.match);
  set_if_present(in, "use_regex", input.use_regex);
  set_if_present(in, "natural_sort", input.natural_sort);
  set_if_present(in, "start", input.start);
  if (in.contains("count") && !in.at("count").is_null()) {
    input.count = in.at("count").get<std::size_t>();
  }
  set_if_present(in, "stride", input.stride);
  set_if_present(in, "files", input.files);
}

void parse_output(const nlohmann::json& root, OutputConfig& output) {
  const auto& out = root.at("output");
  output.folder = require_path(out, "folder");
  set_if_present(out, "mask_template", output.mask_template);
  set_if_present(out, "table_template", output.table_template);
  set_if_present(out, "overwrite", output.overwrite);
}

FilterConfig parse_one_filter(const nlohmann::json& f) {
  FilterConfig filter;
  set_if_present(f, "operation", filter.operation);
  if (f.contains("params")) {
    filter.params = f.at("params");
  }
  return filter;
}

// Populate the filter chain from a `filters` array when present, otherwise from
// a singular legacy `filter` object. `cfg.filter` mirrors the first stage so the
// pybind/legacy single-filter path keeps working.
void parse_filters(const nlohmann::json& root, AppConfig& cfg) {
  if (root.contains("filters") && root.at("filters").is_array()) {
    for (const auto& f : root.at("filters")) {
      cfg.filters.push_back(parse_one_filter(f));
    }
  } else if (root.contains("filter")) {
    cfg.filters.push_back(parse_one_filter(root.at("filter")));
  }
  if (!cfg.filters.empty()) {
    cfg.filter = cfg.filters.front();
  }
}

void parse_feature_filters(const nlohmann::json& root, std::vector<FeatureQuery>& queries) {
  if (!root.contains("feature_filters") || !root.at("feature_filters").is_array()) {
    return;
  }
  for (const auto& q : root.at("feature_filters")) {
    FeatureQuery fq;
    set_if_present(q, "field", fq.field);
    set_if_present(q, "op", fq.op);
    set_if_present(q, "value", fq.value);
    set_if_present(q, "value2", fq.value2);
    queries.push_back(fq);
  }
}

void parse_assembly(const nlohmann::json& root, AssemblyConfig& assembly) {
  if (!root.contains("assembly")) {
    return;
  }
  set_if_present(root.at("assembly"), "connectivity", assembly.connectivity);
}

void parse_msc(const nlohmann::json& root, MscConfig& msc) {
  const auto& m = root.at("msc");
  if (m.contains("persistence_absolute") && !m.at("persistence_absolute").is_null()) {
    msc.persistence_absolute = m.at("persistence_absolute").get<float>();
  }
  if (m.contains("persistence_percent") && !m.at("persistence_percent").is_null()) {
    msc.persistence_percent = m.at("persistence_percent").get<float>();
  }
  // Backward compatibility for older configs.
  if (m.contains("persistence") && !m.at("persistence").is_null()) {
    msc.persistence_absolute = m.at("persistence").get<float>();
  }
  set_if_present(m, "compute_algorithm", msc.compute_algorithm);
  set_if_present(m, "accurate_ascending", msc.accurate_ascending);
  set_if_present(m, "accurate_descending", msc.accurate_descending);
  set_if_present(m, "manifold", msc.manifold);
  set_if_present(m, "requested_parallelism", msc.requested_parallelism);
}

void parse_segments(const nlohmann::json& root, SegmentKeepConfig& seg) {
  if (!root.contains("segments")) {
    return;
  }
  const auto& s = root.at("segments");
  if (s.contains("min_area") && !s.at("min_area").is_null()) seg.min_area = s.at("min_area").get<std::size_t>();
  if (s.contains("max_area") && !s.at("max_area").is_null()) seg.max_area = s.at("max_area").get<std::size_t>();
  if (s.contains("min_value") && !s.at("min_value").is_null()) seg.min_value = s.at("min_value").get<float>();
  if (s.contains("max_value") && !s.at("max_value").is_null()) seg.max_value = s.at("max_value").get<float>();
  if (s.contains("min_mean") && !s.at("min_mean").is_null()) seg.min_mean = s.at("min_mean").get<float>();
  if (s.contains("max_mean") && !s.at("max_mean").is_null()) seg.max_mean = s.at("max_mean").get<float>();
  set_if_present(s, "allow_ids", seg.allow_ids);
  set_if_present(s, "deny_ids", seg.deny_ids);
}

void parse_execution(const nlohmann::json& root, ExecutionConfig& exec) {
  if (!root.contains("execution")) {
    return;
  }
  const auto& e = root.at("execution");
  set_if_present(e, "total_threads", exec.total_threads);
  set_if_present(e, "threads_per_slice", exec.threads_per_slice);
  set_if_present(e, "concurrent_slices", exec.concurrent_slices);
  set_if_present(e, "read_threads", exec.read_threads);
  set_if_present(e, "write_threads", exec.write_threads);
  set_if_present(e, "max_slices_at_a_time", exec.max_slices_at_a_time);
  set_if_present(e, "read_queue_capacity", exec.read_queue_capacity);
  set_if_present(e, "write_queue_capacity", exec.write_queue_capacity);
}

void parse_matching(const nlohmann::json& root, MatchingConfig& matching) {
  if (!root.contains("matching")) {
    return;
  }
  const auto& m = root.at("matching");
  set_if_present(m, "enabled", matching.enabled);
  set_if_present(m, "map_template", matching.map_template);
  set_if_present(m, "global_table_template", matching.global_table_template);
}

void parse_timing(const nlohmann::json& root, TimingConfig& timing) {
  if (!root.contains("timing")) {
    return;
  }
  const auto& t = root.at("timing");
  set_if_present(t, "write_json", timing.write_json);
  set_if_present(t, "write_csv", timing.write_csv);
  if (t.contains("output_path")) {
    timing.output_path = t.at("output_path").get<std::string>();
  }
}

void parse_debug_output(const nlohmann::json& root, DebugOutputConfig& debug) {
  if (!root.contains("debug_output")) {
    return;
  }
  const auto& d = root.at("debug_output");
  set_if_present(d, "write_filter_tiff", debug.write_filter_tiff);
  set_if_present(d, "write_label_tiff", debug.write_label_tiff);
  set_if_present(d, "filter_template", debug.filter_template);
  set_if_present(d, "label_template", debug.label_template);
}

void apply_cli_overrides(const CliOptions& cli, AppConfig& cfg) {
  if (cli.input_folder_override.has_value()) cfg.input.folder = *cli.input_folder_override;
  if (cli.output_folder_override.has_value()) cfg.output.folder = *cli.output_folder_override;
  if (cli.match_override.has_value()) cfg.input.match = *cli.match_override;
  if (cli.start_override.has_value()) cfg.input.start = *cli.start_override;
  if (cli.count_override.has_value()) cfg.input.count = *cli.count_override;
  if (cli.stride_override.has_value()) cfg.input.stride = *cli.stride_override;
  if (cli.worker_override.has_value()) cfg.execution.total_threads = *cli.worker_override;
  if (cli.parallelism_override.has_value()) cfg.msc.requested_parallelism = *cli.parallelism_override;
  if (cli.dump_filter_tiff) cfg.debug_output.write_filter_tiff = true;
  if (cli.dump_label_tiff) cfg.debug_output.write_label_tiff = true;
  if (cli.disable_matching) cfg.matching.enabled = false;
  if (cli.dry_run) cfg.dry_run = true;
}

bool consume_arg_value(int argc, char** argv, int& i, std::string& out) {
  if (i + 1 >= argc) return false;
  ++i;
  out = argv[i];
  return true;
}

}  // namespace

CliOptions parse_cli(int argc, char** argv) {
  CliOptions cli;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    std::string value;

    if (arg == "--config") {
      if (!consume_arg_value(argc, argv, i, value)) throw std::runtime_error("Missing value for --config");
      cli.config_path = value;
    } else if (arg == "--input-folder") {
      if (!consume_arg_value(argc, argv, i, value)) throw std::runtime_error("Missing value for --input-folder");
      cli.input_folder_override = value;
    } else if (arg == "--output-folder") {
      if (!consume_arg_value(argc, argv, i, value)) throw std::runtime_error("Missing value for --output-folder");
      cli.output_folder_override = value;
    } else if (arg == "--match") {
      if (!consume_arg_value(argc, argv, i, value)) throw std::runtime_error("Missing value for --match");
      cli.match_override = value;
    } else if (arg == "--start") {
      if (!consume_arg_value(argc, argv, i, value)) throw std::runtime_error("Missing value for --start");
      cli.start_override = static_cast<std::size_t>(std::stoull(value));
    } else if (arg == "--count") {
      if (!consume_arg_value(argc, argv, i, value)) throw std::runtime_error("Missing value for --count");
      cli.count_override = static_cast<std::size_t>(std::stoull(value));
    } else if (arg == "--stride") {
      if (!consume_arg_value(argc, argv, i, value)) throw std::runtime_error("Missing value for --stride");
      cli.stride_override = static_cast<std::size_t>(std::stoull(value));
    } else if (arg == "--workers") {
      if (!consume_arg_value(argc, argv, i, value)) throw std::runtime_error("Missing value for --workers");
      cli.worker_override = std::stoi(value);
    } else if (arg == "--parallelism") {
      if (!consume_arg_value(argc, argv, i, value)) throw std::runtime_error("Missing value for --parallelism");
      cli.parallelism_override = std::stoi(value);
    } else if (arg == "--dump-filter-tiff") {
      cli.dump_filter_tiff = true;
    } else if (arg == "--dump-label-tiff") {
      cli.dump_label_tiff = true;
    } else if (arg == "--no-matching") {
      cli.disable_matching = true;
    } else if (arg == "--dry-run") {
      cli.dry_run = true;
    } else if (arg == "--help" || arg == "-h") {
      throw std::runtime_error(
          "Usage: mscoupon --config <path> [--input-folder <path>] [--output-folder <path>] "
          "[--match <string>] [--start <n>] [--count <n>] [--stride <n>] [--workers <n>] "
          "[--parallelism <n>] [--dump-filter-tiff] [--dump-label-tiff] [--no-matching] [--dry-run]");
    } else {
      throw std::runtime_error(std::string("Unknown argument: ") + std::string(arg));
    }
  }

  if (cli.config_path.empty()) {
    throw std::runtime_error("Missing required --config");
  }
  return cli;
}

AppConfig load_config(const CliOptions& cli) {
  std::ifstream input(cli.config_path);
  if (!input.good()) {
    throw std::runtime_error("Could not open config file: " + cli.config_path.string());
  }

  nlohmann::json root;
  input >> root;

  AppConfig cfg;
  parse_input(root, cfg.input);
  parse_output(root, cfg.output);
  parse_filters(root, cfg);
  parse_msc(root, cfg.msc);
  parse_segments(root, cfg.segments);
  parse_feature_filters(root, cfg.feature_filters);
  parse_execution(root, cfg.execution);
  parse_matching(root, cfg.matching);
  parse_assembly(root, cfg.assembly);
  parse_timing(root, cfg.timing);
  parse_debug_output(root, cfg.debug_output);

  apply_cli_overrides(cli, cfg);
  validate_config(cfg);
  return cfg;
}

void validate_config(const AppConfig& cfg) {
  if (cfg.input.folder.empty()) throw std::runtime_error("input.folder must not be empty");
  if (cfg.output.folder.empty()) throw std::runtime_error("output.folder must not be empty");
  if (cfg.input.stride == 0) throw std::runtime_error("input.stride must be >= 1");
  if (cfg.execution.threads_per_slice <= 0) throw std::runtime_error("execution.threads_per_slice must be > 0");
  if (cfg.execution.read_threads <= 0) throw std::runtime_error("execution.read_threads must be > 0");
  if (cfg.execution.write_threads <= 0) throw std::runtime_error("execution.write_threads must be > 0");
  if (cfg.execution.max_slices_at_a_time == 0) throw std::runtime_error("execution.max_slices_at_a_time must be > 0");
  if (cfg.execution.read_queue_capacity == 0) throw std::runtime_error("execution.read_queue_capacity must be > 0");
  if (cfg.execution.write_queue_capacity == 0) throw std::runtime_error("execution.write_queue_capacity must be > 0");
  if (!cfg.msc.persistence_absolute.has_value() && !cfg.msc.persistence_percent.has_value()) {
    throw std::runtime_error("msc requires persistence_absolute or persistence_percent.");
  }
  if (cfg.msc.persistence_percent.has_value()) {
    const float p = *cfg.msc.persistence_percent;
    if (p < 0.0f || p > 100.0f) throw std::runtime_error("msc.persistence_percent must be in [0, 100].");
  }
  if (cfg.msc.compute_algorithm != "serial" && cfg.msc.compute_algorithm != "partitioned") {
    throw std::runtime_error("msc.compute_algorithm must be 'serial' or 'partitioned'.");
  }
  if (cfg.msc.requested_parallelism < 0) {
    throw std::runtime_error("msc.requested_parallelism must be >= 0 (0 = library default).");
  }
  if (cfg.matching.enabled) {
    if (cfg.matching.map_template.empty()) throw std::runtime_error("matching.map_template must not be empty when matching is enabled.");
    if (cfg.matching.global_table_template.empty()) {
      throw std::runtime_error("matching.global_table_template must not be empty when matching is enabled.");
    }
  }
  if (cfg.assembly.connectivity != 6 && cfg.assembly.connectivity != 18 &&
      cfg.assembly.connectivity != 26) {
    throw std::runtime_error("assembly.connectivity must be 6, 18, or 26.");
  }
  for (const auto& q : cfg.feature_filters) {
    if (q.field.empty()) throw std::runtime_error("feature_filters[].field must not be empty.");
    if (q.op != "lt" && q.op != "le" && q.op != "gt" && q.op != "ge" &&
        q.op != "eq" && q.op != "between") {
      throw std::runtime_error("feature_filters[].op must be lt/le/gt/ge/eq/between (got '" + q.op + "').");
    }
  }
}

}  // namespace mscoupon
