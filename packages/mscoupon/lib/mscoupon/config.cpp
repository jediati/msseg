#include "mscoupon/config.hpp"

#include "msseg/workflow/stat_channels.hpp"

#include <fstream>
#include <stdexcept>
#include <string_view>

#include "mscoupon/query.hpp"   // is_feature_field / feature_fields (the schema)

namespace mscoupon {
namespace {

// Render the available field names for an error message, so a rejected config
// says what it could have used instead.
std::string join_fields(const std::vector<std::string>& names) {
  std::string out;
  for (const auto& n : names) {
    if (!out.empty()) out += ", ";
    out += n;
  }
  return out;
}

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
  if (root.contains("base_filters") && root.at("base_filters").is_array()) {
    for (const auto& f : root.at("base_filters")) {
      cfg.base_filters.push_back(parse_one_filter(f));
    }
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

void parse_pixel_filters(const nlohmann::json& root, std::vector<PixelFilter>& rules) {
  if (!root.contains("pixel_filters") || !root.at("pixel_filters").is_array()) {
    return;
  }
  for (const auto& r : root.at("pixel_filters")) {
    PixelFilter pf;
    set_if_present(r, "channel", pf.channel);
    set_if_present(r, "mode", pf.mode);
    set_if_present(r, "op", pf.op);
    set_if_present(r, "value", pf.value);
    rules.push_back(pf);
  }
}

void parse_assembly(const nlohmann::json& root, AssemblyConfig& assembly) {
  if (!root.contains("assembly")) {
    return;
  }
  set_if_present(root.at("assembly"), "connectivity", assembly.connectivity);
}

// Read a reduction list ("mean"/"min"/"max"/"std") into four flags. Absent =>
// leave the defaults (all on); present but empty => all off, which is how a
// workflow asks for a channel's extent without paying for the sums.
void parse_reductions(const nlohmann::json& block, const char* key, bool& mean, bool& min,
                      bool& max, bool& std_flag) {
  if (!block.contains(key) || block.at(key).is_null()) return;
  const auto& arr = block.at(key);
  if (!arr.is_array()) throw std::runtime_error(std::string("statistics.") + key + " must be an array.");
  mean = min = max = std_flag = false;
  for (const auto& item : arr) {
    const auto name = item.get<std::string>();
    if (name == "mean" || name == "average") {
      mean = true;
    } else if (name == "min") {
      min = true;
    } else if (name == "max") {
      max = true;
    } else if (name == "std") {
      std_flag = true;
    } else {
      throw std::runtime_error("statistics." + std::string(key) +
                               "[] must be mean/min/max/std (got '" + name + "').");
    }
  }
}

}  // namespace

void parse_statistics_json(const nlohmann::json& root, StatisticsConfig& stats) {
  // Default per-slice quantities: the ones whose per-slice spread actually says
  // something about a 3D feature's shape.
  stats.per_slice_quantities = {"area", "bbox_w", "bbox_h"};
  if (!root.contains("statistics") || root.at("statistics").is_null()) return;
  const auto& s = root.at("statistics");

  if (s.contains("channels") && !s.at("channels").is_null()) {
    const auto& arr = s.at("channels");
    if (!arr.is_array()) throw std::runtime_error("statistics.channels must be an array.");
    stats.spec.base_channel = false;
    stats.spec.filtered_channel = false;
    stats.spec.derived.clear();
    for (const auto& item : arr) {
      // A bare string is one of the two rasters the pipeline already builds; an
      // object requests a DERIVED scale-space channel measured off `base`.
      if (item.is_string()) {
        const auto name = item.get<std::string>();
        if (name == "base") {
          stats.spec.base_channel = true;
        } else if (name == "filtered") {
          stats.spec.filtered_channel = true;
        } else {
          throw std::runtime_error(
              "statistics.channels[] string must be base/filtered (got '" + name +
              "'); a derived channel is an object like "
              "{\"kind\": \"blur\", \"sigmas\": [1.0]}.");
        }
        continue;
      }
      if (!item.is_object()) {
        throw std::runtime_error("statistics.channels[] must be a string or an object.");
      }
      msseg::StatChannelRequest req;
      set_if_present(item, "kind", req.kind);
      if (req.kind == "base") { stats.spec.base_channel = true; continue; }
      if (req.kind == "filtered") { stats.spec.filtered_channel = true; continue; }
      if (!msseg::is_derived_channel_kind(req.kind)) {
        std::string known = "base, filtered";
        for (const auto& k : msseg::derived_channel_kinds()) known += ", " + k;
        throw std::runtime_error("statistics.channels[].kind '" + req.kind +
                                 "' is not known. Available: " + known + ".");
      }
      if (item.contains("sigmas") && !item.at("sigmas").is_null()) {
        req.sigmas = item.at("sigmas").get<std::vector<double>>();
      } else if (item.contains("sigma") && !item.at("sigma").is_null()) {
        req.sigmas = {item.at("sigma").get<double>()};   // singular convenience
      }
      set_if_present(item, "sort_by_absolute_value", req.sort_by_absolute_value);
      set_if_present(item, "name", req.name);
      stats.spec.derived.push_back(std::move(req));
    }
    // Resolve now so a bad sigma or a duplicate channel name is a config error
    // rather than a surprise on the first slice.
    (void)msseg::resolve_stat_channels(stats.spec);
  }
  parse_reductions(s, "reductions", stats.spec.mean, stats.spec.min, stats.spec.max,
                   stats.spec.std);
  set_if_present(s, "extremum", stats.spec.extremum);
  set_if_present(s, "extremum_sample_radius", stats.spec.extremum_sample_radius);
  if (s.contains("relevance") && !s.at("relevance").is_null()) {
    const auto& r = s.at("relevance");
    if (r.is_boolean()) {
      stats.spec.relevance = r.get<bool>();
    } else if (r.is_object()) {
      set_if_present(r, "enabled", stats.spec.relevance);
      set_if_present(r, "low_percentile", stats.spec.relevance_low_percentile);
      set_if_present(r, "high_percentile", stats.spec.relevance_high_percentile);
    } else {
      throw std::runtime_error("statistics.relevance must be a boolean or object.");
    }
  }
  const double rel_lo = stats.spec.relevance_low_percentile;
  const double rel_hi = stats.spec.relevance_high_percentile;
  if (rel_lo < 0.0 || rel_lo > 100.0 || rel_hi < 0.0 || rel_hi > 100.0 ||
      rel_lo > rel_hi) {
    throw std::runtime_error(
        "statistics.relevance percentiles must satisfy 0 <= low_percentile <= "
        "high_percentile <= 100.");
  }

  if (s.contains("per_slice") && !s.at("per_slice").is_null()) {
    const auto& p = s.at("per_slice");
    if (p.contains("quantities") && !p.at("quantities").is_null()) {
      stats.per_slice_quantities = p.at("quantities").get<std::vector<std::string>>();
    }
    parse_reductions(p, "reductions", stats.ps_mean, stats.ps_min, stats.ps_max, stats.ps_std);
  }
}

namespace {

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
  set_if_present(m, "use_gpu_gradient", msc.use_gpu_gradient);
  if (m.contains("use_gpu_stats") && !m.at("use_gpu_stats").is_null()) {
    msc.use_gpu_stats = m.at("use_gpu_stats").get<bool>();
  }
  set_if_present(m, "extremum_sample_radius", msc.extremum_sample_radius);
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
  set_if_present(m, "write_cc_labels", matching.write_cc_labels);
  set_if_present(m, "write_global_labels", matching.write_global_labels);
  set_if_present(m, "cc_label_template", matching.cc_label_template);
  set_if_present(m, "global_label_template", matching.global_label_template);
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
  parse_statistics_json(root, cfg.statistics);
  // `msc.extremum_sample_radius` predates the statistics block; keep honouring it
  // when the newer key was not set, so existing configs and the GUI's params JSON
  // keep working. An explicit statistics.extremum_sample_radius wins.
  if (cfg.statistics.spec.extremum_sample_radius == 0)
    cfg.statistics.spec.extremum_sample_radius = cfg.msc.extremum_sample_radius;
  parse_segments(root, cfg.segments);
  parse_feature_filters(root, cfg.feature_filters);
  parse_pixel_filters(root, cfg.pixel_filters);
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
  if (cfg.msc.extremum_sample_radius < 0) {
    throw std::runtime_error("msc.extremum_sample_radius must be >= 0 (0 = the single critical pixel).");
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
    // row_passes fails closed on an unknown field, so an unvalidated typo would
    // silently drop every feature. Reject it here instead. This also catches a
    // field the `statistics` block switched off -- including the filtered
    // aggregates, which are no longer computed by default.
    if (!is_feature_field(q.field, cfg.statistics.spec)) {
      throw std::runtime_error("feature_filters[].field is not an available statistic (got '" +
                               q.field + "'). Available: " +
                               join_fields(feature_fields(cfg.statistics.spec)));
    }
    if (q.op != "lt" && q.op != "le" && q.op != "gt" && q.op != "ge" &&
        q.op != "eq" && q.op != "between") {
      throw std::runtime_error("feature_filters[].op must be lt/le/gt/ge/eq/between (got '" + q.op + "').");
    }
  }
  for (const auto& r : cfg.pixel_filters) {
    if (r.channel != "base" && r.channel != "filtered") {
      throw std::runtime_error("pixel_filters[].channel must be base/filtered (got '" + r.channel + "').");
    }
    if (r.mode != "keep" && r.mode != "omit") {
      throw std::runtime_error("pixel_filters[].mode must be keep/omit (got '" + r.mode + "').");
    }
    if (r.op != "lt" && r.op != "le" && r.op != "gt" && r.op != "ge") {
      throw std::runtime_error("pixel_filters[].op must be lt/le/gt/ge (got '" + r.op + "').");
    }
  }
  if (cfg.assembly.connectivity != 6 && cfg.assembly.connectivity != 18 &&
      cfg.assembly.connectivity != 26) {
    throw std::runtime_error("assembly.connectivity must be 6/18/26.");
  }
}

}  // namespace mscoupon
