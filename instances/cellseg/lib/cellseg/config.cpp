#include "cellseg/config.hpp"

#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>

namespace cellseg {
namespace {

template <typename T>
void set_if_present(const nlohmann::json& obj, const char* key, T& out) {
  if (obj.contains(key) && !obj.at(key).is_null()) out = obj.at(key).get<T>();
}

template <typename T>
void set_opt_if_present(const nlohmann::json& obj, const char* key, std::optional<T>& out) {
  if (obj.contains(key) && !obj.at(key).is_null()) out = obj.at(key).get<T>();
}

}  // namespace

bool parse_dims_from_filename(const std::filesystem::path& path, std::size_t& x,
                              std::size_t& y, std::size_t& z) {
  // Match a trailing _<X>x<Y>x<Z> before the extension, e.g. cells_128x128x64.raw
  static const std::regex re(R"(_(\d+)x(\d+)x(\d+)(?:\.[A-Za-z0-9]+)?$)");
  const std::string name = path.filename().string();
  std::smatch m;
  if (!std::regex_search(name, m, re)) return false;
  x = static_cast<std::size_t>(std::stoull(m[1].str()));
  y = static_cast<std::size_t>(std::stoull(m[2].str()));
  z = static_cast<std::size_t>(std::stoull(m[3].str()));
  return true;
}

AppConfig config_from_json(const nlohmann::json& root) {
  AppConfig cfg;

  if (root.contains("heavy_lift")) {
    const auto& h = root.at("heavy_lift");
    std::string input;
    set_if_present(h, "input_path", input);
    if (!input.empty()) cfg.heavy.input_path = input;
    set_opt_if_present(h, "dim_x", cfg.heavy.dim_x);
    set_opt_if_present(h, "dim_y", cfg.heavy.dim_y);
    set_opt_if_present(h, "dim_z", cfg.heavy.dim_z);
    set_if_present(h, "blur_sigma", cfg.heavy.blur_sigma);
    set_if_present(h, "accurate_ascending_3m", cfg.heavy.accurate_ascending_3m);
    set_if_present(h, "accurate_descending_3m", cfg.heavy.accurate_descending_3m);
    set_if_present(h, "presimp_threshold", cfg.heavy.presimp_threshold);
    set_if_present(h, "integration_error", cfg.heavy.integration_error);
    set_if_present(h, "gradient_threshold", cfg.heavy.gradient_threshold);
    set_if_present(h, "integration_max_iter", cfg.heavy.integration_max_iter);
    set_opt_if_present(h, "persistence_absolute", cfg.heavy.persistence_absolute);
    set_opt_if_present(h, "persistence_percent", cfg.heavy.persistence_percent);
  }

  if (root.contains("phase_b")) {
    const auto& b = root.at("phase_b");
    set_opt_if_present(b, "persistence_selection", cfg.phase_b.persistence_selection);
    set_if_present(b, "cut_threshold", cfg.phase_b.cut_threshold);
    set_if_present(b, "background_threshold", cfg.phase_b.background_threshold);
  }

  if (root.contains("output")) {
    const auto& o = root.at("output");
    std::string s;
    if (o.contains("seg8_path")) cfg.output.seg8_path = o.at("seg8_path").get<std::string>();
    if (o.contains("ids_path")) cfg.output.ids_path = o.at("ids_path").get<std::string>();
    if (o.contains("merge_tree_path"))
      cfg.output.merge_tree_path = o.at("merge_tree_path").get<std::string>();
  }

  set_if_present(root, "dry_run", cfg.dry_run);
  return cfg;
}

CliOptions parse_cli(int argc, char** argv) {
  CliOptions opts;
  auto need_value = [&](int& i, const char* flag) -> std::string {
    if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + flag);
    return std::string(argv[++i]);
  };
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      throw std::runtime_error(
          "usage: cellseg [--config <cfg.json>] [--input <vol_XxYxZ.raw>] [--dry-run]");
    } else if (arg == "--config" || arg == "-c") {
      opts.config_path = std::filesystem::path(need_value(i, "--config"));
    } else if (arg == "--input" || arg == "-i") {
      opts.input_override = std::filesystem::path(need_value(i, "--input"));
    } else if (arg == "--dry-run") {
      opts.dry_run = true;
    } else if (!arg.empty() && arg[0] != '-' && !opts.config_path) {
      opts.config_path = std::filesystem::path(arg);  // positional config
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  return opts;
}

AppConfig load_config(const CliOptions& cli) {
  AppConfig cfg;
  if (cli.config_path) {
    std::ifstream in(*cli.config_path);
    if (!in.good())
      throw std::runtime_error("Cannot open config: " + cli.config_path->string());
    nlohmann::json root;
    in >> root;
    cfg = config_from_json(root);
  }
  if (cli.input_override) cfg.heavy.input_path = *cli.input_override;
  if (cli.dry_run) cfg.dry_run = true;
  validate_config(cfg);
  return cfg;
}

void validate_config(const AppConfig& cfg) {
  if (cfg.heavy.input_path.empty())
    throw std::runtime_error("cellseg: no input_path (set heavy_lift.input_path or --input)");
  if (cfg.heavy.blur_sigma < 0.0f)
    throw std::runtime_error("cellseg: blur_sigma must be >= 0");
  if (cfg.heavy.persistence_percent && *cfg.heavy.persistence_percent < 0.0f)
    throw std::runtime_error("cellseg: persistence_percent must be >= 0");
}

}  // namespace cellseg
