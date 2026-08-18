#include "mscoupon/sequence.hpp"

#include <algorithm>
#include <cctype>
#include <regex>
#include <stdexcept>

namespace mscoupon {
namespace {

std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

bool extension_matches(const std::filesystem::path& p, const std::vector<std::string>& extensions) {
  const std::string ext = to_lower(p.extension().string());
  for (const auto& candidate : extensions) {
    if (ext == to_lower(candidate)) return true;
  }
  return false;
}

bool name_matches(const std::string& name, const InputConfig& cfg) {
  if (cfg.match.empty()) return true;
  if (cfg.use_regex) return std::regex_search(name, std::regex(cfg.match));
  return name.find(cfg.match) != std::string::npos;
}

std::vector<std::string> tokenize_natural(const std::string& s) {
  std::vector<std::string> out;
  if (s.empty()) return out;
  std::string current;
  bool digit = std::isdigit(static_cast<unsigned char>(s[0])) != 0;
  for (char ch : s) {
    const bool is_digit = std::isdigit(static_cast<unsigned char>(ch)) != 0;
    if (is_digit != digit) {
      out.push_back(current);
      current.clear();
      digit = is_digit;
    }
    current.push_back(ch);
  }
  out.push_back(current);
  return out;
}

bool natural_less(const std::filesystem::path& a, const std::filesystem::path& b) {
  const auto sa = a.filename().string();
  const auto sb = b.filename().string();
  const auto ta = tokenize_natural(sa);
  const auto tb = tokenize_natural(sb);
  const std::size_t n = std::min(ta.size(), tb.size());
  for (std::size_t i = 0; i < n; ++i) {
    const bool ad = !ta[i].empty() && std::isdigit(static_cast<unsigned char>(ta[i][0]));
    const bool bd = !tb[i].empty() && std::isdigit(static_cast<unsigned char>(tb[i][0]));
    if (ad && bd) {
      const auto na = std::stoll(ta[i]);
      const auto nb = std::stoll(tb[i]);
      if (na < nb) return true;
      if (na > nb) return false;
    } else {
      if (ta[i] < tb[i]) return true;
      if (ta[i] > tb[i]) return false;
    }
  }
  return ta.size() < tb.size();
}

std::string render_template(const std::string& template_value, const std::filesystem::path& input_path, int idx) {
  std::string out = template_value;
  const auto stem = input_path.stem().string();
  const auto index = std::to_string(idx);
  const auto replace_all = [](std::string& text, const std::string& from, const std::string& to) {
    std::size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
      text.replace(pos, from.size(), to);
      pos += to.size();
    }
  };
  replace_all(out, "{stem}", stem);
  replace_all(out, "{index}", index);
  return out;
}

}  // namespace

std::vector<SliceJob> build_sequence(const AppConfig& cfg) {
  std::vector<std::filesystem::path> candidates;

  // An explicit `files` list is the authoritative, ordered selection (e.g. a
  // GUI-exported subsequence): use it verbatim, skipping folder scan + match +
  // natural sort + start/stride/count. Relative entries resolve against folder.
  const bool explicit_files = !cfg.input.files.empty();
  if (explicit_files) {
    for (const auto& f : cfg.input.files) {
      std::filesystem::path p(f);
      if (p.is_relative()) p = cfg.input.folder / p;
      if (!std::filesystem::exists(p)) {
        throw std::runtime_error("Listed input file does not exist: " + p.string());
      }
      candidates.push_back(p);
    }
  } else {
    if (!std::filesystem::exists(cfg.input.folder)) {
      throw std::runtime_error("Input folder does not exist: " + cfg.input.folder.string());
    }
    for (const auto& entry : std::filesystem::directory_iterator(cfg.input.folder)) {
      if (!entry.is_regular_file()) continue;
      const auto path = entry.path();
      if (!extension_matches(path, cfg.input.extensions)) continue;
      if (!name_matches(path.filename().string(), cfg.input)) continue;
      candidates.push_back(path);
    }

    if (cfg.input.natural_sort) {
      std::sort(candidates.begin(), candidates.end(), natural_less);
    } else {
      std::sort(candidates.begin(), candidates.end());
    }
  }

  std::vector<SliceJob> jobs;
  // Explicit lists consume every entry in order; scanned candidates honor
  // start/stride/count.
  const std::size_t start = explicit_files ? 0 : cfg.input.start;
  const std::size_t stride = explicit_files ? 1 : cfg.input.stride;
  for (std::size_t i = start; i < candidates.size(); i += stride) {
    if (!explicit_files && cfg.input.count.has_value() && jobs.size() >= *cfg.input.count) break;

    const auto& input_path = candidates[i];
    const int slice_index = static_cast<int>(jobs.size());

    SliceJob job;
    job.input_path = input_path;
    job.slice_index = slice_index;
    job.mask_output_path = cfg.output.folder / render_template(cfg.output.mask_template, input_path, slice_index);
    job.table_output_path = cfg.output.folder / render_template(cfg.output.table_template, input_path, slice_index);
    job.filter_output_path = cfg.output.folder / render_template(cfg.debug_output.filter_template, input_path, slice_index);
    job.label_output_path = cfg.output.folder / render_template(cfg.debug_output.label_template, input_path, slice_index);
    job.cc_label_output_path = cfg.output.folder / render_template(cfg.matching.cc_label_template, input_path, slice_index);
    job.global_label_output_path = cfg.output.folder / render_template(cfg.matching.global_label_template, input_path, slice_index);
    if (!cfg.output.overwrite && (std::filesystem::exists(job.mask_output_path) || std::filesystem::exists(job.table_output_path))) {
      throw std::runtime_error("Output exists while overwrite is false: " + job.input_path.string());
    }
    jobs.push_back(std::move(job));
  }

  if (jobs.empty()) {
    throw std::runtime_error("No input TIFF files matched the configured selection.");
  }
  return jobs;
}

}  // namespace mscoupon
