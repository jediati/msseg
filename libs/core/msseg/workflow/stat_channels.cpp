#include "msseg/workflow/stat_channels.hpp"

#include <cstdio>
#include <stdexcept>

namespace msseg {
namespace {

// The eigenvalue slot names, in diffg's descending order: channel 0 is the
// largest, channel 1 the smallest. The words match what filter_stage.cpp's
// `hessian_eigenvalues` operation already calls them, so a workflow that reads
// a Hessian component as a filter and as a statistic uses one vocabulary.
const char* kHessianSlot[2] = {"largest", "smallest"};

}  // namespace

std::string format_sigma(double sigma) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%g", sigma);
  return std::string(buf);
}

const std::vector<std::string>& derived_channel_kinds() {
  static const std::vector<std::string> kinds = {"blur", "edges", "gradmag",
                                                 "laplacian", "hessian"};
  return kinds;
}

bool is_derived_channel_kind(const std::string& kind) {
  for (const auto& k : derived_channel_kinds()) {
    if (k == kind) return true;
  }
  return false;
}

int channels_per_sigma(const std::string& kind) {
  if (kind == "hessian") return 2;  // 2D: largest and smallest eigenvalue
  if (is_derived_channel_kind(kind)) return 1;
  throw std::runtime_error("Unknown derived statistics channel kind: '" + kind + "'.");
}

std::vector<ResolvedStatChannel> resolve_stat_channels(const StatsSpec& spec) {
  std::vector<ResolvedStatChannel> out;

  // The two rasters the pipeline already builds keep slots 0/1 when enabled, so
  // an empty `derived` list reproduces the previous column order exactly.
  if (spec.base_channel) {
    ResolvedStatChannel c;
    c.name = "base";
    c.kind = "base";
    out.push_back(c);
  }
  if (spec.filtered_channel) {
    ResolvedStatChannel c;
    c.name = "filtered";
    c.kind = "filtered";
    out.push_back(c);
  }

  for (const auto& req : spec.derived) {
    // base/filtered may also be spelled as a request object; they are handled
    // above via the flags, so skip them rather than duplicating a slot.
    if (req.kind == "base" || req.kind == "filtered") continue;
    const int per_sigma = channels_per_sigma(req.kind);
    if (req.sigmas.empty()) {
      throw std::runtime_error("statistics channel '" + req.kind +
                               "' needs at least one sigma.");
    }
    const std::string prefix = req.name.empty() ? req.kind : req.name;
    for (const double sigma : req.sigmas) {
      if (!(sigma > 0.0)) {
        throw std::runtime_error("statistics channel '" + prefix +
                                 "' has a non-positive sigma (" +
                                 format_sigma(sigma) + "); sigmas must be > 0.");
      }
      for (int slot = 0; slot < per_sigma; ++slot) {
        ResolvedStatChannel c;
        c.kind = req.kind;
        c.sigma = sigma;
        c.slot_in_request = slot;
        c.sort_by_absolute_value = req.sort_by_absolute_value;
        c.name = prefix;
        if (per_sigma > 1) c.name += std::string("_") + kHessianSlot[slot];
        c.name += "_s" + format_sigma(sigma);
        out.push_back(c);
      }
    }
  }

  // A duplicate name would silently alias two columns in the feature table and
  // the CSV, so reject it here rather than letting the ambiguity propagate.
  for (std::size_t i = 0; i < out.size(); ++i) {
    for (std::size_t j = i + 1; j < out.size(); ++j) {
      if (out[i].name == out[j].name) {
        throw std::runtime_error("duplicate statistics channel name '" + out[i].name +
                                 "'; give one of them a distinct `name`.");
      }
    }
  }
  return out;
}

}  // namespace msseg
