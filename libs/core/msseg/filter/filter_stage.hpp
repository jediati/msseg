#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "diffg/image.hpp"
#include "diffg/multi_image.hpp"
#include "diffg/options.hpp"
#include "msseg/workflow/params.hpp"
#include "msseg/workflow/stat_channels.hpp"

namespace msseg {

// Apply a diffg filter/transform to a volume. Works for 2D (depth == 1) and
// 3D inputs alike, since every diffg operation is dimension-general.
// Returns the transformed volume; "none"/empty operation returns a copy.
diffg::Image<float> apply_filter(const diffg::Image<float>& input, const FilterParams& filter);

// Apply an ordered chain of filters, feeding each stage's output into the next.
// An empty chain returns a copy of the input.
diffg::Image<float> apply_filter_chain(const diffg::Image<float>& input,
                                       const std::vector<FilterParams>& filters);

// The measurement channels a StatsSpec asks for, as pixels.
//
// `base` and `filtered` are ALIASED, not copied -- the caller already owns those
// rasters and they outlive the bank in every call site. Only the derived
// Gaussian-derivative responses are materialized, and they are computed in a
// SINGLE diffg::apply_filter_bank traversal: filters sharing a 1-D kernel on an
// axis share that axis pass, which is the whole reason a multi-sigma stack is
// affordable per slice.
//
// `channel(k)` is indexed by slot, matching resolve_stat_channels(spec) exactly.
// Every downstream layer resolves names to slots once and then indexes; nothing
// looks a channel up by string inside a pixel loop.
struct StatChannelBank {
  std::vector<ResolvedStatChannel> channels;
  // Derived responses only, planar (channel-slowest). Empty when the spec asks
  // for no derived channel, in which case nothing is computed at all.
  diffg::MultiImage<float> derived;
  // Per-slot pointer into either the caller's base/filtered raster or `derived`.
  std::vector<const float*> data;

  std::size_t size() const { return channels.size(); }
  const float* channel(std::size_t k) const { return data[k]; }
  const std::string& name(std::size_t k) const { return channels[k].name; }
};

// Build the bank for `spec` over the two rasters the pipeline already has.
// `base` is the raster derived channels are computed FROM -- i.e. the
// post-`base_filters` channel, so a normalized workflow's scale-space responses
// are in normalized units too. `filtered` is only ever aliased.
//
// Both images must share dimensions. Throws for an unknown channel kind.
StatChannelBank build_stat_channels(const diffg::Image<float>& base,
                                    const diffg::Image<float>& filtered,
                                    const StatsSpec& spec,
                                    const diffg::ExecutionOptions& exec = {});

}  // namespace msseg
