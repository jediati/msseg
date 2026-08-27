#pragma once

#include <cstddef>
#include <unordered_set>
#include <vector>

#include "mscoupon/config.hpp"
#include "msseg/compute/channel_stats.hpp"
#include "msseg/filter/filter_stage.hpp"
#include "mscoupon/types.hpp"

namespace mscoupon {

// Per-connected-component geometry and extremum machinery.
//
// As with msseg::Msc2DFeatureStat, the per-channel aggregates are NOT members
// here -- they live in a parallel flat msseg::ChannelStats table indexed by
// component id, so a twelve-channel measurement stack does not grow this struct
// by twelve field groups. `filt_min`/`filt_max` stay because they LOCATE the
// component's seeding extremum whether or not `filtered` is a measurement
// channel.
struct CcNodeStat {
  std::size_t area = 0;
  float base_relevance_floor = 0.0f, base_relevance_ceiling = 0.0f;
  float filt_min = 0.0f, filt_max = 0.0f;
  int min_x = 0, min_y = 0, max_x = 0, max_y = 0;

  // This component's seeding extremum: the pixel attaining filt_min for an
  // ascending manifold, filt_max for a descending one -- the same rule the MSC
  // uses per base manifold, applied here to the component.
  //
  // It is derived from the component's own pixels rather than inherited from the
  // MSC feature it came from, for two reasons: a CC node has no MSC identity left
  // (the mask is a union of kept features), and the pixel trim chain runs BEFORE
  // this, so an inherited extremum could name a pixel that was trimmed away.
  float ext_x = -1.0f, ext_y = -1.0f;
  float ext_filtered = 0.0f;
};

// True iff `value op threshold` for op in {lt,le,gt,ge}.
bool pixel_predicate(float value, const std::string& op, double threshold);

// Build the keep mask for the selected regions (labels[p] in keep_ids), refine it
// by the pixel-intensity trim chain, run in-plane connected components (connectivity
// 6 -> 4-connectivity, 18/26 -> 8-connectivity), and compute per-component
// base+filtered statistics.
//   labels   per-pixel MSC feature id (-1 background), size width*height
//   base     original image; filtered  topology field (same dims)
//   keep_ids selected feature ids (per-slice selection)
// Fills `cc_out` (per-pixel CC id, -1 bg / 0..n-1) and `stats_out` (size n), and
// returns the component count n.
//   ascending  manifold direction, which picks the seeding extremum's side
//   stats      which statistics to accumulate (and whether to seek an extremum)
//   bank       the resolved measurement channels for this slice, reused from the
//              MSC stage rather than recomputed -- the scale-space responses do
//              not depend on which features were kept
// `channels_out` is filled in lockstep with `stats_out`: slot k of component c is
// bank.channels[k].
int label_selected_components(const std::vector<int>& labels, int width, int height,
                              const Image2D& base, const Image2D& filtered,
                              const std::unordered_set<int>& keep_ids,
                              const std::vector<PixelFilter>& pixel_filters,
                              int connectivity, bool ascending,
                              const msseg::StatsSpec& stats,
                              const msseg::StatChannelBank& bank,
                              float base_relevance_floor,
                              float base_relevance_ceiling,
                              std::vector<int>& cc_out,
                              std::vector<CcNodeStat>& stats_out,
                              msseg::ChannelStats& channels_out);

}  // namespace mscoupon
