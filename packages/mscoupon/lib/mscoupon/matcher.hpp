#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "mscoupon/cc_stage.hpp"
#include "mscoupon/config.hpp"   // StatisticsConfig
#include "mscoupon/types.hpp"

namespace mscoupon {

// A per-slice raster of GLOBAL feature ids (-1 background), produced by the relabel
// pass at finalize().
struct GlobalLabelRaster {
  int slice_index = 0;
  int width = 0;
  int height = 0;
  std::vector<int> data;
};

// Streaming cross-slice connected-components over per-slice CC nodes. Slices are
// added in ascending order; a node in slice N links to a node in the previously
// added slice when they overlap under the `connectivity` stencil. The union-find
// keeps the FIRST-SEEN (lowest global node index = earliest slice / lowest local
// id) node as the representative, so nothing already emitted needs renumbering.
// finalize() numbers global ids in appearance order and produces the per-slice ->
// global map, the aggregated master table, and per-slice global-id rasters.
class SliceMatcher {
 public:
  // Which statistics to aggregate, and the manifold direction that decides which
  // constituent slice's extremum a merged feature inherits. Must be set before
  // finalize(); the defaults reproduce base-only aggregates with an extremum.
  void configure(const StatisticsConfig& stats, bool ascending,
                 std::vector<msseg::ResolvedStatChannel> channel_schema) {
    stats_ = stats;
    ascending_ = ascending;
    channel_schema_ = std::move(channel_schema);
  }

  // Add one slice's CC labeling (cc_labels: -1 bg, 0..n-1) + per-component stats,
  // linking it to the previously added slice by the `connectivity` stencil.
  // `node_channels` is that slice's flat per-component channel plane, in lockstep
  // with `node_stats`; its slot schema must be the run's resolved channel list.
  void add_slice(const std::vector<int>& cc_labels, int width, int height,
                 const std::vector<CcNodeStat>& node_stats,
                 const msseg::ChannelStats& node_channels, int slice_index,
                 int connectivity);

  // Called once per slice with that slice's relabeled raster. The raster is
  // MOVED in, and the matcher releases its own copy immediately after, so a sink
  // that writes and drops it keeps peak memory flat instead of proportional to
  // the stack depth.
  using GlobalRasterSink = std::function<void(GlobalLabelRaster&&)>;

  // Streaming form: emits per-slice rasters one at a time (ascending slice
  // order) and frees each slice as it goes. Prefer this -- a 2500-slice stack of
  // 3232^2 int32 rasters is ~100 GB, so collecting them all is not an option.
  void finalize(std::vector<FeatureMapRow>& map_out, GlobalFeatureTable& table_out,
                const GlobalRasterSink& emit);

  // Collecting form, for small stacks and tests. Holds every raster at once.
  void finalize(std::vector<FeatureMapRow>& map_out, GlobalFeatureTable& table_out,
                std::vector<GlobalLabelRaster>& rasters_out);

 private:
  int find(int x);
  void unite_first_seen(int a, int b);   // lower index (first-seen) wins

  std::vector<int> parent_;              // per global node index
  std::vector<CcNodeStat> node_stat_;    // per global node index
  // Per-global-node channel plane, appended slice by slice. Flat rather than one
  // table per slice so the merge indexes a node directly by its global index.
  msseg::ChannelStats node_channels_;
  std::vector<int> node_slice_;          // slice index of each node

  StatisticsConfig stats_;
  bool ascending_ = true;
  std::vector<msseg::ResolvedStatChannel> channel_schema_;

  struct SliceRaster {
    int slice_index = 0, width = 0, height = 0, node_base = 0, n = 0;
    std::vector<int> cc;                 // per-pixel CC id (-1 bg)
  };
  std::vector<SliceRaster> slices_;

  // Previous added slice, for cross-slice linking. The labels themselves live in
  // slices_.back() -- keeping a second copy here would cost another full raster.
  int prev_width_ = 0, prev_height_ = 0, prev_node_base_ = -1;
};

}  // namespace mscoupon
