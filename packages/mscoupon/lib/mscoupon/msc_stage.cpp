#include "mscoupon/msc_stage.hpp"

#include <algorithm>
#include <cstddef>

#include "diffg/image.hpp"
#include "msseg/compute/msc2d.hpp"

namespace mscoupon {
namespace {

// Map this instance's MscConfig onto the core msseg::Msc2DParams. This is where
// mscoupon makes its per-instance choices -- notably arc geometry: the current
// segmentation only needs MSC connectivity + the cancellation hierarchy, so we
// leave all arc-geometry dims OFF (it is the dominant per-slice cost). If a later
// approach needs, e.g., the ascending 1-saddle arc geometry (should the filters
// lack the needed sensitivity/specificity), enable that dim here.
msseg::Msc2DParams to_msc_params(const MscConfig& cfg, const msseg::StatsSpec& stats) {
  msseg::Msc2DParams params;
  params.stats = stats;
  params.persistence_absolute = cfg.persistence_absolute;
  params.persistence_percent = cfg.persistence_percent;
  params.compute_algorithm = cfg.compute_algorithm;
  params.use_gpu_gradient = cfg.use_gpu_gradient;
  params.use_gpu_stats = cfg.use_gpu_stats;
  params.accurate_ascending = cfg.accurate_ascending;
  params.accurate_descending = cfg.accurate_descending;
  params.manifold = cfg.manifold;
  params.requested_parallelism = cfg.requested_parallelism;
  params.extremum_sample_radius = cfg.extremum_sample_radius;
  params.build_arc_geometry = {{false, false, false}};
  return params;
}

}  // namespace

// Thin adapter over the core 2D MSC stage. Maps this instance's MscConfig onto
// msseg::Msc2DParams and its Image2D onto a diffg image (depth == 1).
std::vector<int> compute_msc_labels(const Image2D& filtered_image, const MscConfig& cfg) {
  diffg::Image<float> filtered(diffg::Dimensions{
      static_cast<std::size_t>(filtered_image.width), static_cast<std::size_t>(filtered_image.height), 1});
  std::copy(filtered_image.pixels.begin(), filtered_image.pixels.end(), filtered.data());

  // Labels only -- this one-shot path never builds per-feature statistics, so the
  // spec is irrelevant here.
  return msseg::compute_msc2d_labels(filtered, to_msc_params(cfg, msseg::StatsSpec{}));
}

SliceSegmentation segment_slice_pipeline(const diffg::Image<float>& base,
                                         const diffg::Image<float>& filtered,
                                         const MscConfig& cfg, const msseg::StatsSpec& stats,
                                         const msseg::StatChannelBank& bank) {
  msseg::Msc2DPipeline pipe;
  pipe.build(base, filtered, to_msc_params(cfg, stats), &bank);

  SliceSegmentation seg;
  seg.labels = pipe.labels();
  seg.features = pipe.feature_stats();
  seg.feature_channels = pipe.feature_channels();
  seg.channels = pipe.channels();
  seg.base_relevance_floor = pipe.base_relevance_floor();
  seg.base_relevance_ceiling = pipe.base_relevance_ceiling();
  return seg;
}

}  // namespace mscoupon
