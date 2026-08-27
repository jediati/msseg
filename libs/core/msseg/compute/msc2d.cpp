#include "msseg/compute/msc2d.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

#include "msc_2d_lib.h"

#include "msseg/filter/filter_stage.hpp"
#include "msseg/graph/msc_graph.hpp"

namespace msseg {
namespace {

// The linked msc_2d_lib may or may not expose a ComputeOptions/BuilderMode
// surface (it depends on the pinned MSCEER revision). Detect it at compile
// time so we can honor compute_algorithm/accurate flags when available and
// fall back to the legacy 5-arg compute() otherwise.
template <typename MscType, typename = void>
struct HasComputeOptions : std::false_type {};

template <typename MscType>
struct HasComputeOptions<MscType, std::void_t<typename MscType::ComputeOptions>> : std::true_type {};

template <typename MscType>
void compute_with_algorithm(MscType& msc, const float* pixels, int rows, int cols, const Msc2DParams& cfg) {
  if constexpr (HasComputeOptions<MscType>::value) {
    typename MscType::ComputeOptions options;
    options.accurateAsc = cfg.accurate_ascending;
    options.accurateDsc = cfg.accurate_descending;
    // Pass the caller's per-dim arc-geometry choice straight through; the core
    // imposes no policy (the instance decides -- see Msc2DParams).
    options.buildArcGeometry[0] = cfg.build_arc_geometry[0];
    options.buildArcGeometry[1] = cfg.build_arc_geometry[1];
    options.buildArcGeometry[2] = cfg.build_arc_geometry[2];
    if (cfg.requested_parallelism > 0) {
      options.requestedParallelism = cfg.requested_parallelism;
    }
    if (cfg.compute_algorithm == "partitioned") {
      options.builderMode = MscType::BuilderMode::Partitioned;
    } else {
      options.builderMode = MscType::BuilderMode::Serial;
    }
    // Cap the cancellation hierarchy so native re-thresholding (setPersistence +
    // ascending/descending2Manifolds) spans the selectable persistence range. We
    // EXTEND the cap to the configured max persistence but never below MSCEER's
    // default 10%-of-range floor -- the floor keeps the base complex at its default
    // (basePersistence = min(1%, cancel) stays 1%), so a Msc2DPipeline built to a
    // high cap and a one-shot compute_msc2d_labels agree at every persistence.
    float lo = std::numeric_limits<float>::max();
    float hi = std::numeric_limits<float>::lowest();
    const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    for (std::size_t i = 0; i < n; ++i) {
      lo = std::min(lo, pixels[i]);
      hi = std::max(hi, pixels[i]);
    }
    const float range = hi - lo;
    float cap = 0.1f * range;  // MSCEER default floor
    if (cfg.persistence_absolute.has_value()) {
      cap = std::max(cap, *cfg.persistence_absolute);
    } else if (cfg.persistence_percent.has_value()) {
      cap = std::max(cap, range * (*cfg.persistence_percent / 100.0f));
    }
    options.cancelPersistenceAbs = cap;
    msc.compute(pixels, rows, cols, options);
  } else {
    if (cfg.compute_algorithm == "partitioned") {
      throw std::runtime_error(
          "Configured msc.compute_algorithm='partitioned' but linked msc_2d_lib does not expose "
          "ComputeOptions/BuilderMode.");
    }
    msc.compute(pixels, rows, cols, cfg.accurate_ascending, cfg.accurate_descending);
  }
}

}  // namespace

std::vector<int> compute_msc2d_labels(const diffg::Image<float>& filtered, const Msc2DParams& cfg) {
  const int width = static_cast<int>(filtered.dims().width);
  const int height = static_cast<int>(filtered.dims().height);
  if (width <= 0 || height <= 0) {
    throw std::runtime_error("Invalid image dimensions for MSC.");
  }
  if (filtered.dims().depth != 1) {
    throw std::runtime_error("compute_msc2d_labels requires a 2D image (depth == 1).");
  }

  GInt::Msc2D::Msc2D msc;
  compute_with_algorithm(msc, filtered.data(), height, width, cfg);

  float persistence_absolute = 0.0f;
  if (cfg.persistence_absolute.has_value()) {
    persistence_absolute = *cfg.persistence_absolute;
  } else if (cfg.persistence_percent.has_value()) {
    float min_v = std::numeric_limits<float>::max();
    float max_v = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < filtered.size(); ++i) {
      const float v = filtered.data()[i];
      min_v = std::min(min_v, v);
      max_v = std::max(max_v, v);
    }
    const float range = max_v - min_v;
    persistence_absolute = range * (*cfg.persistence_percent / 100.0f);
  } else {
    throw std::runtime_error("MSC persistence not configured.");
  }
  msc.setPersistence(persistence_absolute);

  if (cfg.manifold == "ascending") {
    return msc.ascending2Manifolds().labels;
  }
  if (cfg.manifold == "descending") {
    return msc.descending2Manifolds().labels;
  }
  throw std::runtime_error("Invalid msc.manifold value. Use 'ascending' or 'descending'.");
}

// --------------------------------------------------------------------------- //
// Msc2DPipeline: two-phase 2D facade (build once, re-threshold cheaply).
// --------------------------------------------------------------------------- //
namespace {

double percentile_linear(std::vector<double>& values, double q) {
  const std::size_t n = values.size();
  if (n == 0) return 0.0;
  if (n == 1) return values[0];
  const double h = static_cast<double>(n - 1) * (q / 100.0);
  std::size_t lo = static_cast<std::size_t>(std::floor(h));
  std::size_t hi = static_cast<std::size_t>(std::ceil(h));
  hi = std::min(hi, n - 1);
  lo = std::min(lo, hi);
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(lo),
                   values.end());
  const double a = values[lo];
  double b = a;
  if (hi != lo) {
    std::nth_element(values.begin() + static_cast<std::ptrdiff_t>(lo) + 1,
                     values.begin() + static_cast<std::ptrdiff_t>(hi), values.end());
    b = values[hi];
  }
  return a + (b - a) * (h - static_cast<double>(lo));
}

// Merge two manifolds' geometry + extremum machinery. The per-channel aggregates
// are merged separately, by ChannelStats::merge_region, because they live in a
// flat table rather than in this struct. The extremum fields (ext_*) are
// deliberately NOT merged in either place: a living feature inherits the
// surviving critical point, which select_persistence() stamps from the surviving
// base manifold afterwards.
void accumulate(Msc2DFeatureStat& dst, const Msc2DFeatureStat& src) {
  if (src.area == 0) return;
  if (dst.area == 0) {
    const NodeId keep = dst.feature_id;
    dst = src;
    dst.feature_id = keep;
    return;
  }
  dst.area += src.area;
  dst.filt_min = std::min(dst.filt_min, src.filt_min);
  dst.filt_max = std::max(dst.filt_max, src.filt_max);
  dst.min_x = std::min(dst.min_x, src.min_x);
  dst.min_y = std::min(dst.min_y, src.min_y);
  dst.max_x = std::max(dst.max_x, src.max_x);
  dst.max_y = std::max(dst.max_y, src.max_y);
}

}  // namespace

struct Msc2DPipeline::Impl {
  int width = 0;
  int height = 0;
  float value_range = 0.0f;
  bool ascending = true;
  float current_persistence = 0.0f;
  // Which statistics build() was asked for -- consumers need it to know which
  // fields in leaf_stats/features are meaningful.
  StatsSpec stats;
  float base_relevance_floor = 0.0f;
  float base_relevance_ceiling = 0.0f;

  // The MSC engine is kept alive so persistence re-thresholding uses MSCEER's
  // NATIVE cancellation hierarchy (setPersistence + ascending/descending2Manifolds),
  // not an external merge tree. Adjacent-basin cancellation keeps every living
  // feature spatially connected.
  GInt::Msc2D::Msc2D msc;
  // Native base-extremum node id -> dense 0..M-1 compact id.
  std::unordered_map<int, int> nid_to_compact;
  // Per-base-manifold leaf statistics, indexed by compact base id.
  std::vector<Msc2DFeatureStat> leaf_stats;
  // Per-base-manifold per-channel aggregates, same indexing. Held flat rather
  // than inside leaf_stats so a twelve-channel stack costs one allocation for
  // the slice instead of one per manifold.
  ChannelStats leaf_channels;
  // The measurement channels build() resolved, in slot order.
  std::vector<ResolvedStatChannel> channels;
  // Compact base-extremum id per pixel (row-major), -1 where unlabeled.
  std::vector<int> base_labels;

  // Derived at the current persistence:
  std::vector<int> labels;                       // surviving feature id per pixel
  std::vector<Msc2DFeatureStat> features;        // aggregated per surviving feature
  ChannelStats feature_channels;                 // per-channel aggregates, same order
};

Msc2DPipeline::Msc2DPipeline() : impl_(std::make_unique<Impl>()) {}
Msc2DPipeline::~Msc2DPipeline() = default;
Msc2DPipeline::Msc2DPipeline(Msc2DPipeline&&) noexcept = default;
Msc2DPipeline& Msc2DPipeline::operator=(Msc2DPipeline&&) noexcept = default;

void Msc2DPipeline::build(const diffg::Image<float>& base, const diffg::Image<float>& filtered,
                          const Msc2DParams& cfg, const StatChannelBank* external_bank) {
  const int width = static_cast<int>(filtered.dims().width);
  const int height = static_cast<int>(filtered.dims().height);
  if (width <= 0 || height <= 0) throw std::runtime_error("Invalid image dimensions for MSC.");
  if (filtered.dims().depth != 1) throw std::runtime_error("Msc2DPipeline requires depth == 1.");
  if (base.dims().width != filtered.dims().width || base.dims().height != filtered.dims().height) {
    throw std::runtime_error("Msc2DPipeline: base and filtered images must share dimensions.");
  }
  const bool ascending = (cfg.manifold != "descending");

  impl_->width = width;
  impl_->height = height;
  impl_->ascending = ascending;

  // Value range of the filtered field (for percent->absolute persistence).
  float min_v = std::numeric_limits<float>::max();
  float max_v = std::numeric_limits<float>::lowest();
  for (std::size_t i = 0; i < filtered.size(); ++i) {
    min_v = std::min(min_v, filtered.data()[i]);
    max_v = std::max(max_v, filtered.data()[i]);
  }
  impl_->value_range = max_v - min_v;

  // Heavy: compute the MSC hierarchy once and keep the engine (impl_->msc) alive
  // for cheap native re-thresholding. compute_with_algorithm caps the cancellation
  // hierarchy at the configured max persistence so setPersistence() spans the range.
  compute_with_algorithm(impl_->msc, filtered.data(), height, width, cfg);

  // Base (finest) manifold labeling: at persistence -1 the native base->living
  // remap is the identity, so each labeled pixel carries its BASE extremum node id.
  impl_->msc.setPersistence(-1.0f);
  const GInt::Msc2D::LabelImage base_img =
      ascending ? impl_->msc.ascending2Manifolds() : impl_->msc.descending2Manifolds();

  // Compact the sparse base extremum node ids to a dense 0..M-1 id space, and
  // stamp the compact base id per pixel.
  impl_->nid_to_compact.clear();
  impl_->base_labels.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), -1);
  const std::size_t npix = impl_->base_labels.size();
  for (std::size_t i = 0; i < npix && i < base_img.labels.size(); ++i) {
    const int nid = base_img.labels[i];
    if (nid < 0) continue;
    auto it = impl_->nid_to_compact.find(nid);
    int compact;
    if (it == impl_->nid_to_compact.end()) {
      compact = static_cast<int>(impl_->nid_to_compact.size());
      impl_->nid_to_compact.emplace(nid, compact);
    } else {
      compact = it->second;
    }
    impl_->base_labels[i] = compact;
  }
  const std::size_t num_base = impl_->nid_to_compact.size();

  // Per-base-manifold leaf statistics (one pass over pixels, both images),
  // indexed by compact base id. These are aggregated up to living features on
  // each select_persistence() -- the base constituents are what we precompute.
  const StatsSpec& spec = cfg.stats;
  impl_->stats = spec;
  const bool want_filt_extent = spec.needs_filtered_extent();

  // Resolve the measurement channels and materialize them once. The derived
  // (scale-space) responses collapse into a single diffg filter-bank traversal;
  // base/filtered are aliased, not copied. The bank stays alive only for this
  // accumulation pass -- afterwards the per-manifold cells are all we keep, so a
  // twelve-channel stack does not hold twelve rasters per primed slice.
  diffg::ExecutionOptions bank_exec{};
  bank_exec.threads = std::max(1, cfg.requested_parallelism);
  StatChannelBank owned_bank;
  if (external_bank == nullptr) {
    owned_bank = build_stat_channels(base, filtered, spec, bank_exec);
  }
  const StatChannelBank& bank = external_bank != nullptr ? *external_bank : owned_bank;
  impl_->channels = bank.channels;
  const std::size_t n_ch = bank.size();
  const std::vector<const float*>& chan = bank.data;
  impl_->base_relevance_floor = 0.0f;
  impl_->base_relevance_ceiling = 0.0f;
  if (spec.relevance && spec.base_channel) {
    const bool need_percentiles =
        spec.relevance_low_percentile != 0.0 || spec.relevance_high_percentile != 100.0;
    std::vector<double> finite_base;
    if (need_percentiles) finite_base.reserve(npix);
    float finite_min = std::numeric_limits<float>::max();
    float finite_max = std::numeric_limits<float>::lowest();
    std::size_t finite_count = 0;
    for (std::size_t i = 0; i < npix; ++i) {
      const float value = base.data()[i];
      if (!std::isfinite(value)) continue;
      finite_min = std::min(finite_min, value);
      finite_max = std::max(finite_max, value);
      ++finite_count;
      if (need_percentiles) finite_base.push_back(static_cast<double>(value));
    }
    if (finite_count > 0) {
      impl_->base_relevance_floor =
          spec.relevance_low_percentile == 0.0
              ? finite_min
              : static_cast<float>(
                    percentile_linear(finite_base, spec.relevance_low_percentile));
      impl_->base_relevance_ceiling =
          spec.relevance_high_percentile == 100.0
              ? finite_max
              : static_cast<float>(
                    percentile_linear(finite_base, spec.relevance_high_percentile));
    }
  }

  std::vector<Msc2DFeatureStat>& leaf = impl_->leaf_stats;
  leaf.assign(num_base, Msc2DFeatureStat{});
  impl_->leaf_channels.reset(num_base, n_ch, spec);
  for (std::size_t c = 0; c < num_base; ++c) {
    leaf[c].feature_id = static_cast<NodeId>(c);
    leaf[c].base_relevance_floor = impl_->base_relevance_floor;
    leaf[c].base_relevance_ceiling = impl_->base_relevance_ceiling;
    // Only sentinel the filtered extent when it is actually going to be filled;
    // otherwise a min/max sentinel would survive into merges and CSVs as a
    // 3.4e38 rather than an obvious zero.
    if (want_filt_extent) {
      leaf[c].filt_min = std::numeric_limits<float>::max();
      leaf[c].filt_max = std::numeric_limits<float>::lowest();
    }
    leaf[c].min_x = width;
    leaf[c].min_y = height;
    leaf[c].max_x = -1;
    leaf[c].max_y = -1;
  }
  // Pixel attaining the seeding extremum per base manifold. Only the side the
  // manifold direction actually seeds from is tracked -- the other was always
  // allocated and never read.
  std::vector<std::size_t> arg_ext(num_base, 0);
  for (std::size_t i = 0; i < npix; ++i) {
    const int c = impl_->base_labels[i];
    if (c < 0) continue;
    Msc2DFeatureStat& s = leaf[static_cast<std::size_t>(c)];
    const int x = static_cast<int>(i % static_cast<std::size_t>(width));
    const int y = static_cast<int>(i / static_cast<std::size_t>(width));
    s.area += 1;
    impl_->leaf_channels.add(static_cast<std::size_t>(c), i, chan);
    if (want_filt_extent) {
      const float f = filtered.data()[i];
      if (ascending) {
        if (f < s.filt_min) { s.filt_min = f; arg_ext[static_cast<std::size_t>(c)] = i; }
        s.filt_max = std::max(s.filt_max, f);
      } else {
        if (f > s.filt_max) { s.filt_max = f; arg_ext[static_cast<std::size_t>(c)] = i; }
        s.filt_min = std::min(s.filt_min, f);
      }
    }
    s.min_x = std::min(s.min_x, x);
    s.min_y = std::min(s.min_y, y);
    s.max_x = std::max(s.max_x, x);
    s.max_y = std::max(s.max_y, y);
  }

  // Seeding extremum per base manifold. A base ascending 2-manifold is the basin
  // of exactly one minimum, and every other vertex in the basin flows down to it,
  // so the pixel attaining filt_min IS that minimum -- and symmetrically filt_max
  // for a descending manifold's maximum. Deriving it from the labeling rather
  // than from MSCEER's criticalPoints() keeps this independent of node-id
  // semantics (which differ between the serial and partitioned builders) and
  // always yields a real pixel of the manifold: a maximum is a 2-cell, so its
  // native position is a half-pixel whose value need not even be attained inside
  // the manifold.
  const int ext_radius = spec.extremum_sample_radius > 0 ? spec.extremum_sample_radius
                                                         : cfg.extremum_sample_radius;
  if (spec.extremum) {
    for (std::size_t c = 0; c < num_base; ++c) {
      if (leaf[c].area == 0) continue;
      const std::size_t i = arg_ext[c];
      const int x = static_cast<int>(i % static_cast<std::size_t>(width));
      const int y = static_cast<int>(i / static_cast<std::size_t>(width));
      leaf[c].ext_x = static_cast<float>(x);
      leaf[c].ext_y = static_cast<float>(y);
      leaf[c].ext_filtered = ascending ? leaf[c].filt_min : leaf[c].filt_max;
      // Every measurement channel is sampled at the same pixel, so a scale-space
      // stack reports what the seed looks like at each scale -- the discriminating
      // signal a later material/air model wants.
      impl_->leaf_channels.sample_ext(c, x, y, width, height, ext_radius, chan);
    }
  }
  // A base manifold with no pixels would otherwise carry +/-FLT_MAX sentinels
  // into a merge and out to a CSV.
  for (std::size_t c = 0; c < num_base; ++c) {
    if (leaf[c].area == 0) impl_->leaf_channels.clear_region(c);
  }

  // Initial persistence from cfg (native cancellation).
  float persistence = 0.0f;
  if (cfg.persistence_absolute.has_value()) {
    persistence = *cfg.persistence_absolute;
  } else if (cfg.persistence_percent.has_value()) {
    persistence = impl_->value_range * (*cfg.persistence_percent / 100.0f);
  }
  select_persistence(persistence);
}

int Msc2DPipeline::width() const { return impl_->width; }
int Msc2DPipeline::height() const { return impl_->height; }
float Msc2DPipeline::value_range() const { return impl_->value_range; }
float Msc2DPipeline::current_persistence() const { return impl_->current_persistence; }
float Msc2DPipeline::base_relevance_floor() const { return impl_->base_relevance_floor; }
float Msc2DPipeline::base_relevance_ceiling() const { return impl_->base_relevance_ceiling; }

void Msc2DPipeline::select_persistence(float persistence_absolute) {
  impl_->current_persistence = persistence_absolute;
  // Phase timing, opt-in via MSSEG_TIME_MSC=1. Re-thresholding is the
  // interactive path's dominant cost, so it needs to be attributable.
  const bool time_phases = std::getenv("MSSEG_TIME_MSC") != nullptr;
  const auto t_start = std::chrono::steady_clock::now();
  auto lap = [&](const char* what, std::chrono::steady_clock::time_point& from) {
    if (!time_phases) return;
    const auto now = std::chrono::steady_clock::now();
    std::fprintf(stderr, "  [msc2d] %-10s %6.1f ms\n", what,
                 std::chrono::duration<double, std::milli>(now - from).count());
    from = now;
  };
  auto mark = t_start;

  // Native cancellation: set the living value, then read the surviving-extremum
  // labels. ascending/descending2Manifolds() remaps each base minimum to its
  // living representative via the cancellation hierarchy (cheap -- no gradient or
  // base-manifold recompute). Adjacent-basin merges keep features CONNECTED.
  impl_->msc.setPersistence(persistence_absolute);
  const GInt::Msc2D::LabelImage living_img =
      impl_->ascending ? impl_->msc.ascending2Manifolds() : impl_->msc.descending2Manifolds();
  lap("msceer", mark);

  const std::size_t npix = impl_->base_labels.size();
  impl_->labels.assign(npix, -1);
  for (std::size_t i = 0; i < npix && i < living_img.labels.size(); ++i) {
    const int nid = living_img.labels[i];
    if (nid < 0) continue;
    const auto it = impl_->nid_to_compact.find(nid);
    if (it != impl_->nid_to_compact.end()) impl_->labels[i] = it->second;
  }
  lap("relabel", mark);

  // Aggregate the precomputed per-base leaf stats up to the living features. Every
  // pixel of a base manifold maps to the same living representative, so a pixelwise
  // base->living map is exact; roll up leaf_stats via accumulate().
  std::unordered_map<int, int> base_to_living;
  base_to_living.reserve(impl_->leaf_stats.size() * 2);
  for (std::size_t i = 0; i < npix; ++i) {
    const int b = impl_->base_labels[i];
    const int L = impl_->labels[i];
    if (b >= 0 && L >= 0) base_to_living.emplace(b, L);   // consistent per base manifold
  }
  lap("base_map", mark);

  // Assign each living feature a dense row first, so the geometry vector and the
  // flat channel table can be filled in lockstep -- slot k of row r is always
  // channels()[k] of features()[r]. Rows come out ordered by first contributing
  // base manifold, which also makes the feature order deterministic (it used to
  // follow unordered_map iteration).
  const int n_leaves = static_cast<int>(impl_->leaf_stats.size());
  std::unordered_map<int, int> living_to_row;
  std::vector<int> row_living;
  living_to_row.reserve(impl_->leaf_stats.size() * 2);
  for (int b = 0; b < n_leaves; ++b) {
    if (impl_->leaf_stats[static_cast<std::size_t>(b)].area == 0) continue;
    const auto it = base_to_living.find(b);
    if (it == base_to_living.end()) continue;
    const auto ins = living_to_row.emplace(it->second, static_cast<int>(row_living.size()));
    if (ins.second) row_living.push_back(it->second);
  }

  const std::size_t n_rows = row_living.size();
  const std::size_t n_ch = impl_->leaf_channels.channels();
  impl_->features.assign(n_rows, Msc2DFeatureStat{});
  impl_->feature_channels.reset(n_rows, n_ch, impl_->stats);
  for (std::size_t r = 0; r < n_rows; ++r) {
    impl_->features[r].feature_id = static_cast<NodeId>(row_living[r]);
  }

  for (int b = 0; b < n_leaves; ++b) {
    const Msc2DFeatureStat& src = impl_->leaf_stats[static_cast<std::size_t>(b)];
    if (src.area == 0) continue;
    const auto it = base_to_living.find(b);
    if (it == base_to_living.end()) continue;
    const std::size_t row = static_cast<std::size_t>(living_to_row[it->second]);
    accumulate(impl_->features[row], src);
    impl_->feature_channels.merge_region(row, impl_->leaf_channels,
                                         static_cast<std::size_t>(b));
  }

  for (std::size_t r = 0; r < n_rows; ++r) {
    Msc2DFeatureStat& s = impl_->features[r];
    const int id = row_living[r];
    // A merged feature keeps the SURVIVING extremum, and the living compact id
    // is exactly that surviving extremum's own base-manifold id -- so the seed
    // is a direct lookup, not an accumulation. Every channel's sample at that
    // seed is carried over as a tuple for the same reason.
    if (id >= 0 && id < n_leaves) {
      const Msc2DFeatureStat& seed = impl_->leaf_stats[static_cast<std::size_t>(id)];
      s.ext_x = seed.ext_x;
      s.ext_y = seed.ext_y;
      s.ext_filtered = seed.ext_filtered;
      impl_->feature_channels.copy_ext(r, impl_->leaf_channels, static_cast<std::size_t>(id));
    }
    s.base_relevance_floor = impl_->base_relevance_floor;
    s.base_relevance_ceiling = impl_->base_relevance_ceiling;
  }
  lap("rollup", mark);
}

const std::vector<int>& Msc2DPipeline::labels() const { return impl_->labels; }
std::vector<Msc2DFeatureStat> Msc2DPipeline::feature_stats() const { return impl_->features; }

const std::vector<ResolvedStatChannel>& Msc2DPipeline::channels() const { return impl_->channels; }
const ChannelStats& Msc2DPipeline::feature_channels() const { return impl_->feature_channels; }

const StatsSpec& Msc2DPipeline::stats() const { return impl_->stats; }

}  // namespace msseg
