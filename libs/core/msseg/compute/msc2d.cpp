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

#include "msseg/compute/gpu/stats_gpu.hpp"
#include "msseg/filter/filter_stage.hpp"
#include "msseg/graph/msc_graph.hpp"
#include "msseg/workflow/stat_channels.hpp"

#ifdef MSSEG_HAVE_DIFFG_GPU
#include "api.h"  // diffg_gpu: device-resident / JIT GPU filter bank
#endif

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

// ComputeOptions::useGpuGradient appeared with MSCEER's cuda-gradient work;
// detect the member so an older pin still compiles (the flag then only warns).
template <typename Options, typename = void>
struct HasUseGpuGradient : std::false_type {};

template <typename Options>
struct HasUseGpuGradient<Options,
                         std::void_t<decltype(std::declval<Options&>().useGpuGradient)>>
    : std::true_type {};

// ComputeOptions::simplification selects the MSC hierarchy or the merge-forest
// extremum network; detected so an older pin still builds (and warns).
template <typename Options, typename = void>
struct HasSimplification : std::false_type {};

template <typename Options>
struct HasSimplification<Options,
                         std::void_t<decltype(std::declval<Options&>().simplification)>>
    : std::true_type {};

// Region-scale facade (baseToLiving/paintLabels), from MSCEER's cuda-gradient
// branch: lets a re-select work per base REGION (~100k entries) instead of per
// PIXEL (~10M hash probes), and paint through the persistent GPU label context
// when one exists. Detected so older pins keep the per-pixel path.
template <typename MscType, typename = void>
struct HasRegionApi : std::false_type {};

template <typename MscType>
struct HasRegionApi<
    MscType,
    std::void_t<decltype(std::declval<MscType&>().baseToLiving(true)),
                decltype(std::declval<MscType&>().paintLabels(
                    true, static_cast<const int*>(nullptr), 0, static_cast<int*>(nullptr)))>>
    : std::true_type {};

// releaseGpuResources() (VRAM residency control) is newer than the region API;
// detect it separately so intermediate MSCEER pins still build.
template <typename MscType, typename = void>
struct HasReleaseGpu : std::false_type {};

template <typename MscType>
struct HasReleaseGpu<MscType,
                     std::void_t<decltype(std::declval<MscType&>().releaseGpuResources())>>
    : std::true_type {};

// Runtime kill-switch (MSSEG_REGION_SELECT=0) so the two select paths can be
// A/B-compared for byte identity on real data.
bool region_select_enabled() {
  const char* env = std::getenv("MSSEG_REGION_SELECT");
  return !(env != nullptr && env[0] == '0');
}

// Which simplification the caller asked for, with MSSEG_SIMPLIFICATION as the
// runtime kill-switch (mirrors MSSEG_REGION_SELECT / MSSEG_GPU_STATS, so an A/B
// run can force the MSC without editing a config).
bool merge_forest_wanted(const Msc2DParams& cfg) {
  const char* env = std::getenv("MSSEG_SIMPLIFICATION");
  if (env != nullptr && env[0] != '\0') return std::string(env) == "merge_forest";
  return cfg.simplification == "merge_forest";
}

// GPU statistics accumulation: opt-in via msc.use_gpu_stats, defaulting to
// wherever use_gpu_gradient points, with MSSEG_GPU_STATS=0 as the runtime
// kill-switch (mirrors MSSEG_REGION_SELECT for A/B parity runs).
bool gpu_stats_wanted(const Msc2DParams& cfg) {
  const char* env = std::getenv("MSSEG_GPU_STATS");
  if (env != nullptr && env[0] == '0') return false;
  return cfg.use_gpu_stats.value_or(cfg.use_gpu_gradient);
}

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
    if constexpr (HasUseGpuGradient<typename MscType::ComputeOptions>::value) {
      options.useGpuGradient = cfg.use_gpu_gradient;
    } else {
      if (cfg.use_gpu_gradient) {
        std::fprintf(stderr,
                     "msc2d: msc.use_gpu_gradient requested but the linked "
                     "msc_2d_lib predates ComputeOptions::useGpuGradient; "
                     "using the CPU gradient.\n");
      }
    }
    if constexpr (HasSimplification<typename MscType::ComputeOptions>::value) {
      options.simplification = merge_forest_wanted(cfg)
                                   ? MscType::Simplification::MergeForest
                                   : MscType::Simplification::MscHierarchy;
    } else {
      if (merge_forest_wanted(cfg)) {
        std::fprintf(stderr,
                     "msc2d: msc.simplification='merge_forest' requested but the "
                     "linked msc_2d_lib predates ComputeOptions::simplification; "
                     "using the MSC hierarchy.\n");
      }
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
  // MSCEER-compact base region id -> our compact id (empty when the linked
  // msc_2d_lib lacks the region-scale facade). Built once at persistence -1,
  // where baseToLiving() returns each region's own extremum node id.
  std::vector<int> bridge;
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

namespace {

// GPU leaf accumulation: label CSR + deterministic segmented reduces over
// device-resident channels (diffg JIT bank -- bounded scratch, channels
// materialized one at a time). Fills leaf_stats + leaf_channels exactly like
// the CPU pixel loop; labels/area/bbox/min/max/extremum are exact, sums differ
// only in floating-point association. Returns false on ANY failure, leaving the
// caller to re-init the leaf tables and run the CPU loop. Templated on the
// (private) Impl type so it can stay a free function in this TU.
template <typename ImplT>
bool try_gpu_accumulate(ImplT& impl, const diffg::Image<float>& base,
                        const diffg::Image<float>& filtered, const StatsSpec& spec,
                        int ext_radius) {
#ifndef MSSEG_HAVE_DIFFG_GPU
  (void)impl;
  (void)base;
  (void)filtered;
  (void)spec;
  (void)ext_radius;
  return false;
#else
  if (!gpustats::available()) return false;
  const int width = impl.width;
  const int height = impl.height;
  const std::size_t num_base = impl.leaf_stats.size();
  if (num_base == 0) return false;

  const bool time_phases = std::getenv("MSSEG_TIME_MSC") != nullptr;
  auto mark = std::chrono::steady_clock::now();
  auto lap = [&](const char* what) {
    if (!time_phases) return;
    const auto now = std::chrono::steady_clock::now();
    std::fprintf(stderr, "  [gpu_stats] %-10s %6.1f ms\n", what,
                 std::chrono::duration<double, std::milli>(now - mark).count());
    mark = now;
  };

  // Mirror ChannelStats::reset's decisions about which aggregates exist.
  const bool want_filt_extent = spec.needs_filtered_extent();
  const bool want_sums = spec.mean || spec.std;
  const bool want_sumsq = spec.std;
  const bool want_extent = spec.min || spec.max || spec.relevance;
  const bool want_ext = spec.extremum && want_filt_extent;

  struct StatsGuard {
    gpustats::SliceStats* s = nullptr;
    ~StatsGuard() { gpustats::destroy(s); }
  } sg;
  struct BankGuard {
    diffg_gpu::GpuFilterBank* b = nullptr;
    ~BankGuard() {
      if (b) diffg_gpu::gpu_filter_bank_destroy(b);
    }
  } bg;

  sg.s = gpustats::create(width, height);
  if (!sg.s) return false;

  // Labels: prefer the device image MSCEER's label context can paint directly
  // (our compact ids, via the bridge); fall back to uploading the host image.
  bool have_labels = false;
  if constexpr (HasRegionApi<GInt::Msc2D::Msc2D>::value) {
    if (!impl.bridge.empty()) {
      const void* dev = impl.msc.paintLabelsDevice(impl.ascending, impl.bridge.data(),
                                                   static_cast<int>(impl.bridge.size()));
      if (dev != nullptr) {
        have_labels = gpustats::set_labels(sg.s, nullptr, dev, static_cast<int>(num_base));
      }
    }
  }
  if (!have_labels) {
    have_labels =
        gpustats::set_labels(sg.s, impl.base_labels.data(), nullptr, static_cast<int>(num_base));
  }
  if (!have_labels) return false;
  lap("csr");

  const void* d_base = gpustats::upload(sg.s, base.data(), 0);
  if (!d_base) return false;
  const void* d_filt = nullptr;
  if (want_filt_extent || spec.filtered_channel) {
    d_filt = gpustats::upload(sg.s, filtered.data(), 1);
    if (!d_filt) return false;
  }
  lap("upload");

  // JIT filter bank over the resident base raster for the derived channels.
  // Requests dedup exactly like build_stat_channels: a hessian entry is one
  // request covering two consecutive resolved slots.
  const std::vector<ResolvedStatChannel>& channels = impl.channels;
  std::vector<diffg_gpu::FilterRequest> reqs;
  std::vector<int> bank_slot(channels.size(), -1);
  int next_slot = 0;
  for (std::size_t k = 0; k < channels.size(); ++k) {
    const ResolvedStatChannel& c = channels[k];
    if (c.kind == "base" || c.kind == "filtered") continue;
    if (c.slot_in_request == 0) {
      diffg_gpu::FilterRequest r;
      if (c.kind == "blur") {
        r.kind = diffg_gpu::BankFilterKind::Gaussian;
      } else if (c.kind == "gradmag" || c.kind == "edges") {
        r.kind = diffg_gpu::BankFilterKind::GradientMagnitude;
      } else if (c.kind == "laplacian") {
        r.kind = diffg_gpu::BankFilterKind::Laplacian;
      } else if (c.kind == "hessian") {
        r.kind = diffg_gpu::BankFilterKind::HessianEigenvalues;
      } else {
        return false;  // unknown derived kind -> let the CPU path report it
      }
      r.sigma = c.sigma;
      r.sort_by_absolute_value = c.sort_by_absolute_value;
      reqs.push_back(r);
    }
    bank_slot[k] = next_slot++;
  }
  if (!reqs.empty()) {
    bg.b = diffg_gpu::gpu_filter_bank_create_jit_device(d_base, width, height, reqs.data(),
                                                        static_cast<int>(reqs.size()));
    if (!bg.b || diffg_gpu::gpu_filter_bank_channels(bg.b) != next_slot) return false;
  }
  lap("bank");

  // Geometry + the seeding extremum (arg_ext stays resident for sampling).
  std::vector<int> area(num_base), min_x(num_base), min_y(num_base), max_x(num_base),
      max_y(num_base);
  std::vector<float> fmin, fmax;
  std::vector<std::int64_t> arg_ext;
  if (want_filt_extent) {
    fmin.resize(num_base);
    fmax.resize(num_base);
    arg_ext.resize(num_base);
  }
  if (!gpustats::geometry(sg.s, want_filt_extent ? d_filt : nullptr, impl.ascending, area.data(),
                          min_x.data(), min_y.data(), max_x.data(), max_y.data(),
                          want_filt_extent ? fmin.data() : nullptr,
                          want_filt_extent ? fmax.data() : nullptr,
                          want_filt_extent ? arg_ext.data() : nullptr)) {
    return false;
  }
  lap("geometry");

  // Per channel: materialize (JIT), reduce, and extremum-sample while resident.
  std::vector<double> sum, sumsq;
  std::vector<float> mn, mx, ext_sample;
  if (want_sums) sum.resize(num_base);
  if (want_sumsq) sumsq.resize(num_base);
  if (want_extent) {
    mn.resize(num_base);
    mx.resize(num_base);
  }
  if (want_ext) ext_sample.resize(num_base);
  for (std::size_t k = 0; k < channels.size(); ++k) {
    const ResolvedStatChannel& c = channels[k];
    const void* d_ch = nullptr;
    if (c.kind == "base") {
      d_ch = d_base;
    } else if (c.kind == "filtered") {
      d_ch = d_filt;
    } else {
      d_ch = diffg_gpu::gpu_filter_bank_evaluate_device(bg.b, bank_slot[k]);
    }
    if (!d_ch) return false;
    if ((want_sums || want_extent) &&
        !gpustats::reduce_channel(sg.s, d_ch, want_sums ? sum.data() : nullptr,
                                  want_sumsq ? sumsq.data() : nullptr,
                                  want_extent ? mn.data() : nullptr,
                                  want_extent ? mx.data() : nullptr)) {
      return false;
    }
    for (std::size_t r = 0; r < num_base; ++r) {
      if (area[r] == 0) continue;
      ChannelAccum& a = impl.leaf_channels.cell(r, k);
      if (want_sums) a.sum = sum[r];
      if (want_sumsq) a.sumsq = sumsq[r];
      if (want_extent) {
        a.min = mn[r];
        a.max = mx[r];
      }
    }
    if (want_ext) {
      if (!gpustats::sample_ext(sg.s, d_ch, ext_radius, ext_sample.data())) return false;
      for (std::size_t r = 0; r < num_base; ++r) {
        if (area[r] > 0) impl.leaf_channels.set_ext(r, k, ext_sample[r]);
      }
    }
  }
  lap("channels");

  for (std::size_t r = 0; r < num_base; ++r) {
    Msc2DFeatureStat& s = impl.leaf_stats[r];
    if (area[r] == 0) continue;  // keeps the CPU-identical empty-region state
    s.area = area[r];
    s.min_x = min_x[r];
    s.min_y = min_y[r];
    s.max_x = max_x[r];
    s.max_y = max_y[r];
    if (want_filt_extent) {
      s.filt_min = fmin[r];
      s.filt_max = fmax[r];
    }
    if (want_ext && arg_ext[r] >= 0) {
      s.ext_x = static_cast<float>(arg_ext[r] % width);
      s.ext_y = static_cast<float>(arg_ext[r] / width);
      s.ext_filtered = impl.ascending ? s.filt_min : s.filt_max;
    }
  }
  lap("stamp");
  return true;
#endif  // MSSEG_HAVE_DIFFG_GPU
}

}  // namespace

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

  // Bridge MSCEER's compact base ids to ours while persistence is still -1:
  // baseToLiving() here maps each base region to its own (uncancelled) extremum
  // node id, which is exactly what nid_to_compact keys on. Our compaction (and
  // therefore every downstream id, row order, and CSV) is unchanged -- the
  // bridge only lets select_persistence() work region-scale.
  impl_->bridge.clear();
  if constexpr (HasRegionApi<GInt::Msc2D::Msc2D>::value) {
    const std::vector<int>& r0 = impl_->msc.baseToLiving(ascending);
    impl_->bridge.assign(r0.size(), -1);
    for (std::size_t c = 0; c < r0.size(); ++c) {
      const auto it = impl_->nid_to_compact.find(r0[c]);
      if (it != impl_->nid_to_compact.end()) impl_->bridge[c] = it->second;
    }
  }

  // Per-base-manifold leaf statistics (one pass over pixels, both images),
  // indexed by compact base id. These are aggregated up to living features on
  // each select_persistence() -- the base constituents are what we precompute.
  const StatsSpec& spec = cfg.stats;
  impl_->stats = spec;
  const bool want_filt_extent = spec.needs_filtered_extent();

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
  auto init_leaves = [&](std::size_t n_ch_init) {
    leaf.assign(num_base, Msc2DFeatureStat{});
    impl_->leaf_channels.reset(num_base, n_ch_init, spec);
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
  };
  const int ext_radius = spec.extremum_sample_radius > 0 ? spec.extremum_sample_radius
                                                         : cfg.extremum_sample_radius;

  // GPU accumulation first: the derived channels never materialize on the host
  // (diffg JIT bank) and the reduces run over the resident label CSR. Callers
  // that hand in an external bank already paid for the host rasters, so they
  // keep the CPU loop. Any failure re-inits the leaves and falls through.
  bool gpu_done = false;
  if (external_bank == nullptr && gpu_stats_wanted(cfg)) {
    impl_->channels = resolve_stat_channels(spec);
    init_leaves(impl_->channels.size());
    gpu_done = try_gpu_accumulate(*impl_, base, filtered, spec, ext_radius);
    if (!gpu_done && cfg.use_gpu_stats.value_or(false)) {
      std::fprintf(stderr,
                   "msc2d: msc.use_gpu_stats requested but the GPU accumulation "
                   "is unavailable; using the CPU loop.\n");
    }
  }

  if (!gpu_done) {
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
    const std::vector<const float*>& chan = bank.data;
    init_leaves(bank.size());

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

  // Priming a long sequence must not accumulate one GPU label context per
  // slice; the active slice lazily re-uploads on its first interactive select.
  release_gpu();
}

void Msc2DPipeline::release_gpu() {
  if constexpr (HasReleaseGpu<GInt::Msc2D::Msc2D>::value) {
    impl_->msc.releaseGpuResources();
  }
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

  const std::size_t npix = impl_->base_labels.size();
  const int n_leaves = static_cast<int>(impl_->leaf_stats.size());
  // Our-compact base id -> our-compact living id (-1 = no living owner). Both
  // select paths below fill this; the rollup after it is shared.
  std::vector<int> b2l(static_cast<std::size_t>(n_leaves), -1);

  const bool use_region_select =
      HasRegionApi<GInt::Msc2D::Msc2D>::value && !impl_->bridge.empty() && region_select_enabled();
  if constexpr (HasRegionApi<GInt::Msc2D::Msc2D>::value) {
    if (use_region_select) {
      // Region-scale select: no LabelImage is materialized. baseToLiving() runs
      // the hierarchy walk per base REGION; the ~M-entry lut translates its node
      // ids into our compact ids; paintLabels() stamps the pixels through the
      // persistent GPU label context when one exists (parallel CPU gather
      // otherwise). Byte-identical to the per-pixel path -- MSCEER's parity gate
      // holds paintLabels(baseToLiving) equal to the manifold image.
      impl_->msc.setPersistence(persistence_absolute);
      const std::vector<int>& reg = impl_->msc.baseToLiving(impl_->ascending);
      lap("msceer", mark);

      const std::size_t m = std::min(reg.size(), impl_->bridge.size());
      std::vector<int> lut(reg.size(), -1);
      for (std::size_t c = 0; c < m; ++c) {
        const int nid = reg[c];
        if (nid < 0) continue;
        const auto it = impl_->nid_to_compact.find(nid);
        if (it != impl_->nid_to_compact.end()) lut[c] = it->second;
      }
      impl_->labels.assign(npix, -1);
      impl_->msc.paintLabels(impl_->ascending, lut.data(), static_cast<int>(lut.size()),
                             impl_->labels.data());
      lap("relabel", mark);

      for (std::size_t c = 0; c < m; ++c) {
        const int b = impl_->bridge[c];
        if (b >= 0 && lut[c] >= 0) b2l[static_cast<std::size_t>(b)] = lut[c];
      }
      lap("base_map", mark);
    }
  }
  if (!use_region_select) {
    // Per-pixel path (older msc_2d_lib pins, or MSSEG_REGION_SELECT=0): read the
    // living-extremum LabelImage and hash every pixel's node id to our compact id.
    impl_->msc.setPersistence(persistence_absolute);
    const GInt::Msc2D::LabelImage living_img =
        impl_->ascending ? impl_->msc.ascending2Manifolds() : impl_->msc.descending2Manifolds();
    lap("msceer", mark);

    impl_->labels.assign(npix, -1);
    for (std::size_t i = 0; i < npix && i < living_img.labels.size(); ++i) {
      const int nid = living_img.labels[i];
      if (nid < 0) continue;
      const auto it = impl_->nid_to_compact.find(nid);
      if (it != impl_->nid_to_compact.end()) impl_->labels[i] = it->second;
    }
    lap("relabel", mark);

    // Every pixel of a base manifold maps to the same living representative, so
    // a pixelwise base->living scan is exact (first write wins, all agree).
    for (std::size_t i = 0; i < npix; ++i) {
      const int b = impl_->base_labels[i];
      const int L = impl_->labels[i];
      if (b >= 0 && L >= 0 && b2l[static_cast<std::size_t>(b)] < 0)
        b2l[static_cast<std::size_t>(b)] = L;
    }
    lap("base_map", mark);
  }

  // Assign each living feature a dense row first, so the geometry vector and the
  // flat channel table can be filled in lockstep -- slot k of row r is always
  // channels()[k] of features()[r]. Rows come out ordered by first contributing
  // base manifold, which also makes the feature order deterministic. Living ids
  // are our own compact ids, so a flat vector replaces the old hash map.
  std::vector<int> living_to_row(static_cast<std::size_t>(n_leaves), -1);
  std::vector<int> row_living;
  for (int b = 0; b < n_leaves; ++b) {
    if (impl_->leaf_stats[static_cast<std::size_t>(b)].area == 0) continue;
    const int L = b2l[static_cast<std::size_t>(b)];
    if (L < 0) continue;
    if (living_to_row[static_cast<std::size_t>(L)] < 0) {
      living_to_row[static_cast<std::size_t>(L)] = static_cast<int>(row_living.size());
      row_living.push_back(L);
    }
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
    const int L = b2l[static_cast<std::size_t>(b)];
    if (L < 0) continue;
    const std::size_t row = static_cast<std::size_t>(living_to_row[static_cast<std::size_t>(L)]);
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
