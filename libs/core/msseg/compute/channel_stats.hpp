#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "msseg/workflow/params.hpp"

namespace msseg {

// Mergeable aggregate of one measurement channel over one region's pixels.
//
// Sums are mergeable, so per-base-manifold cells combine cheaply when rolled up
// to living features at a new persistence, and per-slice cells combine into 3D
// features. mean = sum/area and std = sqrt(sumsq/area - mean^2) are DERIVED at
// projection time, never stored -- the same contract Msc2DFeatureStat has always
// had for the base channel.
struct ChannelAccum {
  double sum = 0.0;
  double sumsq = 0.0;
  float min = 0.0f;
  float max = 0.0f;
};

// Flat region-major table of per-channel accumulators.
//
// Layout is `cell(region, slot) = cells[region * n_channels + slot]`, i.e. the
// same shape PerSliceAcc (mscoupon/matcher.cpp) already uses one dimension down.
// Region-major keeps one region's channels contiguous, which is the order both
// the accumulation loop and the rollup touch them in.
//
// Channels are addressed by SLOT throughout. Names are resolved to slots once
// per run by resolve_stat_channels(); nothing here ever hashes a string, because
// this is the pipeline's hot path -- a 12-channel stack means twelve reads per
// pixel, and the per-feature row is rebuilt on every persistence change.
class ChannelStats {
 public:
  ChannelStats() = default;

  // `n_channels` must match the resolved channel list the caller will project
  // with. `spec` decides which of sum/sumsq/min/max are actually maintained;
  // whatever is not asked for stays at zero rather than at a sentinel, so an
  // unused field lands in a CSV as an obvious 0 and not as 3.4e38.
  void reset(std::size_t n_regions, std::size_t n_channels, const StatsSpec& spec) {
    n_regions_ = n_regions;
    n_channels_ = n_channels;
    want_sums_ = spec.mean || spec.std;
    want_sumsq_ = spec.std;
    // min/max are also what `relevance_base` is computed from, so they stay on
    // whenever relevance is, even if a workflow asked for neither reduction. The
    // extremum does NOT need them: it samples its pixel directly.
    want_extent_ = spec.min || spec.max || spec.relevance;
    want_ext_ = spec.extremum;

    cells_.assign(n_regions_ * n_channels_, ChannelAccum{});
    if (want_extent_) {
      for (auto& c : cells_) {
        c.min = std::numeric_limits<float>::max();
        c.max = std::numeric_limits<float>::lowest();
      }
    }
    ext_.assign(want_ext_ ? n_regions_ * n_channels_ : 0, 0.0f);
  }

  std::size_t regions() const { return n_regions_; }
  std::size_t channels() const { return n_channels_; }
  bool empty() const { return n_channels_ == 0 || n_regions_ == 0; }
  bool has_ext() const { return want_ext_; }

  ChannelAccum& cell(std::size_t region, std::size_t slot) {
    return cells_[region * n_channels_ + slot];
  }
  const ChannelAccum& cell(std::size_t region, std::size_t slot) const {
    return cells_[region * n_channels_ + slot];
  }

  // The channel's value at the region's seeding extremum. Unlike the cells this
  // is NOT mergeable: a merged feature inherits the SURVIVING extremum, so it is
  // stamped from the surviving constituent rather than accumulated.
  float ext(std::size_t region, std::size_t slot) const {
    return want_ext_ ? ext_[region * n_channels_ + slot] : 0.0f;
  }
  void set_ext(std::size_t region, std::size_t slot, float value) {
    if (want_ext_) ext_[region * n_channels_ + slot] = value;
  }

  // Accumulate pixel `pixel` into `region` across every channel.
  // `data[slot]` is the channel raster, as handed over by StatChannelBank.
  void add(std::size_t region, std::size_t pixel, const std::vector<const float*>& data) {
    ChannelAccum* row = cells_.data() + region * n_channels_;
    for (std::size_t k = 0; k < n_channels_; ++k) {
      const float v = data[k][pixel];
      ChannelAccum& a = row[k];
      if (want_sums_) {
        a.sum += v;
        if (want_sumsq_) a.sumsq += static_cast<double>(v) * v;
      }
      if (want_extent_) {
        a.min = std::min(a.min, v);
        a.max = std::max(a.max, v);
      }
    }
  }

  // Stamp every channel's value at `pixel` as `region`'s extremum sample.
  // `radius` > 0 averages the (2r+1)^2 window instead of taking the lone pixel,
  // trading the exact critical value for noise robustness.
  void sample_ext(std::size_t region, int x, int y, int width, int height, int radius,
                  const std::vector<const float*>& data) {
    if (!want_ext_) return;
    float* row = ext_.data() + region * n_channels_;
    if (radius <= 0) {
      const std::size_t pixel = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                static_cast<std::size_t>(x);
      for (std::size_t k = 0; k < n_channels_; ++k) row[k] = data[k][pixel];
      return;
    }
    const int x0 = std::max(0, x - radius), x1 = std::min(width - 1, x + radius);
    const int y0 = std::max(0, y - radius), y1 = std::min(height - 1, y + radius);
    const double n = static_cast<double>(x1 - x0 + 1) * static_cast<double>(y1 - y0 + 1);
    for (std::size_t k = 0; k < n_channels_; ++k) {
      const float* channel = data[k];
      double acc = 0.0;
      for (int yy = y0; yy <= y1; ++yy) {
        const std::size_t base = static_cast<std::size_t>(yy) * static_cast<std::size_t>(width);
        for (int xx = x0; xx <= x1; ++xx) acc += channel[base + static_cast<std::size_t>(xx)];
      }
      row[k] = static_cast<float>(acc / n);
    }
  }

  // Merge every channel of `src_region` (in `src`) into `dst_region` here.
  // Extremum samples are deliberately NOT merged; see set_ext.
  void merge_region(std::size_t dst_region, const ChannelStats& src, std::size_t src_region) {
    ChannelAccum* dst = cells_.data() + dst_region * n_channels_;
    const ChannelAccum* s = src.cells_.data() + src_region * src.n_channels_;
    for (std::size_t k = 0; k < n_channels_; ++k) {
      dst[k].sum += s[k].sum;
      dst[k].sumsq += s[k].sumsq;
      dst[k].min = std::min(dst[k].min, s[k].min);
      dst[k].max = std::max(dst[k].max, s[k].max);
    }
  }

  // Copy every channel's extremum sample from one region of `src`. Used where a
  // merged feature inherits the surviving constituent's whole tuple.
  void copy_ext(std::size_t dst_region, const ChannelStats& src, std::size_t src_region) {
    if (!want_ext_ || !src.want_ext_) return;
    float* dst = ext_.data() + dst_region * n_channels_;
    const float* s = src.ext_.data() + src_region * src.n_channels_;
    for (std::size_t k = 0; k < n_channels_; ++k) dst[k] = s[k];
  }

  // Append every region of `src` after this table's current regions.
  //
  // The cross-slice matcher accumulates one flat plane over all slices' CC
  // nodes, so a node is addressed by its global index without a per-slice
  // indirection. An empty table adopts `src`'s shape and flags; otherwise the
  // channel counts must agree, which they do because both come from the same
  // run's resolved channel list.
  void append(const ChannelStats& src) {
    if (n_channels_ == 0 && n_regions_ == 0) {
      n_channels_ = src.n_channels_;
      want_sums_ = src.want_sums_;
      want_sumsq_ = src.want_sumsq_;
      want_extent_ = src.want_extent_;
      want_ext_ = src.want_ext_;
    }
    if (src.n_channels_ != n_channels_) return;
    cells_.insert(cells_.end(), src.cells_.begin(), src.cells_.end());
    if (want_ext_ && src.want_ext_) ext_.insert(ext_.end(), src.ext_.begin(), src.ext_.end());
    else if (want_ext_) ext_.resize(ext_.size() + src.n_regions_ * n_channels_, 0.0f);
    n_regions_ += src.n_regions_;
  }

  // Replace the min/max sentinels of a region that ended up with no pixels, so
  // an empty region reads as 0 rather than as +/-FLT_MAX in a CSV.
  void clear_region(std::size_t region) {
    ChannelAccum* row = cells_.data() + region * n_channels_;
    for (std::size_t k = 0; k < n_channels_; ++k) row[k] = ChannelAccum{};
  }

  const std::vector<ChannelAccum>& cells() const { return cells_; }

 private:
  std::size_t n_regions_ = 0;
  std::size_t n_channels_ = 0;
  bool want_sums_ = false;
  bool want_sumsq_ = false;
  bool want_extent_ = false;
  bool want_ext_ = false;
  std::vector<ChannelAccum> cells_;
  std::vector<float> ext_;
};

}  // namespace msseg
