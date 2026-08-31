#pragma once

// GPU per-region statistics accumulation for the 2D pipeline.
//
// Plain C++ boundary in the diffg_gpu/dgrad_gpu style: no CUDA types, raw
// pointers in, host buffers out. The real implementation (stats_gpu.cu) is
// compiled only with MSSEG_GPU=ON; otherwise stats_gpu_stub.cpp makes every
// call fail cleanly (available() == false, create() == nullptr), so callers
// need no #ifdefs -- they just fall back to the CPU accumulation loop.
//
// The handle owns a pixel CSR over the region labeling (built once with a
// deterministic stable sort, so every reduction has a FIXED pixel order and is
// run-to-run reproducible -- no atomics anywhere). Channel rasters come in as
// device pointers; they interoperate with pointers produced by other
// static-cudart libraries in this process (MSCEER's paintLabelsDevice, diffg's
// GPU filter bank) via the shared CUDA primary context.

#include <cstdint>

namespace msseg {
namespace gpustats {

// True when a usable CUDA device is present (always false in stub builds).
bool available();

struct SliceStats;  // opaque

SliceStats* create(std::int64_t width, std::int64_t height);
void destroy(SliceStats* s);  // safe on nullptr

// Set the per-pixel region labels (dense compact ids, -1 = unlabeled) and
// build the pixel CSR. Exactly one of host_labels / dev_labels should be
// non-null; dev_labels is a borrowed int32 device pointer (e.g. from
// Msc2D::paintLabelsDevice) whose contents are copied, so it may be
// invalidated afterwards. Returns false on any CUDA failure.
bool set_labels(SliceStats* s, const int* host_labels, const void* dev_labels, int n_regions);

// Upload a host float raster (width*height); the returned device pointer is
// owned by the handle and freed on destroy. `slot` is 0 or 1 (two rasters --
// base and filtered -- is all the pipeline stages need resident at once);
// re-uploading a slot frees the previous raster. nullptr on failure.
const void* upload(SliceStats* s, const float* host, int slot);

// Deterministic segmented reduces of one device channel raster over the CSR.
// Each non-null output receives n_regions values (empty regions get 0 / the
// +-FLT_MAX extent sentinels the CPU loop also starts from).
bool reduce_channel(SliceStats* s, const void* dev_channel, double* out_sum,
                    double* out_sumsq, float* out_min, float* out_max);

// Region geometry off the label CSR plus the seeding extremum off the
// filtered raster: area, bbox, filtered extent, and arg_ext = the FIRST pixel
// in raster order attaining filt_min (ascending) / filt_max (descending),
// matching the CPU scan's strict-inequality update rule. arg_ext also stays
// resident on device for sample_ext(). dev_filtered (and the extent/arg
// outputs) may be null when the caller only wants area + bbox.
bool geometry(SliceStats* s, const void* dev_filtered, bool ascending, int* out_area,
              int* out_min_x, int* out_min_y, int* out_max_x, int* out_max_y,
              float* out_filt_min, float* out_filt_max, std::int64_t* out_arg_ext);

// Sample one channel at each region's seeding extremum (requires a prior
// geometry() call with a filtered raster). radius > 0 averages the clamped
// (2r+1)^2 window, accumulating in the same y-then-x order as the CPU's
// ChannelStats::sample_ext so the result is bit-identical. Empty regions
// receive 0.
bool sample_ext(SliceStats* s, const void* dev_channel, int radius, float* out_sample);

}  // namespace gpustats
}  // namespace msseg
