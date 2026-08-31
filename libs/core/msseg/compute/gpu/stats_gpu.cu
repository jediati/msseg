// GPU per-region statistics accumulation (see stats_gpu.hpp for the contract).
//
// Reductions are CSR-segmented (CUB DeviceSegmentedReduce) over a pixel order
// fixed once by a stable sort, so every result is run-to-run deterministic --
// no atomics. Device code here touches only flat arrays; the pipeline types
// (ChannelStats, Msc2DFeatureStat) never cross into this TU.

#include "msseg/compute/gpu/stats_gpu.hpp"

#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace msseg {
namespace gpustats {
namespace {

bool cuda_ok(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return true;
  std::fprintf(stderr, "msseg gpustats: %s failed: %s\n", what, cudaGetErrorString(err));
  return false;
}

// Version-proof reduction functors (the cub::Sum/Min/Max aliases were removed
// from CCCL 3.0, which ships with CUDA 13).
struct SumOp {
  template <typename T>
  __host__ __device__ T operator()(T a, T b) const {
    return a + b;
  }
};
struct MinOp {
  template <typename T>
  __host__ __device__ T operator()(T a, T b) const {
    return b < a ? b : a;
  }
};
struct MaxOp {
  template <typename T>
  __host__ __device__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

struct IsNonNeg {
  __host__ __device__ bool operator()(int v) const { return v >= 0; }
};

// Gather one channel value through the sorted pixel-index array.
struct GatherFloat {
  const float* ch;
  const unsigned int* pidx;
  __host__ __device__ float operator()(long long j) const { return ch[pidx[j]]; }
};
struct GatherDouble {
  const float* ch;
  const unsigned int* pidx;
  __host__ __device__ double operator()(long long j) const {
    return static_cast<double>(ch[pidx[j]]);
  }
};
struct GatherSquare {
  const float* ch;
  const unsigned int* pidx;
  __host__ __device__ double operator()(long long j) const {
    const double v = static_cast<double>(ch[pidx[j]]);
    return v * v;
  }
};
struct GatherX {
  const unsigned int* pidx;
  int w;
  __host__ __device__ int operator()(long long j) const {
    return static_cast<int>(pidx[j] % static_cast<unsigned int>(w));
  }
};

// Packed (orderable value bits << 32 | pixel index) key whose segmented MIN is
// the first raster-order pixel attaining the extremal value -- exactly the CPU
// scan's strict-inequality update. NaNs never win (unless a region is all-NaN).
struct GatherExtKey {
  const float* f;
  const unsigned int* pidx;
  bool ascending;
  __host__ __device__ unsigned long long operator()(long long j) const {
    const unsigned int p = pidx[j];
#ifdef __CUDA_ARCH__
    unsigned int b = __float_as_uint(f[p]);
#else
    unsigned int b;
    std::memcpy(&b, &f[p], sizeof(b));
#endif
    // Map float bits to an unsigned that sorts like the float.
    unsigned int ord = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    if (!ascending) ord = ~ord;                    // MIN key now picks the MAX value
    if (isnan(f[p])) ord = 0xFFFFFFFFu;            // never beats a real value
    return (static_cast<unsigned long long>(ord) << 32) | p;
  }
};

struct IsLabeled {
  __host__ __device__ bool operator()(const thrust::tuple<int, unsigned int>& t) const {
    return thrust::get<0>(t) >= 0;
  }
};

__global__ void bbox_y_kernel(const int* offsets, const unsigned int* pidx, int n_regions, int w,
                              int* min_y, int* max_y) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n_regions) return;
  const int b = offsets[r], e = offsets[r + 1];
  if (e <= b) {
    min_y[r] = 0;
    max_y[r] = -1;
    return;
  }
  // Pixels are sorted by raster index inside a segment, so y is monotonic.
  min_y[r] = static_cast<int>(pidx[b] / static_cast<unsigned int>(w));
  max_y[r] = static_cast<int>(pidx[e - 1] / static_cast<unsigned int>(w));
}

__global__ void decode_argext_kernel(const unsigned long long* keys, const int* offsets,
                                     int n_regions, long long* arg_ext) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n_regions) return;
  arg_ext[r] = (offsets[r + 1] > offsets[r]) ? static_cast<long long>(keys[r] & 0xFFFFFFFFull) : -1;
}

__global__ void sample_ext_kernel(const float* ch, const long long* arg_ext, int n_regions, int w,
                                  int h, int radius, float* out) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n_regions) return;
  const long long idx = arg_ext[r];
  if (idx < 0) {
    out[r] = 0.0f;
    return;
  }
  if (radius <= 0) {
    out[r] = ch[idx];
    return;
  }
  const int x = static_cast<int>(idx % w);
  const int y = static_cast<int>(idx / w);
  const int x0 = max(0, x - radius), x1 = min(w - 1, x + radius);
  const int y0 = max(0, y - radius), y1 = min(h - 1, y + radius);
  const double n = static_cast<double>(x1 - x0 + 1) * static_cast<double>(y1 - y0 + 1);
  // Same y-then-x sequential double accumulation as ChannelStats::sample_ext,
  // so radius > 0 samples are bit-identical to the CPU path.
  double acc = 0.0;
  for (int yy = y0; yy <= y1; ++yy) {
    const long long row = static_cast<long long>(yy) * w;
    for (int xx = x0; xx <= x1; ++xx) acc += ch[row + xx];
  }
  out[r] = static_cast<float>(acc / n);
}

}  // namespace

struct SliceStats {
  long long w = 0, h = 0, n = 0;
  int n_regions = 0;
  long long n_labeled = 0;

  unsigned int* d_pidx = nullptr;  // sorted by (label, raster index)
  int* d_offsets = nullptr;        // n_regions + 1
  float* d_raster[2] = {nullptr, nullptr};
  long long* d_arg_ext = nullptr;

  // Reusable per-region device outputs + CUB temp storage.
  double* d_out_d = nullptr;
  float* d_out_f = nullptr;
  int* d_out_i = nullptr;
  unsigned long long* d_out_k = nullptr;
  void* d_temp = nullptr;
  size_t temp_bytes = 0;

  bool ensure_temp(size_t bytes) {
    if (bytes <= temp_bytes) return true;
    if (d_temp) cudaFree(d_temp);
    d_temp = nullptr;
    temp_bytes = 0;
    if (!cuda_ok(cudaMalloc(&d_temp, bytes), "temp alloc")) return false;
    temp_bytes = bytes;
    return true;
  }
};

bool available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

SliceStats* create(std::int64_t width, std::int64_t height) {
  if (width <= 0 || height <= 0 || !available()) return nullptr;
  SliceStats* s = new SliceStats();
  s->w = width;
  s->h = height;
  s->n = width * height;
  return s;
}

void destroy(SliceStats* s) {
  if (!s) return;
  cudaFree(s->d_pidx);
  cudaFree(s->d_offsets);
  cudaFree(s->d_raster[0]);
  cudaFree(s->d_raster[1]);
  cudaFree(s->d_arg_ext);
  cudaFree(s->d_out_d);
  cudaFree(s->d_out_f);
  cudaFree(s->d_out_i);
  cudaFree(s->d_out_k);
  cudaFree(s->d_temp);
  delete s;
}

bool set_labels(SliceStats* s, const int* host_labels, const void* dev_labels, int n_regions) {
  if (!s || n_regions < 0 || (host_labels == nullptr) == (dev_labels == nullptr)) return false;
  s->n_regions = n_regions;

  int* d_labels = nullptr;
  if (!cuda_ok(cudaMalloc(&d_labels, s->n * sizeof(int)), "labels alloc")) return false;
  const cudaError_t cp =
      host_labels
          ? cudaMemcpy(d_labels, host_labels, s->n * sizeof(int), cudaMemcpyHostToDevice)
          : cudaMemcpy(d_labels, dev_labels, s->n * sizeof(int), cudaMemcpyDeviceToDevice);
  if (!cuda_ok(cp, "labels copy")) {
    cudaFree(d_labels);
    return false;
  }

  int* d_keys = nullptr;
  bool ok = false;
  try {
    thrust::device_ptr<const int> lab(d_labels);
    s->n_labeled = thrust::count_if(thrust::device, lab, lab + s->n, IsNonNeg{});

    cudaFree(s->d_pidx);
    s->d_pidx = nullptr;
    cudaFree(s->d_offsets);
    s->d_offsets = nullptr;
    if (!cuda_ok(cudaMalloc(&s->d_pidx, (s->n_labeled ? s->n_labeled : 1) * sizeof(unsigned int)),
                 "pidx alloc") ||
        !cuda_ok(cudaMalloc(&d_keys, (s->n_labeled ? s->n_labeled : 1) * sizeof(int)),
                 "keys alloc") ||
        !cuda_ok(cudaMalloc(&s->d_offsets, (n_regions + 1) * sizeof(int)), "offsets alloc")) {
      cudaFree(d_labels);
      cudaFree(d_keys);
      return false;
    }

    auto first = thrust::make_zip_iterator(
        thrust::make_tuple(lab, thrust::counting_iterator<unsigned int>(0)));
    auto out = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_ptr<int>(d_keys), thrust::device_ptr<unsigned int>(s->d_pidx)));
    thrust::copy_if(thrust::device, first, first + s->n, out, IsLabeled{});
    thrust::stable_sort_by_key(thrust::device, thrust::device_ptr<int>(d_keys),
                               thrust::device_ptr<int>(d_keys) + s->n_labeled,
                               thrust::device_ptr<unsigned int>(s->d_pidx));
    thrust::lower_bound(thrust::device, thrust::device_ptr<int>(d_keys),
                        thrust::device_ptr<int>(d_keys) + s->n_labeled,
                        thrust::counting_iterator<int>(0),
                        thrust::counting_iterator<int>(n_regions + 1),
                        thrust::device_ptr<int>(s->d_offsets));
    ok = cuda_ok(cudaGetLastError(), "csr build");
  } catch (...) {
    ok = false;
  }
  cudaFree(d_keys);
  cudaFree(d_labels);
  if (!ok) return false;

  // Per-region scratch, sized once the region count is known.
  cudaFree(s->d_out_d);
  cudaFree(s->d_out_f);
  cudaFree(s->d_out_i);
  cudaFree(s->d_out_k);
  cudaFree(s->d_arg_ext);
  s->d_out_d = nullptr;
  s->d_out_f = nullptr;
  s->d_out_i = nullptr;
  s->d_out_k = nullptr;
  s->d_arg_ext = nullptr;
  const int m = n_regions > 0 ? n_regions : 1;
  if (!cuda_ok(cudaMalloc(&s->d_out_d, m * sizeof(double)), "out_d alloc") ||
      !cuda_ok(cudaMalloc(&s->d_out_f, m * sizeof(float)), "out_f alloc") ||
      !cuda_ok(cudaMalloc(&s->d_out_i, m * sizeof(int)), "out_i alloc") ||
      !cuda_ok(cudaMalloc(&s->d_out_k, m * sizeof(unsigned long long)), "out_k alloc") ||
      !cuda_ok(cudaMalloc(&s->d_arg_ext, m * sizeof(long long)), "arg_ext alloc")) {
    return false;
  }
  return true;
}

const void* upload(SliceStats* s, const float* host, int slot) {
  if (!s || !host || slot < 0 || slot > 1) return nullptr;
  if (!s->d_raster[slot] &&
      !cuda_ok(cudaMalloc(&s->d_raster[slot], s->n * sizeof(float)), "raster alloc")) {
    return nullptr;
  }
  if (!cuda_ok(cudaMemcpy(s->d_raster[slot], host, s->n * sizeof(float), cudaMemcpyHostToDevice),
               "raster upload")) {
    return nullptr;
  }
  return s->d_raster[slot];
}

namespace {

template <typename OutT, typename GatherOp, typename ReduceOp>
bool segmented_reduce(SliceStats* s, GatherOp gather, ReduceOp op, OutT init, OutT* d_out,
                      OutT* host_out) {
  auto it = thrust::make_transform_iterator(thrust::counting_iterator<long long>(0), gather);
  size_t bytes = 0;
  cudaError_t err = cub::DeviceSegmentedReduce::Reduce(nullptr, bytes, it, d_out, s->n_regions,
                                                       s->d_offsets, s->d_offsets + 1, op, init);
  if (!cuda_ok(err, "segmented reduce size")) return false;
  if (!s->ensure_temp(bytes)) return false;
  err = cub::DeviceSegmentedReduce::Reduce(s->d_temp, s->temp_bytes, it, d_out, s->n_regions,
                                           s->d_offsets, s->d_offsets + 1, op, init);
  if (!cuda_ok(err, "segmented reduce")) return false;
  return cuda_ok(
      cudaMemcpy(host_out, d_out, s->n_regions * sizeof(OutT), cudaMemcpyDeviceToHost),
      "reduce download");
}

}  // namespace

bool reduce_channel(SliceStats* s, const void* dev_channel, double* out_sum, double* out_sumsq,
                    float* out_min, float* out_max) {
  if (!s || !dev_channel || s->n_regions <= 0) return false;
  const float* ch = static_cast<const float*>(dev_channel);
  if (out_sum &&
      !segmented_reduce<double>(s, GatherDouble{ch, s->d_pidx}, SumOp{}, 0.0, s->d_out_d,
                                out_sum)) {
    return false;
  }
  if (out_sumsq &&
      !segmented_reduce<double>(s, GatherSquare{ch, s->d_pidx}, SumOp{}, 0.0, s->d_out_d,
                                out_sumsq)) {
    return false;
  }
  if (out_min &&
      !segmented_reduce<float>(s, GatherFloat{ch, s->d_pidx}, MinOp{}, FLT_MAX, s->d_out_f,
                               out_min)) {
    return false;
  }
  if (out_max &&
      !segmented_reduce<float>(s, GatherFloat{ch, s->d_pidx}, MaxOp{}, -FLT_MAX, s->d_out_f,
                               out_max)) {
    return false;
  }
  return true;
}

bool geometry(SliceStats* s, const void* dev_filtered, bool ascending, int* out_area,
              int* out_min_x, int* out_min_y, int* out_max_x, int* out_max_y, float* out_filt_min,
              float* out_filt_max, std::int64_t* out_arg_ext) {
  if (!s || s->n_regions <= 0 || !out_area) return false;
  const int m = s->n_regions;

  // area from the CSR offsets.
  std::vector<int> offsets(static_cast<size_t>(m) + 1);
  if (!cuda_ok(cudaMemcpy(offsets.data(), s->d_offsets, (m + 1) * sizeof(int),
                          cudaMemcpyDeviceToHost),
               "offsets download")) {
    return false;
  }
  for (int r = 0; r < m; ++r) out_area[r] = offsets[r + 1] - offsets[r];

  const int block = 256;
  const int grid = (m + block - 1) / block;

  if (out_min_x &&
      !segmented_reduce<int>(s, GatherX{s->d_pidx, static_cast<int>(s->w)}, MinOp{}, INT_MAX,
                             s->d_out_i, out_min_x)) {
    return false;
  }
  if (out_max_x &&
      !segmented_reduce<int>(s, GatherX{s->d_pidx, static_cast<int>(s->w)}, MaxOp{}, -1,
                             s->d_out_i, out_max_x)) {
    return false;
  }
  if (out_min_y && out_max_y) {
    int* d_min_y = s->d_out_i;
    int* d_max_y = nullptr;
    if (!cuda_ok(cudaMalloc(&d_max_y, m * sizeof(int)), "max_y alloc")) return false;
    bbox_y_kernel<<<grid, block>>>(s->d_offsets, s->d_pidx, m, static_cast<int>(s->w), d_min_y,
                                   d_max_y);
    bool ok = cuda_ok(cudaGetLastError(), "bbox_y kernel") &&
              cuda_ok(cudaMemcpy(out_min_y, d_min_y, m * sizeof(int), cudaMemcpyDeviceToHost),
                      "min_y download") &&
              cuda_ok(cudaMemcpy(out_max_y, d_max_y, m * sizeof(int), cudaMemcpyDeviceToHost),
                      "max_y download");
    cudaFree(d_max_y);
    if (!ok) return false;
  }

  if (dev_filtered) {
    const float* f = static_cast<const float*>(dev_filtered);
    if (out_filt_min &&
        !segmented_reduce<float>(s, GatherFloat{f, s->d_pidx}, MinOp{}, FLT_MAX, s->d_out_f,
                                 out_filt_min)) {
      return false;
    }
    if (out_filt_max &&
        !segmented_reduce<float>(s, GatherFloat{f, s->d_pidx}, MaxOp{}, -FLT_MAX, s->d_out_f,
                                 out_filt_max)) {
      return false;
    }
    if (out_arg_ext) {
      auto it = thrust::make_transform_iterator(thrust::counting_iterator<long long>(0),
                                                GatherExtKey{f, s->d_pidx, ascending});
      size_t bytes = 0;
      cudaError_t err = cub::DeviceSegmentedReduce::Reduce(
          nullptr, bytes, it, s->d_out_k, m, s->d_offsets, s->d_offsets + 1, MinOp{},
          0xFFFFFFFFFFFFFFFFull);
      if (!cuda_ok(err, "arg_ext reduce size") || !s->ensure_temp(bytes)) return false;
      err = cub::DeviceSegmentedReduce::Reduce(s->d_temp, s->temp_bytes, it, s->d_out_k, m,
                                               s->d_offsets, s->d_offsets + 1, MinOp{},
                                               0xFFFFFFFFFFFFFFFFull);
      if (!cuda_ok(err, "arg_ext reduce")) return false;
      decode_argext_kernel<<<grid, block>>>(s->d_out_k, s->d_offsets, m, s->d_arg_ext);
      if (!cuda_ok(cudaGetLastError(), "arg_ext decode")) return false;
      static_assert(sizeof(std::int64_t) == sizeof(long long), "int64 layout");
      if (!cuda_ok(cudaMemcpy(out_arg_ext, s->d_arg_ext, m * sizeof(long long),
                              cudaMemcpyDeviceToHost),
                   "arg_ext download")) {
        return false;
      }
    }
  }
  return true;
}

bool sample_ext(SliceStats* s, const void* dev_channel, int radius, float* out_sample) {
  if (!s || !dev_channel || !out_sample || s->n_regions <= 0 || !s->d_arg_ext) return false;
  const int m = s->n_regions;
  const int block = 256;
  const int grid = (m + block - 1) / block;
  sample_ext_kernel<<<grid, block>>>(static_cast<const float*>(dev_channel), s->d_arg_ext, m,
                                     static_cast<int>(s->w), static_cast<int>(s->h), radius,
                                     s->d_out_f);
  if (!cuda_ok(cudaGetLastError(), "sample_ext kernel")) return false;
  return cuda_ok(cudaMemcpy(out_sample, s->d_out_f, m * sizeof(float), cudaMemcpyDeviceToHost),
                 "sample_ext download");
}

}  // namespace gpustats
}  // namespace msseg
