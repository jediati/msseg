// CUDA-off stub for the GPU statistics module (MSSEG_GPU=OFF builds, pip
// wheels, CUDA-less machines). Every entry point fails cleanly so the caller
// takes its CPU path without #ifdefs.

#include "msseg/compute/gpu/stats_gpu.hpp"

#include <cstddef>

namespace msseg {
namespace gpustats {

bool available() { return false; }

SliceStats* create(std::int64_t, std::int64_t) { return nullptr; }
void destroy(SliceStats*) {}

bool set_labels(SliceStats*, const int*, const void*, int) { return false; }
const void* upload(SliceStats*, const float*, int) { return nullptr; }
bool reduce_channel(SliceStats*, const void*, double*, double*, float*, float*) { return false; }
bool geometry(SliceStats*, const void*, bool, int*, int*, int*, int*, int*, float*, float*,
              std::int64_t*) {
  return false;
}
bool sample_ext(SliceStats*, const void*, int, float*) { return false; }

}  // namespace gpustats
}  // namespace msseg
