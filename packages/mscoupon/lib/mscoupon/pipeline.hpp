#pragma once

#include <chrono>
#include <vector>

#include "mscoupon/config.hpp"
#include "mscoupon/sequence.hpp"
#include "mscoupon/types.hpp"

namespace mscoupon {

std::vector<SliceOutput> run_pipeline(const AppConfig& cfg, const std::vector<SliceJob>& jobs,
                                      const std::chrono::steady_clock::time_point& process_start);
void write_timing_report(const AppConfig& cfg, const std::vector<SliceOutput>& outputs);

}  // namespace mscoupon
