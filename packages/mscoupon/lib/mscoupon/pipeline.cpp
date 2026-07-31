#include "mscoupon/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <thread>

#include <nlohmann/json.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mscoupon/filter.hpp"
#include "mscoupon/io.hpp"
#include "mscoupon/msc_stage.hpp"
#include "mscoupon/stats.hpp"

namespace mscoupon {
namespace {

using Clock = std::chrono::steady_clock;

template <typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue(std::size_t capacity) : capacity_(capacity) {}

  void push(T item) {
    std::unique_lock lock(mu_);
    cv_not_full_.wait(lock, [&]() { return closed_ || queue_.size() < capacity_; });
    if (closed_) throw std::runtime_error("Queue is closed");
    queue_.push(std::move(item));
    cv_not_empty_.notify_one();
  }

  std::optional<T> pop() {
    std::unique_lock lock(mu_);
    cv_not_empty_.wait(lock, [&]() { return closed_ || !queue_.empty(); });
    if (queue_.empty()) return std::nullopt;
    T item = std::move(queue_.front());
    queue_.pop();
    cv_not_full_.notify_one();
    return item;
  }

  void close() {
    std::lock_guard lock(mu_);
    closed_ = true;
    cv_not_empty_.notify_all();
    cv_not_full_.notify_all();
  }

 private:
  std::size_t capacity_;
  std::queue<T> queue_;
  bool closed_ = false;
  std::mutex mu_;
  std::condition_variable cv_not_empty_;
  std::condition_variable cv_not_full_;
};

struct LoadedSlice {
  SliceJob job;
  Image2D original;
  StageTiming timing;
  Clock::time_point total_start;
};

struct ProcessedSlice {
  SliceJob job;
  Mask2D mask;
  std::vector<SegmentStat> table;
  std::optional<Image2D> filtered_image;
  std::optional<std::vector<int>> labels;
  int label_width = 0;
  int label_height = 0;
  StageTiming timing;
};

double elapsed_ms(const Clock::time_point& start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace

std::vector<SliceOutput> run_pipeline(const AppConfig& cfg, const std::vector<SliceJob>& jobs,
                                      const std::chrono::steady_clock::time_point& process_start) {
  std::filesystem::create_directories(cfg.output.folder);

  const int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
  const int total_threads = cfg.execution.total_threads > 0 ? cfg.execution.total_threads : std::max(2, hw_threads);
  const int lanes = cfg.execution.concurrent_slices > 0
                        ? cfg.execution.concurrent_slices
                        : std::max(1, (total_threads - cfg.execution.read_threads - cfg.execution.write_threads) /
                                         std::max(1, cfg.execution.threads_per_slice));

  BlockingQueue<SliceJob> read_queue(std::max<std::size_t>(cfg.execution.read_queue_capacity, cfg.execution.max_slices_at_a_time));
  BlockingQueue<LoadedSlice> compute_queue(std::max<std::size_t>(1, cfg.execution.max_slices_at_a_time));
  BlockingQueue<ProcessedSlice> write_queue(std::max<std::size_t>(cfg.execution.write_queue_capacity, cfg.execution.max_slices_at_a_time));

  std::mutex inflight_mu;
  std::condition_variable inflight_cv;
  std::size_t inflight = 0;

  auto acquire_inflight_slot = [&]() {
    std::unique_lock lock(inflight_mu);
    inflight_cv.wait(lock, [&]() { return inflight < cfg.execution.max_slices_at_a_time; });
    ++inflight;
  };
  auto release_inflight_slot = [&]() {
    std::lock_guard lock(inflight_mu);
    if (inflight > 0) --inflight;
    inflight_cv.notify_one();
  };

  std::vector<SliceOutput> outputs;
  std::mutex outputs_mu;
  std::exception_ptr first_error;
  std::mutex error_mu;

  auto capture_error = [&](std::exception_ptr eptr) {
    std::lock_guard lock(error_mu);
    if (!first_error) first_error = eptr;
  };

  std::thread producer([&]() {
    try {
      for (const auto& job : jobs) {
        acquire_inflight_slot();
        read_queue.push(job);
      }
      read_queue.close();
    } catch (...) {
      capture_error(std::current_exception());
      read_queue.close();
    }
  });

  std::vector<std::thread> readers;
  readers.reserve(cfg.execution.read_threads);
  for (int i = 0; i < cfg.execution.read_threads; ++i) {
    readers.emplace_back([&]() {
      try {
        while (true) {
          auto maybe = read_queue.pop();
          if (!maybe.has_value()) break;
          LoadedSlice loaded;
          loaded.job = std::move(*maybe);
          loaded.total_start = Clock::now();

          const auto read_start = Clock::now();
          loaded.original = read_tiff_float32(loaded.job.input_path);
          loaded.timing.read_ms = elapsed_ms(read_start);

          compute_queue.push(std::move(loaded));
        }
      } catch (...) {
        capture_error(std::current_exception());
      }
    });
  }

  std::vector<std::thread> compute_lanes;
  compute_lanes.reserve(lanes);
  for (int lane = 0; lane < lanes; ++lane) {
    compute_lanes.emplace_back([&]() {
      try {
        while (true) {
          auto maybe = compute_queue.pop();
          if (!maybe.has_value()) break;
          LoadedSlice loaded = std::move(*maybe);

#ifdef _OPENMP
          omp_set_num_threads(cfg.execution.threads_per_slice);
#endif

          const auto filter_start = Clock::now();
          Image2D filtered = apply_filter(loaded.original, cfg.filter);
          loaded.timing.filter_ms = elapsed_ms(filter_start);

          const auto msc_start = Clock::now();
          std::vector<int> labels = compute_msc_labels(filtered, cfg.msc);
          loaded.timing.msc_ms = elapsed_ms(msc_start);

          const auto stats_start = Clock::now();
          SegmentTable table = compute_segment_table(loaded.original, labels, loaded.job.slice_index);
          loaded.timing.stats_ms = elapsed_ms(stats_start);

          const auto select_start = Clock::now();
          const auto keep_ids = select_segment_ids(table, cfg.segments);
          Mask2D mask = build_mask(labels, loaded.original.width, loaded.original.height, keep_ids);
          loaded.timing.select_ms = elapsed_ms(select_start);

          loaded.timing.total_ms = elapsed_ms(loaded.total_start);

          ProcessedSlice processed;
          processed.job = loaded.job;
          processed.mask = std::move(mask);
          processed.table = std::move(table.rows);
          if (cfg.debug_output.write_filter_tiff) {
            processed.filtered_image = filtered;
          }
          if (cfg.debug_output.write_label_tiff) {
            processed.labels = labels;
            processed.label_width = loaded.original.width;
            processed.label_height = loaded.original.height;
          }
          processed.timing = loaded.timing;
          write_queue.push(std::move(processed));
        }
      } catch (...) {
        capture_error(std::current_exception());
      }
    });
  }

  std::vector<std::thread> writers;
  writers.reserve(cfg.execution.write_threads);
  for (int i = 0; i < cfg.execution.write_threads; ++i) {
    writers.emplace_back([&]() {
      try {
        while (true) {
          auto maybe = write_queue.pop();
          if (!maybe.has_value()) break;
          ProcessedSlice processed = std::move(*maybe);

          const auto write_start = Clock::now();
          write_tiff_mask_u8(processed.job.mask_output_path, processed.mask);
          write_segment_table_csv(processed.job.table_output_path, processed.table);
          if (cfg.debug_output.write_filter_tiff && processed.filtered_image.has_value()) {
            write_tiff_float32(processed.job.filter_output_path, *processed.filtered_image);
          }
          if (cfg.debug_output.write_label_tiff && processed.labels.has_value()) {
            write_tiff_int32(processed.job.label_output_path, processed.label_width, processed.label_height, *processed.labels);
          }
          processed.timing.write_ms = elapsed_ms(write_start);

          SliceOutput out;
          out.input_path = processed.job.input_path;
          out.mask_output_path = processed.job.mask_output_path;
          out.table_output_path = processed.job.table_output_path;
          out.slice_index = processed.job.slice_index;
          out.timing = processed.timing;
          out.timing.out_time_ms = std::chrono::duration<double, std::milli>(Clock::now() - process_start).count();

          {
            std::lock_guard lock(outputs_mu);
            outputs.push_back(std::move(out));
          }
          release_inflight_slot();
        }
      } catch (...) {
        capture_error(std::current_exception());
      }
    });
  }

  producer.join();
  for (auto& t : readers) t.join();
  compute_queue.close();
  for (auto& t : compute_lanes) t.join();
  write_queue.close();
  for (auto& t : writers) t.join();

  if (first_error) std::rethrow_exception(first_error);

  std::sort(outputs.begin(), outputs.end(), [](const SliceOutput& a, const SliceOutput& b) { return a.slice_index < b.slice_index; });
  return outputs;
}

void write_timing_report(const AppConfig& cfg, const std::vector<SliceOutput>& outputs) {
  if (!cfg.timing.write_json && !cfg.timing.write_csv) return;

  std::filesystem::create_directories(cfg.timing.output_path.parent_path().empty() ? "." : cfg.timing.output_path.parent_path());

  if (cfg.timing.write_json) {
    nlohmann::json root;
    root["count"] = outputs.size();

    nlohmann::json items = nlohmann::json::array();
    for (const auto& out : outputs) {
      items.push_back({
          {"slice_index", out.slice_index},
          {"input", out.input_path.string()},
          {"mask_output", out.mask_output_path.string()},
          {"table_output", out.table_output_path.string()},
          {"read_ms", out.timing.read_ms},
          {"filter_ms", out.timing.filter_ms},
          {"msc_ms", out.timing.msc_ms},
          {"stats_ms", out.timing.stats_ms},
          {"select_ms", out.timing.select_ms},
          {"write_ms", out.timing.write_ms},
          {"total_ms", out.timing.total_ms},
          {"out_time_ms", out.timing.out_time_ms},
      });
    }
    root["items"] = std::move(items);

    std::ofstream json_out(cfg.timing.output_path);
    json_out << root.dump(2) << "\n";
  }

  if (cfg.timing.write_csv) {
    auto csv_path = cfg.timing.output_path;
    csv_path.replace_extension(".csv");
    std::ofstream out(csv_path);
    out << "slice_index,input,read_ms,filter_ms,msc_ms,stats_ms,select_ms,write_ms,total_ms,out_time_ms\n";
    for (const auto& item : outputs) {
      out << item.slice_index << "," << item.input_path.string() << "," << item.timing.read_ms << "," << item.timing.filter_ms << ","
          << item.timing.msc_ms << "," << item.timing.stats_ms << "," << item.timing.select_ms << "," << item.timing.write_ms << ","
          << item.timing.total_ms << "," << item.timing.out_time_ms << "\n";
    }
  }
}

}  // namespace mscoupon
