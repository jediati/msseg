#include "mscoupon/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <nlohmann/json.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mscoupon/cc_stage.hpp"
#include "mscoupon/filter.hpp"
#include "mscoupon/io.hpp"
#include "mscoupon/matcher.hpp"
#include "mscoupon/msc_stage.hpp"
#include "mscoupon/query.hpp"
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
  std::optional<std::vector<int>> labels;   // debug MSC-merged label tiff
  int label_width = 0;
  int label_height = 0;
  // Per-slice connected components of the selected+trimmed raster: the nodes the
  // matcher links into 3D features, and the source for the CC / global label TIFFs.
  std::vector<int> cc_labels;               // -1 bg, 0..n-1
  std::vector<CcNodeStat> node_stats;       // per component
  msseg::ChannelStats node_channels;        // per component, per measurement channel
  std::vector<msseg::ResolvedStatChannel> channels;   // slot schema for node_channels
  int width = 0;
  int height = 0;
  StageTiming timing;
};

double elapsed_ms(const Clock::time_point& start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace

std::vector<SliceOutput> run_pipeline(const AppConfig& cfg, const std::vector<SliceJob>& jobs,
                                      const std::chrono::steady_clock::time_point& process_start) {
  std::filesystem::create_directories(cfg.output.folder);

  // Startup summary: the workflow parameters, so the log records how the run was
  // configured relative to the data.
  {
    std::ostringstream oss;
    oss << "[mscoupon] run: " << jobs.size() << " slices, filters=[";
    for (std::size_t i = 0; i < cfg.filters.size(); ++i) {
      oss << (i ? "," : "") << cfg.filters[i].operation;
    }
    oss << "]";
    if (!cfg.base_filters.empty()) {
      oss << " base_filters=[";
      for (std::size_t i = 0; i < cfg.base_filters.size(); ++i) {
        oss << (i ? "," : "") << cfg.base_filters[i].operation;
      }
      oss << "]";
    }
    oss << " manifold=" << cfg.msc.manifold << " persistence=";
    if (cfg.msc.persistence_absolute.has_value()) oss << *cfg.msc.persistence_absolute << "abs";
    else if (cfg.msc.persistence_percent.has_value()) oss << *cfg.msc.persistence_percent << "%";
    oss << " feature_filters=" << cfg.feature_filters.size()
        << " assembly.connectivity=" << cfg.assembly.connectivity
        << " matching=" << (cfg.matching.enabled ? "on" : "off") << "\n";
    std::cerr << oss.str();
  }

  const int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
  const int total_threads = cfg.execution.total_threads > 0 ? cfg.execution.total_threads : std::max(2, hw_threads);
  const int lanes = cfg.execution.concurrent_slices > 0
                        ? cfg.execution.concurrent_slices
                        : std::max(1, (total_threads - cfg.execution.read_threads - cfg.execution.write_threads) /
                                         std::max(1, cfg.execution.threads_per_slice));

  BlockingQueue<SliceJob> read_queue(std::max<std::size_t>(cfg.execution.read_queue_capacity, cfg.execution.max_slices_at_a_time));
  BlockingQueue<LoadedSlice> compute_queue(std::max<std::size_t>(1, cfg.execution.max_slices_at_a_time));
  BlockingQueue<ProcessedSlice> write_queue(std::max<std::size_t>(cfg.execution.write_queue_capacity, cfg.execution.max_slices_at_a_time));

  // With matching enabled, compute lanes feed a single in-order matcher thread
  // (which forwards each slice unchanged to the writers); otherwise they feed the
  // writers directly, exactly as before.
  BlockingQueue<ProcessedSlice> match_queue(std::max<std::size_t>(cfg.execution.write_queue_capacity, cfg.execution.max_slices_at_a_time));
  BlockingQueue<ProcessedSlice>& compute_out = cfg.matching.enabled ? match_queue : write_queue;

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
          // Two independent channels, both derived from the raw slice:
          //   base     = base_filters(original)  -- what statistics and pixel
          //              filters are measured against
          //   filtered = filters(original)       -- the topology field the MSC runs on
          // Deriving both from `original` (rather than chaining filters onto base)
          // keeps `filters` meaning exactly what it always has, so a config with
          // no base_filters is bit-for-bit identical to the previous behaviour.
          // A `normalize` stage in either chain puts that channel on a two-point
          // [0,1] scale; because the map is affine and order-preserving it cannot
          // change the MSC, only the units the thresholds are read in.
          std::vector<TwoPoint> base_normalizers;
          Image2D base = cfg.base_filters.empty()
                             ? loaded.original
                             : apply_filter_chain(loaded.original, cfg.base_filters,
                                                  &base_normalizers);
          Image2D filtered = apply_filter_chain(loaded.original, cfg.filters);

          // The slice's measurement channels, built ONCE and shared by the MSC
          // stage and the connected-component stage below. The derived
          // scale-space responses collapse into a single diffg filter-bank
          // traversal that shares its separable passes, which is what makes a
          // multi-sigma stack affordable per slice.
          const auto to_diffg = [](const Image2D& img) {
            diffg::Image<float> out(diffg::Dimensions{
                static_cast<std::size_t>(img.width), static_cast<std::size_t>(img.height), 1});
            std::copy(img.pixels.begin(), img.pixels.end(), out.data());
            return out;
          };
          diffg::ExecutionOptions bank_exec{};
          bank_exec.threads = std::max(1, cfg.execution.threads_per_slice);
          const diffg::Image<float> base_img = to_diffg(base);
          const diffg::Image<float> filtered_img = to_diffg(filtered);
          const msseg::StatChannelBank bank = msseg::build_stat_channels(
              base_img, filtered_img, cfg.statistics.spec, bank_exec);
          loaded.timing.filter_ms = elapsed_ms(filter_start);

          // Merge-tree authoritative segmentation (same engine the GUI drives),
          // so an exported config reproduces the viewer's per-slice output.
          const auto msc_start = Clock::now();
          SliceSegmentation seg =
              segment_slice_pipeline(base_img, filtered_img, cfg.msc, cfg.statistics.spec, bank);
          std::vector<int>& labels = seg.labels;
          loaded.timing.msc_ms = elapsed_ms(msc_start);

          const auto stats_start = Clock::now();
          // The columnar per-feature statistics for this slice: one schema, one
          // flat value block. Both the selection chain and the ext_* carry-over
          // below read it, so the row is built once per feature per slice.
          const FeatureTable feature_rows =
              feature_table(seg.features, seg.feature_channels, seg.channels, cfg.statistics.spec);
          const int ext_base_col = feature_rows.column("ext_base");
          SegmentTable table = compute_segment_table(base, labels, loaded.job.slice_index);
          // The CSV rows are accumulated over the base channel only; carry the
          // seeding critical point across from the MSC feature statistics so the
          // table reports the same ext_* the selection queries see.
          {
            std::unordered_map<int, std::size_t> row_of_id;
            row_of_id.reserve(seg.features.size() * 2);
            for (std::size_t i = 0; i < seg.features.size(); ++i) {
              row_of_id.emplace(static_cast<int>(seg.features[i].feature_id), i);
            }
            for (auto& row : table.rows) {
              const auto it = row_of_id.find(row.segment_id);
              if (it == row_of_id.end()) continue;   // background row (-1) has no extremum
              const msseg::Msc2DFeatureStat& f = seg.features[it->second];
              row.ext_x = f.ext_x;
              row.ext_y = f.ext_y;
              row.ext_filtered = f.ext_filtered;
              // ext_base only exists when `base` is a measurement channel.
              row.ext_base = ext_base_col >= 0
                                 ? static_cast<float>(feature_rows.at(
                                       it->second, static_cast<std::size_t>(ext_base_col)))
                                 : 0.0f;
            }
          }
          loaded.timing.stats_ms = elapsed_ms(stats_start);

          const auto select_start = Clock::now();
          // Per-slice selection: legacy size gate intersected with the query chain
          // (evaluated on the 2D merged-region base+filtered stats -- single-source
          // evaluator shared with the GUI).
          std::unordered_set<int> keep_ids = select_segment_ids(table, cfg.segments);
          if (!cfg.feature_filters.empty()) {
            // Field names resolve to columns once for the whole slice, not once
            // per feature.
            const CompiledQueries compiled = compile_queries(feature_rows, cfg.feature_filters);
            std::unordered_set<int> pass;
            for (std::size_t i = 0; i < seg.features.size(); ++i) {
              if (row_passes(feature_rows, i, compiled)) {
                pass.insert(static_cast<int>(seg.features[i].feature_id));
              }
            }
            std::unordered_set<int> both;
            for (const int id : keep_ids) {
              if (pass.count(id)) both.insert(id);
            }
            keep_ids.swap(both);
          }
          // Pixel trim + per-slice connected components over the selected raster.
          std::vector<int> cc_labels;
          std::vector<CcNodeStat> node_stats;
          msseg::ChannelStats node_channels;
          const int n_cc = label_selected_components(
              labels, base.width, base.height, base,
              filtered, keep_ids, cfg.pixel_filters, cfg.assembly.connectivity,
              cfg.msc.manifold != "descending", cfg.statistics.spec, bank,
              seg.base_relevance_floor, seg.base_relevance_ceiling,
              cc_labels, node_stats, node_channels);
          // Per-slice mask = the selected + trimmed CC foreground.
          Mask2D mask;
          mask.width = base.width;
          mask.height = base.height;
          mask.pixels.resize(cc_labels.size());
          for (std::size_t i = 0; i < cc_labels.size(); ++i) {
            mask.pixels[i] = cc_labels[i] >= 0 ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
          }
          loaded.timing.select_ms = elapsed_ms(select_start);

          // Per-slice stage log (MSCEER's own stdout above reports the MSC
          // structure; this adds the pipeline-level summary). Built as one line
          // so parallel compute lanes don't interleave mid-line.
          {
            float imin = std::numeric_limits<float>::max(), imax = std::numeric_limits<float>::lowest();
            for (float v : base.pixels) { imin = std::min(imin, v); imax = std::max(imax, v); }
            float fmin = std::numeric_limits<float>::max(), fmax = std::numeric_limits<float>::lowest();
            for (float v : filtered.pixels) { fmin = std::min(fmin, v); fmax = std::max(fmax, v); }
            std::ostringstream oss;
            oss << "[mscoupon] slice " << loaded.job.slice_index << " ("
                << loaded.job.input_path.filename().string() << "): image[" << imin << "," << imax
                << "] filtered[" << fmin << "," << fmax << "]";
            // The measured landmarks, so a run is reproducible from its log.
            for (const auto& tp : base_normalizers) {
              oss << " norm[" << tp.low << "," << tp.high << "]";
            }
            oss << " regions=" << seg.features.size()
                << " kept=" << keep_ids.size() << " cc=" << n_cc << "\n";
            std::cerr << oss.str();
          }

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
          // Per-slice CC nodes: the matcher's input + the CC/global label rasters.
          processed.width = loaded.original.width;
          processed.height = loaded.original.height;
          processed.cc_labels = std::move(cc_labels);
          processed.node_stats = std::move(node_stats);
          processed.node_channels = std::move(node_channels);
          processed.channels = seg.channels;
          processed.timing = loaded.timing;
          compute_out.push(std::move(processed));
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
          if (cfg.matching.write_cc_labels && !processed.cc_labels.empty()) {
            write_tiff_int32(processed.job.cc_label_output_path, processed.width,
                             processed.height, processed.cc_labels);
          }
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

  // The matcher runs only when enabled: it consumes compute output in arbitrary
  // order, reorders by slice_index, links each slice to the prior one, forwards it
  // unchanged to the writers, and finally emits the derived cross-slice CSVs.
  std::thread matcher_thread;
  if (cfg.matching.enabled) {
    matcher_thread = std::thread([&]() {
      try {
        SliceMatcher matcher;
        matcher.configure(cfg.statistics, cfg.msc.manifold != "descending",
                          msseg::resolve_stat_channels(cfg.statistics.spec));
        std::map<int, ProcessedSlice> pending;
        std::map<int, std::filesystem::path> global_path_of;   // slice -> global TIFF
        int expected = 0;

        auto consume = [&](ProcessedSlice&& ps) {
          matcher.add_slice(ps.cc_labels, ps.width, ps.height, ps.node_stats, ps.node_channels,
                            ps.job.slice_index, cfg.assembly.connectivity);
          global_path_of[ps.job.slice_index] = ps.job.global_label_output_path;
          write_queue.push(std::move(ps));
        };

        while (true) {
          auto maybe = match_queue.pop();
          if (!maybe.has_value()) break;
          const int idx = maybe->job.slice_index;
          pending.emplace(idx, std::move(*maybe));
          while (!pending.empty() && pending.begin()->first == expected) {
            auto it = pending.begin();
            consume(std::move(it->second));
            pending.erase(it);
            ++expected;
          }
        }
        // Flush any stragglers in ascending order. Under normal operation pending
        // is already empty; a gap only remains if an upstream error dropped a
        // slice, in which case the error is rethrown after the joins below.
        for (auto& kv : pending) consume(std::move(kv.second));

        std::vector<FeatureMapRow> map_rows;
        GlobalFeatureTable global_table;
        // Relabel pass: per-slice GLOBAL id rasters (needs the resolved ids, so it
        // runs after finalize -- a late slice can retroactively merge earlier ids).
        // Each raster is written and dropped as it is produced; collecting them
        // first would hold the whole stack in RAM (~100 GB at 2500 slices).
        matcher.finalize(map_rows, global_table, [&](GlobalLabelRaster&& r) {
          if (!cfg.matching.write_global_labels) return;
          const auto it = global_path_of.find(r.slice_index);
          if (it != global_path_of.end()) {
            write_tiff_int32(it->second, r.width, r.height, r.data);
          }
        });
        write_feature_map_csv(cfg.output.folder / cfg.matching.map_template, map_rows);
        write_global_table_csv(cfg.output.folder / cfg.matching.global_table_template,
                               global_table, cfg.statistics);
      } catch (...) {
        capture_error(std::current_exception());
      }
      write_queue.close();
    });
  }

  producer.join();
  for (auto& t : readers) t.join();
  compute_queue.close();
  for (auto& t : compute_lanes) t.join();
  if (cfg.matching.enabled) {
    match_queue.close();
    matcher_thread.join();  // finalizes derived CSVs and closes write_queue
  } else {
    write_queue.close();
  }
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
