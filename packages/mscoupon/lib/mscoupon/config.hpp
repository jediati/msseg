#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "msseg/workflow/params.hpp"  // msseg::StatsSpec

namespace mscoupon {

struct InputConfig {
  std::filesystem::path folder;
  std::vector<std::string> extensions{".tif", ".tiff"};
  std::string match;
  bool use_regex = false;
  bool natural_sort = true;
  std::size_t start = 0;
  std::optional<std::size_t> count;
  std::size_t stride = 1;
  // Optional explicit, ordered file list. When non-empty it takes precedence
  // over folder scanning + match/start/stride (a GUI-exported subsequence is a
  // concrete list). Paths may be absolute or relative to `folder`.
  std::vector<std::string> files;
};

struct OutputConfig {
  std::filesystem::path folder;
  std::string mask_template = "{stem}_mask.tiff";
  std::string table_template = "{stem}_segments.csv";
  bool overwrite = true;
};

struct FilterConfig {
  std::string operation = "none";
  nlohmann::json params = nlohmann::json::object();
};

// One predicate in the feature-selection query chain. Predicates are ANDed:
// a feature is kept only if it satisfies every query. `field` names a per-feature
// statistic (area, mean_base, mean_filtered, min_base, max_base, std_base,
// bbox_w, bbox_h, ...); `op` is one of lt/le/gt/ge/eq/between. `value2` is only
// used by `between` (kept iff value <= stat <= value2).
struct FeatureQuery {
  std::string field;
  std::string op = "gt";
  double value = 0.0;
  double value2 = 0.0;
};

// One pixel-intensity trim rule. Applied per-pixel to the selected raster BEFORE
// per-slice connected components. `channel` is base|filtered, `mode` is keep|omit
// (keep drops pixels that FAIL the predicate; omit drops pixels that PASS), `op`
// is lt/le/gt/ge against `value`. Rules are applied in order (ANDed).
struct PixelFilter {
  std::string channel = "base";   // base | filtered
  std::string mode = "keep";      // keep | omit
  std::string op = "gt";          // lt | le | gt | ge
  double value = 0.0;
};

struct MscConfig {
  std::optional<float> persistence_absolute;
  std::optional<float> persistence_percent = 10.0f;
  std::string compute_algorithm = "serial";
  bool accurate_ascending = true;
  bool accurate_descending = true;
  std::string manifold = "ascending";
  // Discrete-gradient thread count and (in "partitioned" mode) MSC/hierarchy
  // partition count. 0 => leave the msc_2d_lib default (8).
  int requested_parallelism = 0;
  // Window radius for the base-channel sample taken at each feature's seeding
  // extremum (the `ext_base` selection field). 0 => the single critical pixel.
  int extremum_sample_radius = 0;
};

// Which per-feature statistics to compute, and therefore which fields exist to
// query. Wraps msseg::StatsSpec (the 2D/per-pixel half) with the 3D reductions,
// which only mscoupon's cross-slice matcher can produce.
//
// In 2D a reduction runs over a feature's PIXELS. In 3D the same reduction over
// the field still pools voxels -- a mean of per-slice means would be weighted by
// slice count rather than area -- but `per_slice` reductions run over the
// feature's PER-SLICE values, which is the only way `area` and `bbox_w/h` become
// meaningful under mean/min/max/std.
struct StatisticsConfig {
  msseg::StatsSpec spec;
  // 3D only: quantities whose per-slice distribution is reduced across the
  // slices a global feature spans. Names must be per-slice CC fields.
  std::vector<std::string> per_slice_quantities;
  bool ps_mean = true, ps_min = true, ps_max = true, ps_std = true;

  bool any_per_slice() const {
    return !per_slice_quantities.empty() && (ps_mean || ps_min || ps_max || ps_std);
  }
};

struct SegmentKeepConfig {
  std::optional<std::size_t> min_area;
  std::optional<std::size_t> max_area;
  std::optional<float> min_value;
  std::optional<float> max_value;
  std::optional<float> min_mean;
  std::optional<float> max_mean;
  std::vector<int> allow_ids;
  std::vector<int> deny_ids;
};

struct ExecutionConfig {
  int total_threads = 0;
  int threads_per_slice = 4;
  int concurrent_slices = 0;
  int read_threads = 1;
  int write_threads = 1;
  std::size_t max_slices_at_a_time = 4;
  std::size_t read_queue_capacity = 4;
  std::size_t write_queue_capacity = 4;
};

struct MatchingConfig {
  // Cross-slice matching: link per-slice connected components (of the selected +
  // trimmed raster) into 3D features by `assembly.connectivity` between consecutive
  // slices, first-seen representative. Emits a per-slice->global map + an aggregated
  // master table; optionally per-slice CC and global-id label TIFFs.
  bool enabled = true;
  std::string map_template = "feature_map.csv";
  std::string global_table_template = "global_segments.csv";
  bool write_cc_labels = true;         // per-slice in-plane CC id raster (int32)
  bool write_global_labels = true;     // per-slice GLOBAL id raster (int32)
  std::string cc_label_template = "{stem}_cc_i32.tiff";
  std::string global_label_template = "{stem}_global_i32.tiff";
};

// 3D assembly of per-slice connected components into 3D features. `connectivity` is
// the voxel neighborhood (6/18/26); it drives BOTH the in-plane 2D CC structuring
// element and the cross-slice stencil, and is shared by the CLI matcher and the
// GUI's scipy labeling so they produce the same grouping.
struct AssemblyConfig {
  int connectivity = 6;
};

struct TimingConfig {
  bool write_json = true;
  bool write_csv = false;
  std::filesystem::path output_path = "timing_report.json";
};

struct DebugOutputConfig {
  bool write_filter_tiff = false;
  bool write_label_tiff = false;
  std::string filter_template = "{stem}_filter.tiff";
  std::string label_template = "{stem}_labels_i32.tiff";
};

struct AppConfig {
  InputConfig input;
  OutputConfig output;
  // Ordered filter chain (applied output->input). Populated from a `filters`
  // array, or from a singular legacy `filter` object as a one-element chain.
  std::vector<FilterConfig> filters;
  FilterConfig filter;  // legacy single-filter mirror (== filters.front()).
  // Optional chain applied to the BASE channel -- the raster that per-feature
  // statistics and pixel filters are measured against -- before `filters` builds
  // the topology field from it. Its main use is a `normalize` stage, which puts
  // stats and pixel thresholds on a two-point [0,1] scale so one threshold
  // transfers across a stack. Empty (the default) leaves the base raw, which is
  // exactly the pre-existing behaviour.
  std::vector<FilterConfig> base_filters;
  MscConfig msc;
  // Which statistics to compute (and hence which feature_filters fields exist).
  // Omitted => base channel only, all reductions, extremum on: the pre-existing
  // field set minus the filtered aggregates, which had no reader.
  StatisticsConfig statistics;
  SegmentKeepConfig segments;
  // Per-slice selection query chain (ANDed) on 2D merged-region stats. Generalizes
  // `segments`; empty means no query filtering beyond the legacy `segments` gate.
  std::vector<FeatureQuery> feature_filters;
  // Pixel-intensity trim chain, applied to the selected raster before CC.
  std::vector<PixelFilter> pixel_filters;
  ExecutionConfig execution;
  MatchingConfig matching;
  AssemblyConfig assembly;
  TimingConfig timing;
  DebugOutputConfig debug_output;
  bool dry_run = false;
};

struct CliOptions {
  std::filesystem::path config_path;
  std::optional<std::filesystem::path> input_folder_override;
  std::optional<std::filesystem::path> output_folder_override;
  std::optional<std::string> match_override;
  std::optional<std::size_t> start_override;
  std::optional<std::size_t> count_override;
  std::optional<std::size_t> stride_override;
  std::optional<int> worker_override;
  std::optional<int> parallelism_override;
  bool dump_filter_tiff = false;
  bool dump_label_tiff = false;
  bool disable_matching = false;
  bool dry_run = false;
};

CliOptions parse_cli(int argc, char** argv);
AppConfig load_config(const CliOptions& cli);
void validate_config(const AppConfig& cfg);

// Parse a "statistics" block out of `root` (absent => the defaults). Exposed so
// the pybind layer parses it with exactly the same code as the CLI -- the field
// schema is a wire format between the GUI and a batch run, and a second parser
// here would drift into offering fields the CLI rejects.
void parse_statistics_json(const nlohmann::json& root, StatisticsConfig& stats);

}  // namespace mscoupon
