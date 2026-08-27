#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "diffg/image.hpp"
#include "mscoupon/cc_stage.hpp"
#include "mscoupon/config.hpp"
#include "mscoupon/filter.hpp"
#include "mscoupon/gmm.hpp"
#include "mscoupon/histogram_peaks.hpp"
#include "mscoupon/io.hpp"
#include "mscoupon/matcher.hpp"
#include "mscoupon/measure_config.hpp"
#include "mscoupon/normalize.hpp"
#include "mscoupon/query.hpp"
#include "mscoupon/region_measure.hpp"
#include "mscoupon/sequence.hpp"
#include "mscoupon/stats.hpp"
#include "msseg/compute/msc2d.hpp"

namespace {

void expect(bool cond, const char* message) {
  if (!cond) throw std::runtime_error(message);
}

// A field with several Gaussian wells (low = minima) on a raised plane, so the
// ascending 2-manifold decomposition has multiple minima that merge as the
// persistence threshold rises.
diffg::Image<float> make_wells(int w, int h) {
  diffg::Image<float> img(diffg::Dimensions{static_cast<std::size_t>(w), static_cast<std::size_t>(h), 1});
  struct Well { float cx, cy, depth, sigma; };
  const Well wells[] = {{0.28f, 0.30f, 1.0f, 0.14f},
                        {0.72f, 0.32f, 0.7f, 0.12f},
                        {0.50f, 0.72f, 0.4f, 0.10f}};
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float fx = static_cast<float>(x) / (w - 1);
      const float fy = static_cast<float>(y) / (h - 1);
      float v = 0.05f * (fx + fy);  // gentle tilt so wells have distinct depths
      for (const auto& wl : wells) {
        const float dx = fx - wl.cx, dy = fy - wl.cy;
        v -= wl.depth * std::exp(-(dx * dx + dy * dy) / (2.0f * wl.sigma * wl.sigma));
      }
      img.data()[static_cast<std::size_t>(y) * w + x] = v;
    }
  }
  return img;
}

// True iff the two labelings induce the same partition over pixels where both
// are >= 0 (label VALUES may differ; only the grouping must match).
bool same_partition(const std::vector<int>& a, const std::vector<int>& b) {
  if (a.size() != b.size()) return false;
  std::unordered_map<int, int> a2b, b2a;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] < 0 || b[i] < 0) continue;
    auto ab = a2b.emplace(a[i], b[i]);
    if (!ab.second && ab.first->second != b[i]) return false;
    auto ba = b2a.emplace(b[i], a[i]);
    if (!ba.second && ba.first->second != a[i]) return false;
  }
  return true;
}

int distinct_features(const std::vector<int>& labels) {
  std::unordered_set<int> ids;
  for (int v : labels)
    if (v >= 0) ids.insert(v);
  return static_cast<int>(ids.size());
}

// Image2D -> diffg::Image, for tests that build a measurement channel bank.
diffg::Image<float> to_diffg2d(const mscoupon::Image2D& img) {
  diffg::Image<float> out(diffg::Dimensions{static_cast<std::size_t>(img.width),
                                            static_cast<std::size_t>(img.height), 1});
  std::copy(img.pixels.begin(), img.pixels.end(), out.data());
  return out;
}

// The measurement spec the matcher tests use: the base channel only, so the
// channel plane has exactly one slot and "slot 0" means base everywhere below.
const msseg::StatsSpec& base_spec() {
  static const msseg::StatsSpec spec{};
  return spec;
}

// A slice's CC payload: the geometry stats plus the per-channel plane the
// matcher now takes alongside them. Both are indexed by component id.
struct CcSlice {
  std::vector<mscoupon::CcNodeStat> stats;
  msseg::ChannelStats channels;
};

struct NodeSpec {
  std::size_t area;
  float base = 1.0f;
  float filt = 1.0f;
};

// Build a slice of CC nodes, each with a constant base/filtered value.
CcSlice cc_slice(std::initializer_list<NodeSpec> nodes) {
  CcSlice out;
  out.channels.reset(nodes.size(), 1, base_spec());
  std::size_t i = 0;
  for (const NodeSpec& n : nodes) {
    mscoupon::CcNodeStat s;
    s.area = n.area;
    s.filt_min = n.filt; s.filt_max = n.filt;
    s.min_x = 0; s.min_y = 0; s.max_x = 0; s.max_y = 0;
    out.stats.push_back(s);
    msseg::ChannelAccum& a = out.channels.cell(i, 0);
    a.sum = static_cast<double>(n.base) * static_cast<double>(n.area);
    a.sumsq = static_cast<double>(n.base) * n.base * static_cast<double>(n.area);
    a.min = n.base; a.max = n.base;
    ++i;
  }
  return out;
}

// Configure a matcher with the base-only channel schema the helpers above build.
void configure_base(mscoupon::SliceMatcher& m, const mscoupon::StatisticsConfig& cfg,
                    bool ascending = true) {
  m.configure(cfg, ascending, msseg::resolve_stat_channels(cfg.spec));
}

// One named field of one row of the 3D master table, via the same columnar
// projection the CSV writer uses.
double global_field(const mscoupon::GlobalFeatureTable& t, std::size_t row,
                    const std::string& name) {
  const auto projected =
      mscoupon::global_feature_table(t.rows, t.channels, t.schema, base_spec());
  const int c = projected.column(name);
  expect(c >= 0, "the 3D table advertises the requested field");
  return c < 0 ? 0.0 : projected.at(row, static_cast<std::size_t>(c));
}

// global id for a per-slice CC id (segment_id in the map == CC component id).
int global_for(const std::vector<mscoupon::FeatureMapRow>& map, int slice_index, int cc_id) {
  for (const auto& r : map) {
    if (r.slice_index == slice_index && r.segment_id == cc_id) return r.global_id;
  }
  return -1;
}

void test_stats_bbox() {
  mscoupon::Image2D image;
  image.width = 3;
  image.height = 2;
  image.pixels = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};

  const std::vector<int> labels = {1, 1, 2, 1, 2, 2};
  const auto table = mscoupon::compute_segment_table(image, labels, 7);
  expect(table.rows.size() == 2, "Expected two segments");
  expect(table.rows[0].slice_index == 7, "Expected slice index set");
  expect(table.rows[0].min_x <= table.rows[0].max_x, "Expected bbox x ordering");
}

void test_sequence_stride() {
  const auto root = std::filesystem::temp_directory_path() / "mscoupon_sequence_test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  std::ofstream(root / "img_001.tiff").put('\n');
  std::ofstream(root / "img_002.tiff").put('\n');
  std::ofstream(root / "img_003.tiff").put('\n');

  mscoupon::AppConfig cfg;
  cfg.input.folder = root;
  cfg.input.stride = 2;
  cfg.output.folder = root / "out";
  const auto jobs = mscoupon::build_sequence(cfg);
  expect(jobs.size() == 2, "Expected stride to reduce selected jobs");

  std::filesystem::remove_all(root);
}

void test_matcher_links_overlap() {
  mscoupon::SliceMatcher m;
  configure_base(m, mscoupon::StatisticsConfig{});
  // Slice 0 CC: comp0 (px 0-1), comp1 (px 3-4); -1 = background. 6-neighbor.
  const CcSlice s0 = cc_slice({{2}, {2}});
  m.add_slice({0, 0, -1, 1, 1}, 5, 1, s0.stats, s0.channels, 0, 6);
  // Slice 1 CC: comp0 (px 0-1) overlaps slice-0 comp0; slice-0 comp1 has no successor.
  const CcSlice s1 = cc_slice({{2}});
  m.add_slice({0, 0, -1, -1, -1}, 5, 1, s1.stats, s1.channels, 1, 6);

  std::vector<mscoupon::FeatureMapRow> map;
  mscoupon::GlobalFeatureTable table;
  std::vector<mscoupon::GlobalLabelRaster> rasters;
  m.finalize(map, table, rasters);

  expect(map.size() == 3, "three (slice, cc) members");
  expect(global_for(map, 0, 0) == global_for(map, 1, 0), "overlapping component shares a global id");
  expect(global_for(map, 0, 1) != global_for(map, 0, 0), "non-overlapping component is distinct");
  expect(table.rows.size() == 2, "two global features");
  expect(rasters.size() == 2, "one relabeled raster per slice");
}

void test_matcher_merge_unifies() {
  mscoupon::SliceMatcher m;
  configure_base(m, mscoupon::StatisticsConfig{});
  const CcSlice s0 = cc_slice({{2}, {2}});
  m.add_slice({0, 0, -1, 1, 1}, 5, 1, s0.stats, s0.channels, 0, 6);
  // A single slice-1 component spans both columns -> bridges the two slice-0 comps.
  const CcSlice s1 = cc_slice({{5}});
  m.add_slice({0, 0, 0, 0, 0}, 5, 1, s1.stats, s1.channels, 1, 6);

  std::vector<mscoupon::FeatureMapRow> map;
  mscoupon::GlobalFeatureTable table;
  std::vector<mscoupon::GlobalLabelRaster> rasters;
  m.finalize(map, table, rasters);

  expect(table.rows.size() == 1, "merge unifies into one global feature");
  expect(global_for(map, 0, 0) == global_for(map, 0, 1), "both prior components unified");
  expect(global_for(map, 0, 0) == global_for(map, 1, 0), "current component unified with priors");
  expect(table.rows[0].voxel_count == 9, "voxel count sums all members (2+2+5)");
  expect(table.rows[0].first_slice == 0 && table.rows[0].last_slice == 1, "spans both slices");
  expect(table.rows[0].num_slices == 2, "counts two distinct slices");
  // The relabeled slice-0 raster carries the surviving global id everywhere it was set.
  expect(rasters[0].data[0] == table.rows[0].global_id &&
             rasters[0].data[3] == table.rows[0].global_id,
         "relabel pass writes the unified global id into slice 0");
}

void test_matcher_first_seen_ids() {
  mscoupon::SliceMatcher m;
  configure_base(m, mscoupon::StatisticsConfig{});
  // Two distinct components in one slice -> global ids in appearance (node) order.
  const CcSlice s0 = cc_slice({{2, 3.0f}, {2, 7.0f}});
  m.add_slice({0, 0, -1, 1, 1}, 5, 1, s0.stats, s0.channels, 0, 6);

  std::vector<mscoupon::FeatureMapRow> map;
  mscoupon::GlobalFeatureTable table;
  std::vector<mscoupon::GlobalLabelRaster> rasters;
  m.finalize(map, table, rasters);
  expect(global_for(map, 0, 0) == 0, "first-seen component -> global 0");
  expect(global_for(map, 0, 1) == 1, "next component -> global 1");
  // Base-channel aggregates now come out of the channel plane, read by name.
  expect(std::abs(global_field(table, 0, "mean_base") - 3.0) < 1e-5,
         "global 0 mean_base from its node");
  expect(std::abs(global_field(table, 1, "mean_base") - 7.0) < 1e-5,
         "global 1 mean_base from its node");
}

void test_matcher_relevance_range() {
  CcSlice a = cc_slice({{1, 3.0f}});
  a.channels.cell(0, 0).min = 1.0f; a.channels.cell(0, 0).max = 5.0f;
  a.stats[0].base_relevance_floor = 0.0f; a.stats[0].base_relevance_ceiling = 10.0f;
  CcSlice b = cc_slice({{1, 4.0f}});
  b.channels.cell(0, 0).min = 2.0f; b.channels.cell(0, 0).max = 9.0f;
  b.stats[0].base_relevance_floor = 1.0f; b.stats[0].base_relevance_ceiling = 11.0f;

  mscoupon::SliceMatcher m;
  configure_base(m, mscoupon::StatisticsConfig{});
  m.add_slice({0}, 1, 1, a.stats, a.channels, 0, 6);
  m.add_slice({0}, 1, 1, b.stats, b.channels, 1, 6);
  std::vector<mscoupon::FeatureMapRow> map;
  mscoupon::GlobalFeatureTable table;
  std::vector<mscoupon::GlobalLabelRaster> rasters;
  m.finalize(map, table, rasters);
  expect(table.rows.size() == 1, "overlapping relevance nodes merge");
  expect(std::abs(table.rows[0].relevance_base - 8.0f / 12.0f) < 1e-6f,
         "global relevance uses global base extent and carried slice range");
}

void test_global_csv_includes_relevance() {
  const auto path = std::filesystem::temp_directory_path() / "mscoupon_relevance.csv";
  mscoupon::GlobalFeatureTable table;
  mscoupon::GlobalFeatureStat row;
  row.relevance_base = 0.25f;
  table.rows.push_back(row);
  table.schema = msseg::resolve_stat_channels(base_spec());
  table.channels.reset(1, table.schema.size(), base_spec());
  mscoupon::write_global_table_csv(path, table, mscoupon::StatisticsConfig{});
  std::ifstream input(path);
  const std::string text((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  expect(text.find("relevance_base") != std::string::npos,
         "global CSV advertises relevance_base");
  expect(text.find(",0.25") != std::string::npos,
         "global CSV writes the relevance value");
  input.close();
  std::filesystem::remove(path);
}

void test_cc_stage_trim_and_split() {
  // 1x5 raster; MSC region 1 spans px0..3 (value: 5,5,1,5), region 2 = px4.
  const std::vector<int> labels = {1, 1, 1, 1, 2};
  mscoupon::Image2D base; base.width = 5; base.height = 1;
  base.pixels = {5.f, 5.f, 1.f, 5.f, 9.f};
  mscoupon::Image2D filt = base;
  std::unordered_set<int> keep = {1};                 // per-slice selection keeps region 1
  // Pixel trim: keep base >= 2 -> px2 (value 1) is cut, SPLITTING region 1 into two
  // in-plane components (px0-1 and px3).
  std::vector<mscoupon::PixelFilter> rules = {{"base", "keep", "ge", 2.0}};
  std::vector<int> cc; std::vector<mscoupon::CcNodeStat> stats;
  msseg::ChannelStats cc_channels;
  const msseg::StatChannelBank bank =
      msseg::build_stat_channels(to_diffg2d(base), to_diffg2d(filt), base_spec());
  const int n = mscoupon::label_selected_components(labels, 5, 1, base, filt, keep, rules, 6,
                                                    /*ascending=*/true, base_spec(), bank,
                                                    0.0f, 10.0f, cc, stats, cc_channels);
  expect(n == 2, "trim splits the selected region into two components");
  expect(cc[2] == -1, "trimmed pixel (value < 2) is background");
  expect(cc[4] == -1, "unselected region 2 is background");
  expect(cc[0] == cc[1] && cc[0] != cc[3], "two distinct in-plane components");
  std::size_t total = stats[0].area + stats[1].area;
  expect(total == 3, "three surviving pixels across the two components");

  // The per-component extremum is derived from the component's OWN pixels, not
  // inherited from the MSC feature -- which matters precisely here, where the
  // trim removed the region's true minimum (px2, value 1). Component 0 spans
  // px0-1 (both 5), component 1 is px3 (5).
  expect(std::abs(stats[0].ext_filtered - 5.0f) < 1e-6f,
         "component extremum ignores the trimmed-away minimum");
  expect(stats[0].ext_x >= 0.0f && stats[0].ext_x <= 1.0f, "extremum lies inside its component");
  expect(std::abs(stats[1].ext_x - 3.0f) < 1e-6f, "second component's extremum is its own pixel");
  expect(stats[0].base_relevance_floor == 0.0f &&
             stats[0].base_relevance_ceiling == 10.0f,
         "CC carries the slice relevance range");
}

// A 3D feature must inherit the extremum of its DEEPEST constituent slice, and
// as a tuple: reducing ext_x and ext_base independently would report a position
// from one slice with a value from another.
void test_matcher_carries_extremal_tuple() {
  // One 2x1 component per slice, carrying its own extremum tuple.
  const auto slice_of = [](std::size_t area, float ext_filtered, float ext_base, float ext_x) {
    CcSlice out;
    out.channels.reset(1, 1, base_spec());
    mscoupon::CcNodeStat s;
    s.area = area;
    s.min_x = 0; s.max_x = 1; s.min_y = 0; s.max_y = 1;
    s.ext_filtered = ext_filtered;
    s.ext_x = ext_x; s.ext_y = 0.0f;
    out.stats.push_back(s);
    msseg::ChannelAccum& a = out.channels.cell(0, 0);
    a.sum = static_cast<double>(area); a.min = 0.0f; a.max = 1.0f;
    out.channels.set_ext(0, 0, ext_base);
    return out;
  };
  // All overlapping -> a single 3D feature. The deepest (lowest ext_filtered) is
  // the MIDDLE slice, so a first- or last-wins bug would be invisible in a
  // two-slice test.
  const std::vector<int> cc = {0, 0};
  mscoupon::SliceMatcher m;
  mscoupon::StatisticsConfig cfg;
  cfg.per_slice_quantities = {"area"};
  configure_base(m, cfg);
  const CcSlice a = slice_of(2, 0.9f, 0.50f, 0.0f);
  const CcSlice b = slice_of(4, 0.1f, 0.11f, 1.0f);
  const CcSlice c = slice_of(6, 0.5f, 0.30f, 0.0f);
  m.add_slice(cc, 2, 1, a.stats, a.channels, 0, 6);
  m.add_slice(cc, 2, 1, b.stats, b.channels, 1, 6);
  m.add_slice(cc, 2, 1, c.stats, c.channels, 2, 6);

  std::vector<mscoupon::FeatureMapRow> map;
  mscoupon::GlobalFeatureTable table;
  std::vector<mscoupon::GlobalLabelRaster> rasters;
  m.finalize(map, table, rasters);

  expect(table.rows.size() == 1, "the three slices link into one global feature");
  const auto& g = table.rows.front();
  expect(g.num_slices == 3, "feature spans three slices");
  expect(std::abs(g.ext_filtered - 0.1f) < 1e-6f, "ascending 3D extremum is the minimum");
  expect(std::abs(global_field(table, 0, "ext_base") - 0.11) < 1e-6,
         "ext_base comes from the SAME slice");
  expect(std::abs(g.ext_x - 1.0f) < 1e-6f, "ext_x comes from the same slice too");
  expect(g.ext_z == 1, "ext_z names the middle slice");

  // Per-slice reductions run across slices: areas 2, 4, 6.
  expect(std::abs(g.per_slice.at("area_mean") - 4.0) < 1e-9, "area_mean over slices");
  expect(std::abs(g.per_slice.at("area_min") - 2.0) < 1e-9, "area_min over slices");
  expect(std::abs(g.per_slice.at("area_max") - 6.0) < 1e-9, "area_max over slices");
  expect(g.voxel_count == 12, "voxel_count still sums every slice");
}

// A descending manifold seeds from the maximum, so the merge must flip.
void test_matcher_extremum_follows_manifold_direction() {
  const auto slice_of = [](float ext_filtered) {
    CcSlice out;
    out.channels.reset(1, 1, base_spec());
    mscoupon::CcNodeStat s;
    s.area = 2; s.min_x = 0; s.max_x = 1; s.min_y = 0; s.max_y = 1;
    s.ext_filtered = ext_filtered;
    out.stats.push_back(s);
    out.channels.set_ext(0, 0, ext_filtered);
    return out;
  };
  const std::vector<int> cc = {0, 0};
  mscoupon::SliceMatcher m;
  configure_base(m, mscoupon::StatisticsConfig{}, /*ascending=*/false);
  const CcSlice a = slice_of(0.2f);
  const CcSlice b = slice_of(0.8f);
  m.add_slice(cc, 2, 1, a.stats, a.channels, 0, 6);
  m.add_slice(cc, 2, 1, b.stats, b.channels, 1, 6);

  std::vector<mscoupon::FeatureMapRow> map;
  mscoupon::GlobalFeatureTable table;
  std::vector<mscoupon::GlobalLabelRaster> rasters;
  m.finalize(map, table, rasters);
  expect(std::abs(table.rows.front().ext_filtered - 0.8f) < 1e-6f,
         "descending 3D extremum is the maximum");
}

void test_config_matching_flag() {
  mscoupon::AppConfig def;
  expect(def.matching.enabled, "matching is enabled by default");

  const auto dir = std::filesystem::temp_directory_path() / "mscoupon_matching_cfg";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const auto cfg_path = dir / "cfg.json";
  {
    std::ofstream f(cfg_path);
    f << R"({"input":{"folder":")" << (dir / "in").generic_string() << R"("},)"
      << R"("output":{"folder":")" << (dir / "out").generic_string() << R"("},)"
      << R"("msc":{"persistence_percent":10.0}})";
  }
  std::string cfg_str = cfg_path.string();
  std::vector<char> path_buf(cfg_str.begin(), cfg_str.end());
  path_buf.push_back('\0');

  char arg0[] = "mscoupon";
  char arg_config[] = "--config";
  char arg_no_match[] = "--no-matching";

  {
    char* argv[] = {arg0, arg_config, path_buf.data()};
    const auto cli = mscoupon::parse_cli(3, argv);
    const auto loaded = mscoupon::load_config(cli);
    expect(loaded.matching.enabled, "matching stays enabled without --no-matching");
  }
  {
    char* argv[] = {arg0, arg_config, path_buf.data(), arg_no_match};
    const auto cli = mscoupon::parse_cli(4, argv);
    expect(cli.disable_matching, "--no-matching sets the CLI override");
    const auto loaded = mscoupon::load_config(cli);
    expect(!loaded.matching.enabled, "--no-matching disables matching");
  }
  std::filesystem::remove_all(dir);
}

void test_relevance_config() {
  mscoupon::StatisticsConfig stats;
  mscoupon::parse_statistics_json(
      nlohmann::json::parse(
          R"({"statistics":{"relevance":{"enabled":true,"low_percentile":1.0,"high_percentile":99.0}}})"),
      stats);
  expect(stats.spec.relevance, "relevance config enables the field");
  expect(stats.spec.relevance_low_percentile == 1.0 &&
             stats.spec.relevance_high_percentile == 99.0,
         "relevance percentiles parse");

  bool rejected = false;
  try {
    mscoupon::StatisticsConfig invalid;
    mscoupon::parse_statistics_json(
        nlohmann::json::parse(
            R"({"statistics":{"relevance":{"low_percentile":99.0,"high_percentile":1.0}}})"),
        invalid);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  expect(rejected, "relevance rejects reversed percentiles");
}

void test_msc2d_pipeline_monotone_and_stats() {
  const int w = 48, h = 48;
  const diffg::Image<float> field = make_wells(w, h);

  msseg::Msc2DParams cfg;
  cfg.manifold = "ascending";
  cfg.persistence_absolute = 0.0f;
  cfg.persistence_percent.reset();
  cfg.stats.relevance_low_percentile = 25.0;
  cfg.stats.relevance_high_percentile = 75.0;

  msseg::Msc2DPipeline pipe;
  pipe.build(field, field, cfg);  // base == filtered here (topology on the raw field)
  expect(pipe.width() == w && pipe.height() == h, "pipeline reports image dims");
  expect(pipe.value_range() > 0.0f, "value range is positive");
  std::vector<float> sorted(field.data(), field.data() + field.size());
  std::sort(sorted.begin(), sorted.end());
  const auto percentile = [&](double q) {
    const double hq = static_cast<double>(sorted.size() - 1) * q / 100.0;
    const std::size_t lo = static_cast<std::size_t>(std::floor(hq));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(hq));
    return sorted[lo] + (sorted[hi] - sorted[lo]) * static_cast<float>(hq - lo);
  };
  expect(std::abs(pipe.base_relevance_floor() - percentile(25.0)) < 1e-6f &&
             std::abs(pipe.base_relevance_ceiling() - percentile(75.0)) < 1e-6f,
         "pipeline relevance range uses linear percentiles");

  // At persistence 0 the finest decomposition has multiple minima basins.
  pipe.select_persistence(0.0f);
  const int fine = distinct_features(pipe.labels());
  expect(fine >= 2, "finest decomposition has multiple features");

  // Raising the persistence never increases the feature count (monotone merge).
  pipe.select_persistence(pipe.value_range());  // very aggressive
  const int coarse = distinct_features(pipe.labels());
  expect(coarse <= fine, "feature count is monotone non-increasing in persistence");
  expect(coarse >= 1, "at least one feature remains");

  // Aggregated statistics: feature areas partition the labeled pixels exactly.
  pipe.select_persistence(0.3f * pipe.value_range());
  const auto feats = pipe.feature_stats();
  std::unordered_map<int, std::int64_t> area_of;
  std::int64_t labeled = 0;
  for (int v : pipe.labels())
    if (v >= 0) { area_of[v] += 1; ++labeled; }
  std::int64_t feat_area_sum = 0;
  for (std::size_t r = 0; r < feats.size(); ++r) {
    const msseg::Msc2DFeatureStat f = feats[r];
    feat_area_sum += f.area;
    const auto it = area_of.find(static_cast<int>(f.feature_id));
    expect(it != area_of.end() && it->second == f.area,
           "feature area matches its label pixel count");
    const msseg::ChannelAccum& b = pipe.feature_channels().cell(r, 0);
    expect(pipe.channels()[0].name == "base", "slot 0 is the base channel");
    expect(b.min <= b.max, "feature base min<=max");
    expect(f.min_x <= f.max_x && f.min_y <= f.max_y, "feature bbox ordering");
  }
  expect(feat_area_sum == labeled, "aggregated feature areas sum to labeled pixels");
  expect(static_cast<int>(feats.size()) == distinct_features(pipe.labels()),
         "one stat row per surviving feature");
}

void test_msc2d_pipeline_consistency() {
  // The pipeline now re-thresholds via MSCEER's NATIVE cancellation (no merge
  // tree), so its partition matches a fresh native compute_msc2d_labels at EVERY
  // persistence within the shared hierarchy cap -- not only at p=0. Invariants:
  //  (1) pipeline @ p == native compute_msc2d_labels @ p (same base + cancellation);
  //  (2) re-thresholding is idempotent (same persistence -> same labels).
  const int w = 48, h = 48;
  const diffg::Image<float> field = make_wells(w, h);

  msseg::Msc2DParams cfg;
  cfg.manifold = "ascending";
  cfg.persistence_percent.reset();

  // Build once at a cap that covers the tested persistences (the cap floors at
  // MSCEER's 10%-of-range default, so both paths keep the same 1% base complex).
  msseg::Msc2DPipeline pipe;
  msseg::Msc2DParams build_cfg = cfg;
  build_cfg.persistence_absolute = 0.0f;
  pipe.build(field, field, build_cfg);

  // Native match at several persistences within the shared 10% cap.
  for (const float frac : {0.0f, 0.02f, 0.05f, 0.09f}) {
    const float p = frac * pipe.value_range();
    pipe.select_persistence(p);
    msseg::Msc2DParams native = cfg;
    native.persistence_absolute = p;
    const std::vector<int> native_labels = msseg::compute_msc2d_labels(field, native);
    expect(same_partition(pipe.labels(), native_labels),
           "pipeline partition matches native compute_msc2d_labels at this persistence");
  }

  const float p = 0.05f * pipe.value_range();
  pipe.select_persistence(p);
  const std::vector<int> first = pipe.labels();
  pipe.select_persistence(p);
  expect(pipe.labels() == first, "re-thresholding at the same persistence is idempotent");
}

void test_msc2d_pipeline_descending() {
  // Descending (maxima basins) must build and re-threshold without error.
  const int w = 40, h = 40;
  const diffg::Image<float> field = make_wells(w, h);
  msseg::Msc2DParams cfg;
  cfg.manifold = "descending";
  cfg.persistence_absolute = 0.0f;
  cfg.persistence_percent.reset();

  msseg::Msc2DPipeline pipe;
  pipe.build(field, field, cfg);
  pipe.select_persistence(0.0f);
  const int fine = distinct_features(pipe.labels());
  pipe.select_persistence(pipe.value_range());
  const int coarse = distinct_features(pipe.labels());
  expect(fine >= 1 && coarse <= fine, "descending decomposition is monotone");
}

void test_msc2d_extremum_stats() {
  // The seeding critical point of an ascending 2-manifold is its deepest pixel,
  // so ext_filtered must equal filt_min and ext_base must be the BASE channel
  // read at that pixel. Offsetting base from filtered catches a channel mix-up
  // that a base == filtered fixture would hide.
  const int w = 48, h = 48;
  const diffg::Image<float> field = make_wells(w, h);
  diffg::Image<float> base(diffg::Dimensions{static_cast<std::size_t>(w), static_cast<std::size_t>(h), 1});
  for (std::size_t i = 0; i < field.size(); ++i) base.data()[i] = field.data()[i] + 10.0f;

  msseg::Msc2DParams cfg;
  cfg.manifold = "ascending";
  cfg.persistence_absolute = 0.0f;
  cfg.persistence_percent.reset();

  msseg::Msc2DPipeline pipe;
  pipe.build(base, field, cfg);
  pipe.select_persistence(0.0f);

  const auto at = [&](const diffg::Image<float>& img, int x, int y) {
    return img.data()[static_cast<std::size_t>(y) * w + x];
  };
  // ext_<channel> lives in the channel plane; slot 0 is "base" here.
  const auto check_seed = [&](const msseg::Msc2DPipeline& p,
                              const msseg::Msc2DFeatureStat& f, std::size_t row, bool asc) {
    const int px = static_cast<int>(f.ext_x), py = static_cast<int>(f.ext_y);
    expect(f.ext_x == static_cast<float>(px) && f.ext_y == static_cast<float>(py),
           "the extremum sits on a real pixel");
    expect(px >= f.min_x && px <= f.max_x && py >= f.min_y && py <= f.max_y,
           "extremum lies inside the feature's bounding box");
    expect(f.ext_filtered == at(field, px, py), "ext_filtered is the field at the extremum");
    expect(p.channels()[0].name == "base", "slot 0 is the base channel");
    expect(p.feature_channels().ext(row, 0) == at(base, px, py),
           "ext_base samples the BASE channel at the extremum");
    expect(f.ext_filtered == (asc ? f.filt_min : f.filt_max),
           "the seed is the feature's most extreme pixel in the chosen direction");
  };

  std::unordered_map<int, msseg::Msc2DFeatureStat> fine;
  const auto fine_feats = pipe.feature_stats();
  for (std::size_t r = 0; r < fine_feats.size(); ++r) {
    check_seed(pipe, fine_feats[r], r, true);
    fine[static_cast<int>(fine_feats[r].feature_id)] = fine_feats[r];
  }

  // A merged feature inherits the SURVIVING minimum, so its seed is unchanged by
  // re-thresholding -- even though the merged basin may now contain a lower pixel
  // (persistence, not depth, decides which minimum survives).
  pipe.select_persistence(0.5f * pipe.value_range());
  for (const auto& f : pipe.feature_stats()) {
    const auto it = fine.find(static_cast<int>(f.feature_id));
    expect(it != fine.end(), "surviving feature id was present at the finest level");
    expect(f.ext_x == it->second.ext_x && f.ext_y == it->second.ext_y,
           "the surviving extremum does not move when the persistence rises");
    expect(f.ext_filtered == it->second.ext_filtered,
           "the surviving extremum keeps its sampled values");
    expect(f.ext_filtered >= f.filt_min, "the seed value is within the merged feature's range");
  }

  // Descending: the seed is the maximum instead.
  msseg::Msc2DParams dsc = cfg;
  dsc.manifold = "descending";
  msseg::Msc2DPipeline dpipe;
  dpipe.build(base, field, dsc);
  dpipe.select_persistence(0.0f);
  int dsc_features = 0;
  const auto dsc_feats = dpipe.feature_stats();
  for (std::size_t r = 0; r < dsc_feats.size(); ++r) {
    check_seed(dpipe, dsc_feats[r], r, false);
    ++dsc_features;
  }
  expect(dsc_features > 0, "descending decomposition produced features");
}

void test_msc2d_extremum_sample_radius() {
  // A single well in a 9x9 field: the minimum sits at the centre, and a radius
  // of 1 must average the 3x3 base window around it (clamped at the border).
  const int w = 9, h = 9;
  diffg::Image<float> field(diffg::Dimensions{9, 9, 1});
  diffg::Image<float> base(diffg::Dimensions{9, 9, 1});
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float dx = static_cast<float>(x - 4), dy = static_cast<float>(y - 4);
      field.data()[static_cast<std::size_t>(y) * w + x] = dx * dx + dy * dy;
      // Base varies independently, and non-linearly in x, so the window mean is
      // genuinely different from the centre pixel.
      base.data()[static_cast<std::size_t>(y) * w + x] = static_cast<float>(x * x + 10 * y);
    }
  }

  msseg::Msc2DParams cfg;
  cfg.manifold = "ascending";
  cfg.persistence_absolute = 0.0f;
  cfg.persistence_percent.reset();

  msseg::Msc2DPipeline sharp;
  sharp.build(base, field, cfg);
  sharp.select_persistence(0.0f);

  cfg.extremum_sample_radius = 1;
  msseg::Msc2DPipeline blurred;
  blurred.build(base, field, cfg);
  blurred.select_persistence(0.0f);

  // ext_base now lives in the channel plane, at the "base" slot.
  const auto base_slot = [](const msseg::Msc2DPipeline& p) {
    for (std::size_t k = 0; k < p.channels().size(); ++k) {
      if (p.channels()[k].name == "base") return k;
    }
    throw std::runtime_error("no base channel");
  };
  const auto ext_base_of = [&](const msseg::Msc2DPipeline& p, int fid) {
    const auto feats = p.feature_stats();
    for (std::size_t r = 0; r < feats.size(); ++r) {
      if (static_cast<int>(feats[r].feature_id) == fid) {
        return p.feature_channels().ext(r, base_slot(p));
      }
    }
    throw std::runtime_error("feature id not found");
  };
  const auto seed_of = [](const msseg::Msc2DPipeline& p, int fid) {
    for (const auto& f : p.feature_stats())
      if (static_cast<int>(f.feature_id) == fid) return f;
    throw std::runtime_error("feature id not found");
  };
  for (const auto& f : sharp.feature_stats()) {
    const int fid = static_cast<int>(f.feature_id);
    const auto b = seed_of(blurred, fid);
    expect(b.ext_x == f.ext_x && b.ext_y == f.ext_y, "the sample radius does not move the extremum");
    const int px = static_cast<int>(f.ext_x);
    const int py = static_cast<int>(f.ext_y);
    double sum = 0.0;
    int n = 0;
    for (int y = std::max(0, py - 1); y <= std::min(h - 1, py + 1); ++y) {
      for (int x = std::max(0, px - 1); x <= std::min(w - 1, px + 1); ++x) {
        sum += base.data()[static_cast<std::size_t>(y) * w + x];
        ++n;
      }
    }
    const float sharp_ext = ext_base_of(sharp, fid);
    const float blurred_ext = ext_base_of(blurred, fid);
    expect(std::abs(blurred_ext - static_cast<float>(sum / n)) < 1e-4f,
           "radius 1 averages the clamped 3x3 base window");
    expect(sharp_ext == base.data()[static_cast<std::size_t>(py) * w + px],
           "radius 0 reads the single critical pixel");
    expect(blurred_ext != sharp_ext, "the sample radius actually changes ext_base");
  }
}

// Project one synthetic feature under `spec` and return its row as a map.
// Base slot carries sum 500 / sumsq 2600 / extent [1, 9] over area 100 (mean 5);
// filtered slot, when the spec enables it, carries sum 200 / sumsq 500 / [0, 4].
std::unordered_map<std::string, double> synthetic_row(const msseg::StatsSpec& spec) {
  msseg::Msc2DFeatureStat s;
  s.feature_id = 3;
  s.area = 100;
  s.base_relevance_floor = 0.0f; s.base_relevance_ceiling = 10.0f;
  s.filt_min = 0.0f; s.filt_max = 4.0f;
  s.min_x = 5; s.max_x = 14; s.min_y = 0; s.max_y = 19;
  s.ext_x = 7.0f; s.ext_y = 11.0f; s.ext_filtered = 0.0f;

  const auto schema = msseg::resolve_stat_channels(spec);
  msseg::ChannelStats channels;
  channels.reset(1, schema.size(), spec);
  for (std::size_t k = 0; k < schema.size(); ++k) {
    msseg::ChannelAccum& a = channels.cell(0, k);
    if (schema[k].name == "filtered") {
      a.sum = 200.0; a.sumsq = 500.0; a.min = 0.0f; a.max = 4.0f;
      channels.set_ext(0, k, 0.0f);
    } else {
      a.sum = 500.0; a.sumsq = 2600.0; a.min = 1.0f; a.max = 9.0f;
      channels.set_ext(0, k, 1.5f);
    }
  }
  const auto table = mscoupon::feature_table({s}, channels, schema, spec);
  return mscoupon::feature_row(table, 0);
}

void test_feature_query() {
  msseg::StatsSpec spec;  // defaults: base channel, all reductions, extremum on
  const auto row = synthetic_row(spec);

  expect(std::abs(row.at("mean_base") - 5.0) < 1e-9, "mean_base = base_sum/area");
  expect(std::abs(row.at("relevance_base") - 8.0 / 11.0) < 1e-9,
         "relevance_base uses the shifted slice range");
  expect(std::abs(row.at("bbox_w") - 10.0) < 1e-9, "bbox_w = max_x-min_x+1");
  expect(std::abs(row.at("bbox_h") - 20.0) < 1e-9, "bbox_h = max_y-min_y+1");

  const auto Q = [](std::string f, std::string op, double v, double v2 = 0.0) {
    mscoupon::FeatureQuery q; q.field = std::move(f); q.op = std::move(op); q.value = v; q.value2 = v2; return q;
  };
  expect(mscoupon::row_passes(row, {Q("area", "ge", 50)}), "area>=50 passes");
  expect(!mscoupon::row_passes(row, {Q("area", "lt", 50)}), "area<50 fails");
  expect(mscoupon::row_passes(row, {Q("mean_base", "between", 4.0, 6.0)}), "between passes");
  expect(mscoupon::row_passes(row, {Q("area", "ge", 50), Q("mean_base", "gt", 4.0)}),
         "AND-chain of satisfied predicates passes");
  expect(!mscoupon::row_passes(row, {Q("area", "ge", 50), Q("mean_base", "gt", 6.0)}),
         "AND-chain fails when any predicate fails");
  expect(!mscoupon::row_passes(row, {Q("no_such_field", "gt", 0.0)}),
         "unknown field excludes the feature");

  // The extremum sample is queryable, and the schema guard knows it.
  expect(std::abs(row.at("ext_base") - 1.5) < 1e-9, "ext_base is exposed to queries");
  expect(mscoupon::row_passes(row, {Q("ext_base", "lt", 2.0)}), "ext_base predicate passes");
  expect(!mscoupon::row_passes(row, {Q("ext_base", "gt", 2.0)}), "ext_base predicate fails");
  expect(mscoupon::is_feature_field("ext_base", spec) && mscoupon::is_feature_field("area", spec),
         "is_feature_field accepts real statistics");
  expect(!mscoupon::is_feature_field("ext_bse", spec), "is_feature_field rejects a typo");
  expect(mscoupon::is_feature_field("relevance_base", spec),
         "relevance_base is in the default schema");
  expect(mscoupon::row_passes(row, {Q("relevance_base", "gt", 0.7)}),
         "relevance_base is queryable");
  expect(mscoupon::relevance_base_value(0.0, 0.0, 0.0, 0.0) == 0.0,
         "zero-over-zero relevance is defined as zero");
  expect(std::isinf(mscoupon::relevance_base_value(0.0, 1.0, 0.0, 0.0)),
         "nonzero-over-zero relevance is positive infinity");

  // The filtered aggregates had no reader anywhere, so they are off by default
  // -- and a config still naming one must fail loudly rather than silently
  // matching nothing (row_passes fails closed on an unknown field).
  expect(!mscoupon::is_feature_field("mean_filtered", spec),
         "filtered aggregates are not in the default schema");
  expect(row.find("mean_filtered") == row.end(), "and are not computed");
  msseg::StatsSpec with_filtered = spec;
  with_filtered.filtered_channel = true;
  expect(mscoupon::is_feature_field("mean_filtered", with_filtered),
         "opting the filtered channel back in restores them");
  msseg::StatsSpec no_relevance = spec;
  no_relevance.relevance = false;
  expect(!mscoupon::is_feature_field("relevance_base", no_relevance),
         "disabling relevance removes it from the schema");

  // Turning a reduction off removes exactly that field, on every channel.
  msseg::StatsSpec no_std = spec;
  no_std.std = false;
  const auto lean = synthetic_row(no_std);
  expect(lean.find("std_base") == lean.end(), "a disabled reduction is not emitted");
  expect(lean.count("mean_base") == 1, "the others are unaffected");

  // feature_fields() and the projected table share one schema, so the GUI
  // dropdown, the CSV header and config validation cannot disagree.
  const auto names = mscoupon::feature_fields(spec);
  expect(names.size() == row.size(), "feature_fields matches the row it describes");
  for (const auto& n : names) expect(row.count(n) == 1, "every advertised field is produced");
  const auto schema = mscoupon::feature_schema(spec);
  expect(schema.size() == names.size(), "feature_schema and feature_fields agree in size");
  for (std::size_t i = 0; i < schema.size(); ++i) {
    expect(schema[i].name == names[i], "feature_schema and feature_fields agree in order");
  }
}

// A derived channel request expands to one measurement channel per sigma (two
// for hessian), and every reduction of every channel becomes a queryable field.
void test_derived_stat_channels() {
  msseg::StatsSpec spec;
  spec.mean = true; spec.min = true; spec.max = true; spec.std = false;
  spec.derived.push_back(msseg::StatChannelRequest{"blur", {0.7, 1.5, 3.0}, true, ""});
  spec.derived.push_back(msseg::StatChannelRequest{"hessian", {1.5}, true, ""});

  const auto channels = msseg::resolve_stat_channels(spec);
  expect(channels.size() == 1 + 3 + 2, "base + three blurs + two hessian eigenvalues");
  expect(channels[0].name == "base", "base keeps slot 0");
  expect(channels[1].name == "blur_s0.7", "sigma renders with %g");
  expect(channels[3].name == "blur_s3", "a whole sigma drops its trailing zero");
  expect(channels[4].name == "hess_largest_s1.5" || channels[4].name == "hessian_largest_s1.5",
         "hessian expands to a largest slot first");
  expect(channels[5].kind == "hessian" && channels[5].slot_in_request == 1,
         "and a smallest slot second");

  // Every channel contributes its enabled reductions plus an extremum sample.
  expect(mscoupon::is_feature_field("mean_blur_s1.5", spec), "derived aggregates are queryable");
  expect(mscoupon::is_feature_field("max_blur_s3", spec), "on every requested sigma");
  expect(mscoupon::is_feature_field("ext_blur_s1.5", spec),
         "and the seeding extremum is sampled on each");
  expect(!mscoupon::is_feature_field("std_blur_s1.5", spec),
         "a disabled reduction is off on derived channels too");
  expect(!mscoupon::is_feature_field("mean_blur_s2", spec), "an unrequested sigma has no field");

  // The schema carries channel and reduction structurally, so the GUI never has
  // to parse "mean_blur_s0.7" -- and min_x is not read as min-of-channel-x.
  const auto schema = mscoupon::feature_schema(spec);
  bool saw_geometry = false, saw_derived = false;
  for (const auto& f : schema) {
    if (f.name == "min_x") {
      expect(f.channel.empty() && f.reduction.empty(), "min_x is geometry, not a reduction");
      saw_geometry = true;
    }
    if (f.name == "mean_blur_s0.7") {
      expect(f.channel == "blur_s0.7" && f.reduction == "mean",
             "a derived column names its channel and reduction");
      saw_derived = true;
    }
  }
  expect(saw_geometry && saw_derived, "both column kinds are present");

  // An empty derived list is exactly the previous two-channel behaviour.
  msseg::StatsSpec plain;
  expect(msseg::resolve_stat_channels(plain).size() == 1, "default spec is base only");
}

// The measured values must be the filters diffg would compute on their own.
void test_stat_channel_bank_matches_single_filters() {
  const int w = 24, h = 20;
  diffg::Image<float> base(diffg::Dimensions{static_cast<std::size_t>(w),
                                             static_cast<std::size_t>(h), 1});
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      base.data()[static_cast<std::size_t>(y) * w + x] =
          static_cast<float>(std::sin(0.3 * x) * std::cos(0.2 * y) + 0.01 * x * y);
    }
  }
  msseg::StatsSpec spec;
  spec.derived.push_back(msseg::StatChannelRequest{"blur", {1.5}, true, ""});
  const auto bank = msseg::build_stat_channels(base, base, spec);
  expect(bank.size() == 2, "base plus one derived channel");
  expect(bank.channel(0) == base.data(), "the base channel is aliased, not copied");

  msseg::FilterParams blur;
  blur.operation = "blur";
  blur.params["sigma"] = 1.5;
  const auto reference = msseg::apply_filter(base, blur);
  for (std::size_t i = 0; i < reference.size(); ++i) {
    expect(bank.channel(1)[i] == reference.data()[i],
           "the bank is bit-identical to the standalone filter");
  }
}

// ---------------------------------------------------------------------------
// GMM (mscoupon/gmm.hpp)
// ---------------------------------------------------------------------------

// Ground truth for the mixture the GMM tests draw from.
constexpr double kMu1 = 10.0, kSigma1 = 1.0, kWeight1 = 0.3;
constexpr double kMu2 = 20.0, kSigma2 = 2.0, kWeight2 = 0.7;
constexpr std::size_t kGmmSamples = 80000;

// Draw from the known two-component mixture with a fixed generator, as float32
// pixels (what a TIFF slice hands the fitter). Deterministic across runs.
std::vector<float> make_mixture(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  std::normal_distribution<double> g1(kMu1, kSigma1);
  std::normal_distribution<double> g2(kMu2, kSigma2);
  std::vector<float> out(n);
  for (auto& v : out) v = static_cast<float>(coin(rng) < kWeight1 ? g1(rng) : g2(rng));
  return out;
}

// Independent reference for numpy.percentile(..., method="linear"), to check the
// trim cut points against something other than the implementation under test.
double ref_percentile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  const double h = static_cast<double>(v.size() - 1) * q / 100.0;
  const auto lo = static_cast<std::size_t>(std::floor(h));
  const auto hi = std::min(v.size() - 1, static_cast<std::size_t>(std::ceil(h)));
  return v[lo] + (v[hi] - v[lo]) * (h - static_cast<double>(lo));
}

// Reconstruction intensities sit near 1e-4, where a whole slice's variance is
// about 1e-6 -- the same size as sklearn's default reg_covar. At that setting
// every component's variance floors at reg_covar, the two Gaussians become
// indistinguishable and EM collapses both means onto the global mean while
// still reporting converged=true. The normalize filter must not default into
// that, so pin the default preset's behaviour on realistically-scaled data.
void test_gmm_default_preset_survives_low_intensities() {
  // Two populations at 5e-5 and 3.9e-4, sigma 1.2e-4: a scaled-down copy of a
  // real coupon slice's air/metal split.
  std::mt19937_64 rng(4242);
  std::normal_distribution<double> air(5.0e-5, 1.2e-4), metal(3.9e-4, 1.2e-4);
  std::vector<float> px;
  px.reserve(40000);
  for (int i = 0; i < 20000; ++i) px.push_back(static_cast<float>(air(rng)));
  for (int i = 0; i < 20000; ++i) px.push_back(static_cast<float>(metal(rng)));

  const auto opts = mscoupon::parse_gmm_options(nlohmann::json::object());
  const auto r = mscoupon::fit_gmm(px.data(), px.size(), opts);
  expect(r.components.size() == 2, "two components returned");

  const double lo = r.components[0].mean, hi = r.components[1].mean;
  const double sep = hi - lo;
  expect(sep > 1.0e-4, "default preset must separate low-intensity populations");
  // The collapse signature: identical means AND identical sigmas.
  expect(std::abs(r.components[0].sigma - r.components[1].sigma) > 1e-12 || sep > 1.0e-4,
         "components must not be the same Gaussian");
  expect(std::abs(lo - 5.0e-5) < 8.0e-5, "low component near the planted air mean");
  expect(std::abs(hi - 3.9e-4) < 8.0e-5, "high component near the planted metal mean");

  // The explicit opt-in still selects the sklearn-default reg_covar.
  nlohmann::json two = nlohmann::json::object();
  two["preset"] = "two_gaussian";
  expect(mscoupon::parse_gmm_options(two).reg_covar == 1e-6, "two_gaussian keeps reg_covar 1e-6");
  expect(opts.reg_covar == 1e-12, "default preset is 'measure' (reg_covar 1e-12)");
}

void test_gmm_recovers_two_gaussians() {
  const std::vector<float> px = make_mixture(kGmmSamples, 12345);
  const auto r = mscoupon::fit_gmm(px.data(), px.size(), mscoupon::gmm_options_two_gaussian());

  expect(r.components.size() == 2, "two components returned");
  expect(r.n_valid == static_cast<std::int64_t>(kGmmSamples), "all samples are valid");
  expect(r.n_sampled == r.n_valid && r.n_fit == r.n_valid, "no subsample or trim by default");
  expect(r.converged, "EM converged");
  expect(r.components[0].mean < r.components[1].mean, "components sorted by increasing mean");

  const auto& c1 = r.components[0];
  const auto& c2 = r.components[1];
  expect(std::abs(c1.mean - kMu1) < 0.05, "component 1 mean recovered");
  expect(std::abs(c1.sigma - kSigma1) < 0.05, "component 1 sigma recovered");
  expect(std::abs(c1.weight - kWeight1) < 0.01, "component 1 weight recovered");
  expect(std::abs(c2.mean - kMu2) < 0.05, "component 2 mean recovered");
  expect(std::abs(c2.sigma - kSigma2) < 0.05, "component 2 sigma recovered");
  expect(std::abs(c2.weight - kWeight2) < 0.01, "component 2 weight recovered");
  expect(std::abs(c1.weight + c2.weight - 1.0) < 1e-9, "weights sum to one");
}


// A stack's no-data padding is not always 0 -- some reconstructions pad with an
// arbitrary constant. Dropping the wrong value leaves that plateau in the fit as
// a spurious population, which is exactly the failure this guards.
void test_no_data_sentinel_is_configurable() {
  // Two real populations at 10 and 30, plus a large no-data plateau at 43.
  std::vector<float> px;
  px.reserve(30000);
  std::mt19937_64 rng(99);
  std::normal_distribution<double> lo(10.0, 1.0), hi(30.0, 1.0);
  for (int i = 0; i < 5000; ++i) px.push_back(static_cast<float>(lo(rng)));
  for (int i = 0; i < 5000; ++i) px.push_back(static_cast<float>(hi(rng)));
  for (int i = 0; i < 20000; ++i) px.push_back(43.0f);   // the padding

  auto opts = mscoupon::gmm_options_measure();
  opts.compute_hard_stats = false;

  // Default sentinel 0 does not match the padding, so 43 survives and hijacks a
  // component -- the symptom on a real stack.
  opts.omit_value = 0.0;
  const auto naive = mscoupon::fit_gmm(px.data(), px.size(), opts);
  expect(naive.n_valid == static_cast<std::int64_t>(px.size()),
         "sentinel 0 keeps every pixel of a 43-padded stack");
  expect(std::abs(naive.components[1].mean - 43.0) < 3.0,
         "the padding is fitted as a population when the sentinel is wrong");

  // Naming the real sentinel drops the plateau and recovers 10 / 30.
  opts.omit_value = 43.0;
  const auto fixed = mscoupon::fit_gmm(px.data(), px.size(), opts);
  expect(fixed.n_valid == 10000, "the 43 plateau is masked out");
  expect(std::abs(fixed.components[0].mean - 10.0) < 1.0, "low population recovered");
  expect(std::abs(fixed.components[1].mean - 30.0) < 1.0, "high population recovered");

  // JSON: omit_value wins over the legacy boolean, null means keep everything.
  auto j = nlohmann::json::object();
  j["omit_value"] = 43.0;
  expect(mscoupon::parse_gmm_options(j).omit_value.value_or(-1.0) == 43.0,
         "omit_value parses");
  j["omit_zeros"] = true;
  expect(mscoupon::parse_gmm_options(j).omit_value.value_or(-1.0) == 43.0,
         "an explicit omit_value wins over the legacy omit_zeros");
  auto legacy = nlohmann::json::object();
  legacy["omit_zeros"] = false;
  expect(!mscoupon::parse_gmm_options(legacy).omit_value.has_value(),
         "legacy omit_zeros=false still means keep everything");
  auto nulled = nlohmann::json::object();
  nulled["omit_value"] = nullptr;
  expect(!mscoupon::parse_gmm_options(nulled).omit_value.has_value(),
         "omit_value null means keep everything");

  // The histogram measure shares the policy and its n_zero counter follows it.
  mscoupon::HistogramOptions h;
  h.omit_value = 43.0;
  h.bins = 128;
  h.peak_window = 4;
  h.min_peak_distance = 8;
  const auto hr = mscoupon::measure_histogram(px.data(), px.size(), h);
  expect(hr.n_zero == 20000, "n_zero counts the configured sentinel, not literal zeros");
  expect(hr.n_valid == 10000, "the histogram masks the same plateau");
}

void test_gmm_omit_zeros() {
  const std::vector<float> clean = make_mixture(kGmmSamples, 999);

  // Same samples, in the same order, buried in no-data zeros and NaN/Inf.
  std::vector<float> dirty;
  dirty.reserve(clean.size() * 2 + 64);
  for (std::size_t i = 0; i < clean.size(); ++i) {
    dirty.push_back(0.0f);
    if (i % 5000 == 0) dirty.push_back(std::numeric_limits<float>::quiet_NaN());
    if (i % 7000 == 0) dirty.push_back(-std::numeric_limits<float>::infinity());
    dirty.push_back(clean[i]);
  }

  const auto opts = mscoupon::gmm_options_two_gaussian();
  const auto a = mscoupon::fit_gmm(clean.data(), clean.size(), opts);
  const auto b = mscoupon::fit_gmm(dirty.data(), dirty.size(), opts);

  expect(b.n_valid == static_cast<std::int64_t>(clean.size()),
         "mask drops exact zeros and non-finite values");
  for (std::size_t c = 0; c < 2; ++c) {
    expect(std::abs(a.components[c].mean - b.components[c].mean) < 1e-9, "masking preserves mean");
    expect(std::abs(a.components[c].sigma - b.components[c].sigma) < 1e-9, "masking preserves sigma");
    expect(std::abs(a.components[c].weight - b.components[c].weight) < 1e-9,
           "masking preserves weight");
  }

  // With omit_zeros off the zeros become a third population, so the low
  // component is dragged well away from the true mean.
  auto keep_zeros = opts;
  keep_zeros.omit_value.reset();
  const auto z = mscoupon::fit_gmm(dirty.data(), dirty.size(), keep_zeros);
  expect(z.n_valid > b.n_valid, "no sentinel keeps the background pixels");
  expect(z.components[0].mean < kMu1 - 1.0, "background zeros pull the low component down");
}

void test_gmm_downsample() {
  const std::vector<float> px = make_mixture(kGmmSamples, 4242);
  auto opts = mscoupon::gmm_options_two_gaussian();
  opts.downsample_factor = 100;
  const auto r = mscoupon::fit_gmm(px.data(), px.size(), opts);

  expect(r.n_valid == static_cast<std::int64_t>(kGmmSamples), "n_valid counts before subsampling");
  expect(r.n_sampled == r.n_valid / 100, "subsample keeps 1/100 of the valid pixels");
  expect(r.n_fit == r.n_sampled, "no trim configured");
  expect(std::abs(r.components[0].mean - kMu1) < 0.3, "component 1 mean survives 1/100 subsample");
  expect(std::abs(r.components[1].mean - kMu2) < 0.3, "component 2 mean survives 1/100 subsample");
  expect(std::abs(r.components[0].weight - kWeight1) < 0.05, "weights survive 1/100 subsample");

  // The subsample is seeded, so the whole fit is reproducible.
  const auto again = mscoupon::fit_gmm(px.data(), px.size(), opts);
  expect(again.components[0].mean == r.components[0].mean, "same seed -> identical fit");

  auto other_seed = opts;
  other_seed.seed = 7;
  const auto shifted = mscoupon::fit_gmm(px.data(), px.size(), other_seed);
  expect(shifted.components[0].mean != r.components[0].mean, "a different seed draws differently");
}

void test_gmm_quantile_init() {
  const std::vector<float> px = make_mixture(kGmmSamples, 777);
  const auto a = mscoupon::fit_gmm(px.data(), px.size(), mscoupon::gmm_options_two_gaussian());

  auto measure = mscoupon::gmm_options_measure();
  measure.compute_hard_stats = false;
  const auto b = mscoupon::fit_gmm(px.data(), px.size(), measure);

  expect(b.converged, "quantile-initialised EM converged");
  for (std::size_t c = 0; c < 2; ++c) {
    expect(std::abs(a.components[c].mean - b.components[c].mean) < 1e-3,
           "quantile init reaches the same optimum as k-means init");
    expect(std::abs(a.components[c].sigma - b.components[c].sigma) < 1e-3,
           "quantile init reaches the same sigma");
  }
}

void test_gmm_hard_stats() {
  const std::vector<float> px = make_mixture(kGmmSamples, 31337);
  const auto r = mscoupon::fit_gmm(px.data(), px.size(), mscoupon::gmm_options_measure());

  const auto& c1 = r.components[0];
  const auto& c2 = r.components[1];
  expect(c1.n_hard + c2.n_hard == r.n_fit, "hard assignment partitions the fitted pixels");
  expect(std::abs(c1.hard_mean - kMu1) < 0.1, "component 1 hard mean near the true mean");
  expect(std::abs(c1.median - kMu1) < 0.1, "component 1 median near the true mean");
  expect(std::abs(c2.hard_mean - kMu2) < 0.1, "component 2 hard mean near the true mean");
  expect(std::abs(c2.median - kMu2) < 0.1, "component 2 median near the true mean");

  // The mode is a much coarser estimator than the mean or median: near the peak
  // of a Gaussian the density is flatter than the per-bin Poisson noise, so the
  // argmax bin wanders over a sizeable plateau (measured scatter here is a few
  // tenths of a sigma at 512 bins). Assert only that it lands in the peak
  // region; test_gmm_mode_unimodal pins down the estimator itself.
  expect(std::abs(c1.mode - kMu1) < 0.5 * kSigma1, "component 1 mode is in the peak region");
  expect(std::abs(c2.mode - kMu2) < 0.5 * kSigma2, "component 2 mode is in the peak region");

  // Without the flag the hard-assignment fields stay at their defaults.
  auto plain = mscoupon::gmm_options_measure();
  plain.compute_hard_stats = false;
  const auto p = mscoupon::fit_gmm(px.data(), px.size(), plain);
  expect(p.components[0].n_hard == 0, "hard stats are opt-in");
}

void test_gmm_mode_unimodal() {
  // estimate_mode() in isolation: one component, clean unimodal data, so nothing
  // but the histogram-plus-parabolic-interpolation peak finder is under test. It
  // must be unbiased (no systematic drift with the bin count) and land close to
  // the true peak, which for a Gaussian coincides with the mean and median.
  std::mt19937_64 rng(4242);
  std::normal_distribution<double> nd(kMu1, kSigma1);
  std::vector<float> pure(kGmmSamples);
  for (auto& v : pure) v = static_cast<float>(nd(rng));

  double drift = 0.0;
  int trials = 0;
  for (const int bins : {32, 128, 512}) {
    auto opts = mscoupon::gmm_options_measure();
    opts.n_components = 1;
    opts.mode_bins = bins;
    const auto r = mscoupon::fit_gmm(pure.data(), pure.size(), opts);

    expect(r.components.size() == 1, "single-component fit returns one component");
    expect(std::abs(r.components[0].mean - kMu1) < 0.02, "single component mean recovered");
    expect(std::abs(r.components[0].sigma - kSigma1) < 0.02, "single component sigma recovered");
    expect(r.components[0].n_hard == r.n_fit, "one component takes every pixel");
    expect(std::abs(r.components[0].median - kMu1) < 0.05, "median of a Gaussian is its mean");
    expect(std::abs(r.components[0].mode - kMu1) < 0.3, "mode lands on the peak");
    drift += r.components[0].mode - kMu1;
    ++trials;
  }
  expect(std::abs(drift / trials) < 0.15, "the mode estimate is not systematically biased");
}

void test_gmm_trim() {
  const std::vector<float> px = make_mixture(kGmmSamples, 24680);
  std::vector<double> as_double(px.begin(), px.end());

  auto opts = mscoupon::gmm_options_two_gaussian();
  opts.trim_percent = 0.5;
  const auto r = mscoupon::fit_gmm(px.data(), px.size(), opts);

  expect(std::abs(r.trim_lo - ref_percentile(as_double, 0.5)) < 1e-9, "trim_lo is the 0.5th pct");
  expect(std::abs(r.trim_hi - ref_percentile(as_double, 99.5)) < 1e-9, "trim_hi is the 99.5th pct");

  std::int64_t expected = 0;
  for (const double v : as_double)
    if (v >= r.trim_lo && v <= r.trim_hi) ++expected;
  expect(r.n_fit == expected, "trim keeps exactly the pixels inside the cut points");
  expect(r.n_fit < r.n_sampled, "trim removes the tails");
  expect(std::abs(static_cast<double>(r.n_fit) / static_cast<double>(r.n_sampled) - 0.99) < 0.01,
         "0.5% trimming at each end keeps about 99% of the pixels");

  // Untrimmed, trim_lo/trim_hi report the sample range instead.
  const auto u = mscoupon::fit_gmm(px.data(), px.size(), mscoupon::gmm_options_two_gaussian());
  const auto mm = std::minmax_element(as_double.begin(), as_double.end());
  expect(u.trim_lo == *mm.first && u.trim_hi == *mm.second,
         "without trimming the cut points are the data range");
}

void test_gmm_rejects_degenerate_input() {
  const std::vector<float> few = {1.f, 2.f, 3.f, 0.f, 0.f};
  bool threw = false;
  try {
    mscoupon::fit_gmm(few.data(), few.size(), mscoupon::gmm_options_two_gaussian());
  } catch (const std::exception&) {
    threw = true;
  }
  expect(threw, "fewer than 10 valid pixels is an error");

  const std::vector<float> constant(64, 5.f);
  threw = false;
  try {
    mscoupon::fit_gmm(constant.data(), constant.size(), mscoupon::gmm_options_two_gaussian());
  } catch (const std::exception&) {
    threw = true;
  }
  expect(threw, "a constant image cannot be fitted");
}

void test_gmm_integer_pixels() {
  // Integer inputs skip the finite test and are widened to double, mirroring the
  // np.issubdtype(x.dtype, np.floating) branch in the Python.
  const std::vector<float> px = make_mixture(kGmmSamples, 5150);
  std::vector<std::int16_t> ints(px.size());
  for (std::size_t i = 0; i < px.size(); ++i)
    ints[i] = static_cast<std::int16_t>(std::lround(px[i] * 100.0f));

  const auto r = mscoupon::fit_gmm(ints.data(), ints.size(), mscoupon::gmm_options_two_gaussian());
  expect(std::abs(r.components[0].mean - kMu1 * 100.0) < 5.0, "int16 component 1 mean recovered");
  expect(std::abs(r.components[1].mean - kMu2 * 100.0) < 5.0, "int16 component 2 mean recovered");
  expect(std::abs(r.components[0].sigma - kSigma1 * 100.0) < 5.0, "int16 component 1 sigma recovered");
}

void test_gmm_image_overload() {
  mscoupon::Image2D image;
  image.width = 400;
  image.height = 200;
  image.pixels = make_mixture(static_cast<std::size_t>(image.width) * image.height, 606);
  const auto r = mscoupon::fit_gmm(image, mscoupon::gmm_options_two_gaussian());
  expect(r.n_valid == static_cast<std::int64_t>(image.pixels.size()), "image overload sees all pixels");
  expect(std::abs(r.components[0].mean - kMu1) < 0.1, "image overload fits the low component");
  expect(std::abs(r.components[1].mean - kMu2) < 0.1, "image overload fits the high component");
}

// ---------------------------------------------------------------------------
// Two-point normalization (mscoupon/normalize.hpp, histogram_peaks.hpp,
// region_measure.hpp)
// ---------------------------------------------------------------------------

mscoupon::Image2D make_mixture_image(int w, int h, std::uint64_t seed) {
  mscoupon::Image2D image;
  image.width = w;
  image.height = h;
  image.pixels = make_mixture(static_cast<std::size_t>(w) * h, seed);
  return image;
}

void test_two_point_round_trip() {
  const mscoupon::TwoPoint tp{2.0, 6.0};
  expect(tp.valid(), "2..6 is a valid pair");
  expect(std::abs(tp.scale() - 4.0) < 1e-12, "scale is high-low");
  // The headline behaviour: "0.7" means 0.3*low + 0.7*high.
  expect(std::abs(tp.to_raw(0.7) - (0.3 * 2.0 + 0.7 * 6.0)) < 1e-12, "0.7 maps to 0.3lo+0.7hi");
  expect(std::abs(tp.to_raw(0.0) - 2.0) < 1e-12, "0 maps to low");
  expect(std::abs(tp.to_raw(1.0) - 6.0) < 1e-12, "1 maps to high");
  expect(std::abs(tp.to_norm(tp.to_raw(0.42)) - 0.42) < 1e-12, "to_norm inverts to_raw");
  // Values outside the landmarks stay meaningful; nothing clamps by default.
  expect(tp.to_raw(-0.5) < 2.0 && tp.to_raw(1.5) > 6.0, "extrapolates past the landmarks");

  expect(!mscoupon::TwoPoint{5.0, 5.0}.valid(), "degenerate pair is rejected");
  expect(!mscoupon::TwoPoint{6.0, 2.0}.valid(), "inverted pair is rejected");
}

void test_normalize_apply_is_affine() {
  mscoupon::Image2D image = make_mixture_image(64, 64, 909);
  const std::vector<float> raw = image.pixels;

  const mscoupon::TwoPoint tp{kMu1, kMu2};
  mscoupon::apply_two_point(image, tp, /*clamp=*/false);

  expect(image.pixels.size() == raw.size(), "apply is in place, size preserved");
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const double want = (raw[i] - tp.low) / tp.scale();
    expect(std::abs(image.pixels[i] - want) < 1e-4, "every pixel is mapped affinely");
  }

  // A degenerate pair must be a no-op rather than producing inf/nan.
  mscoupon::Image2D untouched = make_mixture_image(8, 8, 910);
  const std::vector<float> before = untouched.pixels;
  mscoupon::apply_two_point(untouched, mscoupon::TwoPoint{1.0, 1.0}, false);
  expect(untouched.pixels == before, "degenerate normalizer leaves pixels alone");
}

// The property that lets normalization be a filter instead of a per-threshold
// transform: statistics computed on the normalized channel are exactly the
// normalized statistics. Note mean is affine but std is SCALE-ONLY -- the
// offset cancels -- which is why no field-kind table is needed anywhere.
void test_normalize_statistics_transform() {
  const mscoupon::Image2D raw = make_mixture_image(96, 96, 911);
  const mscoupon::TwoPoint tp{kMu1, kMu2};

  mscoupon::Image2D norm = raw;
  mscoupon::apply_two_point(norm, tp, false);

  const auto moments = [](const mscoupon::Image2D& im) {
    double sum = 0.0;
    for (const float v : im.pixels) sum += v;
    const double mean = sum / static_cast<double>(im.pixels.size());
    double acc = 0.0;
    for (const float v : im.pixels) acc += (v - mean) * (v - mean);
    return std::pair<double, double>{mean, std::sqrt(acc / static_cast<double>(im.pixels.size()))};
  };

  const auto [raw_mean, raw_std] = moments(raw);
  const auto [norm_mean, norm_std] = moments(norm);

  expect(std::abs(norm_mean - tp.to_norm(raw_mean)) < 1e-4, "mean transforms affinely");
  expect(std::abs(norm_std - raw_std / tp.scale()) < 1e-4, "std transforms by scale only");
}

// Normalization is order-preserving, so it must not perturb the segmentation.
// This is what makes it safe to insert ahead of the MSC.
void test_normalize_preserves_msc_labels() {
  const int w = 48, h = 48;
  const diffg::Image<float> field = make_wells(w, h);

  msseg::Msc2DParams params;
  params.persistence_percent = 10.0f;
  params.manifold = "ascending";

  diffg::Image<float> scaled(diffg::Dimensions{static_cast<std::size_t>(w),
                                               static_cast<std::size_t>(h), 1});
  const double low = 0.25, high = 0.75;  // an arbitrary affine map
  for (std::size_t i = 0; i < field.size(); ++i) {
    scaled.data()[i] = static_cast<float>((field.data()[i] - low) / (high - low));
  }

  msseg::Msc2DPipeline a;
  a.build(field, field, params);
  a.select_persistence(a.value_range() * 0.10f);

  msseg::Msc2DPipeline b;
  b.build(scaled, scaled, params);
  b.select_persistence(b.value_range() * 0.10f);

  const std::vector<int> la = a.labels();
  const std::vector<int> lb = b.labels();
  expect(la.size() == lb.size(), "same raster size");
  expect(la == lb, "an affine rescale leaves the MSC labelling identical");
}

void test_normalize_from_gmm() {
  const mscoupon::Image2D image = make_mixture_image(128, 128, 912);

  nlohmann::json params;
  params["method"] = "gmm";
  params["preset"] = "two_gaussian";
  const mscoupon::NormalizeConfig cfg = mscoupon::parse_normalize_config(params);
  expect(cfg.low_from == "mu_1" && cfg.high_from == "mu_2", "gmm defaults to the two means");

  const mscoupon::TwoPoint tp = mscoupon::measure_two_point(image, cfg);
  expect(std::abs(tp.low - kMu1) < 0.1, "low landmark is the low component mean");
  expect(std::abs(tp.high - kMu2) < 0.1, "high landmark is the high component mean");
}

void test_normalize_manual_and_fallback() {
  nlohmann::json params;
  params["method"] = "manual";
  params["low"] = 3.0;
  params["high"] = 11.0;
  const mscoupon::TwoPoint tp =
      mscoupon::measure_two_point(make_mixture_image(8, 8, 913),
                                  mscoupon::parse_normalize_config(params));
  expect(tp.low == 3.0 && tp.high == 11.0, "manual landmarks are used verbatim");

  nlohmann::json incomplete;
  incomplete["method"] = "manual";
  incomplete["low"] = 1.0;
  bool threw = false;
  try {
    mscoupon::parse_normalize_config(incomplete);
  } catch (const std::exception&) {
    threw = true;
  }
  expect(threw, "manual without both endpoints is rejected");

  // A constant image cannot yield two populations; the manual pair is the
  // documented fallback rather than a divide by zero.
  mscoupon::Image2D flat;
  flat.width = flat.height = 16;
  flat.pixels.assign(256, 4.0f);
  nlohmann::json with_fallback;
  with_fallback["method"] = "histogram";
  with_fallback["low"] = 0.0;
  with_fallback["high"] = 8.0;
  const mscoupon::TwoPoint fb =
      mscoupon::measure_two_point(flat, mscoupon::parse_normalize_config(with_fallback));
  expect(fb.low == 0.0 && fb.high == 8.0, "degenerate measure falls back to the manual pair");
}

// The normalize op runs inside the chain; every other op still goes to core.
void test_normalize_filter_op_in_chain() {
  const mscoupon::Image2D raw = make_mixture_image(64, 64, 914);

  mscoupon::FilterConfig stage;
  stage.operation = "normalize";
  stage.params = nlohmann::json{{"method", "manual"}, {"low", kMu1}, {"high", kMu2}};

  std::vector<mscoupon::TwoPoint> measured;
  const mscoupon::Image2D out = mscoupon::apply_filter_chain(raw, {stage}, &measured);

  expect(measured.size() == 1, "the chain reports the stage's landmarks");
  expect(measured[0].low == kMu1 && measured[0].high == kMu2, "reported landmarks match");
  const mscoupon::TwoPoint tp{kMu1, kMu2};
  for (std::size_t i = 0; i < raw.pixels.size(); ++i) {
    expect(std::abs(out.pixels[i] - tp.to_norm(raw.pixels[i])) < 1e-4, "chain applied the map");
  }

  // An empty chain is still a plain copy.
  const mscoupon::Image2D copy = mscoupon::apply_filter_chain(raw, {});
  expect(copy.pixels == raw.pixels, "empty chain copies through unchanged");
}

void test_histogram_finds_two_peaks() {
  mscoupon::Image2D image = make_mixture_image(400, 400, 915);

  mscoupon::HistogramOptions opts;
  opts.bins = 512;
  const mscoupon::HistogramResult r = mscoupon::measure_histogram(image, opts);

  expect(r.peak_low < r.peak_high, "peaks are ordered by intensity, not height");
  expect(std::abs(r.peak_low - kMu1) < 0.5, "low peak lands on the low population");
  expect(std::abs(r.peak_high - kMu2) < 0.5, "high peak lands on the high population");
  expect(r.hist_lo < r.hist_hi, "histogram support is non-empty");
  expect(r.percentiles.size() == mscoupon::default_percentiles().size(), "full percentile ladder");
  expect(r.n_valid == static_cast<std::int64_t>(image.pixels.size()), "no zeros to drop here");
}

void test_histogram_omit_zeros() {
  // Half the raster is no-data zeros; with omit_zeros they must not become a
  // third population that outvotes the real ones.
  mscoupon::Image2D image;
  image.width = 400;
  image.height = 400;
  image.pixels.assign(static_cast<std::size_t>(400) * 400, 0.0f);
  const std::vector<float> data = make_mixture(image.pixels.size() / 2, 916);
  for (std::size_t i = 0; i < data.size(); ++i) image.pixels[2 * i + 1] = data[i];

  mscoupon::HistogramOptions opts;
  opts.bins = 512;
  const mscoupon::HistogramResult dropped = mscoupon::measure_histogram(image, opts);
  expect(dropped.n_zero == static_cast<std::int64_t>(data.size()), "zeros are counted");
  expect(dropped.n_valid == static_cast<std::int64_t>(data.size()), "zeros are excluded");
  expect(std::abs(dropped.peak_low - kMu1) < 0.5, "low peak survives the zeros");
  expect(std::abs(dropped.peak_high - kMu2) < 0.5, "high peak survives the zeros");

  opts.omit_value.reset();
  const mscoupon::HistogramResult kept = mscoupon::measure_histogram(image, opts);
  expect(kept.n_valid == static_cast<std::int64_t>(image.pixels.size()), "zeros retained on request");
}

void test_region_measure() {
  // Left half at 1.0, right half at 5.0, so each ROI has a known mean.
  mscoupon::Image2D image;
  image.width = 100;
  image.height = 40;
  image.pixels.resize(static_cast<std::size_t>(100) * 40);
  for (int y = 0; y < 40; ++y) {
    for (int x = 0; x < 100; ++x) {
      image.pixels[static_cast<std::size_t>(y) * 100 + x] = x < 50 ? 1.0f : 5.0f;
    }
  }

  const mscoupon::RegionOptions opts;  // zeros retained by default
  const mscoupon::RegionStats left =
      mscoupon::measure_region(image, mscoupon::Rect{0, 40, 0, 50}, opts);
  const mscoupon::RegionStats right =
      mscoupon::measure_region(image, mscoupon::Rect{0, 40, 50, 100}, opts);

  expect(left.n_pixels == 2000 && right.n_pixels == 2000, "rect areas are rows x cols");
  expect(std::abs(left.mean - 1.0) < 1e-6, "left ROI mean");
  expect(std::abs(right.mean - 5.0) < 1e-6, "right ROI mean");
  expect(left.std_dev < 1e-6, "a constant ROI has zero spread");
  expect(std::abs(left.percentiles[5] - 1.0) < 1e-6, "median of a constant ROI");

  // rows are Y and cols are X -- transposing the rect must not be silently accepted.
  bool threw = false;
  try {
    mscoupon::measure_region(image, mscoupon::Rect{0, 100, 0, 40}, opts);
  } catch (const std::exception&) {
    threw = true;
  }
  expect(threw, "a rect taller than the image is rejected");

  const mscoupon::Rect parsed = mscoupon::parse_rect("250:350", "740:840");
  expect(parsed.row0 == 250 && parsed.row1 == 350, "parsed rows");
  expect(parsed.col0 == 740 && parsed.col1 == 840, "parsed cols");
}

// The region measure keeps zeros by default; the other two drop them. This
// asymmetry is deliberate (hand-picked physical ROIs) and easy to regress.
void test_region_zero_policy() {
  mscoupon::Image2D image;
  image.width = image.height = 10;
  image.pixels.assign(100, 0.0f);
  for (std::size_t i = 0; i < 50; ++i) image.pixels[i] = 4.0f;

  mscoupon::RegionOptions opts;
  const mscoupon::RegionStats with_zeros =
      mscoupon::measure_region(image, mscoupon::Rect{0, 10, 0, 10}, opts);
  expect(with_zeros.n_pixels == 100, "zeros are INCLUDED by default for regions");
  expect(std::abs(with_zeros.mean - 2.0) < 1e-6, "mean over all 100 pixels");

  opts.omit_value = 0.0;
  const mscoupon::RegionStats without =
      mscoupon::measure_region(image, mscoupon::Rect{0, 10, 0, 10}, opts);
  expect(without.n_pixels == 50, "zeros dropped on request");
  expect(std::abs(without.mean - 4.0) < 1e-6, "mean over the nonzero half");
}

// Region options are read from the top level, not from inside the "regions"
// rect map. Getting this wrong silently ignored omit_zeros in the C++ while the
// Python honoured it -- the two paths must agree.
void test_region_options_parse_from_top_level() {
  const nlohmann::json cfg = nlohmann::json::parse(R"({
    "omit_zeros": true,
    "regions": {"air": {"rows": "0:10", "cols": "0:10"}}
  })");
  expect(mscoupon::parse_region_options(cfg).omit_value.has_value(),
         "the no-data policy is read at the top level");

  const nlohmann::json defaulted = nlohmann::json::parse(R"({"regions": {}})");
  expect(!mscoupon::parse_region_options(defaulted).omit_value.has_value(),
         "regions keep zeros by default");
}

void test_base_filters_config() {
  const auto dir = std::filesystem::temp_directory_path() / "mscoupon_base_filters_cfg";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);

  const auto load = [&](const std::string& body) {
    const auto cfg_path = dir / "cfg.json";
    {
      std::ofstream f(cfg_path);
      f << R"({"input":{"folder":")" << (dir / "in").generic_string() << R"("},)"
        << R"("output":{"folder":")" << (dir / "out").generic_string() << R"("},)"
        << R"("msc":{"persistence_percent":10.0})" << body << "}";
    }
    std::string cfg_str = cfg_path.string();
    std::vector<char> path_buf(cfg_str.begin(), cfg_str.end());
    path_buf.push_back('\0');
    char arg0[] = "mscoupon";
    char arg_config[] = "--config";
    char* argv[] = {arg0, arg_config, path_buf.data()};
    return mscoupon::load_config(mscoupon::parse_cli(3, argv));
  };

  // base_filters is parsed independently of filters.
  const mscoupon::AppConfig cfg =
      load(R"(,"filters":[{"operation":"blur","params":{"sigma":1.0}}])"
           R"(,"base_filters":[{"operation":"normalize","params":{"method":"gmm"}}])");
  expect(cfg.filters.size() == 1, "topology chain parsed");
  expect(cfg.filters[0].operation == "blur", "topology op parsed");
  expect(cfg.base_filters.size() == 1, "base chain parsed");
  expect(cfg.base_filters[0].operation == "normalize", "base op parsed");
  expect(cfg.base_filters[0].params.at("method") == "gmm", "base op params parsed");

  // Absent base_filters is the pre-existing behaviour: the base stays raw.
  expect(load("").base_filters.empty(), "no base_filters means the base channel stays raw");

  std::filesystem::remove_all(dir);
}

}  // namespace

int main() try {
  test_stats_bbox();
  test_sequence_stride();
  test_matcher_links_overlap();
  test_matcher_merge_unifies();
  test_matcher_first_seen_ids();
  test_matcher_relevance_range();
  test_global_csv_includes_relevance();
  test_cc_stage_trim_and_split();
  test_matcher_carries_extremal_tuple();
  test_matcher_extremum_follows_manifold_direction();
  test_config_matching_flag();
  test_relevance_config();
  test_msc2d_pipeline_monotone_and_stats();
  test_msc2d_pipeline_consistency();
  test_msc2d_pipeline_descending();
  test_msc2d_extremum_stats();
  test_msc2d_extremum_sample_radius();
  test_feature_query();
  test_derived_stat_channels();
  test_stat_channel_bank_matches_single_filters();
  test_gmm_default_preset_survives_low_intensities();
  test_gmm_recovers_two_gaussians();
  test_gmm_omit_zeros();
  test_no_data_sentinel_is_configurable();
  test_gmm_downsample();
  test_gmm_quantile_init();
  test_gmm_hard_stats();
  test_gmm_mode_unimodal();
  test_gmm_trim();
  test_gmm_rejects_degenerate_input();
  test_gmm_integer_pixels();
  test_gmm_image_overload();
  test_two_point_round_trip();
  test_normalize_apply_is_affine();
  test_normalize_statistics_transform();
  test_normalize_preserves_msc_labels();
  test_normalize_from_gmm();
  test_normalize_manual_and_fallback();
  test_normalize_filter_op_in_chain();
  test_histogram_finds_two_peaks();
  test_histogram_omit_zeros();
  test_region_measure();
  test_region_zero_policy();
  test_region_options_parse_from_top_level();
  test_base_filters_config();
  std::cout << "mscoupon tests passed\n";
  return 0;
} catch (const std::exception& e) {
  // Without this, a failed expect() unwinds out of main and MSVC's release CRT
  // fastfails (0xC0000409) with no message at all.
  std::cerr << "mscoupon tests FAILED: " << e.what() << "\n";
  return 1;
}
