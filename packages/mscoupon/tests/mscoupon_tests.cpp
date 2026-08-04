#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "diffg/image.hpp"
#include "mscoupon/config.hpp"
#include "mscoupon/matcher.hpp"
#include "mscoupon/query.hpp"
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

// Build a minimal kept-segment stat row (only the fields the matcher consumes).
mscoupon::SegmentStat kept_row(int segment_id, int slice_index, std::size_t area) {
  mscoupon::SegmentStat s;
  s.segment_id = segment_id;
  s.slice_index = slice_index;
  s.area = area;
  s.min_value = 1.0f;
  s.max_value = 1.0f;
  s.mean_value = 1.0f;
  return s;
}

int global_for(const std::vector<mscoupon::FeatureMapRow>& map, int slice_index, int segment_id) {
  for (const auto& r : map) {
    if (r.slice_index == slice_index && r.segment_id == segment_id) return r.global_id;
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
  // Slice 0: feature id1 (px 0-1) and feature id2 (px 3-4); 0 = background.
  m.add_slice({1, 1, 0, 2, 2}, 5, 1, {1, 2}, {kept_row(1, 0, 2), kept_row(2, 0, 2)}, 0);
  // Slice 1: feature id1 (px 0-1) overlaps slice-0 id1; slice-0 id2 has no successor.
  m.add_slice({1, 1, 0, 0, 0}, 5, 1, {1}, {kept_row(1, 1, 2)}, 1);

  std::vector<mscoupon::FeatureMapRow> map;
  std::vector<mscoupon::GlobalFeatureStat> table;
  m.finalize(map, table);

  expect(map.size() == 3, "three kept (slice,id) members");
  expect(global_for(map, 0, 1) == global_for(map, 1, 1), "overlapping feature shares a global id");
  expect(global_for(map, 0, 2) != global_for(map, 0, 1), "non-overlapping feature is a distinct global id");
  expect(table.size() == 2, "two global features");
}

void test_matcher_merge_unifies() {
  mscoupon::SliceMatcher m;
  m.add_slice({1, 1, 0, 2, 2}, 5, 1, {1, 2}, {kept_row(1, 0, 2), kept_row(2, 0, 2)}, 0);
  // A single slice-1 feature bridges both slice-0 features -> true union.
  m.add_slice({1, 1, 1, 1, 1}, 5, 1, {1}, {kept_row(1, 1, 5)}, 1);

  std::vector<mscoupon::FeatureMapRow> map;
  std::vector<mscoupon::GlobalFeatureStat> table;
  m.finalize(map, table);

  expect(table.size() == 1, "merge unifies into one global feature");
  expect(global_for(map, 0, 1) == global_for(map, 0, 2), "both prior features unified");
  expect(global_for(map, 0, 1) == global_for(map, 1, 1), "current feature unified with priors");
  expect(table[0].voxel_count == 9, "voxel count sums all members (2+2+5)");
  expect(table[0].first_slice == 0 && table[0].last_slice == 1, "spans both slices");
  expect(table[0].num_slices == 2, "counts two distinct slices");
}

void test_matcher_global_ids_deterministic() {
  mscoupon::SliceMatcher m;
  // Rows supplied in reversed id order; global ids must follow (slice, id) order.
  m.add_slice({2, 2, 0, 1, 1}, 5, 1, {1, 2}, {kept_row(2, 0, 2), kept_row(1, 0, 2)}, 0);

  std::vector<mscoupon::FeatureMapRow> map;
  std::vector<mscoupon::GlobalFeatureStat> table;
  m.finalize(map, table);
  expect(global_for(map, 0, 1) == 0, "smallest id appears first -> global 0");
  expect(global_for(map, 0, 2) == 1, "next id -> global 1");
}

void test_matcher_global_table_sorted() {
  mscoupon::SliceMatcher m;
  m.add_slice({1, 0, 2, 2, 2}, 5, 1, {1, 2}, {kept_row(1, 0, 1), kept_row(2, 0, 3)}, 0);

  std::vector<mscoupon::FeatureMapRow> map;
  std::vector<mscoupon::GlobalFeatureStat> table;
  m.finalize(map, table);
  expect(table.size() == 2, "two features");
  expect(table[0].voxel_count >= table[1].voxel_count, "master table sorted by voxel_count descending");
  expect(table[0].global_id == 1, "largest feature (id2, area 3) is global id 1");
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

void test_msc2d_pipeline_monotone_and_stats() {
  const int w = 48, h = 48;
  const diffg::Image<float> field = make_wells(w, h);

  msseg::Msc2DParams cfg;
  cfg.manifold = "ascending";
  cfg.persistence_absolute = 0.0f;
  cfg.persistence_percent.reset();

  msseg::Msc2DPipeline pipe;
  pipe.build(field, field, cfg);  // base == filtered here (topology on the raw field)
  expect(pipe.width() == w && pipe.height() == h, "pipeline reports image dims");
  expect(pipe.value_range() > 0.0f, "value range is positive");

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
  for (const auto& f : feats) {
    feat_area_sum += f.area;
    const auto it = area_of.find(static_cast<int>(f.feature_id));
    expect(it != area_of.end() && it->second == f.area,
           "feature area matches its label pixel count");
    expect(f.base_min <= f.base_max, "feature base min<=max");
    expect(f.min_x <= f.max_x && f.min_y <= f.max_y, "feature bbox ordering");
  }
  expect(feat_area_sum == labeled, "aggregated feature areas sum to labeled pixels");
  expect(static_cast<int>(feats.size()) == distinct_features(pipe.labels()),
         "one stat row per surviving feature");
}

void test_msc2d_pipeline_consistency() {
  // The pipeline is authoritative for segmentation. Two invariants:
  //  (1) at persistence 0 the finest decomposition equals a fresh native
  //      compute_msc2d_labels (the base labeling is shared);
  //  (2) re-thresholding is idempotent (same persistence -> same labels).
  // The merge-tree simplification intentionally differs from GInt's native
  // cancellation ABOVE persistence 0 (branch decomposition vs pairwise cancel),
  // so we do not compare partitions there.
  const int w = 48, h = 48;
  const diffg::Image<float> field = make_wells(w, h);

  msseg::Msc2DParams cfg;
  cfg.manifold = "ascending";
  cfg.persistence_percent.reset();

  msseg::Msc2DPipeline pipe;
  msseg::Msc2DParams build_cfg = cfg;
  build_cfg.persistence_absolute = 0.0f;
  pipe.build(field, field, build_cfg);

  pipe.select_persistence(0.0f);
  msseg::Msc2DParams native = cfg;
  native.persistence_absolute = 0.0f;
  const std::vector<int> native_labels = msseg::compute_msc2d_labels(field, native);
  expect(same_partition(pipe.labels(), native_labels),
         "finest (p=0) decomposition matches native compute_msc2d_labels partition");

  const float p = 0.3f * pipe.value_range();
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

void test_feature_query() {
  // A synthetic feature: area 100, base mean 5, filtered mean 2, bbox 10x20.
  msseg::Msc2DFeatureStat s;
  s.feature_id = 3;
  s.area = 100;
  s.base_sum = 500.0;  s.base_sumsq = 2600.0;  s.base_min = 1.0f;  s.base_max = 9.0f;
  s.filt_sum = 200.0;  s.filt_sumsq = 500.0;   s.filt_min = 0.0f;  s.filt_max = 4.0f;
  s.min_x = 5; s.max_x = 14; s.min_y = 0; s.max_y = 19;
  const auto row = mscoupon::feature_row(s);

  expect(std::abs(row.at("mean_base") - 5.0) < 1e-9, "mean_base = base_sum/area");
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
}

}  // namespace

int main() {
  test_stats_bbox();
  test_sequence_stride();
  test_matcher_links_overlap();
  test_matcher_merge_unifies();
  test_matcher_global_ids_deterministic();
  test_matcher_global_table_sorted();
  test_config_matching_flag();
  test_msc2d_pipeline_monotone_and_stats();
  test_msc2d_pipeline_consistency();
  test_msc2d_pipeline_descending();
  test_feature_query();
  std::cout << "mscoupon tests passed\n";
  return 0;
}
