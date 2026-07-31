#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "mscoupon/config.hpp"
#include "mscoupon/matcher.hpp"
#include "mscoupon/sequence.hpp"
#include "mscoupon/stats.hpp"

namespace {

void expect(bool cond, const char* message) {
  if (!cond) throw std::runtime_error(message);
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

}  // namespace

int main() {
  test_stats_bbox();
  test_sequence_stride();
  test_matcher_links_overlap();
  test_matcher_merge_unifies();
  test_matcher_global_ids_deterministic();
  test_matcher_global_table_sorted();
  test_config_matching_flag();
  std::cout << "mscoupon tests passed\n";
  return 0;
}
