#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "mscoupon/config.hpp"
#include "mscoupon/sequence.hpp"
#include "mscoupon/stats.hpp"

namespace {

void expect(bool cond, const char* message) {
  if (!cond) throw std::runtime_error(message);
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

}  // namespace

int main() {
  test_stats_bbox();
  test_sequence_stride();
  std::cout << "mscoupon tests passed\n";
  return 0;
}
