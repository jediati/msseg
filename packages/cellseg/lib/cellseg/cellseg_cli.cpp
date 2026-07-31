#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cellseg/cell_pipeline.hpp"
#include "cellseg/config.hpp"
#include "cellseg/heavy_lift.hpp"
#include "msseg/io/raw_io.hpp"

namespace cellseg {
namespace {

void write_text(const std::filesystem::path& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  if (!out.good()) throw std::runtime_error("cannot write " + path.string());
  out << text;
}

// Write an int32 LabelVolume down-cast to 8-bit raw + a "<path>.dat" sidecar
// ("width height depth"), matching msseg's raw conventions.
void write_raw_u8(const std::filesystem::path& path, const msseg::LabelVolume& vol) {
  const auto d = vol.dims();
  std::vector<std::uint8_t> bytes(vol.size());
  const std::int32_t* src = vol.data();
  for (std::size_t i = 0; i < vol.size(); ++i)
    bytes[i] = static_cast<std::uint8_t>(src[i] & 0xFF);
  std::ofstream out(path, std::ios::binary);
  if (!out.good()) throw std::runtime_error("cannot write " + path.string());
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  std::ofstream side(path.string() + ".dat");
  side << d.width << " " << d.height << " " << d.depth << "\n";
}

}  // namespace

int run_cli(int argc, char** argv) {
  const CliOptions cli = parse_cli(argc, argv);
  const AppConfig cfg = load_config(cli);

  if (cfg.dry_run) {
    std::cout << "cellseg dry-run\n"
              << "  input:        " << cfg.heavy.input_path.string() << "\n"
              << "  blur_sigma:   " << cfg.heavy.blur_sigma << "\n"
              << "  persistence:  "
              << (cfg.heavy.persistence_absolute
                      ? std::to_string(*cfg.heavy.persistence_absolute) + " (abs)"
                      : std::to_string(cfg.heavy.persistence_percent.value_or(5.0f)) + "%")
              << "\n"
              << "  cut_threshold:        " << cfg.phase_b.cut_threshold << "\n"
              << "  background_threshold: " << cfg.phase_b.background_threshold << "\n"
              << "  -> seg8:  " << cfg.output.seg8_path.string() << "\n"
              << "  -> ids:   " << cfg.output.ids_path.string() << "\n"
              << "  -> tree:  " << cfg.output.merge_tree_path.string() << "\n";
    return 0;
  }

  std::cout << "cellseg: heavy lift on " << cfg.heavy.input_path.string() << " ...\n";
  CellPipeline pipe(run_heavy_lift(cfg.heavy));

  const float p = cfg.phase_b.persistence_selection.value_or(pipe.heavy_persistence());
  pipe.select_persistence(p);
  std::cout << "  value range: " << pipe.value_range()
            << ", persistence: " << pipe.current_persistence() << "\n";

  write_text(cfg.output.merge_tree_path, pipe.merge_tree_json());

  const SegmentResult res = pipe.segment(cfg.phase_b.cut_threshold, cfg.phase_b.background_threshold);
  write_raw_u8(cfg.output.seg8_path, res.seg8);
  msseg::write_raw_labels(cfg.output.ids_path, res.ids);

  std::cout << "  wrote " << cfg.output.seg8_path.string() << ", "
            << cfg.output.ids_path.string() << ", " << cfg.output.merge_tree_path.string() << "\n";
  return 0;
}

}  // namespace cellseg
