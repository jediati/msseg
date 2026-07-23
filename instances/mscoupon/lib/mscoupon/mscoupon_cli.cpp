#include "mscoupon/config.hpp"
#include "mscoupon/pipeline.hpp"
#include "mscoupon/sequence.hpp"

#include <chrono>
#include <iostream>

namespace mscoupon {

int run_cli(int argc, char** argv) {
  const auto process_start = std::chrono::steady_clock::now();
  const CliOptions cli = parse_cli(argc, argv);
  const AppConfig cfg = load_config(cli);

  const auto jobs = build_sequence(cfg);
  std::cout << "Matched " << jobs.size() << " slices.\n";

  if (cfg.dry_run) {
    for (const auto& job : jobs) {
      std::cout << job.input_path.string() << " -> " << job.mask_output_path.string() << "\n";
      if (cfg.debug_output.write_filter_tiff) {
        std::cout << "   filter: " << job.filter_output_path.string() << "\n";
      }
      if (cfg.debug_output.write_label_tiff) {
        std::cout << "   labels: " << job.label_output_path.string() << "\n";
      }
    }
    return 0;
  }

  const auto outputs = run_pipeline(cfg, jobs, process_start);
  write_timing_report(cfg, outputs);
  std::cout << "Processed " << outputs.size() << " slices.\n";
  return 0;
}

}  // namespace mscoupon
