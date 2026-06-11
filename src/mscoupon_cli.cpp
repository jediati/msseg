#include "mscoupon/config.hpp"
#include "mscoupon/pipeline.hpp"
#include "mscoupon/sequence.hpp"

#include <iostream>

namespace mscoupon {

int run_cli(int argc, char** argv) {
  const CliOptions cli = parse_cli(argc, argv);
  const AppConfig cfg = load_config(cli);

  const auto jobs = build_sequence(cfg);
  std::cout << "Matched " << jobs.size() << " slices.\n";

  if (cfg.dry_run) {
    for (const auto& job : jobs) {
      std::cout << job.input_path.string() << " -> " << job.mask_output_path.string() << "\n";
    }
    return 0;
  }

  const auto outputs = run_pipeline(cfg, jobs);
  write_timing_report(cfg, outputs);
  std::cout << "Processed " << outputs.size() << " slices.\n";
  return 0;
}

}  // namespace mscoupon
