// msworkflow: generic JSON-described workflow runner over msseg_core.
//
//   msworkflow <workflow.json> <input.raw> <output_labels.raw>
//
// Input is a raw float32 volume with a "<input.raw>.dat" dimension sidecar
// (width height depth). Output is a raw int32 label volume + sidecar. The
// workflow JSON schema is documented on msseg::parse_workflow.
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "msseg/io/raw_io.hpp"
#include "msseg/volume/types.hpp"
#include "msseg/workflow/pipeline.hpp"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cout << "usage: msworkflow <workflow.json> <input.raw> <output_labels.raw>\n";
    return 1;
  }
  try {
    std::ifstream spec_file(argv[1]);
    if (!spec_file.good()) throw std::runtime_error(std::string("cannot open workflow: ") + argv[1]);
    nlohmann::json spec;
    spec_file >> spec;

    const msseg::WorkflowParams params = msseg::parse_workflow(spec);
    const msseg::Volume input = msseg::read_raw_volume(argv[2]);
    std::printf("loaded volume %zux%zux%zu\n", input.dims().width, input.dims().height, input.dims().depth);

    const msseg::WorkflowResult result = msseg::Pipeline::run(input, params);

    std::size_t labeled = 0;
    for (std::size_t i = 0; i < result.labels.size(); ++i)
      if (result.labels.data()[i] != msseg::kBackgroundLabel) ++labeled;

    std::printf("MSC nodes=%zu arcs=%zu; segmented voxels=%zu/%zu\n", result.graph.nodes.size(),
                result.graph.arcs.size(), labeled, result.labels.size());

    msseg::write_raw_labels(argv[3], result.labels);
    std::printf("wrote labels -> %s\n", argv[3]);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "msworkflow error: " << ex.what() << "\n";
    return 1;
  }
}
