// msworkflow: generic JSON-described workflow runner over msseg_core stages.
//
// Scaffold (M1). The full runner -- parse a JSON workflow spec, walk the core
// stage registry, run filter -> MSC -> simplify -> segment over a volume --
// lands in M5. For now this validates that the generic frontend links the
// portable core and reports its intent.
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "usage: msworkflow <workflow.json> [input] [output]\n"
              << "  Generic MSSeg workflow runner (implementation lands in M5).\n";
    return argc < 2 ? 1 : 0;
  }
  std::cout << "msworkflow scaffold: workflow='" << argv[1] << "' not yet executed (M5).\n";
  return 0;
}
