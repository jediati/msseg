// Python bindings for the mscoupon instance.
//
// Modeled on MSCEER's msc_py (flat m.def functions, py::array in / out). This
// is a skeleton: the batch pipeline is CLI-driven today; the binding surface
// (run a config, process a single slice returning a label array) is fleshed
// out in M4.
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(mscoupon_py, m) {
  m.doc() = "mscoupon instance: 2D Morse-Smale slice segmentation (scaffold).";
  m.def(
      "version", []() { return "0.1.0-scaffold"; }, "Module version tag.");
}
