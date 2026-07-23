// Python bindings for the generic msworkflow runner (scaffold; M5).
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(msworkflow_py, m) {
  m.doc() = "MSSeg generic JSON workflow runner (scaffold).";
  m.def(
      "version", []() { return "0.1.0-scaffold"; }, "Module version tag.");
}
