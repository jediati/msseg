// Python bindings for the generic MSSeg workflow runner (M4).
//
// Exposes run(volume, workflow_json) -> labels, mirroring the msworkflow CLI:
// a float32 volume + a JSON workflow spec in, an int32 label volume out.
// Arrays are C-order with shape (depth, height, width) -- matching diffg's
// row-major (x fastest) layout.
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <nlohmann/json.hpp>

#include "msseg/volume/types.hpp"
#include "msseg/workflow/pipeline.hpp"

namespace py = pybind11;

namespace {

msseg::Volume to_volume(const py::array_t<float, py::array::c_style | py::array::forcecast>& arr) {
  const auto info = arr.request();
  std::size_t w, h, d;
  if (info.ndim == 3) {
    d = static_cast<std::size_t>(info.shape[0]);
    h = static_cast<std::size_t>(info.shape[1]);
    w = static_cast<std::size_t>(info.shape[2]);
  } else if (info.ndim == 2) {
    d = 1;
    h = static_cast<std::size_t>(info.shape[0]);
    w = static_cast<std::size_t>(info.shape[1]);
  } else {
    throw std::runtime_error("expected a 2D (h,w) or 3D (d,h,w) float32 array");
  }
  msseg::Volume v(diffg::Dimensions{w, h, d});
  std::memcpy(v.data(), info.ptr, v.size() * sizeof(float));
  return v;
}

py::array_t<std::int32_t> run(const py::array_t<float, py::array::c_style | py::array::forcecast>& volume,
                              const std::string& workflow_json) {
  const msseg::Volume input = to_volume(volume);
  const msseg::WorkflowParams params = msseg::parse_workflow(nlohmann::json::parse(workflow_json));

  msseg::WorkflowResult result;
  {
    py::gil_scoped_release release;  // heavy C++ work without the GIL
    result = msseg::Pipeline::run(input, params);
  }

  const auto& labels = result.labels;
  py::array_t<std::int32_t> out({static_cast<py::ssize_t>(labels.dims().depth),
                                 static_cast<py::ssize_t>(labels.dims().height),
                                 static_cast<py::ssize_t>(labels.dims().width)});
  std::memcpy(out.request().ptr, labels.data(), labels.size() * sizeof(std::int32_t));
  return out;
}

}  // namespace

PYBIND11_MODULE(msworkflow_py, m) {
  m.doc() = "MSSeg generic JSON workflow runner.";
  m.def("version", []() { return "0.1.0"; }, "Module version tag.");
  m.def("run", &run, py::arg("volume"), py::arg("workflow_json"),
        "Run a JSON-described workflow over a float32 volume (d,h,w) and return an int32 label volume.");
}
