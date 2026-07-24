// Python bindings for the cellseg instance.
//
// Exposes the two-phase 3D cell-segmentation pipeline: a once-per-input heavy
// lift returns a primed CellPipeline object; Phase-B threshold tuning
// (persistence, merge-tree cut, background intensity) is then re-runnable
// cheaply without repeating the heavy lift.
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <nlohmann/json.hpp>

#include "cellseg/cell_pipeline.hpp"
#include "cellseg/config.hpp"
#include "cellseg/heavy_lift.hpp"
#include "cellseg/segment.hpp"
#include "msseg/volume/types.hpp"

namespace py = pybind11;
using cellseg::CellPipeline;

namespace {

msseg::Volume array_to_volume(const py::array_t<float, py::array::c_style | py::array::forcecast>& arr) {
  const auto info = arr.request();
  std::size_t d = 1, h = 0, w = 0;
  if (info.ndim == 3) {
    d = static_cast<std::size_t>(info.shape[0]);
    h = static_cast<std::size_t>(info.shape[1]);
    w = static_cast<std::size_t>(info.shape[2]);
  } else if (info.ndim == 2) {
    h = static_cast<std::size_t>(info.shape[0]);
    w = static_cast<std::size_t>(info.shape[1]);
  } else {
    throw std::runtime_error("cellseg: expected a 2D (h,w) or 3D (d,h,w) float32 array");
  }
  msseg::Volume vol(diffg::Dimensions{w, h, d});
  std::memcpy(vol.data(), info.ptr, vol.size() * sizeof(float));
  return vol;
}

cellseg::HeavyLiftConfig heavy_config(const std::string& params_json) {
  nlohmann::json cfg = params_json.empty() ? nlohmann::json::object() : nlohmann::json::parse(params_json);
  // Accept either a full app config ({"heavy_lift": {...}}) or a flat heavy dict.
  if (!cfg.contains("heavy_lift")) cfg = nlohmann::json{{"heavy_lift", cfg}};
  return cellseg::config_from_json(cfg).heavy;
}

py::tuple segment_result_to_numpy(const cellseg::SegmentResult& res) {
  const auto d = res.seg8.dims();
  const std::array<py::ssize_t, 3> shape{static_cast<py::ssize_t>(d.depth),
                                         static_cast<py::ssize_t>(d.height),
                                         static_cast<py::ssize_t>(d.width)};
  py::array_t<std::uint8_t> seg8(shape);
  auto* sp = static_cast<std::uint8_t*>(seg8.request().ptr);
  const std::int32_t* ss = res.seg8.data();
  for (std::size_t i = 0; i < res.seg8.size(); ++i) sp[i] = static_cast<std::uint8_t>(ss[i] & 0xFF);

  py::array_t<std::int32_t> ids(shape);
  std::memcpy(ids.request().ptr, res.ids.data(), res.ids.size() * sizeof(std::int32_t));
  return py::make_tuple(seg8, ids);
}

}  // namespace

PYBIND11_MODULE(cellseg_py, m) {
  m.doc() = "cellseg instance: two-phase 3D fluorescent-membrane cell segmentation.";
  m.def("version", []() { return "0.1.0"; }, "Module version tag.");

  py::class_<CellPipeline>(m, "CellPipeline",
                           "Primed Morse-Smale complex + cached decompositions; run Phase-B "
                           "threshold tuning (persistence / cut / background) repeatedly.")
      .def("value_range", &CellPipeline::value_range)
      .def("heavy_persistence", &CellPipeline::heavy_persistence)
      .def("current_persistence", &CellPipeline::current_persistence)
      .def("filtered",
           [](CellPipeline& self) {
             const msseg::Volume& f = self.filtered();
             const auto d = f.dims();
             py::array_t<float> out({static_cast<py::ssize_t>(d.depth),
                                     static_cast<py::ssize_t>(d.height),
                                     static_cast<py::ssize_t>(d.width)});
             std::memcpy(out.request().ptr, f.data(), f.size() * sizeof(float));
             return out;
           },
           "The filtered (blurred) volume topology was computed on, as float32 (d,h,w).")
      .def("set_persistence",
           [](CellPipeline& self, float p) {
             py::gil_scoped_release rel;
             self.select_persistence(p);
           },
           py::arg("persistence"),
           "Re-select persistence: re-snapshot, recompute ascending labels + merge tree.")
      .def("merge_tree_json",
           [](CellPipeline& self) {
             py::gil_scoped_release rel;
             return self.merge_tree_json();
           },
           "Nested JSON of the minima->1-saddle merge tree.")
      .def("segment",
           [](CellPipeline& self, float cut_threshold, float background_threshold) {
             cellseg::SegmentResult res = [&] {
               py::gil_scoped_release rel;
               return self.segment(cut_threshold, background_threshold);
             }();
             return segment_result_to_numpy(res);
           },
           py::arg("cut_threshold"), py::arg("background_threshold"),
           "Run Phase-B segmentation; returns (seg8 uint8 (d,h,w), ids int32 (d,h,w)).");

  m.def("heavy_lift",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& volume,
           const std::string& params_json) {
          const cellseg::HeavyLiftConfig hc = heavy_config(params_json);
          msseg::Volume vol = array_to_volume(volume);
          cellseg::CellState st = [&] {
            py::gil_scoped_release rel;
            return cellseg::run_heavy_lift(std::move(vol), hc);
          }();
          return CellPipeline(std::move(st));
        },
        py::arg("volume"), py::arg("params_json") = std::string(),
        "Phase A from an in-memory float32 (d,h,w) volume; returns a primed CellPipeline.");

  m.def("heavy_lift_file",
        [](const std::string& path, const std::string& params_json) {
          cellseg::HeavyLiftConfig hc = heavy_config(params_json);
          hc.input_path = path;
          cellseg::CellState st = [&] {
            py::gil_scoped_release rel;
            return cellseg::run_heavy_lift(hc);
          }();
          return CellPipeline(std::move(st));
        },
        py::arg("path"), py::arg("params_json") = std::string(),
        "Phase A from a '<name>_XxYxZ.raw' file; returns a primed CellPipeline.");
}
