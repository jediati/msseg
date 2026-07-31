// Python bindings for the mscoupon instance (M4).
//
// Exposes the per-slice 2D segmentation the batch pipeline performs: apply a
// diffg filter, then compute the 2D Morse-Smale ascending/descending
// 2-manifold labels. Input is a float32 (h,w) array; output is int32 (h,w).
//
// `filter_slice` exposes the filter stage on its own so callers can also obtain
// the transformed "topo" field (e.g. an edge/gradient-magnitude image) that the
// MSC is computed over, which `segment_slice` does not return.
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <nlohmann/json.hpp>

#include "diffg/image.hpp"
#include "msseg/compute/msc2d.hpp"
#include "msseg/filter/filter_stage.hpp"

namespace py = pybind11;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

// Copy a 2D (h,w) float32 numpy array into a diffg::Image<float>.
diffg::Image<float> to_image(const FloatArray& image, std::size_t& h, std::size_t& w) {
  const auto info = image.request();
  if (info.ndim != 2) throw std::runtime_error("expected a 2D (h,w) float32 array");
  h = static_cast<std::size_t>(info.shape[0]);
  w = static_cast<std::size_t>(info.shape[1]);
  diffg::Image<float> slice(diffg::Dimensions{w, h, 1});
  std::memcpy(slice.data(), info.ptr, slice.size() * sizeof(float));
  return slice;
}

// Parse the optional "filter" block of a params-JSON string into FilterParams.
msseg::FilterParams parse_filter(const nlohmann::json& cfg) {
  msseg::FilterParams filter;
  if (cfg.contains("filter")) {
    filter.operation = cfg["filter"].value("operation", filter.operation);
    if (cfg["filter"].contains("params")) filter.params = cfg["filter"]["params"];
  }
  return filter;
}

// Parse the optional "msc" block of a params-JSON string into Msc2DParams.
msseg::Msc2DParams parse_msc(const nlohmann::json& cfg) {
  msseg::Msc2DParams msc;
  if (cfg.contains("msc")) {
    const auto& m = cfg["msc"];
    if (m.contains("persistence_absolute")) msc.persistence_absolute = m["persistence_absolute"].get<float>();
    if (m.contains("persistence_percent")) msc.persistence_percent = m["persistence_percent"].get<float>();
    msc.compute_algorithm = m.value("compute_algorithm", msc.compute_algorithm);
    msc.accurate_ascending = m.value("accurate_ascending", msc.accurate_ascending);
    msc.accurate_descending = m.value("accurate_descending", msc.accurate_descending);
    msc.manifold = m.value("manifold", msc.manifold);
    msc.requested_parallelism = m.value("requested_parallelism", msc.requested_parallelism);
  }
  return msc;
}

nlohmann::json parse_params(const std::string& params_json) {
  return params_json.empty() ? nlohmann::json::object() : nlohmann::json::parse(params_json);
}

FloatArray filter_slice(const FloatArray& image, const std::string& params_json) {
  std::size_t h = 0, w = 0;
  const diffg::Image<float> slice = to_image(image, h, w);
  const msseg::FilterParams filter = parse_filter(parse_params(params_json));

  diffg::Image<float> filtered;
  {
    py::gil_scoped_release release;
    filtered = msseg::apply_filter(slice, filter);
  }

  FloatArray out({static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(w)});
  std::memcpy(out.request().ptr, filtered.data(), filtered.size() * sizeof(float));
  return out;
}

py::array_t<std::int32_t> segment_slice(const FloatArray& image, const std::string& params_json) {
  std::size_t h = 0, w = 0;
  const diffg::Image<float> slice = to_image(image, h, w);
  const nlohmann::json cfg = parse_params(params_json);
  const msseg::FilterParams filter = parse_filter(cfg);
  const msseg::Msc2DParams msc = parse_msc(cfg);

  std::vector<int> labels;
  {
    py::gil_scoped_release release;
    const diffg::Image<float> filtered = msseg::apply_filter(slice, filter);
    labels = msseg::compute_msc2d_labels(filtered, msc);
  }

  py::array_t<std::int32_t> out({static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(w)});
  std::memcpy(out.request().ptr, labels.data(), labels.size() * sizeof(std::int32_t));
  return out;
}

}  // namespace

PYBIND11_MODULE(mscoupon_py, m) {
  m.doc() = "mscoupon instance: 2D Morse-Smale slice segmentation.";
  m.def("version", []() { return "0.1.0"; }, "Module version tag.");
  m.def("filter_slice", &filter_slice, py::arg("image"), py::arg("params_json") = std::string(),
        "Apply the diffg filter from params_json['filter'] to a float32 (h,w) slice; "
        "returns the transformed float32 (h,w) field (the topo field the MSC runs over).");
  m.def("segment_slice", &segment_slice, py::arg("image"), py::arg("params_json") = std::string(),
        "Filter + 2D MSC segment a float32 (h,w) slice; returns int32 (h,w) manifold labels.");
}
