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
#include <unordered_map>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>

#include "diffg/image.hpp"
#include "mscoupon/config.hpp"
#include "mscoupon/query.hpp"
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

// Parse a "filters" array (or a singular "filter") into an ordered chain.
std::vector<msseg::FilterParams> parse_filter_chain(const nlohmann::json& cfg) {
  std::vector<msseg::FilterParams> chain;
  if (cfg.contains("filters") && cfg["filters"].is_array()) {
    for (const auto& f : cfg["filters"]) {
      msseg::FilterParams fp;
      fp.operation = f.value("operation", fp.operation);
      if (f.contains("params")) fp.params = f["params"];
      chain.push_back(std::move(fp));
    }
  } else if (cfg.contains("filter")) {
    chain.push_back(parse_filter(cfg));
  }
  return chain;
}

// Parse a "feature_filters" JSON array (or a bare array) into query predicates.
std::vector<mscoupon::FeatureQuery> parse_feature_queries(const std::string& queries_json) {
  std::vector<mscoupon::FeatureQuery> out;
  if (queries_json.empty()) return out;
  nlohmann::json j = nlohmann::json::parse(queries_json);
  if (j.is_object() && j.contains("feature_filters")) j = j["feature_filters"];
  if (!j.is_array()) return out;
  for (const auto& q : j) {
    mscoupon::FeatureQuery fq;
    fq.field = q.value("field", std::string());
    fq.op = q.value("op", std::string("gt"));
    fq.value = q.value("value", 0.0);
    fq.value2 = q.value("value2", 0.0);
    out.push_back(std::move(fq));
  }
  return out;
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

// Apply an ordered filter chain (params_json['filters'] array, or a single
// 'filter'); returns the float32 (h,w) field the MSC would run over.
FloatArray filter_chain(const FloatArray& image, const std::string& params_json) {
  std::size_t h = 0, w = 0;
  const diffg::Image<float> slice = to_image(image, h, w);
  const std::vector<msseg::FilterParams> chain = parse_filter_chain(parse_params(params_json));

  diffg::Image<float> filtered;
  {
    py::gil_scoped_release release;
    filtered = msseg::apply_filter_chain(slice, chain);
  }
  FloatArray out({static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(w)});
  std::memcpy(out.request().ptr, filtered.data(), filtered.size() * sizeof(float));
  return out;
}

// Build a primed Msc2DPipeline over `base` (original image) + `filtered` (the
// topology field, already filter-chained). Both are float32 (h,w).
msseg::Msc2DPipeline prime_slice(const FloatArray& base, const FloatArray& filtered,
                                 const std::string& params_json) {
  std::size_t bh = 0, bw = 0, fh = 0, fw = 0;
  const diffg::Image<float> base_img = to_image(base, bh, bw);
  const diffg::Image<float> filt_img = to_image(filtered, fh, fw);
  if (bh != fh || bw != fw) throw std::runtime_error("base and filtered must share shape");
  const msseg::Msc2DParams msc = parse_msc(parse_params(params_json));

  msseg::Msc2DPipeline pipe;
  {
    py::gil_scoped_release release;
    pipe.build(base_img, filt_img, msc);
  }
  return pipe;
}

// Feature id per pixel (int32 h,w) at the pipeline's current persistence.
py::array_t<std::int32_t> pipeline_labels(const msseg::Msc2DPipeline& pipe) {
  const std::vector<int>& labels = pipe.labels();
  const auto h = static_cast<py::ssize_t>(pipe.height());
  const auto w = static_cast<py::ssize_t>(pipe.width());
  py::array_t<std::int32_t> out({h, w});
  std::memcpy(out.request().ptr, labels.data(), labels.size() * sizeof(std::int32_t));
  return out;
}

// Per-surviving-feature statistics as a list of dicts (derived fields).
py::list pipeline_feature_stats(const msseg::Msc2DPipeline& pipe) {
  py::list rows;
  for (const auto& s : pipe.feature_stats()) {
    py::dict d;
    for (const auto& [k, v] : mscoupon::feature_row(s)) d[py::str(k)] = v;
    rows.append(std::move(d));
  }
  return rows;
}

// Evaluate the feature-query chain against a list of feature rows (dicts). Works
// for both 2D features (pipeline_feature_stats) and Python-assembled 3D features.
std::vector<bool> evaluate_queries(const py::list& rows, const std::string& queries_json) {
  const std::vector<mscoupon::FeatureQuery> queries = parse_feature_queries(queries_json);
  std::vector<bool> keep;
  keep.reserve(rows.size());
  for (const auto& item : rows) {
    std::unordered_map<std::string, double> row;
    for (const auto& kv : item.cast<py::dict>()) {
      row[kv.first.cast<std::string>()] = kv.second.cast<double>();
    }
    keep.push_back(mscoupon::row_passes(row, queries));
  }
  return keep;
}

}  // namespace

PYBIND11_MODULE(mscoupon_py, m) {
  m.doc() = "mscoupon instance: 2D Morse-Smale slice segmentation.";
  m.def("version", []() { return "0.1.0"; }, "Module version tag.");
  m.def("filter_slice", &filter_slice, py::arg("image"), py::arg("params_json") = std::string(),
        "Apply the diffg filter from params_json['filter'] to a float32 (h,w) slice; "
        "returns the transformed float32 (h,w) field (the topo field the MSC runs over).");
  m.def("filter_chain", &filter_chain, py::arg("image"), py::arg("params_json") = std::string(),
        "Apply the ordered filter chain from params_json['filters'] (or a single 'filter') "
        "to a float32 (h,w) slice; returns the float32 (h,w) topology field.");
  m.def("segment_slice", &segment_slice, py::arg("image"), py::arg("params_json") = std::string(),
        "Filter + 2D MSC segment a float32 (h,w) slice; returns int32 (h,w) manifold labels.");

  // Two-phase pipeline: prime once, then re-threshold cheaply with the merge tree.
  py::class_<msseg::Msc2DPipeline>(m, "Msc2DPipeline",
      "Primed 2D MSC pipeline: base decomposition + merge tree + statistics, with "
      "cheap persistence re-thresholding. Construct via prime_slice().")
      .def("select_persistence", &msseg::Msc2DPipeline::select_persistence, py::arg("persistence_absolute"),
           "Re-threshold to an absolute persistence (remap labels + re-aggregate stats).")
      .def("current_persistence", &msseg::Msc2DPipeline::current_persistence)
      .def("value_range", &msseg::Msc2DPipeline::value_range,
           "Filtered-field value range (max-min), for percent->absolute persistence.")
      .def("width", &msseg::Msc2DPipeline::width)
      .def("height", &msseg::Msc2DPipeline::height)
      .def("labels", &pipeline_labels, "Feature id per pixel (int32 h,w) at the current persistence.")
      .def("feature_stats", &pipeline_feature_stats,
           "Per-surviving-feature statistics (list of dicts) at the current persistence.")
      .def("merge_tree_json", &msseg::Msc2DPipeline::merge_tree_json,
           "The merge tree (flat JSON) mirroring the manifold merger.");

  m.def("prime_slice", &prime_slice, py::arg("base"), py::arg("filtered"),
        py::arg("params_json") = std::string(),
        "Build a primed Msc2DPipeline over base (original) + filtered (topology field, "
        "already filter-chained), both float32 (h,w). params_json['msc'] configures it.");
  m.def("evaluate_queries", &evaluate_queries, py::arg("rows"), py::arg("queries_json"),
        "Evaluate a feature-query chain (JSON array of {field,op,value[,value2]}) against a "
        "list of feature-stat dicts; returns a list of bool keep flags. Shared 2D/3D evaluator.");
}
