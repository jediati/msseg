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
#include <filesystem>
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
#include "mscoupon/gmm.hpp"
#include "mscoupon/histogram_peaks.hpp"
#include "mscoupon/measure_config.hpp"
#include "mscoupon/query.hpp"
#include "mscoupon/region_measure.hpp"
#include "msseg/compute/msc2d.hpp"
#include "msseg/filter/chain_plan.hpp"
#include "msseg/filter/color_stage.hpp"
#include "msseg/filter/plane_stages.hpp"
#include "msseg/filter/filter_stage.hpp"
#include "msseg/graph/seam_graph.hpp"
#include "msseg/io/tiff_io.hpp"
#include "msseg/workflow/input_slice.hpp"

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

// Copy a (h,w) or planar (C,h,w) float32 numpy array into diffg planes.
diffg::MultiImage<float> to_planes(const FloatArray& image, std::size_t& h, std::size_t& w) {
  const auto info = image.request();
  if (info.ndim != 2 && info.ndim != 3) {
    throw std::runtime_error("expected a (h,w) or planar (C,h,w) float32 array");
  }
  const std::size_t channels = info.ndim == 3 ? static_cast<std::size_t>(info.shape[0]) : 1;
  if (channels == 0) throw std::runtime_error("expected at least one plane");
  h = static_cast<std::size_t>(info.shape[info.ndim - 2]);
  w = static_cast<std::size_t>(info.shape[info.ndim - 1]);
  diffg::MultiImage<float> planes(diffg::Dimensions{w, h, 1}, channels);
  std::memcpy(planes.data(), info.ptr, planes.size() * sizeof(float));
  return planes;
}

bool is_planar(const FloatArray& image) { return image.request().ndim == 3; }

// Copy a computed buffer out as a fresh (h, w) array.
//
// The extent is the INPUT's, but nothing in the core's signatures forces a result
// to match it: apply_filter_chain returns an Image<float> sized by whatever the
// chain built. Today every stage is scalar and extent-preserving, so the counts
// always agree -- but the day one returns planes (docs/design_filter_types.md)
// the bare memcpy this replaces would have run off the end of `out`, silently.
// Checking makes that a diagnosable error instead of a heap overflow.
template <typename Array, typename T>
Array to_2d_array(const T* src, std::size_t count, std::size_t h, std::size_t w, const char* what) {
  if (count != h * w) {
    throw std::runtime_error(std::string(what) + " produced " + std::to_string(count) +
                             " values, but the (" + std::to_string(h) + ", " + std::to_string(w) +
                             ") output holds " + std::to_string(h * w) +
                             "; a stage changed the raster's extent or plane count, which this "
                             "binding cannot return yet.");
  }
  Array out({static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(w)});
  std::memcpy(out.request().ptr, src, count * sizeof(T));
  return out;
}

// A chain result as an array: (h, w) for one plane, planar (C, h, w) for a
// stack. One plane keeps the old shape, so every existing caller -- and every
// existing test -- sees exactly what it saw before; only a chain that asks for
// planes gets three dimensions back.
FloatArray planes_to_array(const diffg::MultiImage<float>& stack, std::size_t h, std::size_t w) {
  const std::size_t c = stack.channels();
  if (c * h * w != stack.size()) {
    throw std::runtime_error("the filter chain produced " + std::to_string(stack.size()) +
                             " values, which is not " + std::to_string(c) + " x " +
                             std::to_string(h) + " x " + std::to_string(w) + ".");
  }
  FloatArray out = c == 1 ? FloatArray({static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(w)})
                          : FloatArray({static_cast<py::ssize_t>(c), static_cast<py::ssize_t>(h),
                                        static_cast<py::ssize_t>(w)});
  std::memcpy(out.request().ptr, stack.data(), stack.size() * sizeof(float));
  return out;
}

// `input.color.default_method` of a params JSON, else the keyword argument.
std::string default_color_method(const nlohmann::json& cfg, const std::string& fallback) {
  if (cfg.contains("input") && cfg["input"].is_object() && cfg["input"].contains("color") &&
      cfg["input"]["color"].is_object()) {
    return cfg["input"]["color"].value("default_method", fallback);
  }
  return fallback;
}

// `input.color.reduce_at`: where the conversion goes when the chain does not
// reduce on its own. Read from the same block, so a caller that already passes a
// params JSON needs no new argument.
std::string reduce_at_of(const nlohmann::json& cfg) {
  if (cfg.contains("input") && cfg["input"].is_object() && cfg["input"].contains("color") &&
      cfg["input"]["color"].is_object()) {
    return cfg["input"]["color"].value("reduce_at", std::string("front"));
  }
  return "front";
}

// The optional colour planes handed alongside base/filtered: None, or a planar
// (C,h,w) float32 array over the same raster. Returns an empty MultiImage for None.
diffg::MultiImage<float> optional_planes(const py::object& color, std::size_t h, std::size_t w) {
  if (color.is_none()) return {};
  const FloatArray arr = color.cast<FloatArray>();
  if (arr.request().ndim != 3) throw std::runtime_error("color must be a planar (C,h,w) float32 array");
  std::size_t ch = 0, cw = 0;
  diffg::MultiImage<float> planes = to_planes(arr, ch, cw);
  if (ch != h || cw != w) throw std::runtime_error("color planes must share the slice's (h,w)");
  return planes;
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

// Parse the optional "statistics" block. Shares the CLI's parser so the GUI and
// a batch run agree on which fields exist -- the schema is a wire format across
// this boundary, so a divergence here shows up as a field the GUI offers and the
// CLI rejects.
msseg::StatsSpec parse_stats_spec(const nlohmann::json& cfg) {
  mscoupon::StatisticsConfig stats;
  mscoupon::parse_statistics_json(cfg, stats);
  return stats.spec;
}

// Parse the optional "msc" block of a params-JSON string into Msc2DParams.
msseg::Msc2DParams parse_msc(const nlohmann::json& cfg) {
  msseg::Msc2DParams msc;
  if (cfg.contains("msc")) {
    const auto& m = cfg["msc"];
    if (m.contains("persistence_absolute")) msc.persistence_absolute = m["persistence_absolute"].get<float>();
    if (m.contains("persistence_percent")) msc.persistence_percent = m["persistence_percent"].get<float>();
    msc.compute_algorithm = m.value("compute_algorithm", msc.compute_algorithm);
    msc.simplification = m.value("simplification", msc.simplification);
    msc.accurate_ascending = m.value("accurate_ascending", msc.accurate_ascending);
    msc.accurate_descending = m.value("accurate_descending", msc.accurate_descending);
    msc.manifold = m.value("manifold", msc.manifold);
    msc.requested_parallelism = m.value("requested_parallelism", msc.requested_parallelism);
    msc.use_gpu_gradient = m.value("use_gpu_gradient", msc.use_gpu_gradient);
    if (m.contains("use_gpu_stats")) msc.use_gpu_stats = m["use_gpu_stats"].get<bool>();
    msc.extremum_sample_radius = m.value("extremum_sample_radius", msc.extremum_sample_radius);
  }
  msc.stats = parse_stats_spec(cfg);
  // Legacy alias: `msc.extremum_sample_radius` predates the statistics block.
  if (msc.stats.extremum_sample_radius == 0)
    msc.stats.extremum_sample_radius = msc.extremum_sample_radius;
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

FloatArray filter_slice(const FloatArray& image, const std::string& params_json,
                        const std::string& default_color_method_kw) {
  std::size_t h = 0, w = 0;
  const nlohmann::json cfg = parse_params(params_json);
  const msseg::FilterParams filter = parse_filter(cfg);

  // Planes in, or a stage that consumes them: the multi-channel entry (one plane
  // + a one-plane-capable colour method is allowed there too). A plane stage may
  // hand back a stack, so the result carries its own channel count.
  if (is_planar(image) || filter.operation == msseg::kColorOperation ||
      msseg::is_plane_operation(filter.operation)) {
    const diffg::MultiImage<float> planes = to_planes(image, h, w);
    const std::string method = default_color_method(cfg, default_color_method_kw);
    diffg::MultiImage<float> filtered;
    {
      py::gil_scoped_release release;
      filtered = msseg::apply_filter_chain_planes(
          planes.view(), std::vector<msseg::FilterParams>{filter}, method, reduce_at_of(cfg));
    }
    return planes_to_array(filtered, h, w);
  }
  diffg::Image<float> filtered;
  {
    const diffg::Image<float> slice = to_image(image, h, w);
    py::gil_scoped_release release;
    filtered = msseg::apply_filter(slice, filter);
  }
  return to_2d_array<FloatArray>(filtered.data(), filtered.size(), h, w, "the filter stage");
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

  return to_2d_array<py::array_t<std::int32_t>>(labels.data(), labels.size(), h, w,
                                                "the segmentation");
}

// Apply an ordered filter chain (params_json['filters'] array, or a single
// 'filter'); returns the float32 (h,w) field the MSC would run over.
FloatArray filter_chain(const FloatArray& image, const std::string& params_json,
                        const std::string& default_color_method_kw) {
  std::size_t h = 0, w = 0;
  const nlohmann::json cfg = parse_params(params_json);
  const std::vector<msseg::FilterParams> chain = parse_filter_chain(cfg);

  const bool head_takes_planes =
      !chain.empty() && (chain.front().operation == msseg::kColorOperation ||
                         msseg::is_plane_operation(chain.front().operation));
  if (is_planar(image) || head_takes_planes) {
    const diffg::MultiImage<float> planes = to_planes(image, h, w);
    const std::string method = default_color_method(cfg, default_color_method_kw);
    diffg::MultiImage<float> filtered;
    {
      py::gil_scoped_release release;
      filtered = msseg::apply_filter_chain_planes(planes.view(), chain, method,
                                                  reduce_at_of(cfg));
    }
    return planes_to_array(filtered, h, w);
  }
  diffg::Image<float> filtered;
  {
    const diffg::Image<float> slice = to_image(image, h, w);
    py::gil_scoped_release release;
    filtered = msseg::apply_filter_chain(slice, chain);
  }
  return to_2d_array<FloatArray>(filtered.data(), filtered.size(), h, w, "the filter chain");
}

// The chain PLAN: what each stage of `chain_json` receives and yields, in
// planes, with no raster in hand. The leading colour conversion a multi-plane
// input gets when the chain does not start with one is IN the list, marked
// index -1, which is the whole point -- it used to exist only inside the runner.
py::dict chain_plan_py(const std::string& chain_json, int channels,
                       const std::string& default_method, const std::string& reduce_at) {
  nlohmann::json arr = chain_json.empty() ? nlohmann::json::array() : nlohmann::json::parse(chain_json);
  if (!arr.is_array()) throw std::runtime_error("chain_plan: expected a JSON array of stages");
  std::vector<msseg::FilterParams> chain;
  for (const auto& f : arr) {
    msseg::FilterParams fp;
    fp.operation = f.value("operation", fp.operation);
    if (f.contains("params") && f["params"].is_object()) fp.params = f["params"];
    chain.push_back(std::move(fp));
  }
  const msseg::ChainPlan plan =
      msseg::plan_chain(chain, static_cast<std::size_t>(channels), default_method, reduce_at);

  py::list stages;
  for (const msseg::StageRecord& rec : plan.stages) {
    py::dict d;
    d["operation"] = rec.stage.operation;
    d["params"] = py::module_::import("json").attr("loads")(rec.stage.params.dump());
    d["in"] = static_cast<int>(rec.in_channels);
    d["out"] = static_cast<int>(rec.out_channels);
    d["index"] = rec.config_index;
    d["synthesized"] = rec.synthesized();
    d["lifted"] = rec.lifted;
    stages.append(std::move(d));
  }
  py::dict out;
  out["stages"] = std::move(stages);
  out["in"] = static_cast<int>(plan.in_channels);
  out["out"] = static_cast<int>(plan.out_channels);
  out["color_stage"] = plan.color_stage;
  return out;
}

// The measurement channels of one slice, as pixels: (names, (C, h, w) float32).
//
// The GUI's 3D assembly runs in Python over whole slices, so it needs the same
// scale-space rasters the CLI measures on. They are computed per slice and
// dropped again rather than cached with each primed slice -- twelve float32
// rasters per slice would dominate a primed subsequence's memory.
py::tuple stat_channel_images(const FloatArray& base, const FloatArray& filtered,
                              const std::string& params_json, const py::object& color) {
  std::size_t bh = 0, bw = 0, fh = 0, fw = 0;
  const diffg::Image<float> base_img = to_image(base, bh, bw);
  const diffg::Image<float> filt_img = to_image(filtered, fh, fw);
  if (bh != fh || bw != fw) throw std::runtime_error("base and filtered must share shape");
  const diffg::MultiImage<float> color_planes = optional_planes(color, bh, bw);
  const msseg::StatsSpec spec = parse_stats_spec(parse_params(params_json));

  msseg::StatChannelBank bank;
  {
    py::gil_scoped_release release;
    bank = msseg::build_stat_channels(base_img, filt_img, spec, {},
                                      color_planes.channels() ? &color_planes : nullptr);
  }

  py::list names;
  for (const auto& c : bank.channels) names.append(py::str(c.name));

  const auto n = static_cast<py::ssize_t>(bank.size());
  FloatArray out({n, static_cast<py::ssize_t>(bh), static_cast<py::ssize_t>(bw)});
  float* dst = static_cast<float*>(out.request().ptr);
  const std::size_t plane = bh * bw;
  for (std::size_t k = 0; k < bank.size(); ++k) {
    std::memcpy(dst + k * plane, bank.channel(k), plane * sizeof(float));
  }
  return py::make_tuple(std::move(names), std::move(out));
}

// Build a primed Msc2DPipeline over `base` (original image) + `filtered` (the
// topology field, already filter-chained). Both are float32 (h,w).
msseg::Msc2DPipeline prime_slice(const FloatArray& base, const FloatArray& filtered,
                                 const std::string& params_json, const py::object& color) {
  std::size_t bh = 0, bw = 0, fh = 0, fw = 0;
  const diffg::Image<float> base_img = to_image(base, bh, bw);
  const diffg::Image<float> filt_img = to_image(filtered, fh, fw);
  if (bh != fh || bw != fw) throw std::runtime_error("base and filtered must share shape");
  const diffg::MultiImage<float> color_planes = optional_planes(color, bh, bw);
  const msseg::Msc2DParams msc = parse_msc(parse_params(params_json));

  msseg::Msc2DPipeline pipe;
  {
    py::gil_scoped_release release;
    pipe.build(base_img, filt_img, msc, nullptr, color_planes.channels() ? &color_planes : nullptr);
  }
  return pipe;
}

// Re-measure a primed pipeline under params_json's statistics block, keeping its
// MSC, labels and arcs. The rasters must be the ones it was primed from (same
// shape); only `statistics` and the measurement flags are read.
void pipeline_remeasure(msseg::Msc2DPipeline& pipe, const std::string& params_json,
                        const FloatArray& base, const FloatArray& filtered,
                        const py::object& color) {
  std::size_t bh = 0, bw = 0, fh = 0, fw = 0;
  const diffg::Image<float> base_img = to_image(base, bh, bw);
  const diffg::Image<float> filt_img = to_image(filtered, fh, fw);
  if (bh != fh || bw != fw) throw std::runtime_error("base and filtered must share shape");
  const diffg::MultiImage<float> color_planes = optional_planes(color, bh, bw);
  const msseg::Msc2DParams msc = parse_msc(parse_params(params_json));
  py::gil_scoped_release release;
  pipe.remeasure(base_img, filt_img, msc, color_planes.channels() ? &color_planes : nullptr);
}

// Feature id per pixel (int32 h,w) at the pipeline's current persistence.
py::array_t<std::int32_t> pipeline_labels(const msseg::Msc2DPipeline& pipe) {
  const std::vector<int>& labels = pipe.labels();
  return to_2d_array<py::array_t<std::int32_t>>(
      labels.data(), labels.size(), static_cast<std::size_t>(pipe.height()),
      static_cast<std::size_t>(pipe.width()), "the pipeline labeling");
}

// Living-region adjacency at the current persistence: (a int32[n], b int32[n],
// saddle float32[n]). Ids are the labels()/feature_id space, a < b, sorted by
// (a, b). Empty arrays when the linked msc_2d_lib lacks livingRegionArcs().
py::tuple pipeline_region_arcs(msseg::Msc2DPipeline& pipe) {
  const std::vector<msseg::Msc2DRegionArc>* arcs = nullptr;
  {
    py::gil_scoped_release release;
    arcs = &pipe.region_arcs();
  }
  const auto n = static_cast<py::ssize_t>(arcs->size());
  py::array_t<std::int32_t> a(n);
  py::array_t<std::int32_t> b(n);
  py::array_t<float> saddle(n);
  auto* pa = static_cast<std::int32_t*>(a.request().ptr);
  auto* pb = static_cast<std::int32_t*>(b.request().ptr);
  auto* ps = static_cast<float*>(saddle.request().ptr);
  for (py::ssize_t i = 0; i < n; ++i) {
    const msseg::Msc2DRegionArc& arc = (*arcs)[static_cast<std::size_t>(i)];
    pa[i] = arc.a;
    pb[i] = arc.b;
    ps[i] = arc.saddle_value;
  }
  return py::make_tuple(std::move(a), std::move(b), std::move(saddle));
}

// The seam graph of ANY int32 (h,w) label raster (-1 background): the crack
// polylines along which regions meet. Returns (a, b, j0, j1, offsets, points
// (P,2), junction_xy (J,2)) -- see msseg/graph/seam_graph.hpp for the
// canonical form the numpy reference (msseg.labeler.seams) reproduces.
py::tuple seam_graph_py(py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> labels) {
  const auto info = labels.request();
  if (info.ndim != 2) throw std::runtime_error("seam_graph: expected a 2D (h,w) int32 array");
  const auto h = static_cast<std::size_t>(info.shape[0]);
  const auto w = static_cast<std::size_t>(info.shape[1]);
  msseg::SeamGraph sg;
  {
    py::gil_scoped_release release;
    sg = msseg::extract_seam_graph(static_cast<const std::int32_t*>(info.ptr), h, w);
  }
  const auto vec_i32 = [](const std::vector<std::int32_t>& v) {
    py::array_t<std::int32_t> out(static_cast<py::ssize_t>(v.size()));
    if (!v.empty()) std::memcpy(out.request().ptr, v.data(), v.size() * sizeof(std::int32_t));
    return out;
  };
  const auto pairs_i32 = [](const std::vector<std::int32_t>& v) {
    py::array_t<std::int32_t> out({static_cast<py::ssize_t>(v.size() / 2), static_cast<py::ssize_t>(2)});
    if (!v.empty()) std::memcpy(out.request().ptr, v.data(), v.size() * sizeof(std::int32_t));
    return out;
  };
  py::array_t<std::int64_t> offsets(static_cast<py::ssize_t>(sg.offsets.size()));
  std::memcpy(offsets.request().ptr, sg.offsets.data(), sg.offsets.size() * sizeof(std::int64_t));
  return py::make_tuple(vec_i32(sg.a), vec_i32(sg.b), vec_i32(sg.j0), vec_i32(sg.j1), std::move(offsets),
                        pairs_i32(sg.points), pairs_i32(sg.junction_xy));
}

// Per-surviving-feature statistics, COLUMNAR: the field names once, then one
// (n_features, n_fields) float64 array.
//
// The previous shape -- one dict per feature -- allocated a py::str and a dict
// entry per field per feature on every persistence change. A twelve-channel
// scale-space stack is ~50 fields, so that is hundreds of thousands of Python
// objects per slider commit; as a block it is one buffer copy. The Python side
// (assembly.py) is already structure-of-arrays, so this removes a conversion.
// Phase -> wall ms of the prime, in the order the phases ran, "total" last
// (a dict keeps insertion order, so the caller can print it as a timeline).
py::dict pipeline_build_timings(const msseg::Msc2DPipeline& pipe) {
  py::dict out;
  for (const auto& t : pipe.build_timings()) out[py::str(t.phase)] = t.ms;
  return out;
}

py::tuple pipeline_feature_table(const msseg::Msc2DPipeline& pipe) {
  const mscoupon::FeatureTable table = mscoupon::feature_table(
      pipe.feature_stats(), pipe.feature_channels(), pipe.channels(), pipe.stats());

  py::list names;
  for (const auto& f : table.fields) names.append(py::str(f.name));

  py::array_t<double> values({static_cast<py::ssize_t>(table.n_rows),
                              static_cast<py::ssize_t>(table.fields.size())});
  if (!table.values.empty()) {
    std::memcpy(values.request().ptr, table.values.data(), table.values.size() * sizeof(double));
  }
  return py::make_tuple(std::move(names), std::move(values));
}

// Back-compatible list-of-dicts view, built from the same table. Convenient for
// small readouts; prefer feature_table() anywhere the feature count is large.
py::list pipeline_feature_stats(const msseg::Msc2DPipeline& pipe) {
  const mscoupon::FeatureTable table = mscoupon::feature_table(
      pipe.feature_stats(), pipe.feature_channels(), pipe.channels(), pipe.stats());
  py::list rows;
  for (std::size_t r = 0; r < table.n_rows; ++r) {
    py::dict d;
    for (std::size_t c = 0; c < table.fields.size(); ++c) {
      d[py::str(table.fields[c].name)] = table.at(r, c);
    }
    rows.append(std::move(d));
  }
  return rows;
}

// The measurement channels a params JSON resolves to, as a list of dicts
// {name, kind, sigma}. The GUI's channel picker and its image-background
// dropdown are generated from this.
py::list stat_channels_py(const std::string& params_json) {
  const msseg::StatsSpec spec = parse_stats_spec(parse_params(params_json));
  py::list out;
  for (const auto& c : msseg::resolve_stat_channels(spec)) {
    py::dict d;
    d["name"] = c.name;
    d["kind"] = c.kind;
    d["sigma"] = c.sigma;
    d["source"] = c.source;
    d["input_channel"] = c.input_channel;
    d["hist"] = c.hist;
    if (c.hist) d["hist_range"] = py::make_tuple(c.hist_lo, c.hist_hi);
    out.append(std::move(d));
  }
  return out;
}

// The full field schema: one dict per column, {name, channel, reduction}. The
// GUI builds its two-level [channel][reduction] picker from this rather than
// parsing names like "mean_blur_s0.7" -- and so "min_x" is never mistaken for
// reduction "min" on a channel "x".
py::list feature_schema_py(const std::string& params_json) {
  const msseg::StatsSpec spec = parse_stats_spec(parse_params(params_json));
  py::list out;
  for (const auto& f : mscoupon::feature_schema(spec)) {
    py::dict d;
    d["name"] = f.name;
    d["channel"] = f.channel;
    d["reduction"] = f.reduction;
    out.append(std::move(d));
  }
  return out;
}

// The field names a given statistics spec produces. The GUI builds its field
// dropdown from this instead of a hand-kept list, so the offered fields are
// exactly the ones the CLI will accept.
std::vector<std::string> feature_fields_py(const std::string& params_json) {
  return mscoupon::feature_fields(parse_stats_spec(parse_params(params_json)));
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

// Evaluate the query chain against a COLUMNAR table: the field names once, then
// an (n, f) float64 block. Field names resolve to column indices once for the
// whole table rather than being hashed per feature, which is what keeps a wide
// channel set usable on a slider.
std::vector<bool> evaluate_queries_table(const std::vector<std::string>& names,
                                         const py::array_t<double, py::array::c_style |
                                                                   py::array::forcecast>& values,
                                         const std::string& queries_json) {
  const auto info = values.request();
  if (info.ndim != 2) throw std::runtime_error("values must be a 2-D (n, f) array");
  const auto n_rows = static_cast<std::size_t>(info.shape[0]);
  const auto n_cols = static_cast<std::size_t>(info.shape[1]);
  if (n_cols != names.size()) {
    throw std::runtime_error("values has " + std::to_string(n_cols) +
                             " columns but " + std::to_string(names.size()) + " names");
  }

  mscoupon::FeatureTable table;
  table.fields.reserve(names.size());
  for (const auto& n : names) table.fields.push_back(mscoupon::FeatureField{n, "", ""});
  table.n_rows = n_rows;
  const double* src = static_cast<const double*>(info.ptr);
  table.values.assign(src, src + n_rows * n_cols);

  const std::vector<mscoupon::FeatureQuery> queries = parse_feature_queries(queries_json);
  const mscoupon::CompiledQueries compiled = mscoupon::compile_queries(table, queries);
  std::vector<bool> keep(n_rows);
  for (std::size_t r = 0; r < n_rows; ++r) keep[r] = mscoupon::row_passes(table, r, compiled);
  return keep;
}

// --- 1-D Gaussian mixture -------------------------------------------------

// The JSON -> options parsers live in the library (mscoupon/measure_config.hpp)
// so the CLI's `normalize` filter op and these bindings accept exactly the same
// keys; there is no second parser to drift.

// Run `fn` if the array holds dtype T; false means "not this dtype, try the
// next". The array is only copied when it is not already C-contiguous T, and
// the GIL is released for the duration of the measure.
template <typename T, typename Fn>
bool run_if_dtype(const py::array& image, const char* what, Fn& fn) {
  if (!image.dtype().equal(py::dtype::of<T>())) return false;
  auto arr = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(image);
  if (!arr)
    throw std::runtime_error(std::string(what) + ": could not view the input as a contiguous array");
  const T* data = arr.data();
  const auto count = static_cast<std::size_t>(arr.size());
  const auto ndim = arr.ndim();
  const int height = ndim == 2 ? static_cast<int>(arr.shape(0)) : 0;
  const int width = ndim == 2 ? static_cast<int>(arr.shape(1)) : 0;
  {
    py::gil_scoped_release release;
    fn(data, count, width, height);
  }
  return true;
}

// Dispatch `fn(const T* data, size_t count, int width, int height)` over every
// real numeric dtype a TIFF can arrive as. width/height are 0 unless the array
// is 2-D. Shared by all three measures so the dtype list cannot drift.
template <typename Fn>
void dispatch_dtype(const py::array& image, const char* what, Fn fn) {
  const bool handled =
      run_if_dtype<float>(image, what, fn) || run_if_dtype<double>(image, what, fn) ||
      run_if_dtype<std::int16_t>(image, what, fn) ||
      run_if_dtype<std::uint16_t>(image, what, fn) ||
      run_if_dtype<std::uint8_t>(image, what, fn) ||
      run_if_dtype<std::int8_t>(image, what, fn) ||
      run_if_dtype<std::int32_t>(image, what, fn) ||
      run_if_dtype<std::uint32_t>(image, what, fn) ||
      run_if_dtype<std::int64_t>(image, what, fn) ||
      run_if_dtype<std::uint64_t>(image, what, fn);
  if (!handled)
    throw std::runtime_error(std::string(what) +
                             ": unsupported dtype; expected a real numeric array "
                             "(float32/64, int8/16/32/64 or uint8/16/32/64)");
}

py::dict gmm_result_dict(const mscoupon::GmmResult& r) {
  py::list components;
  for (const auto& c : r.components) {
    py::dict d;
    d["mean"] = c.mean;
    d["sigma"] = c.sigma;
    d["weight"] = c.weight;
    d["n_hard"] = c.n_hard;
    d["hard_mean"] = c.hard_mean;
    d["median"] = c.median;
    d["mode"] = c.mode;
    components.append(std::move(d));
  }

  py::dict out;
  out["components"] = std::move(components);
  out["n_valid"] = r.n_valid;
  out["n_sampled"] = r.n_sampled;
  out["n_fit"] = r.n_fit;
  out["trim_lo"] = r.trim_lo;
  out["trim_hi"] = r.trim_hi;
  out["log_likelihood"] = r.log_likelihood;
  out["n_iter"] = r.n_iter;
  out["converged"] = r.converged;
  return out;
}

// Fit a 1-D Gaussian mixture to the pixels of any real numeric array. Shape is
// irrelevant -- the pixels are treated as an unordered bag, as in the Python.
py::dict fit_gmm(const py::array& image, const std::string& params_json) {
  const mscoupon::GmmOptions opts = mscoupon::parse_gmm_options(parse_params(params_json));
  mscoupon::GmmResult r;
  dispatch_dtype(image, "gmm", [&](const auto* data, std::size_t count, int, int) {
    r = mscoupon::fit_gmm(data, count, opts);
  });
  return gmm_result_dict(r);
}

// --- Histogram peaks ------------------------------------------------------

py::dict histogram_result_dict(const mscoupon::HistogramResult& r) {
  py::dict out;
  out["peak_low"] = r.peak_low;
  out["peak_high"] = r.peak_high;
  out["peak_low_bin"] = r.peak_low_bin;
  out["peak_high_bin"] = r.peak_high_bin;
  out["peak_low_height"] = r.peak_low_height;
  out["peak_high_height"] = r.peak_high_height;
  out["hist_lo"] = r.hist_lo;
  out["hist_hi"] = r.hist_hi;
  out["n_total_pixels"] = r.n_total;
  out["n_zero_pixels"] = r.n_zero;
  out["n_valid_pixels"] = r.n_valid;
  out["n_sampled_pixels"] = r.n_sampled;
  out["min"] = r.min_value;
  out["max"] = r.max_value;

  const auto& names = mscoupon::default_percentile_names();
  for (std::size_t i = 0; i < names.size() && i < r.percentiles.size(); ++i) {
    out[py::str(names[i])] = r.percentiles[i];
  }
  return out;
}

py::dict measure_histogram(const py::array& image, const std::string& params_json) {
  const mscoupon::HistogramOptions opts =
      mscoupon::parse_histogram_options(parse_params(params_json));
  mscoupon::HistogramResult r;
  dispatch_dtype(image, "histogram", [&](const auto* data, std::size_t count, int, int) {
    r = mscoupon::measure_histogram(data, count, opts);
  });
  return histogram_result_dict(r);
}

// --- Fixed rectangular regions --------------------------------------------

py::dict region_stats_dict(const mscoupon::RegionStats& s) {
  py::dict out;
  out["n_pixels"] = s.n_pixels;
  out["min"] = s.min_value;
  out["max"] = s.max_value;
  out["mean"] = s.mean;
  out["std"] = s.std_dev;

  const auto& names = mscoupon::default_percentile_names();
  for (std::size_t i = 0; i < names.size() && i < s.percentiles.size(); ++i) {
    out[py::str(names[i])] = s.percentiles[i];
  }
  return out;
}

// Measure every rectangle named in params_json["regions"], e.g.
// {"regions": {"air": {"rows": "250:350", "cols": "740:840"}, "metal": {...}}}.
py::dict measure_regions(const py::array& image, const std::string& params_json) {
  const nlohmann::json cfg = parse_params(params_json);
  const mscoupon::RegionOptions opts = mscoupon::parse_region_options(cfg);

  const nlohmann::json& spec = cfg.contains("regions") ? cfg.at("regions") : cfg;
  std::vector<std::pair<std::string, mscoupon::Rect>> rects;
  for (auto it = spec.begin(); it != spec.end(); ++it) {
    if (!it.value().is_object() || !it.value().contains("rows")) continue;
    rects.emplace_back(it.key(), mscoupon::parse_rect_json(it.value()));
  }
  if (rects.empty())
    throw std::runtime_error("region: no rectangles given; expected {\"name\": "
                             "{\"rows\": \"a:b\", \"cols\": \"c:d\"}, ...}");

  std::vector<mscoupon::RegionStats> stats(rects.size());
  dispatch_dtype(image, "region", [&](const auto* data, std::size_t, int width, int height) {
    if (width == 0 || height == 0)
      throw std::runtime_error("region: expected a 2-D image");
    for (std::size_t i = 0; i < rects.size(); ++i) {
      stats[i] = mscoupon::measure_region(data, width, height, rects[i].second, opts);
    }
  });

  py::dict out;
  for (std::size_t i = 0; i < rects.size(); ++i) {
    out[py::str(rects[i].first)] = region_stats_dict(stats[i]);
  }
  return out;
}

// Every sample plane of a TIFF as a planar (C,h,w) float32 array -- the CLI's
// reader, so the GUI loads exactly what a batch run will.
FloatArray read_tiff_planes_py(const std::string& path, const std::string& alpha) {
  msseg::ColorInputPolicy policy;
  policy.alpha = alpha == "keep" ? msseg::ColorInputPolicy::Alpha::Keep : msseg::ColorInputPolicy::Alpha::Drop;
  msseg::InputSlice slice;
  {
    py::gil_scoped_release release;
    slice = msseg::read_tiff_planes(std::filesystem::path(path), policy);
  }
  FloatArray out({static_cast<py::ssize_t>(slice.channels()), static_cast<py::ssize_t>(slice.height()),
                  static_cast<py::ssize_t>(slice.width())});
  std::memcpy(out.request().ptr, slice.planes.data(), slice.planes.size() * sizeof(float));
  return out;
}

}  // namespace

PYBIND11_MODULE(mscoupon_py, m) {
  m.doc() = "mscoupon instance: 2D Morse-Smale slice segmentation.";
  m.def("version", []() { return "0.1.0"; }, "Module version tag.");
  m.def("filter_slice", &filter_slice, py::arg("image"), py::arg("params_json") = std::string(),
        py::arg("default_color_method") = std::string("luminance"),
        "Apply the diffg filter from params_json['filter'] to a float32 (h,w) slice -- or to a "
        "planar (C,h,w) colour slice, where the stage must be (or is synthesized as) a `color` "
        "stage; returns the transformed float32 (h,w) field.");
  m.def("filter_chain", &filter_chain, py::arg("image"), py::arg("params_json") = std::string(),
        py::arg("default_color_method") = std::string("luminance"),
        "Apply the ordered filter chain from params_json['filters'] (or a single 'filter') "
        "to a float32 (h,w) slice or a planar (C,h,w) colour slice (a leading `color` stage, "
        "explicit or defaulted, reduces the planes); returns the float32 (h,w) topology field.");
  m.def("read_tiff_planes", &read_tiff_planes_py, py::arg("path"), py::arg("alpha") = std::string("drop"),
        "Every sample plane of a TIFF as planar (C,h,w) float32 -- the CLI's reader (TinyTIFF), "
        "so a grayscale file is (1,h,w) and RGB (3,h,w); alpha='drop' discards a 4th/2nd sample.");
  m.def("segment_slice", &segment_slice, py::arg("image"), py::arg("params_json") = std::string(),
        "Filter + 2D MSC segment a float32 (h,w) slice; returns int32 (h,w) manifold labels.");

  // Two-phase pipeline: prime once, then re-threshold cheaply with the merge tree.
  py::class_<msseg::Msc2DPipeline>(m, "Msc2DPipeline",
      "Primed 2D MSC pipeline: base decomposition + merge tree + statistics, with "
      "cheap persistence re-thresholding. Construct via prime_slice().")
      .def("select_persistence", &msseg::Msc2DPipeline::select_persistence, py::arg("persistence_absolute"),
           py::call_guard<py::gil_scoped_release>(),
           "Re-threshold to an absolute persistence (remap labels + re-aggregate stats).")
      .def("remeasure", &pipeline_remeasure, py::arg("params_json"), py::arg("base"),
           py::arg("filtered"), py::arg("color") = py::none(),
           "Recompute ONLY the statistics under params_json's `statistics` block, from the "
           "rasters this pipeline was primed from: the MSC, labels(), region_arcs() and the "
           "persistence are kept, feature_table() afterwards equals a fresh prime_slice() "
           "under the same params. build_timings() then holds the re-measure's phases.")
      .def("release_gpu", &msseg::Msc2DPipeline::release_gpu,
           "Free this slice's GPU residue (device label context); host results stay, "
           "the next select re-uploads lazily. Call when the slice stops being active.")
      .def("current_persistence", &msseg::Msc2DPipeline::current_persistence)
      .def("value_range", &msseg::Msc2DPipeline::value_range,
           "Filtered-field value range (max-min), for percent->absolute persistence.")
      .def("base_relevance_floor", &msseg::Msc2DPipeline::base_relevance_floor)
      .def("base_relevance_ceiling", &msseg::Msc2DPipeline::base_relevance_ceiling)
      .def("width", &msseg::Msc2DPipeline::width)
      .def("height", &msseg::Msc2DPipeline::height)
      .def("build_timings", &pipeline_build_timings,
           "Wall ms per phase of the prime, in the order they ran, 'total' last: "
           "{'value_range': .., 'msc': .., 'base_labels': .., 'compact': .., 'bridge': .., "
           "'relevance': .., 'stat_bank': .., 'accumulate': .., 'extremum': .., 'select': .., "
           "'release_gpu': .., 'total': ..}. Also printed to stderr unless MSSEG_TIME_MSC=0.")
      .def("labels", &pipeline_labels, "Feature id per pixel (int32 h,w) at the current persistence.")
      .def("region_arcs", &pipeline_region_arcs,
           "Living-region adjacency at the current persistence: (a, b, saddle) arrays in the "
           "labels() id space; empty on an old msc_2d_lib.")
      .def("feature_stats", &pipeline_feature_stats,
           "Per-living-feature statistics (list of dicts) at the current persistence.")
      .def("feature_table", &pipeline_feature_table,
           "Columnar per-living-feature statistics: (field_names, (n, f) float64 array). "
           "Prefer this over feature_stats() -- it is one buffer copy rather than a dict "
           "per feature, which is what keeps a wide channel set usable on a slider.");

  m.def("seam_graph", &seam_graph_py, py::arg("labels"),
        "Seam graph of an int32 (h,w) label raster (-1 background): the crack polylines along "
        "which regions meet, as (a, b, j0, j1, offsets, points (P,2), junction_xy (J,2)). "
        "Flanks a < b; j0 == j1 == -1 marks a loop, whose points repeat the first corner.");
  m.def("chain_plan", &chain_plan_py, py::arg("chain_json"), py::arg("channels") = 1,
        py::arg("default_method") = std::string("luminance"),
        py::arg("reduce_at") = std::string("front"),
        "Plan a filter chain over `channels` planes, with no raster: "
        "{stages: [{operation, params, in, out, index, synthesized}], in, out, color_stage}. "
        "A multi-plane input whose chain does not start with `color` gets one synthesized at "
        "the FRONT (index -1) exactly as the runner has always done -- it is simply visible now. "
        "Raises for `color` after index 0, a method the plane count refuses, and no planes.");
  m.def("prime_slice", &prime_slice, py::arg("base"), py::arg("filtered"),
        py::arg("params_json") = std::string(), py::arg("color") = py::none(),
        "Build a primed Msc2DPipeline over base (original) + filtered (topology field, "
        "already filter-chained), both float32 (h,w). params_json['msc'] configures it.");
  m.def("evaluate_queries_table", &evaluate_queries_table, py::arg("names"),
        py::arg("values"), py::arg("queries_json"),
        "Evaluate a feature-query chain against a columnar table (names + (n, f) float64). "
        "Same evaluator as evaluate_queries, but names resolve to columns once for the whole "
        "table instead of once per feature.");
  m.def("evaluate_queries", &evaluate_queries, py::arg("rows"), py::arg("queries_json"),
        "Evaluate a feature-query chain (JSON array of {field,op,value[,value2]}) against a "
        "list of feature-stat dicts; returns a list of bool keep flags. Shared 2D/3D evaluator.");

  m.def("feature_fields", &feature_fields_py, py::arg("params_json") = std::string(),
        "Field names the given params JSON's statistics block produces, in table order, i.e. "
        "exactly the fields feature_filters may name. Derived from the same schema the CLI "
        "validates against, so the GUI dropdown cannot drift out of sync.");
  m.def("stat_channel_images", &stat_channel_images, py::arg("base"), py::arg("filtered"),
        py::arg("params_json") = std::string(), py::arg("color") = py::none(),
        "The slice's measurement channels as pixels: (names, (C, h, w) float32), in the same "
        "slot order as stat_channels(). Used by the GUI's 3D assembly so it measures exactly "
        "what the CLI does.");
  m.def("feature_schema", &feature_schema_py, py::arg("params_json") = std::string(),
        "Per-column schema as dicts {name, channel, reduction}, in table order. Drives the "
        "GUI's two-level channel/reduction pickers without re-parsing field names.");
  m.def("stat_channels", &stat_channels_py, py::arg("params_json") = std::string(),
        "The measurement channels the params JSON resolves to, as dicts {name, kind, sigma, "
        "source, input_channel} in slot order -- base/filtered, the raw colour planes, and "
        "every derived scale-space channel (per plane on the colour source).");

  m.def("fit_gmm", &fit_gmm, py::arg("image"), py::arg("params_json") = std::string(),
        "Fit a 1-D Gaussian mixture to the pixels of a real numeric array of any shape "
        "(they are treated as an unordered bag). This is the C++ port of "
        "calculate_2_gaussian_mixture.py / measure_gmm.py.\n\n"
        "params_json['gmm'] (or a bare options object) accepts: preset "
        "('two_gaussian' | 'measure'), n_components, downsample_factor, omit_zeros, "
        "omit_nonfinite, trim_percent, init ('kmeans' | 'quantile'), n_init, max_iter, "
        "tol, reg_covar, seed, compute_hard_stats, mode_bins.\n\n"
        "Returns a dict: components (list of {mean, sigma, weight, n_hard, hard_mean, "
        "median, mode}, sorted by increasing mean), n_valid, n_sampled, n_fit, trim_lo, "
        "trim_hi, log_likelihood, n_iter, converged.");

  m.def("measure_histogram", &measure_histogram, py::arg("image"),
        py::arg("params_json") = std::string(),
        "Locate the two intensity populations by histogram peak finding. This is the C++ "
        "port of measure_im.py: mask zeros/non-finite, random subsample, build a histogram "
        "between two percentiles, smooth it, then take the two strongest separated local "
        "maxima and refine each to sub-bin precision.\n\n"
        "params_json['histogram'] (or a bare options object) accepts: downsample_factor, "
        "omit_zeros, omit_nonfinite, bins, smooth_width, peak_window, min_peak_distance, "
        "hist_lo_percentile, hist_hi_percentile, seed.\n\n"
        "Returns a dict: peak_low, peak_high (ordered by intensity, not height), "
        "peak_*_bin, peak_*_height, hist_lo, hist_hi, min, max, the p0_01..p99_99 "
        "percentile ladder, and the n_*_pixels counts.");

  m.def("measure_regions", &measure_regions, py::arg("image"),
        py::arg("params_json") = std::string(),
        "Measure intensity statistics inside named rectangles of a 2-D image. This is the "
        "C++ port of measure_2_regions.py.\n\n"
        "params_json['regions'] (or a bare object) maps each name to {'rows': 'START:END', "
        "'cols': 'START:END'} -- rows are Y, cols are X, matching image[r0:r1, c0:c1]. "
        "omit_zeros defaults to FALSE here (unlike the other measures): these are "
        "explicitly chosen physical regions, so exact zeros are kept unless asked "
        "otherwise.\n\n"
        "Returns a dict of name -> {n_pixels, min, max, mean, std, p0_01..p99_99}.");
}
