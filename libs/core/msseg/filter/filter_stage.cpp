#include "msseg/filter/filter_stage.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "diffg/blur.hpp"
#include "diffg/differentiator.hpp"
#include "diffg/edges.hpp"
#include "diffg/filter_bank.hpp"
#include "diffg/hessian.hpp"
#include "diffg/image.hpp"
#include "diffg/labeling.hpp"
#include "diffg/laplacian.hpp"
#include "diffg/morphology.hpp"
#include "diffg/options.hpp"
#include "diffg/structure.hpp"
#include "msseg/workflow/stat_channels.hpp"

namespace msseg {
namespace {

double get_double(const nlohmann::json& params, const char* key, double default_value) {
  if (!params.contains(key)) return default_value;
  return params.at(key).get<double>();
}

int get_int(const nlohmann::json& params, const char* key, int default_value) {
  if (!params.contains(key)) return default_value;
  return params.at(key).get<int>();
}

bool get_bool(const nlohmann::json& params, const char* key, bool default_value) {
  if (!params.contains(key)) return default_value;
  return params.at(key).get<bool>();
}

// Materialize a boolean/threshold mask image (values 0.0f / 1.0f) with the
// same dimensions as `like`, from a diffg mask image of any integral type.
template <typename MaskImage>
diffg::Image<float> mask_to_float(const MaskImage& mask, const diffg::Image<float>& like) {
  diffg::Image<float> out(like.dims());
  for (std::size_t i = 0; i < mask.size(); ++i) {
    out.data()[i] = mask.data()[i] > 0 ? 1.0f : 0.0f;
  }
  return out;
}

}  // namespace

diffg::Image<float> apply_filter(const diffg::Image<float>& input, const FilterParams& filter) {
  if (filter.operation == "none" || filter.operation.empty()) {
    return input;
  }

  diffg::ExecutionOptions exec{};
  exec.threads = std::max(1, get_int(filter.params, "threads", 1));

  const auto view = input.view();

  if (filter.operation == "blur") {
    const double sigma = get_double(filter.params, "sigma", 1.0);
    return diffg::blur(view, sigma, exec);
  }

  if (filter.operation == "derivative") {
    const double sigma = get_double(filter.params, "sigma", 1.0);
    const int ox = get_int(filter.params, "order_x", 1);
    const int oy = get_int(filter.params, "order_y", 0);
    const int oz = get_int(filter.params, "order_z", 0);
    return diffg::derivative(view, sigma, ox, oy, oz, exec);
  }

  if (filter.operation == "laplacian") {
    const double sigma = get_double(filter.params, "sigma", 1.0);
    return diffg::laplacian(view, sigma, exec);
  }

  if (filter.operation == "zero_crossings") {
    const double sigma = get_double(filter.params, "sigma", 1.0);
    auto lap = diffg::laplacian(view, sigma, exec);
    auto zc = diffg::zero_crossings(lap.view(), exec);
    return mask_to_float(zc, input);
  }

  if (filter.operation == "hessian_eigenvalues") {
    const double sigma = get_double(filter.params, "sigma", 1.0);
    const bool abs_sort = get_bool(filter.params, "sort_by_absolute_value", true);
    const std::string component = filter.params.value("component", "largest");
    const auto result = diffg::hessian_eigenvalues(view, sigma, abs_sort, exec);
    if (component == "largest") return result.largest;
    if (component == "middle" && result.has_middle) return result.middle;
    if (component == "smallest") return result.smallest;
    throw std::runtime_error("Invalid hessian_eigenvalues component: " + component);
  }

  if (filter.operation == "structure_eigenvalues") {
    const double smoothing_sigma = get_double(filter.params, "smoothing_sigma", 1.0);
    const double integration_sigma = get_double(filter.params, "integration_sigma", 2.0);
    const std::string component = filter.params.value("component", "largest");
    const auto result = diffg::structure_eigenvalues(view, smoothing_sigma, integration_sigma, exec);
    if (component == "largest") return result.largest;
    if (component == "middle" && result.has_middle) return result.middle;
    if (component == "smallest") return result.smallest;
    throw std::runtime_error("Invalid structure_eigenvalues component: " + component);
  }

  if (filter.operation == "edges") {
    const double sigma = get_double(filter.params, "sigma", 1.0);
    diffg::EdgeOptions edge_opts{};
    edge_opts.suppress_nonmax = get_bool(filter.params, "suppress_nonmax", false);
    if (filter.params.contains("low_threshold")) edge_opts.low_threshold = filter.params.at("low_threshold").get<float>();
    if (filter.params.contains("high_threshold")) edge_opts.high_threshold = filter.params.at("high_threshold").get<float>();
    const std::string output = filter.params.value("output", "magnitude");
    const auto result = diffg::edges(view, sigma, edge_opts, exec);
    if (output == "magnitude") return result.gradient_magnitude;
    if (output == "mask" && result.has_mask) return mask_to_float(result.edge_mask, input);
    throw std::runtime_error("Invalid edges output mode: " + output);
  }

  // Grayscale morphology (square structuring element of the given radius).
  if (filter.operation == "erode") {
    return diffg::erode(view, std::max(0, get_int(filter.params, "radius", 1)), exec);
  }
  if (filter.operation == "dilate") {
    return diffg::dilate(view, std::max(0, get_int(filter.params, "radius", 1)), exec);
  }
  if (filter.operation == "open") {
    return diffg::open(view, std::max(0, get_int(filter.params, "radius", 1)), exec);
  }
  if (filter.operation == "close") {
    return diffg::close(view, std::max(0, get_int(filter.params, "radius", 1)), exec);
  }

  // Connected-component labeling: threshold the input into a foreground mask
  // (value > `threshold`, default 0), label it (4-connected), and return the
  // integer component ids widened to float (0 = background).
  if (filter.operation == "label_components") {
    const float threshold = static_cast<float>(get_double(filter.params, "threshold", 0.0));
    diffg::Image<unsigned char> mask(input.dims());
    for (std::size_t i = 0; i < input.size(); ++i) {
      mask.data()[i] = input.data()[i] > threshold ? 1 : 0;
    }
    const auto labels = diffg::label_mask(mask.view(), exec);
    diffg::Image<float> out(input.dims());
    for (std::size_t i = 0; i < out.size(); ++i) {
      out.data()[i] = static_cast<float>(labels.data()[i]);
    }
    return out;
  }

  throw std::runtime_error("Unknown filter operation: " + filter.operation);
}

diffg::Image<float> apply_filter_chain(const diffg::Image<float>& input,
                                       const std::vector<FilterParams>& filters) {
  if (filters.empty()) {
    return input;
  }
  // Apply the first stage against the caller's input, then thread each output
  // into the next stage so we only keep one intermediate alive at a time.
  diffg::Image<float> current = apply_filter(input, filters.front());
  for (std::size_t i = 1; i < filters.size(); ++i) {
    current = apply_filter(current, filters[i]);
  }
  return current;
}

// --------------------------------------------------------------------------- //
// Derived measurement channels (the scale-space stack).
// --------------------------------------------------------------------------- //
namespace {

// One diffg request per (kind, sigma). A `hessian` request yields two output
// channels, which is why the resolved list is walked in lockstep with the
// request list rather than one-to-one.
diffg::FilterRequest to_diffg_request(const ResolvedStatChannel& c) {
  const diffg::Sigma sigma = diffg::Sigma::isotropic(c.sigma);
  if (c.kind == "blur") return diffg::gaussian_filter(sigma);
  if (c.kind == "gradmag" || c.kind == "edges") return diffg::gradient_magnitude_filter(sigma);
  if (c.kind == "laplacian") return diffg::laplacian_filter(sigma);
  if (c.kind == "hessian") return diffg::hessian_filter(sigma, c.sort_by_absolute_value);
  throw std::runtime_error("Unknown derived statistics channel kind: '" + c.kind + "'.");
}

}  // namespace

StatChannelBank build_stat_channels(const diffg::Image<float>& base,
                                    const diffg::Image<float>& filtered,
                                    const StatsSpec& spec,
                                    const diffg::ExecutionOptions& exec) {
  if (base.dims().width != filtered.dims().width ||
      base.dims().height != filtered.dims().height ||
      base.dims().depth != filtered.dims().depth) {
    throw std::runtime_error("build_stat_channels: base and filtered dimensions differ.");
  }

  StatChannelBank bank;
  bank.channels = resolve_stat_channels(spec);
  bank.data.assign(bank.channels.size(), nullptr);

  // Collect the derived slots, deduplicating requests: a `hessian` entry covers
  // two consecutive resolved channels but is one diffg request.
  std::vector<diffg::FilterRequest> requests;
  std::vector<std::size_t> derived_slots;   // resolved index per derived output channel
  for (std::size_t k = 0; k < bank.channels.size(); ++k) {
    const ResolvedStatChannel& c = bank.channels[k];
    if (c.kind == "base" || c.kind == "filtered") continue;
    if (c.slot_in_request == 0) requests.push_back(to_diffg_request(c));
    derived_slots.push_back(k);
  }

  if (!requests.empty()) {
    diffg::FilterBankOptions options;
    options.execution = exec;
    const auto result =
        diffg::apply_filter_bank(base.view(), requests, diffg::OutputShape::MultiChannel, options);
    if (result.channel_count() != derived_slots.size()) {
      throw std::runtime_error("build_stat_channels: filter bank returned " +
                               std::to_string(result.channel_count()) + " channels, expected " +
                               std::to_string(derived_slots.size()) + ".");
    }
    bank.derived = std::move(result.stacked);
    for (std::size_t i = 0; i < derived_slots.size(); ++i) {
      bank.data[derived_slots[i]] = bank.derived.channel_data(i);
    }
  }

  // base/filtered are aliased last, after `derived` has stopped moving, so no
  // pointer stored above can be invalidated by the move.
  for (std::size_t k = 0; k < bank.channels.size(); ++k) {
    if (bank.channels[k].kind == "base") bank.data[k] = base.data();
    else if (bank.channels[k].kind == "filtered") bank.data[k] = filtered.data();
  }
  return bank;
}

}  // namespace msseg
