#include "mscoupon/filter.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "diffg/blur.hpp"
#include "diffg/differentiator.hpp"
#include "diffg/edges.hpp"
#include "diffg/hessian.hpp"
#include "diffg/image.hpp"
#include "diffg/laplacian.hpp"
#include "diffg/options.hpp"
#include "diffg/structure.hpp"

namespace mscoupon {
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

Image2D from_diffg(const diffg::Image<float>& input) {
  Image2D out;
  out.width = static_cast<int>(input.dims().width);
  out.height = static_cast<int>(input.dims().height);
  out.pixels.assign(input.data(), input.data() + input.size());
  return out;
}

diffg::Image<float> to_diffg(const Image2D& input) {
  diffg::Image<float> out(
      diffg::Dimensions{static_cast<std::size_t>(input.width), static_cast<std::size_t>(input.height), 1});
  std::copy(input.pixels.begin(), input.pixels.end(), out.data());
  return out;
}

}  // namespace

Image2D apply_filter(const Image2D& image, const FilterConfig& filter) {
  if (filter.operation == "none" || filter.operation.empty()) {
    return image;
  }

  diffg::ExecutionOptions exec{};
  exec.threads = std::max(1, get_int(filter.params, "threads", 1));

  auto input = to_diffg(image);
  auto view = input.view();

  if (filter.operation == "blur") {
    const double sigma = get_double(filter.params, "sigma", 1.0);
    return from_diffg(diffg::blur(view, sigma, exec));
  }

  if (filter.operation == "derivative") {
    const double sigma = get_double(filter.params, "sigma", 1.0);
    const int ox = get_int(filter.params, "order_x", 1);
    const int oy = get_int(filter.params, "order_y", 0);
    const int oz = get_int(filter.params, "order_z", 0);
    return from_diffg(diffg::derivative(view, sigma, ox, oy, oz, exec));
  }

  if (filter.operation == "laplacian") {
    const double sigma = get_double(filter.params, "sigma", 1.0);
    return from_diffg(diffg::laplacian(view, sigma, exec));
  }

  if (filter.operation == "zero_crossings") {
    const double sigma = get_double(filter.params, "sigma", 1.0);
    auto lap = diffg::laplacian(view, sigma, exec);
    auto zc = diffg::zero_crossings(lap.view(), exec);
    Image2D out;
    out.width = image.width;
    out.height = image.height;
    out.pixels.resize(zc.size(), 0.0f);
    for (std::size_t i = 0; i < zc.size(); ++i) out.pixels[i] = zc.data()[i] > 0 ? 1.0f : 0.0f;
    return out;
  }

  if (filter.operation == "hessian_eigenvalues") {
    const double sigma = get_double(filter.params, "sigma", 1.0);
    const bool abs_sort = get_bool(filter.params, "sort_by_absolute_value", true);
    const std::string component = filter.params.value("component", "largest");
    const auto result = diffg::hessian_eigenvalues(view, sigma, abs_sort, exec);
    if (component == "largest") return from_diffg(result.largest);
    if (component == "middle" && result.has_middle) return from_diffg(result.middle);
    if (component == "smallest") return from_diffg(result.smallest);
    throw std::runtime_error("Invalid hessian_eigenvalues component: " + component);
  }

  if (filter.operation == "structure_eigenvalues") {
    const double smoothing_sigma = get_double(filter.params, "smoothing_sigma", 1.0);
    const double integration_sigma = get_double(filter.params, "integration_sigma", 2.0);
    const std::string component = filter.params.value("component", "largest");
    const auto result = diffg::structure_eigenvalues(view, smoothing_sigma, integration_sigma, exec);
    if (component == "largest") return from_diffg(result.largest);
    if (component == "middle" && result.has_middle) return from_diffg(result.middle);
    if (component == "smallest") return from_diffg(result.smallest);
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
    if (output == "magnitude") return from_diffg(result.gradient_magnitude);
    if (output == "mask" && result.has_mask) {
      Image2D out;
      out.width = image.width;
      out.height = image.height;
      out.pixels.resize(result.edge_mask.size(), 0.0f);
      for (std::size_t i = 0; i < result.edge_mask.size(); ++i) out.pixels[i] = result.edge_mask.data()[i] > 0 ? 1.0f : 0.0f;
      return out;
    }
    throw std::runtime_error("Invalid edges output mode: " + output);
  }

  throw std::runtime_error("Unknown filter operation: " + filter.operation);
}

}  // namespace mscoupon
