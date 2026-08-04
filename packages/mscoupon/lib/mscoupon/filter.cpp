#include "mscoupon/filter.hpp"

#include <algorithm>
#include <cstddef>

#include "diffg/image.hpp"
#include "msseg/filter/filter_stage.hpp"

namespace mscoupon {
namespace {

diffg::Image<float> to_diffg(const Image2D& input) {
  diffg::Image<float> out(
      diffg::Dimensions{static_cast<std::size_t>(input.width), static_cast<std::size_t>(input.height), 1});
  std::copy(input.pixels.begin(), input.pixels.end(), out.data());
  return out;
}

Image2D from_diffg(const diffg::Image<float>& input) {
  Image2D out;
  out.width = static_cast<int>(input.dims().width);
  out.height = static_cast<int>(input.dims().height);
  out.pixels.assign(input.data(), input.data() + input.size());
  return out;
}

}  // namespace

// Thin adapter: this instance keeps its Image2D batch currency and delegates
// the actual transform to the (dimension-general) core filter stage.
Image2D apply_filter(const Image2D& image, const FilterConfig& filter) {
  msseg::FilterParams params;
  params.operation = filter.operation;
  params.params = filter.params;
  return from_diffg(msseg::apply_filter(to_diffg(image), params));
}

Image2D apply_filter_chain(const Image2D& image, const std::vector<FilterConfig>& filters) {
  std::vector<msseg::FilterParams> chain;
  chain.reserve(filters.size());
  for (const auto& f : filters) {
    msseg::FilterParams params;
    params.operation = f.operation;
    params.params = f.params;
    chain.push_back(std::move(params));
  }
  return from_diffg(msseg::apply_filter_chain(to_diffg(image), chain));
}

}  // namespace mscoupon
