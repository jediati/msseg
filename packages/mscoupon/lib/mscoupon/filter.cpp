#include "mscoupon/filter.hpp"

#include <algorithm>
#include <cstddef>

#include "diffg/image.hpp"
#include "msseg/filter/filter_stage.hpp"
#include "mscoupon/normalize.hpp"

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
// the actual transform to the (dimension-general) core filter stage. Routed
// through the chain so a lone `normalize` stage is handled here rather than
// being passed to core, which does not know the operation.
Image2D apply_filter(const Image2D& image, const FilterConfig& filter) {
  return apply_filter_chain(image, {filter});
}

Image2D apply_filter_chain(const Image2D& image, const std::vector<FilterConfig>& filters,
                           std::vector<TwoPoint>* normalizers_out) {
  // `normalize` is an mscoupon op: it needs this package's intensity measures,
  // which the (instance-agnostic) core filter stage knows nothing about. Run of
  // consecutive core ops are batched and delegated as before; a normalize stage
  // is applied here, in place, between them.
  Image2D current = image;
  std::vector<msseg::FilterParams> pending;

  const auto flush = [&]() {
    if (pending.empty()) return;
    current = from_diffg(msseg::apply_filter_chain(to_diffg(current), pending));
    pending.clear();
  };

  for (const auto& f : filters) {
    if (f.operation == kNormalizeOperation) {
      flush();
      const NormalizeConfig cfg = parse_normalize_config(f.params);
      const TwoPoint tp = measure_two_point(current, cfg);
      apply_two_point(current, tp, cfg.clamp);
      if (normalizers_out != nullptr) normalizers_out->push_back(tp);
      continue;
    }
    msseg::FilterParams params;
    params.operation = f.operation;
    params.params = f.params;
    pending.push_back(std::move(params));
  }
  flush();
  return current;
}

}  // namespace mscoupon
