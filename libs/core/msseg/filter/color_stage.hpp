#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "diffg/image.hpp"
#include "diffg/multi_image.hpp"
#include "diffg/options.hpp"
#include "msseg/workflow/params.hpp"

namespace msseg {

// The colour->scalar stage. diffg does no colour-space work ("channels are just
// channels"), so the policy that turns C planes into the one raster a chain
// runs on lives here, in core, where the CLI and every GUI binding share it.
//
// It is a filter stage in the config -- {"operation": "color", "params":
// {"method": ...}} -- but a special one: it is the ONLY stage that consumes
// planes rather than a scalar, so it is valid only as the FIRST stage of a
// chain, and each chain (`filters` for the topology field, `base_filters` for
// the statistics base) picks its own conversion. A chain without one on a
// multi-plane input gets the policy's default method.
inline constexpr const char* kColorOperation = "color";

// Methods, by `params.method`:
//   pick             {channel}                       one plane as-is
//   luminance                                        Rec.709 over planes 0..2
//   weighted         {weights[C]}                    sum_c w_c I_c
//   mean | max | min                                 across planes
//   hsv              {component: hue|saturation|value}
//   optical_density  {i0: "max"|number|[C], eps, and one of stain[C] |
//                     weights[C] | channels[k]}      OD_c = -log10(max(I_c,eps)/I0_c);
//                                                    `stain` is unit-normalized and
//                                                    projected onto (Ruifrok single
//                                                    stain); default = total OD
//   chgradmag        {sigma}                         sqrt(sum_c |grad_c|^2)
//   dizenzo          {sigma, eigen: largest|smallest} eigenvalue of sum_c g_c g_c^T
//   structure        {smoothing_sigma, integration_sigma, eigen}
//                                                    colour structure tensor
// Every method honours `threads`.
const std::vector<std::string>& color_methods();
bool is_color_method(const std::string& method);

// Whether `stage` (an operation == "color" FilterParams) can run on `channels`
// planes: the method must exist, `luminance`/`hsv` need three planes, `pick`
// needs its channel in range, `weighted`/`stain` need C weights. `why` gets the
// reason on false. Shared by config validation and the chain planner so the two
// cannot disagree.
bool color_stage_accepts(const FilterParams& stage, std::size_t channels, std::string* why = nullptr);

// Reduce C planar planes to one scalar field. Throws with the reason from
// color_stage_accepts on a mismatch.
diffg::Image<float> apply_color_stage(diffg::MultiImageView<const float> planes,
                                      const FilterParams& stage);

// The leading part of a chain fed `channels` planes.
struct ColorChainPlan {
  std::optional<FilterParams> color;   // the stage to run first, explicit or synthesized
  std::size_t first_scalar_stage = 0;  // index of the first stage that runs on the scalar
};

// Rules:
//   * a `color` stage anywhere but index 0            -> throw
//   * channels > 1, no leading color stage            -> synthesize {method: default_method}
//   * channels == 1, no leading color stage           -> plan.color empty (today's path, exactly)
//   * any explicit color stage is checked with color_stage_accepts and throws on refusal
ColorChainPlan plan_color_chain(const std::vector<FilterParams>& chain, std::size_t channels,
                                const std::string& default_method);

}  // namespace msseg
