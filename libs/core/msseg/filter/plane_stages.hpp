#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "diffg/image.hpp"
#include "diffg/multi_image.hpp"
#include "msseg/workflow/params.hpp"

namespace msseg {

// The stages that carry PLANES rather than reduce them away.
//
// `color` has been the only plane-consuming stage, and it always yields one
// plane, which is why the chain has had a single carrier since stage 0 (see
// docs/design_filter_types.md). These two do not: they take C planes and yield
// k, so a chain can hold a stack between two stages.
//
//   adapt                the uniform adaptor. Modes:
//     select  {channels: [i...]}            C -> k, the named planes in order
//     project {matrix: [[..C..] x k]}       C -> k, a linear map
//             {preset: "luminance"}         sugar for the Rec.709 row
//   stain_deconvolution  {preset|stains, i0, eps, od}
//                        RGB -> stain concentrations, c = pinv(M) * od
//
// `adapt` is the positionally-free spelling: unlike `color`, whose index-0 rule
// is frozen, it may appear anywhere the arity works out.
inline constexpr const char* kAdaptOperation = "adapt";
inline constexpr const char* kStainOperation = "stain_deconvolution";

bool is_plane_operation(const std::string& operation);

// The modes `adapt` accepts, and the presets `stain_deconvolution` knows, for
// error messages and for the GUI's pickers.
const std::vector<std::string>& adapt_modes();
const std::vector<std::string>& stain_presets();

// How many planes `stage` yields from `in_channels`, or -1 with `why` set when
// the stage cannot run on that many. Shared by the chain planner and the
// appliers so the two cannot disagree -- the same contract `color_stage_accepts`
// has with `apply_color_stage`.
int plane_stage_output_channels(const FilterParams& stage, std::size_t in_channels,
                                std::string* why = nullptr);

// Apply one plane stage. Throws with the reason from
// plane_stage_output_channels on a mismatch.
diffg::MultiImage<float> apply_plane_stage(diffg::MultiImageView<const float> planes,
                                           const FilterParams& stage);

// The unit OD vectors a preset names, column-major as a 3xk matrix (k columns of
// 3). "he" and "hdab" carry two stains; the third column their caller gets is
// Ruifrok's complement, the normalized cross product of the first two, so the
// map is invertible and the light that neither stain explains lands in a
// `residual` plane rather than being smeared across the two that matter.
std::vector<double> stain_matrix(const std::string& preset, std::string* why = nullptr);

}  // namespace msseg
