#include "msseg/filter/chain_plan.hpp"

#include <stdexcept>
#include <string>
#include <utility>

#include "msseg/filter/color_stage.hpp"
#include "msseg/filter/plane_stages.hpp"

namespace msseg {

StageIO stage_io(const FilterParams& stage) {
  StageIO io;
  if (stage.operation == "none" || stage.operation.empty()) {
    io.in = StageIO::In::Any;
    io.out = StageIO::Out::Same;
    return io;
  }
  if (stage.operation == kColorOperation) {
    io.in = StageIO::In::Planes;
    // `optical_density` can hand back the OD planes themselves rather than a
    // projection of them; every other method reduces to one.
    io.out = color_stage_keeps_planes(stage) ? StageIO::Out::Same : StageIO::Out::One;
    return io;
  }
  if (is_plane_operation(stage.operation)) {
    io.in = StageIO::In::Planes;
    io.out = StageIO::Out::Plane;
    return io;
  }
  // Every other core operation, and every operation core does not know, reads
  // one plane and writes one. `hessian_eigenvalues`, `structure_eigenvalues` and
  // `edges` compute more than they return, but what they RETURN is one plane,
  // and a plan describes what runs -- the discarded components are catalogued in
  // docs/design_filter_types.md, not modelled here.
  io.in = StageIO::In::Scalar;
  io.out = StageIO::Out::Same;
  return io;
}

ChainPlan plan_chain(const std::vector<FilterParams>& chain, std::size_t channels,
                     const std::string& default_method) {
  if (channels == 0) {
    throw std::runtime_error("the filter chain was handed no input planes.");
  }
  // The frozen-alias rule, kept verbatim: `color` is a legacy spelling with
  // legacy placement, and moving it is not what this planner is for.
  for (std::size_t i = 1; i < chain.size(); ++i) {
    if (chain[i].operation == kColorOperation) {
      throw std::runtime_error("'color' must be the FIRST stage of a chain (found at index " +
                               std::to_string(i) + "): it consumes the input planes, and every later "
                               "stage runs on a scalar.");
    }
  }

  ChainPlan plan;
  plan.in_channels = channels;

  // Synthesize a conversion only when the chain does not ALREADY consume the
  // stack. `color` was the only plane-consuming stage when this rule was
  // written; `adapt` and `stain_deconvolution` consume it too, and a chain
  // headed by one of those must receive the planes, not a scalar made behind
  // its back. For every chain that predates them the test is unchanged, because
  // `color` is still the only In::Planes stage they contain.
  const bool consumes_planes =
      !chain.empty() && stage_io(chain.front()).in == StageIO::In::Planes;
  if (!consumes_planes && channels > 1) {
    FilterParams synthesized;
    synthesized.operation = kColorOperation;
    synthesized.params = nlohmann::json{{"method", default_method}};
    StageRecord rec;
    rec.stage = std::move(synthesized);
    rec.in_channels = channels;
    rec.out_channels = 1;
    rec.config_index = -1;
    plan.color_stage = 0;
    plan.stages.push_back(std::move(rec));
  }

  std::size_t current = plan.stages.empty() ? channels : plan.stages.back().out_channels;
  for (std::size_t i = 0; i < chain.size(); ++i) {
    const StageIO io = stage_io(chain[i]);
    if (io.in == StageIO::In::Scalar && current != 1) {
      throw std::runtime_error("'" + chain[i].operation + "' (stage " + std::to_string(i) +
                               ") reads one plane, but " + std::to_string(current) +
                               " reach it; reduce them first.");
    }
    StageRecord rec;
    rec.stage = chain[i];
    rec.in_channels = current;
    if (io.out == StageIO::Out::One) {
      rec.out_channels = 1u;
    } else if (io.out == StageIO::Out::Plane) {
      std::string why;
      const int k = plane_stage_output_channels(chain[i], current, &why);
      if (k < 0) throw std::runtime_error(why);
      rec.out_channels = static_cast<std::size_t>(k);
    } else {
      rec.out_channels = current;
    }
    rec.config_index = static_cast<int>(i);
    if (chain[i].operation == kColorOperation) {
      plan.color_stage = static_cast<int>(plan.stages.size());
    }
    current = rec.out_channels;
    plan.stages.push_back(std::move(rec));
  }
  plan.out_channels = current;

  // The colour stage is checked against the RAW plane count, which is what it
  // sees. Shared with config validation so the two cannot disagree.
  if (plan.color_stage >= 0) {
    std::string why;
    if (!color_stage_accepts(plan.stages[static_cast<std::size_t>(plan.color_stage)].stage, channels,
                             &why)) {
      throw std::runtime_error(why);
    }
  }
  return plan;
}

}  // namespace msseg
