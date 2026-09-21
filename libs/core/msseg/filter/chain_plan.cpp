#include "msseg/filter/chain_plan.hpp"

#include <stdexcept>
#include <string>
#include <utility>

#include "msseg/filter/color_stage.hpp"
#include "msseg/filter/plane_stages.hpp"

namespace msseg {
namespace {

// The scalar operations core itself implements. Anything else reaching
// `apply_filter` is a package's own (mscoupon's `normalize`), applied by that
// package between core batches -- so core cannot lift it, and says so.
bool is_core_scalar_operation(const std::string& op) {
  static const char* const kOps[] = {"blur",  "derivative",       "laplacian",
                                     "zero_crossings",           "hessian_eigenvalues",
                                     "structure_eigenvalues",    "edges",
                                     "erode", "dilate",           "open",
                                     "close", "label_components"};
  for (const char* k : kOps) {
    if (op == k) return true;
  }
  return false;
}

}  // namespace

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
  // `label_components` yields ids, and an operation core does not know may be
  // mscoupon's `normalize`, whose landmarks are a population statistic. Neither
  // means anything applied plane by plane.
  io.liftable = stage.operation != "label_components" && is_core_scalar_operation(stage.operation);
  return io;
}

// Would `chain`, fed `channels` planes and given no help, end on a single plane?
//
// Tolerant on purpose: anything it cannot answer means "no", which routes the
// chain to the leading conversion it has always had. That is what keeps a chain
// containing a stage core does not know -- `normalize` -- behaving exactly as it
// did, instead of failing in a probe.
bool chain_ends_scalar(const std::vector<FilterParams>& chain, std::size_t channels) {
  std::size_t current = channels;
  for (const FilterParams& stage : chain) {
    const StageIO io = stage_io(stage);
    if (io.in == StageIO::In::Scalar && current != 1) {
      if (!io.liftable) return false;
      continue;  // lifts: as many planes out as in
    }
    if (io.out == StageIO::Out::One) {
      current = 1;
    } else if (io.out == StageIO::Out::Plane) {
      const int k = plane_stage_output_channels(stage, current, nullptr);
      if (k < 0) return false;
      current = static_cast<std::size_t>(k);
    }
  }
  return current == 1;
}

ChainPlan plan_chain(const std::vector<FilterParams>& chain, std::size_t channels,
                     const std::string& default_method, const std::string& reduce_at) {
  if (reduce_at != kReduceAtFront && reduce_at != kReduceAtEnd && reduce_at != kReduceAtNone) {
    throw std::runtime_error("input.color.reduce_at must be front, end, or none (got '" +
                             reduce_at + "').");
  }
  if (channels == 0) {
    throw std::runtime_error("the filter chain was handed no input planes.");
  }
  // `color` was frozen to index 0 while it was the only plane-consuming stage
  // and everything after it was a scalar. Neither is true now: a chain can
  // carry a stack, so `color{method}` mid-chain has planes to reduce, and
  // refusing it left the methods without an adapt spelling -- luminance,
  // weighted, pick, mean/max/min -- unreachable there.
  //
  // Lifting the rule is strictly WIDENING: it was an error, so no valid config
  // changes meaning. That is the whole reason the freeze was worth having, and
  // the reason it can go.

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
  // ...and only when the chain would not otherwise land on a single plane. A
  // chain carrying its own reduction has said where that happens.
  const bool needs_conversion = !consumes_planes && channels > 1 &&
                                reduce_at == kReduceAtFront && !chain_ends_scalar(chain, channels);
  if (needs_conversion) {
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
    bool lifted = false;
    if (io.in == StageIO::In::Scalar && current != 1) {
      if (!io.liftable) {
        throw std::runtime_error("'" + chain[i].operation + "' (stage " + std::to_string(i) +
                                 ") reads one plane, but " + std::to_string(current) +
                                 " reach it, and it cannot run plane by plane; reduce them first "
                                 "(adapt, mode 'reduce' or 'select').");
      }
      lifted = true;  // run once per plane, as many out as in
    }
    StageRecord rec;
    rec.stage = chain[i];
    rec.in_channels = current;
    rec.lifted = lifted;
    if (lifted) {
      rec.out_channels = current;
    } else if (io.out == StageIO::Out::One) {
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
  // `reduce_at: end` appends the conversion instead, so the chain keeps its
  // planes for every stage that wants them and the reduction is the last thing
  // that happens. A stage that could not lift has already thrown above; that is
  // the honest outcome, since there is no scalar for it to run on.
  if (current > 1 && reduce_at == kReduceAtEnd) {
    FilterParams synthesized;
    synthesized.operation = kColorOperation;
    synthesized.params = nlohmann::json{{"method", default_method}};
    StageRecord rec;
    rec.stage = std::move(synthesized);
    rec.in_channels = current;
    rec.out_channels = 1;
    rec.config_index = -1;
    plan.color_stage = static_cast<int>(plan.stages.size());
    plan.stages.push_back(std::move(rec));
    current = 1;
  }
  plan.out_channels = current;

  // The colour stage is checked against the RAW plane count, which is what it
  // sees. Shared with config validation so the two cannot disagree.
  if (plan.color_stage >= 0) {
    const StageRecord& rec = plan.stages[static_cast<std::size_t>(plan.color_stage)];
    std::string why;
    // Against the planes it actually receives: a leading stage sees the input's,
    // an appended one sees whatever the chain has carried to it.
    if (!color_stage_accepts(rec.stage, rec.in_channels, &why)) throw std::runtime_error(why);
  }
  return plan;
}

}  // namespace msseg
