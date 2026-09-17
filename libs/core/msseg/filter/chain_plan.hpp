#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "msseg/workflow/params.hpp"

namespace msseg {

// What a filter stage consumes and produces, counted in PLANES.
//
// The chain has one carrier today -- a single continuous plane -- and every
// stage that could produce something else reduces in place through a parameter
// of its own (see docs/design_filter_types.md). This type does not change that.
// It states the arity each stage ALREADY has, so that two rules which were
// special cases written out in three different files -- "`color` only at index
// 0", "every stage after it runs on a scalar" -- become one piece of data that
// one planner reads.
struct StageIO {
  enum class In {
    Scalar,  // one plane; the stage cannot see a stack
    Planes,  // the plane stack itself -- `color` alone, today
    Any,     // passes through whatever it is given (`none`)
  };
  enum class Out {
    Same,   // as many planes as came in
    One,    // exactly one, whatever came in
    Plane,  // a plane stage: ask plane_stage_output_channels() for the count
  };
  In in = In::Scalar;
  Out out = Out::Same;
};

// The signature of `stage`.
//
// `adapt` and `stain_deconvolution` report Planes -> Plane: how many planes they
// yield depends on their own parameters, and `plane_stage_output_channels` is
// the single place that decides, so a plan and an applier cannot disagree about
// k. An operation core does not know -- `normalize` is mscoupon's, and a package
// may add others -- is reported as Scalar -> Same, which is what every such
// stage has been. Unknown names are NOT rejected here: naming is validated
// where it always was (`apply_filter` at run time, the package's config parser
// at load time), and this function's subject is arity, not vocabulary.
StageIO stage_io(const FilterParams& stage);

// One stage as it will actually run.
struct StageRecord {
  FilterParams stage;  // the synthesized colour stage is materialized here
  std::size_t in_channels = 1;
  std::size_t out_channels = 1;
  // Where this came from in the caller's chain, or -1 when the planner added it.
  int config_index = -1;
  bool synthesized() const { return config_index < 0; }
};

// A chain, planned.
//
// Computable with NO raster, which is the whole point: config validation runs
// long before a slice is loaded, the GUI draws cards before a Run, and the
// runner needs the same answer at execution. One function, three callers, no
// drift -- the lesson `resolve_stat_channels` and `feature_fields` already
// taught.
struct ChainPlan {
  std::vector<StageRecord> stages;
  std::size_t in_channels = 1;
  std::size_t out_channels = 1;
  int color_stage = -1;  // index into `stages`, or -1 when the chain has none
};

// Thread `channels` planes through `chain`.
//
// Behaviour-preserving by construction. A multi-plane input whose chain does not
// start with `color` still gets one synthesized AT THE FRONT with
// `default_method`; what changes is that the stage is now IN the plan, with
// `synthesized()` true, instead of existing only inside the runner where
// nothing could see it. Moving it is a later and deliberate step -- see
// docs/design_filter_types.md.
//
// Throws on: no input planes; `color` anywhere but index 0 (the frozen-alias
// rule); a colour method the plane count refuses (`color_stage_accepts`); and a
// scalar stage handed a stack, which the leading reduction makes unreachable
// today and which is the guard a multi-plane carrier would rely on.
ChainPlan plan_chain(const std::vector<FilterParams>& chain, std::size_t channels,
                     const std::string& default_method);

}  // namespace msseg
