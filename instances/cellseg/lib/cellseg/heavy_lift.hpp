#pragma once

#include "cellseg/config.hpp"
#include "msseg/compute/msc3d.hpp"
#include "msseg/volume/types.hpp"

namespace cellseg {

// The once-per-input "heavy lift" state (Phase A). Holds the primed Morse-Smale
// complex (with cached ascending/descending base decompositions) and the
// filtered volume, ready for repeated Phase-B threshold tuning. Move-only,
// because Msc3D is move-only.
struct CellState {
  msseg::Msc3D msc;
  msseg::Volume filtered;
  float value_range = 0.0f;
  float heavy_persistence = 0.0f;  // persistence chosen during the heavy lift

  CellState() = default;
  CellState(CellState&&) noexcept = default;
  CellState& operator=(CellState&&) noexcept = default;
  CellState(const CellState&) = delete;
  CellState& operator=(const CellState&) = delete;
};

// Run Phase A from a filename (read -> blur -> gradient -> complex -> simplify
// -> base decompositions). Dimensions come from the filename's _XxYxZ suffix,
// the config overrides, or a "<path>.dat" sidecar (in that order).
CellState run_heavy_lift(const HeavyLiftConfig& cfg);

// Run Phase A from an already-in-memory volume (used by the Python binding).
CellState run_heavy_lift(msseg::Volume input, const HeavyLiftConfig& cfg);

}  // namespace cellseg
