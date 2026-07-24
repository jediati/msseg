#include "cellseg/heavy_lift.hpp"

#include <stdexcept>
#include <utility>

#include "msseg/filter/filter_stage.hpp"
#include "msseg/io/raw_io.hpp"
#include "msseg/workflow/params.hpp"

namespace cellseg {
namespace {

msseg::Msc3DParams make_msc_params(const HeavyLiftConfig& cfg) {
  msseg::Msc3DParams p;
  p.gradient_mode = msseg::Msc3DParams::GradientMode::OnDemandAccurate;
  p.accurate_ascending_3m = cfg.accurate_ascending_3m;
  p.accurate_descending_3m = cfg.accurate_descending_3m;
  p.presimp_threshold = cfg.presimp_threshold;
  p.integration_error = cfg.integration_error;
  p.gradient_threshold = cfg.gradient_threshold;
  p.integration_max_iter = cfg.integration_max_iter;
  p.build_arcs = true;
  p.build_arc_geometry = false;
  return p;
}

float resolve_persistence(const HeavyLiftConfig& cfg, float value_range) {
  if (cfg.persistence_absolute) return *cfg.persistence_absolute;
  const float pct = cfg.persistence_percent ? *cfg.persistence_percent : 5.0f;
  return value_range * (pct / 100.0f);
}

CellState build_from_volume(msseg::Volume input, const HeavyLiftConfig& cfg) {
  CellState state;

  // Filter: Gaussian blur (sigma).
  msseg::FilterParams filter;
  filter.operation = "blur";
  filter.params = {{"sigma", cfg.blur_sigma}};
  state.filtered = msseg::apply_filter(input, filter);

  const msseg::Msc3DParams params = make_msc_params(cfg);
  state.msc.build(state.filtered, params);
  state.msc.compute(params);

  state.value_range = state.msc.value_range();
  state.heavy_persistence = resolve_persistence(cfg, state.value_range);
  state.msc.select_persistence(state.heavy_persistence);

  // Cache both base 3-manifold decompositions once (the expensive pass).
  state.msc.compute_base_decomposition(/*ascending=*/true);
  state.msc.compute_base_decomposition(/*ascending=*/false);
  return state;
}

}  // namespace

CellState run_heavy_lift(msseg::Volume input, const HeavyLiftConfig& cfg) {
  return build_from_volume(std::move(input), cfg);
}

CellState run_heavy_lift(const HeavyLiftConfig& cfg) {
  std::size_t x = 0, y = 0, z = 0;
  bool have_dims = false;
  if (cfg.dim_x && cfg.dim_y && cfg.dim_z) {
    x = *cfg.dim_x;
    y = *cfg.dim_y;
    z = *cfg.dim_z;
    have_dims = true;
  } else if (parse_dims_from_filename(cfg.input_path, x, y, z)) {
    have_dims = true;
  }

  msseg::Volume input =
      have_dims ? msseg::read_raw_volume(cfg.input_path, msseg::Dims3{x, y, z})
                : msseg::read_raw_volume(cfg.input_path);  // "<path>.dat" sidecar
  return build_from_volume(std::move(input), cfg);
}

}  // namespace cellseg
