#include "cellseg/heavy_lift.hpp"

#include <cstdio>
#include <stdexcept>
#include <utility>

#include "msseg/filter/filter_stage.hpp"
#include "msseg/io/raw_io.hpp"
#include "msseg/workflow/params.hpp"

namespace cellseg {
namespace {

inline void stage(const char* msg) { std::fprintf(stderr, "[cellseg] %s\n", msg); std::fflush(stderr); }

msseg::Msc3DParams make_msc_params(const HeavyLiftConfig& cfg) {
  msseg::Msc3DParams p;
  p.gradient_mode = msseg::Msc3DParams::GradientMode::OnDemandAccurate;
  p.accurate_ascending_3m = cfg.accurate_ascending_3m;
  p.accurate_descending_3m = cfg.accurate_descending_3m;
  p.presimp_threshold = cfg.presimp_threshold;
  p.integration_error = cfg.integration_error;
  p.gradient_threshold = cfg.gradient_threshold;
  p.integration_max_iter = cfg.integration_max_iter;
  p.minima_ignore_boundary = cfg.minima_ignore_boundary;
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
  stage("filter: gaussian blur");
  msseg::FilterParams filter;
  filter.operation = "blur";
  filter.params = {{"sigma", cfg.blur_sigma}};
  state.filtered = msseg::apply_filter(input, filter);

  stage("build: discrete gradient");
  msseg::Msc3DParams params = make_msc_params(cfg);
  state.msc.build(state.filtered, params);

  // Resolve the persistence now (value range is known after build) and use it
  // as the hierarchy cap: ComputeHierarchy only cancels up to this persistence.
  state.value_range = state.msc.value_range();
  state.heavy_persistence = resolve_persistence(cfg, state.value_range);
  params.hierarchy_persistence_cap = state.heavy_persistence;
  stage("compute: MS complex + hierarchy");
  state.msc.compute(params);
  stage("select persistence");
  state.msc.select_persistence(state.heavy_persistence);

  // Cache both base 3-manifold decompositions once (the expensive pass).
  stage("base decomposition: ascending");
  state.msc.compute_base_decomposition(/*ascending=*/true);
  stage("base decomposition: descending");
  state.msc.compute_base_decomposition(/*ascending=*/false);
  stage("heavy lift done");
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
