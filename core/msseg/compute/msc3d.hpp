#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "msseg/graph/msc_graph.hpp"
#include "msseg/volume/types.hpp"

namespace msseg {

// Parameters for building the 3D discrete gradient + Morse-Smale complex.
struct Msc3DParams {
  // Discrete gradient source.
  //  - RobinsNoalloc   : in-process Robins steepest-descent pairing.
  //  - GradFile        : load a precomputed gradient from disk.
  //  - OnDemandAccurate: the MSCEER "ondemandaccurate" builder. With both
  //    accuracy flags false (the default) this is plain steepest descent;
  //    enabling a flag refines the corresponding 3-manifold boundaries with a
  //    numeric-integration pass before recomputing the gradient locally.
  enum class GradientMode { RobinsNoalloc, GradFile, OnDemandAccurate };
  GradientMode gradient_mode = GradientMode::RobinsNoalloc;
  std::string grad_file_path;  // used when gradient_mode == GradFile

  // OnDemandAccurate demands (both false == steepest descent).
  bool accurate_ascending_3m = false;   // accurate ascending (minima) 3-manifolds
  bool accurate_descending_3m = false;  // accurate descending (maxima) 3-manifolds
  // Numeric-integration knobs (only used when an accuracy flag is set).
  float presimp_threshold = 0.0f;     // pre-simplification of the extremum graph
  float integration_error = 0.01f;    // adaptive-integration error threshold
  float gradient_threshold = 0.0f;    // low-gradient cutoff
  int integration_max_iter = 1000;    // per-streamline iteration limit

  bool build_arcs = true;
  bool build_arc_geometry = false;
};

// PIMPL facade over the MSCEER GInt 3D typedef stack
// (RegularGrid3D -> TopologicalRegularGrid3D -> ... -> MorseSmaleComplexBasic).
// This is the ONLY translation unit that includes gi_*.h, keeping the
// C++11-era MSCEER headers out of every other TU and out of the python ABI.
class Msc3D {
 public:
  Msc3D();
  ~Msc3D();
  Msc3D(Msc3D&&) noexcept;
  Msc3D& operator=(Msc3D&&) noexcept;
  Msc3D(const Msc3D&) = delete;
  Msc3D& operator=(const Msc3D&) = delete;

  // Load the volume + compute (or load) the discrete gradient.
  void build(const Volume& volume, const Msc3DParams& params);
  // Build arcs and the cancellation/simplification hierarchy.
  void compute(const Msc3DParams& params);
  // Browse the hierarchy at an absolute persistence threshold (O(1)-ish).
  void select_persistence(float persistence);
  // Plain-data snapshot of the currently-selected complex.
  MscGraph snapshot() const;
  // Number of living critical points of a given Morse index at the current
  // persistence (-1 for all dimensions). Convenience for tests/diagnostics.
  int living_node_count(int index_dim = -1) const;
  // Fill the ascending/descending manifold cell set of a snapshot node.
  void fill_manifold(NodeId node_id, bool ascending, std::set<CellIndex>& out) const;

  // Value range (max - min) of the loaded volume. Handy for percent-of-range
  // persistence thresholds (e.g. "5% of the input range").
  float value_range() const;

  // Per-voxel basin labeling: for each vertex, the id of the extremum whose
  // ascending (minima) / descending (maxima) manifold contains it, remapped
  // through the simplification hierarchy at the current persistence. Voxels
  // left unassigned get kBackgroundLabel; real basins are >= 1. Labels are the
  // raw living GInt node id + 1. Mirrors the recipe msc_2d_lib uses for
  // ascending2Manifolds/descending2Manifolds.
  LabelVolume basin_labels(bool ascending);

  // --- Two-phase decomposition (for repeated threshold tuning) -------------
  //
  // Phase A (heavy, run once): compute and cache the base per-voxel extremum
  // labeling (all critical points alive) for the ascending (minima) or
  // descending (maxima) 3-manifold decomposition. This is the expensive
  // fillGeometry pass; repeated calls for the same side are no-ops.
  void compute_base_decomposition(bool ascending);

  // Phase B (cheap, repeatable): per-voxel labels of the *living* extremum (at
  // the current persistence) whose ascending/descending manifold contains each
  // voxel, keyed to the compact NodeIds of the most recent snapshot() (+1;
  // background 0 == kBackgroundLabel). This lets the label volume, the MscGraph
  // snapshot, and the voxel counts all share one NodeId space.
  //
  // Requires a prior snapshot() taken at the current persistence. Calls
  // compute_base_decomposition(ascending) if the base pass has not run yet.
  // When `voxel_counts` is non-null it is resized to the snapshot's node count
  // and filled with the per-NodeId voxel count (0 for non-extremum nodes).
  LabelVolume living_labels(bool ascending, std::vector<std::int64_t>* voxel_counts = nullptr);

  // Lightweight self-test that constructs the core GInt grid types. Used by
  // the build to validate that GInt compiles/links under our C++20 toolchain.
  static bool probe();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace msseg
