#pragma once

#include <memory>
#include <set>
#include <vector>

#include "msseg/graph/msc_graph.hpp"
#include "msseg/volume/types.hpp"

namespace msseg {

// Parameters for building the 3D discrete gradient + Morse-Smale complex.
struct Msc3DParams {
  // Discrete gradient source.
  enum class GradientMode { RobinsNoalloc, GradFile };
  GradientMode gradient_mode = GradientMode::RobinsNoalloc;
  std::string grad_file_path;  // used when gradient_mode == GradFile

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

  // Per-voxel basin labeling: for each vertex, the id of the extremum whose
  // ascending (minima) / descending (maxima) manifold contains it, remapped
  // through the simplification hierarchy at the current persistence. Voxels
  // left unassigned get kBackgroundLabel; real basins are >= 1. Mirrors the
  // recipe msc_2d_lib uses for ascending2Manifolds/descending2Manifolds.
  LabelVolume basin_labels(bool ascending);

  // Lightweight self-test that constructs the core GInt grid types. Used by
  // the build to validate that GInt compiles/links under our C++20 toolchain.
  static bool probe();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace msseg
