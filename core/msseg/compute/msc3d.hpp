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
  // Fill the ascending/descending manifold cell set of a node.
  void fill_manifold(NodeId node_id, bool ascending, std::set<CellIndex>& out) const;

  // Lightweight self-test that constructs the core GInt grid types. Used by
  // the build to validate that GInt compiles/links under our C++20 toolchain.
  static bool probe();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace msseg
