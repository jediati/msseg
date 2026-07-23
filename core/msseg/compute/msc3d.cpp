#include "msseg/compute/msc3d.hpp"

#include <stdexcept>

// This is the ONLY translation unit permitted to include GInt headers.
#include "gi_regular_grid.h"

namespace msseg {

// M1 skeleton: the full canonical typedef stack
//   RegularGrid3D -> TopologicalRegularGrid3D -> RegularGridTrilinearFunction
//   -> LazyMaximumVertexLabeling -> TopologicalMaxVertexMeshFunction
//   -> DiscreteGradientLabeling -> MorseSmaleComplexBasic<float, ...>
// is wired up in M3. For now Impl is a placeholder and the compute methods
// throw; probe() exercises real GInt compilation/linking under our toolchain.
struct Msc3D::Impl {
  // populated in M3
};

Msc3D::Msc3D() : impl_(std::make_unique<Impl>()) {}
Msc3D::~Msc3D() = default;
Msc3D::Msc3D(Msc3D&&) noexcept = default;
Msc3D& Msc3D::operator=(Msc3D&&) noexcept = default;

void Msc3D::build(const Volume&, const Msc3DParams&) {
  throw std::runtime_error("Msc3D::build not implemented yet (M3).");
}

void Msc3D::compute(const Msc3DParams&) {
  throw std::runtime_error("Msc3D::compute not implemented yet (M3).");
}

void Msc3D::select_persistence(float) {
  throw std::runtime_error("Msc3D::select_persistence not implemented yet (M3).");
}

MscGraph Msc3D::snapshot() const {
  throw std::runtime_error("Msc3D::snapshot not implemented yet (M3).");
}

void Msc3D::fill_manifold(NodeId, bool, std::set<CellIndex>&) const {
  throw std::runtime_error("Msc3D::fill_manifold not implemented yet (M3).");
}

bool Msc3D::probe() {
  GInt::RegularGrid3D grid(GInt::Vec3l(4, 4, 4), GInt::Vec3b(false, false, false));
  const GInt::Vec3l& xyz = grid.XYZ();
  return xyz[0] == 4 && xyz[1] == 4 && xyz[2] == 4;
}

}  // namespace msseg
