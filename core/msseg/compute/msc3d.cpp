#include "msseg/compute/msc3d.hpp"

#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// This is the ONLY translation unit permitted to include GInt headers.
// Order matters: basic types + timing first (the labeling/robins headers use
// ThreadedTimer without including it themselves), mirroring extractmsc.h.
#include "gi_basic_types.h"
#include "gi_timing.h"

#include "gi_discrete_gradient_labeling.h"
#include "gi_max_vertex_labeling.h"
#include "gi_morse_smale_complex_basic.h"
#include "gi_regular_grid.h"
#include "gi_regular_grid_trilinear_function.h"
#include "gi_robin_labeling.h"
#include "gi_topological_gradient_using_algorithms.h"
#include "gi_topological_max_vertex_mesh_function.h"
#include "gi_topological_regular_grid.h"

namespace msseg {

// Canonical GInt 3D typedef stack (mirrors MSCEER/extractmsc/extractmsc.h and
// the in-process gradient path in MSCEER/steepest/steepest.cxx).
using GridType = GInt::RegularGrid3D;
using MeshType = GInt::TopologicalRegularGrid3D;
using GridFuncType = GInt::RegularGridTrilinearFunction;
using GradType = GInt::DiscreteGradientLabeling<MeshType>;
using MaxVLType = GInt::MaximumVertexLabeling<MeshType, GridFuncType>;
using TopoFuncType = GInt::TopologicalMaxVertexMeshFunction<MeshType, MaxVLType, GridFuncType, float>;
using MscType = GInt::MorseSmaleComplexBasic<float, MeshType, TopoFuncType, GradType>;
using RobinsType = GInt::RobinsLabelingAlgorithm<MeshType, TopoFuncType>;
using TopoAlgsType = GInt::TopologicalGradientUsingAlgorithms<MeshType, TopoFuncType, GradType>;

struct Msc3D::Impl {
  std::vector<float> raw;  // referenced by gridfunc, kept alive here
  int dim_x = 0, dim_y = 0, dim_z = 0;
  float value_range = 0.0f;
  float selected_persistence = 0.0f;

  std::unique_ptr<GridType> grid;
  std::unique_ptr<GridFuncType> gridfunc;
  std::unique_ptr<MeshType> mesh;
  std::unique_ptr<MaxVLType> maxv;
  std::unique_ptr<TopoFuncType> topofunc;
  std::unique_ptr<GradType> grad;
  std::unique_ptr<MscType> msc;

  // Maps a compact snapshot NodeId back to the GInt node id (populated by
  // snapshot(), used by fill_manifold()).
  std::vector<INT_TYPE> snapshot_gids;

  MscType* mscOrThrow() const {
    if (!msc) throw std::runtime_error("Msc3D: complex not computed. Call build()+compute() first.");
    return msc.get();
  }
};

Msc3D::Msc3D() : impl_(std::make_unique<Impl>()) {}
Msc3D::~Msc3D() = default;
Msc3D::Msc3D(Msc3D&&) noexcept = default;
Msc3D& Msc3D::operator=(Msc3D&&) noexcept = default;

void Msc3D::build(const Volume& volume, const Msc3DParams& params) {
  const int X = static_cast<int>(volume.dims().width);
  const int Y = static_cast<int>(volume.dims().height);
  const int Z = static_cast<int>(volume.dims().depth);
  if (X <= 0 || Y <= 0 || Z <= 0) {
    throw std::runtime_error("Msc3D::build received an empty volume.");
  }

  impl_->dim_x = X;
  impl_->dim_y = Y;
  impl_->dim_z = Z;
  impl_->raw.assign(volume.data(), volume.data() + volume.size());

  impl_->grid = std::make_unique<GridType>(GInt::Vec3l(X, Y, Z), GInt::Vec3b(false, false, false));
  impl_->gridfunc = std::make_unique<GridFuncType>(impl_->grid.get(), impl_->raw.data());
  impl_->value_range = impl_->gridfunc->GetMaxValue() - impl_->gridfunc->GetMinValue();

  impl_->mesh = std::make_unique<MeshType>(impl_->grid.get());

  impl_->maxv = std::make_unique<MaxVLType>(impl_->mesh.get(), impl_->gridfunc.get());
  impl_->maxv->ComputeOutput();

  impl_->topofunc = std::make_unique<TopoFuncType>();
  impl_->topofunc->setMeshAndFuncAndMaxVLabeling(impl_->mesh.get(), impl_->gridfunc.get(), impl_->maxv.get());

  impl_->grad = std::make_unique<GradType>(impl_->mesh.get());
  impl_->grad->ClearAllGradient();

  if (params.gradient_mode == Msc3DParams::GradientMode::GradFile) {
    impl_->grad->load_from_file(params.grad_file_path.c_str());
  } else {
    RobinsType robins(impl_->topofunc.get(), impl_->mesh.get(), impl_->grad.get());
    robins.compute_output();
  }

  TopoAlgsType topo_algs(impl_->topofunc.get(), impl_->mesh.get(), impl_->grad.get());
  topo_algs.setAscendingManifoldDimensions();
}

void Msc3D::compute(const Msc3DParams& params) {
  if (!impl_->grad) throw std::runtime_error("Msc3D::compute called before build().");

  impl_->msc = std::make_unique<MscType>(impl_->grad.get(), impl_->mesh.get(), impl_->topofunc.get());
  impl_->msc->SetBuildArcGeometry(GInt::Vec3b(params.build_arc_geometry, params.build_arc_geometry,
                                              params.build_arc_geometry));
  impl_->msc->ComputeFromGrad();
  // Build the full cancellation hierarchy so any persistence can be browsed.
  impl_->msc->ComputeHierarchy(impl_->value_range);
  impl_->msc->SetSelectPersAbs(0.0f);
  impl_->selected_persistence = 0.0f;
}

void Msc3D::select_persistence(float persistence) {
  impl_->mscOrThrow()->SetSelectPersAbs(persistence);
  impl_->selected_persistence = persistence;
}

int Msc3D::living_node_count(int index_dim) const {
  MscType* msc = impl_->mscOrThrow();
  int count = 0;
  MscType::LivingNodesIterator nit(msc);
  for (nit.begin(); nit.valid(); nit.advance()) {
    if (index_dim < 0 || msc->getNode(nit.value()).dim == index_dim) ++count;
  }
  return count;
}

MscGraph Msc3D::snapshot() const {
  MscType* msc = impl_->mscOrThrow();
  MscGraph graph;
  impl_->snapshot_gids.clear();

  std::unordered_map<INT_TYPE, NodeId> gid_to_compact;

  MscType::LivingNodesIterator nit(msc);
  for (nit.begin(); nit.valid(); nit.advance()) {
    const INT_TYPE gid = nit.value();
    const auto& n = msc->getNode(gid);
    const NodeId compact = static_cast<NodeId>(graph.nodes.size());
    gid_to_compact[gid] = compact;
    impl_->snapshot_gids.push_back(gid);

    GInt::Vec3l c;
    impl_->mesh->cellid2Coords(n.cellindex, c);
    MscNode node;
    node.id = compact;
    node.cell_index = static_cast<CellIndex>(n.cellindex);
    node.index_dim = static_cast<int>(n.dim);
    node.value = n.value;
    node.on_boundary = n.boundary != 0;
    node.pos = {static_cast<float>(c[0]) * 0.5f, static_cast<float>(c[1]) * 0.5f, static_cast<float>(c[2]) * 0.5f};
    graph.nodes.push_back(node);
  }

  graph.adjacency.assign(graph.nodes.size(), {});

  MscType::LivingArcsIterator ait(msc);
  for (ait.begin(); ait.valid(); ait.advance()) {
    const INT_TYPE aid = ait.value();
    const auto& a = msc->getArc(aid);
    const auto lo = gid_to_compact.find(a.lower);
    const auto hi = gid_to_compact.find(a.upper);
    if (lo == gid_to_compact.end() || hi == gid_to_compact.end()) continue;

    MscArc arc;
    arc.id = static_cast<NodeId>(graph.arcs.size());
    arc.lower = lo->second;
    arc.upper = hi->second;
    arc.index_dim = static_cast<int>(a.dim);
    arc.persistence = a.persistence;
    graph.adjacency[static_cast<std::size_t>(arc.lower)].push_back(arc.id);
    graph.adjacency[static_cast<std::size_t>(arc.upper)].push_back(arc.id);
    graph.arcs.push_back(arc);
  }

  return graph;
}

void Msc3D::fill_manifold(NodeId node_id, bool ascending, std::set<CellIndex>& out) const {
  MscType* msc = impl_->mscOrThrow();
  if (node_id < 0 || static_cast<std::size_t>(node_id) >= impl_->snapshot_gids.size()) {
    throw std::runtime_error("Msc3D::fill_manifold: node id out of range (call snapshot() first).");
  }
  std::set<INDEX_TYPE> cells;
  msc->fillGeometry(impl_->snapshot_gids[static_cast<std::size_t>(node_id)], cells, ascending);
  out.clear();
  for (const INDEX_TYPE cid : cells) out.insert(static_cast<CellIndex>(cid));
}

LabelVolume Msc3D::basin_labels(bool ascending) {
  MscType* msc = impl_->mscOrThrow();
  const int target_dim = ascending ? 0 : 3;
  const INDEX_TYPE nvert = impl_->grid->NumElements();

  // Base labeling at full resolution (all critical points alive): each vertex
  // gets the base-node id of the extremum whose manifold covers it.
  std::vector<int> base(static_cast<std::size_t>(nvert), -1);
  msc->SetSelectPersAbs(-1.0f);
  {
    MscType::LivingNodesIterator nit(msc);
    for (nit.begin(); nit.valid(); nit.advance()) {
      const INT_TYPE nid = nit.value();
      if (msc->getNode(nid).dim != target_dim) continue;
      std::set<INDEX_TYPE> manifold;
      msc->fillGeometry(nid, manifold, ascending);
      for (const INDEX_TYPE cid : manifold) {
        if (impl_->mesh->dimension(cid) != 0) continue;
        base[static_cast<std::size_t>(impl_->mesh->VertexNumberFromCellID(cid))] = static_cast<int>(nid);
      }
    }
  }

  // Remap base-node ids to the living extremum they merged into at the current
  // persistence.
  msc->SetSelectPersAbs(impl_->selected_persistence);
  std::unordered_map<int, int> remap;
  {
    MscType::LivingNodesIterator nit(msc);
    for (nit.begin(); nit.valid(); nit.advance()) {
      const INT_TYPE nid = nit.value();
      if (msc->getNode(nid).dim != target_dim) continue;
      std::set<INT_TYPE> constituents;
      msc->GatherNodes(nid, constituents, ascending);
      for (const INT_TYPE c : constituents) remap[static_cast<int>(c)] = static_cast<int>(nid);
    }
  }

  LabelVolume out(diffg::Dimensions{static_cast<std::size_t>(impl_->dim_x), static_cast<std::size_t>(impl_->dim_y),
                                    static_cast<std::size_t>(impl_->dim_z)});
  for (INDEX_TYPE i = 0; i < nvert; ++i) {
    const int b = base[static_cast<std::size_t>(i)];
    const auto it = (b >= 0) ? remap.find(b) : remap.end();
    // Offset by +1 so background (unlabeled) is kBackgroundLabel (0).
    out.data()[i] = (it != remap.end()) ? static_cast<std::int32_t>(it->second) + 1 : kBackgroundLabel;
  }
  return out;
}

bool Msc3D::probe() {
  GInt::RegularGrid3D grid(GInt::Vec3l(4, 4, 4), GInt::Vec3b(false, false, false));
  const GInt::Vec3l& xyz = grid.XYZ();
  return xyz[0] == 4 && xyz[1] == 4 && xyz[2] == 4;
}

}  // namespace msseg
