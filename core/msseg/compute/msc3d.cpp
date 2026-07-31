#include "msseg/compute/msc3d.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
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

// Option 3 "minima ignore boundary": a MorseSmaleComplexBasic whose cancellation
// validity test skips the boundary-equality gate for min->1-saddle (dim-0) arcs
// only. Boundary minima then cancel purely by persistence -- fixing background
// fragmentation, including boundary-min <-> boundary-min through a boundary
// saddle -- while the 1-2 and 2-3 boundary rules are left intact: 1-2 boundary
// cancellations stay feasible and a boundary 2-saddle still cannot cancel with
// an interior maximum. isValid is a protected virtual dispatched from
// ComputeHierarchy, so overriding it here keeps the change behind the GInt
// firewall (no MSCEER source edit). arc.dim is the lower endpoint's dimension.
class CellsegMsc : public MscType {
 public:
  using MscType::MscType;
  bool ignore_boundary_min_saddle = false;

  bool isValid(INT_TYPE /*a*/, GInt::arc<float>& ap) const override {
    if (!(ignore_boundary_min_saddle && ap.dim == 0) &&
        this->nodes[ap.lower].boundary != this->nodes[ap.upper].boundary)
      return false;
    // Endpoints must be connected by exactly one (living) arc.
    return this->countMultiplicity(ap, this->num_cancelled) == 1;
  }
};

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
  // Inverse of snapshot_gids: GInt node id -> compact snapshot NodeId (also
  // populated by snapshot(), used by living_labels()).
  std::unordered_map<INT_TYPE, NodeId> gid_to_compact;

  // Cached base per-vertex extremum labeling (all critical points alive): the
  // base minimum / maximum GInt node id covering each vertex, or -1. Filled by
  // compute_base_decomposition(); shared by basin_labels()/living_labels().
  std::vector<int> base_asc;  // ascending (minima) decomposition
  std::vector<int> base_dsc;  // descending (maxima) decomposition
  bool base_asc_done = false;
  bool base_dsc_done = false;

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
    // Both RobinsNoalloc and OnDemandAccurate start from the in-process Robins
    // steepest-descent pairing. With both OnDemandAccurate accuracy flags off,
    // that IS the requested gradient (the "ondemandaccurate all-false" case).
    RobinsType robins(impl_->topofunc.get(), impl_->mesh.get(), impl_->grad.get());
    robins.compute_output();

    if (params.gradient_mode == Msc3DParams::GradientMode::OnDemandAccurate &&
        (params.accurate_ascending_3m || params.accurate_descending_3m)) {
      // The accurate-3-manifold refinement (numeric integration + local
      // gradient recompute) needs MSCEER's alternate gradient stack
      // (MyRobinsNoalloc / RegularGridMaxMinVertexLabeling3D / path-compressing
      // integrators), which is not yet mirrored here. Fail loudly rather than
      // silently returning the plain steepest-descent gradient.
      throw std::runtime_error(
          "Msc3D: OnDemandAccurate accurate-3-manifold refinement "
          "(accurate_ascending_3m / accurate_descending_3m) is not yet "
          "implemented; use the default (both flags false) for steepest "
          "descent.");
    }
  }

  TopoAlgsType topo_algs(impl_->topofunc.get(), impl_->mesh.get(), impl_->grad.get());
  topo_algs.setAscendingManifoldDimensions();
}

void Msc3D::compute(const Msc3DParams& params) {
  if (!impl_->grad) throw std::runtime_error("Msc3D::compute called before build().");

  // Construct the cellseg MSC subclass so the cancellation hierarchy can, when
  // requested, ignore the boundary gate for min->1-saddle arcs only (Option 3).
  // The flag must be set before ComputeHierarchy; isValid reads it via virtual
  // dispatch. We keep boundary marks untouched so the 1-2 / 2-3 rules survive.
  auto msc = std::make_unique<CellsegMsc>(impl_->grad.get(), impl_->mesh.get(), impl_->topofunc.get());
  msc->ignore_boundary_min_saddle = params.minima_ignore_boundary;
  msc->SetBuildArcGeometry(GInt::Vec3b(params.build_arc_geometry, params.build_arc_geometry,
                                       params.build_arc_geometry));
  msc->ComputeFromGrad();
  impl_->msc = std::move(msc);

  // Build the cancellation hierarchy. A positive cap stops cancelling past that
  // persistence (cheaper; selection then limited to <= cap); otherwise build the
  // full value range so any persistence can be browsed.
  const float hier_cap = params.hierarchy_persistence_cap > 0.0f ? params.hierarchy_persistence_cap
                                                                 : impl_->value_range;
  impl_->msc->ComputeHierarchy(hier_cap);
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
  impl_->gid_to_compact.clear();

  std::unordered_map<INT_TYPE, NodeId>& gid_to_compact = impl_->gid_to_compact;

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

float Msc3D::value_range() const { return impl_->value_range; }

std::vector<float> Msc3D::node_cancellation_persistence() const {
  MscType* msc = impl_->mscOrThrow();
  // Reconstruct cancel_num_to_pers as the running max of the cancellation-record
  // persistences (mirrors how GInt builds it), then index by each node's
  // `destroyed` cancellation time.
  const auto& recs = msc->GetCancellationRecords();
  std::vector<float> cum(recs.size());
  float running = -std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < recs.size(); ++i) {
    running = std::max(running, static_cast<float>(recs[i].persistence));
    cum[i] = running;
  }

  const std::size_t nnodes = impl_->snapshot_gids.size();
  std::vector<float> out(nnodes, std::numeric_limits<float>::quiet_NaN());
  for (std::size_t i = 0; i < nnodes; ++i) {
    const INT_TYPE dt = msc->getNode(impl_->snapshot_gids[i]).destroyed;
    if (dt >= 1 && static_cast<std::size_t>(dt) <= cum.size()) {
      out[i] = cum[static_cast<std::size_t>(dt) - 1];  // absolute cancellation persistence
    }
    // else NaN: never cancelled (the component's surviving extremum)
  }
  return out;
}

void Msc3D::compute_base_decomposition(bool ascending) {
  MscType* msc = impl_->mscOrThrow();
  std::vector<int>& base = ascending ? impl_->base_asc : impl_->base_dsc;
  bool& done = ascending ? impl_->base_asc_done : impl_->base_dsc_done;
  if (done) return;

  const int target_dim = ascending ? 0 : 3;
  // Cell dimension to keep when mapping manifold cells to voxels: ascending
  // (minima) 3-manifolds are enumerated via their 0-cells (vertices);
  // descending (maxima) 3-manifolds contain no 0-cells, so we take their
  // top-dimensional (3-)cells and map each to a vertex via
  // VertexNumberFromCellID -- mirroring msc_2d_lib's ascending/descending
  // 2-manifold recipe (which keeps dim 0 vs dim D=2).
  const int keep_dim = ascending ? 0 : 3;
  const INDEX_TYPE nvert = impl_->grid->NumElements();
  const float saved_pers = impl_->selected_persistence;

  // Base labeling at full resolution (all critical points alive): each vertex
  // gets the base-node id of the extremum whose manifold covers it.
  base.assign(static_cast<std::size_t>(nvert), -1);
  msc->SetSelectPersAbs(-1.0f);
  MscType::LivingNodesIterator nit(msc);
  for (nit.begin(); nit.valid(); nit.advance()) {
    const INT_TYPE nid = nit.value();
    if (msc->getNode(nid).dim != target_dim) continue;
    std::set<INDEX_TYPE> manifold;
    msc->fillGeometry(nid, manifold, ascending);
    for (const INDEX_TYPE cid : manifold) {
      if (impl_->mesh->dimension(cid) != keep_dim) continue;
      base[static_cast<std::size_t>(impl_->mesh->VertexNumberFromCellID(cid))] = static_cast<int>(nid);
    }
  }

  // Restore the caller's persistence selection (the base pass perturbed it).
  msc->SetSelectPersAbs(saved_pers);
  done = true;
}

LabelVolume Msc3D::basin_labels(bool ascending) {
  compute_base_decomposition(ascending);
  MscType* msc = impl_->mscOrThrow();
  const int target_dim = ascending ? 0 : 3;
  const std::vector<int>& base = ascending ? impl_->base_asc : impl_->base_dsc;
  const INDEX_TYPE nvert = impl_->grid->NumElements();

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

LabelVolume Msc3D::living_labels(bool ascending, std::vector<std::int64_t>* voxel_counts) {
  compute_base_decomposition(ascending);
  MscType* msc = impl_->mscOrThrow();
  const int target_dim = ascending ? 0 : 3;
  const std::vector<int>& base = ascending ? impl_->base_asc : impl_->base_dsc;
  const INDEX_TYPE nvert = impl_->grid->NumElements();

  // Remap each base extremum's GInt id to the living extremum (GInt id) it
  // merged into at the current persistence.
  msc->SetSelectPersAbs(impl_->selected_persistence);
  std::unordered_map<int, INT_TYPE> remap;  // base gid -> living gid
  {
    MscType::LivingNodesIterator nit(msc);
    for (nit.begin(); nit.valid(); nit.advance()) {
      const INT_TYPE nid = nit.value();
      if (msc->getNode(nid).dim != target_dim) continue;
      std::set<INT_TYPE> constituents;
      msc->GatherNodes(nid, constituents, ascending);
      for (const INT_TYPE c : constituents) remap[static_cast<int>(c)] = nid;
    }
  }

  if (voxel_counts) voxel_counts->assign(impl_->snapshot_gids.size(), 0);

  LabelVolume out(diffg::Dimensions{static_cast<std::size_t>(impl_->dim_x), static_cast<std::size_t>(impl_->dim_y),
                                    static_cast<std::size_t>(impl_->dim_z)});
  for (INDEX_TYPE i = 0; i < nvert; ++i) {
    const int b = base[static_cast<std::size_t>(i)];
    std::int32_t label = kBackgroundLabel;
    if (b >= 0) {
      const auto rit = remap.find(b);
      if (rit != remap.end()) {
        const auto cit = impl_->gid_to_compact.find(rit->second);
        if (cit != impl_->gid_to_compact.end()) {
          const NodeId compact = cit->second;
          label = static_cast<std::int32_t>(compact) + 1;  // +1 so background stays 0
          if (voxel_counts) ++(*voxel_counts)[static_cast<std::size_t>(compact)];
        }
      }
    }
    out.data()[i] = label;
  }
  return out;
}

bool Msc3D::probe() {
  GInt::RegularGrid3D grid(GInt::Vec3l(4, 4, 4), GInt::Vec3b(false, false, false));
  const GInt::Vec3l& xyz = grid.XYZ();
  return xyz[0] == 4 && xyz[1] == 4 && xyz[2] == 4;
}

}  // namespace msseg
