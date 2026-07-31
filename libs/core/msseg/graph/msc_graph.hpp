#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace msseg {

// Plain-data snapshot of a (simplified) Morse-Smale complex. This is the
// contract between the compute layer (which owns the heavy GInt objects) and
// everything downstream -- segmentation strategies, the viewer, python. No
// GInt types cross this boundary.

using NodeId = std::int64_t;
using CellIndex = std::int64_t;

struct MscNode {
  NodeId id = -1;
  CellIndex cell_index = -1;
  int index_dim = 0;          // Morse index (0=min, up to D=max)
  float value = 0.0f;         // function value at the critical cell
  bool on_boundary = false;
  std::array<float, 3> pos{{0.0f, 0.0f, 0.0f}};
};

struct MscArc {
  NodeId id = -1;
  NodeId lower = -1;          // lower-index endpoint node id
  NodeId upper = -1;          // higher-index endpoint node id
  int index_dim = 0;          // dimension of the separatrix
  float persistence = 0.0f;
  std::vector<std::array<float, 3>> geometry;  // optional polyline geometry
};

struct MscGraph {
  std::vector<MscNode> nodes;
  std::vector<MscArc> arcs;
  // adjacency[node_id] -> list of arc ids incident to that node.
  std::vector<std::vector<NodeId>> adjacency;
};

}  // namespace msseg
