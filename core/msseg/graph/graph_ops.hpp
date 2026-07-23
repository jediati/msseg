#pragma once

#include <vector>

#include "msseg/graph/msc_graph.hpp"

namespace msseg {

// Reusable graph-walking primitives over an MscGraph. These are the substrate
// that segmentation strategies build on. The set here will grow as the
// graph-walking / local-segmentation logic is iterated on.

// Node ids whose Morse index equals `index_dim`.
std::vector<NodeId> nodes_of_index(const MscGraph& graph, int index_dim);

// Node ids reachable from `start` by traversing arcs whose persistence is at
// least `min_persistence` (breadth-first over the living subgraph).
std::vector<NodeId> reachable(const MscGraph& graph, NodeId start, float min_persistence);

}  // namespace msseg
