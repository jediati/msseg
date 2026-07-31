#include "msseg/graph/graph_ops.hpp"

#include <cstddef>
#include <queue>
#include <vector>

namespace msseg {

std::vector<NodeId> nodes_of_index(const MscGraph& graph, int index_dim) {
  std::vector<NodeId> out;
  for (const auto& node : graph.nodes) {
    if (node.index_dim == index_dim) out.push_back(node.id);
  }
  return out;
}

std::vector<NodeId> reachable(const MscGraph& graph, NodeId start, float min_persistence) {
  std::vector<NodeId> out;
  if (start < 0 || static_cast<std::size_t>(start) >= graph.adjacency.size()) return out;

  std::vector<char> seen(graph.nodes.size(), 0);
  std::queue<NodeId> frontier;
  seen[static_cast<std::size_t>(start)] = 1;
  frontier.push(start);

  while (!frontier.empty()) {
    const NodeId cur = frontier.front();
    frontier.pop();
    out.push_back(cur);
    for (const NodeId arc_id : graph.adjacency[static_cast<std::size_t>(cur)]) {
      const MscArc& arc = graph.arcs[static_cast<std::size_t>(arc_id)];
      if (arc.persistence < min_persistence) continue;
      const NodeId other = (arc.lower == cur) ? arc.upper : arc.lower;
      if (other < 0 || static_cast<std::size_t>(other) >= seen.size()) continue;
      if (!seen[static_cast<std::size_t>(other)]) {
        seen[static_cast<std::size_t>(other)] = 1;
        frontier.push(other);
      }
    }
  }
  return out;
}

}  // namespace msseg
