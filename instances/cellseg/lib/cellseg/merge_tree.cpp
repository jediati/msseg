#include "cellseg/merge_tree.hpp"

#include <algorithm>
#include <functional>
#include <set>
#include <utility>

#include <nlohmann/json.hpp>

#include "cellseg/union_find.hpp"

namespace cellseg {

MergeTree build_merge_tree(const msseg::MscGraph& graph,
                           const std::vector<std::int64_t>& min_counts) {
  MergeTree tree;

  // Leaves: one per living minimum. dsu index runs parallel to `min_nodes`.
  std::unordered_map<msseg::NodeId, int> min_to_dsu;
  std::vector<msseg::NodeId> min_nodes;
  for (const auto& n : graph.nodes) {
    if (n.index_dim != 0) continue;
    min_to_dsu[n.id] = static_cast<int>(min_nodes.size());
    min_nodes.push_back(n.id);

    MergeNode leaf;
    leaf.id = static_cast<int>(tree.nodes.size());
    leaf.is_leaf = true;
    leaf.value = n.value;
    leaf.voxel_count = (static_cast<std::size_t>(n.id) < min_counts.size())
                           ? min_counts[static_cast<std::size_t>(n.id)]
                           : 0;
    leaf.msc_node = n.id;
    tree.leaf_of_min[n.id] = leaf.id;
    tree.nodes.push_back(leaf);
  }

  const int nmin = static_cast<int>(min_nodes.size());
  UnionFind uf(nmin);
  std::vector<int> active(static_cast<std::size_t>(nmin));         // tree-node idx per dsu rep
  std::vector<std::int64_t> vox(static_cast<std::size_t>(nmin));   // voxel sum per dsu rep
  for (int i = 0; i < nmin; ++i) {
    active[i] = tree.leaf_of_min[min_nodes[i]];
    vox[i] = tree.nodes[active[i]].voxel_count;
  }

  // 1-saddles in ascending value order.
  std::vector<std::pair<float, msseg::NodeId>> saddles;
  for (const auto& n : graph.nodes)
    if (n.index_dim == 1) saddles.emplace_back(n.value, n.id);
  std::sort(saddles.begin(), saddles.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  for (const auto& [sval, sid] : saddles) {
    // Distinct downward minima of this saddle.
    std::vector<msseg::NodeId> mins;
    for (const msseg::NodeId aid : graph.adjacency[static_cast<std::size_t>(sid)]) {
      const auto& a = graph.arcs[static_cast<std::size_t>(aid)];
      if (a.upper == sid && graph.nodes[static_cast<std::size_t>(a.lower)].index_dim == 0) {
        if (std::find(mins.begin(), mins.end(), a.lower) == mins.end()) mins.push_back(a.lower);
      }
    }
    if (mins.size() < 2) continue;

    // Fold: merge every distinct set this saddle touches (usually just two).
    int acc = uf.find(min_to_dsu[mins[0]]);
    for (std::size_t k = 1; k < mins.size(); ++k) {
      const int d = uf.find(min_to_dsu[mins[k]]);
      if (d == acc) continue;

      MergeNode m;
      m.id = static_cast<int>(tree.nodes.size());
      m.is_leaf = false;
      m.value = sval;
      m.msc_node = sid;
      m.children = {active[acc], active[d]};
      m.voxel_count = vox[acc] + vox[d];
      tree.nodes.push_back(m);

      const int newrep = uf.unite(acc, d);
      active[newrep] = m.id;
      vox[newrep] = m.voxel_count;
      acc = newrep;
    }
  }

  // Roots: the surviving active node of each disjoint set.
  std::set<int> seen;
  for (int i = 0; i < nmin; ++i) {
    const int r = uf.find(i);
    if (seen.insert(r).second) tree.roots.push_back(active[r]);
  }
  return tree;
}

std::string merge_tree_to_json(const MergeTree& tree) {
  std::function<nlohmann::json(int)> emit = [&](int idx) -> nlohmann::json {
    const MergeNode& n = tree.nodes[static_cast<std::size_t>(idx)];
    nlohmann::json j;
    j["id"] = n.id;
    j["type"] = n.is_leaf ? "leaf" : "merger";
    j["value"] = n.value;
    j["voxel_count"] = n.voxel_count;
    j["node_id"] = n.msc_node;
    if (!n.is_leaf) {
      nlohmann::json kids = nlohmann::json::array();
      for (int c : n.children) kids.push_back(emit(c));
      j["children"] = std::move(kids);
    }
    return j;
  };

  nlohmann::json roots = nlohmann::json::array();
  for (int r : tree.roots) roots.push_back(emit(r));
  nlohmann::json out;
  out["roots"] = std::move(roots);
  return out.dump(2);
}

std::unordered_map<msseg::NodeId, int> cut_regions(const MergeTree& tree, float cut_threshold,
                                                   std::vector<std::int64_t>* region_voxels) {
  std::unordered_map<msseg::NodeId, int> region_of_min;
  std::vector<std::int64_t> rvox;
  int next_region = 0;

  std::function<void(int, int)> assign = [&](int idx, int rid) {
    const MergeNode& n = tree.nodes[static_cast<std::size_t>(idx)];
    if (n.is_leaf) {
      region_of_min[n.msc_node] = rid;
      if (static_cast<int>(rvox.size()) <= rid) rvox.resize(static_cast<std::size_t>(rid) + 1, 0);
      rvox[static_cast<std::size_t>(rid)] += n.voxel_count;
      return;
    }
    if (n.value > cut_threshold) {          // cut here: children become new regions
      for (int c : n.children) assign(c, next_region++);
    } else {                                // keep: children stay in this region
      for (int c : n.children) assign(c, rid);
    }
  };

  for (int root : tree.roots) assign(root, next_region++);
  if (region_voxels) *region_voxels = std::move(rvox);
  return region_of_min;
}

}  // namespace cellseg
