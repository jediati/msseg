#include "cellseg/merge_tree.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

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
  // Flat format: a node list (id == index) with children stored as id
  // references, plus a roots id list. Avoids deeply-nested JSON (the tree can be
  // a million-deep caterpillar), so serialization/parsing stay O(1)-shallow.
  nlohmann::json nodes = nlohmann::json::array();
  for (const MergeNode& n : tree.nodes) {
    nlohmann::json j;
    j["id"] = n.id;
    j["type"] = n.is_leaf ? "leaf" : "merger";
    j["value"] = n.value;
    j["voxel_count"] = n.voxel_count;
    j["node_id"] = n.msc_node;
    j["children"] = n.children;  // flat list of child node ids
    nodes.push_back(std::move(j));
  }
  nlohmann::json out;
  out["nodes"] = std::move(nodes);
  out["roots"] = tree.roots;
  return out.dump();
}

std::unordered_map<msseg::NodeId, msseg::NodeId> branch_survivors(
    const MergeTree& tree, float cut_threshold, float select_persistence) {
  std::unordered_map<msseg::NodeId, msseg::NodeId> result;
  const int n = static_cast<int>(tree.nodes.size());
  if (n == 0) return result;

  // Pass 1 (post-order, iterative): lo = deepest leaf value in subtree, rep =
  // that leaf's node index. This is the same low-to-high pass the tree coloring
  // uses; the deepest leaf is each subtree's branch representative.
  std::vector<float> lo(static_cast<std::size_t>(n), 0.0f);
  std::vector<int> rep(static_cast<std::size_t>(n), -1);
  {
    std::vector<std::pair<int, bool>> st;
    for (const int r : tree.roots) st.emplace_back(r, false);
    while (!st.empty()) {
      const auto [idx, done] = st.back();
      st.pop_back();
      const auto& kids = tree.nodes[static_cast<std::size_t>(idx)].children;
      if (kids.empty()) {  // leaf
        lo[static_cast<std::size_t>(idx)] = tree.nodes[static_cast<std::size_t>(idx)].value;
        rep[static_cast<std::size_t>(idx)] = idx;
      } else if (done) {
        int best = kids[0];
        for (const int c : kids)
          if (lo[static_cast<std::size_t>(c)] < lo[static_cast<std::size_t>(best)]) best = c;
        lo[static_cast<std::size_t>(idx)] = lo[static_cast<std::size_t>(best)];
        rep[static_cast<std::size_t>(idx)] = rep[static_cast<std::size_t>(best)];
      } else {
        st.emplace_back(idx, true);
        for (const int c : kids) st.emplace_back(c, false);
      }
    }
  }

  // Pass 2 (pre-order, iterative): for each merger, every child whose branch rep
  // differs from the merger's rep is a *side* branch that dies here. It merges
  // into the deeper (main) branch unless its death is a barrier.
  std::vector<int> parent_leaf(static_cast<std::size_t>(n), -1);  // leaf idx -> leaf it dies into
  std::vector<char> surviving(static_cast<std::size_t>(n), 0);    // leaf idx survives?
  for (const int r : tree.roots) surviving[static_cast<std::size_t>(rep[static_cast<std::size_t>(r)])] = 1;
  {
    std::vector<int> st(tree.roots.rbegin(), tree.roots.rend());
    while (!st.empty()) {
      const int idx = st.back();
      st.pop_back();
      const MergeNode& nd = tree.nodes[static_cast<std::size_t>(idx)];
      if (nd.children.empty()) continue;
      const int mainrep = rep[static_cast<std::size_t>(idx)];
      for (const int c : nd.children) {
        const int leafL = rep[static_cast<std::size_t>(c)];
        if (leafL != mainrep) {  // side branch: dies at this merger
          const float pers = nd.value - lo[static_cast<std::size_t>(c)];  // leaf-to-merge
          parent_leaf[static_cast<std::size_t>(leafL)] = mainrep;
          if (nd.value > cut_threshold && pers >= select_persistence)
            surviving[static_cast<std::size_t>(leafL)] = 1;  // barrier -> its own feature
        }
        st.push_back(c);
      }
    }
  }

  // Resolve each leaf to its surviving ancestor branch (iterative pointer-jump
  // with memoization; no recursion).
  std::vector<int> memo(static_cast<std::size_t>(n), -1);
  for (int i = 0; i < n; ++i) {
    if (!tree.nodes[static_cast<std::size_t>(i)].children.empty()) continue;  // leaves only
    std::vector<int> path;
    int x = i;
    while (!surviving[static_cast<std::size_t>(x)]) {
      if (memo[static_cast<std::size_t>(x)] != -1) { x = memo[static_cast<std::size_t>(x)]; break; }
      const int p = parent_leaf[static_cast<std::size_t>(x)];
      if (p < 0) break;  // safety: a non-root branch always has a parent
      path.push_back(x);
      x = p;
    }
    for (const int p : path) memo[static_cast<std::size_t>(p)] = x;
    result[tree.nodes[static_cast<std::size_t>(i)].msc_node] =
        tree.nodes[static_cast<std::size_t>(x)].msc_node;
  }
  return result;
}

std::unordered_map<msseg::NodeId, int> cut_regions(const MergeTree& tree, float cut_threshold,
                                                   float select_persistence,
                                                   std::vector<std::int64_t>* region_voxels) {
  const std::unordered_map<msseg::NodeId, msseg::NodeId> survivors =
      branch_survivors(tree, cut_threshold, select_persistence);

  // Number surviving branches into regions. Iterative pre-order DFS (the tree
  // may be far too deep to recurse); a region id is allocated the first time a
  // leaf's surviving branch is referenced -- deterministic and mirrored in the
  // Python app so the ids volume and the tree coloring agree.
  std::unordered_map<msseg::NodeId, int> region_of_survivor;
  std::unordered_map<msseg::NodeId, int> region_of_min;
  std::vector<std::int64_t> rvox;
  int next_region = 0;

  std::vector<int> stack(tree.roots.rbegin(), tree.roots.rend());
  while (!stack.empty()) {
    const int idx = stack.back();
    stack.pop_back();
    const MergeNode& nd = tree.nodes[static_cast<std::size_t>(idx)];
    if (nd.children.empty()) {
      const msseg::NodeId mn = nd.msc_node;
      const auto sit = survivors.find(mn);
      const msseg::NodeId sv = (sit != survivors.end()) ? sit->second : mn;
      int rid;
      const auto rit = region_of_survivor.find(sv);
      if (rit == region_of_survivor.end()) {
        rid = next_region++;
        region_of_survivor[sv] = rid;
        rvox.push_back(0);
      } else {
        rid = rit->second;
      }
      region_of_min[mn] = rid;
      rvox[static_cast<std::size_t>(rid)] += nd.voxel_count;
    } else {
      for (auto c = nd.children.rbegin(); c != nd.children.rend(); ++c) stack.push_back(*c);
    }
  }
  if (region_voxels) *region_voxels = std::move(rvox);
  return region_of_min;
}

std::unordered_map<msseg::NodeId, msseg::NodeId> absorb_above_cut_minima(
    const msseg::MscGraph& graph, const MergeTree& tree, float cut_threshold,
    float select_persistence) {
  // Stage-8 partition: each min -> its region's deepest minimum (the survivor).
  std::unordered_map<msseg::NodeId, msseg::NodeId> survivor =
      branch_survivors(tree, cut_threshold, select_persistence);

  // Per-region voxel sums (keyed by survivor) + per-min value, from the leaves.
  std::unordered_map<msseg::NodeId, std::int64_t> region_vox;
  std::unordered_map<msseg::NodeId, float> min_value;
  for (const MergeNode& n : tree.nodes) {
    if (!n.is_leaf) continue;
    const auto it = survivor.find(n.msc_node);
    const msseg::NodeId sv = (it != survivor.end()) ? it->second : n.msc_node;
    region_vox[sv] += n.voxel_count;
    min_value[n.msc_node] = n.value;
  }
  // Background = the heaviest region.
  msseg::NodeId background = -1;
  std::int64_t best = -1;
  for (const auto& [sv, v] : region_vox)
    if (v > best) { best = v; background = sv; }

  const auto below_cut = [&](msseg::NodeId m) {
    const auto it = min_value.find(m);
    return it != min_value.end() && it->second < cut_threshold;
  };
  // A cell = a non-background region whose deepest minimum (survivor) is below
  // the cut, i.e. contains a genuine below-cut minimum.
  const auto is_cell = [&](msseg::NodeId sv) { return sv != background && below_cut(sv); };

  // Min adjacency from living 1-saddles: distinct downward minima are pairwise
  // adjacent with weight = the saddle value.
  std::unordered_map<msseg::NodeId, std::vector<std::pair<msseg::NodeId, float>>> adj;
  for (const auto& node : graph.nodes) {
    if (node.index_dim != 1) continue;
    std::vector<msseg::NodeId> mins;
    for (const msseg::NodeId aid : graph.adjacency[static_cast<std::size_t>(node.id)]) {
      const auto& a = graph.arcs[static_cast<std::size_t>(aid)];
      if (a.upper == node.id && graph.nodes[static_cast<std::size_t>(a.lower)].index_dim == 0) {
        if (std::find(mins.begin(), mins.end(), a.lower) == mins.end()) mins.push_back(a.lower);
      }
    }
    for (std::size_t i = 0; i < mins.size(); ++i)
      for (std::size_t j = i + 1; j < mins.size(); ++j) {
        adj[mins[i]].push_back({mins[j], node.value});
        adj[mins[j]].push_back({mins[i], node.value});
      }
  }

  // Marker-based watershed flood: markers are cell cores (below-cut minima of
  // cell regions); grow only into above-cut minima, lowest connecting saddle
  // first, so each joins the lowest reachable non-background cell.
  struct Front { float w; msseg::NodeId to; msseg::NodeId rep; };
  const auto greater = [](const Front& a, const Front& b) { return a.w > b.w; };
  std::priority_queue<Front, std::vector<Front>, decltype(greater)> pq(greater);

  std::unordered_map<msseg::NodeId, msseg::NodeId> assigned;
  const auto push_neighbors = [&](msseg::NodeId m, msseg::NodeId rep) {
    const auto it = adj.find(m);
    if (it == adj.end()) return;
    for (const auto& [nb, w] : it->second)
      if (!below_cut(nb) && !assigned.count(nb)) pq.push({w, nb, rep});
  };
  for (const auto& [m, sv] : survivor)
    if (below_cut(m) && is_cell(sv)) {  // fixed cell core; seed its above-cut boundary
      assigned[m] = sv;
      push_neighbors(m, sv);
    }
  while (!pq.empty()) {
    const Front f = pq.top();
    pq.pop();
    if (assigned.count(f.to)) continue;  // claimed by a lower saddle already
    assigned[f.to] = f.rep;
    push_neighbors(f.to, f.rep);
  }

  // Reassigned above-cut minima override; every other min keeps its cut region.
  std::unordered_map<msseg::NodeId, msseg::NodeId> result = std::move(survivor);
  for (const auto& [m, rep] : assigned) result[m] = rep;
  return result;
}

}  // namespace cellseg
