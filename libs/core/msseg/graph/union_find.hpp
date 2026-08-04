#pragma once

#include <numeric>
#include <utility>
#include <vector>

namespace msseg {

// Minimal disjoint-set (union by rank, path halving). Payload bookkeeping
// (voxel sums, active tree node, ...) is kept by the caller, keyed on find().
class UnionFind {
 public:
  explicit UnionFind(int n) : parent_(n), rank_(n, 0) {
    std::iota(parent_.begin(), parent_.end(), 0);
  }

  int find(int x) {
    while (parent_[x] != x) {
      parent_[x] = parent_[parent_[x]];  // halving
      x = parent_[x];
    }
    return x;
  }

  // Unite the sets of a and b; returns the new representative (or the shared one
  // if they were already in the same set).
  int unite(int a, int b) {
    int ra = find(a), rb = find(b);
    if (ra == rb) return ra;
    if (rank_[ra] < rank_[rb]) std::swap(ra, rb);
    parent_[rb] = ra;
    if (rank_[ra] == rank_[rb]) ++rank_[ra];
    return ra;
  }

 private:
  std::vector<int> parent_;
  std::vector<int> rank_;
};

}  // namespace msseg
