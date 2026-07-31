#include "mscoupon/matcher.hpp"

#include <algorithm>
#include <limits>

namespace mscoupon {

int SliceMatcher::find(int x) {
  while (parent_[x] != x) {
    parent_[x] = parent_[parent_[x]];  // path halving
    x = parent_[x];
  }
  return x;
}

void SliceMatcher::unite(int a, int b) {
  a = find(a);
  b = find(b);
  if (a == b) return;
  if (rank_[a] < rank_[b]) std::swap(a, b);
  parent_[b] = a;
  if (rank_[a] == rank_[b]) ++rank_[a];
}

int SliceMatcher::node_for(int slice_index, int segment_id) const {
  const auto it = index_.find(key_of(slice_index, segment_id));
  return it == index_.end() ? -1 : it->second;
}

void SliceMatcher::add_slice(const std::vector<int>& labels, int width, int height,
                             const std::unordered_set<int>& keep_ids, const std::vector<SegmentStat>& rows,
                             int slice_index) {
  // 1. Create a node for each kept segment and store its stat payload.
  for (const auto& row : rows) {
    if (keep_ids.count(row.segment_id) == 0) continue;
    const std::int64_t key = key_of(slice_index, row.segment_id);
    if (index_.count(key) != 0) continue;  // defensive: ids are unique per slice
    const int node = static_cast<int>(payload_.size());
    index_.emplace(key, node);
    parent_.push_back(node);
    rank_.push_back(0);
    NodePayload p;
    p.slice_index = slice_index;
    p.segment_id = row.segment_id;
    p.area = row.area;
    p.sum_value = static_cast<double>(row.mean_value) * static_cast<double>(row.area);
    p.min_value = row.min_value;
    p.max_value = row.max_value;
    p.min_x = row.min_x;
    p.min_y = row.min_y;
    p.max_x = row.max_x;
    p.max_y = row.max_y;
    payload_.push_back(p);
  }

  // 2. Link to the previously added slice by 26-neighbor connectivity (a 3x3
  //    stencil into the prior slice). Requires matching dimensions.
  if (prev_slice_index_ >= 0 && prev_width_ == width && prev_height_ == height &&
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height) == labels.size()) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int local = labels[static_cast<std::size_t>(y) * width + x];
        if (keep_ids.count(local) == 0) continue;
        const int cur_node = node_for(slice_index, local);
        if (cur_node < 0) continue;
        for (int dy = -1; dy <= 1; ++dy) {
          const int ny = y + dy;
          if (ny < 0 || ny >= height) continue;
          for (int dx = -1; dx <= 1; ++dx) {
            const int nx = x + dx;
            if (nx < 0 || nx >= width) continue;
            const int prior = prev_labels_[static_cast<std::size_t>(ny) * width + nx];
            if (prev_keep_.count(prior) == 0) continue;
            const int prior_node = node_for(prev_slice_index_, prior);
            if (prior_node >= 0) unite(cur_node, prior_node);
          }
        }
      }
    }
  }

  // 3. Retain this slice as the "prior" for the next add_slice.
  prev_labels_ = labels;
  prev_width_ = width;
  prev_height_ = height;
  prev_slice_index_ = slice_index;
  prev_keep_ = keep_ids;
}

void SliceMatcher::finalize(std::vector<FeatureMapRow>& map_out,
                            std::vector<GlobalFeatureStat>& table_out) {
  map_out.clear();
  table_out.clear();
  const int n = static_cast<int>(payload_.size());
  if (n == 0) return;

  // Deterministic global-id assignment: walk nodes in (slice, id) order and give
  // each newly-seen root the next global id.
  std::vector<int> order(n);
  for (int i = 0; i < n; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (payload_[a].slice_index != payload_[b].slice_index)
      return payload_[a].slice_index < payload_[b].slice_index;
    return payload_[a].segment_id < payload_[b].segment_id;
  });

  std::vector<int> global_of_root(n, -1);
  int next_gid = 0;
  for (const int node : order) {
    const int root = find(node);
    if (global_of_root[root] < 0) global_of_root[root] = next_gid++;
  }

  // Aggregate node payloads into their global feature; count distinct slices.
  struct Accum {
    std::size_t voxel_count = 0;
    double sum_value = 0.0;
    float min_value = std::numeric_limits<float>::max();
    float max_value = std::numeric_limits<float>::lowest();
    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::lowest();
    int max_y = std::numeric_limits<int>::lowest();
    int min_z = std::numeric_limits<int>::max();
    int max_z = std::numeric_limits<int>::lowest();
    std::unordered_set<int> slices;
  };
  std::vector<Accum> accums(next_gid);

  map_out.reserve(n);
  for (const int node : order) {
    const int gid = global_of_root[find(node)];
    const NodePayload& p = payload_[node];
    map_out.push_back(FeatureMapRow{p.slice_index, p.segment_id, gid});

    Accum& a = accums[gid];
    a.voxel_count += p.area;
    a.sum_value += p.sum_value;
    a.min_value = std::min(a.min_value, p.min_value);
    a.max_value = std::max(a.max_value, p.max_value);
    a.min_x = std::min(a.min_x, p.min_x);
    a.min_y = std::min(a.min_y, p.min_y);
    a.max_x = std::max(a.max_x, p.max_x);
    a.max_y = std::max(a.max_y, p.max_y);
    a.min_z = std::min(a.min_z, p.slice_index);
    a.max_z = std::max(a.max_z, p.slice_index);
    a.slices.insert(p.slice_index);
  }

  table_out.reserve(next_gid);
  for (int gid = 0; gid < next_gid; ++gid) {
    const Accum& a = accums[gid];
    GlobalFeatureStat s;
    s.global_id = gid;
    s.voxel_count = a.voxel_count;
    s.num_slices = static_cast<int>(a.slices.size());
    s.first_slice = a.min_z;
    s.last_slice = a.max_z;
    s.min_x = a.min_x;
    s.min_y = a.min_y;
    s.min_z = a.min_z;
    s.max_x = a.max_x;
    s.max_y = a.max_y;
    s.max_z = a.max_z;
    s.min_value = a.min_value;
    s.max_value = a.max_value;
    s.mean_value = a.voxel_count > 0 ? static_cast<float>(a.sum_value / static_cast<double>(a.voxel_count)) : 0.0f;
    table_out.push_back(s);
  }

  std::sort(table_out.begin(), table_out.end(), [](const GlobalFeatureStat& a, const GlobalFeatureStat& b) {
    if (a.voxel_count != b.voxel_count) return a.voxel_count > b.voxel_count;
    return a.global_id < b.global_id;
  });
}

}  // namespace mscoupon
