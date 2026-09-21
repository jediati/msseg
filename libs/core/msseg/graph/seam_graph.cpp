#include "msseg/graph/seam_graph.hpp"

#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>

namespace msseg {

namespace {

using Corner = std::int64_t;

// Lattice geometry over a label raster. Cracks are recomputed from the two
// pixels beside them whenever needed rather than stored: the raster IS the
// crack table, and a label read is cheaper than a side structure.
struct Lattice {
  const std::int32_t* lab;
  std::int64_t w, h, cw;   // cw = w + 1 corners per row

  std::int32_t px(std::int64_t ix, std::int64_t iy) const {
    if (ix < 0 || iy < 0 || ix >= w || iy >= h) return -1;
    return lab[iy * w + ix];
  }
  // Horizontal crack from corner (ix, iy) to (ix+1, iy): between the pixel
  // above (ix, iy-1) and the pixel below (ix, iy).
  bool hcrack(std::int64_t ix, std::int64_t iy, std::int32_t* lo, std::int32_t* hi) const {
    if (ix < 0 || ix >= w) return false;
    const std::int32_t p = px(ix, iy - 1), q = px(ix, iy);
    if (p < 0 || q < 0 || p == q) return false;
    if (lo) *lo = std::min(p, q);
    if (hi) *hi = std::max(p, q);
    return true;
  }
  // Vertical crack from corner (ix, iy) to (ix, iy+1): between the pixel to
  // the left (ix-1, iy) and the pixel to the right (ix, iy).
  bool vcrack(std::int64_t ix, std::int64_t iy, std::int32_t* lo, std::int32_t* hi) const {
    if (iy < 0 || iy >= h) return false;
    const std::int32_t p = px(ix - 1, iy), q = px(ix, iy);
    if (p < 0 || q < 0 || p == q) return false;
    if (lo) *lo = std::min(p, q);
    if (hi) *hi = std::max(p, q);
    return true;
  }
  // Direction d in {0: +x, 1: +y, 2: -x, 3: -y} out of corner (ix, iy).
  bool crack(std::int64_t ix, std::int64_t iy, int d, std::int32_t* lo, std::int32_t* hi) const {
    switch (d) {
      case 0: return hcrack(ix, iy, lo, hi);
      case 1: return vcrack(ix, iy, lo, hi);
      case 2: return hcrack(ix - 1, iy, lo, hi);
      default: return vcrack(ix, iy - 1, lo, hi);
    }
  }
  Corner corner(std::int64_t ix, std::int64_t iy) const { return iy * cw + ix; }
  Corner step(std::int64_t ix, std::int64_t iy, int d) const {
    switch (d) {
      case 0: return corner(ix + 1, iy);
      case 1: return corner(ix, iy + 1);
      case 2: return corner(ix - 1, iy);
      default: return corner(ix, iy - 1);
    }
  }
  // Label-independent crack id of the step out of (ix, iy) in direction d.
  std::int64_t crack_id(std::int64_t ix, std::int64_t iy, int d) const {
    switch (d) {
      case 0: return 2 * corner(ix, iy);
      case 1: return 2 * corner(ix, iy) + 1;
      case 2: return 2 * corner(ix - 1, iy);
      default: return 2 * corner(ix, iy - 1) + 1;
    }
  }
};

constexpr std::uint8_t kJunction = 0x80;   // flag bit in the per-corner degree byte

struct Seam {
  std::int32_t a, b, j0, j1;
  std::vector<std::int32_t> pts;   // x, y, x, y, ...
};

inline std::pair<std::int32_t, std::int32_t> yx(const std::vector<std::int32_t>& pts, std::size_t i) {
  return {pts[2 * i + 1], pts[2 * i]};
}

void reverse_points(std::vector<std::int32_t>& pts) {
  const std::size_t n = pts.size() / 2;
  for (std::size_t i = 0; i < n / 2; ++i) {
    std::swap(pts[2 * i], pts[2 * (n - 1 - i)]);
    std::swap(pts[2 * i + 1], pts[2 * (n - 1 - i) + 1]);
  }
}

// A loop's closed point list (first == last) rotated to start at its smallest
// (y, x) corner and to leave it in +x. That corner's only possible loop
// neighbours are (x+1, y) and (x, y+1) -- anything else would be smaller --
// so "+x first" is always well defined.
void canonicalize_loop(std::vector<std::int32_t>& pts) {
  const std::size_t n = pts.size() / 2 - 1;   // distinct corners
  std::size_t best = 0;
  for (std::size_t i = 1; i < n; ++i)
    if (yx(pts, i) < yx(pts, best)) best = i;
  std::vector<std::int32_t> out;
  out.reserve(pts.size());
  for (std::size_t k = 0; k <= n; ++k) {
    const std::size_t i = (best + k) % n;
    out.push_back(pts[2 * i]);
    out.push_back(pts[2 * i + 1]);
  }
  // out[1] is the second corner; want it at (x+1, y).
  if (!(out[2] == out[0] + 1 && out[3] == out[1])) reverse_points(out);
  pts.swap(out);
}

}  // namespace

SeamGraph extract_seam_graph(const std::int32_t* labels, std::size_t h, std::size_t w) {
  SeamGraph out;
  out.offsets.push_back(0);
  if (h == 0 || w == 0 || labels == nullptr) return out;

  const Lattice L{labels, static_cast<std::int64_t>(w), static_cast<std::int64_t>(h),
                  static_cast<std::int64_t>(w) + 1};
  const std::int64_t ch = L.h + 1;
  const std::int64_t n_corners = L.cw * ch;

  // 1-3. Degree and junction flag per corner, junction list in raster order.
  std::vector<std::uint8_t> deg(static_cast<std::size_t>(n_corners), 0);
  std::vector<Corner> junction_corner;
  for (std::int64_t iy = 0; iy < ch; ++iy) {
    for (std::int64_t ix = 0; ix < L.cw; ++ix) {
      int d = 0;
      std::int64_t key_min = -1, key_max = -1;
      bool first = true;
      for (int dir = 0; dir < 4; ++dir) {
        std::int32_t lo, hi;
        if (!L.crack(ix, iy, dir, &lo, &hi)) continue;
        ++d;
        const std::int64_t key = (static_cast<std::int64_t>(lo) << 32) | static_cast<std::uint32_t>(hi);
        if (first) {
          key_min = key_max = key;
          first = false;
        } else {
          key_min = std::min(key_min, key);
          key_max = std::max(key_max, key);
        }
      }
      if (d == 0) continue;
      const bool junction = d >= 3 || d == 1 || (d == 2 && key_min != key_max);
      const Corner c = L.corner(ix, iy);
      deg[static_cast<std::size_t>(c)] = static_cast<std::uint8_t>(d) | (junction ? kJunction : 0);
      if (junction) junction_corner.push_back(c);
    }
  }
  const auto is_junction = [&](Corner c) { return (deg[static_cast<std::size_t>(c)] & kJunction) != 0; };
  const auto junction_id = [&](Corner c) -> std::int32_t {
    const auto it = std::lower_bound(junction_corner.begin(), junction_corner.end(), c);
    return static_cast<std::int32_t>(it - junction_corner.begin());
  };

  // 4. Chain cracks into seams. `visited` is one bit per lattice edge.
  std::vector<std::uint8_t> visited(static_cast<std::size_t>((2 * n_corners + 7) / 8), 0);
  const auto seen = [&](std::int64_t id) { return (visited[static_cast<std::size_t>(id >> 3)] >> (id & 7)) & 1; };
  const auto mark = [&](std::int64_t id) { visited[static_cast<std::size_t>(id >> 3)] |= static_cast<std::uint8_t>(1u << (id & 7)); };

  std::vector<Seam> seams;
  // Walk from corner (ix, iy) in direction d until a junction (open seam) or
  // the start corner (loop). Returns the end corner; fills pts and flanks.
  const auto walk = [&](std::int64_t ix, std::int64_t iy, int d, Seam& s, bool loop) -> Corner {
    const Corner start = L.corner(ix, iy);
    L.crack(ix, iy, d, &s.a, &s.b);
    s.pts.push_back(static_cast<std::int32_t>(ix));
    s.pts.push_back(static_cast<std::int32_t>(iy));
    for (;;) {
      mark(L.crack_id(ix, iy, d));
      const Corner nxt = L.step(ix, iy, d);
      ix = nxt % L.cw;
      iy = nxt / L.cw;
      s.pts.push_back(static_cast<std::int32_t>(ix));
      s.pts.push_back(static_cast<std::int32_t>(iy));
      if (is_junction(nxt)) return nxt;
      if (loop && nxt == start) return nxt;
      // Degree-2 interior corner: the one crack that is not the way we came.
      const int back = (d + 2) & 3;
      int found = -1;
      for (int dir = 0; dir < 4; ++dir) {
        if (dir == back) continue;
        if (L.crack(ix, iy, dir, nullptr, nullptr)) {
          found = dir;
          break;
        }
      }
      if (found < 0) return nxt;   // cannot happen (degree 2), defensive
      d = found;
    }
  };

  for (const Corner jc : junction_corner) {
    const std::int64_t ix = jc % L.cw, iy = jc / L.cw;
    for (int d = 0; d < 4; ++d) {
      if (!L.crack(ix, iy, d, nullptr, nullptr)) continue;
      if (seen(L.crack_id(ix, iy, d))) continue;
      Seam s;
      const Corner end = walk(ix, iy, d, s, false);
      s.j0 = junction_id(jc);
      s.j1 = junction_id(end);
      // Canonical orientation.
      bool flip = s.j1 < s.j0;
      if (s.j0 == s.j1) {
        const std::size_t n = s.pts.size() / 2;
        flip = yx(s.pts, n - 2) < yx(s.pts, 1);
      }
      if (flip) {
        reverse_points(s.pts);
        std::swap(s.j0, s.j1);
      }
      seams.push_back(std::move(s));
    }
  }
  // Loops: every crack still unvisited lies on a junction-free cycle.
  for (std::int64_t iy = 0; iy < ch; ++iy) {
    for (std::int64_t ix = 0; ix < L.cw; ++ix) {
      for (int d = 0; d < 2; ++d) {   // +x and +y cover every lattice edge once
        if (!L.crack(ix, iy, d, nullptr, nullptr)) continue;
        if (seen(L.crack_id(ix, iy, d))) continue;
        Seam s;
        walk(ix, iy, d, s, true);
        s.j0 = s.j1 = -1;
        canonicalize_loop(s.pts);
        seams.push_back(std::move(s));
      }
    }
  }

  // 5. Deterministic order.
  std::vector<std::size_t> order(seams.size());
  std::iota(order.begin(), order.end(), 0);
  const auto key = [&](std::size_t i) {
    const Seam& s = seams[i];
    const auto p0 = yx(s.pts, 0);
    const auto p1 = yx(s.pts, 1);
    return std::make_tuple(s.a, s.b, s.j0, s.j1, p0.first, p0.second, p1.first, p1.second,
                           static_cast<std::int64_t>(s.pts.size()));
  };
  std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return key(x) < key(y); });

  out.a.reserve(seams.size());
  out.b.reserve(seams.size());
  out.j0.reserve(seams.size());
  out.j1.reserve(seams.size());
  out.offsets.reserve(seams.size() + 1);
  for (const std::size_t i : order) {
    const Seam& s = seams[i];
    out.a.push_back(s.a);
    out.b.push_back(s.b);
    out.j0.push_back(s.j0);
    out.j1.push_back(s.j1);
    out.points.insert(out.points.end(), s.pts.begin(), s.pts.end());
    out.offsets.push_back(static_cast<std::int64_t>(out.points.size() / 2));
  }
  out.junction_xy.reserve(junction_corner.size() * 2);
  for (const Corner c : junction_corner) {
    out.junction_xy.push_back(static_cast<std::int32_t>(c % L.cw));
    out.junction_xy.push_back(static_cast<std::int32_t>(c / L.cw));
  }
  return out;
}

}  // namespace msseg
