#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace msseg {

// The SEAM GRAPH of a 2D label raster: the polylines along which regions meet.
//
// Vocabulary (see docs/seam_labeling.md):
//   corner    a pixel-corner lattice point at integer coordinates; pixel (ix, iy)
//             covers [ix, ix+1) x [iy, iy+1), so the lattice is (w+1) x (h+1).
//   crack     the unit lattice step between two corners that separates two
//             differently labelled pixels (both >= 0; a -1 flank is background
//             and makes no crack). Its id is label-independent:
//             2*c for the horizontal step from corner c = iy*(w+1)+ix to
//             (ix+1, iy), 2*c+1 for the vertical step to (ix, iy+1).
//   junction  a corner where >= 3 regions meet (degree >= 3), or where a seam
//             ends (degree 1), or a degree-2 corner whose two cracks separate
//             different flank pairs.
//   seam      a maximal chain of cracks between the same two regions (the
//             FLANKS, a < b), junction to junction; a LOOP when it closes on
//             itself with no junction (an island), in which case j0 == j1 == -1
//             and its point list repeats the first corner at the end.
//
// The output is canonical -- independent of the traversal order -- so the
// numpy reference (msseg.labeler.seams.seams_from_labels) reproduces it array
// for array: an open seam is oriented so (j0, j1) is minimal (and, for a seam
// from a junction back to itself, so its second corner is the smaller in
// (y, x)); a loop starts at its smallest (y, x) corner and leaves it in +x;
// seams are sorted by (a, b, j0, j1, first corner, second corner, length).
// Junction ids are ranks in raster (y, x) order.
struct SeamGraph {
  std::vector<std::int32_t> a, b;          // flanks, a < b                    [S]
  std::vector<std::int32_t> j0, j1;        // junction ids, -1 for a loop      [S]
  std::vector<std::int64_t> offsets;       // points of seam i: [offsets[i], offsets[i+1])  [S+1]
  std::vector<std::int32_t> points;        // corner (x, y) pairs, flattened  [2P]
  std::vector<std::int32_t> junction_xy;   // corner (x, y) per junction id   [2J]

  std::size_t n_seams() const { return a.size(); }
  std::size_t n_junctions() const { return junction_xy.size() / 2; }
  std::int64_t n_points() const { return offsets.empty() ? 0 : offsets.back(); }
};

// labels: row-major int32 (h, w), -1 = background. O(h*w) time and memory
// (one byte per corner + one bit per lattice edge of scratch). No MSCEER.
SeamGraph extract_seam_graph(const std::int32_t* labels, std::size_t h, std::size_t w);

}  // namespace msseg
