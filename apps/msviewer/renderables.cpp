#include "renderables.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

namespace msviewer {

msseg::WorkflowParams load_workflow(const char* path) {
  std::ifstream in(path);
  if (!in.good()) throw std::runtime_error(std::string("cannot open workflow: ") + path);
  nlohmann::json spec;
  in >> spec;
  return msseg::parse_workflow(spec);
}

namespace {

void color_for_index(int dim, float& r, float& g, float& b) {
  switch (dim) {
    case 0: r = 0.2f; g = 0.4f; b = 1.0f; break;  // minimum   - blue
    case 1: r = 0.2f; g = 1.0f; b = 0.3f; break;  // 1-saddle  - green
    case 2: r = 1.0f; g = 0.9f; b = 0.2f; break;  // 2-saddle  - yellow
    default: r = 1.0f; g = 0.2f; b = 0.2f; break; // maximum   - red
  }
}

void accumulate_bbox(const msseg::MscGraph& g, Vector4& vmin, Vector4& vmax) {
  if (g.nodes.empty()) {
    vmin = Vector4(0, 0, 0, 0);
    vmax = Vector4(1, 1, 1, 0);
    return;
  }
  vmin = Vector4(g.nodes[0].pos[0], g.nodes[0].pos[1], g.nodes[0].pos[2], 0);
  vmax = vmin;
  for (const auto& n : g.nodes) {
    const Vector4 p(n.pos[0], n.pos[1], n.pos[2], 0);
    vmin = Vector4::piecewiseMin(vmin, p);
    vmax = Vector4::piecewiseMax(vmax, p);
  }
}

}  // namespace

CriticalPointsRenderable::CriticalPointsRenderable(msseg::MscGraph graph) : graph_(std::move(graph)) {}

void CriticalPointsRenderable::Render() {
  if (!on_) return;
  glPushAttrib(GL_ALL_ATTRIB_BITS);
  glDisable(GL_LIGHTING);
  glPointSize(7.0f);
  glBegin(GL_POINTS);
  for (const auto& n : graph_.nodes) {
    float r, g, b;
    color_for_index(n.index_dim, r, g, b);
    glColor3f(r, g, b);
    glVertex3f(n.pos[0], n.pos[1], n.pos[2]);
  }
  glEnd();
  glPopAttrib();
}

void CriticalPointsRenderable::BBox(Vector4& vmin, Vector4& vmax) { accumulate_bbox(graph_, vmin, vmax); }

ArcsRenderable::ArcsRenderable(msseg::MscGraph graph) : graph_(std::move(graph)) {}

void ArcsRenderable::Render() {
  if (!on_) return;
  glPushAttrib(GL_ALL_ATTRIB_BITS);
  glDisable(GL_LIGHTING);
  glLineWidth(1.5f);
  glColor3f(0.8f, 0.8f, 0.85f);
  glBegin(GL_LINES);
  for (const auto& a : graph_.arcs) {
    if (a.lower < 0 || a.upper < 0) continue;
    const auto& lo = graph_.nodes[static_cast<std::size_t>(a.lower)];
    const auto& hi = graph_.nodes[static_cast<std::size_t>(a.upper)];
    glVertex3f(lo.pos[0], lo.pos[1], lo.pos[2]);
    glVertex3f(hi.pos[0], hi.pos[1], hi.pos[2]);
  }
  glEnd();
  glPopAttrib();
}

void ArcsRenderable::BBox(Vector4& vmin, Vector4& vmax) { accumulate_bbox(graph_, vmin, vmax); }

}  // namespace msviewer
