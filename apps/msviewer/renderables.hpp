#pragma once

#include "msseg/graph/msc_graph.hpp"
#include "msseg/workflow/pipeline.hpp"

#include "glut_crystal_ball_viewer.h"   // from libs/render (msrender)

namespace msviewer {

// Read + parse a JSON workflow file into WorkflowParams.
msseg::WorkflowParams load_workflow(const char* path);

// Draws Morse-Smale critical points as colored points (by Morse index:
// 0=min blue, 1=green, 2=yellow, 3=max red).
class CriticalPointsRenderable : public RenderableGL {
 public:
  explicit CriticalPointsRenderable(msseg::MscGraph graph);
  void toggle() { on_ = !on_; }
  void Render() override;
  void BBox(Vector4& vmin, Vector4& vmax) override;

 private:
  msseg::MscGraph graph_;
  bool on_ = true;
};

// Draws MS-complex arcs as line segments between their endpoint nodes.
class ArcsRenderable : public RenderableGL {
 public:
  explicit ArcsRenderable(msseg::MscGraph graph);
  void toggle() { on_ = !on_; }
  void Render() override;
  void BBox(Vector4& vmin, Vector4& vmax) override;

 private:
  msseg::MscGraph graph_;
  bool on_ = true;
};

}  // namespace msviewer
