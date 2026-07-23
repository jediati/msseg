// msviewer: one-off OpenGL debug viewer for the MSSeg pipeline (M6).
//
//   msviewer <volume.raw> [workflow.json]
//
// Loads a raw float32 volume (with a "<volume.raw>.dat" dimension sidecar) into
// the reused MSCapps crystal-ball / VSVR2 volume renderer. If a workflow.json
// is given, the MSSeg pipeline is run and its critical points, arcs, and basin
// overlay are added to the scene (renderables.hpp).
//
// Runtime files expected in the working directory: transfer.tf (copied next to
// the exe at build time). Run from the exe directory.
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "msseg/io/raw_io.hpp"
#include "msseg/volume/types.hpp"
#include "msseg/workflow/pipeline.hpp"

#include "third_party/mscapps/glut_crystal_ball_viewer.h"

#include "renderables.hpp"

namespace {

// Scene state kept alive for the duration of the GLUT main loop.
msseg::Volume g_volume;
VolumeRenderableGL* g_vol = nullptr;
SceneGL* g_scene = nullptr;
msviewer::CriticalPointsRenderable* g_cps = nullptr;
msviewer::ArcsRenderable* g_arcs = nullptr;

bool on_key(unsigned char key, int, int) {
  switch (key) {
    case 'v': g_vol->ToggleRender(); break;
    case 'c': if (g_cps) g_cps->toggle(); break;
    case 'a': if (g_arcs) g_arcs->toggle(); break;
    case '-': g_vol->SetNumberOfSlices(150); break;
    case '=': g_vol->SetNumberOfSlices(400); break;
    default: return false;  // let the crystal-ball defaults handle it
  }
  CRYSTAL_BALL_VIEWER::Redisplay();
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: msviewer <volume.raw> [workflow.json]\n");
    return 1;
  }
  try {
    g_volume = msseg::read_raw_volume(argv[1]);
    const int X = static_cast<int>(g_volume.dims().width);
    const int Y = static_cast<int>(g_volume.dims().height);
    const int Z = static_cast<int>(g_volume.dims().depth);
    std::printf("msviewer: loaded volume %dx%dx%d\n", X, Y, Z);

    // GL context must exist before VolumeRenderableGL::Initialize.
    CRYSTAL_BALL_VIEWER::Initialize_GLUT(argc, argv);

    g_vol = new VolumeRenderableGL();
    g_vol->Initialize(X, Y, Z, g_volume.data(), "transfer.tf");
    g_vol->EnableRender();
    g_vol->SetNumberOfSlices(300);

    g_scene = new SceneGL();
    g_scene->AddRenderable(g_vol);

    // Optional: run the pipeline and overlay its topology.
    if (argc >= 3) {
      std::printf("msviewer: running workflow %s\n", argv[2]);
      const msseg::WorkflowParams params = msviewer::load_workflow(argv[2]);
      const msseg::WorkflowResult result = msseg::Pipeline::run(g_volume, params);
      std::printf("msviewer: MSC nodes=%zu arcs=%zu\n", result.graph.nodes.size(), result.graph.arcs.size());
      g_cps = new msviewer::CriticalPointsRenderable(result.graph);
      g_arcs = new msviewer::ArcsRenderable(result.graph);
      g_scene->AddRenderable(g_arcs);
      g_scene->AddRenderable(g_cps);
    }

    CRYSTAL_BALL_VIEWER::SetScene(g_scene);
    CRYSTAL_BALL_VIEWER::SetGraphicKeysFunction(on_key);
    std::printf("msviewer: keys - v:volume c:crit-points a:arcs =/-:slices  (r:reset, mouse:orbit/pan/zoom)\n");
    CRYSTAL_BALL_VIEWER::StartGlutMainLoop();
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "msviewer error: %s\n", ex.what());
    return 1;
  }
}
