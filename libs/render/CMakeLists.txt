# ---------------------------------------------------------------------------
# msrender : shared, instance-agnostic OpenGL 3D renderer (Windows x64, desktop).
#
# The reusable crystal-ball / VSVR2 / glsl / Matrix4 engine (formerly the
# msviewer "legacy" lib). Any app that wants custom 3D visualization links this
# and subclasses its RenderableGL/SceneGL/VolumeRenderableGL. Optional and
# non-portable: only built when MSSEG_BUILD_VIEWER is ON and never enters a wheel.
#
# The 2000s-era MSCapps sources do not survive MSVC's /permissive- (implied by
# /std:c++20), so this target is C++17. Linking C++17 against the C++20
# msseg_core is fine (same compiler/ABI).
# ---------------------------------------------------------------------------
if(NOT TARGET msrender)

include(VendoredGL)   # freeglut::freeglut, glew::glew imported targets (ext/win64)

add_library(msrender STATIC
  msrender/glut_crystal_ball_viewer.cpp
  msrender/vsvr2.cpp
  msrender/glsl.cpp
  msrender/Matrix4.cpp
)
set_target_properties(msrender PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_include_directories(msrender PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/msrender)
target_compile_definitions(msrender PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(msrender PUBLIC freeglut::freeglut glew::glew opengl32 glu32)

endif()  # NOT TARGET msrender
