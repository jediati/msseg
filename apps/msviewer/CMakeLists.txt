# ---------------------------------------------------------------------------
# msviewer: one-off OpenGL debug viewer (Windows x64 only).
#
# The copied MSCapps sources are 2000s-era C++ that does not survive MSVC's
# /permissive- (implied by /std:c++20). The whole viewer is therefore built as
# C++17, which does NOT force /permissive-. Linking a C++17 target against the
# C++20 msseg_core static lib is fine (same compiler/ABI).
# ---------------------------------------------------------------------------

# The reused MSCapps crystal-ball / VSVR2 / glsl / Matrix4 code.
add_library(msviewer_legacy STATIC
  third_party/mscapps/glut_crystal_ball_viewer.cpp
  third_party/mscapps/vsvr2.cpp
  third_party/mscapps/glsl.cpp
  third_party/mscapps/Matrix4.cpp
)
set_target_properties(msviewer_legacy PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_include_directories(msviewer_legacy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mscapps)
target_compile_definitions(msviewer_legacy PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(msviewer_legacy PUBLIC freeglut::freeglut glew::glew opengl32 glu32)

# The app: pipeline glue + renderables. Built C++17 to match the legacy headers
# it includes; msseg headers are C++17-compatible.
add_executable(msviewer main.cpp renderables.cpp)
set_target_properties(msviewer PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_include_directories(msviewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(msviewer PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(msviewer PRIVATE msseg_core msseg_io msviewer_legacy)

# Ship the runtime DLLs and the transfer function next to the exe.
add_custom_command(TARGET msviewer POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:freeglut::freeglut> $<TARGET_FILE_DIR:msviewer>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:glew::glew> $<TARGET_FILE_DIR:msviewer>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/assets/transfer.tf $<TARGET_FILE_DIR:msviewer>
  VERBATIM
)
