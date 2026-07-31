# ---------------------------------------------------------------------------
# msviewer: generic core-level OpenGL debug viewer (Windows x64 only).
#
# Loads a raw float32 volume, optionally runs the core msseg::Pipeline over a
# workflow.json, and renders the volume + Morse-Smale critical points + arcs.
# Instance-agnostic (works over msseg_core / MscGraph); the reusable renderer
# engine lives in libs/render (msrender). A native desktop app, never pip-shipped.
#
# Built C++17 to match the msrender legacy headers it includes; msseg headers
# are C++17-compatible.
# ---------------------------------------------------------------------------
if(NOT TARGET msrender)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../libs/render ${CMAKE_BINARY_DIR}/_msrender)
endif()

add_executable(msviewer main.cpp renderables.cpp)
set_target_properties(msviewer PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_include_directories(msviewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(msviewer PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(msviewer PRIVATE msseg_core msseg_io msrender)

# Ship the runtime DLLs and the transfer function next to the exe.
add_custom_command(TARGET msviewer POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:freeglut::freeglut> $<TARGET_FILE_DIR:msviewer>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:glew::glew> $<TARGET_FILE_DIR:msviewer>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/assets/transfer.tf $<TARGET_FILE_DIR:msviewer>
  VERBATIM
)
