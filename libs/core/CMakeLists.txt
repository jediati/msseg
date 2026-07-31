# ---------------------------------------------------------------------------
# msseg_core : portable Morse-Smale segmentation workflow library.
# Builds on Windows (VS2022), Linux, and HPC. Depends only on diffg,
# nlohmann_json, OpenMP, and (privately) MSCEER's GInt / msc_2d_lib.
#
# This directory is add_subdirectory()'d by every package that needs the core
# (and by the dev root build). The guard makes those inclusions idempotent so
# the targets are defined exactly once per CMake project.
# ---------------------------------------------------------------------------
if(NOT TARGET msseg_core)

add_library(msseg_core STATIC
  msseg/filter/filter_stage.cpp
  msseg/compute/msc2d.cpp
  msseg/compute/msc3d.cpp
  msseg/graph/graph_ops.cpp
  msseg/segment/registry.cpp
  msseg/io/raw_io.cpp
  msseg/workflow/pipeline.cpp
)

# Headers are included as "msseg/..." -> expose this directory publicly.
target_include_directories(msseg_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(msseg_core
  PUBLIC
    diffg::diffg
    nlohmann_json::nlohmann_json
  PRIVATE
    # GInt (header stack) is used only by msc3d.cpp; msc_2d_lib by msc2d.cpp.
    # Keeping them PRIVATE walls the C++11-era MSCEER headers off from every
    # other TU and from the python ABI.
    GInt
    msc_2d_lib
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(msseg_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# msseg_io : portable image-format I/O layer (isolates the TinyTIFF dep).
# ---------------------------------------------------------------------------
add_library(msseg_io STATIC
  msseg/io/tiff_io.cpp
)
target_link_libraries(msseg_io
  PUBLIC
    msseg_core
    ${MSSEG_TINYTIFF_TARGETS}
)

endif()  # NOT TARGET msseg_core
