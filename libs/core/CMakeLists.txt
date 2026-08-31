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
  msseg/graph/merge_tree.cpp
  msseg/segment/registry.cpp
  msseg/io/raw_io.cpp
  msseg/workflow/pipeline.cpp
  msseg/workflow/stat_channels.cpp
)

# GPU statistics accumulation: the real CUDA module with MSSEG_GPU=ON, a stub
# otherwise -- same always-a-target pattern as diffg_gpu, so msc2d.cpp calls
# the API unconditionally and falls back at runtime.
if(MSSEG_GPU)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Same default as MSCEER/diffg: the GPU in this machine.
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  target_sources(msseg_core PRIVATE msseg/compute/gpu/stats_gpu.cu)
  # The target predates the CUDA language enablement, so its architecture
  # property snapshot is empty -- stamp it explicitly.
  set_property(TARGET msseg_core PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
else()
  target_sources(msseg_core PRIVATE msseg/compute/gpu/stats_gpu_stub.cpp)
endif()

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

# diffg's GPU filter bank (plain C++ boundary; a stub target when CUDA is off,
# absent entirely on pre-GPU diffg pins). msc2d.cpp drives it for
# device-resident measurement channels when MSSEG_HAVE_DIFFG_GPU is defined.
if(TARGET diffg_gpu)
  target_link_libraries(msseg_core PRIVATE diffg_gpu)
  target_compile_definitions(msseg_core PRIVATE MSSEG_HAVE_DIFFG_GPU)
endif()

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
