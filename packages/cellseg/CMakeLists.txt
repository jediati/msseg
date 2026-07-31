# cellseg package: the 3D fluorescent-membrane cell segmentation instance, built
# as its own independently pip-installable distribution ("msseg-cellseg").
#
# Works standalone (pip / scikit-build-core) by bootstrapping the shared deps +
# libs/core, and as an add_subdirectory() from the dev root (bootstrap skipped
# because msseg_core already exists).
cmake_minimum_required(VERSION 3.21)

if(NOT TARGET msseg_core)
  project(cellseg LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  option(MSSEG_BUILD_PYTHON "Build python bindings" OFF)
  option(MSSEG_BUILD_TESTS "Build tests" OFF)

  get_filename_component(MSSEG_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
  list(APPEND CMAKE_MODULE_PATH "${MSSEG_REPO_ROOT}/cmake")

  find_package(OpenMP QUIET)
  include(Dependencies)
  add_subdirectory(${MSSEG_REPO_ROOT}/libs/core ${CMAKE_BINARY_DIR}/_msseg_core)
endif()

include(AddInstance)

add_msseg_instance(cellseg
  PYTHON
  NEEDS_IO
  LIB_SOURCES
    lib/cellseg/config.cpp
    lib/cellseg/heavy_lift.cpp
    lib/cellseg/merge_tree.cpp
    lib/cellseg/segment.cpp
    lib/cellseg/cell_pipeline.cpp
    lib/cellseg/cellseg_cli.cpp
)

if(MSSEG_BUILD_TESTS)
  enable_testing()
  add_executable(cellseg_tests tests/cellseg_tests.cpp)
  target_link_libraries(cellseg_tests PRIVATE cellseg_lib)
  add_test(NAME cellseg_tests COMMAND cellseg_tests)
endif()
