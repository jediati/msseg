# mscoupon package: the 2D TIFF-slice Morse-Smale segmentation instance, built
# as its own independently pip-installable distribution ("msseg-mscoupon").
#
# This CMakeLists works two ways:
#   * standalone  (pip / scikit-build-core, or `cmake -S packages/mscoupon`):
#     bootstraps the shared dependencies + libs/core, then builds this instance.
#   * from the dev root (add_subdirectory): the root already provided the deps
#     and msseg_core, so the bootstrap block is skipped.
cmake_minimum_required(VERSION 3.21)

if(NOT TARGET msseg_core)
  project(mscoupon LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  option(MSSEG_BUILD_PYTHON "Build python bindings" OFF)
  option(MSSEG_BUILD_TESTS "Build tests" OFF)

  get_filename_component(MSSEG_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
  list(APPEND CMAKE_MODULE_PATH "${MSSEG_REPO_ROOT}/cmake")

  find_package(OpenMP QUIET)
  include(Dependencies)                                   # diffg, MSCEER, TinyTIFF, json [, pybind11]
  add_subdirectory(${MSSEG_REPO_ROOT}/libs/core ${CMAKE_BINARY_DIR}/_msseg_core)
endif()

include(AddInstance)

add_msseg_instance(mscoupon
  PYTHON
  NEEDS_IO
  LIB_SOURCES
    lib/mscoupon/mscoupon_cli.cpp
    lib/mscoupon/config.cpp
    lib/mscoupon/sequence.cpp
    lib/mscoupon/io.cpp
    lib/mscoupon/filter.cpp
    lib/mscoupon/msc_stage.cpp
    lib/mscoupon/stats.cpp
    lib/mscoupon/matcher.cpp
    lib/mscoupon/pipeline.cpp
)

if(MSSEG_BUILD_TESTS)
  enable_testing()
  add_executable(mscoupon_tests tests/mscoupon_tests.cpp)
  target_link_libraries(mscoupon_tests PRIVATE mscoupon_lib)
  add_test(NAME mscoupon_tests COMMAND mscoupon_tests)
endif()
