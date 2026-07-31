# msworkflow package: the generic JSON-described workflow runner over the core,
# built as its own pip-installable distribution ("msseg-workflow"). Ships a
# native CLI (msworkflow) and the msworkflow_py extension.
#
# Works standalone (pip / scikit-build-core) and as an add_subdirectory() from
# the dev root (bootstrap skipped when msseg_core already exists).
cmake_minimum_required(VERSION 3.21)

if(NOT TARGET msseg_core)
  project(msworkflow LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  option(MSSEG_BUILD_PYTHON "Build python bindings" OFF)

  get_filename_component(MSSEG_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
  list(APPEND CMAKE_MODULE_PATH "${MSSEG_REPO_ROOT}/cmake")

  find_package(OpenMP QUIET)
  include(Dependencies)
  add_subdirectory(${MSSEG_REPO_ROOT}/libs/core ${CMAKE_BINARY_DIR}/_msseg_core)
endif()

add_executable(msworkflow cli/main.cpp)
target_link_libraries(msworkflow PRIVATE msseg_core msseg_io)

if(MSSEG_BUILD_PYTHON)
  pybind11_add_module(msworkflow_py python/msworkflow_py.cpp)
  target_link_libraries(msworkflow_py PRIVATE msseg_core msseg_io)
  install(TARGETS msworkflow_py LIBRARY DESTINATION msseg/workflow COMPONENT msseg_python)
endif()
