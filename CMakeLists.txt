cmake_minimum_required(VERSION 3.21)

project(MSSeg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Static core libraries are linked into shared python modules on Linux/HPC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- Options ---------------------------------------------------------------
# This root CMakeLists is the developer "build & test everything" entry point.
# Each package under packages/<name> is ALSO independently pip-installable via
# its own pyproject.toml + CMakeLists.txt (which bootstrap the deps + libs/core
# themselves). Adding them here reuses the single dependency set below.
option(MSSEG_BUILD_TESTS "Build tests" ON)
option(MSSEG_BUILD_PYTHON "Build python bindings" OFF)
if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
  option(MSSEG_BUILD_VIEWER "Build the OpenGL debug viewer (needs ext/win64 GL binaries)" ON)
else()
  option(MSSEG_BUILD_VIEWER "Build the OpenGL debug viewer" OFF)
endif()
set(MSSEG_DEPS_DIR "" CACHE PATH "Local dependency checkouts for offline/HPC builds")

# --- Dependencies (provisioned once, shared by libs/core + every package) ---
find_package(OpenMP QUIET)
include(Dependencies)   # diffg, MSCEER (GInt/msc_2d_lib), TinyTIFF, nlohmann_json [, pybind11]

# --- Shared library ---------------------------------------------------------
add_subdirectory(libs/core)

# --- Packages (each: instance lib + CLI + pybind module) --------------------
# Their CMakeLists skip their standalone-bootstrap block because msseg_core is
# already defined here (if(NOT TARGET msseg_core)).
add_subdirectory(packages/mscoupon)
add_subdirectory(packages/cellseg)
add_subdirectory(packages/msworkflow)

# --- Optional native 3D viewer (desktop/Windows; never in a wheel) ----------
if(MSSEG_BUILD_VIEWER)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/apps/msviewer/CMakeLists.txt")
    add_subdirectory(libs/render)
    add_subdirectory(apps/msviewer)
  else()
    message(STATUS "MSSeg: viewer enabled but apps/msviewer is absent; skipping.")
  endif()
endif()

# --- Tests -----------------------------------------------------------------
# Per-package tests (mscoupon_tests, cellseg_tests) are wired by each package's
# CMakeLists under MSSEG_BUILD_TESTS. The portable core smoke test lives here.
if(MSSEG_BUILD_TESTS)
  enable_testing()
  add_executable(core_smoke libs/core/tests/core_smoke.cpp)
  target_link_libraries(core_smoke PRIVATE msseg_core)
  add_test(NAME core_smoke COMMAND core_smoke)
endif()
