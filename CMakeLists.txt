cmake_minimum_required(VERSION 3.21)

project(MSSeg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Static core libraries are linked into shared python modules on Linux/HPC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- Options ---------------------------------------------------------------
option(MSSEG_BUILD_INSTANCES "Build verified segmentation instances" ON)
option(MSSEG_BUILD_TESTS "Build tests" ON)
option(MSSEG_BUILD_PYTHON "Build python bindings" OFF)
if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
  option(MSSEG_BUILD_VIEWER "Build the Windows OpenGL debug viewer" ON)
else()
  option(MSSEG_BUILD_VIEWER "Build the Windows OpenGL debug viewer" OFF)
endif()
set(MSSEG_DEPS_DIR "" CACHE PATH "Local dependency checkouts for offline/HPC builds")

# --- Dependencies ----------------------------------------------------------
find_package(OpenMP QUIET)
include(Dependencies)   # diffg, MSCEER (GInt/msc_2d_lib), TinyTIFF, nlohmann_json [, pybind11]
include(AddInstance)    # add_msseg_instance()

# --- Targets ---------------------------------------------------------------
add_subdirectory(core)

if(MSSEG_BUILD_INSTANCES)
  add_subdirectory(instances/mscoupon)
endif()

add_subdirectory(generic/msworkflow)

if(MSSEG_BUILD_VIEWER)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/apps/msviewer/CMakeLists.txt")
    include(VendoredGL)
    add_subdirectory(apps/msviewer)
  else()
    message(STATUS "MSSeg: viewer enabled but apps/msviewer is not present yet (lands in M6); skipping.")
  endif()
endif()

# --- Tests -----------------------------------------------------------------
if(MSSEG_BUILD_TESTS)
  enable_testing()
  if(MSSEG_BUILD_INSTANCES)
    add_executable(mscoupon_tests tests/mscoupon_tests.cpp)
    target_link_libraries(mscoupon_tests PRIVATE mscoupon_lib)
    add_test(NAME mscoupon_tests COMMAND mscoupon_tests)
  endif()

  add_executable(core_smoke tests/core_smoke.cpp)
  target_link_libraries(core_smoke PRIVATE msseg_core)
  add_test(NAME core_smoke COMMAND core_smoke)
endif()
