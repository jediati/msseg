cmake_minimum_required(VERSION 3.20)

project(mscoupon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

set(DIFFG_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(DIFFG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(DIFFG_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

set(MSC_2D_LIB ON CACHE BOOL "Build MSCEER pure C++ 2D library" FORCE)

set(TinyTIFF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  diffg
  GIT_REPOSITORY https://github.com/jediati/diffg.git
  GIT_TAG e06a0eaf8e5cd3ff15fec518af8dd7dc1eaefaaf
)

FetchContent_Declare(
  msceer
  GIT_REPOSITORY https://github.com/sci-visus/MSCEER.git
  GIT_TAG 84b9419ca137d59ecb8d8bdfd2ef2f1aa3175709
)

FetchContent_Declare(
  tinytiff
  GIT_REPOSITORY https://github.com/jkriege2/TinyTIFF.git
  GIT_TAG master
)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)

FetchContent_MakeAvailable(diffg tinytiff nlohmann_json)

FetchContent_GetProperties(msceer)
if(NOT msceer_POPULATED)
  FetchContent_Populate(msceer)
endif()
list(APPEND CMAKE_MODULE_PATH "${msceer_SOURCE_DIR}/cmake")
add_subdirectory(${msceer_SOURCE_DIR} ${msceer_BINARY_DIR} EXCLUDE_FROM_ALL)

set(MSCOUPON_SOURCES
  src/mscoupon_cli.cpp
  src/mscoupon/config.cpp
  src/mscoupon/sequence.cpp
  src/mscoupon/io.cpp
  src/mscoupon/filter.cpp
  src/mscoupon/msc_stage.cpp
  src/mscoupon/stats.cpp
  src/mscoupon/pipeline.cpp
)

add_library(mscoupon_core STATIC ${MSCOUPON_SOURCES})
target_include_directories(mscoupon_core PUBLIC src)
target_link_libraries(mscoupon_core
  PUBLIC
    diffg::diffg
    msc_2d_lib
    nlohmann_json::nlohmann_json
)

set(_tinytiff_candidates TinyTIFF TinyTIFFReader TinyTIFFWriter tinytiffreader tinytiffwriter)
set(_tinytiff_found "")
foreach(candidate IN LISTS _tinytiff_candidates)
  if(TARGET ${candidate})
    list(APPEND _tinytiff_found ${candidate})
  endif()
endforeach()

if(_tinytiff_found STREQUAL "")
  message(FATAL_ERROR "Could not find TinyTIFF targets after FetchContent.")
endif()

target_link_libraries(mscoupon_core PUBLIC ${_tinytiff_found})

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(mscoupon_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(mscoupon src/main.cpp)
target_link_libraries(mscoupon PRIVATE mscoupon_core)

enable_testing()
add_executable(mscoupon_tests tests/mscoupon_tests.cpp src/mscoupon/config.cpp src/mscoupon/sequence.cpp src/mscoupon/stats.cpp)
target_include_directories(mscoupon_tests PRIVATE src)
target_link_libraries(mscoupon_tests PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME mscoupon_tests COMMAND mscoupon_tests)
