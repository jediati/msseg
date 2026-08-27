# ---------------------------------------------------------------------------
# Centralized dependency provisioning for MSSeg.
#
# Default: FetchContent pinned from GitHub (works on Windows/Linux with a
# network). Offline/HPC: set MSSEG_DEPS_DIR to a directory containing local
# checkouts named after each dependency (diffg/, msceer/, tinytiff/, json/,
# pybind11/) and CMake will source them from disk instead of the network.
# Pair with -DFETCHCONTENT_FULLY_DISCONNECTED=ON (see the `hpc` preset).
# ---------------------------------------------------------------------------
include(FetchContent)

# Honor a local dependency mirror for offline builds by pointing CMake's native
# per-dependency source override at MSSEG_DEPS_DIR/<name> when it exists.
function(_msseg_local_override name subdir)
  if(MSSEG_DEPS_DIR)
    set(_candidate "${MSSEG_DEPS_DIR}/${subdir}")
    if(EXISTS "${_candidate}")
      string(TOUPPER "${name}" _upper)
      set(FETCHCONTENT_SOURCE_DIR_${_upper} "${_candidate}" CACHE PATH
          "Local source for ${name}" FORCE)
      message(STATUS "MSSeg: using local ${name} from ${_candidate}")
    endif()
  endif()
endfunction()

_msseg_local_override(diffg diffg)
_msseg_local_override(msceer msceer)
_msseg_local_override(tinytiff tinytiff)
_msseg_local_override(nlohmann_json json)
_msseg_local_override(pybind11 pybind11)

# diffg: options must be set before it is populated.
set(DIFFG_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(DIFFG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(DIFFG_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# MSCEER: build only the 2D facade; the 3D facade builds a placeholder gradient
# so we drive the GInt 3D stack directly from msseg_core instead.
set(MSC_2D_LIB ON CACHE BOOL "Build MSCEER pure C++ 2D library" FORCE)

set(TinyTIFF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(diffg
  GIT_REPOSITORY https://github.com/jediati/diffg.git
  GIT_TAG bdf5623be9b1563518cdd05ff644f74ed6c5e48a   # + batched multi-filter bank (apply_filter_bank)
)
FetchContent_Declare(msceer
  GIT_REPOSITORY https://github.com/sci-visus/MSCEER.git
  GIT_TAG c04aab11cb4fd3b46450f1b70650437a8752858f   # + cheap re-thresholding (compact base ids, no per-pixel hash)
)
FetchContent_Declare(tinytiff
  GIT_REPOSITORY https://github.com/jkriege2/TinyTIFF.git
  GIT_TAG master
)
FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)

FetchContent_MakeAvailable(diffg tinytiff nlohmann_json)

# MSCEER is added via the manual populate path so we can (a) prepend its cmake/
# to CMAKE_MODULE_PATH before its add_subdirectory runs, and (b) EXCLUDE its
# ~35 apps from the default build.
FetchContent_GetProperties(msceer)
if(NOT msceer_POPULATED)
  FetchContent_Populate(msceer)
endif()
list(APPEND CMAKE_MODULE_PATH "${msceer_SOURCE_DIR}/cmake")
add_subdirectory(${msceer_SOURCE_DIR} ${msceer_BINARY_DIR} EXCLUDE_FROM_ALL)

# Resolve whichever TinyTIFF target names this revision exports.
set(_tinytiff_candidates TinyTIFF TinyTIFFReader TinyTIFFWriter tinytiffreader tinytiffwriter)
set(MSSEG_TINYTIFF_TARGETS "")
foreach(candidate IN LISTS _tinytiff_candidates)
  if(TARGET ${candidate})
    list(APPEND MSSEG_TINYTIFF_TARGETS ${candidate})
  endif()
endforeach()
if(MSSEG_TINYTIFF_TARGETS STREQUAL "")
  message(FATAL_ERROR "Could not find TinyTIFF targets after FetchContent.")
endif()
# Included at root scope (include() does not push a scope), so this is visible
# to add_subdirectory(core) and the instances below.

# pybind11 only when building python bindings. Prefer an already-provided
# pybind11 (scikit-build-core injects one during `pip install`), else fetch.
if(MSSEG_BUILD_PYTHON)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    FetchContent_Declare(pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
  endif()
endif()
