# ---------------------------------------------------------------------------
# add_msseg_instance(NAME [PYTHON] [NEEDS_IO] LIB_SOURCES <files...>)
#
# Emits the standard trio for a verified segmentation instance:
#   <NAME>_lib  STATIC library (the workflow definition, links msseg_core)
#   <NAME>      CLI executable  (from <dir>/cli/main.cpp)
#   <NAME>_py   pybind11 module (from <dir>/python/<NAME>_py.cpp), when PYTHON
#               and MSSEG_BUILD_PYTHON; installed into the msseg wheel.
#
# Conventions: instance headers live under <dir>/lib and are included as
# "<name>/...". Call from the instance's own CMakeLists (uses
# CMAKE_CURRENT_SOURCE_DIR).
# ---------------------------------------------------------------------------
function(add_msseg_instance NAME)
  cmake_parse_arguments(ARG "PYTHON;NEEDS_IO" "" "LIB_SOURCES" ${ARGN})

  add_library(${NAME}_lib STATIC ${ARG_LIB_SOURCES})
  target_include_directories(${NAME}_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib)
  target_link_libraries(${NAME}_lib PUBLIC msseg_core)
  if(ARG_NEEDS_IO)
    target_link_libraries(${NAME}_lib PUBLIC msseg_io)
  endif()

  add_executable(${NAME} ${CMAKE_CURRENT_SOURCE_DIR}/cli/main.cpp)
  target_link_libraries(${NAME} PRIVATE ${NAME}_lib)

  if(ARG_PYTHON AND MSSEG_BUILD_PYTHON)
    pybind11_add_module(${NAME}_py ${CMAKE_CURRENT_SOURCE_DIR}/python/${NAME}_py.cpp)
    target_link_libraries(${NAME}_py PRIVATE ${NAME}_lib)
    install(TARGETS ${NAME}_py LIBRARY DESTINATION msseg/${NAME})
  endif()
endfunction()
