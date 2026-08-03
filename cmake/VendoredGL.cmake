# ---------------------------------------------------------------------------
# Imported targets for the vendored, Windows-x64 OpenGL dependencies used by
# the (optional) debug viewer. Binaries live under ext/win64/. Only included
# when MSSEG_BUILD_VIEWER is ON (Windows x64).
#
# NOTE (M1 stub): the ext/win64 binaries are copied in during M6. This module
# defines the imported targets only if the files are present, so configuring
# without the viewer binaries does not fail.
# ---------------------------------------------------------------------------
set(_gl_root ${CMAKE_SOURCE_DIR}/ext/win64)

if(EXISTS ${_gl_root}/freeglut/lib/freeglut.lib AND NOT TARGET freeglut::freeglut)
  # GLOBAL so the imported target is visible in apps/msviewer (a subdirectory)
  # and to its POST_BUILD $<TARGET_FILE:...> generator expression.
  add_library(freeglut::freeglut SHARED IMPORTED GLOBAL)
  set_target_properties(freeglut::freeglut PROPERTIES
    IMPORTED_IMPLIB   ${_gl_root}/freeglut/lib/freeglut.lib
    IMPORTED_LOCATION ${_gl_root}/freeglut/bin/freeglut.dll
    INTERFACE_INCLUDE_DIRECTORIES ${_gl_root}/freeglut/include)
endif()

if(EXISTS ${_gl_root}/glew/lib/glew32.lib AND NOT TARGET glew::glew)
  add_library(glew::glew SHARED IMPORTED GLOBAL)
  set_target_properties(glew::glew PROPERTIES
    IMPORTED_IMPLIB   ${_gl_root}/glew/lib/glew32.lib
    IMPORTED_LOCATION ${_gl_root}/glew/bin/glew32.dll
    INTERFACE_INCLUDE_DIRECTORIES ${_gl_root}/glew/include)
endif()
