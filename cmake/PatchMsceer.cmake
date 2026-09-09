# ---------------------------------------------------------------------------
# In-place source fixups for the pinned MSCEER checkout.
#
# Every entry here is a bug that has to be fixed in MSCEER's own sources -- the
# offending templates are instantiated inside MSCEER's own translation units, so
# nothing MSSeg can do at its include path or its call sites reaches them. Each
# fixup is a no-op once the pin advances past a revision that carries it, so the
# right way to retire one is to bump the pin in Dependencies.cmake and leave
# this file alone.
#
# Everything here MUST be idempotent: CMake re-configures over an already
# patched tree, and MSSEG_DEPS_DIR points at a checkout that is reused across
# build trees.
# ---------------------------------------------------------------------------

function(msseg_patch_msceer msceer_source_dir)
  # DigitizeSegmentInternal(): the four 2D/3D ASC/DSC overloads fall off the end
  # without returning their ADVECTION_EVENT. MSVC merely warns (C4715) and
  # returns whatever is in the return register, which is why this never showed
  # on Windows; GCC treats the fall-through as unreachable, so at -O0 the
  # function ends in a trap (SIGILL) and at -O2 control runs on into the
  # following code and reads past the end of `intersection_index` -- a segfault
  # in every accurate-gradient 2D MSC build under GCC. The value is discarded at
  # the one call site (IntegrateStreamline), so returning NONE only makes the
  # existing behaviour well defined.
  set(header "${msceer_source_dir}/include/gi_numeric_streamline_integrator.h")
  if(NOT EXISTS "${header}")
    message(WARNING "MSSeg: ${header} not found; skipping the MSCEER return-type fixup.")
    return()
  endif()

  file(READ "${header}" contents)
  # Each broken overload ends with `start_hex = next_hex;` followed by exactly
  # three closing braces (the else, the for, then the function itself) with
  # nothing but whitespace between them. Slipping the return in before the last
  # brace stops matching once it is there, which is what makes this idempotent.
  string(REGEX REPLACE
         "start_hex = next_hex;([ \t\r\n]*)}([ \t\r\n]*)}([ \t\r\n]*)}"
         "start_hex = next_hex;\\1}\\2}\\3return ADVECTION_EVENT::NONE;\n\t\t}"
         patched "${contents}")

  if(patched STREQUAL contents)
    return()
  endif()
  file(WRITE "${header}" "${patched}")
  message(STATUS "MSSeg: patched MSCEER DigitizeSegmentInternal to return a value "
                 "(see cmake/PatchMsceer.cmake).")
endfunction()
