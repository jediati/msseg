# Generic JSON-workflow runner: a sibling to the verified instances that runs
# any workflow described in JSON over the core stage registry.
add_executable(msworkflow cli/main.cpp)
target_link_libraries(msworkflow PRIVATE msseg_core msseg_io)

if(MSSEG_BUILD_PYTHON)
  pybind11_add_module(msworkflow_py python/msworkflow_py.cpp)
  target_link_libraries(msworkflow_py PRIVATE msseg_core msseg_io)
  install(TARGETS msworkflow_py LIBRARY DESTINATION msseg/workflow COMPONENT msseg_python)
endif()
