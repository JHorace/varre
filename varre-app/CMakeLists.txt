# Workspace subtree matching the Rust crate `varre-app`.

find_package(SDL3 3.2 CONFIG REQUIRED)
if(NOT TARGET SDL3::SDL3)
  message(FATAL_ERROR "Unable to resolve SDL3::SDL3 target.")
endif()

if(NOT TARGET varre_engine)
  message(FATAL_ERROR "varre_app requires varre_engine target to be available.")
endif()

if(NOT TARGET varre_app_dependencies)
  add_library(varre_app_dependencies INTERFACE)
endif()
add_library(varre::app_dependencies ALIAS varre_app_dependencies)

target_compile_features(varre_app_dependencies INTERFACE cxx_std_23)
target_link_libraries(
  varre_app_dependencies
  INTERFACE
    varre_engine
    SDL3::SDL3
)

add_subdirectory(src)
