# App target structure mirrors Rust binaries, excluding unfinished `hello_iced`.
add_library(varre_app_core STATIC)
add_library(varre::app_core ALIAS varre_app_core)

target_compile_features(varre_app_core PUBLIC cxx_std_23)
target_link_libraries(varre_app_core PUBLIC varre::app_dependencies)
target_include_directories(
  varre_app_core
  PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}/include"
)

add_subdirectory(app_core)
add_subdirectory(triangle)
add_subdirectory(mesh_simple)
