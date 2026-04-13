add_executable(varre_app_mesh_simple main.cpp)
target_compile_features(varre_app_mesh_simple PRIVATE cxx_std_23)
target_link_libraries(
  varre_app_mesh_simple
  PRIVATE
    varre::app_core
)
