add_executable(varre_app_triangle main.cpp)
target_compile_features(varre_app_triangle PRIVATE cxx_std_23)
target_link_libraries(
  varre_app_triangle
  PRIVATE
    varre::app_core
)
