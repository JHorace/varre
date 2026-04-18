add_executable(varre_app_simple_stress main.cpp)
target_compile_features(varre_app_simple_stress PRIVATE cxx_std_23)
target_link_libraries(
  varre_app_simple_stress
  PRIVATE
    varre::app_core
)
