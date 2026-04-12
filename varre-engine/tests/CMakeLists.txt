add_executable(varre_engine_tests engine_runtime.tests.cpp)
target_compile_features(varre_engine_tests PRIVATE cxx_std_23)
target_link_libraries(
  varre_engine_tests
  PRIVATE
    varre_engine
    varre::engine_test_dependencies
)

add_test(NAME varre_engine_tests COMMAND varre_engine_tests)
