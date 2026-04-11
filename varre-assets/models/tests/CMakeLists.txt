include(FetchContent)

find_package(Catch2 3 CONFIG QUIET)
if(NOT Catch2_FOUND)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.6.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(Catch2)
endif()

add_executable(varre_assets_models_tests model_loader.tests.cpp)
target_compile_features(varre_assets_models_tests PRIVATE cxx_std_23)
target_link_libraries(
  varre_assets_models_tests
  PRIVATE
    varre_assets_models
    Catch2::Catch2WithMain
)

add_test(NAME varre_assets_models_tests COMMAND varre_assets_models_tests)

