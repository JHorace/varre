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

add_executable(varre_assets_shaders_codegen_tests shader_codegen.tests.cpp)
target_compile_features(varre_assets_shaders_codegen_tests PRIVATE cxx_std_23)
target_compile_definitions(
  varre_assets_shaders_codegen_tests
  PRIVATE
    VARRE_SHADER_CODEGEN_EXE=\"$<TARGET_FILE:shader_codegen>\"
    VARRE_SHADERS_CORPUS_VALID_DIR=\"${CMAKE_CURRENT_LIST_DIR}/corpus/valid\"
    VARRE_SHADERS_CORPUS_INVALID_DIR=\"${CMAKE_CURRENT_LIST_DIR}/corpus/invalid\"
    VARRE_SHADERS_FIXTURES_DIR=\"${CMAKE_CURRENT_LIST_DIR}/fixtures\"
)
target_link_libraries(
  varre_assets_shaders_codegen_tests
  PRIVATE
    Catch2::Catch2WithMain
)

add_test(NAME varre_assets_shaders_codegen_tests COMMAND varre_assets_shaders_codegen_tests)

