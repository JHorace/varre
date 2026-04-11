set(VARRE_ASSETS_MODELS_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(VARRE_ASSETS_MODELS_GEN_HEADER "${VARRE_ASSETS_MODELS_GEN_DIR}/models_generated.hpp")
set(VARRE_ASSETS_MODELS_GEN_SOURCE "${VARRE_ASSETS_MODELS_GEN_DIR}/models_generated.cpp")
set(VARRE_ASSETS_MODELS_INPUT_DIR "${CMAKE_CURRENT_LIST_DIR}/obj")

file(
  GLOB_RECURSE
  VARRE_ASSETS_MODELS_OBJ_FILES
  CONFIGURE_DEPENDS
  "${VARRE_ASSETS_MODELS_INPUT_DIR}/*.obj"
)

set(VARRE_ASSETS_MODELS_GENERATOR_DEPS
  model_codegen
  ${VARRE_ASSETS_MODELS_OBJ_FILES}
)

add_custom_command(
  OUTPUT
    "${VARRE_ASSETS_MODELS_GEN_HEADER}"
    "${VARRE_ASSETS_MODELS_GEN_SOURCE}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${VARRE_ASSETS_MODELS_GEN_DIR}"
  COMMAND
    $<TARGET_FILE:model_codegen>
    --input-root "${VARRE_ASSETS_MODELS_INPUT_DIR}"
    --output-dir "${VARRE_ASSETS_MODELS_GEN_DIR}"
    --emit-hpp "${VARRE_ASSETS_MODELS_GEN_HEADER}"
    --emit-cpp "${VARRE_ASSETS_MODELS_GEN_SOURCE}"
    --namespace "varre::assets"
  DEPENDS ${VARRE_ASSETS_MODELS_GENERATOR_DEPS}
  COMMENT "Generating model asset registry"
  VERBATIM
)

add_custom_target(
  varre_assets_models_codegen
  DEPENDS
    "${VARRE_ASSETS_MODELS_GEN_HEADER}"
    "${VARRE_ASSETS_MODELS_GEN_SOURCE}"
)

add_library(
  varre_assets_models STATIC
  "${VARRE_ASSETS_MODELS_GEN_SOURCE}"
)

add_dependencies(varre_assets_models varre_assets_models_codegen)

target_compile_features(varre_assets_models PUBLIC cxx_std_23)
target_include_directories(
  varre_assets_models
  PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}/include"
    "${VARRE_ASSETS_MODELS_GEN_DIR}"
)

# Keep nested model-asset directory visible in CMake hierarchy.
add_subdirectory(obj)
