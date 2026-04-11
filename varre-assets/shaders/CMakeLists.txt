set(VARRE_ASSETS_SHADERS_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(VARRE_ASSETS_SHADERS_GEN_HEADER "${VARRE_ASSETS_SHADERS_GEN_DIR}/shaders_generated.hpp")
set(VARRE_ASSETS_SHADERS_GEN_SOURCE "${VARRE_ASSETS_SHADERS_GEN_DIR}/shaders_generated.cpp")
set(VARRE_ASSETS_SHADERS_INPUT_DIR "${CMAKE_CURRENT_LIST_DIR}/slang")

file(
  GLOB_RECURSE
  VARRE_ASSETS_SHADERS_SLANG_FILES
  CONFIGURE_DEPENDS
  "${VARRE_ASSETS_SHADERS_INPUT_DIR}/*.slang"
)

set(
  VARRE_ASSETS_SHADERS_GENERATOR_ARGS
  --input-root "${VARRE_ASSETS_SHADERS_INPUT_DIR}"
  --output-dir "${VARRE_ASSETS_SHADERS_GEN_DIR}"
  --emit-hpp "${VARRE_ASSETS_SHADERS_GEN_HEADER}"
  --emit-cpp "${VARRE_ASSETS_SHADERS_GEN_SOURCE}"
  --namespace "varre::assets"
)

set(VARRE_ASSETS_SHADERS_GENERATOR_DEPS
  shader_codegen
  ${VARRE_ASSETS_SHADERS_SLANG_FILES}
)

add_custom_command(
  OUTPUT
    "${VARRE_ASSETS_SHADERS_GEN_HEADER}"
    "${VARRE_ASSETS_SHADERS_GEN_SOURCE}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${VARRE_ASSETS_SHADERS_GEN_DIR}"
  COMMAND $<TARGET_FILE:shader_codegen> ${VARRE_ASSETS_SHADERS_GENERATOR_ARGS}
  DEPENDS ${VARRE_ASSETS_SHADERS_GENERATOR_DEPS}
  COMMENT "Generating shader asset registry"
  VERBATIM
)

add_custom_target(
  varre_assets_shaders_codegen
  DEPENDS
    "${VARRE_ASSETS_SHADERS_GEN_HEADER}"
    "${VARRE_ASSETS_SHADERS_GEN_SOURCE}"
)

add_library(
  varre_assets_shaders STATIC
  "${VARRE_ASSETS_SHADERS_GEN_SOURCE}"
)

add_dependencies(varre_assets_shaders varre_assets_shaders_codegen)

target_compile_features(varre_assets_shaders PUBLIC cxx_std_23)
target_include_directories(
  varre_assets_shaders
  PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}/include"
    "${VARRE_ASSETS_SHADERS_GEN_DIR}"
)
