# Workspace subtree matching the Rust crate `varre-assets`.
# This defines CMake structure and stub targets only.

option(
  VARRE_ASSETS_BUILD_CODEGEN_TOOLS
  "Build placeholder code-generation executables for assets."
  ON
)

if(VARRE_ASSETS_BUILD_CODEGEN_TOOLS)
  add_subdirectory(codegen)
endif()

add_subdirectory(shaders)
add_subdirectory(models)
add_subdirectory(src)
