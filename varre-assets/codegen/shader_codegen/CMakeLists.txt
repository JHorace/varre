include(FetchContent)

set(
  VARRE_SLANGC_EXECUTABLE
  ""
  CACHE FILEPATH
  "Path to the slangc executable used by shader_codegen."
)
if(VARRE_SLANGC_EXECUTABLE STREQUAL "")
  find_program(
    VARRE_SLANGC_EXECUTABLE
    NAMES slangc
    DOC "Path to the slangc executable used by shader_codegen."
  )
endif()
if(VARRE_SLANGC_EXECUTABLE STREQUAL "")
  message(FATAL_ERROR "slangc executable not found. Set VARRE_SLANGC_EXECUTABLE to a valid path.")
endif()

find_package(SPIRV-Tools CONFIG QUIET)
if(NOT SPIRV-Tools_FOUND)
  set(SPIRV_SKIP_TESTS ON CACHE BOOL "Skip SPIRV-Tools tests." FORCE)
  set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "Skip SPIRV-Tools CLI tools." FORCE)
  FetchContent_Declare(
    SPIRV-Tools
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools.git
    GIT_TAG v2025.4
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(SPIRV-Tools)
endif()

set(VARRE_SPIRV_TOOLS_TARGET "")
foreach(target_candidate SPIRV-Tools::SPIRV-Tools SPIRV-Tools SPIRV-Tools-static)
  if(TARGET ${target_candidate})
    set(VARRE_SPIRV_TOOLS_TARGET ${target_candidate})
    break()
  endif()
endforeach()

if(VARRE_SPIRV_TOOLS_TARGET STREQUAL "")
  message(FATAL_ERROR "Unable to resolve a SPIRV-Tools library target.")
endif()

add_executable(shader_codegen main.cpp)
target_compile_features(shader_codegen PRIVATE cxx_std_23)
target_compile_definitions(
  shader_codegen
  PRIVATE
    VARRE_SLANGC_EXECUTABLE=\"${VARRE_SLANGC_EXECUTABLE}\"
)
target_link_libraries(shader_codegen PRIVATE ${VARRE_SPIRV_TOOLS_TARGET})
