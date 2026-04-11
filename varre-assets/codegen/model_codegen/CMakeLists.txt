include(FetchContent)

find_package(tinyobjloader CONFIG QUIET)
if(NOT tinyobjloader_FOUND)
  FetchContent_Declare(
    tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG v2.0.0rc13
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(tinyobjloader)
endif()

add_executable(model_codegen main.cpp)
target_compile_features(model_codegen PRIVATE cxx_std_23)

if(TARGET tinyobjloader::tinyobjloader)
  target_link_libraries(model_codegen PRIVATE tinyobjloader::tinyobjloader)
elseif(TARGET tinyobjloader)
  target_link_libraries(model_codegen PRIVATE tinyobjloader)
else()
  message(FATAL_ERROR "tinyobjloader target not found")
endif()
