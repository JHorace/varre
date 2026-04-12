add_library(varre_engine STATIC)
add_library(varre::engine ALIAS varre_engine)

target_compile_features(varre_engine PUBLIC cxx_std_23)
target_link_libraries(varre_engine PUBLIC varre::engine_dependencies)
target_include_directories(
  varre_engine
  PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}/include"
)

# Engine source layout mirrors the Rust module structure.
add_subdirectory(core)
add_subdirectory(sync)
add_subdirectory(descriptors)
add_subdirectory(geometry)
add_subdirectory(render_context)
