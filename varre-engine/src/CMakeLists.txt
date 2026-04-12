add_library(
  varre_engine STATIC
  engine.cpp
  surface.cpp
  swapchain.cpp
  frame_loop.cpp
  descriptors.cpp
  material_descriptors.cpp
  upload_context.cpp
)
add_library(varre::engine ALIAS varre_engine)

target_compile_features(varre_engine PUBLIC cxx_std_23)
target_link_libraries(varre_engine PUBLIC varre::engine_dependencies)
target_include_directories(
  varre_engine
  PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}/include"
)

# Engine source layout mirrors the Rust module structure.
add_subdirectory(geometry)
add_subdirectory(render_context)
