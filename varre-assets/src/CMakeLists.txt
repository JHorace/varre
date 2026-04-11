add_library(varre_assets INTERFACE)

target_link_libraries(
  varre_assets
  INTERFACE
    varre_assets_shaders
    varre_assets_models
)

target_include_directories(
  varre_assets
  INTERFACE
    "${CMAKE_CURRENT_LIST_DIR}/include"
)

