target_sources(
  varre_engine
  PRIVATE
    pass_frame_loop.cpp
    pass_mode.cpp
)

# App-level sample contexts remain intentionally out of engine internals.
add_subdirectory(triangle)
add_subdirectory(mesh_simple)
