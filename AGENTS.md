# AGENTS.md

## Scope
This file defines repository-specific guidance for AI coding agents working in `/home/jsumihiro/repos/varre`.

## Project Goal
Convert `https://github.com/JHorace/vulkan-rust-engine` into a C++/CMake project.

## Explicit Exclusions
- Do not convert or implement `varre-iced-renderer` yet.
- Do not convert or implement the `hello_iced` application target yet.
- Do not add compatibility/fallback paths for Vulkan runtimes below 1.3.

## Runtime Baseline
- Vulkan **1.3+ only**.
- Engine/runtime design assumes dynamic rendering, synchronization2, and shader objects.
- Pre-1.3 runtime support is intentionally out of scope.

## Current Project State (Actual)
- Top-level CMake project `varre` with conditional subprojects:
  - `VARRE_BUILD_ASSETS` -> `varre-assets`
  - `VARRE_BUILD_ENGINE` -> `varre-engine`
  - `VARRE_BUILD_APPS` -> `varre-app`
- C++ standard is set globally to C++23 in `cmake/ProjectOptions.cmake`.
- `include(CTest)` is enabled at root; tests are controlled through `BUILD_TESTING`.

### `varre-assets` status
- Active codegen executables:
  - `model_codegen` (implemented; uses `tinyobjloader`)
  - `shader_codegen` (stub; prints unimplemented and exits non-zero)
- Active generated asset targets:
  - `varre_assets_models_codegen` custom target generates:
    - `models_generated.hpp`
    - `models_generated.cpp`
  - `varre_assets_models` static library builds generated model source.
  - `varre_assets_shaders_codegen` custom target copies placeholder templates to:
    - `shaders_generated.hpp`
    - `shaders_generated.cpp`
  - `varre_assets_shaders` static library links shader stubs + generated placeholders.
  - `varre_assets` interface library links `varre_assets_models` and `varre_assets_shaders`.
- Model generation options (CMake cache):
  - `VARRE_ASSETS_MODELS_FLIP_HANDEDNESS`
  - `VARRE_ASSETS_MODELS_FLIP_WINDING`
  - `VARRE_ASSETS_MODELS_FLIP_UV_V`
- `varre-assets/models/tests` defines `varre_assets_models_tests` when `BUILD_TESTING=ON`.
  - Tests use Catch2 and may fetch it via `FetchContent` if not preinstalled.

### `varre-engine` and `varre-app` status
- Both are currently CMake structure scaffolds only.
- Nested module trees are present to mirror Rust layout, but concrete targets are intentionally deferred.
- Present placeholder paths include:
  - `varre-engine/src/geometry`
  - `varre-engine/src/render_context/{triangle,mesh_simple}`
  - `varre-app/src/{triangle,mesh_simple}`

## Engine Execution Plan (Current)
Use this sequence when implementing `varre-engine` internals.

1. Create `varre_engine` and wire it to `varre::engine_dependencies`.
2. Define public engine API headers first (`Engine`, `DeviceContext`, `RenderContext`, config structs, error types).
3. Port module structure from Rust with near 1:1 C++ mapping:
   - `physical_device_utils`
   - `extensions`
   - `memory_utils`
   - `command_buffers`
   - `mesh_utils`
   - `shader_utils`
   - `render_context/triangle`
   - `render_context/mesh_simple`
   - window/surface provider abstraction (no direct SDL in engine)
4. Implement instance creation with `vk::raii::Context` + `vk::raii::Instance`, validation toggles, and debug messenger.
5. Implement physical-device selection and queue-family discovery.
6. Implement logical device creation using Vulkan 1.3 + extension feature chains (`vk::StructureChain`).
7. Implement per-frame state with RAII synchronization (`NUM_FRAMES_IN_FLIGHT = 3`).
8. Implement command infrastructure (`vk::raii::CommandPool`, per-frame command buffers, one-time submit).
9. Implement surface/swapchain path behind engine-facing platform interfaces.
10. Port shader integration to `varre_assets` generated shader APIs.
11. Port mesh/model upload integration to `varre_assets` models APIs + VMA-backed buffers/images.
12. Port texture upload path (staging, transitions, view/sampler creation) with `stb_image`.
13. Port render contexts in order: `triangle`, then `mesh_simple`.
14. Add cleanup/lifetime checks and validation-friendly errors for swapchain recreation flows.
15. Add tests for queue selection, feature-chain composition, shader/model lookup integration, and smoke initialization.

### Planned Deviations / Improvements
- Use `vk::raii` ownership consistently instead of manual destroy paths.
- Keep surface/window provider outside `varre-engine`.
- Use VMA from first engine implementation (avoid handwritten Vulkan memory allocation paths).
- Use `fmt`/`spdlog` for structured diagnostics.
- Normalize extension/feature enablement via typed helper builders (`vk::StructureChain`).
- Prefer explicit frame-context structs over hidden mutable synchronization state.

### Recommended Implementation Order
1. Core init path: instance -> device -> queues -> command pool.
2. Swapchain + frame-loop primitives.
3. Assets integration (shaders/models/textures).
4. Render contexts (`triangle`, then `mesh_simple`).
5. Test hardening and resize/recreate edge cases.

## Implementation Rules
- Implement CMake structure first, then target internals.
- Preserve nested module/subproject layout when adding targets.
- Keep placeholder/stub behavior explicit for unfinished components.
- Avoid introducing functionality for excluded targets.
- Do not replace placeholder shader flow with real shader embedding until `shader_codegen` is intentionally implemented.
- Do not implement legacy fallback paths (e.g., pre-1.3 compatibility modes) unless explicitly requested.

## Build & Validation
- Configure: `cmake -S . -B build`
- Build: `cmake --build build -j4`
- Prefer validating only impacted targets when possible.
- Typical targeted builds:
  - `cmake --build build --target model_codegen`
  - `cmake --build build --target varre_assets_models`
  - `cmake --build build --target varre_assets_shaders`
- Tests (when enabled): `ctest --test-dir build --output-on-failure`

## Formatting & Style
- Use `.clang-format` at repo root.
- Maximum line length: `160`.
- Use Doxygen-style comments for public APIs and non-trivial components.

## Editing Guidelines
- Keep changes minimal and scoped to the requested task.
- Do not remove existing generated-file flows unless replacing them with working equivalents.
- Maintain deterministic code generation behavior where applicable (stable ordering, stable identifiers).
- Keep generated output locations in build directories (do not move generated files into source tree).

## When Adding New Targets
- Add targets in the relevant nested `CMakeLists.txt`, not only at root.
- Keep public headers under `include/varre/...` so includes stay stable (e.g. `#include <varre/assets/models.hpp>`).
- Ensure dependencies are explicit (`add_dependencies`, `target_link_libraries`, include dirs).
- For generated artifacts, model dependencies explicitly in CMake (`add_custom_command` + `OUTPUT` + `DEPENDS` + `VERBATIM`).
