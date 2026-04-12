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

### `varre-engine` status
- `varre_engine` static library target is active and links `varre::engine_dependencies`.
- Implemented modules:
  - `core`: engine initialization, queue/device profile resolution, surface/swapchain primitives.
  - `sync`: `FrameLoop` + upload context.
  - `assets`: model upload, shader object cache, texture upload.
  - `descriptors`: descriptor/pipeline-layout/material helpers.
  - `render_context`: pass mode (`PassGraph`, `PassExecutor`) and `PassFrameLoop`.
- Design baseline:
  - Vulkan 1.3+ only.
  - Dynamic rendering + synchronization2 + `VK_EXT_shader_object` first-class.
  - No legacy `VkPipeline`/`VkRenderPass` fallback path.
- Tests:
  - `varre_engine_tests` target exists under `varre-engine/tests`.
  - Covers queue/feature profile behavior, pass dependency/barrier flows, cross-queue timeline sync, shader/model integration, and swapchain recreate smoke path.
- Still intentionally placeholder:
  - `varre-engine/src/geometry`
  - `varre-engine/src/render_context/{triangle,mesh_simple}` (app-level examples are no longer engine internals).

### `varre-app` status
- `varre-app` remains scaffolded (`triangle`, `mesh_simple` CMake placeholders).
- Porting priority is now app migration on top of pass mode.

## App Port Plan (Current Priority)
Use this sequence for converting old Rust app targets.

1. Lock architecture boundaries:
   - Keep `varre-engine` window-system agnostic.
   - Put SDL3 window/event loop + Vulkan surface creation in `varre-app`.
2. Create shared `varre_app_core` runtime:
   - SDL lifecycle.
   - Vulkan instance extension query via SDL.
   - SDL-created `VkSurfaceKHR` adoption into `SurfaceContext`.
   - Engine/swapchain/`PassFrameLoop` bootstrap and shutdown.
3. Define app-facing interfaces:
   - Scene lifecycle hooks (`init`, `build_pass_graph`, `on_swapchain_recreated`, `on_event`, `shutdown`).
   - Per-frame context describing swapchain image/view/extent/frame index.
4. Wire real app targets:
   - Convert `triangle` and `mesh_simple` into concrete executables.
   - Link against `varre_engine`, `varre_assets`, and SDL3.
5. Port `triangle` first as the pass-mode reference path.
6. Port `mesh_simple` second with model upload dependency integration.
7. Add ImGui-ready hooks now (no backend integration yet):
   - Keep a no-op UI overlay interface to avoid future runtime refactors.
8. Add app-level smoke coverage:
   - startup/shutdown,
   - resize/minimize/swapchain recreate behavior,
   - extension/feature diagnostic paths.

### Planned Deviations / Improvements
- Use `vk::raii` ownership consistently instead of manual destroy paths.
- Keep surface/window provider outside `varre-engine`.
- Use VMA from first engine implementation (avoid handwritten Vulkan memory allocation paths).
- Use `fmt`/`spdlog` for structured diagnostics.
- Normalize extension/feature enablement via typed helper builders (`vk::StructureChain`).
- Prefer explicit frame-context structs over hidden mutable synchronization state.
- Keep SDL3 dependencies out of `varre-engine`.
- Keep ImGui integration staged behind app-layer hooks.

### Recommended Implementation Order (Apps)
1. Implement `varre_app_core` + SDL3 surface/bootstrap path.
2. Port `triangle` end-to-end with pass mode.
3. Port `mesh_simple` on the same runtime.
4. Add ImGui hook scaffolding (no renderer backend yet).
5. Add app-level smoke tests and tighten recreate handling.

## Deferred Cleanup Plan
- Post-app-migration engine structure/naming cleanup is recorded in `docs/post_app_migration_engine_cleanup.md`.
- Keep this deferred until app migration is complete unless explicitly requested otherwise.

## Implementation Rules
- Implement CMake structure first, then target internals.
- Preserve nested module/subproject layout when adding targets.
- Keep placeholder/stub behavior explicit for unfinished components.
- Avoid introducing functionality for excluded targets.
- Do not replace placeholder shader flow with real shader embedding until `shader_codegen` is intentionally implemented.
- Do not implement legacy fallback paths (e.g., pre-1.3 compatibility modes) unless explicitly requested.
- Do not add SDL3 or ImGui dependencies to `varre-engine`; keep those in `varre-app`.
- New app rendering code should use pass mode (`PassGraph`/`PassExecutor`/`PassFrameLoop`) only.

## Build & Validation
- Configure: `cmake -S . -B build`
- Build: `cmake --build build -j4`
- Prefer validating only impacted targets when possible.
- Typical targeted builds:
  - `cmake --build build --target model_codegen`
  - `cmake --build build --target varre_assets_models`
  - `cmake --build build --target varre_assets_shaders`
  - `cmake --build build --target varre_engine`
  - `cmake --build build --target varre_engine_tests`
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
