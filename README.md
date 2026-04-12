# varre

`varre` is an in-progress **conversion of the Rust project [`JHorace/vulkan-rust-engine`](https://github.com/JHorace/vulkan-rust-engine) to C++ with CMake**.

This repository is being built using **guided AI generation**:
- AI is used to draft and implement incremental changes.
- The conversion direction, constraints, and acceptance are human-guided.
- Structure and behavior are kept explicit, with placeholders where features are not implemented yet.

## Conversion Approach

- Preserve Rust crate/module structure where practical.
- Establish CMake target layout first, then fill internals.
- Keep unfinished work as explicit stubs instead of implicit partial behavior.
- Maintain deterministic code generation paths for assets.
- Target Vulkan runtime baseline: **1.3+ only**.
- Pre-1.3 runtime compatibility is intentionally out of scope.
- Use pass-mode rendering with `VK_EXT_shader_object` and fully dynamic state only.

## Current Status

- `varre-assets`
  - `model_codegen`: implemented
  - `shader_codegen`: stub (intentionally unimplemented)
  - `varre_assets_models`: generated model asset library
  - `varre_assets_shaders`: placeholder shader asset library
  - `varre_assets`: interface target aggregating assets
- `varre-engine`: active library target with substantial implementation.
  - Core Vulkan init (`EngineContext`), queue discovery/topology, logical device + feature chain composition.
  - Surface/swapchain primitives (`SurfaceContext`, `SwapchainContext`) with recreate flow.
  - Frame orchestration (`FrameLoop`) and pass-mode orchestration (`PassFrameLoop`).
  - Pass graph execution (`PassExecutor`/`PassGraph`) for graphics + compute + transfer with queue-aware scheduling.
  - Shader-object path + dynamic-state command encoding (`VK_EXT_shader_object`; no legacy pipeline/renderpass fallback).
  - Asset services: shader object cache, model upload service with pass dependency wait wiring, texture upload service.
  - Descriptor/pipeline-layout caches and material descriptor helpers.
  - Runtime tests target `varre_engine_tests` (device/queue profile behavior, pass barriers/sync, asset integration, swapchain smoke/recreate).
- `varre-app`: still scaffolded at target level (`triangle`, `mesh_simple` placeholders), now prioritized for migration.

## App Port Plan (As Of 2026-04-12)

Planned migration sequence for old Rust app targets:

1. Lock architecture boundaries.
   - `varre-engine` remains windowing/platform agnostic.
   - `varre-app` owns SDL3 window/event loop + Vulkan surface creation.
2. Create shared `varre_app_core`.
   - SDL bootstrap/teardown.
   - SDL Vulkan instance extension query and surface creation/adoption.
   - Engine + swapchain + `PassFrameLoop` bootstrap and shutdown orchestration.
3. Define reusable app interfaces.
   - Scene lifecycle hooks (`init`, `build_pass_graph`, `on_swapchain_recreated`, `on_event`, `shutdown`).
   - Per-frame context struct (swapchain image/view, extent, frame index, timing).
4. Wire concrete app targets.
   - Real `triangle` and `mesh_simple` executables linked to `varre_app_core`, `varre_engine`, `varre_assets`, and SDL3.
5. Port `triangle` first as pass-mode reference.
   - Fully dynamic state + shader-object binding through pass mode.
6. Port `mesh_simple` second.
   - Integrate model upload dependency waits and scene input handling.
7. Prepare for ImGui integration without enabling it yet.
   - Add no-op UI hook interfaces so later ImGui backend integration does not require runtime refactors.
8. Add app-level smoke validation.
   - Startup/shutdown, resize/minimize/recreate, and error-path diagnostics.

### Explicitly Not Implemented Yet

- `varre-iced-renderer`
- `hello_iced` app target

## Build

```bash
cmake -S . -B build
cmake --build build -j4
```

Run tests (if enabled):

```bash
ctest --test-dir build --output-on-failure
```

## Repository Layout

- `varre-assets/` asset codegen and generated asset libraries
- `varre-engine/` engine library + runtime tests
- `varre-app/` app target scaffolding (migration in progress)
- `cmake/` shared CMake options/modules

## Deferred Plan

- Post-app-migration engine structure cleanup is tracked in `docs/post_app_migration_engine_cleanup.md`.

## Notes

This is an active conversion workspace, not a feature-complete engine yet. APIs, targets, and generated outputs may evolve as Rust functionality is progressively ported to C++.
