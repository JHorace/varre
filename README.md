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

## Current Status

- `varre-assets`
  - `model_codegen`: implemented
  - `shader_codegen`: stub (intentionally unimplemented)
  - `varre_assets_models`: generated model asset library
  - `varre_assets_shaders`: placeholder shader asset library
  - `varre_assets`: interface target aggregating assets
- `varre-engine`: CMake/module scaffolding in place, target internals deferred
- `varre-app`: CMake/module scaffolding in place, target internals deferred

## Engine Port Roadmap (As Of 2026-04-12)

Planned implementation sequence for `varre-engine`:

1. Create `varre_engine` library target and wire `varre::engine_dependencies`.
2. Define public API headers first (`Engine`, `DeviceContext`, `RenderContext`, config structs, errors).
3. Port Rust module layout with near 1:1 C++ mapping.
4. Implement core Vulkan init path:
   - instance + validation + debug messenger (`vk::raii`)
   - physical device + queue discovery
   - logical device + feature chains (`vk::StructureChain`)
   - queue handles and frame state (`NUM_FRAMES_IN_FLIGHT = 3`)
   - command pools/buffers and one-time submit path
5. Implement surface/swapchain flow behind platform-provided interfaces (no windowing dependency inside engine).
6. Integrate generated assets (`varre_assets` shaders/models), then texture upload path.
7. Port render contexts in order: `triangle`, then `mesh_simple`.
8. Add lifecycle/recreation correctness checks and tests (queue selection, feature composition, asset lookup, smoke init).

Explicit design choices for the port:

- Prefer `vk::raii` ownership over manual Vulkan destroy patterns.
- Keep platform/window adapter boundaries outside `varre-engine`.
- Use VMA for memory allocation from first engine implementation.
- Use structured diagnostics (`fmt`/`spdlog`) rather than ad hoc prints.
- Keep unfinished components as explicit stubs.
- Use dynamic rendering + synchronization2 + `VK_EXT_shader_object` as first-class requirements (no legacy pipeline/renderpass fallback path).

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
- `varre-engine/` engine module scaffolding
- `varre-app/` app module scaffolding
- `cmake/` shared CMake options/modules

## Deferred Plan

- Post-app-migration engine structure cleanup is tracked in `docs/post_app_migration_engine_cleanup.md`.

## Notes

This is an active conversion workspace, not a feature-complete engine yet. APIs, targets, and generated outputs may evolve as Rust functionality is progressively ported to C++.
