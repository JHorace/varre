# AGENTS.md

## Scope
This file defines repository-specific guidance for AI coding agents working in `/home/jsumihiro/repos/varre`.

## Project Goal
Convert `https://github.com/JHorace/vulkan-rust-engine` into a C++/CMake project.

## Explicit Exclusions
- Do not convert or implement `varre-iced-renderer` yet.
- Do not convert or implement the `hello_iced` application target yet.

## Current Structure (Expected)
- Top-level CMake project with nested subprojects:
  - `varre-assets`
  - `varre-engine`
  - `varre-app`
- `varre-assets` currently has:
  - `shader_codegen` executable (stub)
  - `model_codegen` executable (implemented)
  - `varre_assets_shaders` library (stub wiring)
  - `varre_assets_models` library (generated via `model_codegen`)
  - `varre_assets` interface target

## Implementation Rules
- Implement CMake structure first, then target internals.
- Preserve nested module/subproject layout when adding targets.
- Keep placeholder/stub behavior explicit for unfinished components.
- Avoid introducing functionality for excluded targets.

## Build & Validation
- Configure: `cmake -S . -B build`
- Build: `cmake --build build -j4`
- Prefer validating only impacted targets when possible.

## Formatting & Style
- Use `.clang-format` at repo root.
- Maximum line length: `160`.
- Use Doxygen-style comments for public APIs and non-trivial components.

## Editing Guidelines
- Keep changes minimal and scoped to the requested task.
- Do not remove existing generated-file flows unless replacing them with working equivalents.
- Maintain deterministic code generation behavior where applicable (stable ordering, stable identifiers).

## When Adding New Targets
- Add targets in the relevant nested `CMakeLists.txt`, not only at root.
- Keep public headers under `include/varre/...` so includes stay stable (e.g. `#include <varre/assets/models.hpp>`).
- Ensure dependencies are explicit (`add_dependencies`, `target_link_libraries`, include dirs).

