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

## Current Status

- `varre-assets`
  - `model_codegen`: implemented
  - `shader_codegen`: stub (intentionally unimplemented)
  - `varre_assets_models`: generated model asset library
  - `varre_assets_shaders`: placeholder shader asset library
  - `varre_assets`: interface target aggregating assets
- `varre-engine`: CMake/module scaffolding in place, target internals deferred
- `varre-app`: CMake/module scaffolding in place, target internals deferred

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

## Notes

This is an active conversion workspace, not a feature-complete engine yet. APIs, targets, and generated outputs may evolve as Rust functionality is progressively ported to C++.
