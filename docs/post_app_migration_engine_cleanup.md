# Post-App-Migration Engine Cleanup Plan

Status: Deferred until `varre-app` migration is complete.

This document records the engine structure/naming cleanup plan that should be resumed after app targets are fully migrated.

## Goals

- Keep the shader-object-only direction explicit in naming and APIs.
- Reduce `pass_mode` complexity by separating responsibilities.
- Remove temporary compatibility layers once app code is migrated.

## Planned Work

1. Remove temporary shader header shims:
   - drop `include/varre/engine/assets/shaders.hpp` shim once all callsites use `shader_objects.hpp`
   - drop `include/varre/engine/shaders.hpp` shim once all callsites use `shader_objects.hpp`
2. Normalize shader-object naming across engine code:
   - prefer `shader_object*` names over pipeline-era `shader*` naming where practical
   - keep public include paths aligned with canonical headers
3. Split `render_context/pass_mode.cpp` by responsibility:
   - validation/debug checks
   - pass dependency/barrier planning
   - command recording/execution
4. Tighten pass-mode interfaces:
   - keep only app-facing pass API surface public
   - move internal planner/executor helpers behind internal headers
5. Revisit directory boundaries for clarity:
   - keep assets/descriptors/render_context boundaries explicit
   - move cross-cutting helpers to dedicated internal utility locations if still shared

## When To Execute

Run this cleanup only after:

- `triangle` and `mesh_simple` are fully migrated to app-level pass mode, and
- app-facing APIs stabilize enough that renames/removals will not churn migration work.
