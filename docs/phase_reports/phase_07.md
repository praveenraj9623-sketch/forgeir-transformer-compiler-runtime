# Phase 07: Deterministic scheduling and static tensor-memory planning

## Scope

Milestone 7 adds a deterministic topological execution schedule, tensor lifetime analysis, a checked
aligned reusable arena planner, JSON/CSV/SVG evidence, and the `plan-memory` CLI command.

The milestone does not execute graph operators, allocate the described arena, copy tensor data,
schedule a device, implement GPU or MLIR behavior, or run a memory benchmark. Reported byte counts
are static calculations from verified graph shapes and dtypes.

## Implementation

`build_execution_schedule` independently constructs dependency edges and uses Kahn ordering with
stable operation-ID tie-breaking. Relative graph order is added as a constraint between
side-effect-marked operations. Schedule entries retain stable operation IDs, semantic names, types,
inputs, and outputs.

For every graph value, planning records definition, first-use, and final-use indices; an inclusive
lifetime interval; raw and aligned byte sizes; alignment; storage class; protection and immutability
flags; and an optional arena offset. A buffer whose final use is operation `N` remains live while
operation `N` defines its output. Declared outputs extend to the one-past-final-operation boundary.

Storage ownership is explicit:

- inputs are protected external buffers;
- parameters and constants are immutable external buffers;
- intermediates use reusable arena storage;
- declared outputs use dedicated arena storage retained beyond the schedule and never selected from
  a reusable free block.

The CPU default is 64-byte alignment. `MemoryPlannerOptions` permits a backend name and power-of-two
alignment override. Managed tensors are ordered by definition index and stable value ID. Expired
blocks are coalesced; deterministic best fit selects the smallest sufficient block and then the
lowest offset. Output allocation always extends the arena.

All descriptor multiplication, alignment rounding, byte accumulation, block range, arena growth,
and live-byte accumulation is checked. Planning rejects nonpositive/dynamic dimensions, missing
lifetime information, missing definitions, cycles, invalid alignment, size overflow, live allocation
overlap, output-region reuse, and invalid external-storage ownership.

The CLI writes `schedule.json`, `memory_plan.json`, `timeline.csv`, and a deterministic `timeline.svg`
generated from the actual inclusive lifetimes and arena ranges. No graph schema change was required;
the schedule and memory plan each use their own report schema version `1.0`.

## Files created

- `cpp/include/forgeir/runtime/execution_schedule.hpp`
- `cpp/src/runtime/execution_schedule.cpp`
- `cpp/include/forgeir/runtime/memory_planner.hpp`
- `cpp/src/runtime/memory_planner.cpp`
- `tests/cpp/memory_planner_test.cpp`
- `tests/python/test_memory_planning.py`
- `docs/architecture/memory_planner.md`
- `docs/phase_reports/phase_07.md`

`CMakeLists.txt` and `cpp/src/core/cli_main.cpp` were updated to build the planner, register tests, and
provide the CLI surface. Four real-plan artifacts were generated under
`artifacts/reports/milestone_07/tiny_transformer_O2`.

## Verified environment

- Host: Windows
- Compiler: MSVC 19.42.34433.0
- Build: Debug, C++17
- ForgeIR: 0.1.0
- Python: 3.11 from the existing repository virtual environment
- Optional compiled features: CUDA false, HIP false, MLIR false
- Planner backend: CPU
- Planner alignment: 64 bytes

No package was installed and no dependency, model, or data was downloaded.

## Commands and observed outcomes

The repository environment was selected with:

    $env:VIRTUAL_ENV = (Resolve-Path '.venv').Path
    $env:Path = "$env:VIRTUAL_ENV\Scripts;$env:Path"
    $env:PYTHONPATH = "$(Resolve-Path 'python');$(Resolve-Path 'build\windows-msvc-debug\python')"

C++ formatting:

    $formatter = 'C:\Program Files\LLVM\bin\clang-format.exe'
    $cppFiles = @(rg --files cpp bindings tests/cpp | Where-Object { $_ -match '\.(cpp|hpp|h)$' })
    & $formatter --dry-run --Werror $cppFiles

Outcome: exit code 0 with no formatting violations.

Python formatting, lint, and typing:

    python -m ruff format --check python tests
    python -m ruff check python tests
    python -m mypy python\forgeir

Outcomes: all 22 Python files were formatted; Ruff reported `All checks passed!`; mypy reported
`Success: no issues found in 12 source files`.

Windows Debug configuration and build:

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\configure.ps1 -Configuration Debug
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -Configuration Debug

Outcome: configuration used local dependency sources and active-environment pybind11. All strict
warning-as-error targets built successfully; the final validation build reported
`ninja: no work to do`.

Complete C++ and CTest suite:

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\test.ps1 -Configuration Debug

Outcome: `100% tests passed, 0 tests failed out of 64`. Thirteen new CTest cases cover schedule
determinism, definition/first/final-use calculations, inclusive lifetimes, valid reuse, overlap
rejection, dedicated outputs, 64/128-byte alignment, invalid alignment, exact deterministic best-fit
selection, repeated-plan identity, naive bounds, immutable external parameters/constants, byte-size
overflow, unresolved dimensions, missing lifetime information, and the CLI artifact command. All
earlier compiler, optimizer, graph-loader, CLI, and binding tests also passed.

Complete Python and integration suite:

    python -m pytest tests\python -q

Outcome: `38 passed`. The new integration test invokes the CLI twice, requires byte-identical
schedule/plan/CSV/SVG artifacts, checks every live interval against every arena range, verifies
alignment and output protection, validates external storage roles, confirms calculated bounds, and
parses the SVG as XML. All earlier reference, export, cross-language, verification, and optimization
tests passed.

Optimized TinyTransformer plan generation:

    .\build\windows-msvc-debug\bin\forgeir_cli.exe plan-memory artifacts\graphs\milestone_06\O2\tiny_transformer_block.graph.json --output-dir artifacts\reports\milestone_07\tiny_transformer_O2

Outcome: exit code 0. The CLI semantically verified graph hash
`f0dced73e7b6868bb9cc2e82ae0e12bec321cdd5bdadb60601285e7813a36dcd`, built a 35-operation
schedule, planned all 35 graph values, and wrote all four requested artifacts.

Independent artifact validation loaded the written JSON/CSV/SVG, recomputed the naive and peak-live
totals, checked all allocation/lifetime pairs, checked all output regions against every other region,
checked all alignments, counted CSV rows, and parsed the SVG document.

Outcome:

- allocation overlap check: passed;
- output protection check: passed;
- alignment check: passed;
- schedule operations: 35;
- tensor lifetimes: 35;
- arena-managed tensors: 25;
- external tensors: 10;
- timeline CSV rows: 35;
- timeline SVG valid XML: true.

Repository hygiene:

    rg -n "TODO|NotImplementedError|fake return" cpp\include\forgeir\runtime cpp\src\runtime tests\cpp\memory_planner_test.cpp tests\python\test_memory_planning.py

Outcome: no forbidden critical-path placeholder was found.

## Static TinyTransformer memory-plan calculations

| Calculation | Bytes |
|---|---:|
| Naive independently aligned managed allocations | 950272 |
| Planned aligned arena | 262144 |
| Peak simultaneously live aligned tensors | 229376 |

The calculated reuse ratio is `950272 / 262144 = 3.625`. The corresponding static reuse fraction is
`0.7241379310344828`, or approximately 72.41% fewer planned arena bytes than the naive independently
aligned total. Inputs, parameters, and constants are external and excluded from both comparison
totals. These are graph-plan calculations, not measured CPU/GPU memory usage or performance claims.

## Generated artifacts

- `artifacts/reports/milestone_07/tiny_transformer_O2/schedule.json`
  - SHA-256: `0e2ffb19a45ab73190fecaaa2963570e86f4035fc0ec0d1376dc1f32bba16cf0`
- `artifacts/reports/milestone_07/tiny_transformer_O2/memory_plan.json`
  - SHA-256: `ae1f6a034f497b970c54d3ae8d785990f009aa9b251cb705a20bb17b552ef68c`
- `artifacts/reports/milestone_07/tiny_transformer_O2/timeline.csv`
  - SHA-256: `15c630500bbdcfe993fe2835d245c263a5c7ebda8d6ffa25d1a9b9154289f23c`
- `artifacts/reports/milestone_07/tiny_transformer_O2/timeline.svg`
  - SHA-256: `41aeca5649099ce2ec0d2e1635cbd0b10e1f9c02bc255f2bb86d7d185a806a9d`

## Environment limitation

This session verified the Windows MSVC Debug preset. No Linux execution environment was available,
so Linux GCC presets were not run. CUDA, HIP, and MLIR remained disabled and were not built or tested.
The planner did not allocate or measure GPU memory and makes no hardware-memory or performance claim.
