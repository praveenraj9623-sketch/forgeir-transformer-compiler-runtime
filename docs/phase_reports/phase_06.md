# Phase 06: Deterministic graph optimization passes

## Scope

Milestone 6 adds deterministic C++ graph rewriting, O0/O1/O2 pipelines, rewrite evidence, canonical
optimized-graph serialization, pass reports, before/after DOT files, and a Python reference evaluator
for semantic-equivalence testing.

The milestone does not add C++ operator execution, scheduling, GPU or MLIR code, transformer changes,
or benchmark and performance claims. `FuseResidualRMSNormPass` was not added because the current
operation contract cannot express that fusion with fully specified and testable semantics.

## Implementation

The implemented transformations are:

- `CanonicalisationPass`: normalizes negative axes and known default attributes, removes redundant
  false side-effect markers, and restores stable topological order while retaining relative order
  between side-effect-marked operations.
- `ConstantFoldingPass`: folds only side-effect-free scalar Add, Mul, Div, or exact GELU expressions
  with literal constant operands. Integer overflow and division by zero prevent a fold. The maximum
  folded tensor bound is 1,024 elements; this milestone accepts only the stricter scalar subset.
- `DeadCodeEliminationPass`: retains declared inputs, declared outputs, side-effect operations, and
  all transitively required producers.
- `RedundantReshapeEliminationPass` and `RedundantTransposeEliminationPass`: remove only identity
  layout operations whose intermediate output is not a declared graph output.
- `FuseBiasGELUPass`: safely fuses single-use Linear/MatMul, rank-one bias Add, and exact GELU chains
  into the existing terminal operation while retaining its output ID and semantic name.

Every rewrite contains a reason and ordered added, removed, modified, and removed-value ID lists.
`PassManager` performs semantic verification before and after every pass. The CLI reloads each written
graph through `GraphLoader`, which rechecks structural contracts and the canonical graph hash. Golden
tests rerun every transformation and require the second run to report `changed=false`.

Optimization levels are:

- O0: verification only.
- O1: canonicalisation, constant folding, dead-code elimination.
- O2: O1 plus redundant reshape/transpose elimination and safe bias-GELU fusion.

The Python CPU reference evaluator supports the controlled graph-schema operations and both verified
fused projection forms. It validates graph hashes, parameter keys, and every produced tensor shape
and dtype, and fails explicitly on an unsupported contract.

## Files created

Core graph writing and optimization:

- `cpp/include/forgeir/ir/graph_writer.hpp`
- `cpp/src/ir/graph_writer.cpp`
- `cpp/include/forgeir/passes/canonicalisation_pass.hpp`
- `cpp/include/forgeir/passes/constant_folding_pass.hpp`
- `cpp/include/forgeir/passes/dead_code_elimination_pass.hpp`
- `cpp/include/forgeir/passes/redundant_reshape_elimination_pass.hpp`
- `cpp/include/forgeir/passes/redundant_transpose_elimination_pass.hpp`
- `cpp/include/forgeir/passes/fuse_bias_gelu_pass.hpp`
- `cpp/include/forgeir/passes/optimization.hpp`
- `cpp/src/passes/canonicalisation_pass.cpp`
- `cpp/src/passes/constant_folding_pass.cpp`
- `cpp/src/passes/dead_code_elimination_pass.cpp`
- `cpp/src/passes/redundant_reshape_elimination_pass.cpp`
- `cpp/src/passes/redundant_transpose_elimination_pass.cpp`
- `cpp/src/passes/fuse_bias_gelu_pass.cpp`
- `cpp/src/passes/optimization.cpp`
- `cpp/src/passes/rewrite_utils.hpp`
- `cpp/src/passes/rewrite_utils.cpp`

Reference validation and tests:

- `python/forgeir/reference/evaluator.py`
- `tests/cpp/optimization_pass_test.cpp`
- `tests/python/test_optimization_equivalence.py`
- `tests/golden/milestone_06_pass_expectations.json`

Documentation:

- `docs/architecture/optimization_passes.md`
- `docs/phase_reports/phase_06.md`

Generated artifacts are listed below. Existing IR, loader, verifier, pass-manager, CLI, Python package
exports, CMake target sources, graph-contract documentation, and deterministic attention test setup
were updated to support and verify the new behavior.

## Verified environment

- Host: Windows
- Compiler: MSVC 19.42.34433.0
- Build: Debug, C++17
- ForgeIR: 0.1.0
- Python: 3.11 from the existing repository virtual environment
- Optional compiled features: CUDA false, HIP false, MLIR false

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

Outcomes: all 21 Python files were formatted; Ruff reported `All checks passed!`; mypy reported
`Success: no issues found in 12 source files`.

Windows Debug configuration and build:

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\configure.ps1 -Configuration Debug
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -Configuration Debug

Outcome: the Windows MSVC Debug preset configured with local GoogleTest and nlohmann/json sources and
pybind11 discovered from the active virtual environment. All targets built successfully with ForgeIR
warnings treated as errors; the final build reported `ninja: no work to do` after validation rebuilds.

Complete C++ and CTest suite:

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\test.ps1 -Configuration Debug

Outcome: `100% tests passed, 0 tests failed out of 51`. Ten Milestone 6 cases cover pass goldens,
side-effect safety, declared-output retention, Linear and MatMul fusion, O0/O1/O2 combined pipelines,
determinism, rewrite reasons, and idempotence. All earlier C++ and CLI tests also passed.

Complete Python and integration suite:

    python -m pytest tests\python -q

Outcome: `37 passed`. The suite includes original/optimized transformer equivalence at all three
levels, JSON Schema and graph-hash validation, and exact before/after equality for numerically active
Linear and MatMul bias-GELU fusion graphs. The first full-suite run exposed an older attention test
that constructed parameters from ambient global RNG state. Its fixture now uses the permanent seed
42 inside a forked RNG scope; the complete suite passed after that deterministic-fixture correction.

Real transformer optimization:

    .\build\windows-msvc-debug\bin\forgeir_cli.exe optimize artifacts\graphs\milestone_03\default\tiny_transformer_block.graph.json --level O0 --output artifacts\graphs\milestone_06\O0\tiny_transformer_block.graph.json
    .\build\windows-msvc-debug\bin\forgeir_cli.exe optimize artifacts\graphs\milestone_03\default\tiny_transformer_block.graph.json --level O1 --output artifacts\graphs\milestone_06\O1\tiny_transformer_block.graph.json
    .\build\windows-msvc-debug\bin\forgeir_cli.exe optimize artifacts\graphs\milestone_03\default\tiny_transformer_block.graph.json --level O2 --output artifacts\graphs\milestone_06\O2\tiny_transformer_block.graph.json

All commands exited 0. O0 reported no change. O1 and O2 each recorded three canonical axis rewrites.
All pass-stage verifications succeeded with zero errors and zero warnings.

Semantic and artifact validation used the default deterministic model, input, Milestone 2 NPZ
weights, JSON Schema draft 2020-12 validation, embedded graph-hash verification, and
`forgeir.reference.evaluate_graph` for the original and each optimized graph.

Outcome: the reference block, original graph, O0 graph, O1 graph, and O2 graph were exactly equal for
the tested deterministic input; every maximum absolute difference was `0.0`. Every expected graph,
pass report, before DOT, and after DOT artifact existed.

Repository hygiene:

    rg -n "TODO|NotImplementedError|fake return" cpp\include\forgeir\ir cpp\include\forgeir\passes cpp\src\ir cpp\src\passes python\forgeir\reference tests\cpp\optimization_pass_test.cpp tests\python\test_optimization_equivalence.py

Outcome: no forbidden critical-path placeholder was found.

## Real transformer operation counts and hashes

| Level | Changed | Operations | Difference | Graph hash | Semantic maximum absolute difference |
|---|---:|---:|---:|---|---:|
| O0 | false | 35 -> 35 | 0 | `a8a805297dc8483d2407043342d0d15433d42f73e28fd96b53cce3faaa0ec322` | 0.0 |
| O1 | true | 35 -> 35 | 0 | `f0dced73e7b6868bb9cc2e82ae0e12bec321cdd5bdadb60601285e7813a36dcd` | 0.0 |
| O2 | true | 35 -> 35 | 0 | `f0dced73e7b6868bb9cc2e82ae0e12bec321cdd5bdadb60601285e7813a36dcd` | 0.0 |

The real exported transformer has no bounded constant-only expression, dead node, identity layout
operation, or explicit projection-plus-bias-Add-plus-GELU chain. O1/O2 therefore change only the
three negative axes to canonical nonnegative values and do not remove an operation. Synthetic golden
and semantic tests exercise each removing and fusing transformation; both safe fusion forms change
6 operations to 4 with exact output equality.

## Artifact paths

For each of `O0`, `O1`, and `O2`, directory `artifacts/graphs/milestone_06/<level>` contains:

- `tiny_transformer_block.graph.json`
- `tiny_transformer_block.graph.pass_report.json`
- `tiny_transformer_block.graph.before.dot`
- `tiny_transformer_block.graph.after.dot`

Architecture and test evidence:

- `docs/architecture/optimization_passes.md`
- `tests/golden/milestone_06_pass_expectations.json`
- `tests/cpp/optimization_pass_test.cpp`
- `tests/python/test_optimization_equivalence.py`

## Environment limitation

This session verified the Windows MSVC Debug preset. A Linux execution environment was not available,
so the existing Linux GCC presets were not executed. CUDA, HIP, and MLIR remained disabled and were
not built or tested. No residual/RMSNorm fusion was emitted because its semantics are not representable
by the current verified operation contract.
