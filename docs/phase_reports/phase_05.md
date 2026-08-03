# Phase 05: Type and shape inference with pass infrastructure

## Scope

Milestone 5 adds reusable C++ pass infrastructure, deterministic pass management, shape inference,
dtype propagation, semantic graph verification, JSON verification reports, CLI and pybind11
verification surfaces, invalid-contract tests, and deterministic property-based tests.

This milestone adds no operator execution, fusion, constant folding, dead-code elimination, graph
rewrite, runtime scheduling, GPU integration, or MLIR integration.

## Implementation

`Pass` returns an explicit status, changed flag, and ordered diagnostics. `PassManager` owns passes
with `unique_ptr`, preserves insertion order, performs full semantic verification before and after
every pass, aggregates changed state and diagnostics, and stops immediately after a failed stage.
The standard pipeline runs `ShapeInferencePass`, `DTypePropagationPass`, then
`GraphVerifierPass`.

Graph schema version `1.0` already requires every value descriptor. Inference therefore recomputes
the descriptor implied by each operation and verifies the declaration. It does not silently invent
missing or ambiguous dimensions. All three standard passes are analysis-only and report
`changed=false`, making repeated runs idempotent.

The verifier covers every schema operation and validates MatMul contracting/batch dimensions,
Linear weight and optional bias dimensions, Add/Mul/Div broadcasting, RMSNorm final dimension and
weight, Softmax axis, explicit Reshape element count, Transpose permutations, CausalMask attention
dimensions, dtype compatibility, and output descriptors. Every diagnostic contains an operation ID
and value ID.

## Verified environment

- Host: Windows
- Compiler: MSVC 19.42.34433.0
- Build: Debug, C++17
- Python: 3.11.9 from the repository virtual environment
- Hypothesis: 6.165.0 from the existing locked virtual environment
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

Outcomes: all 19 Python files were formatted; Ruff reported `All checks passed!`; mypy reported
`Success: no issues found in 11 source files`.

Windows Debug configuration and build:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
    .\scripts\windows\configure.ps1 -Configuration Debug
    .\scripts\windows\build.ps1 -Configuration Debug

Outcome: configuration used the existing local dependencies and active-environment pybind11. All
Debug targets built successfully with ForgeIR warnings treated as errors.

Complete C++/CTest suite:

    .\scripts\windows\test.ps1 -Configuration Debug

Outcome: `100% tests passed, 0 tests failed out of 41`. Tests cover deterministic order,
pre-/post-verification, changed aggregation, failure propagation, idempotence, the real transformer,
valid and invalid MatMul, Linear, bias, broadcast, Div, RMSNorm, Softmax, Reshape, Transpose,
CausalMask, and dtype cases, plus all earlier tests and both CLI verification modes.

Explicit property and binding verification tests:

    python -m pytest tests\python\test_pass_verification.py -q

Outcome: `4 passed`. Three tests use Hypothesis with seed 42, no example database, and 40 examples
per property. They cover Add/Mul broadcasting, valid element-preserving reshapes, and invalid
element-changing reshapes. The fourth test verifies the full transformer through pybind11.

Complete Python and integration suite:

    python -m pytest tests\python -q

Outcome: `35 passed`, including the existing Python-export-to-C++-load integration test.

Full transformer verification and durable report generation:

    .\build\windows-msvc-debug\bin\forgeir_cli.exe verify artifacts\graphs\milestone_03\default\tiny_transformer_block.graph.json --report artifacts\reports\milestone_05\tiny_transformer_verification.json

Outcome: exit code 0; verification succeeded with `changed=false`, 0 errors, and 0 warnings. All nine
pre-pass, pass, and post-pass execution records succeeded. The report SHA-256 is
`208e23d7d839c79be998245a33265b154ff9ac1c9efb3a080b44e3e59047bb4f`.

Read-only Python verification API:

    python -c "import json, forgeir_py; r=forgeir_py.verify_graph(r'artifacts\graphs\milestone_03\default\tiny_transformer_block.graph.json'); print(json.dumps({'success':r['success'],'changed':r['changed'],'diagnostic_counts':r['diagnostic_counts']},sort_keys=True))"

Outcome:

    {"changed": false, "diagnostic_counts": {"error": 0, "warning": 0}, "success": true}

Repository hygiene:

    rg -n -i "TODO|NotImplementedError|fake return|\bpass\b" <Milestone-5 critical paths>
    git diff --check

Outcome: no forbidden placeholder implementation was present and the whitespace check passed.
Matches for the word `pass` were legitimate pass-interface names and report fields.

During test iteration, the first negative CTest run exposed unspecified function-argument evaluation
order when moving a diagnostics vector while reading its first message. The message is now copied
before the move. The complete suites above passed after that correction.

## Verified transformer result

- Graph: `artifacts/graphs/milestone_03/default/tiny_transformer_block.graph.json`
- Graph schema: `1.0`
- Graph hash: `a8a805297dc8483d2407043342d0d15433d42f73e28fd96b53cce3faaa0ec322`
- Verification success: true
- Changed: false
- Error diagnostics: 0
- Warning diagnostics: 0
- Standard passes: 3
- Verified pre/pass/post stages: 9

## Artifact paths

- Verification report: `artifacts/reports/milestone_05/tiny_transformer_verification.json`
- Pass architecture: `docs/architecture/pass_manager.md`
- C++ tests: `tests/cpp/pass_manager_test.cpp`
- Property and binding tests: `tests/python/test_pass_verification.py`

## Environment limitation

This session verified the Windows MSVC Debug preset. A Linux environment was not available, so the
existing Linux GCC presets were not executed. CUDA, HIP, and MLIR remained disabled and were not
built or tested, as required by Milestone 5 scope.
