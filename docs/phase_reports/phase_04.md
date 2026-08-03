# Phase 04: C++ typed IR and structural graph verification

## Scope

Milestone 4 adds a typed, read-only C++ representation of ForgeIR graph schema version `1.0`, JSON
loading with nlohmann/json, overflow-safe descriptor sizing, structural and topological verification,
the `inspect-graph` CLI command, a read-only pybind11 summary function, C++ negative tests, and a
Python-export-to-C++-load integration test.

This milestone does not execute, schedule, rewrite, or optimise graph operations. It adds no tensor
storage, runtime, compiler pass, GPU integration, or MLIR integration.

## Implemented contract

The typed model contains `DataType`, `Shape`, `TensorDescriptor`, `ValueKind`, `Value`,
`OperationType`, `Operation`, `Graph`, `GraphLoader`, `Diagnostic`, `Status`, and `Result<T>`.
Dimensions are `int64_t`; element counts, byte sizes, and parameter-byte accumulation are checked
before multiplication or addition. No type owns a raw pointer.

The loader supports exactly `Input`, `Parameter`, `Constant`, `MatMul`, `Linear`, `Add`, `Mul`,
`Div`, `RMSNorm`, `GELU`, `Softmax`, `Reshape`, `Transpose`, and `CausalMask`. It rejects malformed
JSON, unsupported schema versions and operations, invalid dtypes and hashes, negative and zero
dimensions, size overflow, duplicate IDs, missing operands or outputs, multiple or missing
producers, cycles, value-kind mismatches, and disconnected required outputs. Stable operation-ID
tie-breaking produces a deterministic topological representation but is not an execution schedule.

## Verified environment

- Host: Windows
- Compiler: MSVC 19.42.34433.0
- Build: Debug, C++17
- Python: 3.11.9 from the repository virtual environment
- clang-format: 22.1.8
- Optional compiled features: CUDA false, HIP false, MLIR false

Local dependency sources were used. No package was installed and no dependency, model, or data was
downloaded.

## Commands and observed outcomes

The repository environment was selected for commands that use Python:

    $env:VIRTUAL_ENV = (Resolve-Path '.venv').Path
    $env:Path = "$env:VIRTUAL_ENV\Scripts;$env:Path"
    $env:PYTHONPATH = "$(Resolve-Path 'python');$(Resolve-Path 'build\windows-msvc-debug\python')"

Windows Debug configuration:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
    .\scripts\windows\configure.ps1 -Configuration Debug

Outcome: CMake configuration and generation succeeded with local nlohmann/json, local GoogleTest,
and pybind11 discovered from the active virtual environment.

Windows Debug build:

    .\scripts\windows\build.ps1 -Configuration Debug

Outcome: all Debug targets built successfully, including `forgeir_core`, `forgeir_cli`,
`forgeir_tests`, and `forgeir_py`. Warnings remained errors for ForgeIR targets.

C++ formatting check:

    $formatter = 'C:\Program Files\LLVM\bin\clang-format.exe'
    $files = @(rg --files cpp bindings tests/cpp | Where-Object { $_ -match '\.(cpp|hpp|h)$' })
    & $formatter --dry-run --Werror $files

Outcome: exit code 0 with no formatting violations.

Python formatting, lint, and typing:

    python -m ruff format --check python tests
    python -m ruff check python tests
    python -m mypy python\forgeir

Outcomes: all 18 Python files were already formatted after the configured formatter was applied;
Ruff reported `All checks passed!`; mypy reported no issues in 11 source files.

C++ and binding tests:

    .\scripts\windows\test.ps1 -Configuration Debug

Outcome: `100% tests passed, 0 tests failed out of 24`. Coverage includes the valid golden graph,
malformed JSON, missing operand/output IDs, cycle detection, element-count and byte-size overflow,
negative and zero dimensions, unsupported operations and schema versions, duplicate values, invalid
dtypes, invalid graph and parameter hashes, graph-hash mismatch, disconnected outputs, and stable
topological order. Existing version and doctor tests, the new CLI inspection test, and the Python
binding smoke test also passed.

Explicit cross-language integration:

    python -m pytest tests\python\test_cpp_graph_integration.py -q

Outcome: `1 passed`. A deterministic Python FX export was loaded through the C++ pybind11 function,
and schema version, all counts, operation histogram, and estimated parameter bytes matched the Python
contract.

Full Python regression suite:

    python -m pytest tests\python -q

Outcome: `31 passed`.

Real transformer graph inspection:

    .\build\windows-msvc-debug\bin\forgeir_cli.exe inspect-graph artifacts\graphs\milestone_03\default\tiny_transformer_block.graph.json

Outcome: the embedded canonical graph hash verified and inspection succeeded.

Read-only Python graph summary:

    python -c "import json, forgeir_py; print(json.dumps(forgeir_py.graph_summary(r'artifacts\graphs\milestone_03\default\tiny_transformer_block.graph.json'), sort_keys=True))"

Outcome: the Python binding returned the same summary as the CLI.

During implementation, the first compile detected invalid lvalue calls to an rvalue-qualified
`Result<T>::take_value`; the API was corrected and the complete build/test sequence above passed. The
first repository-wide Ruff formatter check identified five pre-existing formatting-only differences
in Milestone 2/3 Python files; the configured formatter was applied without changing behavior, and
all lint, typing, and regression tests then passed.

## Verified graph summary

- Schema version: `1.0`
- Inputs: 1
- Outputs: 1
- Values: 35
- Operations: 35
- Estimated parameter bytes: 656384
- Operation histogram:
  - `Add`: 2
  - `CausalMask`: 1
  - `Constant`: 1
  - `GELU`: 1
  - `Input`: 1
  - `Linear`: 6
  - `MatMul`: 2
  - `Mul`: 1
  - `Parameter`: 8
  - `RMSNorm`: 2
  - `Reshape`: 4
  - `Softmax`: 1
  - `Transpose`: 5

## Artifact paths

- Real inspected graph: `artifacts/graphs/milestone_03/default/tiny_transformer_block.graph.json`
- Normative schema: `config/graph/forgeir_graph.schema.json`
- Milestone 4 golden graph: `tests/golden/milestone_04_valid_graph.json`
- C++ IR architecture: `docs/architecture/cpp_ir.md`

## Environment limitation

This host verified the Windows MSVC Debug preset only. The existing Linux GCC presets were not run
because no Linux environment is available in this session. CUDA, HIP, and MLIR remain disabled and
were neither built nor tested, as required by the milestone scope.
