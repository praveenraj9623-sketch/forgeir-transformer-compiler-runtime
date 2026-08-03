# Phase 08: Correct float32 CPU execution

## Scope

Milestone 8 adds aligned RAII CPU tensor storage, execution through the Milestone 7 arena plan, an
explicit CPU backend registry, float32 operator kernels, per-operation tracing, pybind11 runtime
surfaces, and numerical validation against PyTorch.

This milestone does not add graph transformations, dynamic shapes, tensors of other dtypes, GPU,
CUDA, HIP, MLIR, quantization, distributed execution, benchmarks, or performance claims.

## Implementation

`TensorStorage` is move-only and owns its aligned allocation through `unique_ptr` with an aligned
deleter. Runtime session loading structurally and semantically verifies the graph, creates the
deterministic schedule and memory plan, allocates one 64-byte-aligned arena, and selects the CPU
backend explicitly. The real O2 TinyTransformer uses the verified Milestone 7 arena size of 262,144
bytes and all 35 planned schedule entries.

Input and parameter NumPy arrays remain external immutable buffers for the synchronous execution.
The binding rejects dtype conversion and non-contiguous arrays instead of copying them. Parameters
are resolved by archive key, validated for shape and byte size, and checked against the graph's raw
content SHA-256 before use. Constants use immutable aligned storage outside the reusable arena.

The CPU backend implements batched MatMul, Linear, Add, Mul, Div, RMSNorm, exact GELU, stable
Softmax, Reshape, materializing Transpose, CausalMask, and the three-input fused bias plus exact GELU
forms for Linear and MatMul. The direct reference MatMul passed its focused gate before the
32-element cache-tiled implementation was enabled. A deterministic test requires tiled and reference
outputs to be exactly equal while preserving increasing contracting-dimension accumulation order.

Execution traces contain operation ID, canonical operation type, kernel name, output shape, elapsed
microseconds, and the planned arena offset or null for external storage. Elapsed values are run
diagnostics, not benchmark evidence.

The pybind11 API is `load_graph`, `execute`, `get_outputs`, and `get_trace`. Intermediate values named
in `capture_values` are copied when defined so subsequent arena reuse cannot alter checkpoint
evidence. Declared graph outputs are always retained.

## Files created

Runtime headers:

- `cpp/include/forgeir/runtime/tensor_storage.hpp`
- `cpp/include/forgeir/runtime/cpu_backend.hpp`
- `cpp/include/forgeir/runtime/runtime_session.hpp`

Runtime sources:

- `cpp/src/runtime/tensor_storage.cpp`
- `cpp/src/runtime/cpu_backend.cpp`
- `cpp/src/runtime/runtime_session.cpp`

Tests and documentation:

- `tests/cpp/cpu_backend_test.cpp`
- `tests/python/test_cpu_runtime.py`
- `docs/architecture/cpu_runtime.md`
- `docs/phase_reports/phase_08.md`

`CMakeLists.txt`, `bindings/python_module.cpp`, and
`python/forgeir/reference/evaluator.py` were updated to build the runtime, expose its API, and return
explicitly requested reference checkpoints without changing the evaluator's default declared-output
behavior.

The generated numerical report is
`artifacts/reports/milestone_08/numerical_parity.json`. It is a validation artifact and remains under
the repository's generated-artifact ignore policy.

## Verified environment

- Host: Windows
- Compiler: MSVC 19.42.34433.0
- CMake: 3.29.5-msvc4
- Build modes: Debug and Release, C++17
- Python: 3.11.9 from the existing repository virtual environment
- PyTorch: 2.13.0+cpu
- NumPy: 2.4.6
- Execution device/backend: CPU
- Optional compiled features: CUDA false, HIP false, MLIR false

No package was installed, no dependency or model was downloaded, and no network fallback ran.

## Commands and observed outcomes

The existing repository environment was selected with:

    $env:VIRTUAL_ENV = (Resolve-Path '.venv').Path
    $env:Path = "$env:VIRTUAL_ENV\Scripts;$env:Path"

### Reference-before-tiled gate

The initial attempt to call bare `cmake --build` after configuration found that `cmake` was not on
the ordinary PowerShell `PATH`; the already-built test binary therefore contained no new tests. The
repository build script, which resolves the configured toolchain, was used for all actual builds.

The reference kernel was then built and run with:

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -Configuration Debug
    .\build\windows-msvc-debug\bin\forgeir_tests.exe --gtest_filter=CpuReferenceMatMul.*:TensorStorage.*

The first run exposed one incorrect hand-calculated expected value in the test: the implementation
returned `4`, while the fixture expected `5`. Manual matrix multiplication confirmed `4`; the test
expectation was corrected. The rerun passed all 3 selected tests.

Only after that pass, the tiled path was enabled and checked with:

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -Configuration Debug
    .\build\windows-msvc-debug\bin\forgeir_tests.exe --gtest_filter=CpuReferenceMatMul.*:CpuTiledMatMul.*:TensorStorage.*

Outcome: all 4 selected tests passed, including exact tiled/reference equality.

### Python operator and transformer validation

The targeted suite ran with the Debug binding:

    $env:PYTHONPATH = "$(Resolve-Path 'build\windows-msvc-debug\python');$(Resolve-Path 'python')"
    python -m pytest tests\python\test_cpu_runtime.py -q

Outcome: `17 passed`. Thirteen parametrized cases compare every required standalone operator and both
fused projection forms with PyTorch. The remaining tests cover real transformer checkpoints and final
output, trace schema and order, IEEE NaN/Inf behavior, missing/tampered/wrong-shape/wrong-dtype and
non-contiguous weights, unknown backends, and output access before execution.

### Formatting, linting, typing, and hygiene

    $formatter = 'C:\Program Files\LLVM\bin\clang-format.exe'
    $cppFiles = @(rg --files cpp bindings tests/cpp | Where-Object { $_ -match '\.(cpp|hpp|h)$' })
    & $formatter --dry-run --Werror $cppFiles
    python -m ruff format --check python tests
    python -m ruff check python tests
    python -m mypy python\forgeir
    rg -n "TODO|NotImplementedError|fake return" cpp\include\forgeir\runtime cpp\src\runtime bindings\python_module.cpp tests\cpp\cpu_backend_test.cpp tests\python\test_cpu_runtime.py
    git diff --check

Outcomes: clang-format passed; all 23 Python files were formatted; Ruff reported `All checks
passed!`; mypy reported no issues in 12 source files; the critical-path placeholder scan found no
matches; the whitespace check passed. Git emitted only the existing Windows LF-to-CRLF working-copy
notices.

### Complete Debug validation

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\configure.ps1 -Configuration Debug
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -Configuration Debug
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\test.ps1 -Configuration Debug
    python -m pytest tests\python -q

Outcomes: Debug configured and built with strict ForgeIR warnings as errors; CTest reported `100%
tests passed, 0 tests failed out of 75`; the full Python and integration suite reported `55 passed`.

### Complete Release validation

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\configure.ps1 -Configuration Release
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -Configuration Release
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\test.ps1 -Configuration Release
    $env:PYTHONPATH = "$(Resolve-Path 'build\windows-msvc-release\python');$(Resolve-Path 'python')"
    python -c "import forgeir_py; print(forgeir_py.__file__)"
    python -m pytest tests\python -q

Outcomes: Release configured and built successfully; CTest reported `100% tests passed, 0 tests
failed out of 75`. The import command confirmed the Release `.pyd`, and the complete Python and
integration suite reported `55 passed` using that binding.

## Numerical parity

The acceptance rule for every finite value is:

    abs(actual - expected) <= 2e-6 + 2e-5 * abs(expected)

Non-finite positions require matching IEEE classification. The PyTorch model, controlled Python
graph evaluator, C++ checkpoint results, and final C++ output all satisfied this contract.

For the final value `v0034`, generated by the Release-binding test:

- maximum absolute error: `5.662441253662109e-07`;
- maximum pointwise relative error: `0.0023864314425736666`.

The pointwise relative denominator is `max(abs(expected), 1e-12)`. Its maximum occurs near zero, so
it can exceed the relative coefficient while the combined absolute-plus-relative acceptance rule
passes. The largest captured-checkpoint absolute error was `2.86102294921875e-06` at attention score
value `v0016`; that checkpoint also passed the combined criterion.

## Environment limitations

The Windows MSVC Debug and Release presets were both executed. A Linux environment was not available,
so the existing Linux GCC presets could not be run in this session. CUDA, HIP, and MLIR remained
disabled and were not built or tested. No GPU hardware result, benchmark, speedup, throughput, or
other performance claim is made.
