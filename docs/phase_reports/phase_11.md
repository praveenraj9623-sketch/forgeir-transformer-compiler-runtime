# Phase 11: Optional MLIR/StableHLO textual bridge

## Scope

Milestone 11 adds an optional, deterministic textual lowering bridge for a controlled
ForgeIR-to-StableHLO/MLIR subset. It does not add an MLIR dependency to the normal build, implement
a complete MLIR compiler, define a custom production dialect, execute MLIR, or extend operator
semantics.

## Delivered contracts

- `FORGEIR_ENABLE_MLIR` remains `OFF` by default. With `OFF`, no bridge source or external MLIR
  library is compiled, `doctor` reports `mlir=false`, and `emit-mlir` returns the structured
  `mlir_bridge_disabled` error without creating an output file.
- With `ON`, `forgeir_core` includes the textual bridge and `doctor` reports `mlir=true`. The bridge
  still links no MLIR or StableHLO library.
- Lowering begins only after the existing semantic verification pipeline succeeds.
- Function inputs, parameters, outputs, constants, MatMul, Linear, Add, Mul, Div, Reshape,
  Transpose, exact GELU, RMSNorm, and valid-axis stable Softmax have deterministic typed lowerings.
- StableHLO broadcasting is explicit. MatMul/Linear use `dot_general`; GELU, RMSNorm, and Softmax
  are documented primitive decompositions.
- Every emitted tensor type is ranked and explicit. Stable value IDs are preserved as SSA names,
  and every source operation has a deterministic `forgeir.op_id` comment.
- `CausalMask` is deliberately unsupported in this milestone. It returns
  `mlir.unsupported_operation` with operation and value IDs; it is never skipped or mocked.
- The CLI detects `stablehlo-opt` or `mlir-opt` on `PATH`. When neither exists, it reports exactly
  `tool unavailable`, leaves all external-success fields false, and relies only on deterministic
  golden and structural tests.

## Files created

- `cpp/include/forgeir/mlir/stablehlo_lowering.hpp`
- `cpp/src/mlir/stablehlo_lowering.cpp`
- `tests/cpp/mlir_bridge_test.cpp`
- `tests/golden/milestone_11_linear.mlir`
- `docs/architecture/mlir_bridge.md`
- `docs/phase_reports/phase_11.md`

The milestone also updates `CMakeLists.txt`, `cmake/build_config.hpp.in`,
`cpp/src/core/cli_main.cpp`, and `tests/cpp/core_smoke_test.cpp` to connect the feature flag, CLI,
configuration diagnostic, and feature-state migration coverage.

## Commands and outcomes

The active repository Python environment and MSVC/Ninja environment were initialized without
installing or downloading anything:

```powershell
$env:VIRTUAL_ENV=(Resolve-Path '.venv').Path
$env:Path="$env:VIRTUAL_ENV\Scripts;$env:Path"
. .\scripts\windows\common.ps1
Enable-ForgeIRWindowsToolchain
$forgeirPython=Get-ForgeIRActivePython
$pybind11Dir=& $forgeirPython -m pybind11 --cmakedir
```

The optional bridge configuration was created and built with:

```powershell
cmake -S . -B build\windows-msvc-debug-mlir -G Ninja `
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=cl -DBUILD_TESTING=ON `
  -DFORGEIR_ENABLE_CUDA=OFF -DFORGEIR_ENABLE_HIP=OFF `
  -DFORGEIR_ENABLE_MLIR=ON "-Dpybind11_DIR=$pybind11Dir"
cmake --build build\windows-msvc-debug-mlir -- -j1
```

Outcome: configuration and all targets succeeded under MSVC 19.42.34433.0 with `/W4 /WX`.

Targeted bridge and CLI tests:

```powershell
ctest --test-dir build\windows-msvc-debug-mlir `
  -R "MlirBridge|forgeir_cli_emit_mlir" --output-on-failure
```

Outcome: 7/7 passed, covering golden determinism, ranked types and shapes, source IDs, the required
operation subset and decompositions, structured unsupported failure, tool-unavailable truthfulness,
and the enabled CLI.

Complete MLIR-enabled C++/CLI suite:

```powershell
ctest --test-dir build\windows-msvc-debug-mlir --output-on-failure
```

Outcome: 84/84 passed in 2.61 seconds. This includes all existing CPU tests and the six new bridge
GoogleTests plus the enabled CLI CTest.

Enabled CLI emission:

```powershell
build\windows-msvc-debug-mlir\bin\forgeir_cli.exe emit-mlir `
  tests\golden\milestone_04_valid_graph.json `
  --output build\windows-msvc-debug-mlir\milestone_11_linear.generated.mlir
```

Outcome: exit code 0. The JSON reported `available=false`, `syntax_verified=false`,
`canonicalization_attempted=false`, `canonicalization_succeeded=false`, and
`status="tool unavailable"`. The generated file and committed golden fixture both have SHA-256
`1b8fd6a945a31b464345f9866605b856bee681b7d9d76dc59bbf4cde4ece68be`.

The standard feature-disabled configuration was then rebuilt from its normal preset:

```powershell
cmake --preset windows-msvc-debug "-Dpybind11_DIR=$pybind11Dir"
cmake --build --preset windows-msvc-debug -- -j1
ctest --preset windows-msvc-debug --output-on-failure
```

Outcome: the configure summary showed `FORGEIR_ENABLE_MLIR="OFF"`; all targets built and 78/78
tests passed in 2.36 seconds. `forgeir_cli doctor` reported `mlir=false`. Calling `emit-mlir` in this
build returned exit code 3 with `mlir_bridge_disabled` and created no output file.

Python checks and complete integration suite, using the normal debug binding:

```powershell
$env:PYTHONPATH="$(Resolve-Path 'build\windows-msvc-debug\python');$(Resolve-Path 'python')"
python -m ruff format --check python tests
python -m ruff check python tests
python -m mypy python\forgeir
python -m pytest tests\python -q
```

Outcome: 38 files were already formatted, Ruff passed, mypy passed 25 source files, and 69/69
Python tests passed in 18.39 seconds.

C++ format verification:

```powershell
$clangFormat = `
  'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\Llvm\x64\bin\clang-format.exe'
$forgeirFiles = @(rg --files cpp bindings tests\cpp -g '*.cpp' -g '*.hpp')
& $clangFormat --dry-run --Werror $forgeirFiles
```

Outcome: passed.

External validation-tool probe:

```powershell
Get-Command stablehlo-opt -ErrorAction SilentlyContinue
Get-Command mlir-opt -ErrorAction SilentlyContinue
```

Outcome: neither command was available. External MLIR syntax validation and external
canonicalization/CSE were therefore not run and are not marked successful.

## Environment limitations and validation corrections

The C: drive had only about 64 MB free when the separate optional Debug build first linked. The
linker correctly failed with insufficient-space errors. Only recoverable ForgeIR build products
were cleaned with CMake's `clean` target, first for Release and then by swapping between the normal
and optional Debug build directories. No source, configuration, test, golden, documentation, or
third-party source was removed. Serial `-j1` linking then succeeded in each feature state.

The first complete MLIR-enabled suite exposed a pre-existing smoke assertion that required the MLIR
diagnostic flag always to be false. The test was migrated to compare against the generated
`FORGEIR_MLIR_COMPILED` value. It now verifies both the default-disabled and enabled contracts.

## Final status

Milestone 11 is complete. The CPU-only project builds and passes with no MLIR tools installed. The
optional textual bridge builds and passes its golden, type/shape, determinism, decomposition, CLI,
and structured-error coverage. External MLIR validation was not available on this host, so no
external validation success is claimed.
