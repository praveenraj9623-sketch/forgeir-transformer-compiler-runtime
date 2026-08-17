# Phase 12: Optional handwritten NVIDIA CUDA backend (complete)

## Status

Milestone 12 is **complete**. The optional backend was built from a clean public clone and executed
on a real Google Colab NVIDIA Tesla T4. The CUDA-enabled CTest suite passed 83/83 tests, and all 36
PyTorch CUDA parity/validation cases passed. The measured result is tracked at
`benchmarks/results/cuda/milestone_12.json`.

The initial implementation and offline review occurred on an AMD-only Windows host. That history
is retained below because it documents the deliberate hardware gate and the absence of fabricated
GPU claims before matching NVIDIA execution became available.

## Delivered implementation scope

- `FORGEIR_ENABLE_CUDA` defaults to `OFF`; the normal CPU build neither enables CUDA nor searches
  for the CUDA toolkit. The generated build diagnostic uses the actual compiled feature state.
- CUDA-enabled Windows/MSVC and Linux/GCC Release presets enable CUDA C++17, require
  `CUDAToolkit`, link `CUDA::cudart`, and compile the optional CUDA binding and tests.
- Handwritten float32 CUDA kernels implement exact GELU, RMSNorm, stable final-axis row Softmax,
  and FusedBiasGELU. No ATen operation implements a ForgeIR kernel.
- Kernel requests validate sizes with checked multiplication, supported width, auxiliary tensor
  shape, epsilon, contiguity at the Python boundary, device ownership, device/managed pointer
  provenance, and 16-byte alignment.
- CUDA allocations, copies, device/runtime/driver queries, event operations, launches, and
  synchronizations have explicit error handling. Device memory and events use RAII.
- Warm-ups are mandatory. Measured kernel launches use CUDA start/stop events and exclude setup,
  allocation, copies, and JSON rendering.
- The CUDA backend is registered only in CUDA-enabled builds. GELU, RMSNorm, and Softmax accept
  validated device pointers through the backend. FusedBiasGELU is available through the focused
  CUDA validation/profile surface because the current graph runtime does not yet own a CUDA arena
  or implement CUDA MatMul/Linear.
- The parity harness uses seed 42, identical inputs and auxiliary tensors for PyTorch CUDA and
  ForgeIR, exact GELU, widths 1, 3, 7, 31, 127, 257, 1,023, and 4,096, and an extreme finite-value
  case. It writes JSON only after every case satisfies the documented combined tolerance.
- The Colab notebook stops on every failed command, checks real hardware and tooling before the
  build, runs the complete tests and parity workflow, validates the result, and exports it.

This milestone does not implement CUDA MatMul/Linear, a complete device-resident graph arena, full
graph CUDA execution, multi-stream scheduling, or a distributed CUDA runtime.

## Files created

- `backends/cuda/include/forgeir/backends/cuda/cuda_backend.hpp`
- `backends/cuda/src/cuda_backend.cu`
- `backends/cuda/src/python_module.cpp`
- `backends/cuda/tests/cuda_backend_test.cu`
- `backends/cuda/notebooks/forgeir_cuda_milestone_12.ipynb`
- `backends/cuda/requirements-colab.txt`
- `python/forgeir/benchmark/cuda_validation.py`
- `tests/python/test_cuda_validation_contract.py`
- `scripts/linux/configure_cuda.sh`
- `scripts/linux/build_cuda.sh`
- `scripts/linux/test_cuda.sh`
- `benchmarks/results/cuda/.gitkeep`
- `docs/architecture/cuda_backend.md`
- `docs/phase_reports/phase_12.md`

The milestone also updates `CMakeLists.txt`, `CMakePresets.json`,
`cmake/ForgeIRWarnings.cmake`, `cmake/build_config.hpp.in`,
`cpp/src/runtime/cpu_backend.cpp`, and `pyproject.toml` to connect the optional build, generated
diagnostic, backend registry, warning policy, test registration, binding, and typing configuration.
The former `.gitkeep` files in populated CUDA source directories were removed.

## Historical offline-development hardware probe

Before the final Colab validation, the local Windows development host was probed with:

Commands:

```powershell
Get-Command nvidia-smi -ErrorAction SilentlyContinue
Get-Command nvcc -ErrorAction SilentlyContinue
Get-CimInstance Win32_VideoController |
  Select-Object Name, DriverVersion, AdapterRAM
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Outcome on 2026-08-03:

- `nvidia-smi`: unavailable
- `nvcc`: unavailable
- display adapters: AMD Radeon RX 6750 XT and AMD Radeon(TM) Graphics, both driver
  `32.0.21045.1000`
- Python: 3.11.9
- PyTorch: `2.13.0+cpu`
- `torch.cuda.is_available()`: `False`

At that stage there was no NVIDIA GPU model, compute capability, CUDA runtime version, CUDA driver
version, NVCC version, launch geometry, CUDA timing, maximum absolute error, or maximum relative
error to report from the local machine.

## Historical CPU-only regression verification

Commands:

```powershell
$env:VIRTUAL_ENV=(Resolve-Path .venv).Path
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\windows\configure.ps1 -Configuration Debug
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\windows\build.ps1 -Configuration Debug
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\windows\test.ps1 -Configuration Debug
```

Outcome: the preset configured with `FORGEIR_ENABLE_CUDA="OFF"`, Ninja had no remaining work,
and 78/78 C++/CLI/pybind CTest cases passed in 2.61 seconds. `forgeir_cli doctor` reported
`"cuda": false`, MSVC 19.42.34433.0, Debug, Windows, and C++17.

Python commands:

```powershell
python -m ruff format --check python tests
python -m ruff check python tests
python -m mypy python\forgeir
python -m pytest tests\python -q
```

Outcome: 40 files were already formatted, Ruff passed, mypy passed 26 source files, and 72/72
Python tests passed in 19.68 seconds. The three new offline CUDA-contract tests are included in
that total; they validate the case matrix, parity-metric calculation, and refusal to create a
result file when CUDA is unavailable.

C++ and script-format commands:

```powershell
$clangFormat = `
  'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\Llvm\x64\bin\clang-format.exe'
& $clangFormat --dry-run --Werror `
  backends/cuda/include/forgeir/backends/cuda/cuda_backend.hpp `
  backends/cuda/src/cuda_backend.cu `
  backends/cuda/src/python_module.cpp `
  backends/cuda/tests/cuda_backend_test.cu `
  cpp/src/runtime/cpu_backend.cpp
bash -n /mnt/c/Users/admin/Desktop/ForgeIR/scripts/linux/configure_cuda.sh `
  /mnt/c/Users/admin/Desktop/ForgeIR/scripts/linux/build_cuda.sh `
  /mnt/c/Users/admin/Desktop/ForgeIR/scripts/linux/test_cuda.sh
```

Outcome: both checks passed.

## Historical local CUDA build and execution gate

CUDA-enabled configure command:

```powershell
cmake --preset windows-msvc-cuda-release
```

Outcome: exit code 1 at `enable_language(CUDA)` with `Failed to find nvcc. Compiler requires the
CUDA toolkit.` This is the expected truthful failure on this host; the build did not fall back to
CPU under a CUDA label.

Parity command, using the normal CPU binding only so package imports resolve:

```powershell
$env:PYTHONPATH="$(Resolve-Path build\windows-msvc-debug\python);$(Resolve-Path python)"
python -m forgeir.benchmark.cuda_validation `
  --output benchmarks\results\cuda\milestone_12.json `
  --warmup 10 --iterations 50
```

Outcome at that time: exit code 1 with
`RuntimeError: PyTorch reports no CUDA-capable NVIDIA device`. No local result was created, and no
fake benchmark or parity artifact was used to close the milestone.

## Historical notebook validation limitation

The notebook was validated offline as JSON with nbformat 4 metadata and 10 cells. Every subprocess
uses `check=True`, and the notebook has explicit goal, setup, build, check, export, and next-step
sections. The AMD-only host could not execute the notebook's NVIDIA checks. That limitation was
subsequently resolved by the final Tesla T4 Colab run described below.

## Final NVIDIA hardware validation

The final validation ran from a clean public clone at commit
`fe04aeeb39940722ec7dc0bc10561e93bef75752` (`fix: make CUDA validation clean-clone reproducible`)
on a Google Colab NVIDIA Tesla T4. The `linux-gcc-cuda-release` preset configured and built with
CUDA 12.5/NVCC. The generated evidence file is:

```text
benchmarks/results/cuda/milestone_12.json
```

Its SHA-256 is `b7fb4cdf1869ff002556fb5e893a77a6634ca3fb5e0fe4d198ffa1dd4be36409`.

Hardware and software metadata recorded by that file:

- GPU: NVIDIA Tesla T4, device ordinal 0
- Compute capability: 7.5
- CUDA runtime version: 12050
- CUDA driver version: 13000
- NVCC version: 12.5.82
- PyTorch version: 2.6.0+cu124
- PyTorch CUDA build: 12.4
- Deterministic seed: 42
- Warm-up iterations per case: 10
- Measured iterations per case: 50

CTest outcome:

- 83/83 tests passed
- 100% tests passed
- 0 failures
- Total CTest time: 7.35 seconds

PyTorch CUDA parity outcome:

- Status: `passed`
- Validation cases: 36
- Combined-tolerance violations: 0 across all cases
- Maximum absolute error: `7.152557373046875e-07`
- Maximum relative error: `2.2558165948922222e-07`
- Configured tolerance: `5e-06 + 5e-05 * abs(expected)`

The result contains CUDA-event timing observations for ForgeIR and PyTorch kernels, but Milestone
12 makes no speedup claim. Its completion claim is correctness and real-hardware execution of the
four documented handwritten kernels.

## Real-hardware portability and clean-clone repairs

The first NVIDIA build exposed two portability defects that the Windows-only offline review could
not reveal:

1. `backends/cuda/src/cuda_backend.cu` now explicitly includes `<math_constants.h>` so
   `CUDART_INF_F` is declared under CUDA 12.5/NVCC.
2. `diagnostic_json()` in `cpp/src/core/cli_main.cpp` is compiled only when
   `FORGEIR_MLIR_COMPILED` is true, preventing GCC `-Werror` from rejecting an unused function in
   the CUDA-enabled, MLIR-disabled build.

The clean-clone run also exposed tests that depended on ignored development graphs. The canonical
35-operation transformer graphs were moved into deterministic committed fixtures under
`tests/fixtures`, C++ and Python tests were redirected to those fixtures, and deterministic seed-42
weights are generated in pytest temporary storage. Feature diagnostic tests now compare CUDA, HIP,
and MLIR states with their generated `FORGEIR_*_COMPILED` macros. This preserves the original
runtime memory-plan assertions while making the complete test suite independent of ignored
`artifacts/` content.

## Remaining scope limitations

Milestone 12 remains a focused optional kernel backend, not a complete CUDA graph runtime. It does
not provide:

- CUDA MatMul or Linear kernels;
- a complete device-resident graph arena;
- full graph CUDA execution;
- multi-stream scheduling; or
- a distributed CUDA runtime.

## Final status

Milestone 12 is complete. The CPU-only build remains optional and independent of CUDA, and the
documented handwritten CUDA kernels have clean-clone build, test, PyTorch parity, and real Tesla T4
execution evidence. The completion claim is limited to the implemented kernel scope above.
