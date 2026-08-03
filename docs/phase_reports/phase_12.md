# Phase 12: Optional handwritten NVIDIA CUDA backend (incomplete)

## Status

Milestone 12 is **incomplete**. The reviewable build integration, handwritten CUDA sources,
validation harness, tests, scripts, and Colab workflow are implemented, but this host has no
NVIDIA GPU or CUDA toolkit. Consequently, the CUDA target was not compiled, no CUDA kernel was
executed, no PyTorch CUDA parity error was measured, and no result JSON was created. This phase
must not be marked complete until the documented workflow passes on matching NVIDIA hardware.

## Delivered offline-reviewable scope

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

This milestone does not implement CUDA MatMul/Linear, a device-resident graph arena, complete graph
execution on CUDA, multi-stream scheduling, or a production GPU backend.

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

## Hardware and tool probe

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

There is therefore no NVIDIA GPU model, compute capability, CUDA runtime version, CUDA driver
version, NVCC version, launch geometry, CUDA timing, maximum absolute error, or maximum relative
error to report from this machine.

## CPU-only regression verification

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

## CUDA build and execution gate

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

Outcome: exit code 1 with `RuntimeError: PyTorch reports no CUDA-capable NVIDIA device`.
`benchmarks/results/cuda/milestone_12.json` does not exist; the directory contains only
`.gitkeep`. No fake benchmark or parity artifact was created.

## Notebook validation limitation

The notebook is valid JSON with nbformat 4 metadata and 10 cells. Every subprocess uses
`check=True`, and the notebook has explicit goal, setup, build, check, export, and next-step
sections. Local `nbformat`, `nbclient`, and Jupyter modules are unavailable, and installing them is
outside this milestone's authorization. More importantly, this AMD-only machine cannot execute
the notebook's NVIDIA checks. The notebook has therefore not been marked executed or validated on
hardware.

## Required completion evidence

On a real NVIDIA Colab or Linux host, set the repository URL in
`backends/cuda/notebooks/forgeir_cuda_milestone_12.ipynb` and run every cell top to bottom. A future
completion report must include a successfully built CUDA preset, passing CUDA CTest and PyTorch
CUDA parity cases, `benchmarks/results/cuda/milestone_12.json`, and the actual GPU model, compute
capability, CUDA runtime, driver, NVCC version, launch geometry, maximum absolute error, and maximum
relative error from that same run.

## Final status

Milestone 12 remains incomplete. Offline-reviewable code and failure gates are present and the
CPU-only project is unchanged and fully passing, but no real NVIDIA execution evidence exists.
