# Optional handwritten NVIDIA CUDA backend

## Status and scope

Milestone 12 is implemented for offline review but is not complete. The development host has no
NVIDIA GPU, NVIDIA driver tools, or CUDA compiler, so none of the CUDA sources or parity claims have
been executed locally. No CUDA benchmark JSON exists in the repository.

This work is a focused float32 kernel lab for GELU, final-axis RMSNorm, final-axis row Softmax, and
FusedBiasGELU. It is not a complete CUDA graph runtime. MatMul, Linear, device-resident graph
storage, CUDA memory planning, multi-stream execution, and end-to-end TinyTransformer CUDA
execution are not implemented.

## Build boundary

`FORGEIR_ENABLE_CUDA` defaults to `OFF`. In that configuration:

- CMake does not enable the CUDA language or search for the CUDA toolkit;
- no CUDA header, source, library, Python module, or CUDA test is compiled;
- `forgeir_core`, `forgeir_cli`, `forgeir_py`, and all CPU tests retain their normal C++17 build;
- `doctor` reports `cuda=false`.

With `FORGEIR_ENABLE_CUDA=ON`, CMake enables CUDA 17, requires `CUDAToolkit`, links `CUDA::cudart`,
builds `forgeir_cuda_py`, and compiles CUDA tests. If no architecture is supplied, CMake selects the
native NVIDIA architecture. The generated build diagnostic then reports `cuda=true`.

CUDA Release presets are provided for Linux/GCC and Windows/MSVC. The Colab workflow uses
`linux-gcc-cuda-release` through `scripts/linux/configure_cuda.sh`, `build_cuda.sh`, and
`test_cuda.sh`. Those scripts fail if `nvcc`, `nvidia-smi`, an active Python 3.11 virtual
environment, or a reported NVIDIA GPU is missing.

## Kernel implementation

All four kernels are handwritten CUDA C++ in `backends/cuda/src/cuda_backend.cu`; ForgeIR does not
delegate their computation to ATen or PyTorch:

- `GELU` applies the exact `0.5*x*(1+erf(x/sqrt(2)))` convention elementwise.
- `FusedBiasGELU` adds a rank-one row bias and applies the same exact GELU in one kernel.
- `RMSNorm` assigns one CUDA block per row, reduces the sum of squares through shared memory,
  applies the graph epsilon, and multiplies by the rank-one weight.
- `Softmax` assigns one block per row and performs shared-memory maximum and sum reductions before
  normalization. Subtracting the row maximum is part of the kernel.

The fixed block size is 256 threads. Elementwise kernels use a one-dimensional ceiling-divided
grid and no dynamic shared memory. RMSNorm and Softmax use one grid block per row and 1,024 bytes
of dynamic shared memory. The maximum reviewed row width is 16,384; threads stride across wider
rows within that bound. Odd, smaller-than-block, and non-power-of-two widths are intentional test
cases.

## Validation and error handling

The CUDA surface is float32-only by type. It validates positive static rows and width, checked
element and byte products, operation-specific auxiliary sizes, final-axis Softmax, positive finite
RMSNorm epsilon, C-contiguity, current device ownership, CUDA device/managed memory, and 16-byte
pointer alignment. Unsupported operations and shapes return explicit `Status` errors.

`cudaPointerGetAttributes` verifies device provenance before direct backend launches. Every
fallible allocation, copy, device query, event operation, launch check, and synchronization is
checked. `cudaPeekAtLastError` follows each kernel launch; warm-ups end with a checked device
synchronization, and each measured launch is synchronized through its stop event. RAII deleters
terminate if CUDA resource cleanup itself fails because destructors cannot return a `Status`.

`BackendRegistry::create("cuda")` constructs `CudaBackend` only in a CUDA-enabled build. It can
launch GELU, RMSNorm, and final-axis Softmax on validated device pointers. FusedBiasGELU is exposed
through the dedicated validation/profile API. The existing `RuntimeSession` still owns a CPU arena
and accepts NumPy host buffers, so selecting `cuda` there will fail pointer validation rather than
silently treating host memory as device memory. A device-resident session is future work and is not
claimed here.

## Timing and metadata

`run_cuda_host_profile` performs a positive configurable warm-up count, then records every measured
kernel launch with CUDA start/stop events. Setup, host/device copies, allocation, and report
rendering are outside the kernel event interval. Each result records raw kernel milliseconds and:

- GPU model and selected device ordinal;
- compute capability;
- CUDA runtime and driver integer versions;
- CMake-detected NVCC version;
- block size, grid size, and dynamic shared-memory bytes.

These values become evidence only after the code runs on NVIDIA hardware. The present source and
documentation contain no measured timings or performance claims.

## PyTorch CUDA parity protocol

`python -m forgeir.benchmark.cuda_validation` requires both `torch.cuda.is_available()` and the
compiled `forgeir_cuda_py` module. With deterministic seed 42 it exercises every kernel at widths
1, 3, 7, 31, 127, 257, 1,023, and 4,096, plus an extreme finite-value case. This covers single
element, small, odd, larger-than-block, non-power-of-two, and large supported rows.

PyTorch performs each reference operation on the same selected CUDA device with identical inputs,
weights, bias, epsilon, warm-ups, and measured iteration count. PyTorch and ForgeIR use separate
CUDA-event samples. Outputs must match NaN and infinity classifications and satisfy:

```text
abs(actual - expected) <= 5e-6 + 5e-5 * abs(expected)
```

The command stops at the first failure and writes JSON only after every case passes. A successful
real-hardware run writes `benchmarks/results/cuda/milestone_12.json`; this file must never be copied
from a different run or synthesized by hand.

## Colab workflow

`backends/cuda/notebooks/forgeir_cuda_milestone_12.ipynb` is a top-to-bottom Colab workflow. It
requires the user to set the real Git repository URL, refuses to reuse a stale clone, verifies
`nvidia-smi`, `nvcc`, and PyTorch CUDA, installs only
`backends/cuda/requirements-colab.txt`, builds the CUDA preset, runs all tests, validates the result
schema/status/hardware fields, and exports a ZIP. Every subprocess uses `check=True`; no cell
suppresses a failed command.

The notebook cannot be executed on the current AMD-only host. Its JSON structure and failure
handling are checked offline, but successful Colab execution and the resulting hardware evidence
remain required before this milestone can be marked complete.
