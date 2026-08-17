# ForgeIR

ForgeIR aims to provide an MLIR-aware transformer graph compiler, heterogeneous execution runtime, custom GPU kernel lab, memory planner, numerical-validation framework, quantization module, and distributed-execution planner.

Status: experimental systems implementation; verified claims are limited to reproducible phase reports.

Planned backends: CPU, CUDA and HIP

## Verified NVIDIA CUDA validation

ForgeIR's optional handwritten float32 CUDA kernels for GELU, RMSNorm, row Softmax, and
FusedBiasGELU were validated on an NVIDIA Tesla T4. The clean-clone CUDA-enabled suite passed
83/83 CTest tests, and all 36 PyTorch CUDA parity/validation cases passed. The measured maximum
absolute error was `7.152557373046875e-07`; the measured maximum relative error was
`2.2558165948922222e-07`.

These results validate the listed kernels only and do not claim a PyTorch speedup or complete CUDA
graph execution. See [the Milestone 12 phase report](docs/phase_reports/phase_12.md) for provenance,
protocol, hardware metadata, and scope limitations.

Planned compiler flow: PyTorch FX → ForgeIR typed graph → optimisation passes → memory planning → runtime/MLIR bridge
