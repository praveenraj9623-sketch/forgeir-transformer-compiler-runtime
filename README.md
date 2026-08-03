# ForgeIR

ForgeIR aims to provide an MLIR-aware transformer graph compiler, heterogeneous execution runtime, custom GPU kernel lab, memory planner, numerical-validation framework, quantization module, and distributed-execution planner.

Status: scaffold only; no implementation or benchmark claims

Planned backends: CPU, CUDA and HIP

Planned compiler flow: PyTorch FX → ForgeIR typed graph → optimisation passes → memory planning → runtime/MLIR bridge
