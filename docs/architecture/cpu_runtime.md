# Float32 CPU runtime

## Scope

Milestone 8 executes verified ForgeIR graph schema `1.0` graphs on the CPU using float32 tensors. It
adds no GPU, MLIR, graph-rewrite, dynamic-shape, quantization, or benchmark behavior. The backend
registry requires the caller to select `cpu`; an unknown backend fails explicitly.

## Storage and ownership

`TensorStorage` is a move-only RAII allocation backed by C++17 aligned `operator new` and aligned
`operator delete`. Allocation size and alignment are validated before allocation. The CPU runtime
uses the Milestone 7 memory plan unchanged: one 64-byte-aligned arena is allocated at the planned
size, every managed value uses its planned offset, reusable intermediates share only non-overlapping
lifetimes, and declared outputs occupy their protected retained regions.

Graph inputs and parameters are non-owning, immutable views for the duration of `execute`. Python
arrays must be exact float32, have the declared shape and byte size, and be C-contiguous. No
`forcecast`, implicit contiguity conversion, or hidden layout copy occurs. Each parameter is selected
by its graph `archive_key`, and its raw C-order bytes must match the operation's `content_sha256`.
Constants are immutable, aligned runtime-owned scalar storage outside the reusable arena.

Every computed output is contiguous. `Transpose` explicitly materializes its declared permutation.
`Reshape` accepts only a contiguous input, preserves element count, and copies the contiguous logical
sequence into its separately planned output allocation. Non-contiguous input views are rejected at
the backend boundary instead of being silently copied.

## Operators

The CPU backend implements:

- batched, broadcast-compatible `MatMul`;
- `Linear` with `[out_features, in_features]` weights;
- right-aligned broadcast `Add`, `Mul`, and float32 `Div`;
- final-axis `RMSNorm` with the graph epsilon;
- exact `GELU` with `approximate="none"`;
- numerically stable `Softmax` on any verified axis;
- `Reshape`, materializing `Transpose`, and `CausalMask` with `masked_value="-inf"`;
- the Milestone 6 three-input Linear/MatMul fused-bias exact-GELU contract.

The reference MatMul uses direct row/column/contracting-dimension loops and float32 accumulation. It
was built and tested before the tiled path was enabled. The tiled kernel uses 32-element tiles while
retaining increasing contracting-dimension accumulation order, and is tested for exact equality with
the reference kernel on deterministic batched tensors. It is a correctness implementation; no
speedup or throughput claim is made.

Softmax subtracts the row maximum before exponentiation. RMSNorm computes
`x * (1 / sqrt(mean(x*x) + epsilon)) * weight`. GELU computes
`0.5 * x * (1 + erf(x / sqrt(2)))`, matching the documented PyTorch exact convention. IEEE NaN and
infinity results propagate; they are not replaced with finite sentinels.

## Execution session and tracing

Loading a session structurally loads, semantically verifies, schedules, and plans the graph before
allocating the arena. Execution follows the deterministic Milestone 7 schedule. Requested
intermediate checkpoints are copied when defined so later arena reuse cannot change them. Declared
outputs are always retained.

Every trace record contains the operation ID, canonical operation type, CPU kernel name, output
shape, elapsed microseconds, and arena offset or null for external storage. Elapsed values are
diagnostic observations from the individual run, not benchmark evidence or performance claims.

The pybind11 surface is:

    session = forgeir_py.load_graph(path, backend="cpu")
    forgeir_py.execute(session, inputs, parameters, capture_values=[])
    outputs = forgeir_py.get_outputs(session, value_ids=[])
    trace = forgeir_py.get_trace(session)

`inputs` maps graph value IDs to NumPy arrays. `parameters` maps graph archive keys to NumPy arrays.
With no `value_ids`, `get_outputs` returns declared graph outputs. A supplied list returns values that
were named in `capture_values` (declared outputs are always available).

## Numerical acceptance

Standalone operators, transformer checkpoints, and the final transformer output are compared with
the PyTorch reference using:

    abs(actual - expected) <= 2e-6 + 2e-5 * abs(expected)

Non-finite reference positions must have matching IEEE classifications. The durable parity report at
`artifacts/reports/milestone_08/numerical_parity.json` records maximum absolute and pointwise relative
errors for every checked value. Pointwise relative error uses
`abs(actual - expected) / max(abs(expected), 1e-12)`; therefore its maximum may exceed the relative
tolerance near zero while the combined acceptance inequality still passes.

## Rejected contracts

Execution fails explicitly for non-float32 graph values, non-contiguous external arrays, shape or
byte-size mismatch, missing or content-hash-mismatched parameters, unsupported GELU approximations,
unsupported causal-mask sentinels, invalid axes or permutations, invalid broadcasting or matrix
contracts, allocation overflow, and an unknown backend. No fallback mock or implicit data conversion
is used.
