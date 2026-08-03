# Compiler pass manager and semantic verification

## Scope

Milestone 5 adds reusable compiler-pass infrastructure and semantic analysis for the typed graph from
Milestone 4. It does not execute operations and contains no fusion, constant folding, dead-code
elimination, or other graph transformation.

ForgeIR graph schema version `1.0` requires a shape and dtype on every value. Shape inference and
dtype propagation therefore recompute the descriptor implied by each operation and compare it with
the declared descriptor. They never invent an omitted or ambiguous descriptor. These analysis-only
passes return `changed=false`, so rerunning them is idempotent.

## Pass contract

`Pass` has a stable name and a `run(Graph&)` method. A `PassResult` contains an explicit `Status`, a
changed flag, and ordered diagnostics. Passes are owned by `PassManager` with `unique_ptr`; no owning
raw pointers are used.

`PassManager` runs passes in insertion order. For every registered pass it performs:

1. full semantic pre-pass verification;
2. the pass itself;
3. full semantic post-pass verification.

Each stage records pass name, stage, success, changed state, and diagnostic count. The manager stops
at the first failed stage, preserves diagnostics already produced, and never runs later passes after
failure. Changed state is the logical OR of completed stage results.

The standard verification pipeline registers, in order:

1. `ShapeInferencePass`;
2. `DTypePropagationPass`;
3. `GraphVerifierPass`.

`GraphVerifierPass` combines the shape and dtype analyses. Reusing it for pre- and post-pass checks
keeps one semantic contract for direct verification and future transformation passes.

## Shape rules

- `Input`, `Parameter`, and scalar `Constant` values retain their explicit boundary shapes.
- `MatMul` requires rank at least two, matching contracting dimensions, and broadcast-compatible
  batch dimensions. Its result is the broadcast batch followed by the left row and right column
  dimensions.
- `Linear` requires an input with a final dimension, a rank-two weight shaped
  `[out_features, in_features]`, matching feature attributes, and an optional rank-one bias of
  `[out_features]`. The result replaces the input final dimension with `out_features`.
- `Add`, `Mul`, and `Div` use right-aligned broadcasting: dimensions must be equal or one.
- `RMSNorm` requires a non-scalar input, a rank-one weight matching the final input dimension, a
  final-dimension axis, and positive epsilon.
- `GELU`, `Softmax`, and `CausalMask` preserve shape. Softmax validates its normalized axis.
  CausalMask additionally requires equal final query/key dimensions.
- `Reshape` requires explicit positive target dimensions and equal checked element counts. Values
  such as `-1` are rejected rather than inferred silently.
- `Transpose` requires a rank-matched permutation containing every dimension exactly once.

The checked MatMul, Reshape, Transpose, CausalMask, and broadcast rules jointly verify the attention
head split, score tensor, causal mask, probability tensor, context tensor, and head merge declared by
the exported transformer graph.

## Dtype rules

Shape-only operations preserve dtype. MatMul, Linear, Add, and Mul require one common numeric dtype.
Div requires float32 to avoid ambiguous integer-division semantics. RMSNorm, GELU, Softmax, and
CausalMask require float32. Scalar Constant dtype is inferred from its JSON boolean, integer, or
floating value and checked against its output. Boundary Input and Parameter dtypes remain explicit.

## Diagnostics and report

Every diagnostic contains severity, stable code, message, operation ID, and value ID. Diagnostics
are emitted in deterministic operation order. A verification report contains:

- report and graph schema versions;
- graph hash;
- overall success and changed state;
- status code and message;
- error and warning counts;
- ordered diagnostics;
- ordered pass-stage execution records.

Run verification and print the JSON report with:

    forgeir_cli verify <graph>

An optional durable report path is supported:

    forgeir_cli verify <graph> --report <path>

Python exposes the same read-only report as `forgeir_py.verify_graph(path)`. Loading errors raise a
Python value error; semantic failures return a report with `success=false` and contextual
diagnostics.
