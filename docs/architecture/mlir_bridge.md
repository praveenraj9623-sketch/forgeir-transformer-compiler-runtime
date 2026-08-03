# Optional MLIR/StableHLO textual bridge

## Scope and feature boundary

Milestone 11 adds a deterministic textual lowering bridge, not a complete MLIR compiler and not a
custom production dialect. The bridge is compiled only when CMake is configured with:

```text
-DFORGEIR_ENABLE_MLIR=ON
```

The default remains `OFF`. No MLIR or StableHLO headers, libraries, packages, or downloads are
required by either configuration. A normal CPU-only ForgeIR build therefore retains its existing
build and runtime behavior. When the option is disabled, `forgeir_cli emit-mlir` fails explicitly
with the structured code `mlir_bridge_disabled`.

## Lowering contract

`lower_to_stablehlo` first runs the complete ForgeIR semantic verification pipeline. Failed graphs
produce the existing ordered verification diagnostics and no module text. A successful lowering
emits one `module` containing `func.func @main`:

- ForgeIR `Input` and `Parameter` values become ordered function arguments. Parameter bytes remain
  external; the bridge does not embed the NPZ archive.
- Declared ForgeIR outputs become ordered function result types and `func.return` operands.
- Every tensor is a ranked MLIR tensor with an explicit `i1`, `i64`, or `f32` element type.
- ForgeIR values retain their stable IDs as SSA names. Lowering-only temporaries use a deterministic
  `%forgeirN` sequence.
- Operations are visited in verified ForgeIR order. Each source operation has a preceding
  `forgeir.op_id` comment, including boundary operations that emit no body operation.

The emitted text uses the `func`, `stablehlo`, and, for exact GELU, `chlo` dialects. Generic
operation syntax is used so attributes and complete function signatures remain explicit.

## Implemented subset

| ForgeIR operation | Textual lowering |
|---|---|
| `Input`, `Parameter` | Typed `func.func` arguments |
| `Constant` | Scalar `stablehlo.constant` |
| `MatMul` | Broadcast normalization plus `stablehlo.dot_general` |
| `Linear` | Weight transpose plus `stablehlo.dot_general`; safe fused bias/GELU forms are retained |
| `Add`, `Mul`, `Div` | Elementwise StableHLO operations with explicit `broadcast_in_dim` as needed |
| `Reshape` | `stablehlo.reshape` |
| `Transpose` | `stablehlo.transpose` with an explicit permutation |
| `GELU` | Exact decomposition using divide, `chlo.erf`, add, and multiply |
| `RMSNorm` | Square, final-axis reduction, mean, epsilon, `rsqrt`, broadcast, and scale |
| `Softmax` | Max subtraction, exponential, sum reduction, broadcast, and divide |

Softmax accepts only an already verified static ranked float32 tensor and a valid normalized axis.
RMSNorm accepts only verified final-axis float32 normalization with a rank-one weight and positive
epsilon. GELU accepts only the documented exact (`approximate="none"`) convention. These
restrictions prevent a textual lowering from silently changing ForgeIR semantics.

## Not implemented and limitations

- `CausalMask` has no Milestone 11 lowering. It returns diagnostic code
  `mlir.unsupported_operation` with the source operation and output value IDs. Consequently the
  current full TinyTransformer graph stops at that operation; the bridge does not omit or mock it.
- Dynamic or unranked tensors, quantized types, GPU lowering, bufferization, code generation,
  execution-engine integration, and runtime invocation are outside this milestone.
- The text is a controlled interoperability artifact. It is not evidence of a complete StableHLO
  import pipeline or production compiler.
- Exact GELU uses CHLO `erf`; a consumer must register CHLO or perform an equivalent supported
  legalization.
- The bridge does not load parameter data, execute operations, or make numerical or performance
  claims.

## CLI and external validation

With the feature enabled:

```text
forgeir_cli emit-mlir <graph> --output <file>
```

The command writes deterministic module text, then searches `PATH` for `stablehlo-opt` followed by
`mlir-opt`. If a tool is found, the bridge invokes it once for parse/syntax verification and then
attempts `--canonicalize --cse`, preserving a diagnostic log and any canonical output. A syntax
failure returns a non-zero exit code.

If neither tool exists, emission still succeeds because external validation is optional. The JSON
status is `tool unavailable`, with `available=false`, `syntax_verified=false`, and
`canonicalization_succeeded=false`. Golden-output and structural lowering tests still run, but the
result is never described as externally validated.

The deterministic Linear golden fixture is
`tests/golden/milestone_11_linear.mlir`. Unit tests additionally cover explicit types and shapes,
source IDs, every implemented operation family, stable decompositions, repeated-output identity,
and structured unsupported-operation failure.
