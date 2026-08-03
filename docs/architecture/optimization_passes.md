# Deterministic graph optimization passes

## Scope

Milestone 6 adds deterministic graph rewriting on the verified, typed JSON graph contract. It does
not execute the graph in C++, schedule operations, introduce runtime behavior, or make performance
claims. Every pipeline verifies the graph before and after each pass and aborts on the first
diagnostic error.

The optional residual/RMSNorm fusion is intentionally absent. The current schema has no fused
operation whose numerical order and evaluator semantics could express that rewrite without changing
the public operation contract.

## Rewrite contract

Each pass returns explicit status, changed state, ordered diagnostics, and ordered rewrite evidence.
Every rewrite record contains the pass and stage, a nonempty reason, operation IDs added, removed, or
modified, and removed value IDs. Existing value and operation IDs and semantic names are retained
whenever their nodes survive. Declared output IDs are immutable across optimization.

`PassManager` surrounds every transformation with semantic `GraphVerifierPass` runs. The CLI also
serializes and reloads the result through `GraphLoader`, providing structural, graph-hash, shape, and
dtype checks. A second execution of each pass produces no rewrite and reports `changed=false`.

Canonical operation order is a stable topological order with operation ID as its deterministic
tie-breaker. Relative order between side-effect-marked operations is retained as an additional
ordering constraint.

## Passes

### CanonicalisationPass

Negative Softmax and RMSNorm axes are converted to their equivalent nonnegative index. Exact GELU,
stable Softmax, and causal-mask diagonal defaults are materialized when omitted. A redundant
`side_effect=false` marker is removed. Operations are placed in stable topological order. The pass
does not change value IDs, names, shapes, dtypes, or declared outputs.

### ConstantFoldingPass

Only scalar `Add`, `Mul`, `Div`, and exact `GELU` expressions whose operands are literal `Constant`
values are eligible. Integer addition and multiplication are bounds-checked, division by zero is
not folded, and side-effect-marked operations are never folded. The public limit is 1,024 tensor
elements; the initial implementation is deliberately stricter and accepts only one-element scalar
outputs. The terminal operation and output value IDs remain stable while the operation becomes a
`Constant` with its computed value and fold provenance.

### DeadCodeEliminationPass

Liveness starts from all declared outputs, declared inputs, and outputs of side-effect-marked
operations. Producer operands are followed transitively. Only operations and values outside that
live set are removed. Declared outputs and side-effect paths are therefore retained even when their
values otherwise appear unused.

### RedundantReshapeEliminationPass

An identity reshape is removed only when input and output shapes are identical, the reshape has no
side effect, and its output is not a declared graph output. All uses are redirected to the input;
the reshape operation and its intermediate output value are removed.

### RedundantTransposeEliminationPass

An identity permutation is removed under the same output, use-replacement, and side-effect safety
rules as an identity reshape. Nonidentity permutations remain unchanged.

### FuseBiasGELUPass

The pass recognizes either of these exact single-use chains:

    Linear(input, weight) -> Add(projection, bias) -> GELU
    MatMul(left, right) -> Add(projection, bias) -> GELU

The bias must be a rank-one parameter or constant matching the final output dimension. The
projection and Add must have exactly one use, neither intermediate may be a declared output, and all
three operations must be side-effect free. GELU must use the exact `none` approximation. For Linear,
the source projection must explicitly be bias-free.

The terminal GELU operation ID, output value ID, and semantic names are preserved. Its operation
type becomes the original projection type, its inputs become the two projection inputs plus bias,
and attributes record `bias=true`, exact fused GELU semantics, and the three source operation IDs.
The projection and Add operations and their intermediate values are removed. The Python reference
evaluator defines and tests the fused contract for both Linear and MatMul.

## Optimization levels

| Level | Ordered pipeline |
|---|---|
| `O0` | Verification only |
| `O1` | Canonicalisation, constant folding, dead-code elimination |
| `O2` | Canonicalisation, constant folding, redundant reshape elimination, redundant transpose elimination, bias-GELU fusion, dead-code elimination |

No pass is registered solely to increase the pipeline length.

## CLI artifacts

Run optimization with:

    forgeir_cli optimize <input> --level O0|O1|O2 --output <path>

The command writes the canonical optimized graph, `<output-stem>.pass_report.json`,
`<output-stem>.before.dot`, and `<output-stem>.after.dot`. The optimized graph contains its refreshed
canonical SHA-256 hash. The pass report contains input/output hashes, before/after operation and
value counts, the fold-size limit, every pre/pass/post verification execution, diagnostic counts,
and detailed rewrite evidence.

## Semantic validation

`forgeir.reference.evaluate_graph` is a deterministic CPU-only evaluator for the controlled graph
operations. It verifies the graph hash, loads external parameters from the supplied NPZ archive,
checks every produced tensor against declared shape and dtype, and fails explicitly on unsupported
contracts. Tests compare the original and optimized transformer outputs at O0, O1, and O2, and also
exercise numerically active Linear/MatMul bias-GELU fusion graphs.
