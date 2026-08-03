# C++ typed intermediate representation

## Scope

Milestone 4 introduces a read-only C++ representation of ForgeIR graph schema version `1.0`, a JSON
loader, and structural verification. It does not execute, schedule, rewrite, or optimise operations.
The normative wire contract remains `config/graph/forgeir_graph.schema.json`.

## Typed model

`DataType` supports exactly `bool`, `float32`, and `int64`. `Shape` stores dimensions as `int64_t`.
`TensorDescriptor` pairs a dtype with a shape and exposes checked element-count and byte-size
calculations. `ValueKind`, `Value`, `OperationType`, `Operation`, and `Graph` represent the ordered
contract records without owning raw pointers. Operation attributes remain read-only JSON because
schema version 1.0 assigns different attributes to each canonical operation.

The operation enum contains exactly the schema operations: `Input`, `Parameter`, `Constant`,
`MatMul`, `Linear`, `Add`, `Mul`, `Div`, `RMSNorm`, `GELU`, `Softmax`, `Reshape`, `Transpose`, and
`CausalMask`.

`Status` carries an explicit status code and message. `Result<T>` contains either a value or a
non-success status. `Diagnostic` is the durable structured diagnostic record with severity, code,
message, and optional entity ID. Loading failures never substitute a partial graph.

## Size safety

Shapes reject negative dimensions and, because schema version 1.0 has no explicit zero-dimension
opt-in, reject zero dimensions. A scalar has an empty shape and one element. Each element-count
multiplication is checked against `uint64_t` before it occurs. Tensor byte sizes check the product of
element count and dtype width, and graph summaries check addition overflow while accumulating
parameter storage.

These calculations estimate descriptor storage only. The loader allocates no tensor data.

## Structural verification

`GraphLoader` parses with nlohmann/json and performs these checks before returning a graph:

1. The document and required fields have the expected JSON types and schema version is `1.0`.
2. Contract hashes have lowercase SHA-256 syntax and `graph_hash` matches canonical JSON with only
   the hash field omitted.
3. Value and operation IDs are well formed and unique; dtypes, kinds, shapes, and operation names are
   supported.
4. Every operand and output value exists, every value has exactly one producer, and declaration
   operations agree with their output value kinds.
5. Operation arities match the controlled schema contract.
6. Kahn topological verification rejects cycles. Ready operations are selected by stable operation
   ID, producing a deterministic topological order without creating an execution schedule.
7. Every required graph output is transitively reachable from a required graph input.

Parameters remain external to JSON. The loader validates manifest and parameter digest syntax but
does not load NPZ data or execute an operation.

## Inspection surfaces

The command below loads and verifies a graph before printing a JSON summary:

    forgeir_cli inspect-graph <path>

The summary contains schema version, input/output/value/operation counts, a lexicographically ordered
operation histogram, and overflow-checked estimated parameter bytes. `forgeir_py.graph_summary(path)`
exposes the same information as a read-only Python dictionary. The cross-language test exports an FX
graph in Python, loads it through this C++ binding, and compares every summary field.
