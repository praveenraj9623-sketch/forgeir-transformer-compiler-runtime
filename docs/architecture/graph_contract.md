# ForgeIR graph contract

## Scope

Milestone 3 traces the deterministic Milestone 2 transformer block with PyTorch FX and lowers it into
a versioned ForgeIR JSON graph. The exporter is a controlled translation layer: it reads the FX node
sequence but never serializes arbitrary FX objects, Python callables, or module internals. There is no
C++ parser or execution runtime in this milestone.

The graph schema version is `1.0`. The normative JSON Schema is
`config/graph/forgeir_graph.schema.json` and uses JSON Schema draft 2020-12.

## Top-level contract

Every graph contains:

- `graph_schema_version`: exactly `1.0`.
- `producer_version`: the ForgeIR producer version.
- `model_configuration_hash`: SHA-256 of the canonical Milestone 2 configuration object.
- `graph_hash`: deterministic SHA-256 of the graph with only `graph_hash` omitted.
- `inputs` and `outputs`: ordered stable value IDs.
- `values`: ordered value records.
- `operations`: ordered operation records.
- `weight_manifest`: portable manifest reference plus manifest and weight-archive hashes.

Arrays preserve graph order. Object keys are serialized lexicographically with UTF-8, no insignificant
whitespace, no non-finite numbers, and one final newline in the file. This canonical form is used for
hashing and disk serialization.

## Values

A value record contains a stable ID such as `v0000`, a semantic name, shape, dtype, and one kind:
`input`, `parameter`, `constant`, `intermediate`, or `output`. IDs are allocated monotonically in
lowering order. Shapes use nonnegative integer dimensions. Version 1.0 permits bool, float32, and
int64 value dtypes; the current transformer block graph uses float32.

Parameter values contain no weight bytes. Their `Parameter` operations identify an NPZ archive key
and the tensor content SHA-256.

## Operations

An operation contains a stable ID such as `op0000`, canonical type, semantic name, ordered input and
output value IDs, and operation-specific attributes.

| Canonical operation | Contract role |
|---|---|
| `Input` | Declares a graph input value. |
| `Parameter` | Declares an external weight value. |
| `Constant` | Declares a scalar or tensor literal. |
| `MatMul` | Matrix multiplication for attention scores or context; optimized form may add a verified rank-one bias and exact fused activation attributes. |
| `Linear` | Learned projection. The exporter emits a bias-free two-input form; optimized form may add a verified rank-one bias and exact fused activation attributes. |
| `Add` | Residual elementwise addition. |
| `Mul` | Elementwise or scalar multiplication. |
| `Div` | Elementwise or scalar division available in schema version 1.0. |
| `RMSNorm` | RMS normalization with external learned scale and epsilon. |
| `GELU` | Exact or explicitly attributed GELU. |
| `Softmax` | Stable Softmax over an attributed axis. |
| `Reshape` | Shape-only layout reinterpretation. |
| `Transpose` | Dimension permutation. |
| `CausalMask` | Replaces future attention scores with negative infinity. |

No FX node is ignored. Placeholders become `Input`; supported modules and functions lower explicitly;
the FX output identifies the ordered ForgeIR output. Every other node raises a structured error.

The Milestone 6 optimized projection form is backward compatible within schema version `1.0`:
operation attributes are extensible and operation inputs were already represented as an ordered
array. Unoptimized two-input `Linear` and `MatMul` remain valid. A fused projection adds a third,
rank-one parameter or constant input, `bias=true`, `fused_activation="GELU"`, and
`fused_activation_approximate="none"`. Load-time and semantic verification reject an invalid bias
shape or operand count. Migration coverage loads and verifies both the original and fused forms.

## Controlled attention expansion

RMSNorm and causal attention are treated as explicit FX leaf boundaries so tracing does not execute
shape validation against symbolic proxies. The exporter then expands the known
`CausalSelfAttention` module into separate Q/K/V `Linear` operations, head `Reshape` and `Transpose`
operations, score `MatMul`, scale `Constant` and `Mul`, `CausalMask`, stable `Softmax`, context
`MatMul`, head merge, and output `Linear`. This expansion reads only the reviewed Milestone 2 module
type and configuration.

## External weights

The CLI receives a Milestone 2 artifact directory and a portable manifest reference. Before tracing
or writing a graph, the exporter:

1. Verifies `manifest.sha256` against every required Milestone 2 payload file.
2. Requires the artifact configuration to equal the export configuration.
3. Resolves each graph parameter through `tensor_metadata.json`.
4. Requires the parameter to name `weight_tensors.npz` and an existing archive key.
5. Rechecks the loaded tensor shape, dtype, and raw-content SHA-256.

The graph records the portable manifest reference, manifest SHA-256, archive filename, archive
SHA-256, and per-parameter archive keys/content hashes. Weight bytes remain exclusively in the NPZ.
Missing or modified weights fail export.

## Unsupported-node errors

Unsupported nodes raise `UnsupportedFxNodeError`. Its machine-readable representation contains
`node_name`, `fx_target`, recursively described `arguments`, and `reason`. The exporter never omits
the node or substitutes a mock operation.

## Graph hash

To compute the graph hash, copy the complete graph object, remove only `graph_hash`, serialize the
copy with canonical JSON, encode as UTF-8, and compute SHA-256. Verification repeats these steps and
compares the result with the embedded lowercase hexadecimal digest.

## Artifacts and CLI

Run after generating the default Milestone 2 reference artifacts:

    python -m forgeir.export.fx_export

Optional `--weights-dir`, `--output-dir`, and `--weight-manifest-reference` arguments select the
physical weights, output directory, and portable reference recorded in the graph. One export writes:

- `tiny_transformer_block.graph.json`: canonical machine-readable graph contract.
- `tiny_transformer_block.dot`: deterministic operation-level DOT graph for human inspection.

Small golden operation-order and operation-count files under `tests/golden` protect the version 1.0
lowering order without embedding external weight bytes.
