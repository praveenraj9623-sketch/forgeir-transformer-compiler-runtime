# Phase 03: Versioned deterministic FX graph export

## Scope

Milestone 3 traces the Milestone 2 transformer block with PyTorch FX and lowers it through a
controlled exporter into ForgeIR graph schema version 1.0. It adds canonical JSON serialization,
deterministic graph hashing, external weight resolution, JSON Schema validation, structured
unsupported-node failures, DOT output, tests, and golden operation fixtures.

This milestone does not implement a C++ graph parser, graph execution runtime, compiler passes, FX
runtime execution, GPU code, or MLIR code.

## Contract

The normative schema is `config/graph/forgeir_graph.schema.json`. Every graph records the schema
version, producer version, model configuration hash, graph hash, ordered inputs, ordered outputs,
ordered values, ordered operations, and a Milestone 2 weight-manifest reference. Values contain stable
IDs, semantic names, shapes, dtypes, and kinds. Parameter bytes remain in `weight_tensors.npz`.

The exporter verifies the complete Milestone 2 manifest, configuration equality, every parameter
archive key, shape, dtype, and raw-content hash before writing the graph. No FX node is skipped.
Unsupported nodes raise a structured error containing node name, FX target, arguments, and reason.

## Verified environment

- Host: Windows
- Python: 3.11.9 from the repository virtual environment
- PyTorch: 2.13.0+cpu
- JSON Schema validator: jsonschema 4.26.0
- Device used for reference regeneration: CPU

No package was installed and no dependency, model, or data was downloaded.

## Commands and observed outcomes

The repository virtual environment and source/build paths were selected with:

    $env:VIRTUAL_ENV = (Resolve-Path '.venv').Path
    $env:Path = "$env:VIRTUAL_ENV\Scripts;$env:Path"
    $env:PYTHONPATH = "$(Resolve-Path 'python');$(Resolve-Path 'build\windows-msvc-debug\python')"

Python lint:

    python -m ruff check python tests

Outcome: `All checks passed!`

Strict source typing:

    python -m mypy python\forgeir

Outcome: `Success: no issues found in 11 source files`.

Python tests:

    python -m pytest tests\python -q

Outcome: `30 passed`. Milestone 3 coverage includes schema validity, canonical JSON, deterministic
graph and DOT bytes, deterministic graph hashes, stable golden operation order/counts, structured
unsupported-node errors, missing-weight rejection, tampered-weight rejection, graph-hash tamper
rejection, and the complete graph input/output/value/parameter contract.

Default reference regeneration:

    python -m forgeir.reference.generate --output-dir artifacts\references\milestone_02\default_run_1

Outcome: succeeded and verified the Milestone 2 artifact manifest before returning.

Default controlled FX export:

    python -m forgeir.export.fx_export

Outcome: succeeded. It wrote canonical graph JSON and deterministic DOT output. The embedded graph hash
is `a8a805297dc8483d2407043342d0d15433d42f73e28fd96b53cce3faaa0ec322`.

Independent schema, canonical serialization, and graph-hash validation loaded the emitted graph and
`config/graph/forgeir_graph.schema.json`, then ran:

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(graph)
    verify_graph_hash(graph)
    assert graph_text == canonical_json_text(graph) + "\n"

Outcome:

- JSON Schema valid: true
- Graph hash valid: true
- Canonical JSON valid: true
- Ordered operations: 35
- Ordered values: 35
- Ordered inputs: 1
- Ordered outputs: 1

Existing C++ and Milestone 1 regression tests:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
    .\scripts\windows\test.ps1 -Configuration Debug

Outcome: `100% tests passed, 0 tests failed out of 5`.

## Operation counts

- `Add`: 2
- `CausalMask`: 1
- `Constant`: 1
- `GELU`: 1
- `Input`: 1
- `Linear`: 6
- `MatMul`: 2
- `Mul`: 1
- `Parameter`: 8
- `RMSNorm`: 2
- `Reshape`: 4
- `Softmax`: 1
- `Transpose`: 5

Total: 35 operations.

## Generated artifact paths

- Graph JSON:
  `artifacts/graphs/milestone_03/default/tiny_transformer_block.graph.json`
- DOT graph:
  `artifacts/graphs/milestone_03/default/tiny_transformer_block.dot`
- Referenced weight manifest:
  `artifacts/references/milestone_02/default_run_1/manifest.sha256`
- JSON Schema:
  `config/graph/forgeir_graph.schema.json`
- Golden operation order:
  `tests/golden/tiny_transformer_block_v1.operation_order.json`
- Golden operation counts:
  `tests/golden/tiny_transformer_block_v1.operation_counts.json`

## Environment limitation

The Windows host continues to require repository-local pytest temporary storage and a process-scoped
PowerShell execution-policy bypass, as documented in earlier phases. No persistent system policy was
changed. No C++ parser or execution runtime was built or tested because both are outside Milestone 3.
