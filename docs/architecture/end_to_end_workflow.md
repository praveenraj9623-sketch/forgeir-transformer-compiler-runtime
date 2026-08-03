# Stable end-to-end workflow

## Scope

Milestone 9 connects the completed deterministic reference generator, FX exporter, C++ graph loader
and verifier, optimizer, static memory planner, float32 CPU runtime, numerical comparator, and report
writer. It adds orchestration and durable evidence only. It does not add graph operations, compiler
passes, device backends, model architectures, benchmarks, or performance claims.

## Entry points

The installed console entry point is:

    forgeir pipeline --config <config> --output-dir <directory>

From an uninstalled source checkout, the equivalent command is:

    python -m forgeir.cli pipeline --config <config> --output-dir <directory>

The typed Python API is:

    from pathlib import Path
    from forgeir.pipeline import PipelineResult, run_pipeline

    result: PipelineResult = run_pipeline(
        Path("config/pipeline/default.json"),
        Path("artifacts/pipeline"),
    )

`PipelineConfig`, `PipelineStage`, `StageStatus`, `ArtifactDigest`, `StageResult`,
`PipelineResult`, `PipelineRunner`, and `run_pipeline` are public immutable or typed workflow
contracts.

## Configuration

Pipeline configuration schema `1.0` is a strict JSON object. `config/pipeline/default.json` is the
normative default example. It contains:

- the complete validated `TinyTransformerConfig` under `model`;
- `optimization_level`, one of `O0`, `O1`, or `O2`;
- a positive power-of-two `alignment_bytes`;
- finite nonnegative absolute and relative numerical tolerances.

The model seed must be exactly 42. Any other seed fails before an output directory is created. The
canonical configuration JSON is hashed with SHA-256. The run ID is
`run-<first-16-config-hash-hex>`, so an output root contains an isolated, configuration-identified
run directory.

## Stage contract

The fixed stage order is:

| # | Stage | Primary behavior |
|---:|---|---|
| 1 | `generate-reference` | Generate and verify deterministic configuration, input, weights, expected output, tensor metadata, and reference manifest. |
| 2 | `export` | Trace and export canonical graph JSON and DOT after revalidating every reference artifact. |
| 3 | `inspect` | Load through the C++ graph loader and write its structural summary. |
| 4 | `verify` | Run the complete C++ semantic pass pipeline and persist its diagnostics. |
| 5 | `optimize` | Run the selected native O0/O1/O2 optimizer and write graph, pass report, and before/after DOT. |
| 6 | `plan-memory` | Generate the deterministic schedule, memory plan, CSV, and SVG. |
| 7 | `run` | Execute the optimized graph through the CPU runtime and write deterministic input, output, and structural trace artifacts. |
| 8 | `compare` | Apply the configured combined absolute-plus-relative parity rule to the PyTorch and C++ outputs. |
| 9 | `report` | Write deterministic JSON and Markdown summaries from actual stage data. |

The execution trace stored by the pipeline retains operation ID, type, kernel, output shape, and
arena offset. Per-call elapsed microseconds remain available through the Milestone 8 Python runtime
API but are intentionally excluded from the reproducibility artifact because they are nondeterministic
observations rather than correctness evidence.

## Integrity and determinism

Reference and runtime NPZ archives use stable member order, NPY content, fixed ZIP timestamps, fixed
permissions, and no compression. JSON uses sorted keys and a final newline. Native optimizer reports
are normalized to run-relative artifact paths. Run manifests contain only run-relative paths and no
timestamps or elapsed durations.

After every successful stage, each produced file is recorded with its SHA-256 and byte size. Before
the next stage, the runner rechecks every recorded input and output. A missing, resized, or modified
artifact fails with `stale_artifact`; it is never reused. The graph exporter independently validates
the reference manifest and parameter content hashes, and the CPU runtime independently validates
every external parameter content hash.

`run_manifest.json` contains the configuration hash, seed, stage order, typed stage results, input
hashes observed by each stage, and the final artifact ledger. `run_manifest.sha256` hashes the
manifest. `status.json` records success or the first failed stage, preserved diagnostics, and skipped
downstream stages. The status file itself is part of the artifact ledger.

Two equivalent configurations written with identical input bytes produce byte-identical manifests
even when their isolated run directories have different parent paths.

## Fail-fast and replacement behavior

Stages execute sequentially and stop at the first failure. The failed stage retains structured
diagnostics and any partial files with recorded hashes. Every later stage is marked `skipped`. The CLI
returns 0 only for success or a successful dry run, 1 for a workflow failure, and 2 for setup or
argument failure.

An existing config-derived run directory is never silently reused. Without `--force`, its manifest,
manifest digest, and every recorded artifact are validated. A valid existing run reports
`run_directory_exists`; an invalid one reports `stale_artifact`. Both return nonzero. Only an explicit
`--force` permits deletion and fresh recreation of that exact run directory.

## Presentation modes

Default output uses Rich to show a human-readable stage table and final artifact paths.

- `--json` emits the complete typed result as machine-readable JSON.
- `--quiet` suppresses normal and diagnostic console output while retaining exit status and durable
  status artifacts.
- `--verbose` adds artifact SHA-256 values, byte sizes, and full diagnostics to Rich output.
- `--dry-run` returns the nine planned stages without creating the output root, run directory, or any
  execution artifact.

`--json` and `--quiet` are mutually exclusive. Failure diagnostics are never discarded from the
typed result or durable status, regardless of presentation mode.
