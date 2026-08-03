# Phase 09: End-to-End Workflow

## Scope

Milestone 9 connects the completed reference generator, FX exporter, C++ graph
inspection and verification, optimizer, memory planner, CPU runtime, numerical
comparison, and report generator behind one deterministic CLI and typed Python
API. No Milestone 10 work is included.

## Delivered contracts

- `forgeir pipeline --config <config> --output-dir <directory>` is declared as
  the installed console command. The source-tree equivalent used for local
  validation is `python -m forgeir.cli pipeline ...`.
- The fixed stage order is `generate-reference`, `export`, `inspect`, `verify`,
  `optimize`, `plan-memory`, `run`, `compare`, and `report`.
- `PipelineConfig`, `PipelineStage`, `StageResult`, `ArtifactDigest`, and
  `PipelineResult` are immutable typed Python result contracts.
- Seed 42 is mandatory. A configuration that requests another seed is rejected
  before a run directory is created.
- Each configuration receives a deterministic isolated run ID. Important
  inputs and stage outputs are recorded with their SHA-256 digest and size.
- Every stage validates the complete artifact ledger before executing. A hash
  or size mismatch stops the pipeline at the first affected stage and preserves
  structured diagnostics in `status.json`.
- Existing run directories are never reused. Without `--force`, a valid run is
  reported as already existing and a modified run is reported as stale. Only an
  explicit `--force` replaces the exact configuration-derived run directory.
- `--dry-run` returns the planned stage sequence without creating the output
  directory. Human-readable Rich, `--json`, `--quiet`, and `--verbose` output
  modes are supported.
- Reference NPZ archives use fixed ZIP metadata, stable entry order, and
  uncompressed deterministic NPY members so repeated manifests are identical.
- Runtime timing remains available through the runtime API, but the pipeline's
  deterministic execution-trace artifact records only structural trace fields.

## Files created

- `config/pipeline/default.json`
- `python/forgeir/cli/__init__.py`
- `python/forgeir/cli/__main__.py`
- `python/forgeir/cli/main.py`
- `python/forgeir/pipeline/__init__.py`
- `python/forgeir/pipeline/runner.py`
- `python/forgeir/pipeline/types.py`
- `tests/python/test_end_to_end_pipeline.py`
- `docs/architecture/end_to_end_workflow.md`
- `docs/phase_reports/phase_09.md`

The package root and project metadata were extended to export the typed API,
declare Rich, and install the `forgeir` console entry point. The reference
artifact writer was made byte-deterministic.

## Verification environment

- Windows 10 AMD64
- Python 3.11.9 from `.venv`
- CMake 3.29.5-msvc4
- Ninja 1.13.2
- MSVC 19.42.34433.0
- Debug and Release pybind11 modules supplied through `PYTHONPATH`

No package was installed and no dependency was downloaded during this
milestone.

## Commands and outcomes

The active environment for Python and pipeline commands was:

```powershell
$env:VIRTUAL_ENV=(Resolve-Path '.venv').Path
$env:Path="$env:VIRTUAL_ENV\Scripts;$env:Path"
$env:PYTHONPATH="$(Resolve-Path 'build\windows-msvc-release\python');$(Resolve-Path 'python')"
```

Static Python validation:

```powershell
python -m ruff format --check python tests
python -m ruff check python tests
python -m mypy python\forgeir
```

Outcome: 30 files were already formatted; Ruff reported `All checks passed!`;
mypy reported success across 18 source files.

Complete Python suite:

```powershell
python -m pytest tests\python -q
```

Outcome: 64 tests passed in 11.33 seconds. This includes successful complete
pipeline, numerical parity, tampered weight, invalid graph, stale artifact,
forced replacement, repeated determinism, non-42 seed rejection, all CLI output
modes, and artifact-free dry-run coverage.

Debug C++ build and tests:

```powershell
cmake --build --preset windows-msvc-debug
ctest --preset windows-msvc-debug --output-on-failure
```

Outcome: Ninja reported no work to do and all 75 tests passed in 2.02 seconds.

Release C++ build and tests:

```powershell
cmake --build --preset windows-msvc-release
ctest --preset windows-msvc-release --output-on-failure
```

Outcome: Ninja reported no work to do and all 75 tests passed in 1.73 seconds.

Dry-run validation:

```powershell
python -m forgeir.cli pipeline --config config\pipeline\default.json --output-dir work\milestone09_dry --dry-run --json
```

Outcome: all nine stages were reported as planned and
`work\milestone09_dry` was not created.

Clean repeated pipeline validation began with both output roots absent:

```powershell
python -m forgeir.cli pipeline --config config\pipeline\default.json --output-dir artifacts\pipeline\milestone_09\run_1 --quiet
python -m forgeir.cli pipeline --config config\pipeline\default.json --output-dir artifacts\pipeline\milestone_09\run_2 --quiet
```

Both commands returned exit code 0. Each run recorded nine succeeded stages and
no failure stage. The generated manifests were compared byte for byte and both
had SHA-256:

```text
b750c4ee74c4fafc5c73141eac349b9c8ec569c3258506a2c580c66db44563c7
```

## Verified result

- Run ID: `run-a0fa72a0f8f1ee46`
- Configuration SHA-256:
  `a0fa72a0f8f1ee4687dddd5d6dd2a0aeb5e6f8e2a96580e8b8fe51f00cae807c`
- Optimized graph SHA-256:
  `681b7009cef039c59515650e69d1b520f96f2c8faeadeca1df114f7800f7494d`
- Operation count: 35 before optimization and 35 after O2 optimization
- Memory-plan calculation: 950272 naive bytes and 262144 planned bytes
- Numerical comparison: passed, zero tolerance violations,
  maximum absolute error `5.662441253662109e-07`, and maximum relative error
  `0.0023864314425736666`
- Primary run directory:
  `artifacts/pipeline/milestone_09/run_1/run-a0fa72a0f8f1ee46`
- Determinism run directory:
  `artifacts/pipeline/milestone_09/run_2/run-a0fa72a0f8f1ee46`

The memory values are static plan calculations. The numerical values are test
outputs, not performance claims.

## Environment limitation

The Windows MSVC/Ninja paths were validated locally. A Linux GCC/Ninja host was
not available in this Windows environment, so Linux execution was not rerun in
Milestone 9. The console-script entry point is declared in `pyproject.toml`; the
equivalent module command was used because installing the local package was
explicitly outside the milestone workflow.
