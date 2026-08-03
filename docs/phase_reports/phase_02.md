# Phase 02: Deterministic PyTorch reference transformer

## Scope

Milestone 2 implements an immutable validated TinyTransformer configuration, a deterministic CPU
PyTorch reference model, reference artifact serialization and verification, and behavioral tests.
It does not implement FX export, C++ IR, compiler passes, runtime execution, graph export, GPU code,
MLIR code, pretrained model loading, external data loading, or benchmarks.

## Implementation

The reference model uses explicit token embedding, two RMSNorm modules, separate query/key/value
projections, scaled causal multi-head attention, a manually stabilized Softmax, an output projection,
two residual additions, and an exact GELU MLP. Dropout is absent and factory-created models are in
evaluation mode. The implementation does not call
`torch.nn.functional.scaled_dot_product_attention`.

The configuration validates positive and bounded dimensions, head divisibility, float32 support,
finite positive epsilon, seed range, and total parameter and activation element budgets before
constructing the model.

## Verified environment

- Host: Windows
- Python: 3.11.9 from the repository virtual environment
- PyTorch: 2.13.0+cpu
- NumPy: 2.4.6
- mypy: 1.20.2
- Device used: CPU
- Deterministic seed: 42

No package was installed, no dependency was downloaded, no pretrained model was loaded, and no
internet data was used.

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

Outcome: `Success: no issues found in 6 source files`.

Python tests:

    python -m pytest tests\python -q

Outcome: `23 passed`. Coverage includes configuration validation and immutability, allocation bounds,
the RMSNorm equation, attention shape and normalization, strict causal masking, deterministic
parameters/inputs/repeated outputs, finite output, changed output for changed input, evaluation mode,
dropout absence, tensor metadata hashes, repeatable manifests, manifest tamper rejection, and the
module CLI.

Default reference generation run 1:

    python -m forgeir.reference.generate --output-dir artifacts\references\milestone_02\default_run_1

Default reference generation run 2:

    python -m forgeir.reference.generate --output-dir artifacts\references\milestone_02\default_run_2

Manifest comparison:

    $first = Get-Content artifacts\references\milestone_02\default_run_1\manifest.sha256 -Raw
    $second = Get-Content artifacts\references\milestone_02\default_run_2\manifest.sha256 -Raw
    if ($first -cne $second) { throw 'Generated manifests differ.' }

Outcome: the manifests were byte-identical. Their SHA-256 was
`e5d85fd1682ff4be448404e6c9544e34b18356e27359f6f045752da534058be6`.
Each generation also ran the manifest verifier before returning.

The identical manifest records:

- `configuration.json`: `e3f94ffe8a79a93010bd5132aed61c8ed8e0e789e9db14beb8bbf7ec0f109412`
- `expected_output.npz`: `ab9ab72823d93c80e60eeb0c34dd40a83e7e0eab3addbd5af55b695c9e5bbf73`
- `input_tensor.npz`: `7a666dee063a26399ac9cea72fa94f7596c904cf64afff5f3fd296bf23ef03ee`
- `tensor_metadata.json`: `c0c63ff6e1f72c785165804fea4cdaa61f429d7e4c5fdf4276b10db43fecef23`
- `weight_tensors.npz`: `246eb4f777093c8355392ce5cf202cdee92fd138f7287aff811f36cb2283e5d6`

Existing C++ and Milestone 1 regression tests:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
    .\scripts\windows\test.ps1 -Configuration Debug

Outcome: `100% tests passed, 0 tests failed out of 5`.

## Artifact locations

- `artifacts/references/milestone_02/default_run_1`
- `artifacts/references/milestone_02/default_run_2`

Each directory contains `configuration.json`, `input_tensor.npz`, `weight_tensors.npz`,
`expected_output.npz`, `tensor_metadata.json`, and `manifest.sha256`.

## Environment limitation

The host denied pytest access to its default `AppData\Local\Temp\pytest-of-admin` directory. The
repository config now directs pytest temporary files to the ignored `.pytest_tmp` directory, allowing
the required unmodified pytest command to pass. As in Milestone 1, PowerShell scripts were invoked
with a process-scoped execution-policy bypass; no persistent policy was changed.

No GPU or matching GPU hardware was used or claimed. FX export and graph export remain intentionally
unimplemented.
