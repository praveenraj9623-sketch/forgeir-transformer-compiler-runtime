# CPU benchmark protocol

## Scope and non-goals

Milestone 10 measures the existing float32 CPU implementations without changing operator semantics,
numeric conventions, scheduling, or memory planning. It compares the deterministic transformer
block boundary in PyTorch eager CPU with ForgeIR CPU graphs optimized at O0, O1, and O2. The token
embedding is prepared before timing for every implementation; the measured block input is therefore
identical `[batch, sequence, hidden]` float32 content.

Measurements apply only to the captured machine and software build. They are not portable speed
claims. A report must state explicitly whenever a measured ForgeIR mean latency is higher than the
PyTorch eager mean latency.

## Protocol configurations

The test-sized protocol is `config/benchmark/smoke.json`:

- one warm-up;
- three measured iterations;
- one separately profiled PyTorch iteration;
- shape `[1, 4, 8]`, intermediate size 16, and two attention heads.

The documented local protocol is `config/benchmark/full_local_cpu.json`:

- five warm-ups;
- twenty measured iterations;
- twenty separately profiled PyTorch iterations;
- shape `[2, 32, 128]`, intermediate size 384, and four attention heads.

Both use seed 42, float32, one process, one PyTorch intra-operation thread, and the single-threaded
ForgeIR CPU runtime. PyTorch inter-operation thread count and `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
and `OPENBLAS_NUM_THREADS` are captured rather than silently changed. CPU affinity and process
priority are left to the operating-system scheduler and reported as unchanged.

Full configurations are never selected by unit tests. Unit tests invoke only the smoke configuration
and assert correctness, schema, finite measurements, sample counts, and report provenance. They do
not assert that one implementation is faster than another.

## Inputs and weights

Reference artifacts are generated once per invocation from `TinyTransformerConfig` and seed 42.
The exporter produces one controlled graph. That graph is independently optimized to O0, O1, and
O2. All ForgeIR sessions receive the exact same contiguous block-input NumPy array and parameter
archive. The benchmark also compares every regenerated PyTorch state tensor against the NPZ tensor
before measuring.

The measured JSON records SHA-256 identities for the runtime input content, input archive, weight
archive, reference manifest, graph files, and canonical graph hashes. Correctness is checked after
measurement with:

    abs(actual - expected) <= 2e-6 + 2e-5 * abs(expected)

NaN and positive/negative infinity classifications must also match. These correctness tolerances and
a deliberately generous 60-second single-iteration gross-failure ceiling are the only regression
thresholds. There is no machine-specific speed threshold.

## Timed regions and clocks

ForgeIR warm-ups and measured iterations call an already-loaded `RuntimeSession`. The C++ timed
region starts immediately before `RuntimeSession::execute` and ends immediately after it returns.
It uses `std::chrono::steady_clock`. The existing per-operation trace uses the same clock around each
stable operation ID. Input and parameter contract checks, output retention, arena execution, and
trace collection performed by `execute` are included. Graph parsing and session construction are
not included.

PyTorch warm-ups and measured iterations call the already-created evaluation-mode
`TinyTransformerBlock` under `torch.inference_mode()`. End-to-end timing uses Python's monotonic
`time.perf_counter_ns`. CPU eager work is synchronous. PyTorch operation-level data is collected in
a separate profiler run as self CPU time grouped by PyTorch operator family. Profiler measurements
are not substituted for the end-to-end samples and must not be summed or directly equated to stable
ForgeIR operation IDs.

The following work is excluded from end-to-end latency and is either recorded as separate setup
timing or omitted from timing:

- reference generation and artifact hashing;
- token embedding and input/weight loading;
- FX tracing and graph export;
- O0/O1/O2 optimization and artifact writing;
- graph parsing, verification, scheduling, planning, and session loading;
- PyTorch profiler passes;
- output extraction and correctness comparison;
- environment capture;
- JSON, CSV, HTML, and manifest generation.

## Statistics

Raw microsecond samples are retained in `benchmark_measurements.json`. Statistics describe the full
population of retained measured iterations:

- minimum and maximum are the smallest and largest sample;
- mean is the arithmetic mean;
- standard deviation is the population standard deviation (`ddof = 0`);
- p50 and p95 use linear interpolation at ranks `(n - 1) * 0.50` and `(n - 1) * 0.95` in the sorted
  sample sequence;
- attempted samples per second is
  `measured_iterations * batch_size / summed_measured_seconds`.

“Attempted samples” means transformer input sequences presented to completed measured calls. It is
reported for protocol transparency and is not a service-capacity or concurrent-throughput claim.
Planned arena bytes come from the static memory plan associated with each already-loaded ForgeIR
session; they are not a measured resident-memory value.

## Environment capture

Each measured JSON report records:

- CPU model and physical/logical core counts;
- detailed operating system and machine identity;
- ForgeIR compiler, build type, and version;
- PyTorch and Python versions;
- Git commit, dirty-worktree flag, and a SHA-256 identity of porcelain status;
- active process and thread settings and relevant thread-pool environment variables;
- all configured input dimensions and iteration counts;
- clock sources and the list of excluded setup/report work.

A dirty worktree is not hidden. It limits exact source reconstruction from the recorded commit and
must be mentioned when interpreting local results.

## Reports and provenance

The command is:

    forgeir benchmark --config <config> --output-dir <new-directory>

The source-tree equivalent is:

    python -m forgeir.cli benchmark --config <config> --output-dir <new-directory>

An existing directory fails unless `--force` is explicit. The runner first writes
`benchmark_measurements.json`. It then reloads only that JSON to produce
`benchmark_summary.csv` and the self-contained static `benchmark_report.html`. HTML generation does
not inspect live runtime objects, rerun measurements, or obtain independent values. A SHA-256
manifest covers the configuration, input and weight identities, graphs, measured JSON, CSV, and
HTML.
