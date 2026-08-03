# Phase 10: CPU Benchmarking and Operation Profiling

## Scope

Milestone 10 adds a setup-separated, raw-sample CPU benchmark for PyTorch eager and ForgeIR O0,
O1, and O2. It does not change graph operations, operator numeric conventions, scheduling, memory
planning, or backend selection. No later milestone work is included.

## Delivered contracts

- `benchmark_runtime` performs warm-ups and measured `RuntimeSession::execute` calls in C++.
- Whole-session and existing per-operation timing use `std::chrono::steady_clock`.
- The binding `forgeir_py.benchmark_execute` returns raw end-to-end and stable operation-ID samples,
  clock identity, and memory-plan values.
- `forgeir benchmark --config <config> --output-dir <new-directory>` and the equivalent
  `python -m forgeir.cli benchmark ...` run the comparison.
- PyTorch eager and ForgeIR O0/O1/O2 use one generated input and one verified weight archive.
- JSON retains raw samples and p50, p95, minimum, maximum, mean, population standard deviation,
  attempted samples per second, and planned arena bytes.
- PyTorch operator-family self CPU time is gathered in a separate profiler phase; it is not
  substituted for unprofiled PyTorch end-to-end latency.
- CSV and static HTML are rendered only by reloading `benchmark_measurements.json`.
- Correctness tolerance and a 60-second gross-failure ceiling are the only thresholds. No
  machine-specific speed threshold exists.
- Host reports capture CPU model, physical/logical cores, OS, compiler, build type, PyTorch,
  Python, Git commit/dirty status, and process/thread settings.

## Files created

- `cpp/include/forgeir/runtime/benchmark.hpp`
- `cpp/src/runtime/benchmark.cpp`
- `tests/cpp/benchmark_test.cpp`
- `config/benchmark/smoke.json`
- `config/benchmark/full_local_cpu.json`
- `python/forgeir/benchmark/__init__.py`
- `python/forgeir/benchmark/__main__.py`
- `python/forgeir/benchmark/environment.py`
- `python/forgeir/benchmark/report.py`
- `python/forgeir/benchmark/runner.py`
- `python/forgeir/benchmark/statistics.py`
- `python/forgeir/benchmark/types.py`
- `tests/python/test_cpu_benchmark.py`
- `docs/architecture/benchmark_protocol.md`
- `docs/phase_reports/phase_10.md`

The build source list, Python binding, package exports, and CLI were extended. The shared MSVC
warning profile now adds `/EHsc`; a clean regeneration had exposed MSVC C4530 because `/WX` was
enabled without exception-unwinding semantics. This is a build-contract correction, not an
operator implementation or benchmark tuning change.

## Full benchmark environment

- CPU: AMD Ryzen 5 7600X 6-Core Processor
- Cores: 6 physical, 12 logical
- OS: Windows 10, build identity `Windows-10-10.0.26200-SP0`
- Compiler: MSVC 19.42.34433.0
- Build: Release
- PyTorch: 2.13.0+cpu
- Python: 3.11.9
- Git commit: `18c8b09e81119f75fb7a9c68279512b57ba404c8`
- Git worktree dirty: true
- Process count: 1
- PyTorch intra-operation threads: 1
- PyTorch inter-operation threads: 6 (captured, unchanged)
- ForgeIR runtime threads: 1
- Affinity and process priority: unchanged

The dirty-worktree flag is material: the commit alone does not reconstruct the exact benchmark
source. The report includes a SHA-256 identity of the captured porcelain status.

## Commands and outcomes

The Python environment used for validation was:

```powershell
$env:VIRTUAL_ENV=(Resolve-Path '.venv').Path
$env:Path="$env:VIRTUAL_ENV\Scripts;$env:Path"
$env:PYTHONPATH="$(Resolve-Path 'build\windows-msvc-release\python');$(Resolve-Path 'python')"
```

Windows Debug and Release configuration/build used the repository scripts because the parent shell
did not inherit the Visual Studio developer environment and local script execution policy was
restricted:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\configure.ps1 -Configuration Debug
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\build.ps1 -Configuration Debug
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\configure.ps1 -Configuration Release
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\build.ps1 -Configuration Release
```

Outcome: both configurations and builds succeeded with strict warnings treated as errors.

Targeted benchmark tests:

```powershell
ctest --test-dir build\windows-msvc-debug -R RuntimeBenchmark --output-on-failure
python -m pytest tests\python\test_cpu_benchmark.py -q
```

Outcome: 2 C++ benchmark contract tests and 5 Python smoke/report tests passed. The smoke test uses
the smoke configuration only; the full protocol is not invoked by normal unit tests.

Standalone smoke benchmark:

```powershell
python -m forgeir.cli benchmark --config config\benchmark\smoke.json --output-dir benchmarks\results\milestone_10\smoke
```

Outcome: success with one warm-up and three measured iterations. On the tiny `[1, 4, 8]` shape,
measured means were 198.500 microseconds for PyTorch, 48.533 for ForgeIR O0, 48.100 for O1, and
57.900 for O2. This smoke result is dominated by tiny-workload dispatch characteristics and is not
used as a general performance conclusion.

Documented full local CPU benchmark:

```powershell
python -m forgeir.cli benchmark --config config\benchmark\full_local_cpu.json --output-dir benchmarks\results\milestone_10\full_local_cpu
```

Outcome: success with five warm-ups, twenty measured end-to-end iterations, twenty separately
profiled PyTorch iterations, and zero correctness violations for every ForgeIR level.

Python static checks and complete test suite:

```powershell
python -m ruff format --check python tests
python -m ruff check python tests
python -m mypy python\forgeir
python -m pytest tests\python -q
```

Outcome: 38 files were already formatted, Ruff passed, mypy passed 25 source files, and all 69
Python tests passed in 19.20 seconds.

C++ formatting and complete tests:

```powershell
$clangFormat = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\Llvm\x64\bin\clang-format.exe'
$files = @(rg --files cpp bindings tests\cpp -g '*.cpp' -g '*.hpp')
& $clangFormat --dry-run --Werror $files
ctest --preset windows-msvc-debug --output-on-failure
ctest --preset windows-msvc-release --output-on-failure
```

Outcome: formatting passed; all 77 Debug tests passed in 2.08 seconds and all 77 Release tests
passed in 1.84 seconds.

## Measured full local results

All latency values are microseconds. Attempted samples per second counts the two input sequences in
each completed measured iteration.

| Implementation | Min | p50 | p95 | Max | Mean | Stddev | Attempted samples/s | Planned arena bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch eager CPU | 382.200 | 387.250 | 417.245 | 458.000 | 393.935 | 16.664 | 5076.980 | n/a |
| ForgeIR CPU O0 | 5822.700 | 5872.300 | 6368.100 | 6389.000 | 5938.205 | 164.994 | 336.802 | 262144 |
| ForgeIR CPU O1 | 5726.400 | 5796.350 | 6340.470 | 6545.100 | 5872.025 | 207.483 | 340.598 | 262144 |
| ForgeIR CPU O2 | 5688.400 | 5725.000 | 5921.985 | 6107.900 | 5756.850 | 96.531 | 347.412 | 262144 |

ForgeIR was slower on this measured host and protocol. By mean latency, O0 was 15.074 times slower,
O1 was 14.906 times slower, and O2 was 14.614 times slower than PyTorch eager CPU. These statements
describe this one report; they are not universal speed claims.

The largest O2 ForgeIR operation means were the second MLP Linear (`op0033`, 882.925 microseconds),
first MLP Linear (`op0030`, 873.750 microseconds), and parameter validation for their weights
(`op0032`, 538.765 microseconds; `op0029`, 532.085 microseconds). Parameter integrity checking is
part of the current `RuntimeSession::execute` contract and was not removed to improve results.

Every O0/O1/O2 output had maximum absolute error `5.66244125366211e-07`, maximum pointwise relative
error `0.0023864314425736666`, and zero combined-tolerance violations.

## Report artifacts

- `benchmarks/results/milestone_10/full_local_cpu/benchmark_measurements.json`
- `benchmarks/results/milestone_10/full_local_cpu/benchmark_summary.csv`
- `benchmarks/results/milestone_10/full_local_cpu/benchmark_report.html`
- `benchmarks/results/milestone_10/full_local_cpu/report_manifest.sha256`

The measured JSON SHA-256 is
`c5dbf9aa0556beba514619d2f9913f170128e4b56b82aeb351ce38738eb3519e`. Independent verification of
all manifest entries found zero integrity failures.

## Limitations

- Results come from one Windows host, one Release build, one thread setting, and twenty iterations.
- Linux, other CPUs, other compilers, multiple threads, process pinning, and controlled system load
  were not measured.
- The worktree was dirty, so the Git commit is not a complete source identity by itself.
- PyTorch operator data uses a separate profiler clock and groups operator families; it cannot be
  equated one-to-one with stable ForgeIR operation IDs or summed into the unprofiled end-to-end time.
- ForgeIR end-to-end time includes its normal execution validation, output retention, and trace
  collection. Setup, parsing, and rendering are excluded and separately identified.
- Planned arena bytes are static planner calculations, not measured resident or peak process memory.
- Attempted samples per second is a serial protocol calculation, not a concurrent throughput or
  capacity claim.

No package was installed or downloaded during Milestone 10.
