"""Truthful setup-separated CPU benchmark orchestration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

import forgeir_py
from forgeir.benchmark.environment import capture_environment
from forgeir.benchmark.report import render_reports_from_json
from forgeir.benchmark.statistics import attempted_samples_per_second, latency_statistics
from forgeir.benchmark.types import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkArtifacts,
    BenchmarkConfig,
)
from forgeir.export.fx_export import export_fx_graph
from forgeir.reference import (
    array_content_sha256,
    create_deterministic_model,
    generate_reference_artifacts,
    verify_manifest,
)
from forgeir.reference.artifacts import file_sha256

FloatArray = NDArray[np.float32]


def _elapsed_milliseconds(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON document must contain an object: {path}")
    return cast(dict[str, Any], document)


def _native_cli(repository_root: Path) -> Path:
    executable_name = "forgeir_cli.exe" if os.name == "nt" else "forgeir_cli"
    binding_directory = Path(forgeir_py.__file__).resolve().parent
    candidates = (
        binding_directory.parent / "bin" / executable_name,
        repository_root / "build" / "windows-msvc-release" / "bin" / executable_name,
        repository_root / "build" / "windows-msvc-debug" / "bin" / executable_name,
        repository_root / "build" / "linux-gcc-release" / "bin" / executable_name,
        repository_root / "build" / "linux-gcc-debug" / "bin" / executable_name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError("forgeir_cli must be built before running CPU benchmarks")


def _optimize_graph(
    repository_root: Path, input_graph: Path, output_graph: Path, level: str
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(_native_cli(repository_root)),
            "optimize",
            str(input_graph),
            "--level",
            level,
            "--output",
            str(output_graph),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"ForgeIR {level} optimization failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    report = json.loads(completed.stdout)
    if not isinstance(report, dict):
        raise ValueError(f"ForgeIR {level} optimization did not produce a JSON report")
    return cast(dict[str, Any], report)


def _assert_matching_model_weights(model: torch.nn.Module, weight_archive: Path) -> None:
    with np.load(weight_archive, allow_pickle=False) as archive:
        for name, parameter in sorted(model.state_dict().items()):
            archive_key = name.replace(".", "__")
            if archive_key not in archive:
                raise ValueError(f"reference weight archive is missing {archive_key}")
            expected = parameter.detach().cpu().contiguous().numpy()
            if not np.array_equal(expected, archive[archive_key]):
                raise ValueError(f"PyTorch and ForgeIR weight content differs for {archive_key}")


def _pytorch_end_to_end(
    block: torch.nn.Module,
    hidden_states: torch.Tensor,
    warmup_count: int,
    measured_iteration_count: int,
) -> tuple[list[float], torch.Tensor]:
    with torch.inference_mode():
        for _ in range(warmup_count):
            block(hidden_states)
        samples: list[float] = []
        output: torch.Tensor | None = None
        for _ in range(measured_iteration_count):
            start = time.perf_counter_ns()
            output = cast(torch.Tensor, block(hidden_states))
            end = time.perf_counter_ns()
            samples.append((end - start) / 1_000.0)
    if output is None:
        raise ValueError("PyTorch measured iteration count must be positive")
    return samples, output


def _pytorch_operation_profile(
    block: torch.nn.Module, hidden_states: torch.Tensor, iterations: int
) -> list[dict[str, object]]:
    if iterations == 0:
        return []
    iteration_profiles: list[dict[str, tuple[float, int]]] = []
    with torch.inference_mode():
        for _ in range(iterations):
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as profile:
                block(hidden_states)
            events: dict[str, tuple[float, int]] = {}
            for event in profile.key_averages():
                events[str(event.key)] = (float(event.self_cpu_time_total), int(event.count))
            iteration_profiles.append(events)
    operation_names = sorted({name for profile in iteration_profiles for name in profile})
    result: list[dict[str, object]] = []
    for name in operation_names:
        samples = [profile.get(name, (0.0, 0))[0] for profile in iteration_profiles]
        call_counts = [profile.get(name, (0.0, 0))[1] for profile in iteration_profiles]
        result.append(
            {
                "scope": "pytorch_operator_family_self_cpu_time",
                "operation_id": name,
                "operation_type": name,
                "kernel": "PyTorch CPU profiler",
                "samples_microseconds": samples,
                "call_counts_per_profiled_iteration": call_counts,
                "statistics": latency_statistics(samples),
            }
        )
    return result


def _external_tensors(
    reference_directory: Path, hidden_states: torch.Tensor
) -> tuple[dict[str, FloatArray], dict[str, FloatArray]]:
    runtime_input = np.ascontiguousarray(hidden_states.detach().cpu().numpy(), dtype=np.float32)
    with np.load(reference_directory / "weight_tensors.npz", allow_pickle=False) as archive:
        parameters = {
            name: np.ascontiguousarray(archive[name], dtype=np.float32) for name in archive.files
        }
    return {"v0000": runtime_input}, parameters


def _forgeir_result(
    implementation: str,
    level: str,
    session: object,
    inputs: dict[str, FloatArray],
    parameters: dict[str, FloatArray],
    config: BenchmarkConfig,
) -> dict[str, object]:
    raw = cast(
        dict[str, Any],
        forgeir_py.benchmark_execute(
            session,
            inputs,
            parameters,
            config.warmup_count,
            config.measured_iteration_count,
        ),
    )
    samples = cast(list[float], raw["end_to_end_samples_microseconds"])
    operations: list[dict[str, object]] = []
    for operation in cast(list[dict[str, Any]], raw["operations"]):
        operation_samples = cast(list[float], operation["samples_microseconds"])
        operations.append(
            {
                "scope": "forgeir_canonical_operation",
                "operation_id": operation["operation_id"],
                "operation_type": operation["operation_type"],
                "kernel": operation["kernel"],
                "output_shape": operation["output_shape"],
                "samples_microseconds": operation_samples,
                "statistics": latency_statistics(operation_samples),
            }
        )
    return {
        "implementation": implementation,
        "optimization_level": level,
        "clock_source": raw["clock_source"],
        "clock_is_steady": raw["clock_is_steady"],
        "end_to_end": {
            "samples_microseconds": samples,
            "statistics": latency_statistics(samples),
        },
        "attempted_samples_per_second": attempted_samples_per_second(
            samples, config.model.batch_size
        ),
        "planned_arena_bytes": raw["planned_arena_bytes"],
        "peak_live_bytes": raw["peak_live_bytes"],
        "naive_allocation_bytes": raw["naive_allocation_bytes"],
        "operation_profile_contract": (
            "Each stable ForgeIR operation ID is timed during the measured end-to-end "
            "iterations with std::chrono::steady_clock. Runtime tracing overhead is included."
        ),
        "per_operations": operations,
    }


def _correctness(
    expected: FloatArray,
    actual: FloatArray,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return {
            "passed": False,
            "reason": "shape or dtype mismatch",
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
            "expected_dtype": str(expected.dtype),
            "actual_dtype": str(actual.dtype),
        }
    finite = np.isfinite(expected)
    classifications_match = bool(
        np.array_equal(np.isnan(expected), np.isnan(actual))
        and np.array_equal(np.isposinf(expected), np.isposinf(actual))
        and np.array_equal(np.isneginf(expected), np.isneginf(actual))
    )
    absolute_error = np.abs(actual[finite] - expected[finite])
    tolerance = absolute_tolerance + relative_tolerance * np.abs(expected[finite])
    relative_error = absolute_error / np.maximum(np.abs(expected[finite]), np.float32(1.0e-12))
    violations = int(np.count_nonzero(absolute_error > tolerance))
    return {
        "passed": classifications_match and violations == 0,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "maximum_absolute_error": float(absolute_error.max(initial=0.0)),
        "maximum_relative_error": float(relative_error.max(initial=0.0)),
        "violation_count": violations,
        "nonfinite_classifications_match": classifications_match,
    }


def _conclusions(results: list[dict[str, object]]) -> list[str]:
    pytorch_end_to_end = cast(dict[str, object], results[0]["end_to_end"])
    pytorch_statistics = cast(dict[str, float], pytorch_end_to_end["statistics"])
    pytorch_mean = pytorch_statistics["mean_microseconds"]
    conclusions: list[str] = []
    for result in results[1:]:
        statistics = cast(
            dict[str, Any], cast(dict[str, object], result["end_to_end"])["statistics"]
        )
        mean = float(statistics["mean_microseconds"])
        ratio = mean / pytorch_mean
        name = str(result["implementation"])
        if ratio > 1.0:
            conclusions.append(
                f"{name} was {ratio:.3f}x slower than PyTorch eager CPU by measured mean latency."
            )
        elif ratio < 1.0:
            conclusions.append(
                f"{name} measured {1.0 / ratio:.3f}x lower mean latency than PyTorch eager CPU."
            )
        else:
            conclusions.append(f"{name} and PyTorch eager CPU had equal measured mean latency.")
    return conclusions


def _manifest(output_directory: Path, paths: list[Path]) -> Path:
    manifest = output_directory / "report_manifest.sha256"
    lines = [
        f"{file_sha256(path)}  {path.relative_to(output_directory).as_posix()}\n"
        for path in sorted(paths)
    ]
    manifest.write_text("".join(lines), encoding="utf-8")
    return manifest


def run_cpu_benchmark(
    config_path: Path, output_directory: Path, *, force: bool = False
) -> BenchmarkArtifacts:
    """Run the configured CPU comparison and persist raw measurements before rendering."""
    config_path = config_path.resolve()
    output_directory = output_directory.resolve()
    config = BenchmarkConfig.load(config_path)
    repository_root = Path(__file__).resolve().parents[3]
    if output_directory.exists():
        if not force:
            raise ValueError(f"benchmark output directory already exists: {output_directory}")
        unsafe_directories = {
            repository_root.resolve(),
            Path.home().resolve(),
            Path(output_directory.anchor).resolve(),
        }
        if output_directory in unsafe_directories:
            raise ValueError("refusing to remove an unsafe benchmark output directory")
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    config_copy = output_directory / "benchmark_config.json"
    _write_json(config_copy, config.as_dict())

    torch.set_num_threads(config.torch_num_threads)
    setup_timings: dict[str, float] = {}
    reference_directory = output_directory / "reference"
    start = time.perf_counter_ns()
    reference_paths = generate_reference_artifacts(reference_directory, config.model)
    verify_manifest(reference_directory)
    setup_timings["reference_generation_milliseconds"] = _elapsed_milliseconds(start)

    export_directory = output_directory / "export"
    start = time.perf_counter_ns()
    export_result = export_fx_graph(
        export_directory,
        reference_directory,
        config=config.model,
        weight_manifest_reference="reference/manifest.sha256",
    )
    setup_timings["fx_export_milliseconds"] = _elapsed_milliseconds(start)

    graph_paths: dict[str, Path] = {}
    graph_reports: dict[str, dict[str, Any]] = {}
    for level in config.optimization_levels:
        level_directory = output_directory / "graphs" / level
        level_directory.mkdir(parents=True, exist_ok=True)
        output_graph = level_directory / "tiny_transformer_block.graph.json"
        start = time.perf_counter_ns()
        graph_reports[level] = _optimize_graph(
            repository_root, export_result.graph_path, output_graph, level
        )
        setup_timings[f"optimization_{level}_milliseconds"] = _elapsed_milliseconds(start)
        graph_paths[level] = output_graph

    model = create_deterministic_model(config.model)
    _assert_matching_model_weights(model, reference_paths.weight_tensors)
    with np.load(reference_paths.input_tensor, allow_pickle=False) as archive:
        input_ids = torch.from_numpy(np.ascontiguousarray(archive["input_ids"]))
    with torch.inference_mode():
        hidden_states = model.token_embedding(input_ids).contiguous()
    inputs, parameters = _external_tensors(reference_directory, hidden_states)

    start = time.perf_counter_ns()
    pytorch_samples, pytorch_output = _pytorch_end_to_end(
        model.block,
        hidden_states,
        config.warmup_count,
        config.measured_iteration_count,
    )
    setup_timings["pytorch_operation_profile_milliseconds"] = 0.0
    profile_start = time.perf_counter_ns()
    pytorch_operations = _pytorch_operation_profile(
        model.block, hidden_states, config.pytorch_operation_profile_iterations
    )
    setup_timings["pytorch_operation_profile_milliseconds"] = _elapsed_milliseconds(profile_start)
    pytorch_result: dict[str, object] = {
        "implementation": "PyTorch eager CPU",
        "optimization_level": None,
        "clock_source": "Python time.perf_counter_ns (monotonic performance counter)",
        "clock_is_steady": True,
        "end_to_end": {
            "samples_microseconds": pytorch_samples,
            "statistics": latency_statistics(pytorch_samples),
        },
        "attempted_samples_per_second": attempted_samples_per_second(
            pytorch_samples, config.model.batch_size
        ),
        "planned_arena_bytes": None,
        "peak_live_bytes": None,
        "naive_allocation_bytes": None,
        "operation_profile_contract": (
            "PyTorch operator-family self CPU times are collected in a separate profiler run "
            "and are not included in end-to-end samples. Zero configured profile iterations "
            "produce no PyTorch per-operation rows."
        ),
        "per_operations": pytorch_operations,
    }

    results: list[dict[str, object]] = [pytorch_result]
    sessions: dict[str, object] = {}
    for level in config.optimization_levels:
        start = time.perf_counter_ns()
        session = forgeir_py.load_graph(str(graph_paths[level]), "cpu")
        setup_timings[f"session_load_{level}_milliseconds"] = _elapsed_milliseconds(start)
        sessions[level] = session
        results.append(
            _forgeir_result(f"ForgeIR CPU {level}", level, session, inputs, parameters, config)
        )

    expected = np.ascontiguousarray(pytorch_output.detach().cpu().numpy(), dtype=np.float32)
    correctness: dict[str, dict[str, object]] = {}
    for level in config.optimization_levels:
        outputs = cast(dict[str, FloatArray], forgeir_py.get_outputs(sessions[level]))
        if len(outputs) != 1:
            raise ValueError(f"ForgeIR {level} runtime returned {len(outputs)} declared outputs")
        actual = np.ascontiguousarray(next(iter(outputs.values())), dtype=np.float32)
        correctness[level] = _correctness(
            expected,
            actual,
            config.absolute_tolerance,
            config.relative_tolerance,
        )

    gross_limit_microseconds = config.gross_failure_max_iteration_seconds * 1_000_000.0
    gross_failures = [
        str(result["implementation"])
        for result in results
        if max(
            cast(list[float], cast(dict[str, object], result["end_to_end"])["samples_microseconds"])
        )
        > gross_limit_microseconds
    ]
    success = not gross_failures and all(item["passed"] is True for item in correctness.values())
    conclusions = _conclusions(results)
    if gross_failures:
        conclusions.append(
            "Gross-failure ceiling exceeded by: " + ", ".join(sorted(gross_failures)) + "."
        )

    environment = capture_environment(repository_root)
    graph_identities = {
        level: {
            "path": graph_paths[level].relative_to(output_directory).as_posix(),
            "file_sha256": file_sha256(graph_paths[level]),
            "graph_hash": _load_json(graph_paths[level])["graph_hash"],
            "operation_counts": graph_reports[level].get("operation_counts", {}),
        }
        for level in config.optimization_levels
    }
    measured_document = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "success": success,
        "configuration_hash": config.configuration_hash,
        "protocol": {
            "warmup_count": config.warmup_count,
            "measured_iteration_count": config.measured_iteration_count,
            "pytorch_operation_profile_iterations": config.pytorch_operation_profile_iterations,
            "process_settings": {
                "process_count": config.process_count,
                "torch_num_threads": config.torch_num_threads,
                "forgeir_runtime_threads": 1,
                "affinity": "unchanged",
            },
            "input_shapes": {
                "token_ids": list(input_ids.shape),
                "block_input": list(hidden_states.shape),
            },
            "statistic_definitions": {
                "p50_p95": "linear interpolation at ranks (n - 1) * 0.50 and (n - 1) * 0.95",
                "mean": "arithmetic mean over retained measured iterations",
                "standard_deviation": "population standard deviation over measured iterations",
                "attempted_samples_per_second": (
                    "measured_iteration_count * batch_size divided by summed measured seconds"
                ),
            },
            "clock_sources": {
                "forgeir": "std::chrono::steady_clock in C++",
                "pytorch_end_to_end": "Python time.perf_counter_ns monotonic clock",
                "pytorch_operation_profile": "PyTorch CPU profiler self time",
            },
            "excluded_from_end_to_end_latency": [
                "reference generation",
                "FX export",
                "graph optimization",
                "graph parsing and session loading",
                "input and weight loading",
                "PyTorch operation-profiler runs",
                "correctness comparison",
                "environment capture",
                "JSON, CSV, and HTML rendering",
            ],
        },
        "thresholds": {
            "correctness": {
                "absolute_tolerance": config.absolute_tolerance,
                "relative_tolerance": config.relative_tolerance,
            },
            "gross_failure_max_iteration_seconds": config.gross_failure_max_iteration_seconds,
            "machine_specific_speed_regression_thresholds": [],
        },
        "environment": environment,
        "input_identity": {
            "seed": 42,
            "runtime_input_content_sha256": array_content_sha256(inputs["v0000"]),
            "input_archive_sha256": file_sha256(reference_paths.input_tensor),
            "weight_archive_sha256": file_sha256(reference_paths.weight_tensors),
            "reference_manifest_sha256": file_sha256(reference_paths.manifest),
        },
        "graphs": graph_identities,
        "setup_timings_milliseconds": setup_timings,
        "results": results,
        "correctness": correctness,
        "conclusions": conclusions,
    }
    measured_json = output_directory / "benchmark_measurements.json"
    _write_json(measured_json, measured_document)
    summary_csv, report_html = render_reports_from_json(measured_json)
    manifest_paths = [
        config_copy,
        reference_paths.input_tensor,
        reference_paths.weight_tensors,
        reference_paths.manifest,
        export_result.graph_path,
        *graph_paths.values(),
        measured_json,
        summary_csv,
        report_html,
    ]
    manifest = _manifest(output_directory, manifest_paths)
    return BenchmarkArtifacts(
        success=success,
        output_directory=output_directory,
        measured_json=measured_json,
        summary_csv=summary_csv,
        report_html=report_html,
        manifest=manifest,
        conclusions=tuple(conclusions),
    )
