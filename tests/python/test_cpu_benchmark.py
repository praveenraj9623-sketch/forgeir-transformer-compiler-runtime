"""Smoke coverage for truthful CPU benchmarking and JSON-derived reports."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from forgeir.benchmark import (
    BenchmarkArtifacts,
    BenchmarkConfig,
    attempted_samples_per_second,
    latency_statistics,
    render_reports_from_json,
    run_cpu_benchmark,
)
from forgeir.cli import main

ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = ROOT / "config" / "benchmark" / "smoke.json"
FULL_CONFIG = ROOT / "config" / "benchmark" / "full_local_cpu.json"


@pytest.fixture(scope="module")
def smoke_benchmark(tmp_path_factory: pytest.TempPathFactory) -> BenchmarkArtifacts:
    output = tmp_path_factory.mktemp("cpu_benchmark") / "smoke"
    return run_cpu_benchmark(SMOKE_CONFIG, output)


def test_latency_statistics_contract() -> None:
    statistics = latency_statistics([1.0, 2.0, 3.0, 4.0])
    assert statistics == {
        "sample_count": 4,
        "minimum_microseconds": 1.0,
        "p50_microseconds": 2.5,
        "p95_microseconds": pytest.approx(3.85),
        "maximum_microseconds": 4.0,
        "mean_microseconds": 2.5,
        "standard_deviation_microseconds": pytest.approx(math.sqrt(1.25)),
    }
    assert attempted_samples_per_second([100.0, 100.0], batch_size=2) == 20_000.0
    with pytest.raises(ValueError, match="at least one sample"):
        latency_statistics([])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        latency_statistics([math.inf])


def test_benchmark_configuration_is_strict_and_full_is_not_a_test_default(tmp_path: Path) -> None:
    smoke = BenchmarkConfig.load(SMOKE_CONFIG)
    full = BenchmarkConfig.load(FULL_CONFIG)
    assert smoke.model.seed == 42
    assert smoke.optimization_levels == ("O0", "O1", "O2")
    assert full.measured_iteration_count == 20
    assert full.measured_iteration_count > smoke.measured_iteration_count

    document = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    document["model"]["seed"] = 41
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="seed 42"):
        BenchmarkConfig.load(invalid)


def test_smoke_benchmark_measures_all_implementations(
    smoke_benchmark: BenchmarkArtifacts,
) -> None:
    assert smoke_benchmark.success
    document = json.loads(smoke_benchmark.measured_json.read_text(encoding="utf-8"))
    assert document["success"] is True
    assert document["protocol"]["warmup_count"] == 1
    assert document["protocol"]["measured_iteration_count"] == 3
    assert document["thresholds"]["machine_specific_speed_regression_thresholds"] == []
    assert [result["implementation"] for result in document["results"]] == [
        "PyTorch eager CPU",
        "ForgeIR CPU O0",
        "ForgeIR CPU O1",
        "ForgeIR CPU O2",
    ]
    for result in document["results"]:
        samples = result["end_to_end"]["samples_microseconds"]
        statistics = result["end_to_end"]["statistics"]
        assert len(samples) == 3
        assert statistics["sample_count"] == 3
        assert statistics["minimum_microseconds"] >= 0.0
        assert statistics["p50_microseconds"] >= 0.0
        assert statistics["p95_microseconds"] >= 0.0
        assert statistics["maximum_microseconds"] >= statistics["minimum_microseconds"]
        assert statistics["mean_microseconds"] >= 0.0
        assert statistics["standard_deviation_microseconds"] >= 0.0
        assert result["attempted_samples_per_second"] > 0.0
    for result in document["results"][1:]:
        assert result["clock_source"] == "std::chrono::steady_clock"
        assert result["clock_is_steady"] is True
        assert result["planned_arena_bytes"] > 0
        assert result["per_operations"]
        assert all(
            len(operation["samples_microseconds"]) == 3 for operation in result["per_operations"]
        )
    assert all(check["passed"] for check in document["correctness"].values())


def test_reports_exist_and_html_is_rendered_only_from_measured_json(
    smoke_benchmark: BenchmarkArtifacts, tmp_path: Path
) -> None:
    assert smoke_benchmark.summary_csv.is_file()
    assert smoke_benchmark.report_html.is_file()
    assert smoke_benchmark.manifest.is_file()
    csv_text = smoke_benchmark.summary_csv.read_text(encoding="utf-8")
    assert "attempted_samples_per_second" in csv_text
    assert "ForgeIR CPU O2" in csv_text

    document = json.loads(smoke_benchmark.measured_json.read_text(encoding="utf-8"))
    document["conclusions"] = ["JSON-only rendering sentinel"]
    measured_copy = tmp_path / "benchmark_measurements.json"
    measured_copy.write_text(json.dumps(document), encoding="utf-8")
    _, html_path = render_reports_from_json(measured_copy)
    assert "JSON-only rendering sentinel" in html_path.read_text(encoding="utf-8")


def test_existing_output_requires_explicit_force(
    smoke_benchmark: BenchmarkArtifacts, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "benchmark",
            "--config",
            str(SMOKE_CONFIG),
            "--output-dir",
            str(smoke_benchmark.output_directory),
            "--json",
        ]
    )
    assert exit_code == 2
    failure = json.loads(capsys.readouterr().out)
    assert failure["diagnostics"][0]["code"] == "benchmark_failed"
    assert "already exists" in failure["diagnostics"][0]["message"]
