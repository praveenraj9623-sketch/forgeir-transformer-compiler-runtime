"""Integration tests for the stable Milestone 9 pipeline workflow."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import forgeir.pipeline.runner as runner_module
from forgeir.pipeline import (
    PIPELINE_STAGES,
    PipelineResult,
    PipelineStage,
    StageStatus,
    run_pipeline,
)
from forgeir.reference.artifacts import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def _config_document() -> dict[str, object]:
    return {
        "pipeline_schema_version": "1.0",
        "model": {
            "vocabulary_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "sequence_length": 4,
            "batch_size": 1,
            "epsilon": 1.0e-5,
            "dtype": "float32",
            "seed": 42,
        },
        "optimization_level": "O2",
        "alignment_bytes": 64,
        "absolute_tolerance": 2.0e-6,
        "relative_tolerance": 2.0e-5,
    }


def _write_config(path: Path) -> Path:
    path.write_text(
        json.dumps(_config_document(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


@pytest.fixture(scope="module")
def successful_pipeline(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PipelineResult]:
    directory = tmp_path_factory.mktemp("pipeline_success")
    config_path = _write_config(directory / "config.json")
    result = run_pipeline(config_path, directory / "runs")
    assert result.success
    yield result


def test_successful_complete_pipeline_has_all_stages_and_valid_hashes(
    successful_pipeline: PipelineResult,
) -> None:
    result = successful_pipeline
    assert [stage.stage for stage in result.stages] == list(PIPELINE_STAGES)
    assert all(stage.status is StageStatus.SUCCEEDED for stage in result.stages)
    assert result.manifest_path is not None
    assert result.status_path is not None
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    status = json.loads(result.status_path.read_text(encoding="utf-8"))
    assert manifest["success"] is True
    assert status["success"] is True
    assert manifest["seed"] == 42
    assert manifest["stage_order"] == [stage.value for stage in PIPELINE_STAGES]
    for artifact in manifest["artifacts"]:
        path = result.run_directory / artifact["path"]
        assert path.is_file()
        assert path.stat().st_size == artifact["size_bytes"]
        assert file_sha256(path) == artifact["sha256"]
    digest, filename = (
        (result.run_directory / "run_manifest.sha256").read_text().strip().split("  ")
    )
    assert filename == "run_manifest.json"
    assert digest == file_sha256(result.manifest_path)


def test_pipeline_output_parity_report_passes(successful_pipeline: PipelineResult) -> None:
    comparison = json.loads(
        (successful_pipeline.run_directory / "08_compare" / "comparison.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        (successful_pipeline.run_directory / "09_report" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    assert comparison["passed"] is True
    assert comparison["violation_count"] == 0
    assert report["success"] is True
    assert report["numerical_comparison"] == comparison


def test_tampered_weight_fails_at_first_stage_with_preserved_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path / "config.json")
    real_generate = runner_module.generate_reference_artifacts

    def generate_then_tamper(*args: object, **kwargs: object) -> Any:
        paths = real_generate(*args, **kwargs)
        with paths.weight_tensors.open("ab") as stream:
            stream.write(b"tampered-weight")
        return paths

    monkeypatch.setattr(runner_module, "generate_reference_artifacts", generate_then_tamper)
    result = run_pipeline(config_path, tmp_path / "runs")
    assert not result.success
    assert result.failure_stage is PipelineStage.GENERATE_REFERENCE
    assert result.stages[0].status is StageStatus.FAILED
    assert all(stage.status is StageStatus.SKIPPED for stage in result.stages[1:])
    assert "SHA-256 mismatch" in str(result.diagnostics[0]["message"])
    assert result.status_path is not None and result.status_path.is_file()


def test_invalid_exported_graph_stops_before_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path / "config.json")
    real_export = runner_module.export_fx_graph

    def export_then_invalidate(*args: object, **kwargs: object) -> Any:
        result = real_export(*args, **kwargs)
        graph = json.loads(result.graph_path.read_text(encoding="utf-8"))
        graph["producer_version"] = "invalid-after-export"
        result.graph_path.write_text(
            json.dumps(graph, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8"
        )
        return result

    monkeypatch.setattr(runner_module, "export_fx_graph", export_then_invalidate)
    result = run_pipeline(config_path, tmp_path / "runs")
    assert not result.success
    assert result.failure_stage is PipelineStage.EXPORT
    assert result.stages[0].status is StageStatus.SUCCEEDED
    assert result.stages[1].status is StageStatus.FAILED
    assert result.stages[2].status is StageStatus.SKIPPED
    assert "graph hash mismatch" in str(result.diagnostics[0]["message"])


def test_stale_artifact_is_rejected_and_force_recreates_exact_run(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.json")
    output_root = tmp_path / "runs"
    first = run_pipeline(config_path, output_root)
    assert first.success and first.manifest_path is not None
    original_manifest = first.manifest_path.read_bytes()
    cli_failure = subprocess.run(
        [
            sys.executable,
            "-m",
            "forgeir.cli",
            "pipeline",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_root),
            "--json",
        ],
        cwd=ROOT,
        env=_cli_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert cli_failure.returncode != 0
    assert json.loads(cli_failure.stdout)["success"] is False
    report_path = first.run_directory / "09_report" / "report.md"
    report_path.write_text(report_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    stale = run_pipeline(config_path, output_root)
    assert not stale.success
    assert stale.diagnostics[0]["code"] == "stale_artifact"
    assert "hash or size mismatch" in str(stale.diagnostics[0]["message"])

    replaced = run_pipeline(config_path, output_root, force=True)
    assert replaced.success and replaced.manifest_path is not None
    assert replaced.manifest_path.read_bytes() == original_manifest


def test_repeated_isolated_runs_have_byte_identical_manifests(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.json")
    first = run_pipeline(config_path, tmp_path / "first")
    second = run_pipeline(config_path, tmp_path / "second")
    assert first.success and second.success
    assert first.run_id == second.run_id
    assert first.manifest_path is not None and second.manifest_path is not None
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert file_sha256(first.manifest_path) == file_sha256(second.manifest_path)


def test_dry_run_plans_every_stage_without_creating_output(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.json")
    output_root = tmp_path / "absent"
    result = run_pipeline(config_path, output_root, dry_run=True)
    assert result.success and result.dry_run
    assert not output_root.exists()
    assert [stage.stage for stage in result.stages] == list(PIPELINE_STAGES)
    assert all(stage.status is StageStatus.PLANNED for stage in result.stages)
    assert all(not stage.artifacts for stage in result.stages)


def test_pipeline_rejects_any_seed_other_than_42_before_creating_output(tmp_path: Path) -> None:
    document = _config_document()
    model = document["model"]
    assert isinstance(model, dict)
    model["seed"] = 7
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    output_root = tmp_path / "absent"
    with pytest.raises(ValueError, match="requires deterministic seed 42"):
        run_pipeline(config_path, output_root)
    assert not output_root.exists()


def _cli_environment() -> dict[str, str]:
    environment = os.environ.copy()
    python_paths = [str(ROOT / "python"), str(Path(runner_module.forgeir_py.__file__).parent)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    return environment


def test_cli_help_json_quiet_and_rich_dry_run_modes(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.json")
    base = [
        sys.executable,
        "-m",
        "forgeir.cli",
        "pipeline",
        "--config",
        str(config_path),
        "--output-dir",
        str(tmp_path / "dry"),
        "--dry-run",
    ]
    json_result = subprocess.run(
        [*base, "--json"],
        cwd=ROOT,
        env=_cli_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(json_result.stdout)
    assert document["dry_run"] is True
    assert len(document["stages"]) == 9

    quiet_result = subprocess.run(
        [*base, "--quiet"],
        cwd=ROOT,
        env=_cli_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert quiet_result.stdout == ""
    assert quiet_result.stderr == ""

    rich_result = subprocess.run(
        [*base, "--verbose"],
        cwd=ROOT,
        env=_cli_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "ForgeIR pipeline dry run" in rich_result.stdout
    assert "generate-reference" in rich_result.stdout
    assert not (tmp_path / "dry").exists()
