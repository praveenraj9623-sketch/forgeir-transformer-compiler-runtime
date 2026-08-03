"""Deterministic fail-fast orchestration for all completed ForgeIR components."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

import forgeir_py
from forgeir.export.contract import verify_graph_hash
from forgeir.export.fx_export import export_fx_graph
from forgeir.reference import (
    create_deterministic_model,
    generate_reference_artifacts,
    verify_manifest,
    write_deterministic_npz,
)
from forgeir.reference.artifacts import file_sha256

from .types import (
    PIPELINE_SCHEMA_VERSION,
    PIPELINE_STAGES,
    ArtifactDigest,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    StageResult,
    StageStatus,
)


@dataclass(frozen=True, slots=True)
class _StageExecution:
    artifacts: tuple[Path, ...]
    summary: dict[str, object]
    success: bool = True
    diagnostics: tuple[dict[str, object], ...] = ()


class _StageFailure(RuntimeError):
    def __init__(self, code: str, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def diagnostic(self) -> dict[str, object]:
        return {"code": self.code, "message": str(self), "details": self.details}


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return cast(dict[str, Any], document)


def _native_json(text: str) -> object:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


class PipelineRunner:
    """One isolated deterministic pipeline run."""

    def __init__(
        self,
        config_path: Path,
        output_directory: Path,
        *,
        force: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.source_config_path = config_path.resolve()
        self.config = PipelineConfig.load(self.source_config_path)
        self.output_directory = output_directory.resolve()
        self.run_directory = self.output_directory / self.config.run_id
        self.force = force
        self.dry_run = dry_run
        self.repository_root = Path(__file__).resolve().parents[3]
        self.status_path = self.run_directory / "status.json"
        self.manifest_path = self.run_directory / "run_manifest.json"
        self.manifest_digest_path = self.run_directory / "run_manifest.sha256"
        self._ledger: dict[str, ArtifactDigest] = {}
        self._stage_results: list[StageResult] = []

    def run(self) -> PipelineResult:
        """Execute every stage in order, stopping and persisting the first failure."""
        if self.dry_run:
            return PipelineResult(
                success=True,
                dry_run=True,
                run_id=self.config.run_id,
                run_directory=self.run_directory,
                stages=tuple(
                    StageResult(stage=stage, status=StageStatus.PLANNED)
                    for stage in PIPELINE_STAGES
                ),
            )

        preparation_failure = self._prepare_run_directory()
        if preparation_failure is not None:
            return preparation_failure

        self._record_setup_inputs()
        self._persist(success=False, failure_stage=None, diagnostics=())

        stage_functions = {
            PipelineStage.GENERATE_REFERENCE: self._generate_reference,
            PipelineStage.EXPORT: self._export,
            PipelineStage.INSPECT: self._inspect,
            PipelineStage.VERIFY: self._verify,
            PipelineStage.OPTIMIZE: self._optimize,
            PipelineStage.PLAN_MEMORY: self._plan_memory,
            PipelineStage.RUN: self._execute,
            PipelineStage.COMPARE: self._compare,
            PipelineStage.REPORT: self._report,
        }
        for stage in PIPELINE_STAGES:
            input_hashes = dict(sorted((path, item.sha256) for path, item in self._ledger.items()))
            try:
                self._validate_ledger()
                execution = stage_functions[stage]()
                artifacts = self._record_artifacts(execution.artifacts)
                stage_result = StageResult(
                    stage=stage,
                    status=StageStatus.SUCCEEDED if execution.success else StageStatus.FAILED,
                    input_hashes=input_hashes,
                    artifacts=artifacts,
                    diagnostics=execution.diagnostics,
                    summary=execution.summary,
                )
                self._stage_results.append(stage_result)
                if not execution.success:
                    self._append_skipped_stages(stage)
                    self._persist(
                        success=False,
                        failure_stage=stage,
                        diagnostics=execution.diagnostics,
                    )
                    return self._result(
                        success=False,
                        failure_stage=stage,
                        diagnostics=execution.diagnostics,
                    )
                self._persist(success=False, failure_stage=None, diagnostics=())
            except Exception as error:
                diagnostic = self._diagnostic(error)
                partial_artifacts = self._record_artifacts(self._partial_stage_files(stage))
                self._stage_results.append(
                    StageResult(
                        stage=stage,
                        status=StageStatus.FAILED,
                        input_hashes=input_hashes,
                        artifacts=partial_artifacts,
                        diagnostics=(diagnostic,),
                    )
                )
                self._append_skipped_stages(stage)
                self._persist(
                    success=False,
                    failure_stage=stage,
                    diagnostics=(diagnostic,),
                )
                return self._result(
                    success=False,
                    failure_stage=stage,
                    diagnostics=(diagnostic,),
                )

        self._persist(success=True, failure_stage=None, diagnostics=())
        return self._result(success=True, failure_stage=None, diagnostics=())

    def _result(
        self,
        *,
        success: bool,
        failure_stage: PipelineStage | None,
        diagnostics: tuple[dict[str, object], ...],
    ) -> PipelineResult:
        return PipelineResult(
            success=success,
            dry_run=False,
            run_id=self.config.run_id,
            run_directory=self.run_directory,
            stages=tuple(self._stage_results),
            manifest_path=self.manifest_path,
            status_path=self.status_path,
            failure_stage=failure_stage,
            diagnostics=diagnostics,
        )

    def _prepare_run_directory(self) -> PipelineResult | None:
        if self.run_directory.exists() and not self.force:
            diagnostic = self._validate_existing_run()
            failure_status_path = (
                self.output_directory / f"{self.config.run_id}.failure_status.json"
            )
            _write_json(
                failure_status_path,
                {
                    "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
                    "success": False,
                    "run_id": self.config.run_id,
                    "failure_stage": PipelineStage.GENERATE_REFERENCE.value,
                    "diagnostics": [diagnostic],
                },
            )
            return PipelineResult(
                success=False,
                dry_run=False,
                run_id=self.config.run_id,
                run_directory=self.run_directory,
                stages=(),
                status_path=failure_status_path,
                failure_stage=PipelineStage.GENERATE_REFERENCE,
                diagnostics=(diagnostic,),
            )
        if self.run_directory.exists():
            resolved_run = self.run_directory.resolve()
            resolved_root = self.output_directory.resolve()
            if resolved_run.parent != resolved_root or resolved_run.name != self.config.run_id:
                raise ValueError("refusing to force-remove a run directory outside the output root")
            shutil.rmtree(resolved_run)
        self.run_directory.mkdir(parents=True, exist_ok=False)
        return None

    def _validate_existing_run(self) -> dict[str, object]:
        if not self.manifest_path.is_file():
            return {
                "code": "stale_artifact",
                "message": (
                    "existing run directory has no run_manifest.json; use --force to replace it"
                ),
                "details": {"run_id": self.config.run_id},
            }
        try:
            if not self.manifest_digest_path.is_file():
                raise ValueError("missing run_manifest.sha256")
            digest_parts = self.manifest_digest_path.read_text(encoding="utf-8").strip().split("  ")
            if digest_parts != [file_sha256(self.manifest_path), "run_manifest.json"]:
                raise ValueError("run manifest digest mismatch")
            manifest = _load_json(self.manifest_path)
            if manifest.get("configuration_hash") != self.config.configuration_hash:
                raise ValueError("configuration hash mismatch")
            artifacts = manifest.get("artifacts")
            if not isinstance(artifacts, list):
                raise ValueError("manifest artifacts must be an array")
            for raw_artifact in artifacts:
                if not isinstance(raw_artifact, dict):
                    raise ValueError("manifest artifact entry must be an object")
                relative = raw_artifact.get("path")
                digest = raw_artifact.get("sha256")
                size = raw_artifact.get("size_bytes")
                if (
                    not isinstance(relative, str)
                    or not isinstance(digest, str)
                    or not isinstance(size, int)
                ):
                    raise ValueError("manifest artifact entry has invalid fields")
                path = (self.run_directory / relative).resolve()
                if not path.is_relative_to(self.run_directory.resolve()) or not path.is_file():
                    raise ValueError(f"missing or unsafe artifact {relative}")
                if path.stat().st_size != size or file_sha256(path) != digest:
                    raise ValueError(f"hash or size mismatch for {relative}")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return {
                "code": "stale_artifact",
                "message": f"existing run artifacts failed validation: {error}",
                "details": {"run_id": self.config.run_id},
            }
        return {
            "code": "run_directory_exists",
            "message": "validated run directory already exists; use --force to replace it",
            "details": {"run_id": self.config.run_id},
        }

    def _record_setup_inputs(self) -> None:
        input_directory = self.run_directory / "input"
        input_directory.mkdir(parents=True)
        source_copy = input_directory / "source_config.json"
        normalized_copy = input_directory / "normalized_config.json"
        shutil.copyfile(self.source_config_path, source_copy)
        normalized_copy.write_text(self.config.canonical_json + "\n", encoding="utf-8")
        self._record_artifacts((source_copy, normalized_copy))

    def _artifact(self, path: Path) -> ArtifactDigest:
        resolved = path.resolve()
        run_root = self.run_directory.resolve()
        if not resolved.is_relative_to(run_root) or not resolved.is_file():
            raise _StageFailure("artifact_missing", f"expected run artifact is missing: {path}")
        relative = resolved.relative_to(run_root).as_posix()
        return ArtifactDigest(relative, file_sha256(resolved), resolved.stat().st_size)

    def _record_artifacts(self, paths: tuple[Path, ...]) -> tuple[ArtifactDigest, ...]:
        artifacts: list[ArtifactDigest] = []
        for path in sorted(set(paths), key=lambda item: item.as_posix()):
            artifact = self._artifact(path)
            self._ledger[artifact.path] = artifact
            artifacts.append(artifact)
        return tuple(artifacts)

    def _validate_ledger(self) -> None:
        for relative, recorded in sorted(self._ledger.items()):
            path = (self.run_directory / relative).resolve()
            if not path.is_relative_to(self.run_directory.resolve()) or not path.is_file():
                raise _StageFailure(
                    "stale_artifact",
                    f"recorded artifact is missing before stage execution: {relative}",
                )
            actual_size = path.stat().st_size
            actual_hash = file_sha256(path)
            if actual_size != recorded.size_bytes or actual_hash != recorded.sha256:
                raise _StageFailure(
                    "stale_artifact",
                    f"recorded artifact changed before stage execution: {relative}",
                    {
                        "expected_sha256": recorded.sha256,
                        "actual_sha256": actual_hash,
                        "expected_size_bytes": recorded.size_bytes,
                        "actual_size_bytes": actual_size,
                    },
                )

    def _persist(
        self,
        *,
        success: bool,
        failure_stage: PipelineStage | None,
        diagnostics: tuple[dict[str, object], ...],
    ) -> None:
        status = {
            "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
            "success": success,
            "run_id": self.config.run_id,
            "configuration_hash": self.config.configuration_hash,
            "failure_stage": failure_stage.value if failure_stage is not None else None,
            "diagnostics": list(diagnostics),
            "stages": [stage.as_dict() for stage in self._stage_results],
        }
        _write_json(self.status_path, status)
        status_artifact = self._artifact(self.status_path)
        self._ledger[status_artifact.path] = status_artifact
        manifest = {
            "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
            "run_id": self.config.run_id,
            "configuration_hash": self.config.configuration_hash,
            "seed": 42,
            "success": success,
            "failure_stage": failure_stage.value if failure_stage is not None else None,
            "stage_order": [stage.value for stage in PIPELINE_STAGES],
            "stages": [stage.as_dict() for stage in self._stage_results],
            "artifacts": [self._ledger[path].as_dict() for path in sorted(self._ledger)],
        }
        _write_json(self.manifest_path, manifest)
        self.manifest_digest_path.write_text(
            f"{file_sha256(self.manifest_path)}  run_manifest.json\n", encoding="utf-8"
        )

    def _append_skipped_stages(self, failed_stage: PipelineStage) -> None:
        failed_index = PIPELINE_STAGES.index(failed_stage)
        for stage in PIPELINE_STAGES[failed_index + 1 :]:
            self._stage_results.append(StageResult(stage=stage, status=StageStatus.SKIPPED))

    def _diagnostic(self, error: Exception) -> dict[str, object]:
        if isinstance(error, _StageFailure):
            return error.diagnostic()
        return {
            "code": "stage_exception",
            "message": str(error),
            "details": {"exception_type": type(error).__name__},
        }

    def _stage_directory(self, stage: PipelineStage) -> Path:
        index = PIPELINE_STAGES.index(stage) + 1
        return self.run_directory / f"{index:02d}_{stage.value.replace('-', '_')}"

    def _partial_stage_files(self, stage: PipelineStage) -> tuple[Path, ...]:
        directory = self._stage_directory(stage)
        if not directory.is_dir():
            return ()
        return tuple(path for path in directory.rglob("*") if path.is_file())

    def _generate_reference(self) -> _StageExecution:
        directory = self._stage_directory(PipelineStage.GENERATE_REFERENCE)
        paths = generate_reference_artifacts(directory, self.config.model)
        verify_manifest(directory)
        artifacts = (
            paths.configuration,
            paths.input_tensor,
            paths.weight_tensors,
            paths.expected_output,
            paths.tensor_metadata,
            paths.manifest,
        )
        return _StageExecution(artifacts, {"seed": self.config.model.seed})

    def _export(self) -> _StageExecution:
        reference_directory = self._stage_directory(PipelineStage.GENERATE_REFERENCE)
        directory = self._stage_directory(PipelineStage.EXPORT)
        result = export_fx_graph(
            directory,
            reference_directory,
            config=self.config.model,
            weight_manifest_reference="01_generate_reference/manifest.sha256",
        )
        graph = _load_json(result.graph_path)
        verify_graph_hash(graph)
        forgeir_py.graph_summary(str(result.graph_path))
        return _StageExecution(
            (result.graph_path, result.dot_path),
            {"graph_hash": result.graph_hash, "operation_counts": result.operation_counts},
        )

    def _inspect(self) -> _StageExecution:
        graph_path = (
            self._stage_directory(PipelineStage.EXPORT) / "tiny_transformer_block.graph.json"
        )
        summary = cast(dict[str, object], forgeir_py.graph_summary(str(graph_path)))
        path = self._stage_directory(PipelineStage.INSPECT) / "graph_summary.json"
        _write_json(path, summary)
        return _StageExecution((path,), summary)

    def _verify(self) -> _StageExecution:
        graph_path = (
            self._stage_directory(PipelineStage.EXPORT) / "tiny_transformer_block.graph.json"
        )
        report = cast(dict[str, object], forgeir_py.verify_graph(str(graph_path)))
        path = self._stage_directory(PipelineStage.VERIFY) / "verification_report.json"
        _write_json(path, report)
        success = report.get("success") is True
        diagnostics: tuple[dict[str, object], ...] = (
            ()
            if success
            else (
                {
                    "code": "graph_verification_failed",
                    "message": "graph semantic verification failed",
                    "details": {"report": path.name},
                },
            )
        )
        return _StageExecution((path,), {"verified": success}, success, diagnostics)

    def _native_cli(self) -> Path:
        binding_directory = Path(forgeir_py.__file__).resolve().parent
        executable_name = "forgeir_cli.exe" if os.name == "nt" else "forgeir_cli"
        candidates = (
            binding_directory.parent / "bin" / executable_name,
            self.repository_root / "build" / "windows-msvc-release" / "bin" / executable_name,
            self.repository_root / "build" / "windows-msvc-debug" / "bin" / executable_name,
            self.repository_root / "build" / "linux-gcc-release" / "bin" / executable_name,
            self.repository_root / "build" / "linux-gcc-debug" / "bin" / executable_name,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise _StageFailure(
            "native_cli_missing",
            "forgeir_cli must be built before running the end-to-end pipeline",
        )

    def _run_native(self, arguments: list[str]) -> dict[str, Any]:
        command = [str(self._native_cli()), *arguments]
        completed = subprocess.run(
            command,
            cwd=self.repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise _StageFailure(
                "native_command_failed",
                f"native command failed with exit code {completed.returncode}",
                {
                    "command": [Path(command[0]).name, *arguments],
                    "exit_code": completed.returncode,
                    "stdout": _native_json(completed.stdout),
                    "stderr": _native_json(completed.stderr),
                },
            )
        parsed = _native_json(completed.stdout)
        if not isinstance(parsed, dict):
            raise _StageFailure(
                "native_output_invalid", "native command did not return a JSON object"
            )
        return cast(dict[str, Any], parsed)

    def _optimize(self) -> _StageExecution:
        input_graph = (
            self._stage_directory(PipelineStage.EXPORT) / "tiny_transformer_block.graph.json"
        )
        directory = self._stage_directory(PipelineStage.OPTIMIZE)
        directory.mkdir(parents=True, exist_ok=True)
        output_graph = directory / "tiny_transformer_block.graph.json"
        summary = self._run_native(
            [
                "optimize",
                str(input_graph),
                "--level",
                self.config.optimization_level,
                "--output",
                str(output_graph),
            ]
        )
        artifact_base = output_graph.parent / output_graph.stem
        pass_report = Path(f"{artifact_base}.pass_report.json")
        before_dot = Path(f"{artifact_base}.before.dot")
        after_dot = Path(f"{artifact_base}.after.dot")
        report = _load_json(pass_report)
        report["artifacts"] = {
            "optimized_graph": output_graph.relative_to(self.run_directory).as_posix(),
            "pass_report": pass_report.relative_to(self.run_directory).as_posix(),
            "before_dot": before_dot.relative_to(self.run_directory).as_posix(),
            "after_dot": after_dot.relative_to(self.run_directory).as_posix(),
        }
        _write_json(pass_report, report)
        optimized = _load_json(output_graph)
        verify_graph_hash(optimized)
        forgeir_py.graph_summary(str(output_graph))
        return _StageExecution(
            (output_graph, pass_report, before_dot, after_dot),
            {
                "optimization_level": self.config.optimization_level,
                "graph_hash": optimized["graph_hash"],
                "operation_counts": summary.get("operation_counts", {}),
            },
        )

    def _plan_memory(self) -> _StageExecution:
        graph_path = (
            self._stage_directory(PipelineStage.OPTIMIZE) / "tiny_transformer_block.graph.json"
        )
        directory = self._stage_directory(PipelineStage.PLAN_MEMORY)
        summary = self._run_native(
            [
                "plan-memory",
                str(graph_path),
                "--alignment",
                str(self.config.alignment_bytes),
                "--output-dir",
                str(directory),
            ]
        )
        artifacts = tuple(
            directory / name
            for name in ("schedule.json", "memory_plan.json", "timeline.csv", "timeline.svg")
        )
        return _StageExecution(
            artifacts,
            {
                "alignment_bytes": summary["alignment_bytes"],
                "planned_bytes": summary["planned_bytes"],
                "naive_allocation_bytes": summary["naive_allocation_bytes"],
            },
        )

    def _execute(self) -> _StageExecution:
        reference_directory = self._stage_directory(PipelineStage.GENERATE_REFERENCE)
        graph_path = (
            self._stage_directory(PipelineStage.OPTIMIZE) / "tiny_transformer_block.graph.json"
        )
        with np.load(reference_directory / "input_tensor.npz", allow_pickle=False) as archive:
            input_ids = torch.from_numpy(np.ascontiguousarray(archive["input_ids"]))
        model = create_deterministic_model(self.config.model)
        with torch.no_grad():
            hidden_states = model.token_embedding(input_ids).contiguous()
        with np.load(reference_directory / "weight_tensors.npz", allow_pickle=False) as archive:
            parameters = {name: np.ascontiguousarray(archive[name]) for name in archive.files}

        session = forgeir_py.load_graph(str(graph_path), "cpu")
        runtime_input = np.ascontiguousarray(hidden_states.numpy())
        forgeir_py.execute(session, {"v0000": runtime_input}, parameters)
        outputs = cast(
            dict[str, np.ndarray[Any, np.dtype[np.float32]]], forgeir_py.get_outputs(session)
        )
        trace = cast(list[dict[str, object]], forgeir_py.get_trace(session))
        deterministic_trace = [
            {key: value for key, value in item.items() if key != "elapsed_microseconds"}
            for item in trace
        ]

        directory = self._stage_directory(PipelineStage.RUN)
        directory.mkdir(parents=True, exist_ok=True)
        input_path = directory / "runtime_input.npz"
        output_path = directory / "runtime_output.npz"
        trace_path = directory / "execution_trace.json"
        write_deterministic_npz(input_path, {"v0000": runtime_input})
        write_deterministic_npz(output_path, outputs)
        graph = _load_json(graph_path)
        _write_json(
            trace_path,
            {
                "trace_schema_version": "1.0",
                "graph_hash": graph["graph_hash"],
                "records": deterministic_trace,
            },
        )
        return _StageExecution(
            (input_path, output_path, trace_path),
            {"output_ids": sorted(outputs), "operation_count": len(trace)},
        )

    def _compare(self) -> _StageExecution:
        reference_path = (
            self._stage_directory(PipelineStage.GENERATE_REFERENCE) / "expected_output.npz"
        )
        runtime_path = self._stage_directory(PipelineStage.RUN) / "runtime_output.npz"
        graph_path = (
            self._stage_directory(PipelineStage.OPTIMIZE) / "tiny_transformer_block.graph.json"
        )
        graph = _load_json(graph_path)
        output_ids = graph.get("outputs")
        if (
            not isinstance(output_ids, list)
            or len(output_ids) != 1
            or not isinstance(output_ids[0], str)
        ):
            raise _StageFailure("output_contract_invalid", "graph must declare exactly one output")
        output_id = output_ids[0]
        with np.load(reference_path, allow_pickle=False) as archive:
            expected = np.ascontiguousarray(archive["hidden_states"])
        with np.load(runtime_path, allow_pickle=False) as archive:
            actual = np.ascontiguousarray(archive[output_id])
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise _StageFailure(
                "output_contract_mismatch",
                "runtime output shape or dtype does not match the reference output",
                {
                    "actual_shape": list(actual.shape),
                    "expected_shape": list(expected.shape),
                    "actual_dtype": str(actual.dtype),
                    "expected_dtype": str(expected.dtype),
                },
            )

        finite = np.isfinite(expected)
        classification_match = bool(
            np.array_equal(np.isnan(actual), np.isnan(expected))
            and np.array_equal(np.isposinf(actual), np.isposinf(expected))
            and np.array_equal(np.isneginf(actual), np.isneginf(expected))
        )
        finite_actual = actual[finite]
        finite_expected = expected[finite]
        absolute_error = np.abs(finite_actual - finite_expected)
        tolerance = self.config.absolute_tolerance + self.config.relative_tolerance * np.abs(
            finite_expected
        )
        violation_count = int(np.count_nonzero(absolute_error > tolerance))
        relative_error = absolute_error / np.maximum(np.abs(finite_expected), np.float32(1.0e-12))
        passed = classification_match and violation_count == 0
        comparison = {
            "comparison_schema_version": "1.0",
            "output_id": output_id,
            "passed": passed,
            "absolute_tolerance": self.config.absolute_tolerance,
            "relative_tolerance": self.config.relative_tolerance,
            "max_absolute_error": float(absolute_error.max(initial=0.0)),
            "max_relative_error": float(relative_error.max(initial=0.0)),
            "finite_value_count": int(np.count_nonzero(finite)),
            "nonfinite_classification_match": classification_match,
            "violation_count": violation_count,
        }
        path = self._stage_directory(PipelineStage.COMPARE) / "comparison.json"
        _write_json(path, comparison)
        diagnostics: tuple[dict[str, object], ...] = (
            ()
            if passed
            else (
                {
                    "code": "numerical_parity_failed",
                    "message": "runtime output does not satisfy the configured parity tolerance",
                    "details": comparison,
                },
            )
        )
        return _StageExecution((path,), comparison, passed, diagnostics)

    def _report(self) -> _StageExecution:
        graph_summary = _load_json(
            self._stage_directory(PipelineStage.INSPECT) / "graph_summary.json"
        )
        memory_plan = _load_json(
            self._stage_directory(PipelineStage.PLAN_MEMORY) / "memory_plan.json"
        )
        comparison = _load_json(self._stage_directory(PipelineStage.COMPARE) / "comparison.json")
        graph = _load_json(
            self._stage_directory(PipelineStage.OPTIMIZE) / "tiny_transformer_block.graph.json"
        )
        report = {
            "report_schema_version": "1.0",
            "run_id": self.config.run_id,
            "configuration_hash": self.config.configuration_hash,
            "seed": 42,
            "success": True,
            "graph_hash": graph["graph_hash"],
            "graph_summary": graph_summary,
            "memory_summary": memory_plan["summary"],
            "numerical_comparison": comparison,
        }
        directory = self._stage_directory(PipelineStage.REPORT)
        json_path = directory / "report.json"
        markdown_path = directory / "report.md"
        _write_json(json_path, report)
        memory_summary = cast(dict[str, object], memory_plan["summary"])
        markdown_path.write_text(
            "\n".join(
                (
                    "# ForgeIR pipeline report",
                    "",
                    f"- Run ID: `{self.config.run_id}`",
                    "- Status: success",
                    "- Seed: 42",
                    f"- Graph hash: `{graph['graph_hash']}`",
                    f"- Operations: {graph_summary['operation_count']}",
                    f"- Planned arena bytes: {memory_summary['planned_bytes']}",
                    f"- Maximum absolute error: {comparison['max_absolute_error']}",
                    f"- Maximum relative error: {comparison['max_relative_error']}",
                    "",
                    "Memory values are static planner calculations. No performance claim is made.",
                    "",
                )
            ),
            encoding="utf-8",
        )
        return _StageExecution(
            (json_path, markdown_path),
            {
                "graph_hash": graph["graph_hash"],
                "max_absolute_error": comparison["max_absolute_error"],
                "max_relative_error": comparison["max_relative_error"],
            },
        )


def run_pipeline(
    config_path: Path,
    output_directory: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> PipelineResult:
    """Run the typed end-to-end workflow and return a durable structured result."""
    return PipelineRunner(
        config_path,
        output_directory,
        force=force,
        dry_run=dry_run,
    ).run()
