"""Typed public contracts for the deterministic end-to-end pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from forgeir.reference import TinyTransformerConfig

PIPELINE_SCHEMA_VERSION = "1.0"


class PipelineStage(StrEnum):
    """Stable ordered pipeline stages."""

    GENERATE_REFERENCE = "generate-reference"
    EXPORT = "export"
    INSPECT = "inspect"
    VERIFY = "verify"
    OPTIMIZE = "optimize"
    PLAN_MEMORY = "plan-memory"
    RUN = "run"
    COMPARE = "compare"
    REPORT = "report"


PIPELINE_STAGES: tuple[PipelineStage, ...] = tuple(PipelineStage)


class StageStatus(StrEnum):
    """Machine-readable stage state."""

    PLANNED = "planned"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Immutable validated pipeline configuration."""

    model: TinyTransformerConfig
    optimization_level: str = "O2"
    alignment_bytes: int = 64
    absolute_tolerance: float = 2.0e-6
    relative_tolerance: float = 2.0e-5

    def __post_init__(self) -> None:
        if self.model.seed != 42:
            raise ValueError("the end-to-end pipeline requires deterministic seed 42")
        if self.optimization_level not in {"O0", "O1", "O2"}:
            raise ValueError("optimization_level must be O0, O1, or O2")
        if (
            isinstance(self.alignment_bytes, bool)
            or not isinstance(self.alignment_bytes, int)
            or self.alignment_bytes <= 0
            or self.alignment_bytes & (self.alignment_bytes - 1)
        ):
            raise ValueError("alignment_bytes must be a positive power of two")
        for name, value in (
            ("absolute_tolerance", self.absolute_tolerance),
            ("relative_tolerance", self.relative_tolerance),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @classmethod
    def load(cls, path: Path) -> PipelineConfig:
        """Load and strictly validate a pipeline JSON document."""
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(f"unable to read pipeline config {path}: {error}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"pipeline config is not valid JSON: {error}") from error
        if not isinstance(document, dict):
            raise ValueError("pipeline config must be a JSON object")
        expected_keys = {
            "pipeline_schema_version",
            "model",
            "optimization_level",
            "alignment_bytes",
            "absolute_tolerance",
            "relative_tolerance",
        }
        if set(document) != expected_keys:
            missing = sorted(expected_keys - set(document))
            extra = sorted(set(document) - expected_keys)
            raise ValueError(f"pipeline config fields mismatch; missing={missing}, extra={extra}")
        if document["pipeline_schema_version"] != PIPELINE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported pipeline schema version {document['pipeline_schema_version']!r}"
            )
        model_document = document["model"]
        if not isinstance(model_document, dict):
            raise ValueError("pipeline config model must be an object")
        model_keys = set(TinyTransformerConfig().as_dict())
        if set(model_document) != model_keys:
            missing = sorted(model_keys - set(model_document))
            extra = sorted(set(model_document) - model_keys)
            raise ValueError(f"model config fields mismatch; missing={missing}, extra={extra}")
        model = TinyTransformerConfig(**cast(dict[str, Any], model_document))
        return cls(
            model=model,
            optimization_level=cast(str, document["optimization_level"]),
            alignment_bytes=cast(int, document["alignment_bytes"]),
            absolute_tolerance=cast(float, document["absolute_tolerance"]),
            relative_tolerance=cast(float, document["relative_tolerance"]),
        )

    def as_dict(self) -> dict[str, object]:
        """Return the canonical JSON-compatible configuration."""
        return {
            "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
            "model": self.model.as_dict(),
            "optimization_level": self.optimization_level,
            "alignment_bytes": self.alignment_bytes,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
        }

    @property
    def canonical_json(self) -> str:
        """Return deterministic compact JSON used for identity hashing."""
        return json.dumps(self.as_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @property
    def configuration_hash(self) -> str:
        """Return SHA-256 of the canonical pipeline configuration."""
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def run_id(self) -> str:
        """Return the stable config-derived isolated run-directory name."""
        return f"run-{self.configuration_hash[:16]}"


@dataclass(frozen=True, slots=True)
class ArtifactDigest:
    """One run-relative artifact identity."""

    path: str
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True, slots=True)
class StageResult:
    """Typed result for one pipeline stage."""

    stage: PipelineStage
    status: StageStatus
    input_hashes: dict[str, str] = field(default_factory=dict)
    artifacts: tuple[ArtifactDigest, ...] = ()
    diagnostics: tuple[dict[str, object], ...] = ()
    summary: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "input_hashes": dict(sorted(self.input_hashes.items())),
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
            "diagnostics": list(self.diagnostics),
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Complete typed outcome for a pipeline or dry run."""

    success: bool
    dry_run: bool
    run_id: str
    run_directory: Path
    stages: tuple[StageResult, ...]
    manifest_path: Path | None = None
    status_path: Path | None = None
    failure_stage: PipelineStage | None = None
    diagnostics: tuple[dict[str, object], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
            "success": self.success,
            "dry_run": self.dry_run,
            "run_id": self.run_id,
            "run_directory": str(self.run_directory),
            "manifest_path": str(self.manifest_path) if self.manifest_path is not None else None,
            "status_path": str(self.status_path) if self.status_path is not None else None,
            "failure_stage": self.failure_stage.value if self.failure_stage is not None else None,
            "diagnostics": list(self.diagnostics),
            "stages": [stage.as_dict() for stage in self.stages],
        }
