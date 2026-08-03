"""Validated public contracts for reproducible CPU benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from forgeir.reference import TinyTransformerConfig

BENCHMARK_SCHEMA_VERSION = "1.0"
REQUIRED_OPTIMIZATION_LEVELS = ("O0", "O1", "O2")


def _bounded_integer(name: str, value: object, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Immutable benchmark protocol configuration."""

    name: str
    model: TinyTransformerConfig
    warmup_count: int
    measured_iteration_count: int
    pytorch_operation_profile_iterations: int
    process_count: int
    torch_num_threads: int
    gross_failure_max_iteration_seconds: float
    absolute_tolerance: float
    relative_tolerance: float
    optimization_levels: tuple[str, ...] = REQUIRED_OPTIMIZATION_LEVELS

    def __post_init__(self) -> None:
        if not self.name or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in self.name
        ):
            raise ValueError("benchmark name must use lowercase letters, digits, '-' or '_'")
        if self.model.seed != 42:
            raise ValueError("CPU benchmarks require deterministic seed 42")
        _bounded_integer("warmup_count", self.warmup_count, minimum=0, maximum=10000)
        _bounded_integer(
            "measured_iteration_count",
            self.measured_iteration_count,
            minimum=1,
            maximum=100000,
        )
        _bounded_integer(
            "pytorch_operation_profile_iterations",
            self.pytorch_operation_profile_iterations,
            minimum=0,
            maximum=1000,
        )
        if self.pytorch_operation_profile_iterations > self.measured_iteration_count:
            raise ValueError(
                "pytorch_operation_profile_iterations may not exceed measured_iteration_count"
            )
        if self.process_count != 1:
            raise ValueError("Milestone 10 benchmarks require exactly one process")
        _bounded_integer("torch_num_threads", self.torch_num_threads, minimum=1, maximum=256)
        _positive_finite(
            "gross_failure_max_iteration_seconds", self.gross_failure_max_iteration_seconds
        )
        for name, value in (
            ("absolute_tolerance", self.absolute_tolerance),
            ("relative_tolerance", self.relative_tolerance),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.optimization_levels != REQUIRED_OPTIMIZATION_LEVELS:
            raise ValueError("optimization_levels must be exactly ['O0', 'O1', 'O2']")

    @classmethod
    def load(cls, path: Path) -> BenchmarkConfig:
        """Load a strict benchmark configuration document."""
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(f"unable to read benchmark config {path}: {error}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"benchmark config is not valid JSON: {error}") from error
        if not isinstance(document, dict):
            raise ValueError("benchmark config must be a JSON object")
        expected_keys = {
            "benchmark_schema_version",
            "name",
            "model",
            "warmup_count",
            "measured_iteration_count",
            "pytorch_operation_profile_iterations",
            "process_count",
            "torch_num_threads",
            "gross_failure_max_iteration_seconds",
            "absolute_tolerance",
            "relative_tolerance",
            "optimization_levels",
        }
        if set(document) != expected_keys:
            missing = sorted(expected_keys - set(document))
            extra = sorted(set(document) - expected_keys)
            raise ValueError(f"benchmark config fields mismatch; missing={missing}, extra={extra}")
        if document["benchmark_schema_version"] != BENCHMARK_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported benchmark schema version {document['benchmark_schema_version']!r}"
            )
        model_document = document["model"]
        if not isinstance(model_document, dict):
            raise ValueError("benchmark model config must be an object")
        model_keys = set(TinyTransformerConfig().as_dict())
        if set(model_document) != model_keys:
            missing = sorted(model_keys - set(model_document))
            extra = sorted(set(model_document) - model_keys)
            raise ValueError(f"model config fields mismatch; missing={missing}, extra={extra}")
        levels = document["optimization_levels"]
        if not isinstance(levels, list) or not all(isinstance(level, str) for level in levels):
            raise TypeError("optimization_levels must be an array of strings")
        return cls(
            name=cast(str, document["name"]),
            model=TinyTransformerConfig(**cast(dict[str, Any], model_document)),
            warmup_count=cast(int, document["warmup_count"]),
            measured_iteration_count=cast(int, document["measured_iteration_count"]),
            pytorch_operation_profile_iterations=cast(
                int, document["pytorch_operation_profile_iterations"]
            ),
            process_count=cast(int, document["process_count"]),
            torch_num_threads=cast(int, document["torch_num_threads"]),
            gross_failure_max_iteration_seconds=cast(
                float, document["gross_failure_max_iteration_seconds"]
            ),
            absolute_tolerance=cast(float, document["absolute_tolerance"]),
            relative_tolerance=cast(float, document["relative_tolerance"]),
            optimization_levels=tuple(cast(list[str], levels)),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
            "name": self.name,
            "model": self.model.as_dict(),
            "warmup_count": self.warmup_count,
            "measured_iteration_count": self.measured_iteration_count,
            "pytorch_operation_profile_iterations": self.pytorch_operation_profile_iterations,
            "process_count": self.process_count,
            "torch_num_threads": self.torch_num_threads,
            "gross_failure_max_iteration_seconds": self.gross_failure_max_iteration_seconds,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "optimization_levels": list(self.optimization_levels),
        }

    @property
    def configuration_hash(self) -> str:
        canonical = json.dumps(
            self.as_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BenchmarkArtifacts:
    """Paths and measured outcome produced by one benchmark invocation."""

    success: bool
    output_directory: Path
    measured_json: Path
    summary_csv: Path
    report_html: Path
    manifest: Path
    conclusions: tuple[str, ...]
