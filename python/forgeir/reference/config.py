"""Validated configuration for the deterministic reference transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch

ConfigValue = int | float | str


@dataclass(frozen=True, slots=True)
class TinyTransformerConfig:
    """Immutable dimensions and numerical settings for the reference model."""

    vocabulary_size: int = 256
    hidden_size: int = 128
    intermediate_size: int = 384
    num_heads: int = 4
    sequence_length: int = 32
    batch_size: int = 2
    epsilon: float = 1e-5
    dtype: str = "float32"
    seed: int = 42

    SUPPORTED_DTYPES: ClassVar[frozenset[str]] = frozenset({"float32"})
    DIMENSION_LIMITS: ClassVar[dict[str, int]] = {
        "vocabulary_size": 65_536,
        "hidden_size": 4_096,
        "intermediate_size": 16_384,
        "num_heads": 128,
        "sequence_length": 4_096,
        "batch_size": 256,
    }
    MAX_PARAMETER_ELEMENTS: ClassVar[int] = 100_000_000
    MAX_ACTIVATION_ELEMENTS: ClassVar[int] = 100_000_000
    MAX_SEED: ClassVar[int] = (1 << 63) - 1

    def __post_init__(self) -> None:
        for name, upper_bound in self.DIMENSION_LIMITS.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            if value > upper_bound:
                raise ValueError(f"{name} exceeds the safe upper bound of {upper_bound}")

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not isinstance(self.dtype, str):
            raise TypeError("dtype must be a string")
        if self.dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(sorted(self.SUPPORTED_DTYPES))
            raise ValueError(f"unsupported dtype {self.dtype!r}; supported dtypes: {supported}")
        if isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (int, float)):
            raise TypeError("epsilon must be a real number")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.seed < 0 or self.seed > self.MAX_SEED:
            raise ValueError(f"seed must be between 0 and {self.MAX_SEED}")

        parameter_elements = (
            self.vocabulary_size * self.hidden_size
            + 4 * self.hidden_size * self.hidden_size
            + 2 * self.hidden_size * self.intermediate_size
            + 2 * self.hidden_size
        )
        if parameter_elements > self.MAX_PARAMETER_ELEMENTS:
            raise ValueError(
                "configuration exceeds the safe parameter element budget of "
                f"{self.MAX_PARAMETER_ELEMENTS}"
            )

        hidden_elements = self.batch_size * self.sequence_length * self.hidden_size
        attention_elements = (
            self.batch_size
            * self.num_heads
            * self.sequence_length
            * self.sequence_length
        )
        activation_elements = 8 * hidden_elements + 2 * attention_elements
        if activation_elements > self.MAX_ACTIVATION_ELEMENTS:
            raise ValueError(
                "configuration exceeds the safe activation element budget of "
                f"{self.MAX_ACTIVATION_ELEMENTS}"
            )

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return the validated PyTorch dtype."""
        if self.dtype == "float32":
            return torch.float32
        raise ValueError(f"unsupported dtype {self.dtype!r}")

    @property
    def head_size(self) -> int:
        """Return the per-head hidden width."""
        return self.hidden_size // self.num_heads

    def as_dict(self) -> dict[str, ConfigValue]:
        """Return a JSON-compatible representation with a stable field order."""
        return {
            "vocabulary_size": self.vocabulary_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_heads": self.num_heads,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "epsilon": self.epsilon,
            "dtype": self.dtype,
            "seed": self.seed,
        }
