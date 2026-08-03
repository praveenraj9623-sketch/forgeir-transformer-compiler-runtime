"""Structured failures for controlled FX graph export."""

from __future__ import annotations

from dataclasses import dataclass


class FxExportError(RuntimeError):
    """Base class for deterministic graph export failures."""


@dataclass(frozen=True, slots=True)
class UnsupportedNodeDetails:
    """Machine-readable context for one unsupported FX node."""

    node_name: str
    fx_target: str
    arguments: object
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "node_name": self.node_name,
            "fx_target": self.fx_target,
            "arguments": self.arguments,
            "reason": self.reason,
        }


class UnsupportedFxNodeError(FxExportError):
    """Raised when an FX node has no explicit ForgeIR lowering."""

    def __init__(self, details: UnsupportedNodeDetails) -> None:
        self.details = details
        super().__init__(
            f"unsupported FX node {details.node_name!r} targeting "
            f"{details.fx_target!r}: {details.reason}"
        )

    def as_dict(self) -> dict[str, object]:
        return self.details.as_dict()


class WeightResolutionError(FxExportError):
    """Raised when a graph parameter cannot resolve to reference weights."""


class WeightIntegrityError(FxExportError):
    """Raised when reference weight artifacts fail integrity validation."""
