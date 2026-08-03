"""Deterministic PyTorch reference model and artifacts."""

from forgeir.reference.artifacts import (
    ArtifactPaths,
    array_content_sha256,
    generate_reference_artifacts,
    verify_manifest,
)
from forgeir.reference.config import TinyTransformerConfig
from forgeir.reference.model import (
    CausalSelfAttention,
    RMSNorm,
    TinyTransformerBlock,
    TinyTransformerModel,
    create_deterministic_input,
    create_deterministic_model,
)

__all__ = [
    "ArtifactPaths",
    "CausalSelfAttention",
    "RMSNorm",
    "TinyTransformerBlock",
    "TinyTransformerConfig",
    "TinyTransformerModel",
    "array_content_sha256",
    "create_deterministic_input",
    "create_deterministic_model",
    "generate_reference_artifacts",
    "verify_manifest",
]
