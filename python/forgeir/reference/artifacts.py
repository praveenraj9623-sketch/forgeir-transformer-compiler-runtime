"""Deterministic reference artifact serialization and verification."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from forgeir.reference.config import TinyTransformerConfig
from forgeir.reference.model import create_deterministic_input, create_deterministic_model

Array = NDArray[np.generic]
MANIFEST_PAYLOAD_FILES = (
    "configuration.json",
    "expected_output.npz",
    "input_tensor.npz",
    "tensor_metadata.json",
    "weight_tensors.npz",
)


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """Paths produced by one reference generation run."""

    output_directory: Path
    configuration: Path
    input_tensor: Path
    weight_tensors: Path
    expected_output: Path
    tensor_metadata: Path
    manifest: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "output_directory": str(self.output_directory),
            "configuration": str(self.configuration),
            "input_tensor": str(self.input_tensor),
            "weight_tensors": str(self.weight_tensors),
            "expected_output": str(self.expected_output),
            "tensor_metadata": str(self.tensor_metadata),
            "manifest": str(self.manifest),
        }


@dataclass(frozen=True, slots=True)
class TensorMetadata:
    """Portable identity and layout information for one serialized tensor."""

    artifact: str
    archive_key: str
    semantic_name: str
    shape: tuple[int, ...]
    dtype: str
    content_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact": self.artifact,
            "archive_key": self.archive_key,
            "semantic_name": self.semantic_name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "content_sha256": self.content_sha256,
        }


def array_content_sha256(array: Array) -> str:
    """Hash contiguous tensor content bytes without archive-specific metadata."""
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_to_numpy(tensor: Tensor) -> Array:
    return tensor.detach().cpu().contiguous().numpy()


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_deterministic_npz(path: Path, tensors: Mapping[str, Array]) -> None:
    """Write an NPZ archive with stable member order and fixed ZIP metadata."""
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in tensors.items():
            member = io.BytesIO()
            write_array = cast(Callable[..., None], cast(Any, np.lib.format.write_array))
            write_array(member, np.ascontiguousarray(array), allow_pickle=False)
            information = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            information.compress_type = zipfile.ZIP_STORED
            information.create_system = 3
            information.external_attr = 0o600 << 16
            archive.writestr(information, member.getvalue())


def _metadata(
    *, artifact: str, archive_key: str, semantic_name: str, array: Array
) -> TensorMetadata:
    return TensorMetadata(
        artifact=artifact,
        archive_key=archive_key,
        semantic_name=semantic_name,
        shape=tuple(int(dimension) for dimension in array.shape),
        dtype=str(array.dtype),
        content_sha256=array_content_sha256(array),
    )


def generate_reference_artifacts(
    output_directory: Path,
    config: TinyTransformerConfig | None = None,
) -> ArtifactPaths:
    """Generate a complete deterministic artifact set on CPU."""
    resolved_config = config or TinyTransformerConfig()
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    paths = ArtifactPaths(
        output_directory=output_directory,
        configuration=output_directory / "configuration.json",
        input_tensor=output_directory / "input_tensor.npz",
        weight_tensors=output_directory / "weight_tensors.npz",
        expected_output=output_directory / "expected_output.npz",
        tensor_metadata=output_directory / "tensor_metadata.json",
        manifest=output_directory / "manifest.sha256",
    )

    _write_json(
        paths.configuration,
        {
            "schema_version": 1,
            "semantic_name": "forgeir.reference.tiny_transformer_config",
            "configuration": resolved_config.as_dict(),
        },
    )

    model = create_deterministic_model(resolved_config)
    input_ids = create_deterministic_input(resolved_config)
    with torch.inference_mode():
        expected_output = model(input_ids)

    input_array = _tensor_to_numpy(input_ids)
    output_array = _tensor_to_numpy(expected_output)
    input_tensors: dict[str, Array] = {"input_ids": input_array}
    output_tensors: dict[str, Array] = {"hidden_states": output_array}
    write_deterministic_npz(paths.input_tensor, input_tensors)
    write_deterministic_npz(paths.expected_output, output_tensors)

    metadata_entries = [
        _metadata(
            artifact=paths.input_tensor.name,
            archive_key="input_ids",
            semantic_name="model.input_ids",
            array=input_array,
        ),
        _metadata(
            artifact=paths.expected_output.name,
            archive_key="hidden_states",
            semantic_name="model.expected_output.hidden_states",
            array=output_array,
        ),
    ]

    weight_tensors: dict[str, Array] = {}
    for parameter_name, parameter in sorted(model.state_dict().items()):
        archive_key = parameter_name.replace(".", "__")
        weight_array = _tensor_to_numpy(parameter)
        weight_tensors[archive_key] = weight_array
        metadata_entries.append(
            _metadata(
                artifact=paths.weight_tensors.name,
                archive_key=archive_key,
                semantic_name=f"model.parameter.{parameter_name}",
                array=weight_array,
            )
        )
    write_deterministic_npz(paths.weight_tensors, weight_tensors)

    _write_json(
        paths.tensor_metadata,
        {
            "schema_version": 1,
            "hash_algorithm": "sha256",
            "tensors": [entry.as_dict() for entry in metadata_entries],
        },
    )

    manifest_lines = [
        f"{file_sha256(output_directory / filename)}  {filename}\n"
        for filename in MANIFEST_PAYLOAD_FILES
    ]
    paths.manifest.write_text("".join(manifest_lines), encoding="utf-8")
    verify_manifest(output_directory)
    return paths


def verify_manifest(output_directory: Path) -> None:
    """Raise ValueError unless every expected artifact matches its manifest digest."""
    output_directory = output_directory.resolve()
    manifest_path = output_directory / "manifest.sha256"
    if not manifest_path.is_file():
        raise ValueError(f"missing manifest: {manifest_path}")

    recorded: dict[str, str] = {}
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"invalid manifest line {line_number}")
        digest, filename = parts
        if filename in recorded:
            raise ValueError(f"duplicate manifest entry: {filename}")
        recorded[filename] = digest

    expected = set(MANIFEST_PAYLOAD_FILES)
    if set(recorded) != expected:
        raise ValueError("manifest file set does not match the reference artifact schema")
    for filename in MANIFEST_PAYLOAD_FILES:
        artifact_path = output_directory / filename
        if not artifact_path.is_file():
            raise ValueError(f"missing artifact: {filename}")
        actual_digest = file_sha256(artifact_path)
        if not hmac.compare_digest(actual_digest, recorded[filename]):
            raise ValueError(f"SHA-256 mismatch for {filename}")
