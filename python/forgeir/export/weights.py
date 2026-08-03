"""Resolution of graph parameters against Milestone 2 reference artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from forgeir.export.contract import canonical_json_sha256
from forgeir.export.errors import WeightIntegrityError, WeightResolutionError
from forgeir.reference.artifacts import (
    array_content_sha256,
    file_sha256,
    verify_manifest,
)
from forgeir.reference.config import TinyTransformerConfig


@dataclass(frozen=True, slots=True)
class WeightReference:
    """Validated external parameter reference used by the graph contract."""

    semantic_name: str
    archive: str
    archive_key: str
    shape: tuple[int, ...]
    dtype: str
    content_sha256: str


class WeightManifest:
    """Validated view of one Milestone 2 artifact directory."""

    def __init__(
        self,
        directory: Path,
        config: TinyTransformerConfig,
        logical_reference: str,
    ) -> None:
        self.directory = directory.resolve()
        self.logical_reference = logical_reference.replace("\\", "/")
        if not self.logical_reference:
            raise WeightResolutionError("weight manifest reference must not be empty")
        try:
            verify_manifest(self.directory)
        except ValueError as error:
            raise WeightIntegrityError(str(error)) from error

        self.manifest_path = self.directory / "manifest.sha256"
        self.archive_path = self.directory / "weight_tensors.npz"
        self._manifest_entries = self._read_manifest_entries()
        self._metadata = self._read_tensor_metadata()
        self._validate_configuration(config)

    @property
    def configuration_hash(self) -> str:
        configuration_path = self.directory / "configuration.json"
        document = json.loads(configuration_path.read_text(encoding="utf-8"))
        return canonical_json_sha256(document["configuration"])

    def _read_manifest_entries(self) -> dict[str, str]:
        entries: dict[str, str] = {}
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            digest, filename = line.split("  ", maxsplit=1)
            entries[filename] = digest
        return entries

    def _read_tensor_metadata(self) -> dict[str, dict[str, object]]:
        metadata_path = self.directory / "tensor_metadata.json"
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
        tensors = document.get("tensors")
        if not isinstance(tensors, list):
            raise WeightIntegrityError("tensor_metadata.json has no tensor list")
        by_semantic_name: dict[str, dict[str, object]] = {}
        for entry in tensors:
            if not isinstance(entry, dict):
                raise WeightIntegrityError("tensor metadata entry must be an object")
            semantic_name = entry.get("semantic_name")
            if not isinstance(semantic_name, str):
                raise WeightIntegrityError("tensor metadata semantic_name must be a string")
            if semantic_name in by_semantic_name:
                raise WeightIntegrityError(f"duplicate tensor metadata: {semantic_name}")
            by_semantic_name[semantic_name] = cast(dict[str, object], entry)
        return by_semantic_name

    def _validate_configuration(self, config: TinyTransformerConfig) -> None:
        configuration_path = self.directory / "configuration.json"
        document = json.loads(configuration_path.read_text(encoding="utf-8"))
        if document.get("configuration") != config.as_dict():
            raise WeightResolutionError(
                "reference artifact configuration does not match the exported model configuration"
            )

    def resolve(self, parameter_name: str) -> WeightReference:
        semantic_name = f"model.parameter.{parameter_name}"
        entry = self._metadata.get(semantic_name)
        if entry is None:
            raise WeightResolutionError(f"missing weight metadata for {semantic_name}")

        artifact = entry.get("artifact")
        archive_key = entry.get("archive_key")
        shape = entry.get("shape")
        dtype = entry.get("dtype")
        content_sha256 = entry.get("content_sha256")
        if artifact != "weight_tensors.npz":
            raise WeightResolutionError(f"{semantic_name} does not reference weight_tensors.npz")
        if not isinstance(archive_key, str):
            raise WeightIntegrityError(f"invalid archive key for {semantic_name}")
        if not isinstance(shape, list) or not all(isinstance(value, int) for value in shape):
            raise WeightIntegrityError(f"invalid shape for {semantic_name}")
        if not isinstance(dtype, str) or not isinstance(content_sha256, str):
            raise WeightIntegrityError(f"invalid dtype or content hash for {semantic_name}")

        archive = cast(Any, np.load)(self.archive_path, allow_pickle=False)
        try:
            if archive_key not in archive.files:
                raise WeightResolutionError(
                    f"missing archive key {archive_key!r} for {semantic_name}"
                )
            array = archive[archive_key]
        finally:
            archive.close()
        resolved_shape = tuple(int(value) for value in array.shape)
        if resolved_shape != tuple(shape) or str(array.dtype) != dtype:
            raise WeightIntegrityError(f"shape or dtype mismatch for {semantic_name}")
        if array_content_sha256(array) != content_sha256:
            raise WeightIntegrityError(f"content hash mismatch for {semantic_name}")
        return WeightReference(
            semantic_name=semantic_name,
            archive="weight_tensors.npz",
            archive_key=archive_key,
            shape=resolved_shape,
            dtype=dtype,
            content_sha256=content_sha256,
        )

    def contract_reference(self) -> dict[str, object]:
        return {
            "reference": self.logical_reference,
            "sha256": file_sha256(self.manifest_path),
            "weight_archive": "weight_tensors.npz",
            "weight_archive_sha256": self._manifest_entries["weight_tensors.npz"],
        }
