"""Tests for deterministic reference artifact serialization and verification."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forgeir_py
from forgeir.reference import (
    TinyTransformerConfig,
    array_content_sha256,
    generate_reference_artifacts,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parents[2]


def _artifact_config() -> TinyTransformerConfig:
    return TinyTransformerConfig(
        vocabulary_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        sequence_length=4,
        batch_size=1,
    )


def test_artifacts_are_repeatable_and_tensor_hashes_match(tmp_path: Path) -> None:
    first = generate_reference_artifacts(tmp_path / "first", _artifact_config())
    second = generate_reference_artifacts(tmp_path / "second", _artifact_config())
    assert first.manifest.read_bytes() == second.manifest.read_bytes()
    verify_manifest(first.output_directory)
    verify_manifest(second.output_directory)

    metadata = json.loads(first.tensor_metadata.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["hash_algorithm"] == "sha256"
    assert metadata["tensors"]
    for tensor in metadata["tensors"]:
        assert set(tensor) == {
            "artifact",
            "archive_key",
            "semantic_name",
            "shape",
            "dtype",
            "content_sha256",
        }
        archive_path = first.output_directory / tensor["artifact"]
        with np.load(archive_path, allow_pickle=False) as archive:
            array = archive[tensor["archive_key"]]
        assert list(array.shape) == tensor["shape"]
        assert str(array.dtype) == tensor["dtype"]
        assert tensor["semantic_name"]
        assert array_content_sha256(array) == tensor["content_sha256"]


def test_manifest_verification_rejects_modified_artifact(tmp_path: Path) -> None:
    paths = generate_reference_artifacts(tmp_path / "tampered", _artifact_config())
    paths.configuration.write_text(
        paths.configuration.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_manifest(paths.output_directory)


def test_reference_generation_module_cli(tmp_path: Path) -> None:
    output_directory = tmp_path / "cli"
    environment = os.environ.copy()
    python_paths = [str(ROOT / "python"), str(Path(forgeir_py.__file__).resolve().parent)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "forgeir.reference.generate",
            "--output-dir",
            str(output_directory),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    reported_paths = json.loads(result.stdout)
    assert Path(reported_paths["output_directory"]) == output_directory.resolve()
    verify_manifest(output_directory)
