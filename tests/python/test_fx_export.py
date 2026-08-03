"""Contract tests for deterministic controlled PyTorch FX export."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import torch
from torch import nn

from forgeir.export.contract import canonical_json_text, verify_graph_hash
from forgeir.export.errors import (
    UnsupportedFxNodeError,
    WeightIntegrityError,
    WeightResolutionError,
)
from forgeir.export.fx_export import export_fx_graph
from forgeir.reference import TinyTransformerConfig, generate_reference_artifacts
from forgeir.reference.artifacts import file_sha256

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "config" / "graph" / "forgeir_graph.schema.json"
GOLDEN_DIRECTORY = ROOT / "tests" / "golden"


def _small_config() -> TinyTransformerConfig:
    return TinyTransformerConfig(
        vocabulary_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        sequence_length=4,
        batch_size=1,
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _export(tmp_path: Path, name: str = "export") -> tuple[dict[str, Any], Path, Path]:
    config = _small_config()
    weights = generate_reference_artifacts(tmp_path / "weights", config)
    result = export_fx_graph(
        tmp_path / name,
        weights.output_directory,
        config=config,
        weight_manifest_reference="golden/reference/manifest.sha256",
    )
    return _load_json(result.graph_path), result.graph_path, result.dot_path


def _rewrite_manifest_digest(directory: Path, filename: str) -> None:
    manifest_path = directory / "manifest.sha256"
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    replacement = f"{file_sha256(directory / filename)}  {filename}"
    rewritten = [replacement if line.endswith(f"  {filename}") else line for line in lines]
    manifest_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def test_exported_graph_validates_against_schema_and_canonical_json(tmp_path: Path) -> None:
    graph, graph_path, _ = _export(tmp_path)
    schema = _load_json(SCHEMA_PATH)
    jsonschema.Draft202012Validator(schema).validate(graph)
    verify_graph_hash(graph)
    assert graph["graph_schema_version"] == "1.0"
    assert graph_path.read_text(encoding="utf-8") == canonical_json_text(graph) + "\n"


def test_export_is_deterministic_with_stable_golden_operation_order(tmp_path: Path) -> None:
    config = _small_config()
    weights = generate_reference_artifacts(tmp_path / "weights", config)
    first = export_fx_graph(
        tmp_path / "first",
        weights.output_directory,
        config=config,
        weight_manifest_reference="golden/reference/manifest.sha256",
    )
    second = export_fx_graph(
        tmp_path / "second",
        weights.output_directory,
        config=config,
        weight_manifest_reference="golden/reference/manifest.sha256",
    )
    assert first.graph_hash == second.graph_hash
    assert first.graph_path.read_bytes() == second.graph_path.read_bytes()
    assert first.dot_path.read_bytes() == second.dot_path.read_bytes()

    golden_order = _load_json(GOLDEN_DIRECTORY / "tiny_transformer_block_v1.operation_order.json")
    operations = first.graph["operations"]
    operation_types = [operation["type"] for operation in operations]
    assert operation_types == golden_order["operation_types"]
    assert [operation["id"] for operation in operations] == [
        f"op{index:04d}" for index in range(len(operations))
    ]
    golden_counts = _load_json(GOLDEN_DIRECTORY / "tiny_transformer_block_v1.operation_counts.json")
    assert dict(sorted(Counter(operation_types).items())) == golden_counts


def test_unsupported_fx_node_has_structured_error(tmp_path: Path) -> None:
    class UnsupportedBlock(nn.Module):
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return torch.sin(hidden_states)

    config = _small_config()
    weights = generate_reference_artifacts(tmp_path / "weights", config)
    with pytest.raises(UnsupportedFxNodeError) as caught:
        export_fx_graph(
            tmp_path / "unsupported",
            weights.output_directory,
            config=config,
            weight_manifest_reference="golden/reference/manifest.sha256",
            block=UnsupportedBlock(),
        )
    details = caught.value.as_dict()
    assert details["node_name"] == "sin"
    assert str(details["fx_target"]).endswith(".sin")
    assert details["arguments"] == {
        "args": [{"node": "hidden_states"}],
        "kwargs": {},
    }
    assert "canonical call_function lowering" in str(details["reason"])


def test_missing_weight_fails_without_skipping_parameter(tmp_path: Path) -> None:
    config = _small_config()
    weights = generate_reference_artifacts(tmp_path / "weights", config)
    metadata = _load_json(weights.tensor_metadata)
    missing_name = "model.parameter.block.attention.query_projection.weight"
    metadata["tensors"] = [
        tensor for tensor in metadata["tensors"] if tensor["semantic_name"] != missing_name
    ]
    weights.tensor_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rewrite_manifest_digest(weights.output_directory, weights.tensor_metadata.name)
    with pytest.raises(WeightResolutionError, match="missing weight metadata"):
        export_fx_graph(
            tmp_path / "missing",
            weights.output_directory,
            config=config,
            weight_manifest_reference="golden/reference/manifest.sha256",
        )


def test_tampered_weight_archive_hash_fails(tmp_path: Path) -> None:
    config = _small_config()
    weights = generate_reference_artifacts(tmp_path / "weights", config)
    with weights.weight_tensors.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(WeightIntegrityError, match="SHA-256 mismatch"):
        export_fx_graph(
            tmp_path / "tampered",
            weights.output_directory,
            config=config,
            weight_manifest_reference="golden/reference/manifest.sha256",
        )


def test_graph_hash_verification_rejects_modified_graph(tmp_path: Path) -> None:
    graph, _, _ = _export(tmp_path)
    graph["producer_version"] = "tampered"
    with pytest.raises(ValueError, match="graph hash mismatch"):
        verify_graph_hash(graph)


def test_graph_input_output_value_and_weight_contract(tmp_path: Path) -> None:
    graph, _, dot_path = _export(tmp_path)
    values = {value["id"]: value for value in graph["values"]}
    assert len(graph["inputs"]) == 1
    assert len(graph["outputs"]) == 1
    input_value = values[graph["inputs"][0]]
    output_value = values[graph["outputs"][0]]
    assert input_value == {
        "id": "v0000",
        "semantic_name": "block.input.hidden_states",
        "shape": [1, 4, 8],
        "dtype": "float32",
        "kind": "input",
    }
    assert output_value["semantic_name"] == "block.output.hidden_states"
    assert output_value["shape"] == [1, 4, 8]
    assert output_value["dtype"] == "float32"
    assert output_value["kind"] == "output"

    value_ids = set(values)
    for operation in graph["operations"]:
        assert set(operation["inputs"]).issubset(value_ids)
        assert set(operation["outputs"]).issubset(value_ids)
    parameters = [
        operation for operation in graph["operations"] if operation["type"] == "Parameter"
    ]
    assert len(parameters) == 8
    for parameter in parameters:
        assert parameter["attributes"]["archive"] == "weight_tensors.npz"
        assert len(parameter["attributes"]["content_sha256"]) == 64
    assert graph["weight_manifest"]["reference"] == "golden/reference/manifest.sha256"
    assert len(graph["weight_manifest"]["weight_archive_sha256"]) == 64
    assert dot_path.read_text(encoding="utf-8").startswith("digraph ForgeIR {\n")
