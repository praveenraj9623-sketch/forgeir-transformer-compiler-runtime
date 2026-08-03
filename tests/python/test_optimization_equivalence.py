"""End-to-end semantic-equivalence tests for Milestone 6 optimization."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, cast

import jsonschema
import numpy as np
import torch

from forgeir.export.contract import compute_graph_hash, write_canonical_json
from forgeir.reference import (
    TinyTransformerConfig,
    create_deterministic_input,
    create_deterministic_model,
    evaluate_graph,
)

ROOT = Path(__file__).resolve().parents[2]
REAL_GRAPH = (
    ROOT / "artifacts" / "graphs" / "milestone_03" / "default" / "tiny_transformer_block.graph.json"
)
REAL_WEIGHTS = (
    ROOT / "artifacts" / "references" / "milestone_02" / "default_run_1" / "weight_tensors.npz"
)
GRAPH_SCHEMA = ROOT / "config" / "graph" / "forgeir_graph.schema.json"


def _cli() -> Path:
    candidates = (
        ROOT / "build" / "windows-msvc-debug" / "bin" / "forgeir_cli.exe",
        ROOT / "build" / "linux-gcc-debug" / "bin" / "forgeir_cli",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("forgeir_cli debug executable must be built before Python integration tests")


def _optimize(input_path: Path, output_path: Path, level: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(_cli()),
            "optimize",
            str(input_path),
            "--level",
            level,
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert isinstance(report, dict)
    return cast(dict[str, Any], report)


def _validate_schema(graph_path: Path) -> dict[str, Any]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    schema = json.loads(GRAPH_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(graph)
    assert isinstance(graph, dict)
    return cast(dict[str, Any], graph)


def test_real_transformer_o0_o1_o2_match_reference_evaluator(tmp_path: Path) -> None:
    config = TinyTransformerConfig()
    model = create_deterministic_model(config)
    input_ids = create_deterministic_input(config)
    with torch.no_grad():
        hidden_states = model.token_embedding(input_ids)
        expected = model.block(hidden_states)

    baseline = evaluate_graph(REAL_GRAPH, {"v0000": hidden_states}, REAL_WEIGHTS)["v0034"]
    torch.testing.assert_close(baseline, expected, rtol=1.0e-6, atol=1.0e-6)

    for level in ("O0", "O1", "O2"):
        optimized_path = tmp_path / level / "tiny_transformer_block.graph.json"
        optimized_path.parent.mkdir(parents=True)
        report = _optimize(REAL_GRAPH, optimized_path, level)
        _validate_schema(optimized_path)
        actual = evaluate_graph(optimized_path, {"v0000": hidden_states}, REAL_WEIGHTS)["v0034"]
        torch.testing.assert_close(actual, baseline, rtol=0.0, atol=0.0)
        assert report["operation_counts"] == {"before": 35, "after": 35, "change": 0}


def _fusion_graph(operation_type: str) -> dict[str, Any]:
    zero_hash = "0" * 64
    graph: dict[str, Any] = {
        "graph_schema_version": "1.0",
        "producer_version": "0.1.0",
        "model_configuration_hash": zero_hash,
        "inputs": ["v0000"],
        "outputs": ["v0005"],
        "values": [
            {
                "id": "v0000",
                "semantic_name": "input",
                "shape": [1, 2],
                "dtype": "float32",
                "kind": "input",
            },
            {
                "id": "v0001",
                "semantic_name": "weight",
                "shape": [2, 2],
                "dtype": "float32",
                "kind": "parameter",
            },
            {
                "id": "v0002",
                "semantic_name": "bias",
                "shape": [2],
                "dtype": "float32",
                "kind": "parameter",
            },
            {
                "id": "v0003",
                "semantic_name": "projection",
                "shape": [1, 2],
                "dtype": "float32",
                "kind": "intermediate",
            },
            {
                "id": "v0004",
                "semantic_name": "biased",
                "shape": [1, 2],
                "dtype": "float32",
                "kind": "intermediate",
            },
            {
                "id": "v0005",
                "semantic_name": "gelu_output",
                "shape": [1, 2],
                "dtype": "float32",
                "kind": "output",
            },
        ],
        "operations": [
            {
                "id": "op0000",
                "type": "Input",
                "semantic_name": "input",
                "inputs": [],
                "outputs": ["v0000"],
                "attributes": {},
            },
            {
                "id": "op0001",
                "type": "Parameter",
                "semantic_name": "weight",
                "inputs": [],
                "outputs": ["v0001"],
                "attributes": {
                    "archive": "weight_tensors.npz",
                    "archive_key": "weight",
                    "content_sha256": zero_hash,
                },
            },
            {
                "id": "op0002",
                "type": "Parameter",
                "semantic_name": "bias",
                "inputs": [],
                "outputs": ["v0002"],
                "attributes": {
                    "archive": "weight_tensors.npz",
                    "archive_key": "bias",
                    "content_sha256": zero_hash,
                },
            },
            {
                "id": "op0003",
                "type": operation_type,
                "semantic_name": "projection",
                "inputs": ["v0000", "v0001"],
                "outputs": ["v0003"],
                "attributes": {"bias": False, "in_features": 2, "out_features": 2},
            },
            {
                "id": "op0004",
                "type": "Add",
                "semantic_name": "bias_add",
                "inputs": ["v0003", "v0002"],
                "outputs": ["v0004"],
                "attributes": {},
            },
            {
                "id": "op0005",
                "type": "GELU",
                "semantic_name": "gelu",
                "inputs": ["v0004"],
                "outputs": ["v0005"],
                "attributes": {"approximate": "none"},
            },
        ],
        "weight_manifest": {
            "reference": "manifest.sha256",
            "sha256": zero_hash,
            "weight_archive": "weight_tensors.npz",
            "weight_archive_sha256": zero_hash,
        },
    }
    graph["graph_hash"] = compute_graph_hash(graph)
    return graph


def test_o2_bias_gelu_fusion_is_semantically_equivalent(tmp_path: Path) -> None:
    weights_path = tmp_path / "weight_tensors.npz"
    np.savez(
        weights_path,
        weight=np.array([[1.0, -0.5], [0.25, 2.0]], dtype=np.float32),
        bias=np.array([0.125, -0.25], dtype=np.float32),
    )
    input_tensor = torch.tensor([[0.5, -1.5]], dtype=torch.float32)
    for operation_type in ("Linear", "MatMul"):
        graph = _fusion_graph(operation_type)
        graph_path = tmp_path / operation_type / "fusion.graph.json"
        optimized_path = tmp_path / operation_type / "O2" / "fusion.graph.json"
        optimized_path.parent.mkdir(parents=True)
        write_canonical_json(graph_path, graph)
        before = evaluate_graph(graph_path, {"v0000": input_tensor}, weights_path)["v0005"]
        report = _optimize(graph_path, optimized_path, "O2")
        optimized = _validate_schema(optimized_path)
        after = evaluate_graph(optimized_path, {"v0000": input_tensor}, weights_path)["v0005"]

        torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)
        assert report["operation_counts"] == {"before": 6, "after": 4, "change": -2}
        assert optimized["outputs"] == ["v0005"]
        assert optimized["operations"][-1]["id"] == "op0005"
        assert optimized["operations"][-1]["type"] == operation_type
        assert optimized["operations"][-1]["attributes"]["fused_activation"] == "GELU"
