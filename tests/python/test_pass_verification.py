"""Property and integration tests for the C++ semantic verifier."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, given, seed, settings
from hypothesis import strategies as st

import forgeir_py
from forgeir.export.contract import canonical_json_text, compute_graph_hash

ROOT = Path(__file__).resolve().parents[2]
REAL_GRAPH = ROOT / "tests" / "fixtures" / "tiny_transformer_block_v1.graph.json"
SHAPES = st.lists(st.integers(min_value=1, max_value=4), min_size=0, max_size=4)
PROPERTY_SETTINGS = settings(
    max_examples=40,
    database=None,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _base_graph(
    values: list[dict[str, object]],
    operations: list[dict[str, object]],
    inputs: list[str],
    output: str,
) -> dict[str, object]:
    zero_hash = "0" * 64
    graph: dict[str, object] = {
        "graph_schema_version": "1.0",
        "producer_version": "0.1.0",
        "model_configuration_hash": zero_hash,
        "inputs": inputs,
        "outputs": [output],
        "values": values,
        "operations": operations,
        "weight_manifest": {
            "reference": "property/manifest.sha256",
            "sha256": zero_hash,
            "weight_archive": "weight_tensors.npz",
            "weight_archive_sha256": zero_hash,
        },
    }
    graph["graph_hash"] = compute_graph_hash(graph)
    return graph


def _write_graph(path: Path, graph: dict[str, object]) -> None:
    path.write_text(canonical_json_text(graph) + "\n", encoding="utf-8")


def _binary_graph(
    left: list[int], right: list[int], output: list[int], operation_type: str
) -> dict[str, object]:
    values: list[dict[str, object]] = [
        {
            "id": "v0000",
            "semantic_name": "left",
            "shape": left,
            "dtype": "float32",
            "kind": "input",
        },
        {
            "id": "v0001",
            "semantic_name": "right",
            "shape": right,
            "dtype": "float32",
            "kind": "input",
        },
        {
            "id": "v0002",
            "semantic_name": "output",
            "shape": output,
            "dtype": "float32",
            "kind": "output",
        },
    ]
    operations: list[dict[str, object]] = [
        {
            "id": "op0000",
            "type": "Input",
            "semantic_name": "left",
            "inputs": [],
            "outputs": ["v0000"],
            "attributes": {},
        },
        {
            "id": "op0001",
            "type": "Input",
            "semantic_name": "right",
            "inputs": [],
            "outputs": ["v0001"],
            "attributes": {},
        },
        {
            "id": "op0002",
            "type": operation_type,
            "semantic_name": "broadcast",
            "inputs": ["v0000", "v0001"],
            "outputs": ["v0002"],
            "attributes": {},
        },
    ]
    return _base_graph(values, operations, ["v0000", "v0001"], "v0002")


def _reshape_graph(input_shape: list[int], target_shape: list[int]) -> dict[str, object]:
    values: list[dict[str, object]] = [
        {
            "id": "v0000",
            "semantic_name": "input",
            "shape": input_shape,
            "dtype": "float32",
            "kind": "input",
        },
        {
            "id": "v0001",
            "semantic_name": "output",
            "shape": target_shape,
            "dtype": "float32",
            "kind": "output",
        },
    ]
    operations: list[dict[str, object]] = [
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
            "type": "Reshape",
            "semantic_name": "reshape",
            "inputs": ["v0000"],
            "outputs": ["v0001"],
            "attributes": {"shape": target_shape},
        },
    ]
    return _base_graph(values, operations, ["v0000"], "v0001")


def _broadcast_shape(left: list[int], right: list[int]) -> list[int] | None:
    result: list[int] = []
    for offset in range(1, max(len(left), len(right)) + 1):
        left_dimension = left[-offset] if offset <= len(left) else 1
        right_dimension = right[-offset] if offset <= len(right) else 1
        if left_dimension != right_dimension and left_dimension != 1 and right_dimension != 1:
            return None
        result.append(max(left_dimension, right_dimension))
    result.reverse()
    return result


def _assert_diagnostic_context(report: dict[str, Any]) -> None:
    diagnostics = report["diagnostics"]
    assert isinstance(diagnostics, list)
    for diagnostic in diagnostics:
        assert diagnostic["operation_id"].startswith("op")
        assert diagnostic["value_id"].startswith("v")


def test_full_transformer_verifies_through_python_binding() -> None:
    report = forgeir_py.verify_graph(str(REAL_GRAPH))
    assert report["success"] is True
    assert report["changed"] is False
    assert report["diagnostic_counts"] == {"error": 0, "warning": 0}
    assert [
        execution["pass"] for execution in report["pass_executions"] if execution["stage"] == "pass"
    ] == ["ShapeInferencePass", "DTypePropagationPass", "GraphVerifierPass"]


@seed(42)
@PROPERTY_SETTINGS
@given(left=SHAPES, right=SHAPES, operation_type=st.sampled_from(["Add", "Mul"]))
def test_broadcast_contract_property(
    left: list[int], right: list[int], operation_type: str, tmp_path: Path
) -> None:
    inferred = _broadcast_shape(left, right)
    declared_output = inferred if inferred is not None else [1]
    graph_path = tmp_path / "broadcast.graph.json"
    _write_graph(graph_path, _binary_graph(left, right, declared_output, operation_type))

    report = forgeir_py.verify_graph(str(graph_path))
    assert report["success"] is (inferred is not None)
    _assert_diagnostic_context(report)
    if inferred is None:
        assert report["diagnostics"][0]["code"] == "shape.broadcast.incompatible"


@seed(42)
@PROPERTY_SETTINGS
@given(input_shape=SHAPES)
def test_reshape_preserves_element_count_property(input_shape: list[int], tmp_path: Path) -> None:
    target_shape = [math.prod(input_shape)]
    graph_path = tmp_path / "valid_reshape.graph.json"
    _write_graph(graph_path, _reshape_graph(input_shape, target_shape))

    report = forgeir_py.verify_graph(str(graph_path))
    assert report["success"] is True
    assert report["diagnostic_counts"] == {"error": 0, "warning": 0}


@seed(42)
@PROPERTY_SETTINGS
@given(input_shape=SHAPES)
def test_reshape_rejects_changed_element_count_property(
    input_shape: list[int], tmp_path: Path
) -> None:
    target_shape = [math.prod(input_shape) + 1]
    graph_path = tmp_path / "invalid_reshape.graph.json"
    _write_graph(graph_path, _reshape_graph(input_shape, target_shape))

    report = forgeir_py.verify_graph(str(graph_path))
    assert report["success"] is False
    assert report["diagnostics"][0]["code"] == "shape.reshape.element_count"
    _assert_diagnostic_context(report)
