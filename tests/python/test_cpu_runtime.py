"""Numerical and contract tests for the Milestone 8 float32 CPU runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import forgeir_py
from forgeir.export.contract import compute_graph_hash, write_canonical_json
from forgeir.reference import (
    TinyTransformerConfig,
    create_deterministic_input,
    create_deterministic_model,
    evaluate_graph,
)
from forgeir.reference.artifacts import array_content_sha256

ROOT = Path(__file__).resolve().parents[2]
REAL_GRAPH = ROOT / "tests" / "fixtures" / "tiny_transformer_block_v1_o2.graph.json"
ABSOLUTE_TOLERANCE = 2.0e-6
RELATIVE_TOLERANCE = 2.0e-5
ZERO_HASH = "0" * 64


@dataclass(frozen=True)
class RuntimeCase:
    graph_path: Path
    inputs: dict[str, np.ndarray[Any, np.dtype[np.float32]]]
    parameters: dict[str, np.ndarray[Any, np.dtype[np.float32]]]
    output_id: str


def _runtime_case(
    tmp_path: Path,
    name: str,
    operation_type: str,
    operands: list[np.ndarray[Any, np.dtype[np.float32]]],
    parameter_indices: set[int],
    output_shape: tuple[int, ...],
    attributes: dict[str, object],
) -> RuntimeCase:
    values: list[dict[str, object]] = []
    operations: list[dict[str, object]] = []
    graph_inputs: list[str] = []
    inputs: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {}
    parameters: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {}
    operation_inputs: list[str] = []
    for index, raw_array in enumerate(operands):
        array = np.ascontiguousarray(raw_array, dtype=np.float32)
        value_id = f"v{index:04d}"
        operation_id = f"op{index:04d}"
        operation_inputs.append(value_id)
        is_parameter = index in parameter_indices
        values.append(
            {
                "id": value_id,
                "semantic_name": f"{name}.operand.{index}",
                "shape": list(array.shape),
                "dtype": "float32",
                "kind": "parameter" if is_parameter else "input",
            }
        )
        if is_parameter:
            archive_key = f"parameter_{index}"
            parameters[archive_key] = array
            operation_attributes: dict[str, object] = {
                "archive": "weight_tensors.npz",
                "archive_key": archive_key,
                "content_sha256": array_content_sha256(array),
            }
            declaration_type = "Parameter"
        else:
            inputs[value_id] = array
            graph_inputs.append(value_id)
            operation_attributes = {}
            declaration_type = "Input"
        operations.append(
            {
                "id": operation_id,
                "type": declaration_type,
                "semantic_name": f"{name}.operand.{index}",
                "inputs": [],
                "outputs": [value_id],
                "attributes": operation_attributes,
            }
        )

    output_id = f"v{len(operands):04d}"
    values.append(
        {
            "id": output_id,
            "semantic_name": f"{name}.output",
            "shape": list(output_shape),
            "dtype": "float32",
            "kind": "output",
        }
    )
    operations.append(
        {
            "id": f"op{len(operands):04d}",
            "type": operation_type,
            "semantic_name": name,
            "inputs": operation_inputs,
            "outputs": [output_id],
            "attributes": attributes,
        }
    )
    graph: dict[str, object] = {
        "graph_schema_version": "1.0",
        "producer_version": "0.1.0",
        "model_configuration_hash": ZERO_HASH,
        "inputs": graph_inputs,
        "outputs": [output_id],
        "values": values,
        "operations": operations,
        "weight_manifest": {
            "reference": "manifest.sha256",
            "sha256": ZERO_HASH,
            "weight_archive": "weight_tensors.npz",
            "weight_archive_sha256": ZERO_HASH,
        },
    }
    graph["graph_hash"] = compute_graph_hash(graph)
    graph_path = tmp_path / f"{name}.graph.json"
    write_canonical_json(graph_path, graph)
    return RuntimeCase(graph_path, inputs, parameters, output_id)


def _execute_case(case: RuntimeCase) -> tuple[np.ndarray[Any, np.dtype[np.float32]], list[object]]:
    session = forgeir_py.load_graph(str(case.graph_path), "cpu")
    forgeir_py.execute(session, case.inputs, case.parameters)
    output = forgeir_py.get_outputs(session)[case.output_id]
    return output, list(forgeir_py.get_trace(session))


def _operator_cases() -> list[
    tuple[
        str,
        str,
        list[np.ndarray[Any, np.dtype[np.float32]]],
        set[int],
        dict[str, object],
        torch.Tensor,
    ]
]:
    left = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) / 7.0
    right = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5) / 11.0
    linear_input = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) / 9.0
    weight = np.arange(5 * 4, dtype=np.float32).reshape(5, 4) / 13.0
    broadcast_left = np.array([[[1.0], [-2.0], [3.0]]], dtype=np.float32)
    broadcast_right = np.array([[0.5, 2.0, -1.0, 4.0]], dtype=np.float32)
    norm_input = np.array([[1.0, -2.0, 3.0, -4.0], [0.5, 1.5, -2.5, 3.5]], dtype=np.float32)
    norm_weight = np.array([1.0, 0.75, 1.25, 0.5], dtype=np.float32)
    softmax_input = np.array([[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -998.0]], dtype=np.float32)
    transpose_input = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    mask_input = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    fused_input = np.array([[0.5, -1.5, 2.0], [1.25, 0.25, -0.5]], dtype=np.float32)
    fused_weight = np.array(
        [[1.0, -0.5, 0.25], [0.5, 1.5, -1.0], [-0.75, 0.5, 1.25]], dtype=np.float32
    )
    fused_bias = np.array([0.125, -0.25, 0.5], dtype=np.float32)

    torch_left = torch.from_numpy(left)
    torch_right = torch.from_numpy(right)
    torch_linear_input = torch.from_numpy(linear_input)
    torch_weight = torch.from_numpy(weight)
    torch_broadcast_left = torch.from_numpy(broadcast_left)
    torch_broadcast_right = torch.from_numpy(broadcast_right)
    torch_norm_input = torch.from_numpy(norm_input)
    torch_norm_weight = torch.from_numpy(norm_weight)
    torch_softmax_input = torch.from_numpy(softmax_input)
    torch_fused_input = torch.from_numpy(fused_input)
    torch_fused_weight = torch.from_numpy(fused_weight)
    torch_fused_bias = torch.from_numpy(fused_bias)
    stable_max = torch_softmax_input.amax(dim=1, keepdim=True)
    stable_exp = torch.exp(torch_softmax_input - stable_max)

    return [
        ("matmul", "MatMul", [left, right], set(), {}, torch.matmul(torch_left, torch_right)),
        (
            "linear",
            "Linear",
            [linear_input, weight],
            {1},
            {"bias": False, "in_features": 4, "out_features": 5},
            torch.matmul(torch_linear_input, torch_weight.transpose(-2, -1)),
        ),
        (
            "add",
            "Add",
            [broadcast_left, broadcast_right],
            set(),
            {},
            torch_broadcast_left + torch_broadcast_right,
        ),
        (
            "mul",
            "Mul",
            [broadcast_left, broadcast_right],
            set(),
            {},
            torch_broadcast_left * torch_broadcast_right,
        ),
        (
            "div",
            "Div",
            [broadcast_left, broadcast_right],
            set(),
            {},
            torch_broadcast_left / torch_broadcast_right,
        ),
        (
            "rms_norm",
            "RMSNorm",
            [norm_input, norm_weight],
            {1},
            {"axis": -1, "epsilon": 1.0e-5},
            torch_norm_input
            * torch.rsqrt(torch_norm_input.square().mean(dim=-1, keepdim=True) + 1.0e-5)
            * torch_norm_weight,
        ),
        (
            "gelu",
            "GELU",
            [norm_input],
            set(),
            {"approximate": "none"},
            torch.nn.functional.gelu(torch_norm_input, approximate="none"),
        ),
        (
            "softmax",
            "Softmax",
            [softmax_input],
            set(),
            {"axis": 1, "stable": True},
            stable_exp / stable_exp.sum(dim=1, keepdim=True),
        ),
        (
            "reshape",
            "Reshape",
            [transpose_input],
            set(),
            {"shape": [4, 6]},
            torch.from_numpy(transpose_input).reshape(4, 6),
        ),
        (
            "transpose",
            "Transpose",
            [transpose_input],
            set(),
            {"permutation": [2, 0, 1]},
            torch.from_numpy(transpose_input).permute(2, 0, 1),
        ),
        (
            "causal_mask",
            "CausalMask",
            [mask_input],
            set(),
            {"diagonal": 0, "masked_value": "-inf"},
            torch.from_numpy(mask_input).masked_fill(
                torch.triu(torch.ones((4, 4), dtype=torch.bool), diagonal=1), -torch.inf
            ),
        ),
        (
            "fused_bias_gelu_linear",
            "Linear",
            [fused_input, fused_weight, fused_bias],
            {1, 2},
            {
                "bias": True,
                "in_features": 3,
                "out_features": 3,
                "fused_activation": "GELU",
                "fused_activation_approximate": "none",
            },
            torch.nn.functional.gelu(
                torch.matmul(torch_fused_input, torch_fused_weight.transpose(-2, -1))
                + torch_fused_bias,
                approximate="none",
            ),
        ),
        (
            "fused_bias_gelu_matmul",
            "MatMul",
            [fused_input, fused_weight.transpose().copy(), fused_bias],
            {1, 2},
            {
                "bias": True,
                "fused_activation": "GELU",
                "fused_activation_approximate": "none",
            },
            torch.nn.functional.gelu(
                torch.matmul(torch_fused_input, torch_fused_weight.transpose(-2, -1))
                + torch_fused_bias,
                approximate="none",
            ),
        ),
    ]


@pytest.mark.parametrize(
    ("name", "operation_type", "operands", "parameter_indices", "attributes", "expected"),
    _operator_cases(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_standalone_cpu_operator_matches_pytorch(
    tmp_path: Path,
    name: str,
    operation_type: str,
    operands: list[np.ndarray[Any, np.dtype[np.float32]]],
    parameter_indices: set[int],
    attributes: dict[str, object],
    expected: torch.Tensor,
) -> None:
    case = _runtime_case(
        tmp_path,
        name,
        operation_type,
        operands,
        parameter_indices,
        tuple(expected.shape),
        attributes,
    )
    actual, trace = _execute_case(case)
    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
        equal_nan=True,
    )
    assert trace[-1]["operation_type"] == operation_type
    assert trace[-1]["output_shape"] == list(expected.shape)
    assert trace[-1]["elapsed_microseconds"] >= 0.0
    assert trace[-1]["arena_offset"] is not None


def test_complete_transformer_checkpoints_and_final_output_match_pytorch(
    tmp_path: Path, deterministic_reference_weights: Path
) -> None:
    config = TinyTransformerConfig()
    model = create_deterministic_model(config)
    input_ids = create_deterministic_input(config)
    checkpoints = ("v0002", "v0004", "v0016", "v0019", "v0020", "v0021", "v0026", "v0028", "v0031")
    with torch.no_grad():
        hidden_states = model.token_embedding(input_ids).contiguous()
        expected_model_output = model.block(hidden_states)
    expected = evaluate_graph(
        REAL_GRAPH,
        {"v0000": hidden_states},
        deterministic_reference_weights,
        capture_value_ids=checkpoints,
    )
    torch.testing.assert_close(expected["v0034"], expected_model_output, rtol=1.0e-6, atol=1.0e-6)

    with np.load(deterministic_reference_weights, allow_pickle=False) as archive:
        parameters = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    session = forgeir_py.load_graph(str(REAL_GRAPH), "cpu")
    forgeir_py.execute(
        session,
        {"v0000": hidden_states.numpy()},
        parameters,
        list(checkpoints),
    )
    actual_values = forgeir_py.get_outputs(session, [*checkpoints, "v0034"])

    checkpoint_errors: dict[str, dict[str, float]] = {}
    for value_id in (*checkpoints, "v0034"):
        actual = actual_values[value_id]
        expected_array = expected[value_id].detach().contiguous().numpy()
        np.testing.assert_allclose(
            actual,
            expected_array,
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
        )
        finite = np.isfinite(expected_array)
        absolute_error = np.abs(actual[finite] - expected_array[finite])
        relative_error = absolute_error / np.maximum(
            np.abs(expected_array[finite]), np.float32(1.0e-12)
        )
        checkpoint_errors[value_id] = {
            "max_absolute_error": float(absolute_error.max(initial=0.0)),
            "max_relative_error": float(relative_error.max(initial=0.0)),
        }

    trace = list(forgeir_py.get_trace(session))
    assert len(trace) == 35
    assert [item["operation_id"] for item in trace] == [f"op{index:04d}" for index in range(35)]
    assert trace[-1]["arena_offset"] is not None

    parity_report = tmp_path / "numerical_parity.json"
    parity_report.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "graph": str(REAL_GRAPH.relative_to(ROOT)).replace("\\", "/"),
                "binding": str(Path(forgeir_py.__file__).resolve()),
                "absolute_tolerance": ABSOLUTE_TOLERANCE,
                "relative_tolerance": RELATIVE_TOLERANCE,
                "checkpoints": checkpoint_errors,
                "final": checkpoint_errors["v0034"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_document = json.loads(parity_report.read_text(encoding="utf-8"))
    assert report_document["final"] == checkpoint_errors["v0034"]


def test_ieee_nan_and_infinity_propagation(tmp_path: Path) -> None:
    numerator = np.array([1.0, 0.0, np.nan], dtype=np.float32)
    denominator = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    case = _runtime_case(tmp_path, "ieee_div", "Div", [numerator, denominator], set(), (3,), {})
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = numerator / denominator
    actual, _ = _execute_case(case)
    assert np.isposinf(actual[0])
    assert np.isnan(actual[1])
    assert np.isnan(actual[2])
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))


def test_malformed_and_noncontiguous_weights_fail_explicitly(tmp_path: Path) -> None:
    input_tensor = np.array([[1.0, 2.0]], dtype=np.float32)
    weight = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    case = _runtime_case(
        tmp_path,
        "malformed_weight",
        "Linear",
        [input_tensor, weight],
        {1},
        (1, 2),
        {"bias": False, "in_features": 2, "out_features": 2},
    )

    session = forgeir_py.load_graph(str(case.graph_path))
    with pytest.raises(ValueError, match="missing external parameter"):
        forgeir_py.execute(session, case.inputs, {})

    tampered = {"parameter_1": weight.copy()}
    tampered["parameter_1"][0, 0] += 1.0
    with pytest.raises(ValueError, match="content SHA-256"):
        forgeir_py.execute(session, case.inputs, tampered)

    with pytest.raises(ValueError, match="shape does not match"):
        forgeir_py.execute(
            session,
            case.inputs,
            {"parameter_1": np.ones((1, 2), dtype=np.float32)},
        )

    with pytest.raises(TypeError, match="dtype float32"):
        forgeir_py.execute(session, case.inputs, {"parameter_1": weight.astype(np.float64)})

    noncontiguous = np.arange(8, dtype=np.float32).reshape(2, 4)[:, ::2]
    assert not noncontiguous.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        forgeir_py.execute(session, case.inputs, {"parameter_1": noncontiguous})


def test_unknown_backend_and_pre_execution_outputs_fail(tmp_path: Path) -> None:
    value = np.array([1.0, -1.0], dtype=np.float32)
    case = _runtime_case(
        tmp_path,
        "backend_contract",
        "GELU",
        [value],
        set(),
        (2,),
        {"approximate": "none"},
    )
    with pytest.raises(ValueError, match="no implementation"):
        forgeir_py.load_graph(str(case.graph_path), "gpu")
    session = forgeir_py.load_graph(str(case.graph_path), "cpu")
    with pytest.raises(ValueError, match="before successful execution"):
        forgeir_py.get_outputs(session)
