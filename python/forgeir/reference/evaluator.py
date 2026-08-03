"""Deterministic CPU evaluator for ForgeIR reference and optimization validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from forgeir.export.contract import verify_graph_hash

_DTYPES: dict[str, torch.dtype] = {
    "bool": torch.bool,
    "float32": torch.float32,
    "int64": torch.int64,
}


def _document(graph: Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(graph, Path):
        loaded = json.loads(graph.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("graph document must be a JSON object")
        return cast(dict[str, Any], loaded)
    return dict(graph)


def _value_metadata(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values = document.get("values")
    if not isinstance(values, list):
        raise ValueError("graph values must be an array")
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict) or not isinstance(value.get("id"), str):
            raise ValueError("every graph value must have a string ID")
        result[value["id"]] = cast(dict[str, Any], value)
    return result


def _validate_tensor(value_id: str, tensor: Tensor, metadata: Mapping[str, Any]) -> None:
    shape = metadata.get("shape")
    dtype_name = metadata.get("dtype")
    if not isinstance(shape, list) or not all(isinstance(item, int) for item in shape):
        raise ValueError(f"value {value_id} has invalid shape metadata")
    if not isinstance(dtype_name, str) or dtype_name not in _DTYPES:
        raise ValueError(f"value {value_id} has invalid dtype metadata")
    if list(tensor.shape) != shape:
        raise ValueError(f"value {value_id} expected shape {shape}, received {list(tensor.shape)}")
    if tensor.dtype != _DTYPES[dtype_name]:
        raise TypeError(f"value {value_id} expected dtype {dtype_name}, received {tensor.dtype}")
    if tensor.device.type != "cpu":
        raise ValueError("reference graph evaluation supports CPU tensors only")


def _stable_softmax(input_tensor: Tensor, axis: int) -> Tensor:
    maximum = input_tensor.amax(dim=axis, keepdim=True)
    exponentials = torch.exp(input_tensor - maximum)
    return exponentials / exponentials.sum(dim=axis, keepdim=True)


def _constant(attributes: Mapping[str, Any], dtype: torch.dtype, shape: list[int]) -> Tensor:
    if "value" not in attributes:
        raise ValueError("Constant operation requires a value attribute")
    return torch.tensor(attributes["value"], dtype=dtype, device="cpu").reshape(shape)


def _projection(
    operation_type: str, operands: list[Tensor], attributes: Mapping[str, Any]
) -> Tensor:
    if operation_type == "Linear":
        result = torch.matmul(operands[0], operands[1].transpose(-2, -1))
    else:
        result = torch.matmul(operands[0], operands[1])
    if len(operands) == 3:
        result = result + operands[2]
    activation = attributes.get("fused_activation")
    if activation is not None:
        if activation != "GELU" or attributes.get("fused_activation_approximate") != "none":
            raise ValueError(f"unsupported fused activation contract: {activation!r}")
        result = torch.nn.functional.gelu(result, approximate="none")
    return result


def _evaluate_operation(
    operation_type: str,
    operands: list[Tensor],
    attributes: Mapping[str, Any],
    output_dtype: torch.dtype,
    output_shape: list[int],
) -> Tensor:
    if operation_type == "Constant":
        return _constant(attributes, output_dtype, output_shape)
    if operation_type in {"Linear", "MatMul"}:
        return _projection(operation_type, operands, attributes)
    if operation_type == "Add":
        return operands[0] + operands[1]
    if operation_type == "Mul":
        return operands[0] * operands[1]
    if operation_type == "Div":
        return operands[0] / operands[1]
    if operation_type == "RMSNorm":
        axis = int(attributes["axis"])
        epsilon = float(attributes["epsilon"])
        mean_square = operands[0].square().mean(dim=axis, keepdim=True)
        return operands[0] * torch.rsqrt(mean_square + epsilon) * operands[1]
    if operation_type == "GELU":
        approximate = attributes.get("approximate", "none")
        if approximate != "none":
            raise ValueError(f"unsupported GELU approximation: {approximate!r}")
        return torch.nn.functional.gelu(operands[0], approximate="none")
    if operation_type == "Softmax":
        return _stable_softmax(operands[0], int(attributes["axis"]))
    if operation_type == "Reshape":
        return operands[0].reshape(output_shape)
    if operation_type == "Transpose":
        permutation = attributes.get("permutation")
        if not isinstance(permutation, list) or not all(
            isinstance(index, int) for index in permutation
        ):
            raise ValueError("Transpose requires an integer permutation")
        return operands[0].permute(*permutation)
    if operation_type == "CausalMask":
        diagonal = int(attributes.get("diagonal", 0))
        query_length = operands[0].shape[-2]
        key_length = operands[0].shape[-1]
        mask = torch.triu(
            torch.ones((query_length, key_length), dtype=torch.bool), diagonal=diagonal + 1
        )
        return operands[0].masked_fill(mask, -torch.inf)
    raise ValueError(f"unsupported reference evaluator operation: {operation_type}")


def evaluate_graph(
    graph: Path | Mapping[str, Any],
    inputs: Mapping[str, Tensor],
    weight_archive: Path,
    *,
    capture_value_ids: tuple[str, ...] = (),
) -> dict[str, Tensor]:
    """Evaluate a controlled graph and return declared outputs plus requested checkpoints."""
    document = _document(graph)
    verify_graph_hash(document)
    metadata = _value_metadata(document)
    tensors: dict[str, Tensor] = {}
    for value_id, tensor in inputs.items():
        if value_id not in metadata:
            raise ValueError(f"input value {value_id} is absent from the graph")
        _validate_tensor(value_id, tensor, metadata[value_id])
        tensors[value_id] = tensor

    operations = document.get("operations")
    if not isinstance(operations, list):
        raise ValueError("graph operations must be an array")
    with np.load(weight_archive, allow_pickle=False) as weights, torch.no_grad():
        for raw_operation in operations:
            if not isinstance(raw_operation, dict):
                raise ValueError("every graph operation must be an object")
            operation = cast(dict[str, Any], raw_operation)
            operation_type = operation.get("type")
            output_ids = operation.get("outputs")
            input_ids = operation.get("inputs")
            attributes = operation.get("attributes")
            if (
                not isinstance(operation_type, str)
                or not isinstance(output_ids, list)
                or len(output_ids) != 1
                or not isinstance(output_ids[0], str)
                or not isinstance(input_ids, list)
                or not all(isinstance(item, str) for item in input_ids)
                or not isinstance(attributes, dict)
            ):
                raise ValueError("operation does not satisfy the controlled graph contract")
            output_id = output_ids[0]
            output_metadata = metadata[output_id]
            if operation_type == "Input":
                if output_id not in tensors:
                    raise ValueError(f"missing tensor for graph input {output_id}")
                continue
            if operation_type == "Parameter":
                archive_key = attributes.get("archive_key")
                if not isinstance(archive_key, str) or archive_key not in weights:
                    raise ValueError(f"missing parameter archive key for {output_id}")
                tensors[output_id] = torch.from_numpy(weights[archive_key].copy())
            else:
                try:
                    operands = [tensors[value_id] for value_id in input_ids]
                except KeyError as error:
                    raise ValueError(
                        f"operation references unavailable value {error.args[0]}"
                    ) from error
                shape = output_metadata.get("shape")
                dtype_name = output_metadata.get("dtype")
                if not isinstance(shape, list) or not all(isinstance(item, int) for item in shape):
                    raise ValueError(f"value {output_id} has invalid shape metadata")
                if not isinstance(dtype_name, str) or dtype_name not in _DTYPES:
                    raise ValueError(f"value {output_id} has invalid dtype metadata")
                tensors[output_id] = _evaluate_operation(
                    operation_type,
                    operands,
                    attributes,
                    _DTYPES[dtype_name],
                    cast(list[int], shape),
                )
            _validate_tensor(output_id, tensors[output_id], output_metadata)

    output_ids = document.get("outputs")
    if not isinstance(output_ids, list) or not all(isinstance(item, str) for item in output_ids):
        raise ValueError("graph outputs must be an array of value IDs")
    requested = list(output_ids)
    for value_id in capture_value_ids:
        if value_id not in tensors:
            raise ValueError(f"requested checkpoint {value_id} was not produced")
        if value_id not in requested:
            requested.append(value_id)
    return {value_id: tensors[value_id] for value_id in requested}
