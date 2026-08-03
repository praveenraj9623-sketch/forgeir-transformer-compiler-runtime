"""Controlled PyTorch FX lowering into the ForgeIR graph contract."""

from __future__ import annotations

import argparse
import json
import operator
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import fx, nn

from forgeir.export.contract import (
    GRAPH_SCHEMA_VERSION,
    PRODUCER_VERSION,
    compute_graph_hash,
    verify_graph_hash,
    write_canonical_json,
)
from forgeir.export.errors import UnsupportedFxNodeError, UnsupportedNodeDetails
from forgeir.export.weights import WeightManifest, WeightReference
from forgeir.reference.config import TinyTransformerConfig
from forgeir.reference.model import (
    CausalSelfAttention,
    RMSNorm,
    create_deterministic_model,
)

DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHT_DIRECTORY = (
    DEFAULT_REPOSITORY_ROOT / "artifacts" / "references" / "milestone_02" / "default_run_1"
)
DEFAULT_OUTPUT_DIRECTORY = (
    DEFAULT_REPOSITORY_ROOT / "artifacts" / "graphs" / "milestone_03" / "default"
)
DEFAULT_WEIGHT_MANIFEST_REFERENCE = (
    "artifacts/references/milestone_02/default_run_1/manifest.sha256"
)


@dataclass(frozen=True, slots=True)
class FxExportResult:
    """Generated graph artifacts and deterministic summary."""

    graph_path: Path
    dot_path: Path
    graph_hash: str
    operation_counts: dict[str, int]
    graph: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "graph_path": str(self.graph_path),
            "dot_path": str(self.dot_path),
            "graph_hash": self.graph_hash,
            "operation_counts": self.operation_counts,
        }


class _ReferenceTracer(fx.Tracer):
    def is_leaf_module(self, module: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(module, (RMSNorm, CausalSelfAttention)):
            return True
        return super().is_leaf_module(module, module_qualified_name)


def trace_reference_block(block: nn.Module) -> fx.GraphModule:
    """Trace a transformer block while retaining controlled composite boundaries."""
    tracer = _ReferenceTracer()
    graph = tracer.trace(block)
    return fx.GraphModule(block, graph)


def _describe_argument(argument: object) -> object:
    if isinstance(argument, fx.Node):
        return {"node": argument.name}
    if isinstance(argument, tuple | list):
        return [_describe_argument(value) for value in argument]
    if isinstance(argument, dict):
        return {
            str(key): _describe_argument(value)
            for key, value in sorted(argument.items(), key=lambda item: str(item[0]))
        }
    if argument is None or isinstance(argument, bool | int | float | str):
        return argument
    return repr(argument)


def _target_name(target: object) -> str:
    if isinstance(target, str):
        return target
    qualified_name = getattr(target, "__qualname__", None)
    if isinstance(qualified_name, str):
        return qualified_name
    return repr(target)


def _unsupported(node: fx.Node, reason: str) -> UnsupportedFxNodeError:
    return UnsupportedFxNodeError(
        UnsupportedNodeDetails(
            node_name=node.name,
            fx_target=_target_name(node.target),
            arguments={
                "args": _describe_argument(node.args),
                "kwargs": _describe_argument(node.kwargs),
            },
            reason=reason,
        )
    )


class _GraphBuilder:
    def __init__(self, config: TinyTransformerConfig, weights: WeightManifest) -> None:
        self.config = config
        self.weights = weights
        self.values: list[dict[str, object]] = []
        self.operations: list[dict[str, object]] = []
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self._value_by_id: dict[str, dict[str, object]] = {}
        self._parameter_values: dict[str, str] = {}

    def add_value(
        self,
        semantic_name: str,
        shape: tuple[int, ...],
        dtype: str,
        kind: str,
    ) -> str:
        value_id = f"v{len(self.values):04d}"
        value: dict[str, object] = {
            "id": value_id,
            "semantic_name": semantic_name,
            "shape": list(shape),
            "dtype": dtype,
            "kind": kind,
        }
        self.values.append(value)
        self._value_by_id[value_id] = value
        return value_id

    def add_operation(
        self,
        operation_type: str,
        semantic_name: str,
        inputs: list[str],
        outputs: list[str],
        attributes: dict[str, object] | None = None,
    ) -> None:
        self.operations.append(
            {
                "id": f"op{len(self.operations):04d}",
                "type": operation_type,
                "semantic_name": semantic_name,
                "inputs": inputs,
                "outputs": outputs,
                "attributes": attributes or {},
            }
        )

    def input(self) -> str:
        value_id = self.add_value(
            "block.input.hidden_states",
            (self.config.batch_size, self.config.sequence_length, self.config.hidden_size),
            self.config.dtype,
            "input",
        )
        self.add_operation("Input", "block.input", [], [value_id])
        self.inputs.append(value_id)
        return value_id

    def parameter(self, parameter_name: str) -> str:
        existing = self._parameter_values.get(parameter_name)
        if existing is not None:
            return existing
        reference = self.weights.resolve(parameter_name)
        value_id = self.add_value(
            reference.semantic_name,
            reference.shape,
            reference.dtype,
            "parameter",
        )
        self.add_operation(
            "Parameter",
            reference.semantic_name,
            [],
            [value_id],
            _weight_attributes(reference),
        )
        self._parameter_values[parameter_name] = value_id
        return value_id

    def constant(self, semantic_name: str, value: float) -> str:
        value_id = self.add_value(semantic_name, (), self.config.dtype, "constant")
        self.add_operation("Constant", semantic_name, [], [value_id], {"value": value})
        return value_id

    def emit(
        self,
        operation_type: str,
        semantic_name: str,
        inputs: list[str],
        output_shape: tuple[int, ...],
        *,
        dtype: str | None = None,
        attributes: dict[str, object] | None = None,
    ) -> str:
        output_id = self.add_value(
            f"{semantic_name}.output", output_shape, dtype or self.config.dtype, "intermediate"
        )
        self.add_operation(operation_type, semantic_name, inputs, [output_id], attributes or {})
        return output_id

    def shape(self, value_id: str) -> tuple[int, ...]:
        shape = self._value_by_id[value_id]["shape"]
        return tuple(cast(list[int], shape))

    def mark_output(self, value_id: str) -> None:
        value = self._value_by_id[value_id]
        value["semantic_name"] = "block.output.hidden_states"
        value["kind"] = "output"
        self.outputs.append(value_id)


def _weight_attributes(reference: WeightReference) -> dict[str, object]:
    return {
        "archive": reference.archive,
        "archive_key": reference.archive_key,
        "content_sha256": reference.content_sha256,
    }


class _ControlledLowerer:
    def __init__(
        self,
        graph_module: fx.GraphModule,
        config: TinyTransformerConfig,
        weights: WeightManifest,
    ) -> None:
        self.graph_module = graph_module
        self.config = config
        self.builder = _GraphBuilder(config, weights)
        self.node_values: dict[fx.Node, str] = {}

    def lower(self) -> dict[str, object]:
        for node in self.graph_module.graph.nodes:
            if node.op == "placeholder":
                self.node_values[node] = self.builder.input()
            elif node.op == "call_module":
                self.node_values[node] = self._lower_module(node)
            elif node.op == "call_function":
                self.node_values[node] = self._lower_function(node)
            elif node.op == "output":
                self._lower_output(node)
            else:
                raise _unsupported(node, f"FX node kind {node.op!r} is not supported")

        if not self.builder.outputs:
            raise ValueError("FX graph has no output")
        return {
            "inputs": self.builder.inputs,
            "outputs": self.builder.outputs,
            "values": self.builder.values,
            "operations": self.builder.operations,
        }

    def _node_input(self, node: fx.Node, index: int = 0) -> str:
        if len(node.args) <= index or not isinstance(node.args[index], fx.Node):
            raise _unsupported(node, f"argument {index} must reference an FX node")
        argument = cast(fx.Node, node.args[index])
        if argument not in self.node_values:
            raise _unsupported(node, f"argument {index} has no lowered value")
        return self.node_values[argument]

    def _lower_module(self, node: fx.Node) -> str:
        if not isinstance(node.target, str):
            raise _unsupported(node, "call_module target must be a qualified module name")
        module = self.graph_module.get_submodule(node.target)
        input_id = self._node_input(node)
        qualified_name = f"block.{node.target}"
        if isinstance(module, RMSNorm):
            weight_id = self.builder.parameter(f"{qualified_name}.weight")
            return self.builder.emit(
                "RMSNorm",
                qualified_name,
                [input_id, weight_id],
                self.builder.shape(input_id),
                attributes={"epsilon": module.epsilon, "axis": -1},
            )
        if isinstance(module, CausalSelfAttention):
            return self._lower_attention(input_id, qualified_name, module)
        if isinstance(module, nn.Linear):
            return self._lower_linear(input_id, qualified_name, module)
        if isinstance(module, nn.GELU):
            return self.builder.emit(
                "GELU",
                qualified_name,
                [input_id],
                self.builder.shape(input_id),
                attributes={"approximate": module.approximate},
            )
        raise _unsupported(node, f"module type {type(module).__qualname__!r} has no lowering")

    def _lower_linear(self, input_id: str, qualified_name: str, module: nn.Linear) -> str:
        bias = cast(torch.Tensor | None, module.bias)
        if bias is not None:
            raise ValueError(f"{qualified_name} bias is unsupported by the reference contract")
        weight_id = self.builder.parameter(f"{qualified_name}.weight")
        output_shape = (*self.builder.shape(input_id)[:-1], module.out_features)
        return self.builder.emit(
            "Linear",
            qualified_name,
            [input_id, weight_id],
            output_shape,
            attributes={
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bias": False,
            },
        )

    def _lower_attention(
        self, input_id: str, qualified_name: str, module: CausalSelfAttention
    ) -> str:
        batch = self.config.batch_size
        sequence = self.config.sequence_length
        hidden = self.config.hidden_size
        heads = self.config.num_heads
        head_size = self.config.head_size
        split_shape = (batch, sequence, heads, head_size)
        head_shape = (batch, heads, sequence, head_size)

        query = self._lower_projection(input_id, qualified_name, "query_projection")
        query = self.builder.emit(
            "Reshape",
            f"{qualified_name}.query.reshape",
            [query],
            split_shape,
            attributes={"shape": list(split_shape)},
        )
        query = self.builder.emit(
            "Transpose",
            f"{qualified_name}.query.transpose",
            [query],
            head_shape,
            attributes={"permutation": [0, 2, 1, 3]},
        )

        key = self._lower_projection(input_id, qualified_name, "key_projection")
        key = self.builder.emit(
            "Reshape",
            f"{qualified_name}.key.reshape",
            [key],
            split_shape,
            attributes={"shape": list(split_shape)},
        )
        key = self.builder.emit(
            "Transpose",
            f"{qualified_name}.key.transpose_heads",
            [key],
            head_shape,
            attributes={"permutation": [0, 2, 1, 3]},
        )
        key_transposed_shape = (batch, heads, head_size, sequence)
        key = self.builder.emit(
            "Transpose",
            f"{qualified_name}.key.transpose_scores",
            [key],
            key_transposed_shape,
            attributes={"permutation": [0, 1, 3, 2]},
        )

        value = self._lower_projection(input_id, qualified_name, "value_projection")
        value = self.builder.emit(
            "Reshape",
            f"{qualified_name}.value.reshape",
            [value],
            split_shape,
            attributes={"shape": list(split_shape)},
        )
        value = self.builder.emit(
            "Transpose",
            f"{qualified_name}.value.transpose",
            [value],
            head_shape,
            attributes={"permutation": [0, 2, 1, 3]},
        )

        score_shape = (batch, heads, sequence, sequence)
        scores = self.builder.emit(
            "MatMul", f"{qualified_name}.scores.matmul", [query, key], score_shape
        )
        scale = self.builder.constant(f"{qualified_name}.scores.scale", module.scale)
        scores = self.builder.emit(
            "Mul", f"{qualified_name}.scores.scale_mul", [scores, scale], score_shape
        )
        scores = self.builder.emit(
            "CausalMask",
            f"{qualified_name}.causal_mask",
            [scores],
            score_shape,
            attributes={"diagonal": 0, "masked_value": "-inf"},
        )
        probabilities = self.builder.emit(
            "Softmax",
            f"{qualified_name}.softmax",
            [scores],
            score_shape,
            attributes={"axis": -1, "stable": True},
        )
        context = self.builder.emit(
            "MatMul", f"{qualified_name}.context.matmul", [probabilities, value], head_shape
        )
        merged_shape = (batch, sequence, heads, head_size)
        context = self.builder.emit(
            "Transpose",
            f"{qualified_name}.context.transpose",
            [context],
            merged_shape,
            attributes={"permutation": [0, 2, 1, 3]},
        )
        output_shape = (batch, sequence, hidden)
        context = self.builder.emit(
            "Reshape",
            f"{qualified_name}.context.reshape",
            [context],
            output_shape,
            attributes={"shape": list(output_shape)},
        )
        return self._lower_projection(context, qualified_name, "output_projection")

    def _lower_projection(self, input_id: str, qualified_name: str, name: str) -> str:
        module = cast(nn.Linear, self.graph_module.get_submodule(f"{qualified_name[6:]}.{name}"))
        return self._lower_linear(input_id, f"{qualified_name}.{name}", module)

    def _lower_function(self, node: fx.Node) -> str:
        if node.target not in (operator.add, torch.add):
            raise _unsupported(node, "only Add has a canonical call_function lowering")
        left = self._node_input(node, 0)
        right = self._node_input(node, 1)
        if self.builder.shape(left) != self.builder.shape(right):
            raise _unsupported(node, "Add inputs must have identical shapes")
        return self.builder.emit(
            "Add", f"block.{node.name}", [left, right], self.builder.shape(left)
        )

    def _lower_output(self, node: fx.Node) -> None:
        output_id = self._node_input(node)
        self.builder.mark_output(output_id)


def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def graph_to_dot(graph: dict[str, object]) -> str:
    """Create a deterministic operation-level DOT representation."""
    operations = cast(list[dict[str, object]], graph["operations"])
    producer: dict[str, str] = {}
    lines = ["digraph ForgeIR {", "  rankdir=LR;"]
    for operation in operations:
        operation_id = cast(str, operation["id"])
        operation_type = cast(str, operation["type"])
        semantic_name = cast(str, operation["semantic_name"])
        label = _dot_escape(f"{operation_id} | {operation_type} | {semantic_name}")
        lines.append(f'  {operation_id} [shape=box,label="{label}"];')
        for output_id in cast(list[str], operation["outputs"]):
            producer[output_id] = operation_id
    for operation in operations:
        operation_id = cast(str, operation["id"])
        for input_id in cast(list[str], operation["inputs"]):
            source = producer.get(input_id)
            if source is not None:
                lines.append(f'  {source} -> {operation_id} [label="{input_id}"];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def export_fx_graph(
    output_directory: Path,
    weight_directory: Path,
    *,
    config: TinyTransformerConfig | None = None,
    weight_manifest_reference: str = DEFAULT_WEIGHT_MANIFEST_REFERENCE,
    block: nn.Module | None = None,
) -> FxExportResult:
    """Trace and export one controlled transformer block graph."""
    resolved_config = config or TinyTransformerConfig()
    weights = WeightManifest(weight_directory, resolved_config, weight_manifest_reference)
    resolved_block = block or create_deterministic_model(resolved_config).block
    resolved_block.eval()
    graph_module = trace_reference_block(resolved_block)
    lowered = _ControlledLowerer(graph_module, resolved_config, weights).lower()

    graph: dict[str, object] = {
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "producer_version": PRODUCER_VERSION,
        "model_configuration_hash": weights.configuration_hash,
        "inputs": lowered["inputs"],
        "outputs": lowered["outputs"],
        "values": lowered["values"],
        "operations": lowered["operations"],
        "weight_manifest": weights.contract_reference(),
    }
    graph["graph_hash"] = compute_graph_hash(graph)
    verify_graph_hash(graph)

    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    graph_path = output_directory / "tiny_transformer_block.graph.json"
    dot_path = output_directory / "tiny_transformer_block.dot"
    write_canonical_json(graph_path, graph)
    dot_path.write_text(graph_to_dot(graph), encoding="utf-8")

    operation_types = [
        cast(str, operation["type"])
        for operation in cast(list[dict[str, object]], graph["operations"])
    ]
    operation_counts = dict(sorted(Counter(operation_types).items()))
    return FxExportResult(
        graph_path=graph_path,
        dot_path=dot_path,
        graph_hash=cast(str, graph["graph_hash"]),
        operation_counts=operation_counts,
        graph=graph,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the deterministic transformer block to ForgeIR graph JSON."
    )
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHT_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--weight-manifest-reference",
        default=DEFAULT_WEIGHT_MANIFEST_REFERENCE,
        help="Portable manifest reference recorded in the graph contract.",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    result = export_fx_graph(
        args.output_dir,
        args.weights_dir,
        weight_manifest_reference=args.weight_manifest_reference,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
