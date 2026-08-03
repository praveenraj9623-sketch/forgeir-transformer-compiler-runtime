"""Cross-language FX export and C++ graph-loading contract test."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import forgeir_py
from forgeir.export.fx_export import export_fx_graph
from forgeir.reference import TinyTransformerConfig, generate_reference_artifacts


def test_python_export_cpp_load_summary_match(tmp_path: Path) -> None:
    config = TinyTransformerConfig(
        vocabulary_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        sequence_length=4,
        batch_size=1,
    )
    weights = generate_reference_artifacts(tmp_path / "weights", config)
    exported = export_fx_graph(
        tmp_path / "graph",
        weights.output_directory,
        config=config,
        weight_manifest_reference="weights/manifest.sha256",
    )

    values = exported.graph["values"]
    operations = exported.graph["operations"]
    assert isinstance(values, list)
    assert isinstance(operations, list)

    expected_parameter_bytes = sum(
        _element_count(value["shape"]) * _dtype_width(value["dtype"])
        for value in values
        if value["kind"] == "parameter"
    )
    expected_histogram = dict(Counter(operation["type"] for operation in operations))

    summary = forgeir_py.graph_summary(str(exported.graph_path))
    assert summary == {
        "schema_version": "1.0",
        "input_count": len(exported.graph["inputs"]),
        "output_count": len(exported.graph["outputs"]),
        "value_count": len(values),
        "operation_count": len(operations),
        "operation_histogram": expected_histogram,
        "estimated_parameter_bytes": expected_parameter_bytes,
    }


def _element_count(shape: object) -> int:
    assert isinstance(shape, list)
    result = 1
    for dimension in shape:
        assert isinstance(dimension, int)
        result *= dimension
    return result


def _dtype_width(dtype: object) -> int:
    widths = {"bool": 1, "float32": 4, "int64": 8}
    assert isinstance(dtype, str)
    return widths[dtype]
