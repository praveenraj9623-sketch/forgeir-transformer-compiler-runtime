"""Integration coverage for deterministic scheduling and static memory-plan artifacts."""

from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as element_tree
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "artifacts" / "graphs" / "milestone_06" / "O2" / "tiny_transformer_block.graph.json"


def _cli() -> Path:
    candidates = (
        ROOT / "build" / "windows-msvc-debug" / "bin" / "forgeir_cli.exe",
        ROOT / "build" / "linux-gcc-debug" / "bin" / "forgeir_cli",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("forgeir_cli debug executable must be built before integration tests")


def _run_plan(output_directory: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(_cli()),
            "plan-memory",
            str(GRAPH),
            "--output-dir",
            str(output_directory),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert isinstance(summary, dict)
    return cast(dict[str, Any], summary)


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return cast(dict[str, Any], document)


def _ranges_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_start = cast(int, left["arena_offset"])
    right_start = cast(int, right["arena_offset"])
    left_end = left_start + cast(int, left["aligned_byte_size"])
    right_end = right_start + cast(int, right["aligned_byte_size"])
    return left_start < right_end and right_start < left_end


def _lifetimes_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_interval = cast(dict[str, int], left["lifetime_interval"])
    right_interval = cast(dict[str, int], right["lifetime_interval"])
    return not (
        left_interval["end"] < right_interval["start"]
        or right_interval["end"] < left_interval["start"]
    )


def test_cli_memory_plan_artifacts_are_deterministic_and_safe(tmp_path: Path) -> None:
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_summary = _run_plan(first_directory)
    second_summary = _run_plan(second_directory)

    first_comparable = dict(first_summary)
    second_comparable = dict(second_summary)
    first_comparable.pop("artifacts")
    second_comparable.pop("artifacts")
    assert first_comparable == second_comparable

    artifact_names = ("schedule.json", "memory_plan.json", "timeline.csv", "timeline.svg")
    for name in artifact_names:
        assert (first_directory / name).read_bytes() == (second_directory / name).read_bytes()

    schedule = _load_json(first_directory / "schedule.json")
    plan = _load_json(first_directory / "memory_plan.json")
    operations = cast(list[dict[str, Any]], schedule["operations"])
    tensors = cast(list[dict[str, Any]], plan["tensors"])
    assert [operation["index"] for operation in operations] == list(range(len(operations)))
    assert len({operation["operation_id"] for operation in operations}) == len(operations)

    arena_tensors = [tensor for tensor in tensors if tensor["arena_offset"] is not None]
    for tensor in arena_tensors:
        assert cast(int, tensor["arena_offset"]) % cast(int, tensor["alignment_bytes"]) == 0
    for index, left in enumerate(arena_tensors):
        for right in arena_tensors[index + 1 :]:
            if _lifetimes_overlap(left, right):
                assert not _ranges_overlap(left, right)

    outputs = [tensor for tensor in tensors if tensor["kind"] == "output"]
    assert outputs
    for output in outputs:
        assert output["protected"] is True
        assert output["storage_class"] == "arena_output"
        assert output["final_use_index"] == len(operations)

    for tensor in tensors:
        if tensor["kind"] == "input":
            assert tensor["external"] is True
            assert tensor["protected"] is True
            assert tensor["arena_offset"] is None
        if tensor["kind"] in {"parameter", "constant"}:
            assert tensor["external"] is True
            assert tensor["immutable"] is True
            assert tensor["protected"] is True
            assert tensor["arena_offset"] is None

    summary = cast(dict[str, float | int], plan["summary"])
    assert summary["planned_bytes"] <= summary["naive_allocation_bytes"]
    assert summary["peak_live_bytes"] <= summary["naive_allocation_bytes"]
    assert summary["reuse_ratio"] >= 1.0
    element_tree.parse(first_directory / "timeline.svg")
