"""Canonical ForgeIR graph contract serialization and hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

GRAPH_SCHEMA_VERSION = "1.0"
PRODUCER_VERSION = "0.1.0"


def canonical_json_text(document: object) -> str:
    """Serialize JSON with deterministic key order and no insignificant whitespace."""
    return json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_sha256(document: object) -> str:
    """Hash canonical UTF-8 JSON bytes."""
    return hashlib.sha256(canonical_json_text(document).encode("utf-8")).hexdigest()


def compute_graph_hash(graph: Mapping[str, object]) -> str:
    """Compute the graph hash while excluding the self-referential graph_hash field."""
    payload = dict(graph)
    payload.pop("graph_hash", None)
    return canonical_json_sha256(payload)


def verify_graph_hash(graph: Mapping[str, object]) -> None:
    """Raise ValueError unless the embedded deterministic graph hash is correct."""
    recorded = graph.get("graph_hash")
    if not isinstance(recorded, str):
        raise ValueError("graph_hash must be a string")
    calculated = compute_graph_hash(graph)
    if recorded != calculated:
        raise ValueError(f"graph hash mismatch: expected {recorded}, calculated {calculated}")


def write_canonical_json(path: Path, document: object) -> None:
    """Write canonical JSON followed by one newline."""
    path.write_text(canonical_json_text(document) + "\n", encoding="utf-8")
