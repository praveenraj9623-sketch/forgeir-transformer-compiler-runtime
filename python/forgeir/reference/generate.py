"""CLI for deterministic TinyTransformer reference artifact generation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from forgeir.reference.artifacts import generate_reference_artifacts

DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[3] / "artifacts" / "references" / "tiny_transformer_default"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic ForgeIR TinyTransformer reference artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Artifact output directory (default: %(default)s)",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    paths = generate_reference_artifacts(args.output_dir)
    print(json.dumps(paths.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
