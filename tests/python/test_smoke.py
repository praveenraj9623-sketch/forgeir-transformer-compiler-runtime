"""Milestone 1 Python and CLI smoke tests."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import forgeir
import forgeir_py

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_VERSION = "0.1.0"
EXPECTED_DIAGNOSTIC_KEYS = {
    "forgeir_version",
    "compiler",
    "build_type",
    "operating_system",
    "cpp_standard",
    "python_version",
    "features",
}


def _cli_path() -> Path:
    executable_name = "forgeir_cli.exe" if sys.platform == "win32" else "forgeir_cli"
    for preset in (
        "windows-msvc-debug",
        "windows-msvc-release",
        "linux-gcc-debug",
        "linux-gcc-release",
    ):
        candidate = ROOT / "build" / preset / "bin" / executable_name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("forgeir_cli has not been built with a supported preset")


def _assert_diagnostic_schema(diagnostic: dict[str, Any], *, from_python: bool) -> None:
    assert set(diagnostic) == EXPECTED_DIAGNOSTIC_KEYS
    assert diagnostic["forgeir_version"] == EXPECTED_VERSION
    assert isinstance(diagnostic["compiler"], str) and diagnostic["compiler"]
    assert diagnostic["build_type"] in {"Debug", "Release"}
    assert isinstance(diagnostic["operating_system"], str) and diagnostic["operating_system"]
    assert diagnostic["cpp_standard"] == "C++17"
    if from_python:
        assert isinstance(diagnostic["python_version"], str)
        assert diagnostic["python_version"].startswith("3.11.")
    else:
        assert diagnostic["python_version"] is None
    assert diagnostic["features"] == {"cuda": False, "hip": False, "mlir": False}


def test_python_binding_version() -> None:
    assert forgeir_py.version() == EXPECTED_VERSION
    assert forgeir.version() == EXPECTED_VERSION
    assert forgeir.__version__ == EXPECTED_VERSION


def test_cli_version() -> None:
    result = subprocess.run(
        [_cli_path(), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == f"ForgeIR {EXPECTED_VERSION}"


def test_cli_diagnostic_schema() -> None:
    result = subprocess.run(
        [_cli_path(), "doctor"],
        check=True,
        capture_output=True,
        text=True,
    )
    _assert_diagnostic_schema(json.loads(result.stdout), from_python=False)


def test_python_diagnostic_schema() -> None:
    _assert_diagnostic_schema(forgeir_py.doctor(), from_python=True)
