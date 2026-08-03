"""Test path setup for locally built ForgeIR artifacts."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON_PACKAGE_DIR = ROOT / "python"
BUILD_MODULE_DIRS = (
    ROOT / "build" / "windows-msvc-debug" / "python",
    ROOT / "build" / "windows-msvc-release" / "python",
    ROOT / "build" / "linux-gcc-debug" / "python",
    ROOT / "build" / "linux-gcc-release" / "python",
)

sys.path.insert(0, str(PYTHON_PACKAGE_DIR))
for module_dir in BUILD_MODULE_DIRS:
    if module_dir.is_dir():
        sys.path.insert(0, str(module_dir))
