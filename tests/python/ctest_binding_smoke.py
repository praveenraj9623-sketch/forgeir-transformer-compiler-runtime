"""CTest entry point for the built Python extension."""

import importlib
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("expected: module-dir package-dir expected-version")

    module_dir, package_dir, expected_version = sys.argv[1:]
    sys.path.insert(0, str(Path(module_dir)))
    sys.path.insert(0, str(Path(package_dir)))

    binding = importlib.import_module("forgeir_py")
    package = importlib.import_module("forgeir")
    if binding.version() != expected_version or package.__version__ != expected_version:
        raise RuntimeError("Python binding version does not match the CMake project version")
    print(f"ForgeIR Python binding {binding.version()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
