# Build system

ForgeIR uses CMake 3.28 or newer and Ninja for deterministic single-configuration builds.
C++17 is required without compiler-specific language extensions. The supported host combinations are
MSVC with Ninja on Windows and GCC with Ninja on Linux.

## Presets and outputs

`CMakePresets.json` defines Debug and Release presets for both supported hosts. Each preset writes to
`build/<preset-name>`. Executables are emitted under `bin`, static libraries under `lib`, and the
Python extension under `python` within that preset directory.

The Windows and Linux scripts select these presets and fail explicitly when their required toolchain
or an active Python 3.11 virtual environment is unavailable. The Windows scripts discover Visual
Studio Build Tools, bundled CMake, and bundled Ninja through `vswhere`.

## Dependencies

Configuration prefers `third_party/googletest` and `third_party/nlohmann_json`. If either local source
is absent, configuration fails locally. A FetchContent fallback pinned to GoogleTest v1.17.0 or
nlohmann/json v3.12.0 is reachable only when the `CI` environment variable identifies a CI run.
ForgeIR targets receive strict warnings and warnings-as-errors; those flags are not applied to
third-party compilation.

pybind11 is never located through a user-specific path. CMake invokes the active Python interpreter as
`python -m pybind11 --cmakedir` and uses the returned package directory.

## Targets

- `forgeir_core`: static library containing version and build diagnostics.
- `forgeir_cli`: command-line smoke executable supporting `--version` and `doctor`.
- `forgeir_tests`: GoogleTest executable for the C++ foundation.
- `forgeir_py`: pybind11 module exposing version and build diagnostics to Python.

CUDA, HIP, and MLIR options default to disabled. Milestone 1 fails explicitly if they are enabled
because no integration is implemented yet. The diagnostic schema reports their compiled state as
false and reports the active Python version only when diagnostics are requested through Python.
