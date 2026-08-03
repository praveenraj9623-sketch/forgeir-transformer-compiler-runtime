# Phase 01: Repository foundation

## Scope

Milestone 1 establishes the CMake repository foundation, host environment diagnostics, Python
package metadata, cross-platform build scripts, and smoke tests. It does not implement tensors,
graphs, operators, compiler passes, transformer models, GPU code, MLIR code, or benchmarks.

## Verified environment

- Host: Windows
- Compiler: MSVC 19.42.34433.0
- CMake: 3.29.5-msvc4
- Ninja: 1.12.1
- Python: 3.11.9 from the repository virtual environment
- pybind11: 3.0.4 from the repository virtual environment
- GoogleTest: local `third_party/googletest`, v1.17.0
- nlohmann/json: local `third_party/nlohmann_json`, v3.12.0

No package was installed, no dependency was downloaded, and no FetchContent fallback was invoked.

## Commands and observed outcomes

The repository virtual environment was selected for each validation shell with:

    $env:VIRTUAL_ENV = (Resolve-Path '.venv').Path
    $env:Path = "$env:VIRTUAL_ENV\Scripts;$env:Path"

The pybind11 CMake directory was discovered dynamically:

    python -m pybind11 --cmakedir

Outcome: succeeded and returned the pybind11 CMake package inside the active repository virtual
environment. No user-specific path is present in build configuration.

Python lint:

    python -m ruff check python tests

Outcome: `All checks passed!`

Windows Debug configuration:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
    .\scripts\windows\configure.ps1 -Configuration Debug

The script discovered pybind11 with `python -m pybind11 --cmakedir` and then executed the equivalent
of `cmake --preset windows-msvc-debug -Dpybind11_DIR=<discovered-directory>`.

Outcome: succeeded with MSVC 19.42.34433.0, local dependencies, Python 3.11.9, and pybind11 3.0.4.
No network fallback ran.

Windows Debug build:

    .\scripts\windows\build.ps1 -Configuration Debug

The script executed `cmake --build --preset windows-msvc-debug`.

Outcome: succeeded. The `forgeir_core`, `forgeir_cli`, `forgeir_tests`, and `forgeir_py` targets were
built with ForgeIR warnings treated as errors.

Python smoke tests:

    python -m pytest tests/python -q

Outcome: `4 passed`.

CTest smoke tests:

    .\scripts\windows\test.ps1 -Configuration Debug

The script executed `ctest --preset windows-msvc-debug --output-on-failure`.

Outcome: `100% tests passed, 0 tests failed out of 5`. The passing tests covered the core version,
environment diagnostic schema, CLI version, CLI doctor output, and Python binding version.

CLI version smoke command:

    .\build\windows-msvc-debug\bin\forgeir_cli.exe --version

Outcome: `ForgeIR 0.1.0`.

CLI doctor smoke command:

    .\build\windows-msvc-debug\bin\forgeir_cli.exe doctor

Outcome: succeeded and reported ForgeIR 0.1.0, MSVC 19.42.34433.0, Debug, Windows, C++17, a null
Python version for the native CLI, and CUDA/HIP/MLIR compiled states of `false`.

Python binding smoke command:

    $env:PYTHONPATH = "$(Resolve-Path 'build\windows-msvc-debug\python');$(Resolve-Path 'python')"
    python -c "import forgeir_py; print(forgeir_py.version())"

Outcome: `0.1.0`. The Python diagnostic schema test additionally observed Python 3.11.9 and
CUDA/HIP/MLIR compiled states of `false`.

C++ format and repository hygiene checks:

    clang-format --dry-run --Werror --style=file <Milestone-1 C++ files>
    rg -n -i "TODO|NotImplementedError|fake return|\bpass\b" cpp bindings python tests cmake CMakeLists.txt
    git diff --check

Outcome: formatting and whitespace checks passed; the critical-path placeholder scan found no
matches.

## Environment limitations

The host PowerShell policy blocks direct execution of `.venv\Scripts\Activate.ps1`. Verification used
the same virtual environment by setting `VIRTUAL_ENV` and placing its `Scripts` directory first on
`PATH`. Project PowerShell scripts were run with a process-scoped execution-policy bypass; no machine
or user policy was changed.

Linux GCC presets and shell scripts were created but could not be executed on the Windows host. CUDA,
HIP, and MLIR were not compiled or tested in this milestone.
