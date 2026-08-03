#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"

configuration="${1:-Debug}"
root="$(forgeir_root)"
preset="$(forgeir_linux_preset "${configuration}")"
forgeir_require_linux_tools
pybind11_cmake_dir="$(python -m pybind11 --cmakedir)"

cd -- "${root}"
cmake --preset "${preset}" "-Dpybind11_DIR=${pybind11_cmake_dir}"
