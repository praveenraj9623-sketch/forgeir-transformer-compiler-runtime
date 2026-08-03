#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"

root="$(forgeir_root)"
forgeir_require_linux_tools
for tool in nvcc nvidia-smi; do
    command -v "${tool}" >/dev/null 2>&1 || {
        printf 'Required CUDA tool not found: %s\n' "${tool}" >&2
        exit 1
    }
done
nvidia-smi -L | grep -q 'GPU ' || {
    printf '%s\n' 'No NVIDIA GPU was reported by nvidia-smi.' >&2
    exit 1
}
pybind11_cmake_dir="$(python -m pybind11 --cmakedir)"

cd -- "${root}"
cmake --preset linux-gcc-cuda-release "-Dpybind11_DIR=${pybind11_cmake_dir}"
