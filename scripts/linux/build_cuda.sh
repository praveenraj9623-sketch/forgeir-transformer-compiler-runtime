#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"

root="$(forgeir_root)"
forgeir_require_linux_tools
command -v nvcc >/dev/null 2>&1 || {
    printf '%s\n' 'Required CUDA compiler not found: nvcc' >&2
    exit 1
}

cd -- "${root}"
cmake --build --preset linux-gcc-cuda-release
