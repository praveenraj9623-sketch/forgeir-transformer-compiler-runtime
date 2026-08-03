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

cd -- "${root}"
ctest --preset linux-gcc-cuda-release --output-on-failure
export PYTHONPATH="${root}/build/linux-gcc-cuda-release/python:${root}/python"
python -m forgeir.benchmark.cuda_validation \
    --output "${root}/benchmarks/results/cuda/milestone_12.json" \
    --warmup 10 \
    --iterations 50
