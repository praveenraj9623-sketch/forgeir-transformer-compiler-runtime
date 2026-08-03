#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"

configuration="${1:-Debug}"
root="$(forgeir_root)"
preset="$(forgeir_linux_preset "${configuration}")"
target="${root}/build/${preset}"
case "${target}" in
    "${root}"/build/*)
        rm -rf -- "${target}"
        printf 'Removed %s\n' "${target}"
        ;;
    *)
        printf 'Refusing to clean unsafe path: %s\n' "${target}" >&2
        exit 1
        ;;
esac
