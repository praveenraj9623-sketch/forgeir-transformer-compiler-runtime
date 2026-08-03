#!/usr/bin/env bash

forgeir_root() {
    local script_dir
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    cd -- "${script_dir}/../.." && pwd
}

forgeir_linux_preset() {
    local configuration="${1:-Debug}"
    case "${configuration}" in
        Debug) printf '%s\n' 'linux-gcc-debug' ;;
        Release) printf '%s\n' 'linux-gcc-release' ;;
        *) printf 'Unsupported configuration: %s\n' "${configuration}" >&2; return 2 ;;
    esac
}

forgeir_require_linux_tools() {
    local tool
    for tool in python cmake ninja g++; do
        command -v "${tool}" >/dev/null 2>&1 || {
            printf 'Required tool not found: %s\n' "${tool}" >&2
            return 1
        }
    done
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        printf '%s\n' 'Activate a Python 3.11 virtual environment first.' >&2
        return 1
    fi
}
