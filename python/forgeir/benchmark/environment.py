"""Host, build, language, and source identity capture for benchmark reports."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from pathlib import Path

import torch

import forgeir_py


def _command(arguments: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    result = completed.stdout.strip()
    return result or None


def _windows_cpu_model() -> str | None:
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        return str(value).strip()
    except (ImportError, OSError):
        return None


def _linux_cpu_information() -> tuple[str | None, int | None]:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return None, None
    model: str | None = None
    physical_cores: set[tuple[str, str]] = set()
    physical_id: str | None = None
    core_id: str | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines() + [""]:
        if not line:
            if physical_id is not None and core_id is not None:
                physical_cores.add((physical_id, core_id))
            physical_id = None
            core_id = None
            continue
        key, separator, value = line.partition(":")
        if not separator:
            continue
        normalized = key.strip()
        resolved = value.strip()
        if normalized in {"model name", "Hardware"} and model is None:
            model = resolved
        elif normalized == "physical id":
            physical_id = resolved
        elif normalized == "core id":
            core_id = resolved
    return model, len(physical_cores) or None


def _cpu_information() -> tuple[str, int | None]:
    system = platform.system()
    if system == "Windows":
        windows_model = _windows_cpu_model() or platform.processor() or platform.machine()
        physical_text = _command(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum",
            ]
        )
        try:
            physical = int(physical_text) if physical_text is not None else None
        except ValueError:
            physical = None
        return windows_model, physical
    if system == "Linux":
        linux_model, physical = _linux_cpu_information()
        return linux_model or platform.processor() or platform.machine(), physical
    if system == "Darwin":
        darwin_model = _command(["sysctl", "-n", "machdep.cpu.brand_string"])
        physical_text = _command(["sysctl", "-n", "hw.physicalcpu"])
        try:
            physical = int(physical_text) if physical_text is not None else None
        except ValueError:
            physical = None
        return darwin_model or platform.processor() or platform.machine(), physical
    return platform.processor() or platform.machine(), None


def _git_information(repository_root: Path) -> dict[str, object]:
    commit = _command(["git", "rev-parse", "HEAD"], cwd=repository_root)
    status = _command(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=repository_root
    )
    status_text = status or ""
    return {
        "commit": commit or "unavailable",
        "worktree_dirty": bool(status_text),
        "status_sha256": hashlib.sha256(status_text.encode("utf-8")).hexdigest(),
    }


def capture_environment(repository_root: Path) -> dict[str, object]:
    """Capture the host and active ForgeIR/PyTorch build without changing it."""
    cpu_model, physical_cores = _cpu_information()
    build = forgeir_py.doctor()
    return {
        "cpu": {
            "model": cpu_model,
            "physical_core_count": physical_cores,
            "logical_core_count": os.cpu_count(),
        },
        "operating_system": {
            "forgeir_identity": build["operating_system"],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "compiler": build["compiler"],
        "build_type": build["build_type"],
        "forgeir_version": build["forgeir_version"],
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "git": _git_information(repository_root),
        "thread_environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "forgeir_runtime_threads": 1,
            "process_count": 1,
            "cpu_affinity": "unchanged; controlled by the operating-system scheduler",
        },
    }
