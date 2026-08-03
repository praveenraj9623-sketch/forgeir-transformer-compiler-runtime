"""Real-hardware PyTorch CUDA parity and profiling for ForgeIR CUDA kernels."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as torch_functional

FloatArray = npt.NDArray[np.float32]


class CudaBinding(Protocol):
    MAXIMUM_ROW_WIDTH: int
    KERNEL_BLOCK_SIZE: int

    def device_metadata(self) -> Mapping[str, Any]: ...

    def run_kernel(
        self,
        kernel_name: str,
        input_array: FloatArray,
        auxiliary: FloatArray | None,
        *,
        rows: int,
        width: int,
        epsilon: float,
        warmup_iterations: int,
        measured_iterations: int,
    ) -> Mapping[str, Any]: ...


CUDA_CASES: tuple[tuple[int, int, bool], ...] = (
    (1, 1, False),
    (3, 3, False),
    (5, 7, False),
    (3, 31, False),
    (3, 127, False),
    (3, 257, False),
    (2, 1023, False),
    (2, 4096, False),
    (1, 7, True),
)


def _binding() -> CudaBinding:
    return cast(CudaBinding, importlib.import_module("forgeir_cuda_py"))


def _timing_statistics(samples: Sequence[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("CUDA timing statistics require at least one sample")
    ordered = sorted(float(sample) for sample in samples)
    if any(not math.isfinite(sample) or sample < 0.0 for sample in ordered):
        raise ValueError("CUDA event timing samples must be finite and non-negative")
    return {
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": ordered[(len(ordered) - 1) // 2],
        "standard_deviation_ms": statistics.pstdev(ordered),
    }


def _parity_metrics(actual: FloatArray, expected: FloatArray) -> dict[str, float | int]:
    if actual.shape != expected.shape:
        raise ValueError(f"CUDA output shape {actual.shape} does not match {expected.shape}")
    if not np.array_equal(np.isnan(actual), np.isnan(expected)):
        raise ValueError("CUDA output NaN classifications differ from PyTorch CUDA")
    if not np.array_equal(np.isposinf(actual), np.isposinf(expected)):
        raise ValueError("CUDA output positive-infinity classifications differ from PyTorch CUDA")
    if not np.array_equal(np.isneginf(actual), np.isneginf(expected)):
        raise ValueError("CUDA output negative-infinity classifications differ from PyTorch CUDA")
    finite = np.isfinite(expected)
    finite_actual = actual[finite].astype(np.float64)
    finite_expected = expected[finite].astype(np.float64)
    if finite_expected.size == 0:
        return {"maximum_absolute_error": 0.0, "maximum_relative_error": 0.0, "finite_count": 0}
    absolute = np.abs(finite_actual - finite_expected)
    relative = absolute / np.maximum(np.abs(finite_expected), 1.0e-12)
    return {
        "maximum_absolute_error": float(absolute.max(initial=0.0)),
        "maximum_relative_error": float(relative.max(initial=0.0)),
        "finite_count": int(finite_expected.size),
    }


def _input_array(random: np.random.Generator, rows: int, width: int, extreme: bool) -> FloatArray:
    values = random.normal(0.0, 1.5, size=(rows, width)).astype(np.float32)
    if extreme:
        largest = np.finfo(np.float32).max
        pattern = np.array(
            [largest, largest, -largest, 1000.0, -1000.0, 0.0, 1.0], dtype=np.float32
        )
        values[0, : pattern.size] = pattern
    return np.ascontiguousarray(values)


def _pytorch_reference(
    kernel_name: str, input_tensor: torch.Tensor, auxiliary: torch.Tensor | None, epsilon: float
) -> torch.Tensor:
    if kernel_name == "GELU":
        return torch_functional.gelu(input_tensor, approximate="none")
    if kernel_name == "RMSNorm":
        if auxiliary is None:
            raise ValueError("RMSNorm requires a weight")
        mean_square = input_tensor.square().mean(dim=-1, keepdim=True)
        return input_tensor * torch.rsqrt(mean_square + epsilon) * auxiliary
    if kernel_name == "Softmax":
        return torch.softmax(input_tensor, dim=-1)
    if kernel_name == "FusedBiasGELU":
        if auxiliary is None:
            raise ValueError("FusedBiasGELU requires a bias")
        return torch_functional.gelu(input_tensor + auxiliary, approximate="none")
    raise ValueError(f"unsupported CUDA kernel: {kernel_name}")


def _pytorch_cuda_timings(
    operation: Callable[[], torch.Tensor], warmup: int, iterations: int
) -> tuple[torch.Tensor, list[float]]:
    if warmup <= 0 or iterations <= 0:
        raise ValueError("CUDA timing requires positive warm-up and measured iterations")
    output = operation()
    for _ in range(warmup):
        output = operation()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        stop = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start.record()
        output = operation()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    return output, samples


def _checked_mapping(value: Any, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{role} must be a mapping")
    return cast(Mapping[str, Any], value)


def run_cuda_validation(*, warmup: int, iterations: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch reports no CUDA-capable NVIDIA device")
    if warmup <= 0 or iterations <= 0:
        raise ValueError("warmup and iterations must be positive")
    binding = _binding()
    metadata = dict(binding.device_metadata())
    torch_device_name = torch.cuda.get_device_name(0)
    if metadata.get("gpu_model") != torch_device_name:
        raise RuntimeError(
            "ForgeIR CUDA and PyTorch CUDA selected different devices: "
            f"{metadata.get('gpu_model')!r} versus {torch_device_name!r}"
        )
    torch.manual_seed(42)
    random = np.random.default_rng(42)
    epsilon = 1.0e-5
    absolute_tolerance = 5.0e-6
    relative_tolerance = 5.0e-5
    cases: list[dict[str, Any]] = []
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0

    for rows, width, extreme in CUDA_CASES:
        if width > int(binding.MAXIMUM_ROW_WIDTH):
            raise RuntimeError(f"test width {width} exceeds the compiled CUDA contract")
        input_array = _input_array(random, rows, width, extreme)
        weight = np.ascontiguousarray(random.uniform(0.5, 1.5, size=width).astype(np.float32))
        bias = np.ascontiguousarray(random.normal(0.0, 0.25, size=width).astype(np.float32))
        for kernel_name in ("GELU", "RMSNorm", "Softmax", "FusedBiasGELU"):
            auxiliary_array: FloatArray | None
            if kernel_name == "RMSNorm":
                auxiliary_array = weight
            elif kernel_name == "FusedBiasGELU":
                auxiliary_array = bias
            else:
                auxiliary_array = None
            input_tensor = torch.from_numpy(input_array).to(device="cuda", dtype=torch.float32)
            auxiliary_tensor = (
                None
                if auxiliary_array is None
                else torch.from_numpy(auxiliary_array).to(device="cuda", dtype=torch.float32)
            )
            operation = partial(
                _pytorch_reference, kernel_name, input_tensor, auxiliary_tensor, epsilon
            )
            expected_tensor, torch_timings = _pytorch_cuda_timings(operation, warmup, iterations)
            expected = np.ascontiguousarray(expected_tensor.detach().cpu().numpy())
            raw_forgeir = binding.run_kernel(
                kernel_name,
                input_array,
                auxiliary_array,
                rows=rows,
                width=width,
                epsilon=epsilon,
                warmup_iterations=warmup,
                measured_iterations=iterations,
            )
            forgeir_result = _checked_mapping(raw_forgeir, "ForgeIR CUDA result")
            actual = np.ascontiguousarray(np.asarray(forgeir_result["output"], dtype=np.float32))
            metrics = _parity_metrics(actual, expected)
            max_abs = float(metrics["maximum_absolute_error"])
            max_rel = float(metrics["maximum_relative_error"])
            maximum_absolute_error = max(maximum_absolute_error, max_abs)
            maximum_relative_error = max(maximum_relative_error, max_rel)
            violations = np.abs(actual.astype(np.float64) - expected.astype(np.float64)) > (
                absolute_tolerance + relative_tolerance * np.abs(expected.astype(np.float64))
            )
            violation_count = int(np.count_nonzero(violations & np.isfinite(expected)))
            if violation_count != 0:
                raise RuntimeError(
                    f"{kernel_name} rows={rows} width={width} has {violation_count} "
                    "combined-tolerance violations"
                )
            launch = dict(_checked_mapping(forgeir_result["launch"], "launch metadata"))
            timings = [
                float(value)
                for value in cast(Sequence[float], forgeir_result["kernel_milliseconds"])
            ]
            cases.append(
                {
                    "kernel": kernel_name,
                    "rows": rows,
                    "width": width,
                    "extreme_finite_input": extreme,
                    "parity": {**metrics, "combined_tolerance_violations": violation_count},
                    "forgeir_cuda_event_timing": _timing_statistics(timings),
                    "pytorch_cuda_event_timing": _timing_statistics(torch_timings),
                    "launch": launch,
                }
            )

    metadata["pytorch_device_name"] = torch_device_name
    metadata["pytorch_version"] = torch.__version__
    metadata["pytorch_cuda_build"] = torch.version.cuda
    return {
        "report_schema_version": "1.0",
        "status": "passed",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seed": 42,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "hardware": metadata,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
        "case_count": len(cases),
        "cases": cases,
    }


def write_cuda_validation(output: Path, *, warmup: int, iterations: int) -> dict[str, Any]:
    report = run_cuda_validation(warmup=warmup, iterations=iterations)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real-hardware ForgeIR CUDA parity and CUDA-event profiling"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = write_cuda_validation(
        arguments.output, warmup=arguments.warmup, iterations=arguments.iterations
    )
    summary = {
        "status": report["status"],
        "output": str(arguments.output),
        "gpu_model": cast(Mapping[str, Any], report["hardware"])["gpu_model"],
        "maximum_absolute_error": report["maximum_absolute_error"],
        "maximum_relative_error": report["maximum_relative_error"],
        "case_count": report["case_count"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
