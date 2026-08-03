"""Transparent latency statistics calculated from retained raw samples."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def _percentile(sorted_samples: Sequence[float], percentile: float) -> float:
    rank = (len(sorted_samples) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_samples[lower])
    fraction = rank - lower
    return float(sorted_samples[lower] * (1.0 - fraction) + sorted_samples[upper] * fraction)


def latency_statistics(samples_microseconds: Sequence[float]) -> dict[str, float | int]:
    """Return population statistics using linear-interpolated percentiles."""
    if not samples_microseconds:
        raise ValueError("latency statistics require at least one sample")
    samples = [float(sample) for sample in samples_microseconds]
    if any(not math.isfinite(sample) or sample < 0.0 for sample in samples):
        raise ValueError("latency samples must be finite and nonnegative")
    ordered = sorted(samples)
    return {
        "sample_count": len(samples),
        "minimum_microseconds": ordered[0],
        "p50_microseconds": _percentile(ordered, 0.50),
        "p95_microseconds": _percentile(ordered, 0.95),
        "maximum_microseconds": ordered[-1],
        "mean_microseconds": statistics.fmean(samples),
        "standard_deviation_microseconds": statistics.pstdev(samples),
    }


def attempted_samples_per_second(samples_microseconds: Sequence[float], batch_size: int) -> float:
    """Calculate attempted input sequences per measured execution second."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    total_microseconds = math.fsum(float(sample) for sample in samples_microseconds)
    if not math.isfinite(total_microseconds) or total_microseconds <= 0.0:
        raise ValueError("total measured latency must be finite and positive")
    return len(samples_microseconds) * batch_size * 1_000_000.0 / total_microseconds
