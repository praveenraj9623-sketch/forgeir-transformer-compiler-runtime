from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forgeir.benchmark import cuda_validation


def test_cuda_case_matrix_covers_required_width_categories() -> None:
    widths = [width for _, width, _ in cuda_validation.CUDA_CASES]
    assert 1 in widths
    assert any(width % 2 == 1 for width in widths)
    assert max(widths) >= 4096
    assert any(extreme for _, _, extreme in cuda_validation.CUDA_CASES)


def test_parity_metrics_report_exact_arrays() -> None:
    values = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
    metrics = cuda_validation._parity_metrics(values, values.copy())
    assert metrics == {
        "maximum_absolute_error": 0.0,
        "maximum_relative_error": 0.0,
        "finite_count": 3,
    }


def test_no_result_is_written_without_cuda(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "must_not_exist.json"
    monkeypatch.setattr(cuda_validation.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA-capable NVIDIA device"):
        cuda_validation.write_cuda_validation(output, warmup=1, iterations=1)
    assert not output.exists()
