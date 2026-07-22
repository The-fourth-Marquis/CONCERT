#!/usr/bin/env python3
"""Explicit device-selection contracts for AnchorFlow."""

from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v15.anchor_flow import (
    _resolve_torch_device,
    fit_predict_anchor_residual_flow,
)


def _synthetic_inputs() -> dict[str, np.ndarray]:
    control_expr = np.array(
        [[0.0, 0.0], [0.5, 0.2], [1.0, 0.8], [1.5, 1.0]],
        dtype=np.float32,
    )
    context = np.arange(4, dtype=np.float32)[:, None]
    return {
        "control_expr": control_expr,
        "control_context": context,
        "query_expr": control_expr[[1, 2]],
        "query_context": context[[1, 2]],
        "reference_expr": control_expr + np.array([1.0, -0.5], dtype=np.float32),
        "reference_context": context.copy(),
    }


def test_explicit_cpu_device_is_recorded() -> None:
    result = fit_predict_anchor_residual_flow(
        **_synthetic_inputs(),
        latent_dim=2,
        hidden_dim=8,
        epochs=2,
        integration_steps=2,
        device="cpu",
        seed=41,
    )
    assert result.metadata["device"] == "cpu"
    assert result.metadata["precision"] == "float32"


def test_cuda_request_fails_clearly_when_cuda_is_unavailable() -> None:
    original = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        try:
            _resolve_torch_device("cuda:0")
        except RuntimeError as error:
            assert "CUDA device requested but CUDA is unavailable" in str(error)
        else:
            raise AssertionError("CUDA request unexpectedly succeeded")
    finally:
        torch.cuda.is_available = original


if __name__ == "__main__":
    test_explicit_cpu_device_is_recorded()
    test_cuda_request_fails_clearly_when_cuda_is_unavailable()
    print("ok")
