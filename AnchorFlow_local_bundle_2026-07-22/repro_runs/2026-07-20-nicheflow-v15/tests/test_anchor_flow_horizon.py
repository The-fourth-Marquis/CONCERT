#!/usr/bin/env python3
"""Integration horizon controls how far the anchor follows the residual field."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v15.anchor_flow import fit_predict_anchor_residual_flow


def test_zero_integration_horizon_returns_initial_anchor() -> None:
    control = np.array(
        [[0.0, 0.0], [0.5, 0.2], [1.0, 0.8], [1.5, 1.0]],
        dtype=np.float32,
    )
    reference = control + np.array([1.0, -0.5], dtype=np.float32)
    query = control[[1, 2]]
    context = np.arange(4, dtype=np.float32)[:, None]

    result = fit_predict_anchor_residual_flow(
        control_expr=control,
        control_context=context,
        query_expr=query,
        query_context=context[[1, 2]],
        reference_expr=reference,
        reference_context=context,
        latent_dim=2,
        hidden_dim=16,
        epochs=10,
        integration_steps=4,
        integration_horizon=0.0,
        seed=47,
    )

    assert np.allclose(result.prediction, result.initial_prediction)
    assert result.metadata["integration_horizon"] == 0.0


if __name__ == "__main__":
    test_zero_integration_horizon_returns_initial_anchor()
    print("ok")
