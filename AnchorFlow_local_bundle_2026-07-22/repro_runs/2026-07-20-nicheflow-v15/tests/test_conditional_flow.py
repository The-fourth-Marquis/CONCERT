#!/usr/bin/env python3
"""Trainable conditional residual flow contracts."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v15.flow import fit_predict_conditional_flow


def test_conditional_flow_learns_a_nonzero_velocity_field() -> None:
    control_expr = np.array(
        [[-1.0, -0.5], [-0.5, 0.0], [0.0, 0.5], [0.5, 1.0], [1.0, 1.5]],
        dtype=np.float32,
    )
    reference_expr = np.column_stack(
        [
            control_expr[:, 0] + 0.6 * control_expr[:, 1] ** 2,
            control_expr[:, 1] + 0.4 * control_expr[:, 0],
        ]
    ).astype(np.float32)
    query_expr = np.array(
        [[-0.75, -0.2], [0.25, 0.7]],
        dtype=np.float32,
    )
    control_context = np.column_stack(
        [np.linspace(0.0, 1.0, control_expr.shape[0]), np.zeros(5)]
    ).astype(np.float32)
    reference_context = control_context.copy()
    query_context = np.array(
        [[0.15, 0.0], [0.65, 0.0]],
        dtype=np.float32,
    )

    result = fit_predict_conditional_flow(
        control_expr=control_expr,
        control_context=control_context,
        query_expr=query_expr,
        query_context=query_context,
        reference_expr=reference_expr,
        reference_context=reference_context,
        latent_dim=2,
        hidden_dim=32,
        epochs=40,
        integration_steps=12,
        learning_rate=5e-3,
        seed=29,
    )

    assert result.prediction.shape == query_expr.shape
    assert np.isfinite(result.prediction).all()
    assert result.metadata["output_mode"] == "conditional_residual_flow"
    assert result.metadata["uses_evaluation_target"] is False
    assert result.metadata["training_loss_last"] < result.metadata["training_loss_first"]
    assert not np.allclose(result.prediction, query_expr)


if __name__ == "__main__":
    test_conditional_flow_learns_a_nonzero_velocity_field()
    print("ok")
