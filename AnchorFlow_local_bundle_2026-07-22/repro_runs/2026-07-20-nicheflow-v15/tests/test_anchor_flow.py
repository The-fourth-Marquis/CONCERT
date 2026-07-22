#!/usr/bin/env python3
"""Anchor-initialized residual flow contracts."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v15.anchor_flow import fit_predict_anchor_residual_flow


def test_anchor_flow_starts_near_context_matched_target_and_trains() -> None:
    control_expr = np.array(
        [[0.0, 0.0], [0.5, 0.2], [1.0, 0.8], [1.5, 1.0]],
        dtype=np.float32,
    )
    reference_expr = control_expr + np.array([1.0, -0.5], dtype=np.float32)
    query_expr = control_expr[[1, 2]]
    control_context = np.arange(4, dtype=np.float32)[:, None]
    reference_context = control_context.copy()
    query_context = control_context[[1, 2]]

    result = fit_predict_anchor_residual_flow(
        control_expr=control_expr,
        control_context=control_context,
        query_expr=query_expr,
        query_context=query_context,
        reference_expr=reference_expr,
        reference_context=reference_context,
        latent_dim=2,
        hidden_dim=24,
        epochs=30,
        integration_steps=8,
        anchor_temperature=0.02,
        source_skip_weight=0.0,
        seed=41,
    )

    expected = reference_expr[[1, 2]]
    initial_error = np.mean(np.abs(result.initial_prediction - expected))
    source_error = np.mean(np.abs(query_expr - expected))
    assert initial_error < source_error
    assert result.prediction.shape == query_expr.shape
    assert np.isfinite(result.prediction).all()
    assert result.metadata["output_mode"] == "anchor_initialized_residual_flow"
    assert result.metadata["training_loss_last"] < result.metadata["training_loss_first"]
    assert result.metadata["uses_evaluation_target"] is False


if __name__ == "__main__":
    test_anchor_flow_starts_near_context_matched_target_and_trains()
    print("ok")
