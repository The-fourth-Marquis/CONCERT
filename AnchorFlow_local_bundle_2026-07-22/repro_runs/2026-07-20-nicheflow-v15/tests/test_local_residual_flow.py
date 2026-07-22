#!/usr/bin/env python3
"""Local target anchor plus source-context residual contracts."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v15.local_residual_flow import (
    fit_predict_local_residual_flow,
)


def test_constant_target_shift_is_recovered_without_losing_query_geometry() -> None:
    control = np.array(
        [[0.0, 0.0], [0.5, 1.0], [1.0, -0.5], [1.5, 0.25]],
        dtype=np.float32,
    )
    shift = np.array([1.0, -0.75], dtype=np.float32)
    reference = control + shift
    control_context = np.arange(4, dtype=np.float32)[:, None]
    reference_context = control_context.copy()
    query = np.array([[0.25, 2.0], [1.25, -2.0]], dtype=np.float32)
    query_context = np.array([[0.5], [2.5]], dtype=np.float32)

    result = fit_predict_local_residual_flow(
        control_expr=control,
        control_context=control_context,
        query_expr=query,
        query_context=query_context,
        reference_expr=reference,
        reference_context=reference_context,
        latent_dim=2,
        hidden_dim=16,
        epochs=15,
        integration_steps=4,
        integration_horizon=0.0,
        anchor_temperature=0.2,
        modulation_cap=0.5,
        seed=61,
    )

    expected = query + shift
    assert np.allclose(result.base_prediction, expected, atol=1e-5)
    assert np.allclose(result.prediction, expected, atol=1e-5)
    assert result.metadata["output_mode"] == "local_anchor_source_residual_flow"
    assert result.metadata["uses_evaluation_target"] is False


if __name__ == "__main__":
    test_constant_target_shift_is_recovered_without_losing_query_geometry()
    print("ok")
