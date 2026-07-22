#!/usr/bin/env python3
"""Structured decoder contracts for source identity and bounded modulation."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v15.structured_flow import fit_predict_structured_flow


def test_zero_modulation_preserves_source_pairwise_geometry() -> None:
    control = np.array(
        [[0.0, 0.0], [0.3, 1.0], [0.8, -1.0], [1.0, 0.5]],
        dtype=np.float32,
    )
    shift = np.array([1.0, -0.5], dtype=np.float32)
    reference = control + shift
    query = np.array(
        [[0.2, 3.0], [0.7, -2.0], [1.2, 0.25]],
        dtype=np.float32,
    )
    control_context = np.linspace(0.0, 1.0, 4)[:, None].astype(np.float32)
    reference_context = control_context.copy()
    query_context = np.array([[0.1], [0.5], [0.9]], dtype=np.float32)

    result = fit_predict_structured_flow(
        control_expr=control,
        control_context=control_context,
        query_expr=query,
        query_context=query_context,
        reference_expr=reference,
        reference_context=reference_context,
        latent_dim=2,
        hidden_dim=16,
        epochs=25,
        integration_steps=8,
        integration_horizon=0.0,
        modulation_cap=0.5,
        seed=53,
    )

    predicted_pairs = result.prediction[:, None, :] - result.prediction[None, :, :]
    source_pairs = query[:, None, :] - query[None, :, :]
    assert np.allclose(predicted_pairs, source_pairs, atol=1e-5)
    assert np.allclose(
        result.prediction.mean(axis=0),
        query.mean(axis=0) + shift,
        atol=0.25,
    )
    assert result.metadata["output_mode"] == "structured_source_shift_flow"
    assert result.metadata["uses_evaluation_target"] is False


if __name__ == "__main__":
    test_zero_modulation_preserves_source_pairwise_geometry()
    print("ok")
