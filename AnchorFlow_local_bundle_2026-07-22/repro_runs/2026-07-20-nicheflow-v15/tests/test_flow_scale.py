#!/usr/bin/env python3
"""Conditional flow training should be invariant to expression units."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v15.flow import fit_predict_conditional_flow


def test_flow_training_is_stable_under_global_expression_rescaling() -> None:
    control = np.array(
        [[0.0, 0.2], [0.3, 0.5], [0.7, 0.9], [1.0, 1.1]],
        dtype=np.float32,
    )
    reference = control + np.array([0.4, -0.2], dtype=np.float32)
    query = np.array([[0.2, 0.4], [0.8, 1.0]], dtype=np.float32)
    context = np.linspace(0.0, 1.0, 4)[:, None].astype(np.float32)
    query_context = np.array([[0.25], [0.75]], dtype=np.float32)
    settings = {
        "control_context": context,
        "query_context": query_context,
        "reference_context": context,
        "latent_dim": 2,
        "hidden_dim": 16,
        "epochs": 8,
        "integration_steps": 4,
        "seed": 37,
    }

    base = fit_predict_conditional_flow(
        control_expr=control,
        query_expr=query,
        reference_expr=reference,
        **settings,
    )
    scaled = fit_predict_conditional_flow(
        control_expr=100.0 * control,
        query_expr=100.0 * query,
        reference_expr=100.0 * reference,
        **settings,
    )

    assert np.isclose(
        base.metadata["training_loss_first"],
        scaled.metadata["training_loss_first"],
        rtol=1e-4,
    )
    assert np.allclose(scaled.prediction, 100.0 * base.prediction, rtol=1e-4)
    assert base.metadata["latent_standardized"] is True


if __name__ == "__main__":
    test_flow_training_is_stable_under_global_expression_rescaling()
    print("ok")
