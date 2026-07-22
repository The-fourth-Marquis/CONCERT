#!/usr/bin/env python3
"""Contracts for shared global shift plus bounded context modulation."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v15.factorized_flow import fit_predict_factorized_flow


def test_zero_cap_preserves_source_pairwise_geometry() -> None:
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
    result = fit_predict_factorized_flow(
        control_expr=control,
        control_context=control_context,
        query_expr=query,
        query_context=query_context,
        reference_expr=reference,
        reference_context=reference_context,
        latent_dim=2,
        hidden_dim=16,
        epochs=20,
        integration_steps=8,
        integration_horizon=0.25,
        modulation_cap=0.0,
        seed=71,
    )
    predicted_pairs = result.prediction[:, None, :] - result.prediction[None, :, :]
    source_pairs = query[:, None, :] - query[None, :, :]
    assert np.allclose(predicted_pairs, source_pairs, atol=1e-5)
    assert result.metadata["max_modulation_norm"] <= 1e-8
    assert result.metadata["output_mode"] == "factorized_global_shift_flow"


def test_modulation_is_hard_capped() -> None:
    rng = np.random.default_rng(3)
    control = rng.normal(size=(6, 4)).astype(np.float32)
    reference = (control + 0.4).astype(np.float32)
    query = rng.normal(size=(5, 4)).astype(np.float32)
    context = rng.normal(size=(6, 2)).astype(np.float32)
    result = fit_predict_factorized_flow(
        control_expr=control,
        control_context=context,
        query_expr=query,
        query_context=rng.normal(size=(5, 2)).astype(np.float32),
        reference_expr=reference,
        reference_context=context,
        latent_dim=3,
        hidden_dim=16,
        epochs=20,
        integration_steps=8,
        integration_horizon=1.0,
        modulation_cap=0.1,
        seed=73,
    )
    assert result.metadata["max_modulation_norm"] <= 0.1 + 1e-6


if __name__ == "__main__":
    test_zero_cap_preserves_source_pairwise_geometry()
    test_modulation_is_hard_capped()
    print("ok")
