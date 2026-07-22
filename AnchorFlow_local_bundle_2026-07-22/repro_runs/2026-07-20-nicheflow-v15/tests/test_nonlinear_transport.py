#!/usr/bin/env python3
"""Tests for the first nonlinear reference-only transport prototype."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v15.model import fit_predict_nonlinear_transport
from nicheflow_v15.predict import predict_task_reference_only


def test_nonlinear_transport_returns_finite_nonidentity_predictions() -> None:
    control_expr = np.array(
        [[0.0, 0.0], [0.2, 0.1], [0.8, 0.7], [1.0, 1.0]],
        dtype=np.float32,
    )
    reference_expr = np.array(
        [[0.0, 0.1], [0.25, 0.15], [1.45, 0.9], [2.0, 1.2]],
        dtype=np.float32,
    )
    query_expr = np.array([[0.4, 0.3], [0.9, 0.8]], dtype=np.float32)
    control_context = np.column_stack(
        [np.linspace(0.0, 1.0, 4), np.array([0.0, 0.0, 1.0, 1.0])]
    ).astype(np.float32)
    reference_context = control_context.copy()
    query_context = np.array([[0.4, 0.0], [0.9, 1.0]], dtype=np.float32)

    result = fit_predict_nonlinear_transport(
        control_expr=control_expr,
        control_context=control_context,
        query_expr=query_expr,
        query_context=query_context,
        reference_expr=reference_expr,
        reference_context=reference_context,
        latent_dim=2,
        hidden_dim=16,
        seed=7,
    )

    assert result.prediction.shape == query_expr.shape
    assert np.isfinite(result.prediction).all()
    assert not np.allclose(result.prediction, query_expr)
    assert result.metadata["output_mode"] == "nonlinear_monge_map"
    assert result.metadata["uses_evaluation_target"] is False


def test_task_prediction_is_invariant_to_evaluation_expression_mutation() -> None:
    v2_root = ROOT.parent / "2026-07-11-nicheflow-v2" / "src"
    sys.path.insert(0, str(v2_root))
    from nicheflow_v2.data import load_manifest, load_task_payload
    from nicheflow_v2.splits import split_target_pool

    payload = load_task_payload(load_manifest().iloc[0])
    split = split_target_pool(
        payload,
        reference_fraction=0.70,
        seed=20260711,
        split_type="random_stratified",
    )
    altered = payload.with_target_evaluation_expr(
        split,
        payload.target_expr[split.evaluation_indices] + 1000.0,
    )
    settings = {"latent_dim": 4, "hidden_dim": 8, "seed": 11}
    original = predict_task_reference_only(payload, split, **settings)
    changed = predict_task_reference_only(altered, split, **settings)

    assert np.allclose(original.prediction, changed.prediction)
    assert original.metadata["n_reference"] == split.reference_indices.size
    assert original.metadata["uses_evaluation_target"] is False


if __name__ == "__main__":
    test_nonlinear_transport_returns_finite_nonidentity_predictions()
    test_task_prediction_is_invariant_to_evaluation_expression_mutation()
    print("ok")
