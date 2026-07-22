#!/usr/bin/env python3
"""Ablation contract for expression-only versus context-aware OT."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v15.model import fit_predict_nonlinear_transport


def test_context_weight_changes_coupling_and_is_reported() -> None:
    control_expr = np.array(
        [[0.0, 0.0], [0.05, 0.0], [1.0, 1.0], [1.05, 1.0]],
        dtype=np.float32,
    )
    reference_expr = np.array(
        [[0.0, 0.0], [0.05, 0.0], [1.0, 1.0], [1.05, 1.0]],
        dtype=np.float32,
    )
    query_expr = np.array([[0.5, 0.5]], dtype=np.float32)
    control_context = np.array(
        [[0.0], [1.0], [0.0], [1.0]],
        dtype=np.float32,
    )
    reference_context = np.array(
        [[1.0], [0.0], [1.0], [0.0]],
        dtype=np.float32,
    )
    query_context = np.array([[0.5]], dtype=np.float32)
    common = dict(
        control_expr=control_expr,
        control_context=control_context,
        query_expr=query_expr,
        query_context=query_context,
        reference_expr=reference_expr,
        reference_context=reference_context,
        latent_dim=2,
        hidden_dim=8,
        sinkhorn_epsilon=0.2,
        seed=3,
    )

    expression_only = fit_predict_nonlinear_transport(
        **common,
        context_weight=0.0,
    )
    context_aware = fit_predict_nonlinear_transport(
        **common,
        context_weight=4.0,
    )

    assert not np.allclose(expression_only.coupling, context_aware.coupling)
    assert expression_only.metadata["context_weight"] == 0.0
    assert context_aware.metadata["context_weight"] == 4.0


if __name__ == "__main__":
    test_context_weight_changes_coupling_and_is_reported()
    print("ok")
