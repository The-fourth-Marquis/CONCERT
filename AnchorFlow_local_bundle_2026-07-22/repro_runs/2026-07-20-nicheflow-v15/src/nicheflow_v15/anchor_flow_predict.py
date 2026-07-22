"""Task wrapper for anchor-initialized residual flow."""

from __future__ import annotations

from typing import Any

import numpy as np

from .anchor_flow import AnchorFlowResult, fit_predict_anchor_residual_flow
from .predict import _eligible_context


def predict_task_anchor_residual_flow(
    payload: Any,
    split: Any,
    *,
    graph_k: int = 8,
    latent_dim: int = 16,
    hidden_dim: int = 64,
    context_weight: float = 1.0,
    anchor_temperature: float = 0.25,
    source_skip_weight: float = 0.20,
    sinkhorn_epsilon: float = 0.1,
    sinkhorn_iterations: int = 100,
    epochs: int = 80,
    integration_steps: int = 16,
    integration_horizon: float = 1.0,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    distribution_weight: float = 0.25,
    identity_weight: float = 0.05,
    device: str = "cpu",
    seed: int = 20260720,
) -> AnchorFlowResult:
    if getattr(split, "status", None) != "ok":
        raise ValueError(
            f"Cannot predict with split status {getattr(split, status, None)}"
        )
    reference_indices = np.asarray(split.reference_indices, dtype=int)
    control_context, query_context, reference_context = _eligible_context(
        payload,
        reference_indices,
        graph_k=int(graph_k),
    )
    result = fit_predict_anchor_residual_flow(
        control_expr=payload.control_expr,
        control_context=control_context,
        query_expr=payload.query_expr,
        query_context=query_context,
        reference_expr=payload.target_expr[reference_indices],
        reference_context=reference_context,
        latent_dim=int(latent_dim),
        hidden_dim=int(hidden_dim),
        context_weight=float(context_weight),
        anchor_temperature=float(anchor_temperature),
        source_skip_weight=float(source_skip_weight),
        sinkhorn_epsilon=float(sinkhorn_epsilon),
        sinkhorn_iterations=int(sinkhorn_iterations),
        epochs=int(epochs),
        integration_steps=int(integration_steps),
        integration_horizon=float(integration_horizon),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        distribution_weight=float(distribution_weight),
        identity_weight=float(identity_weight),
        device=str(device),
        seed=int(seed),
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "task_id": str(payload.task_id),
            "access_level": "L3",
            "reference_fraction": float(split.reference_fraction),
            "split_seed": int(split.seed),
            "split_type": str(split.split_type),
            "graph_k": int(graph_k),
        }
    )
    return AnchorFlowResult(
        prediction=result.prediction,
        initial_prediction=result.initial_prediction,
        coupling=result.coupling,
        metadata=metadata,
    )
