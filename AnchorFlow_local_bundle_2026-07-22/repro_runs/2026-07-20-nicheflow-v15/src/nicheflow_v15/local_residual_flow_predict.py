"""Task wrapper for local anchor plus source-context residual flow."""

from __future__ import annotations

from typing import Any

import numpy as np

from .local_residual_flow import (
    LocalResidualFlowResult,
    fit_predict_local_residual_flow,
)
from .predict import _eligible_context


def predict_task_local_residual_flow(
    payload: Any,
    split: Any,
    *,
    graph_k: int = 8,
    latent_dim: int = 16,
    hidden_dim: int = 64,
    context_weight: float = 1.0,
    anchor_temperature: float = 0.25,
    sinkhorn_epsilon: float = 0.1,
    sinkhorn_iterations: int = 100,
    epochs: int = 80,
    integration_steps: int = 16,
    integration_horizon: float = 0.25,
    modulation_cap: float = 0.5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    distribution_weight: float = 0.25,
    identity_weight: float = 0.05,
    seed: int = 20260720,
) -> LocalResidualFlowResult:
    if getattr(split, "status", None) != "ok":
        raise ValueError(
            f"Cannot predict with split status {getattr(split, 'status', None)}"
        )
    reference_indices = np.asarray(split.reference_indices, dtype=int)
    control_context, query_context, reference_context = _eligible_context(
        payload,
        reference_indices,
        graph_k=int(graph_k),
    )
    control_lookup = {
        int(global_index): row
        for row, global_index in enumerate(payload.control_indices)
    }
    query_control_indices = np.asarray(
        [control_lookup[int(index)] for index in payload.query_indices],
        dtype=int,
    )
    result = fit_predict_local_residual_flow(
        control_expr=payload.control_expr,
        control_context=control_context,
        query_expr=payload.query_expr,
        query_context=query_context,
        reference_expr=payload.target_expr[reference_indices],
        reference_context=reference_context,
        query_control_indices=query_control_indices,
        latent_dim=int(latent_dim),
        hidden_dim=int(hidden_dim),
        context_weight=float(context_weight),
        anchor_temperature=float(anchor_temperature),
        sinkhorn_epsilon=float(sinkhorn_epsilon),
        sinkhorn_iterations=int(sinkhorn_iterations),
        epochs=int(epochs),
        integration_steps=int(integration_steps),
        integration_horizon=float(integration_horizon),
        modulation_cap=float(modulation_cap),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        distribution_weight=float(distribution_weight),
        identity_weight=float(identity_weight),
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
            "query_self_exclusions": int(query_control_indices.size),
        }
    )
    return LocalResidualFlowResult(
        prediction=result.prediction,
        base_prediction=result.base_prediction,
        coupling=result.coupling,
        metadata=metadata,
    )
