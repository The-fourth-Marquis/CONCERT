"""Task-level reference-only prediction wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from .graph import build_niche_adjacency
from .model import NonlinearTransportResult, fit_predict_nonlinear_transport


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lower = values.min(axis=0, keepdims=True)
    span = np.maximum(values.max(axis=0, keepdims=True) - lower, 1e-6)
    return (values - lower) / span


def _context_features(
    position: np.ndarray,
    niche: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    adjacency = build_niche_adjacency(position, niche, k=k)
    neighbor_niche = adjacency @ np.asarray(niche, dtype=np.float32)
    boundary_distance = np.abs(np.asarray(niche, dtype=np.float32) - 0.5)
    return np.column_stack(
        [
            np.asarray(position, dtype=np.float32),
            np.asarray(niche, dtype=np.float32),
            boundary_distance,
            neighbor_niche,
        ]
    ).astype(np.float32)


def _eligible_context(
    payload: Any,
    reference_indices: np.ndarray,
    *,
    graph_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_position = payload.target_pos[reference_indices]
    reference_niche = payload.target_niche[reference_indices]
    if payload.is_cross_slide:
        source_position = _minmax(
            np.vstack([payload.control_pos, payload.query_pos])
        )
        target_position = _minmax(reference_position)
        n_control = payload.control_pos.shape[0]
        control_position = source_position[:n_control]
        query_position = source_position[n_control:]
    else:
        all_position = _minmax(
            np.vstack(
                [
                    payload.control_pos,
                    payload.query_pos,
                    reference_position,
                ]
            )
        )
        control_end = payload.control_pos.shape[0]
        query_end = control_end + payload.query_pos.shape[0]
        control_position = all_position[:control_end]
        query_position = all_position[control_end:query_end]
        target_position = all_position[query_end:]

    source_position = np.vstack([control_position, query_position])
    source_niche = np.concatenate([payload.control_niche, payload.query_niche])
    source_context = _context_features(
        source_position,
        source_niche,
        k=graph_k,
    )
    n_control = payload.control_expr.shape[0]
    control_context = source_context[:n_control]
    query_context = source_context[n_control:]
    reference_context = _context_features(
        target_position,
        reference_niche,
        k=graph_k,
    )
    return control_context, query_context, reference_context


def predict_task_reference_only(
    payload: Any,
    split: Any,
    *,
    graph_k: int = 8,
    latent_dim: int = 32,
    hidden_dim: int = 64,
    ridge: float = 1e-2,
    context_weight: float = 1.0,
    modulation_strength: float = 1.0,
    sinkhorn_epsilon: float = 0.1,
    sinkhorn_iterations: int = 100,
    seed: int = 20260720,
) -> NonlinearTransportResult:
    """Predict a v2 task using target-reference rows and never evaluation rows."""

    if getattr(split, 'status', None) != "ok":
        raise ValueError(
            f"Cannot predict with split status {getattr(split, 'status', None)}"
        )
    reference_indices = np.asarray(split.reference_indices, dtype=int)
    control_context, query_context, reference_context = _eligible_context(
        payload,
        reference_indices,
        graph_k=int(graph_k),
    )
    result = fit_predict_nonlinear_transport(
        control_expr=payload.control_expr,
        control_context=control_context,
        query_expr=payload.query_expr,
        query_context=query_context,
        reference_expr=payload.target_expr[reference_indices],
        reference_context=reference_context,
        latent_dim=int(latent_dim),
        hidden_dim=int(hidden_dim),
        ridge=float(ridge),
        context_weight=float(context_weight),
        modulation_strength=float(modulation_strength),
        sinkhorn_epsilon=float(sinkhorn_epsilon),
        sinkhorn_iterations=int(sinkhorn_iterations),
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
    return NonlinearTransportResult(
        prediction=result.prediction,
        coupling=result.coupling,
        metadata=metadata,
    )
