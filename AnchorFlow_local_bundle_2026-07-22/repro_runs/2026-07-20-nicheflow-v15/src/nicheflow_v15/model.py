"""Reference-only nonlinear Monge-map prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .transport import entropic_sinkhorn


@dataclass(frozen=True)
class NonlinearTransportResult:
    prediction: np.ndarray
    coupling: np.ndarray
    metadata: dict[str, Any]


def _matrix(name: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def _standardize(
    control: np.ndarray,
    query: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pooled = np.vstack([control, query, reference])
    mean = pooled.mean(axis=0, keepdims=True)
    scale = np.maximum(pooled.std(axis=0, keepdims=True), 1e-6)
    return (
        (control - mean) / scale,
        (query - mean) / scale,
        (reference - mean) / scale,
    )


def _low_rank_basis(
    centered: np.ndarray,
    *,
    latent_dim: int,
    seed: int,
) -> np.ndarray:
    effective_dim = min(
        int(latent_dim),
        centered.shape[1],
        max(1, centered.shape[0] - 1),
    )
    sketch_dim = min(min(centered.shape), effective_dim + 8)
    rng = np.random.default_rng(int(seed))
    projection = rng.normal(
        0.0,
        1.0,
        size=(centered.shape[1], sketch_dim),
    )
    left, _ = np.linalg.qr(centered @ projection, mode="reduced")
    _, _, right = np.linalg.svd(left.T @ centered, full_matrices=False)
    return right[:effective_dim].T


def _random_features(
    values: np.ndarray,
    *,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    hidden = np.tanh(values @ weights + bias)
    return np.column_stack(
        [
            np.ones(values.shape[0], dtype=np.float64),
            values,
            hidden,
        ]
    )


def fit_predict_nonlinear_transport(
    *,
    control_expr: np.ndarray,
    control_context: np.ndarray,
    query_expr: np.ndarray,
    query_context: np.ndarray,
    reference_expr: np.ndarray,
    reference_context: np.ndarray,
    latent_dim: int = 32,
    hidden_dim: int = 64,
    ridge: float = 1e-2,
    context_weight: float = 1.0,
    modulation_strength: float = 1.0,
    sinkhorn_epsilon: float = 0.1,
    sinkhorn_iterations: int = 100,
    seed: int = 20260720,
) -> NonlinearTransportResult:
    """Fit a nonlinear reference-only map and predict target-state queries.

    Target evaluation expression is intentionally absent from this API.
    """

    control_expr = _matrix("control_expr", control_expr)
    query_expr = _matrix("query_expr", query_expr)
    reference_expr = _matrix("reference_expr", reference_expr)
    control_context = _matrix("control_context", control_context)
    query_context = _matrix("query_context", query_context)
    reference_context = _matrix("reference_context", reference_context)
    n_genes = control_expr.shape[1]
    if query_expr.shape[1] != n_genes or reference_expr.shape[1] != n_genes:
        raise ValueError("expression matrices must share the same gene dimension")
    n_context = control_context.shape[1]
    if query_context.shape[1] != n_context or reference_context.shape[1] != n_context:
        raise ValueError("context matrices must share the same feature dimension")
    if control_context.shape[0] != control_expr.shape[0]:
        raise ValueError("control context and expression row counts differ")
    if query_context.shape[0] != query_expr.shape[0]:
        raise ValueError("query context and expression row counts differ")
    if reference_context.shape[0] != reference_expr.shape[0]:
        raise ValueError("reference context and expression row counts differ")
    if int(latent_dim) < 1 or int(hidden_dim) < 1:
        raise ValueError("latent_dim and hidden_dim must be positive")
    if float(ridge) <= 0.0:
        raise ValueError("ridge must be positive")
    if float(context_weight) < 0.0:
        raise ValueError("context_weight must be non-negative")
    if not 0.0 <= float(modulation_strength) <= 1.0:
        raise ValueError("modulation_strength must be between zero and one")

    expression_pool = np.vstack([control_expr, reference_expr])
    expression_mean = expression_pool.mean(axis=0, keepdims=True)
    centered = expression_pool - expression_mean
    basis = _low_rank_basis(
        centered,
        latent_dim=int(latent_dim),
        seed=int(seed),
    )
    effective_latent_dim = basis.shape[1]
    control_latent = (control_expr - expression_mean) @ basis
    query_latent = (query_expr - expression_mean) @ basis
    reference_latent = (reference_expr - expression_mean) @ basis

    latent_scale = np.maximum(
        np.vstack([control_latent, reference_latent]).std(axis=0, keepdims=True),
        1e-6,
    )
    control_latent_z = control_latent / latent_scale
    reference_latent_z = reference_latent / latent_scale
    control_context_z, query_context_z, reference_context_z = _standardize(
        control_context,
        query_context,
        reference_context,
    )
    latent_cost = (
        (
            control_latent_z[:, None, :]
            - reference_latent_z[None, :, :]
        )
        ** 2
    ).mean(axis=2)
    context_cost = (
        (
            control_context_z[:, None, :]
            - reference_context_z[None, :, :]
        )
        ** 2
    ).mean(axis=2)
    cost = latent_cost + float(context_weight) * context_cost
    coupling = entropic_sinkhorn(
        cost,
        epsilon=float(sinkhorn_epsilon),
        iterations=int(sinkhorn_iterations),
    )
    conditional = coupling / np.maximum(coupling.sum(axis=1, keepdims=True), 1e-12)
    barycentric_target = conditional @ reference_latent
    target_delta = barycentric_target - control_latent

    control_features = np.column_stack([control_latent_z, control_context_z])
    query_features = np.column_stack([query_latent / latent_scale, query_context_z])
    rng = np.random.default_rng(int(seed))
    random_weights = rng.normal(
        0.0,
        1.0 / np.sqrt(max(control_features.shape[1], 1)),
        size=(control_features.shape[1], int(hidden_dim)),
    )
    random_bias = rng.uniform(-np.pi, np.pi, size=(int(hidden_dim),))
    design = _random_features(
        control_features,
        weights=random_weights,
        bias=random_bias,
    )
    query_design = _random_features(
        query_features,
        weights=random_weights,
        bias=random_bias,
    )
    gram = design.T @ design
    penalty = float(ridge) * np.eye(gram.shape[0], dtype=np.float64)
    coefficients = np.linalg.solve(
        gram + penalty,
        design.T @ target_delta,
    )
    predicted_latent = query_latent + query_design @ coefficients
    raw_delta = (predicted_latent - query_latent) @ basis.T
    global_shift = (
        reference_expr.mean(axis=0, keepdims=True)
        - control_expr.mean(axis=0, keepdims=True)
    )
    centered_modulation = raw_delta - raw_delta.mean(axis=0, keepdims=True)
    prediction = (
        query_expr
        + global_shift
        + float(modulation_strength) * centered_modulation
    )

    return NonlinearTransportResult(
        prediction=prediction.astype(np.float32),
        coupling=coupling,
        metadata={
            "output_mode": "nonlinear_monge_map",
            "decoder_contract": "intrinsic_identity_plus_global_shift_plus_centered_modulation",
            "uses_target_reference": True,
            "uses_evaluation_target": False,
            "calibration": False,
            "latent_dim": int(effective_latent_dim),
            "hidden_dim": int(hidden_dim),
            "n_control": int(control_expr.shape[0]),
            "n_reference": int(reference_expr.shape[0]),
            "n_query": int(query_expr.shape[0]),
            "sinkhorn_epsilon": float(sinkhorn_epsilon),
            "ridge": float(ridge),
            "context_weight": float(context_weight),
            "modulation_strength": float(modulation_strength),
            "seed": int(seed),
        },
    )
