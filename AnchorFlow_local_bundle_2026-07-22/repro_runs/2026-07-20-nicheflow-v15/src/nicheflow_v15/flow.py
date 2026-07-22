"""Trainable conditional residual flow in a low-dimensional expression latent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .model import _low_rank_basis, _matrix, _standardize
from .transport import entropic_sinkhorn


@dataclass(frozen=True)
class ConditionalFlowResult:
    prediction: np.ndarray
    coupling: np.ndarray
    metadata: dict[str, Any]


class _VelocityNetwork(torch.nn.Module):
    def __init__(self, latent_dim: int, context_dim: int, hidden_dim: int) -> None:
        super().__init__()
        input_dim = int(latent_dim) + int(context_dim) + 1
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, int(hidden_dim)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_dim), int(latent_dim)),
        )

    def forward(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(torch.cat([latent, time, condition], dim=1))


def _covariance(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    denominator = max(int(values.shape[0]) - 1, 1)
    return centered.T @ centered / float(denominator)


def fit_predict_conditional_flow(
    *,
    control_expr: np.ndarray,
    control_context: np.ndarray,
    query_expr: np.ndarray,
    query_context: np.ndarray,
    reference_expr: np.ndarray,
    reference_context: np.ndarray,
    latent_dim: int = 32,
    hidden_dim: int = 64,
    context_weight: float = 1.0,
    sinkhorn_epsilon: float = 0.1,
    sinkhorn_iterations: int = 100,
    epochs: int = 80,
    integration_steps: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    distribution_weight: float = 0.25,
    identity_weight: float = 0.01,
    seed: int = 20260720,
) -> ConditionalFlowResult:
    """Fit and integrate a reference-only conditional residual velocity field."""

    control_expr = _matrix("control_expr", control_expr)
    query_expr = _matrix("query_expr", query_expr)
    reference_expr = _matrix("reference_expr", reference_expr)
    control_context = _matrix("control_context", control_context)
    query_context = _matrix("query_context", query_context)
    reference_context = _matrix("reference_context", reference_context)
    if query_expr.shape[1] != control_expr.shape[1]:
        raise ValueError("query and control expression dimensions differ")
    if reference_expr.shape[1] != control_expr.shape[1]:
        raise ValueError("reference and control expression dimensions differ")
    if control_context.shape[0] != control_expr.shape[0]:
        raise ValueError("control context row count differs")
    if query_context.shape[0] != query_expr.shape[0]:
        raise ValueError("query context row count differs")
    if reference_context.shape[0] != reference_expr.shape[0]:
        raise ValueError("reference context row count differs")
    if control_context.shape[1] != query_context.shape[1]:
        raise ValueError("query and control context dimensions differ")
    if control_context.shape[1] != reference_context.shape[1]:
        raise ValueError("reference and control context dimensions differ")
    if int(epochs) < 1 or int(integration_steps) < 1:
        raise ValueError("epochs and integration_steps must be positive")
    if float(context_weight) < 0.0:
        raise ValueError("context_weight must be non-negative")
    if float(distribution_weight) < 0.0 or float(identity_weight) < 0.0:
        raise ValueError("regularization weights must be non-negative")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    expression_pool = np.vstack([control_expr, reference_expr])
    expression_mean = expression_pool.mean(axis=0, keepdims=True)
    basis = _low_rank_basis(
        expression_pool - expression_mean,
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
    barycentric_target_z = conditional @ reference_latent_z
    target_delta_z = barycentric_target_z - control_latent_z

    device = torch.device("cpu")
    control = torch.tensor(control_latent_z, dtype=torch.float32, device=device)
    target = torch.tensor(barycentric_target_z, dtype=torch.float32, device=device)
    delta = torch.tensor(target_delta_z, dtype=torch.float32, device=device)
    control_z = control
    query = torch.tensor(query_latent / latent_scale, dtype=torch.float32, device=device)
    query_z = query
    control_context_tensor = torch.tensor(
        control_context_z,
        dtype=torch.float32,
        device=device,
    )
    query_context_tensor = torch.tensor(
        query_context_z,
        dtype=torch.float32,
        device=device,
    )
    condition = torch.cat([control_z, control_context_tensor], dim=1)
    query_condition = torch.cat([query_z, query_context_tensor], dim=1)
    target_cov = _covariance(target)

    model = _VelocityNetwork(
        latent_dim=effective_latent_dim,
        context_dim=condition.shape[1],
        hidden_dim=int(hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    loss_history: list[float] = []
    n_control = int(control.shape[0])
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        time = torch.rand((n_control, 1), dtype=torch.float32, device=device)
        interpolated = control + time * delta
        predicted_velocity = model(interpolated, time, condition)
        endpoint = control + model(
            control,
            torch.ones((n_control, 1), dtype=torch.float32, device=device),
            condition,
        )
        velocity_loss = torch.nn.functional.smooth_l1_loss(
            predicted_velocity,
            delta,
        )
        endpoint_loss = torch.nn.functional.smooth_l1_loss(endpoint, target)
        covariance_loss = torch.nn.functional.mse_loss(
            _covariance(endpoint),
            target_cov,
        )
        source_distances = torch.cdist(control, control)
        endpoint_distances = torch.cdist(endpoint, endpoint)
        identity_loss = torch.nn.functional.smooth_l1_loss(
            endpoint_distances,
            source_distances,
        )
        loss = (
            velocity_loss
            + float(distribution_weight) * (endpoint_loss + 0.1 * covariance_loss)
            + float(identity_weight) * identity_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        latent = query.clone()
        step_size = 1.0 / float(integration_steps)
        for step in range(int(integration_steps)):
            time = torch.full(
                (latent.shape[0], 1),
                float(step) * step_size,
                dtype=torch.float32,
                device=device,
            )
            latent = latent + step_size * model(
                latent,
                time,
                query_condition,
            )
    predicted_delta = (
        latent.cpu().numpy() - query_latent / latent_scale
    ) * latent_scale
    prediction = query_expr + predicted_delta @ basis.T
    return ConditionalFlowResult(
        prediction=prediction.astype(np.float32),
        coupling=coupling,
        metadata={
            "output_mode": "conditional_residual_flow",
            "decoder_contract": "source_plus_integrated_latent_delta",
            "latent_standardized": True,
            "uses_target_reference": True,
            "uses_evaluation_target": False,
            "calibration": False,
            "latent_dim": int(effective_latent_dim),
            "hidden_dim": int(hidden_dim),
            "epochs": int(epochs),
            "integration_steps": int(integration_steps),
            "context_weight": float(context_weight),
            "distribution_weight": float(distribution_weight),
            "identity_weight": float(identity_weight),
            "training_loss_first": float(loss_history[0]),
            "training_loss_last": float(loss_history[-1]),
            "training_loss_min": float(min(loss_history)),
            "n_control": int(control_expr.shape[0]),
            "n_reference": int(reference_expr.shape[0]),
            "n_query": int(query_expr.shape[0]),
            "seed": int(seed),
        },
    )
