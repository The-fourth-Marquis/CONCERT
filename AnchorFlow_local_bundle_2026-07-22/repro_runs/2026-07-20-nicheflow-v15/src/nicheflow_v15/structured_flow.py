"""Source-identity structured conditional residual flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .anchor_flow import _anchor_from_context
from .flow import _VelocityNetwork, _covariance
from .model import _low_rank_basis, _matrix, _standardize
from .transport import entropic_sinkhorn


@dataclass(frozen=True)
class StructuredFlowResult:
    prediction: np.ndarray
    coupling: np.ndarray
    metadata: dict[str, Any]


def fit_predict_structured_flow(
    *,
    control_expr: np.ndarray,
    control_context: np.ndarray,
    query_expr: np.ndarray,
    query_context: np.ndarray,
    reference_expr: np.ndarray,
    reference_context: np.ndarray,
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
) -> StructuredFlowResult:
    """Train flow modulation while keeping source identity as the main path."""

    control_expr = _matrix("control_expr", control_expr)
    query_expr = _matrix("query_expr", query_expr)
    reference_expr = _matrix("reference_expr", reference_expr)
    control_context = _matrix("control_context", control_context)
    query_context = _matrix("query_context", query_context)
    reference_context = _matrix("reference_context", reference_context)
    if len({control_expr.shape[1], query_expr.shape[1], reference_expr.shape[1]}) != 1:
        raise ValueError("expression matrices must share genes")
    if len({control_context.shape[1], query_context.shape[1], reference_context.shape[1]}) != 1:
        raise ValueError("context matrices must share features")
    if control_context.shape[0] != control_expr.shape[0]:
        raise ValueError("control context row count differs")
    if query_context.shape[0] != query_expr.shape[0]:
        raise ValueError("query context row count differs")
    if reference_context.shape[0] != reference_expr.shape[0]:
        raise ValueError("reference context row count differs")
    if int(epochs) < 1 or int(integration_steps) < 1:
        raise ValueError("epochs and integration_steps must be positive")
    if not 0.0 <= float(integration_horizon) <= 1.0:
        raise ValueError("integration_horizon must be between zero and one")
    if float(modulation_cap) < 0.0:
        raise ValueError("modulation_cap must be non-negative")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    control_context_z, query_context_z, reference_context_z = _standardize(
        control_context,
        query_context,
        reference_context,
    )
    control_anchor = _anchor_from_context(
        control_context_z,
        reference_context_z,
        reference_expr,
        temperature=float(anchor_temperature),
    )
    global_shift = (
        control_anchor.mean(axis=0, keepdims=True)
        - control_expr.mean(axis=0, keepdims=True)
    )
    base_control = control_expr + global_shift
    base_query = query_expr + global_shift
    expression_pool = np.vstack([base_control, reference_expr])
    expression_mean = expression_pool.mean(axis=0, keepdims=True)
    basis = _low_rank_basis(
        expression_pool - expression_mean,
        latent_dim=int(latent_dim),
        seed=int(seed),
    )
    source_control_latent = (control_expr - expression_mean) @ basis
    source_query_latent = (query_expr - expression_mean) @ basis
    base_control_latent = (base_control - expression_mean) @ basis
    base_query_latent = (base_query - expression_mean) @ basis
    reference_latent = (reference_expr - expression_mean) @ basis
    latent_scale = np.maximum(
        np.vstack([base_control_latent, reference_latent]).std(
            axis=0,
            keepdims=True,
        ),
        1e-6,
    )
    source_control_z = source_control_latent / latent_scale
    source_query_z = source_query_latent / latent_scale
    base_control_z = base_control_latent / latent_scale
    base_query_z = base_query_latent / latent_scale
    reference_z = reference_latent / latent_scale

    expression_cost = (
        (source_control_z[:, None, :] - reference_z[None, :, :]) ** 2
    ).mean(axis=2)
    context_cost = (
        (control_context_z[:, None, :] - reference_context_z[None, :, :]) ** 2
    ).mean(axis=2)
    coupling = entropic_sinkhorn(
        expression_cost + float(context_weight) * context_cost,
        epsilon=float(sinkhorn_epsilon),
        iterations=int(sinkhorn_iterations),
    )
    conditional = coupling / np.maximum(coupling.sum(axis=1, keepdims=True), 1e-12)
    target_z = conditional @ reference_z
    correction = target_z - base_control_z
    correction = correction - correction.mean(axis=0, keepdims=True)

    device = torch.device("cpu")
    base = torch.tensor(base_control_z, dtype=torch.float32, device=device)
    target = torch.tensor(
        base_control_z + correction,
        dtype=torch.float32,
        device=device,
    )
    correction_tensor = torch.tensor(correction, dtype=torch.float32, device=device)
    source = torch.tensor(source_control_z, dtype=torch.float32, device=device)
    condition = torch.tensor(
        np.column_stack([source_control_z, control_context_z]),
        dtype=torch.float32,
        device=device,
    )
    query_state = torch.tensor(base_query_z, dtype=torch.float32, device=device)
    query_condition = torch.tensor(
        np.column_stack([source_query_z, query_context_z]),
        dtype=torch.float32,
        device=device,
    )
    model = _VelocityNetwork(
        latent_dim=basis.shape[1],
        context_dim=condition.shape[1],
        hidden_dim=int(hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    history: list[float] = []
    n_control = int(base.shape[0])
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        time = torch.rand((n_control, 1), dtype=torch.float32, device=device)
        state = base + time * correction_tensor
        velocity = model(state, time, condition)
        endpoint = base + model(
            base,
            torch.ones((n_control, 1), dtype=torch.float32, device=device),
            condition,
        )
        endpoint_correction = endpoint - base
        endpoint_correction = endpoint_correction - endpoint_correction.mean(
            dim=0,
            keepdim=True,
        )
        velocity_loss = torch.nn.functional.smooth_l1_loss(
            velocity,
            correction_tensor,
        )
        endpoint_loss = torch.nn.functional.smooth_l1_loss(endpoint, target)
        covariance_loss = torch.nn.functional.mse_loss(
            _covariance(endpoint_correction),
            _covariance(correction_tensor),
        )
        identity_loss = torch.nn.functional.smooth_l1_loss(
            torch.cdist(endpoint, endpoint),
            torch.cdist(source, source),
        )
        loss = (
            velocity_loss
            + float(distribution_weight) * (endpoint_loss + 0.1 * covariance_loss)
            + float(identity_weight) * identity_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        history.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        latent = query_state.clone()
        step_size = float(integration_horizon) / float(integration_steps)
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
    modulation = latent.cpu().numpy() - base_query_z
    modulation = modulation - modulation.mean(axis=0, keepdims=True)
    if float(modulation_cap) > 0.0:
        norm = np.sqrt(np.mean(modulation**2, axis=1, keepdims=True))
        modulation *= np.minimum(
            1.0,
            float(modulation_cap) / np.maximum(norm, 1e-8),
        )
    else:
        modulation[:] = 0.0
    prediction = query_expr + global_shift + (
        modulation * latent_scale
    ) @ basis.T
    return StructuredFlowResult(
        prediction=prediction.astype(np.float32),
        coupling=coupling,
        metadata={
            "output_mode": "structured_source_shift_flow",
            "decoder_contract": "source_identity_plus_global_shift_plus_bounded_centered_modulation",
            "uses_target_reference": True,
            "uses_evaluation_target": False,
            "latent_standardized": True,
            "integration_steps": int(integration_steps),
            "integration_horizon": float(integration_horizon),
            "modulation_cap": float(modulation_cap),
            "context_weight": float(context_weight),
            "anchor_temperature": float(anchor_temperature),
            "distribution_weight": float(distribution_weight),
            "identity_weight": float(identity_weight),
            "training_loss_first": float(history[0]),
            "training_loss_last": float(history[-1]),
            "training_loss_min": float(min(history)),
            "mean_global_shift": float(np.sqrt(np.mean(global_shift**2))),
            "mean_modulation_norm": float(np.sqrt(np.mean(modulation**2))),
            "n_control": int(control_expr.shape[0]),
            "n_reference": int(reference_expr.shape[0]),
            "n_query": int(query_expr.shape[0]),
            "seed": int(seed),
        },
    )
