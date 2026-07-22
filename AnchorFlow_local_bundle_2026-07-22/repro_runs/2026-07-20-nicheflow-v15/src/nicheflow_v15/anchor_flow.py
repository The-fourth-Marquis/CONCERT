"""Anchor-initialized trainable conditional residual flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .flow import _VelocityNetwork, _covariance
from .model import _low_rank_basis, _matrix, _standardize
from .transport import entropic_sinkhorn


@dataclass(frozen=True)
class AnchorFlowResult:
    prediction: np.ndarray
    initial_prediction: np.ndarray
    coupling: np.ndarray
    metadata: dict[str, Any]

def _resolve_torch_device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    return device


def _anchor_from_context(
    source_context: np.ndarray,
    reference_context: np.ndarray,
    reference_expr: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    cost = (
        (
            source_context[:, None, :]
            - reference_context[None, :, :]
        )
        ** 2
    ).mean(axis=2)
    logits = -cost / max(float(temperature), 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return weights @ reference_expr


def fit_predict_anchor_residual_flow(
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
    device: str | torch.device = "cpu",
    seed: int = 20260720,
) -> AnchorFlowResult:
    """Learn a conditional correction field around a reference-only anchor."""

    control_expr = _matrix("control_expr", control_expr)
    query_expr = _matrix("query_expr", query_expr)
    reference_expr = _matrix("reference_expr", reference_expr)
    control_context = _matrix("control_context", control_context)
    query_context = _matrix("query_context", query_context)
    reference_context = _matrix("reference_context", reference_context)
    if len(
        {
            control_expr.shape[1],
            query_expr.shape[1],
            reference_expr.shape[1],
        }
    ) != 1:
        raise ValueError("expression matrices must share genes")
    if len(
        {
            control_context.shape[1],
            query_context.shape[1],
            reference_context.shape[1],
        }
    ) != 1:
        raise ValueError("context matrices must share features")
    if control_context.shape[0] != control_expr.shape[0]:
        raise ValueError("control context row count differs")
    if query_context.shape[0] != query_expr.shape[0]:
        raise ValueError("query context row count differs")
    if reference_context.shape[0] != reference_expr.shape[0]:
        raise ValueError("reference context row count differs")
    if not 0.0 <= float(source_skip_weight) <= 1.0:
        raise ValueError("source_skip_weight must be between zero and one")
    if float(anchor_temperature) <= 0.0:
        raise ValueError("anchor_temperature must be positive")
    if int(epochs) < 1 or int(integration_steps) < 1:
        raise ValueError("epochs and integration_steps must be positive")
    if not 0.0 <= float(integration_horizon) <= 1.0:
        raise ValueError("integration_horizon must be between zero and one")

    device = _resolve_torch_device(device)
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
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
    query_anchor = _anchor_from_context(
        query_context_z,
        reference_context_z,
        reference_expr,
        temperature=float(anchor_temperature),
    )
    control_initial = (
        (1.0 - float(source_skip_weight)) * control_anchor
        + float(source_skip_weight) * control_expr
    )
    query_initial = (
        (1.0 - float(source_skip_weight)) * query_anchor
        + float(source_skip_weight) * query_expr
    )

    expression_pool = np.vstack([control_initial, reference_expr])
    expression_mean = expression_pool.mean(axis=0, keepdims=True)
    basis = _low_rank_basis(
        expression_pool - expression_mean,
        latent_dim=int(latent_dim),
        seed=int(seed),
    )
    initial_control_latent = (control_initial - expression_mean) @ basis
    initial_query_latent = (query_initial - expression_mean) @ basis
    source_control_latent = (control_expr - expression_mean) @ basis
    source_query_latent = (query_expr - expression_mean) @ basis
    reference_latent = (reference_expr - expression_mean) @ basis
    latent_scale = np.maximum(
        np.vstack([initial_control_latent, reference_latent]).std(
            axis=0,
            keepdims=True,
        ),
        1e-6,
    )
    initial_control_z = initial_control_latent / latent_scale
    initial_query_z = initial_query_latent / latent_scale
    source_control_z = source_control_latent / latent_scale
    source_query_z = source_query_latent / latent_scale
    reference_z = reference_latent / latent_scale

    expression_cost = (
        (
            source_control_z[:, None, :]
            - reference_z[None, :, :]
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
    cost = expression_cost + float(context_weight) * context_cost
    coupling = entropic_sinkhorn(
        cost,
        epsilon=float(sinkhorn_epsilon),
        iterations=int(sinkhorn_iterations),
    )
    conditional = coupling / np.maximum(coupling.sum(axis=1, keepdims=True), 1e-12)
    target_z = conditional @ reference_z
    correction_z = target_z - initial_control_z

    initial = torch.tensor(initial_control_z, dtype=torch.float32, device=device)
    target = torch.tensor(target_z, dtype=torch.float32, device=device)
    correction = torch.tensor(correction_z, dtype=torch.float32, device=device)
    source = torch.tensor(source_control_z, dtype=torch.float32, device=device)
    query_initial_tensor = torch.tensor(
        initial_query_z,
        dtype=torch.float32,
        device=device,
    )
    condition = torch.tensor(
        np.column_stack([source_control_z, control_context_z]),
        dtype=torch.float32,
        device=device,
    )
    query_condition = torch.tensor(
        np.column_stack([source_query_z, query_context_z]),
        dtype=torch.float32,
        device=device,
    )
    target_covariance = _covariance(target)
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
    n_control = int(initial.shape[0])
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        time = torch.rand((n_control, 1), dtype=torch.float32, device=device)
        state = initial + time * correction
        predicted_velocity = model(state, time, condition)
        endpoint = initial + model(
            initial,
            torch.ones((n_control, 1), dtype=torch.float32, device=device),
            condition,
        )
        velocity_loss = torch.nn.functional.smooth_l1_loss(
            predicted_velocity,
            correction,
        )
        endpoint_loss = torch.nn.functional.smooth_l1_loss(endpoint, target)
        covariance_loss = torch.nn.functional.mse_loss(
            _covariance(endpoint),
            target_covariance,
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
        latent = query_initial_tensor.clone()
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
    correction_full = (
        latent.cpu().numpy() - initial_query_z
    ) * latent_scale @ basis.T
    prediction = query_initial + correction_full
    return AnchorFlowResult(
        prediction=prediction.astype(np.float32),
        initial_prediction=query_initial.astype(np.float32),
        coupling=coupling,
        metadata={
            "output_mode": "anchor_initialized_residual_flow",
            "initialization": "reference_context_anchor_with_source_skip",
            "decoder_contract": "anchor_plus_integrated_latent_correction",
            "uses_target_reference": True,
            "uses_evaluation_target": False,
            "calibration": False,
            "latent_standardized": True,
            "latent_dim": int(basis.shape[1]),
            "hidden_dim": int(hidden_dim),
            "anchor_temperature": float(anchor_temperature),
            "source_skip_weight": float(source_skip_weight),
            "epochs": int(epochs),
            "integration_steps": int(integration_steps),
            "integration_horizon": float(integration_horizon),
            "distribution_weight": float(distribution_weight),
            "identity_weight": float(identity_weight),
            "device": str(device),
            "precision": "float32",
            "training_loss_first": float(history[0]),
            "training_loss_last": float(history[-1]),
            "training_loss_min": float(min(history)),
            "mean_correction_norm": float(
                np.sqrt(np.mean(correction_full**2))
            ),
            "n_control": int(control_expr.shape[0]),
            "n_reference": int(reference_expr.shape[0]),
            "n_query": int(query_expr.shape[0]),
            "seed": int(seed),
        },
    )
