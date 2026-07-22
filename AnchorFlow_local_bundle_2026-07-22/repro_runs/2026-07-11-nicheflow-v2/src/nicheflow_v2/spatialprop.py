"""Reference-only SpatialProp-style spot-GNN comparator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .baselines import calibrate_to_reference
from .data import TaskPayload
from .splits import TargetSplit


@dataclass(frozen=True)
class SpatialPropPrediction:
    prediction: np.ndarray
    access_level: str = "L2"
    calibration: bool = True
    uses_target_reference: bool = True
    backend: str = "torch_spatialprop_style"


class TorchSpotGNN(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim),
            torch.nn.ReLU(),
        )
        self.message = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(features)
        hidden = self.message(adjacency @ hidden)
        return self.decoder(hidden)


def build_knn_adjacency(position: np.ndarray, *, k: int) -> np.ndarray:
    position = np.asarray(position, dtype=np.float32)
    if position.ndim != 2 or position.shape[0] == 0:
        raise ValueError("position must be a non-empty matrix")
    n_rows = int(position.shape[0])
    k_eff = max(1, min(int(k), n_rows - 1)) if n_rows > 1 else 0
    distances = np.sqrt(
        ((position[:, None, :] - position[None, :, :]) ** 2).sum(axis=2)
    )
    positive = distances[distances > 0]
    sigma = max(float(np.median(positive)) if positive.size else 1.0, 1e-6)
    adjacency = np.eye(n_rows, dtype=np.float32)
    if k_eff:
        neighbors = np.argsort(distances, axis=1)[:, 1 : k_eff + 1]
        rows = np.arange(n_rows)[:, None]
        adjacency[rows, neighbors] = np.exp(
            -(distances[rows, neighbors] ** 2) / (2.0 * sigma**2)
        )
    adjacency /= np.maximum(adjacency.sum(axis=1, keepdims=True), 1e-8)
    return adjacency


def neighbor_expression_input(
    expression: np.ndarray,
    adjacency: np.ndarray,
) -> np.ndarray:
    no_self = np.asarray(adjacency, dtype=np.float32).copy()
    np.fill_diagonal(no_self, 0.0)
    row_sum = no_self.sum(axis=1, keepdims=True)
    fallback = row_sum[:, 0] <= 1e-8
    output = (no_self / np.maximum(row_sum, 1e-8)) @ np.asarray(
        expression, dtype=np.float32
    )
    if fallback.any():
        output[fallback] = expression[fallback]
    return output.astype(np.float32)


def predict_spatialprop_reference_only(
    payload: TaskPayload,
    split: TargetSplit,
    *,
    k: int = 8,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    epochs: int = 80,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 20260711,
    device: str = "cpu",
) -> SpatialPropPrediction:
    """Train only on target-reference nodes and predict selected source queries."""
    split.validate(payload.target_expr.shape[0])
    if split.status != "ok":
        raise ValueError(f"Cannot train on split status {split.status}: {split.reason}")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    reference_expr = payload.target_expr[split.reference_indices].astype(np.float32)
    reference_pos = payload.target_pos[split.reference_indices].astype(np.float32)
    query_expr = payload.query_expr.astype(np.float32)
    query_pos = payload.query_pos.astype(np.float32)
    expression = np.vstack([reference_expr, query_expr]).astype(np.float32)
    position = np.vstack([reference_pos, query_pos]).astype(np.float32)
    n_reference = int(reference_expr.shape[0])

    adjacency_np = build_knn_adjacency(position, k=k)
    features_np = neighbor_expression_input(expression, adjacency_np)
    feature_mean = features_np[:n_reference].mean(axis=0, keepdims=True)
    feature_std = np.maximum(
        features_np[:n_reference].std(axis=0, keepdims=True), 1e-6
    )
    target_mean = reference_expr.mean(axis=0, keepdims=True)
    target_std = np.maximum(reference_expr.std(axis=0, keepdims=True), 0.1)
    features_z = (features_np - feature_mean) / feature_std
    target_z = (reference_expr - target_mean) / target_std

    device_obj = torch.device(device)
    features = torch.tensor(features_z, dtype=torch.float32, device=device_obj)
    targets = torch.tensor(target_z, dtype=torch.float32, device=device_obj)
    adjacency = torch.tensor(adjacency_np, dtype=torch.float32, device=device_obj)
    model = TorchSpotGNN(
        input_dim=features.shape[1],
        hidden_dim=int(hidden_dim),
        latent_dim=int(latent_dim),
        output_dim=targets.shape[1],
    ).to(device_obj)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    for _ in range(max(1, int(epochs))):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(features, adjacency)[:n_reference]
        loss = torch.nn.functional.smooth_l1_loss(output, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predicted_z = model(features, adjacency)[n_reference:].cpu().numpy()
    prediction = predicted_z * target_std + target_mean
    prediction = calibrate_to_reference(prediction, reference_expr)
    return SpatialPropPrediction(prediction=prediction.astype(np.float32))
