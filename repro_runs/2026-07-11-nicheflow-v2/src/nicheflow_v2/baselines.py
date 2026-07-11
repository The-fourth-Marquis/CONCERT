"""Reference-only reconstructions of existing Mainline A methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from .data import TaskPayload
from .splits import TargetSplit


@dataclass(frozen=True)
class PredictionResult:
    prediction: np.ndarray
    method: str
    access_level: str
    calibration: bool
    uses_target_reference: bool
    components: dict[str, Any]


def _reference_arrays(payload: TaskPayload, split: TargetSplit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split.status != "ok":
        raise ValueError(f"Cannot predict with split status {split.status}: {split.reason}")
    idx = split.reference_indices
    return payload.target_expr[idx], payload.target_pos[idx], payload.target_niche[idx]


def _scaled_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    distances = np.sqrt(((query[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2))
    positive = distances[distances > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    return (distances / max(scale, 1e-6)).astype(np.float32)


def _softmax_negative(distance: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = -np.asarray(distance, dtype=np.float64) / max(float(temperature), 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return (weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _knn_prediction(payload: TaskPayload, split: TargetSplit, k: int) -> np.ndarray:
    reference_expr, reference_pos, _ = _reference_arrays(payload, split)
    distances = _scaled_distances(payload.query_pos, reference_pos)
    k_eff = max(1, min(int(k), reference_expr.shape[0]))
    neighbors = np.argsort(distances, axis=1)[:, :k_eff]
    rows = np.arange(payload.query_expr.shape[0])[:, None]
    selected_distances = distances[rows, neighbors]
    inverse = 1.0 / np.maximum(selected_distances, 1e-3)
    weights = inverse / inverse.sum(axis=1, keepdims=True)
    return np.einsum("nk,nkg->ng", weights, reference_expr[neighbors]).astype(np.float32)


def _kernel_prediction(payload: TaskPayload, split: TargetSplit, kernel: str) -> np.ndarray:
    reference_expr, reference_pos, _ = _reference_arrays(payload, split)
    distances = _scaled_distances(payload.query_pos, reference_pos)
    if kernel == "gaussian":
        weights = np.exp(-0.5 * distances**2)
    elif kernel == "cauchy":
        weights = 1.0 / (1.0 + distances**2)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return (weights @ reference_expr).astype(np.float32)


def _graph_diffusion_prediction(payload: TaskPayload, split: TargetSplit, steps: int = 3) -> np.ndarray:
    prediction = _kernel_prediction(payload, split, "gaussian")
    query_distances = _scaled_distances(payload.query_pos, payload.query_pos)
    affinity = np.exp(-0.5 * query_distances**2)
    np.fill_diagonal(affinity, 1.0)
    affinity /= np.maximum(affinity.sum(axis=1, keepdims=True), 1e-12)
    for _ in range(int(steps)):
        prediction = 0.7 * prediction + 0.3 * (affinity @ prediction)
    return prediction.astype(np.float32)


def _linear_covariate_prediction(payload: TaskPayload, split: TargetSplit) -> np.ndarray:
    reference_expr, reference_pos, reference_niche = _reference_arrays(payload, split)
    train = np.column_stack([reference_pos, reference_niche]).astype(np.float32)
    query = np.column_stack([payload.query_pos, payload.query_niche]).astype(np.float32)
    model = Ridge(alpha=1.0)
    model.fit(train, reference_expr)
    return np.asarray(model.predict(query), dtype=np.float32)


def calibrate_to_reference(
    prediction: np.ndarray,
    reference_expr: np.ndarray,
    *,
    strength: float = 0.25,
) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float32)
    reference_expr = np.asarray(reference_expr, dtype=np.float32)
    pred_mean = prediction.mean(axis=0, keepdims=True)
    pred_std = prediction.std(axis=0, keepdims=True)
    ref_mean = reference_expr.mean(axis=0, keepdims=True)
    ref_std = reference_expr.std(axis=0, keepdims=True)
    scaled = (prediction - pred_mean) / np.maximum(pred_std, 1e-4)
    scaled = scaled * np.maximum(ref_std, 1e-4) + ref_mean
    mixed = (1.0 - float(strength)) * prediction + float(strength) * scaled
    lower = np.quantile(reference_expr, 0.01, axis=0, keepdims=True)
    upper = np.quantile(reference_expr, 0.99, axis=0, keepdims=True)
    return np.clip(mixed, lower, upper).astype(np.float32)


def _legacy_residual_prediction(payload: TaskPayload, split: TargetSplit) -> np.ndarray:
    reference_expr, reference_pos, reference_niche = _reference_arrays(payload, split)
    target_to_query = _scaled_distances(reference_pos, payload.query_pos)
    control_anchor = np.argmin(target_to_query, axis=1)
    residual_examples = reference_expr - payload.query_expr[control_anchor]
    query_to_target = _scaled_distances(payload.query_pos, reference_pos)
    context_distance = query_to_target + np.abs(payload.query_niche[:, None] - reference_niche[None, :])
    context_distance += 0.75 * (
        (payload.query_niche[:, None] >= 0.5) != (reference_niche[None, :] >= 0.5)
    )
    weights = _softmax_negative(context_distance, temperature=0.75)
    return (payload.query_expr + weights @ residual_examples).astype(np.float32)


def _adaptive_anchor_name(payload: TaskPayload) -> str:
    task_type = str(payload.task.get("task_type", ""))
    benchmark_kind = str(payload.task.get("benchmark_kind", ""))
    if benchmark_kind == "exact_available" or task_type in {
        "border",
        "periphery",
        "cross_slide",
        "niche_core",
    }:
        return "cauchy"
    return "knn10"


def _anchor_prediction(payload: TaskPayload, split: TargetSplit, name: str) -> np.ndarray:
    if name == "knn5":
        return _knn_prediction(payload, split, 5)
    if name == "knn10":
        return _knn_prediction(payload, split, 10)
    if name in {"gaussian", "cauchy"}:
        return _kernel_prediction(payload, split, name)
    raise ValueError(f"Unknown anchor: {name}")


EXISTING_METHODS: tuple[str, ...] = (
    "source_identity_l0",
    "target_mean_l1",
    "knn5_l1",
    "knn10_l1",
    "gaussian_l1",
    "cauchy_l1",
    "graph_diffusion_l1",
    "linear_covariate_ridge_l1",
    "legacy_nicheflow_lite_l3",
    "legacy_v1_fixed_80_20_l3",
    "legacy_v1_fixed_80_20_calibrated_l3",
)


def predict_existing_method(
    method: str,
    payload: TaskPayload,
    split: TargetSplit,
) -> PredictionResult:
    if method not in EXISTING_METHODS:
        raise ValueError(f"Unknown existing method: {method}")
    reference_expr, _, _ = _reference_arrays(payload, split)
    calibration = False
    uses_reference = method != "source_identity_l0"
    components: dict[str, Any] = {}
    if method == "source_identity_l0":
        prediction = payload.query_expr.copy()
        access_level = "L0"
    elif method == "target_mean_l1":
        prediction = np.repeat(reference_expr.mean(axis=0, keepdims=True), payload.query_expr.shape[0], axis=0)
        access_level = "L1"
    elif method == "knn5_l1":
        prediction = _knn_prediction(payload, split, 5)
        access_level = "L1"
    elif method == "knn10_l1":
        prediction = _knn_prediction(payload, split, 10)
        access_level = "L1"
    elif method == "gaussian_l1":
        prediction = _kernel_prediction(payload, split, "gaussian")
        access_level = "L1"
    elif method == "cauchy_l1":
        prediction = _kernel_prediction(payload, split, "cauchy")
        access_level = "L1"
    elif method == "graph_diffusion_l1":
        prediction = _graph_diffusion_prediction(payload, split)
        access_level = "L1"
    elif method == "linear_covariate_ridge_l1":
        prediction = _linear_covariate_prediction(payload, split)
        access_level = "L1"
    elif method == "legacy_nicheflow_lite_l3":
        prediction = _legacy_residual_prediction(payload, split)
        access_level = "L3"
        components["residual_contract"] = "legacy_full_prediction"
    else:
        anchor_name = _adaptive_anchor_name(payload)
        anchor = _anchor_prediction(payload, split, anchor_name)
        residual_prediction = _legacy_residual_prediction(payload, split)
        prediction = (0.80 * anchor + 0.20 * residual_prediction).astype(np.float32)
        access_level = "L3"
        components.update({"anchor": anchor_name, "anchor_weight": 0.80, "residual_prediction_weight": 0.20})
        if method.endswith("calibrated_l3"):
            prediction = calibrate_to_reference(prediction, reference_expr)
            calibration = True
    return PredictionResult(
        prediction=np.asarray(prediction, dtype=np.float32),
        method=method,
        access_level=access_level,
        calibration=calibration,
        uses_target_reference=uses_reference,
        components=components,
    )
