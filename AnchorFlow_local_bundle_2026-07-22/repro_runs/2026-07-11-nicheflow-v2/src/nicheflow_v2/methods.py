"""Explicit residual matching and reliability-gated NicheFlow-v2 methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .baselines import calibrate_to_reference, predict_existing_method
from .data import TaskPayload
from .splits import TargetSplit


V2_METHODS: tuple[str, ...] = (
    "anchor_only",
    "residual_prediction_only",
    "fixed_80_20",
    "fixed_80_20_calibrated",
    "nicheflow_v2_gated",
    "nicheflow_v2_gated_calibrated",
)


@dataclass(frozen=True)
class V2PredictionResult:
    prediction: np.ndarray
    method: str
    access_level: str
    calibration: bool
    uses_target_reference: bool
    components: dict[str, Any]
    gate: np.ndarray | None = None


@dataclass(frozen=True)
class _ResidualComponents:
    delta: np.ndarray
    prediction: np.ndarray
    query_target_weights: np.ndarray
    query_target_distance: np.ndarray
    residual_examples: np.ndarray


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lower = values.min(axis=0, keepdims=True)
    span = np.maximum(values.max(axis=0, keepdims=True) - lower, 1e-6)
    return (values - lower) / span


def _position_features(
    payload: TaskPayload,
    split: TargetSplit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_pos = payload.target_pos[split.reference_indices]
    if payload.is_cross_slide:
        source = np.vstack([payload.control_pos, payload.query_pos])
        source_normalized = _minmax(source)
        control_pos = source_normalized[: payload.control_pos.shape[0]]
        query_pos = source_normalized[payload.control_pos.shape[0] :]
        target_pos = _minmax(reference_pos)
    else:
        all_pos = np.vstack([payload.control_pos, payload.query_pos, reference_pos])
        normalized = _minmax(all_pos)
        control_end = payload.control_pos.shape[0]
        query_end = control_end + payload.query_pos.shape[0]
        control_pos = normalized[:control_end]
        query_pos = normalized[control_end:query_end]
        target_pos = normalized[query_end:]
    return control_pos, query_pos, target_pos


def _context_features(pos: np.ndarray, niche: np.ndarray) -> np.ndarray:
    niche = np.asarray(niche, dtype=np.float32)
    boundary_distance = np.abs(niche - 0.5)
    return np.column_stack([pos, niche, boundary_distance]).astype(np.float32)


def _context_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    spatial = np.sqrt(((left[:, None, :2] - right[None, :, :2]) ** 2).sum(axis=2))
    niche = np.abs(left[:, None, 2] - right[None, :, 2])
    side = (left[:, None, 2] >= 0.5) != (right[None, :, 2] >= 0.5)
    return (spatial + niche + 0.75 * side.astype(np.float32)).astype(np.float32)


def _softmax_negative(distance: np.ndarray, temperature: float = 0.75) -> np.ndarray:
    logits = -np.asarray(distance, dtype=np.float64) / max(float(temperature), 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return (weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _build_residual_components(payload: TaskPayload, split: TargetSplit) -> _ResidualComponents:
    if split.status != "ok":
        raise ValueError(f"Cannot predict with split status {split.status}: {split.reason}")
    reference_expr = payload.target_expr[split.reference_indices]
    reference_niche = payload.target_niche[split.reference_indices]
    control_pos, query_pos, target_pos = _position_features(payload, split)
    control_context = _context_features(control_pos, payload.control_niche)
    query_context = _context_features(query_pos, payload.query_niche)
    target_context = _context_features(target_pos, reference_niche)
    target_control_distance = _context_distance(target_context, control_context)
    target_control_weights = _softmax_negative(target_control_distance)
    control_anchor = target_control_weights @ payload.control_expr
    residual_examples = reference_expr - control_anchor
    query_target_distance = _context_distance(query_context, target_context)
    query_target_weights = _softmax_negative(query_target_distance)
    delta = (query_target_weights @ residual_examples).astype(np.float32)
    prediction = (payload.query_expr + delta).astype(np.float32)
    return _ResidualComponents(
        delta=delta,
        prediction=prediction,
        query_target_weights=query_target_weights,
        query_target_distance=query_target_distance,
        residual_examples=residual_examples.astype(np.float32),
    )


def context_matched_delta(payload: TaskPayload, split: TargetSplit) -> np.ndarray:
    """Return a pure context-matched perturbation delta, never source + delta."""
    return _build_residual_components(payload, split).delta


def residual_prediction(payload: TaskPayload, delta_context: np.ndarray) -> np.ndarray:
    delta_context = np.asarray(delta_context, dtype=np.float32)
    if delta_context.shape != payload.query_expr.shape:
        raise ValueError("delta_context must have the same shape as query expression")
    return (payload.query_expr + delta_context).astype(np.float32)


def _anchor_method(payload: TaskPayload) -> str:
    task_type = str(payload.task.get("task_type", ""))
    benchmark_kind = str(payload.task.get("benchmark_kind", ""))
    if benchmark_kind == "exact_available" or task_type in {
        "border",
        "periphery",
        "cross_slide",
        "niche_core",
    }:
        return "cauchy_l1"
    return "knn10_l1"


def _anchor(payload: TaskPayload, split: TargetSplit) -> tuple[np.ndarray, str]:
    method = _anchor_method(payload)
    return predict_existing_method(method, payload, split).prediction, method


def _reliability_gate(
    payload: TaskPayload,
    split: TargetSplit,
    components: _ResidualComponents,
    anchor: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    weights = np.asarray(components.query_target_weights, dtype=np.float64)
    n_reference = weights.shape[1]
    if n_reference <= 1:
        normalized_entropy = np.zeros(weights.shape[0], dtype=np.float64)
    else:
        normalized_entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=1) / np.log(n_reference)
    nearest_distance = components.query_target_distance.min(axis=1).astype(np.float64)
    residual_disagreement = np.sqrt(((anchor - components.prediction) ** 2).mean(axis=1)).astype(np.float64)
    disagreement_scale = max(float(np.median(residual_disagreement[residual_disagreement > 0])) if np.any(residual_disagreement > 0) else 1.0, 1e-6)
    normalized_disagreement = residual_disagreement / disagreement_scale
    boundary_uncertainty = 1.0 - np.clip(2.0 * np.abs(payload.query_niche - 0.5), 0.0, 1.0)
    cross_slide_factor = 0.75 if payload.is_cross_slide else 1.0
    quality = (
        (1.0 - normalized_entropy)
        * np.exp(-nearest_distance)
        * np.exp(-normalized_disagreement)
        * (1.0 - 0.25 * boundary_uncertainty)
        * cross_slide_factor
    )
    gate = np.clip(0.05 + 0.75 * quality, 0.05, 0.80).astype(np.float32)
    diagnostics = {
        "gate_mean": float(gate.mean()),
        "gate_std": float(gate.std()),
        "matching_entropy_mean": float(normalized_entropy.mean()),
        "nearest_match_distance_mean": float(nearest_distance.mean()),
        "anchor_residual_disagreement_mean": float(residual_disagreement.mean()),
        "boundary_uncertainty_mean": float(boundary_uncertainty.mean()),
    }
    return gate, diagnostics


def predict_method(method: str, payload: TaskPayload, split: TargetSplit) -> V2PredictionResult:
    if method not in V2_METHODS:
        raise ValueError(f"Unknown NicheFlow-v2 method: {method}")
    reference_expr = payload.target_expr[split.reference_indices]
    anchor, anchor_method = _anchor(payload, split)
    components = _build_residual_components(payload, split)
    metadata: dict[str, Any] = {
        "anchor_method": anchor_method,
        "residual_contract": "pure_delta_then_source_plus_delta",
    }
    gate: np.ndarray | None = None
    calibration = False
    if method == "anchor_only":
        prediction = anchor
        access_level = "L1"
    elif method == "residual_prediction_only":
        prediction = components.prediction
        access_level = "L3"
    elif method in {"fixed_80_20", "fixed_80_20_calibrated"}:
        prediction = (0.80 * anchor + 0.20 * components.prediction).astype(np.float32)
        metadata.update({"anchor_weight": 0.80, "residual_prediction_weight": 0.20})
        access_level = "L3"
        if method.endswith("calibrated"):
            prediction = calibrate_to_reference(prediction, reference_expr)
            calibration = True
    else:
        gate, diagnostics = _reliability_gate(payload, split, components, anchor)
        metadata.update(diagnostics)
        prediction = ((1.0 - gate[:, None]) * anchor + gate[:, None] * components.prediction).astype(np.float32)
        access_level = "L4"
        if method.endswith("calibrated"):
            prediction = calibrate_to_reference(prediction, reference_expr)
            calibration = True
    return V2PredictionResult(
        prediction=np.asarray(prediction, dtype=np.float32),
        method=method,
        access_level=access_level,
        calibration=calibration,
        uses_target_reference=True,
        components=metadata,
        gate=gate,
    )
