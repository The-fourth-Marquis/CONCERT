"""Leakage-aware distribution, perturbation-effect, and DE metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _mean_pairwise_distance(left: np.ndarray, right: np.ndarray) -> float:
    distances = np.sqrt(((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=2))
    return float(distances.mean())


def energy_distance(prediction: np.ndarray, target: np.ndarray) -> float:
    """Multivariate sample energy distance; lower is better."""
    prediction = _as_2d(prediction, "prediction")
    target = _as_2d(target, "target")
    if prediction.shape[1] != target.shape[1]:
        raise ValueError("prediction and target must have the same gene dimension")
    value = (
        2.0 * _mean_pairwise_distance(prediction, target)
        - _mean_pairwise_distance(prediction, prediction)
        - _mean_pairwise_distance(target, target)
    )
    return float(max(value, 0.0))


def safe_correlation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    method: str,
) -> tuple[float, str]:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    if left.shape != right.shape:
        raise ValueError("Correlation inputs must have identical shapes")
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 2:
        return float("nan"), "fewer_than_two_valid_values"
    left = left[mask]
    right = right[mask]
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan"), "constant_input"
    if method == "pearson":
        value = float(np.corrcoef(left, right)[0, 1])
    elif method == "spearman":
        value = float(spearmanr(left, right).statistic)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    return float(np.clip(np.round(value, 12), -1.0, 1.0)), ""


def _safe_r2(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, str]:
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    if denominator <= 1e-12:
        return float("nan"), "constant_truth"
    value = 1.0 - float(np.sum((predicted - truth) ** 2)) / denominator
    return value, ""


def perturbation_effect_metrics(
    prediction: np.ndarray,
    source: np.ndarray,
    evaluation_target: np.ndarray,
) -> dict[str, Any]:
    prediction = _as_2d(prediction, "prediction")
    source = _as_2d(source, "source")
    evaluation_target = _as_2d(evaluation_target, "evaluation_target")
    if len({prediction.shape[1], source.shape[1], evaluation_target.shape[1]}) != 1:
        raise ValueError("prediction, source, and evaluation_target must share genes")
    predicted_delta = prediction.mean(axis=0) - source.mean(axis=0)
    true_delta = evaluation_target.mean(axis=0) - source.mean(axis=0)
    error = predicted_delta - true_delta
    pcc, pcc_reason = safe_correlation(predicted_delta, true_delta, method="pearson")
    spearman, spearman_reason = safe_correlation(predicted_delta, true_delta, method="spearman")
    r2, r2_reason = _safe_r2(predicted_delta, true_delta)
    return {
        "delta_mae": float(np.mean(np.abs(error))),
        "delta_rmse": float(np.sqrt(np.mean(error**2))),
        "delta_pcc": pcc,
        "delta_pcc_reason": pcc_reason,
        "delta_spearman": spearman,
        "delta_spearman_reason": spearman_reason,
        "delta_r2": r2,
        "delta_r2_reason": r2_reason,
    }


def select_de_indices(reference_delta: np.ndarray, *, top_k: int) -> np.ndarray:
    delta = np.asarray(reference_delta, dtype=np.float64).ravel()
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    valid = np.flatnonzero(np.isfinite(delta))
    if valid.size == 0:
        return np.array([], dtype=int)
    order = np.argsort(-np.abs(delta[valid]), kind="stable")
    return valid[order[: min(int(top_k), valid.size)]].astype(int)


def de_direction_metrics(
    predicted_delta: np.ndarray,
    true_delta: np.ndarray,
    reference_delta: np.ndarray,
    *,
    top_k: int,
) -> dict[str, Any]:
    predicted_delta = np.asarray(predicted_delta, dtype=np.float64).ravel()
    true_delta = np.asarray(true_delta, dtype=np.float64).ravel()
    reference_delta = np.asarray(reference_delta, dtype=np.float64).ravel()
    if not (predicted_delta.shape == true_delta.shape == reference_delta.shape):
        raise ValueError("DE delta vectors must have identical shapes")
    selected = select_de_indices(reference_delta, top_k=top_k)
    prefix = f"de_top{top_k}"
    if selected.size < 2:
        return {
            f"{prefix}_n_genes": int(selected.size),
            f"{prefix}_mse": float("nan"),
            f"{prefix}_delta_pcc": float("nan"),
            f"{prefix}_delta_pcc_reason": "fewer_than_two_valid_genes",
            f"{prefix}_direction_accuracy": float("nan"),
            f"{prefix}_opposite_direction_fraction": float("nan"),
        }
    predicted = predicted_delta[selected]
    truth = true_delta[selected]
    valid = np.isfinite(predicted) & np.isfinite(truth) & (np.abs(truth) > 1e-12)
    pcc, pcc_reason = safe_correlation(predicted, truth, method="pearson")
    if not valid.any():
        direction_accuracy = float("nan")
        opposite_fraction = float("nan")
    else:
        predicted_sign = np.sign(predicted[valid])
        truth_sign = np.sign(truth[valid])
        direction_accuracy = float(np.mean(predicted_sign == truth_sign))
        opposite_fraction = float(np.mean(predicted_sign == -truth_sign))
    return {
        f"{prefix}_n_genes": int(selected.size),
        f"{prefix}_mse": float(np.mean((predicted - truth) ** 2)),
        f"{prefix}_delta_pcc": pcc,
        f"{prefix}_delta_pcc_reason": pcc_reason,
        f"{prefix}_direction_accuracy": direction_accuracy,
        f"{prefix}_opposite_direction_fraction": opposite_fraction,
    }


def _binary_niche_labels(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    if not np.isfinite(values).all():
        raise ValueError("niche values contain non-finite values")
    return (values >= 0.5).astype(np.int8)


def niche_stratified_energy_distance(
    prediction: np.ndarray,
    prediction_niche: np.ndarray,
    evaluation_target: np.ndarray,
    target_niche: np.ndarray,
) -> tuple[float, int, str]:
    """Target-count-weighted held-out E-distance over shared binary niche strata."""
    prediction = _as_2d(prediction, "prediction")
    evaluation_target = _as_2d(evaluation_target, "evaluation_target")
    pred_labels = _binary_niche_labels(prediction_niche)
    target_labels = _binary_niche_labels(target_niche)
    if prediction.shape[0] != pred_labels.size or evaluation_target.shape[0] != target_labels.size:
        raise ValueError("niche labels must align with expression rows")
    values: list[float] = []
    weights: list[int] = []
    for label in (0, 1):
        pred_mask = pred_labels == label
        target_mask = target_labels == label
        if pred_mask.any() and target_mask.any():
            values.append(energy_distance(prediction[pred_mask], evaluation_target[target_mask]))
            weights.append(int(target_mask.sum()))
    if not values:
        return float("nan"), 0, "no_shared_niche_strata"
    return float(np.average(values, weights=weights)), len(values), ""


def niche_wise_delta_correlation(
    prediction: np.ndarray,
    source: np.ndarray,
    source_niche: np.ndarray,
    evaluation_target: np.ndarray,
    target_niche: np.ndarray,
) -> tuple[float, int, str]:
    """Mean gene-wise delta PCC across shared binary niche strata."""
    prediction = _as_2d(prediction, "prediction")
    source = _as_2d(source, "source")
    evaluation_target = _as_2d(evaluation_target, "evaluation_target")
    source_labels = _binary_niche_labels(source_niche)
    target_labels = _binary_niche_labels(target_niche)
    if prediction.shape != source.shape or source.shape[0] != source_labels.size:
        raise ValueError("prediction, source, and source_niche must align")
    if evaluation_target.shape[0] != target_labels.size:
        raise ValueError("evaluation target and target_niche must align")
    values: list[float] = []
    for label in (0, 1):
        source_mask = source_labels == label
        target_mask = target_labels == label
        if not source_mask.any() or not target_mask.any():
            continue
        predicted_delta = prediction[source_mask].mean(axis=0) - source[source_mask].mean(axis=0)
        true_delta = evaluation_target[target_mask].mean(axis=0) - source[source_mask].mean(axis=0)
        value, _ = safe_correlation(predicted_delta, true_delta, method="pearson")
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), 0, "no_defined_shared_niche_strata"
    return float(np.mean(values)), len(values), ""


def spatial_map_correlation(
    prediction: np.ndarray,
    prediction_pos: np.ndarray,
    evaluation_target: np.ndarray,
    target_pos: np.ndarray,
    selected_genes: np.ndarray,
) -> tuple[float, str]:
    """Nearest-map correlation on reference-selected genes in registered coordinates."""
    prediction = _as_2d(prediction, "prediction")
    evaluation_target = _as_2d(evaluation_target, "evaluation_target")
    prediction_pos = _as_2d(prediction_pos, "prediction_pos")
    target_pos = _as_2d(target_pos, "target_pos")
    genes = np.asarray(selected_genes, dtype=int).ravel()
    if prediction.shape[0] != prediction_pos.shape[0] or evaluation_target.shape[0] != target_pos.shape[0]:
        raise ValueError("expression and position rows must align")
    if prediction_pos.shape[1] != target_pos.shape[1]:
        raise ValueError("prediction and target positions must share dimensions")
    if genes.size == 0:
        return float("nan"), "no_selected_genes"
    distances = np.sqrt(
        ((prediction_pos[:, None, :] - target_pos[None, :, :]) ** 2).sum(axis=2)
    )
    nearest = np.argmin(distances, axis=1)
    observed_map = evaluation_target[nearest][:, genes]
    predicted_map = prediction[:, genes]
    return safe_correlation(predicted_map.ravel(), observed_map.ravel(), method="pearson")


def distribution_metrics(
    prediction: np.ndarray,
    evaluation_target: np.ndarray,
    full_target: np.ndarray,
) -> dict[str, float]:
    return {
        "heldout_edistance": energy_distance(prediction, evaluation_target),
        "fullpool_edistance": energy_distance(prediction, full_target),
    }


def _pairwise_upper_distances(values: np.ndarray) -> np.ndarray:
    values = _as_2d(values, "values")
    if values.shape[0] < 2:
        return np.array([], dtype=np.float64)
    distances = np.sqrt(((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2))
    return distances[np.triu_indices(values.shape[0], k=1)]


def boundary_leakage_score(
    prediction: np.ndarray,
    target_reference: np.ndarray,
    boundary_labels: np.ndarray,
) -> float:
    """Mean absolute deviation from label-stratified target references."""
    prediction = _as_2d(prediction, "prediction")
    target_reference = _as_2d(target_reference, "target_reference")
    labels = np.asarray(boundary_labels).astype(str)
    if prediction.shape != target_reference.shape or prediction.shape[0] != labels.size:
        raise ValueError("prediction, target_reference, and labels must align by row")
    errors: list[float] = []
    for label in np.unique(labels):
        mask = labels == label
        reference = target_reference[mask].mean(axis=0)
        errors.append(float(np.abs(prediction[mask] - reference).mean()))
    return float(np.mean(errors)) if errors else float("nan")


def source_individuality_score(
    prediction: np.ndarray,
    source: np.ndarray,
) -> tuple[float, str]:
    """Correlation between predicted and source pairwise-distance structures."""
    prediction_distances = _pairwise_upper_distances(prediction)
    source_distances = _pairwise_upper_distances(source)
    return safe_correlation(prediction_distances, source_distances, method="pearson")


def distance_response_decay_score(
    distances: np.ndarray,
    response_magnitude: np.ndarray,
) -> tuple[float, str]:
    """Positive values indicate response magnitude decreases with distance."""
    correlation, reason = safe_correlation(distances, response_magnitude, method="pearson")
    return (-correlation if np.isfinite(correlation) else correlation), reason


def cross_niche_drop(
    rows: pd.DataFrame,
    *,
    metric: str = "delta_pcc",
    in_niche_task: str = "niche_core",
    cross_niche_task: str = "cross_niche_core_to_border",
) -> float:
    in_niche = rows.loc[rows["task_type"] == in_niche_task, metric].dropna()
    cross_niche = rows.loc[rows["task_type"] == cross_niche_task, metric].dropna()
    if in_niche.empty or cross_niche.empty:
        return float("nan")
    return float(in_niche.mean() - cross_niche.mean())


METRIC_REGISTRY: dict[str, dict[str, str]] = {
    "heldout_edistance": {"version": "v2.0", "direction": "lower", "scope": "distribution", "causal_proof": "no"},
    "fullpool_edistance": {"version": "v2.0", "direction": "lower", "scope": "compatibility", "causal_proof": "no"},
    "niche_stratified_edistance": {"version": "v2.0", "direction": "lower", "scope": "niche_distribution", "causal_proof": "no"},
    "niche_wise_delta_pcc": {"version": "v2.0", "direction": "higher", "scope": "niche_group_effect", "causal_proof": "no"},
    "spatial_map_pcc": {"version": "v2.0", "direction": "higher", "scope": "registered_spatial_proxy", "causal_proof": "no"},
    "delta_mae": {"version": "v2.0", "direction": "lower", "scope": "group_effect", "causal_proof": "no"},
    "delta_rmse": {"version": "v2.0", "direction": "lower", "scope": "group_effect", "causal_proof": "no"},
    "delta_pcc": {"version": "v2.0", "direction": "higher", "scope": "group_effect", "causal_proof": "no"},
    "delta_spearman": {"version": "v2.0", "direction": "higher", "scope": "group_effect", "causal_proof": "no"},
    "delta_r2": {"version": "v2.0", "direction": "higher", "scope": "group_effect", "causal_proof": "no"},
    "boundary_leakage_score": {"version": "v2.0", "direction": "lower", "scope": "spatial_proxy", "causal_proof": "no"},
    "source_individuality_score": {"version": "v2.0", "direction": "higher", "scope": "structure_proxy", "causal_proof": "no"},
    "distance_response_decay_score": {"version": "v2.0", "direction": "higher", "scope": "spatial_proxy", "causal_proof": "no"},
}
