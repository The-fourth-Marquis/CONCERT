"""Leakage-aware distribution, perturbation-effect, and DE metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
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


def distribution_metrics(
    prediction: np.ndarray,
    evaluation_target: np.ndarray,
    full_target: np.ndarray,
) -> dict[str, float]:
    return {
        "heldout_edistance": energy_distance(prediction, evaluation_target),
        "fullpool_edistance": energy_distance(prediction, full_target),
    }


METRIC_REGISTRY: dict[str, dict[str, str]] = {
    "heldout_edistance": {"version": "v2.0", "direction": "lower", "scope": "distribution"},
    "fullpool_edistance": {"version": "v2.0", "direction": "lower", "scope": "compatibility"},
    "delta_mae": {"version": "v2.0", "direction": "lower", "scope": "group_effect"},
    "delta_rmse": {"version": "v2.0", "direction": "lower", "scope": "group_effect"},
    "delta_pcc": {"version": "v2.0", "direction": "higher", "scope": "group_effect"},
    "delta_spearman": {"version": "v2.0", "direction": "higher", "scope": "group_effect"},
    "delta_r2": {"version": "v2.0", "direction": "higher", "scope": "group_effect"},
}
