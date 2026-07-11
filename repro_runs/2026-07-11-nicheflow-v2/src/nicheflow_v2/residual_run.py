"""End-to-end NicheFlow-v2 residual/gating evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import write_artifacts
from .config import ExperimentConfig
from .data import TaskPayload, load_task_payload
from .methods import V2_METHODS, V2PredictionResult, predict_method
from .metrics import (
    boundary_leakage_score,
    de_direction_metrics,
    distance_response_decay_score,
    distribution_metrics,
    perturbation_effect_metrics,
    source_individuality_score,
)
from .splits import TargetSplit, split_target_pool


@dataclass(frozen=True)
class ResidualRunResult:
    metrics: pd.DataFrame
    coverage: pd.DataFrame
    split_manifest: pd.DataFrame
    gate_summary: pd.DataFrame
    outputs: tuple[Path, ...]


def _target_reference(payload: TaskPayload, split: TargetSplit) -> tuple[np.ndarray, np.ndarray]:
    evaluation_expr = payload.target_expr[split.evaluation_indices]
    evaluation_niche = payload.target_niche[split.evaluation_indices]
    query_labels = np.where(payload.query_niche >= 0.5, "tumor_like", "boundary_or_normal_like")
    target_labels = np.where(evaluation_niche >= 0.5, "tumor_like", "boundary_or_normal_like")
    fallback = evaluation_expr.mean(axis=0)
    rows = []
    for label in query_labels:
        mask = target_labels == label
        rows.append(evaluation_expr[mask].mean(axis=0) if mask.any() else fallback)
    return np.asarray(rows, dtype=np.float32), query_labels


def _metric_row(
    payload: TaskPayload,
    split: TargetSplit,
    result: V2PredictionResult,
) -> dict[str, Any]:
    evaluation_target = payload.target_expr[split.evaluation_indices]
    reference_target = payload.target_expr[split.reference_indices]
    row: dict[str, Any] = {
        "task_id": payload.task_id,
        "benchmark_kind": str(payload.task.get("benchmark_kind", "")),
        "task_type": str(payload.task.get("task_type", "")),
        "source_sample": str(payload.task.get("source_sample", "")),
        "target_sample": str(payload.task.get("target_sample", "")),
        "target_perturbation": str(payload.task.get("target_perturbation", "")),
        "model_family": "nicheflow_v2_lightweight",
        "method": result.method,
        "access_level": result.access_level,
        "reference_fraction": split.reference_fraction,
        "n_reference": int(split.reference_indices.size),
        "n_evaluation": int(split.evaluation_indices.size),
        "split_seed": split.seed,
        "split_type": split.split_type,
        "calibration": result.calibration,
        "uses_target_reference": result.uses_target_reference,
        "eligible_for_primary": True,
        "leakage_mutation_check": True,
        "claim_scope": "observed_perturbation_relocation_heldout_target_evaluation",
    }
    row.update(result.components)
    row.update(distribution_metrics(result.prediction, evaluation_target, payload.target_expr))
    row.update(perturbation_effect_metrics(result.prediction, payload.query_expr, evaluation_target))
    predicted_delta = result.prediction.mean(axis=0) - payload.query_expr.mean(axis=0)
    true_delta = evaluation_target.mean(axis=0) - payload.query_expr.mean(axis=0)
    reference_delta = reference_target.mean(axis=0) - payload.query_expr.mean(axis=0)
    row.update(de_direction_metrics(predicted_delta, true_delta, reference_delta, top_k=20))
    row.update(de_direction_metrics(predicted_delta, true_delta, reference_delta, top_k=50))
    target_reference, labels = _target_reference(payload, split)
    row["boundary_leakage_score"] = boundary_leakage_score(result.prediction, target_reference, labels)
    individuality, individuality_reason = source_individuality_score(result.prediction, payload.query_expr)
    row["source_individuality_score"] = individuality
    row["source_individuality_reason"] = individuality_reason
    reference_pos = payload.target_pos[split.reference_indices]
    distances = np.sqrt(((payload.query_pos[:, None, :] - reference_pos[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    response = np.sqrt(((result.prediction - payload.query_expr) ** 2).mean(axis=1))
    decay, decay_reason = distance_response_decay_score(distances, response)
    row["distance_response_decay_score"] = decay
    row["distance_response_decay_reason"] = decay_reason
    if result.gate is not None:
        row["gate_min"] = float(result.gate.min())
        row["gate_max"] = float(result.gate.max())
    return row


def _gate_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    means = metrics.groupby("method", as_index=True).mean(numeric_only=True)

    def difference(left: str, right: str, column: str) -> float:
        return float(means.loc[left, column] - means.loc[right, column])

    rows = [
        {
            "criterion": "gated_uncalibrated_vs_anchor_mean_edistance",
            "value": difference("nicheflow_v2_gated", "anchor_only", "heldout_edistance"),
            "threshold": 0.0,
            "direction": "lower",
        },
        {
            "criterion": "gated_calibrated_vs_anchor_mean_edistance",
            "value": difference("nicheflow_v2_gated_calibrated", "anchor_only", "heldout_edistance"),
            "threshold": 0.0,
            "direction": "lower",
        },
        {
            "criterion": "calibration_effect_on_gated_edistance",
            "value": difference("nicheflow_v2_gated_calibrated", "nicheflow_v2_gated", "heldout_edistance"),
            "threshold": 0.0,
            "direction": "descriptive",
        },
        {
            "criterion": "gated_mean_gate_std",
            "value": float(means.loc["nicheflow_v2_gated", "gate_std"]),
            "threshold": 0.0,
            "direction": "higher",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["passed"] = np.where(
        frame["direction"] == "lower",
        frame["value"] < frame["threshold"],
        np.where(frame["direction"] == "higher", frame["value"] > frame["threshold"], pd.NA),
    )
    frame["n_task_split_rows"] = int(metrics[["task_id", "reference_fraction", "split_seed", "split_type"]].drop_duplicates().shape[0])
    return frame


def run_residual_gating(
    *,
    manifest: pd.DataFrame,
    output_dir: Path,
    config: ExperimentConfig,
) -> ResidualRunResult:
    metric_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for task in manifest.to_dict(orient="records"):
        payload = load_task_payload(task, config)
        for fraction in config.reference_fractions:
            for seed in config.split_seeds:
                for split_type in config.split_types:
                    split = split_target_pool(
                        payload,
                        reference_fraction=fraction,
                        seed=seed,
                        split_type=split_type,
                        min_reference=config.min_reference,
                        min_evaluation=config.min_evaluation,
                    )
                    split_rows.append(
                        {
                            "task_id": payload.task_id,
                            "reference_fraction": fraction,
                            "split_seed": seed,
                            "split_type": split_type,
                            "split_status": split.status,
                            "split_reason": split.reason,
                            "n_reference": int(split.reference_indices.size),
                            "n_evaluation": int(split.evaluation_indices.size),
                        }
                    )
                    if split.status != "ok":
                        for method in V2_METHODS:
                            coverage_rows.append(
                                {
                                    "task_id": payload.task_id,
                                    "model_family": "nicheflow_v2_lightweight",
                                    "method": method,
                                    "eligibility_status": "unavailable",
                                    "reason": split.reason,
                                    "eligible_for_primary": False,
                                    "leakage_mutation_check": False,
                                }
                            )
                        continue
                    altered = payload.with_target_evaluation_expr(
                        split,
                        payload.target_expr[split.evaluation_indices] + 1000.0,
                    )
                    for method in V2_METHODS:
                        result = predict_method(method, payload, split)
                        changed = predict_method(method, altered, split)
                        leakage_check = bool(np.allclose(result.prediction, changed.prediction))
                        if not leakage_check:
                            raise AssertionError(f"Evaluation leakage in {payload.task_id}/{method}")
                        metric_rows.append(_metric_row(payload, split, result))
                        coverage_rows.append(
                            {
                                "task_id": payload.task_id,
                                "model_family": "nicheflow_v2_lightweight",
                                "method": method,
                                "reference_fraction": fraction,
                                "split_seed": seed,
                                "split_type": split_type,
                                "eligibility_status": "exact",
                                "reason": "reference-only v2 reconstruction",
                                "eligible_for_primary": True,
                                "leakage_mutation_check": leakage_check,
                            }
                        )
    metrics = pd.DataFrame(metric_rows)
    coverage = pd.DataFrame(coverage_rows)
    split_manifest = pd.DataFrame(split_rows)
    gate_summary = _gate_summary(metrics)
    outputs = write_artifacts(
        metrics,
        coverage,
        split_manifest,
        output_dir,
        {
            "claim_scope": "observed perturbation relocation with held-out target evaluation",
            "model_family": "nicheflow_v2_lightweight",
            "n_tasks_requested": int(manifest.shape[0]),
        },
        config=config,
    )
    gate_path = Path(output_dir) / "gate_summary.csv"
    gate_summary.to_csv(gate_path, index=False)
    outputs.append(gate_path)
    return ResidualRunResult(
        metrics=metrics,
        coverage=coverage,
        split_manifest=split_manifest,
        gate_summary=gate_summary,
        outputs=tuple(outputs),
    )
