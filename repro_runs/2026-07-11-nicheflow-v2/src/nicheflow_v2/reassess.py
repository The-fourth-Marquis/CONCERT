"""Reassess existing methods under disjoint target reference/evaluation pools."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .baselines import EXISTING_METHODS, predict_existing_method
from .config import ExperimentConfig
from .data import TaskPayload, load_task_payload
from .metrics import (
    boundary_leakage_score,
    de_direction_metrics,
    distance_response_decay_score,
    distribution_metrics,
    perturbation_effect_metrics,
    source_individuality_score,
)
from .splits import TargetSplit, split_target_pool


def _model_family(method: str) -> str:
    if method.startswith("legacy_v1"):
        return "nicheflow_v1"
    if method.startswith("legacy_nicheflow"):
        return "nicheflow_lite"
    return "a2_reference_only_baseline"


def _target_reference_for_queries(
    payload: TaskPayload,
    split: TargetSplit,
) -> tuple[np.ndarray, np.ndarray]:
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


def _nearest_reference_distance(payload: TaskPayload, split: TargetSplit) -> np.ndarray:
    reference_pos = payload.target_pos[split.reference_indices]
    distances = np.sqrt(((payload.query_pos[:, None, :] - reference_pos[None, :, :]) ** 2).sum(axis=2))
    return distances.min(axis=1).astype(np.float32)


def _method_metric_row(
    payload: TaskPayload,
    split: TargetSplit,
    method: str,
) -> tuple[dict[str, Any], bool]:
    result = predict_existing_method(method, payload, split)
    changed = payload.with_target_evaluation_expr(
        split,
        payload.target_expr[split.evaluation_indices] + 1000.0,
    )
    changed_result = predict_existing_method(method, changed, split)
    leakage_check = bool(np.allclose(result.prediction, changed_result.prediction))
    if not leakage_check:
        raise AssertionError(f"Evaluation-expression leakage detected for {payload.task_id}/{method}")
    evaluation_target = payload.target_expr[split.evaluation_indices]
    reference_target = payload.target_expr[split.reference_indices]
    row: dict[str, Any] = {
        "task_id": payload.task_id,
        "benchmark_kind": str(payload.task.get("benchmark_kind", "")),
        "task_type": str(payload.task.get("task_type", "")),
        "source_sample": str(payload.task.get("source_sample", "")),
        "target_sample": str(payload.task.get("target_sample", "")),
        "target_perturbation": str(payload.task.get("target_perturbation", "")),
        "model_family": _model_family(method),
        "method": method,
        "access_level": result.access_level,
        "reference_fraction": split.reference_fraction,
        "n_reference": int(split.reference_indices.size),
        "n_evaluation": int(split.evaluation_indices.size),
        "split_seed": split.seed,
        "split_type": split.split_type,
        "calibration": result.calibration,
        "uses_target_reference": result.uses_target_reference,
        "eligible_for_primary": True,
        "leakage_mutation_check": leakage_check,
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
    target_reference, query_labels = _target_reference_for_queries(payload, split)
    row["boundary_leakage_score"] = boundary_leakage_score(
        result.prediction,
        target_reference,
        query_labels,
    )
    individuality, individuality_reason = source_individuality_score(result.prediction, payload.query_expr)
    row["source_individuality_score"] = individuality
    row["source_individuality_reason"] = individuality_reason
    response = np.sqrt(((result.prediction - payload.query_expr) ** 2).mean(axis=1))
    decay, decay_reason = distance_response_decay_score(
        _nearest_reference_distance(payload, split),
        response,
    )
    row["distance_response_decay_score"] = decay
    row["distance_response_decay_reason"] = decay_reason
    return row, leakage_check


def _coverage_row(
    payload: TaskPayload,
    split: TargetSplit,
    *,
    model_family: str,
    method: str,
    status: str,
    reason: str,
    leakage_check: bool,
) -> dict[str, Any]:
    return {
        "task_id": payload.task_id,
        "model_family": model_family,
        "method": method,
        "reference_fraction": split.reference_fraction,
        "split_seed": split.seed,
        "split_type": split.split_type,
        "eligibility_status": status,
        "reason": reason,
        "eligible_for_primary": status in {"exact", "approximate"},
        "leakage_mutation_check": leakage_check,
    }


def reassess_manifest(
    manifest: pd.DataFrame,
    cfg: ExperimentConfig,
    *,
    include_expensive: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for task in manifest.to_dict(orient="records"):
        payload = load_task_payload(task, cfg)
        for fraction in cfg.reference_fractions:
            for seed in cfg.split_seeds:
                for split_type in cfg.split_types:
                    split = split_target_pool(
                        payload,
                        reference_fraction=fraction,
                        seed=seed,
                        split_type=split_type,
                        min_reference=cfg.min_reference,
                        min_evaluation=cfg.min_evaluation,
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
                            "evaluation_blocks": ";".join(map(str, split.evaluation_blocks)),
                        }
                    )
                    if split.status != "ok":
                        for method in EXISTING_METHODS:
                            coverage_rows.append(
                                _coverage_row(
                                    payload,
                                    split,
                                    model_family=_model_family(method),
                                    method=method,
                                    status="unavailable",
                                    reason=split.reason,
                                    leakage_check=False,
                                )
                            )
                        continue
                    for method in EXISTING_METHODS:
                        row, leakage_check = _method_metric_row(payload, split, method)
                        metric_rows.append(row)
                        status = "approximate" if method.startswith("legacy_") else "exact"
                        coverage_rows.append(
                            _coverage_row(
                                payload,
                                split,
                                model_family=_model_family(method),
                                method=method,
                                status=status,
                                reason="reference-only reconstruction",
                                leakage_check=leakage_check,
                            )
                        )
                    coverage_rows.append(
                        _coverage_row(
                            payload,
                            split,
                            model_family="available_concert",
                            method="available_concert_fixed_fullpool_output",
                            status="not_eligible_for_heldout_primary",
                            reason="fixed output may have used evaluation target rows",
                            leakage_check=False,
                        )
                    )
                    coverage_rows.append(
                        _coverage_row(
                            payload,
                            split,
                            model_family="spatialprop_style_spot_gnn",
                            method="spatialprop_reference_only_retrain",
                            status="unavailable" if not include_expensive else "approximate",
                            reason="excluded_expensive_smoke" if not include_expensive else "expensive runner not connected",
                            leakage_check=False,
                        )
                    )
    return pd.DataFrame(metric_rows), pd.DataFrame(coverage_rows), pd.DataFrame(split_rows)
