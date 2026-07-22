"""Reassess existing methods under disjoint target reference/evaluation pools."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .baselines import EXISTING_METHODS, PredictionResult, predict_existing_method
from .config import ExperimentConfig, assert_writable_output
from .data import TaskPayload, load_task_payload
from .metrics import (
    boundary_leakage_score,
    de_direction_metrics,
    distance_response_decay_score,
    distribution_metrics,
    niche_stratified_energy_distance,
    niche_wise_delta_correlation,
    perturbation_effect_metrics,
    select_de_indices,
    source_individuality_score,
    spatial_map_correlation,
)
from .spatialprop import predict_spatialprop_reference_only
from .splits import TargetSplit, split_target_pool


def _model_family(method: str) -> str:
    if method.startswith("spatialprop_"):
        return "spatialprop_style_spot_gnn"
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
    *,
    result_override: PredictionResult | None = None,
    leakage_check_override: bool | None = None,
) -> tuple[dict[str, Any], bool]:
    if result_override is None:
        result = predict_existing_method(method, payload, split)
        changed = payload.with_target_evaluation_expr(
            split,
            payload.target_expr[split.evaluation_indices] + 1000.0,
        )
        changed_result = predict_existing_method(method, changed, split)
        leakage_check = bool(np.allclose(result.prediction, changed_result.prediction))
    else:
        result = result_override
        if leakage_check_override is None:
            raise ValueError("result_override requires leakage_check_override")
        leakage_check = bool(leakage_check_override)
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
    niche_edistance, niche_count, niche_reason = niche_stratified_energy_distance(
        result.prediction,
        payload.query_niche,
        evaluation_target,
        payload.target_niche[split.evaluation_indices],
    )
    row["niche_stratified_edistance"] = niche_edistance
    row["niche_stratified_n_strata"] = niche_count
    row["niche_stratified_reason"] = niche_reason
    row.update(perturbation_effect_metrics(result.prediction, payload.query_expr, evaluation_target))
    niche_delta, niche_delta_count, niche_delta_reason = niche_wise_delta_correlation(
        result.prediction,
        payload.query_expr,
        payload.query_niche,
        evaluation_target,
        payload.target_niche[split.evaluation_indices],
    )
    row["niche_wise_delta_pcc"] = niche_delta
    row["niche_wise_delta_n_strata"] = niche_delta_count
    row["niche_wise_delta_reason"] = niche_delta_reason
    predicted_delta = result.prediction.mean(axis=0) - payload.query_expr.mean(axis=0)
    true_delta = evaluation_target.mean(axis=0) - payload.query_expr.mean(axis=0)
    reference_delta = reference_target.mean(axis=0) - payload.query_expr.mean(axis=0)
    row.update(de_direction_metrics(predicted_delta, true_delta, reference_delta, top_k=20))
    row.update(de_direction_metrics(predicted_delta, true_delta, reference_delta, top_k=50))
    if payload.is_cross_slide:
        row["spatial_map_pcc"] = float("nan")
        row["spatial_map_pcc_reason"] = "unregistered_cross_slide"
    else:
        row["spatial_map_pcc"], row["spatial_map_pcc_reason"] = spatial_map_correlation(
            result.prediction,
            payload.query_pos,
            evaluation_target,
            payload.target_pos[split.evaluation_indices],
            select_de_indices(reference_delta, top_k=20),
        )
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


def _cached_spatialprop_result(
    payload: TaskPayload,
    split: TargetSplit,
    cfg: ExperimentConfig,
) -> PredictionResult:
    cache_dir = cfg.run_root / "results" / "spatialprop_cache"
    assert_writable_output(cache_dir, cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity = "|".join(
        [
            payload.task_id,
            str(split.reference_fraction),
            str(split.seed),
            split.split_type,
            str(cfg.spatialprop_k),
            str(cfg.spatialprop_hidden_dim),
            str(cfg.spatialprop_latent_dim),
            str(cfg.spatialprop_epochs),
        ]
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    cache_path = cache_dir / f"{digest}.npz"
    if cache_path.exists():
        with np.load(cache_path) as cached:
            prediction = cached["prediction"].astype(np.float32)
    else:
        trained = predict_spatialprop_reference_only(
            payload,
            split,
            k=cfg.spatialprop_k,
            hidden_dim=cfg.spatialprop_hidden_dim,
            latent_dim=cfg.spatialprop_latent_dim,
            epochs=cfg.spatialprop_epochs,
            seed=split.seed,
        )
        prediction = trained.prediction
        np.savez_compressed(cache_path, prediction=prediction)
    return PredictionResult(
        prediction=prediction,
        method="spatialprop_reference_only_retrain",
        access_level="L2",
        calibration=True,
        uses_target_reference=True,
        components={
            "model_backend": "torch_spatialprop_style",
            "training_epochs": cfg.spatialprop_epochs,
            "cache_key": digest,
        },
    )


def _spatialprop_leakage_probe(
    payload: TaskPayload,
    split: TargetSplit,
    cfg: ExperimentConfig,
) -> bool:
    altered = payload.with_target_evaluation_expr(
        split,
        payload.target_expr[split.evaluation_indices] + 1000.0,
    )
    settings = {
        "k": cfg.spatialprop_k,
        "hidden_dim": min(cfg.spatialprop_hidden_dim, 8),
        "latent_dim": min(cfg.spatialprop_latent_dim, 4),
        "epochs": 1,
        "seed": split.seed,
    }
    original = predict_spatialprop_reference_only(payload, split, **settings)
    changed = predict_spatialprop_reference_only(altered, split, **settings)
    return bool(np.allclose(original.prediction, changed.prediction))


def reassess_manifest(
    manifest: pd.DataFrame,
    cfg: ExperimentConfig,
    *,
    include_expensive: bool = False,
    only_expensive: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if only_expensive and not include_expensive:
        raise ValueError("only_expensive requires include_expensive")
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
                        unavailable_methods = (
                            ("spatialprop_reference_only_retrain",)
                            if only_expensive
                            else EXISTING_METHODS
                            + (
                                ("spatialprop_reference_only_retrain",)
                                if include_expensive
                                else ()
                            )
                        )
                        for method in unavailable_methods:
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
                    if not only_expensive:
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
                    if include_expensive:
                        leakage_check = _spatialprop_leakage_probe(payload, split, cfg)
                        result = _cached_spatialprop_result(payload, split, cfg)
                        row, _ = _method_metric_row(
                            payload,
                            split,
                            "spatialprop_reference_only_retrain",
                            result_override=result,
                            leakage_check_override=leakage_check,
                        )
                        metric_rows.append(row)
                        coverage_rows.append(
                            _coverage_row(
                                payload,
                                split,
                                model_family="spatialprop_style_spot_gnn",
                                method="spatialprop_reference_only_retrain",
                                status="approximate",
                                reason="reference-only SpatialProp-style retraining",
                                leakage_check=leakage_check,
                            )
                        )
                    elif not only_expensive:
                        coverage_rows.append(
                            _coverage_row(
                                payload,
                                split,
                                model_family="spatialprop_style_spot_gnn",
                                method="spatialprop_reference_only_retrain",
                                status="unavailable",
                                reason="excluded_expensive_smoke",
                                leakage_check=False,
                            )
                        )
    return pd.DataFrame(metric_rows), pd.DataFrame(coverage_rows), pd.DataFrame(split_rows)


def merge_fast_and_expensive(
    fast_metrics: pd.DataFrame,
    fast_coverage: pd.DataFrame,
    expensive_metrics: pd.DataFrame,
    expensive_coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace fast-lane unavailable coverage with completed expensive rows."""
    expensive_methods = set(expensive_coverage["method"].astype(str))
    retained_coverage = fast_coverage[
        ~fast_coverage["method"].astype(str).isin(expensive_methods)
    ]
    retained_metrics = fast_metrics[
        ~fast_metrics["method"].astype(str).isin(expensive_methods)
    ]
    metrics = pd.concat([retained_metrics, expensive_metrics], ignore_index=True)
    coverage = pd.concat([retained_coverage, expensive_coverage], ignore_index=True)
    keys = ["task_id", "method", "reference_fraction", "split_seed", "split_type"]
    if set(keys).issubset(metrics.columns) and metrics.duplicated(keys).any():
        raise ValueError("Merged reassessment metrics contain duplicate keys")
    if set(keys).issubset(coverage.columns) and coverage.duplicated(keys).any():
        raise ValueError("Merged reassessment coverage contains duplicate keys")
    return metrics, coverage
