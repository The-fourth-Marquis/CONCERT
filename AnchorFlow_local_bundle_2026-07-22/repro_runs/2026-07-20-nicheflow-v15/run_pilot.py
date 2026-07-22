#!/usr/bin/env python3
"""Run a representative, leakage-controlled NicheFlow-v1.5 pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Iterable

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))
sys.path.insert(
    0,
    str(REPO_ROOT / "repro_runs" / "2026-07-11-nicheflow-v2" / "src"),
)

from nicheflow_v15.predict import predict_task_reference_only
from nicheflow_v2.baselines import predict_existing_method
from nicheflow_v2.data import load_manifest, load_task_payload
from nicheflow_v2.metrics import (
    boundary_leakage_score,
    distribution_metrics,
    niche_stratified_energy_distance,
    perturbation_effect_metrics,
    source_individuality_score,
)
from nicheflow_v2.splits import split_target_pool


@dataclass(frozen=True)
class PilotConfig:
    name: str
    latent_dim: int
    hidden_dim: int
    context_weight: float
    sinkhorn_epsilon: float
    ridge: float
    modulation_strength: float = 1.0


DEFAULT_CONFIGS: tuple[PilotConfig, ...] = (
    PilotConfig(
        name="expression_only_l8",
        latent_dim=8,
        hidden_dim=16,
        context_weight=0.0,
        sinkhorn_epsilon=0.1,
        ridge=0.01,
    ),
    PilotConfig(
        name="context_l8",
        latent_dim=8,
        hidden_dim=16,
        context_weight=1.0,
        sinkhorn_epsilon=0.1,
        ridge=0.01,
    ),
    PilotConfig(
        name="context_l16",
        latent_dim=16,
        hidden_dim=32,
        context_weight=1.0,
        sinkhorn_epsilon=0.1,
        ridge=0.01,
    ),
)


INTRINSIC_CONFIGS: tuple[PilotConfig, ...] = (
    PilotConfig(
        name="mean_shift_l16",
        latent_dim=16,
        hidden_dim=32,
        context_weight=1.0,
        sinkhorn_epsilon=0.1,
        ridge=0.01,
        modulation_strength=0.0,
    ),
    PilotConfig(
        name="modulation_025_l16",
        latent_dim=16,
        hidden_dim=32,
        context_weight=1.0,
        sinkhorn_epsilon=0.1,
        ridge=0.01,
        modulation_strength=0.25,
    ),
    PilotConfig(
        name="modulation_100_l16",
        latent_dim=16,
        hidden_dim=32,
        context_weight=1.0,
        sinkhorn_epsilon=0.1,
        ridge=0.01,
        modulation_strength=1.0,
    ),
)


def _representative_task_ids(manifest: pd.DataFrame) -> tuple[str, ...]:
    chosen: list[str] = []
    exact = manifest.loc[manifest["benchmark_kind"] == "exact_available"]
    if not exact.empty:
        chosen.append(str(exact.iloc[0]["task_id"]))
    for task_type in (
        "border",
        "cross_niche_core_to_border",
        "cross_slide",
        "niche_core",
        "periphery",
    ):
        matches = manifest.loc[manifest["task_type"] == task_type]
        if not matches.empty:
            chosen.append(str(matches.iloc[0]["task_id"]))
    return tuple(dict.fromkeys(chosen))


def _target_reference_for_queries(
    payload: object,
    split: object,
) -> tuple[np.ndarray, np.ndarray]:
    evaluation = payload.target_expr[split.evaluation_indices]
    evaluation_niche = payload.target_niche[split.evaluation_indices]
    query_labels = np.where(
        payload.query_niche >= 0.5,
        "tumor_like",
        "boundary_or_normal_like",
    )
    target_labels = np.where(
        evaluation_niche >= 0.5,
        "tumor_like",
        "boundary_or_normal_like",
    )
    fallback = evaluation.mean(axis=0)
    rows = []
    for label in query_labels:
        mask = target_labels == label
        rows.append(evaluation[mask].mean(axis=0) if mask.any() else fallback)
    return np.asarray(rows, dtype=np.float32), query_labels


def _metric_bundle(
    prediction: np.ndarray,
    payload: object,
    split: object,
) -> dict[str, float]:
    evaluation = payload.target_expr[split.evaluation_indices]
    evaluation_niche = payload.target_niche[split.evaluation_indices]
    distribution = distribution_metrics(
        prediction,
        evaluation,
        payload.target_expr,
    )
    effects = perturbation_effect_metrics(
        prediction,
        payload.query_expr,
        evaluation,
    )
    niche_distance, _, _ = niche_stratified_energy_distance(
        prediction,
        payload.query_niche,
        evaluation,
        evaluation_niche,
    )
    target_reference, labels = _target_reference_for_queries(payload, split)
    boundary = boundary_leakage_score(
        prediction,
        target_reference,
        labels,
    )
    individuality, _ = source_individuality_score(
        prediction,
        payload.query_expr,
    )
    return {
        "heldout_edistance": float(distribution["heldout_edistance"]),
        "niche_stratified_edistance": float(niche_distance),
        "delta_pcc": float(effects["delta_pcc"]),
        "delta_mae": float(effects["delta_mae"]),
        "boundary_leakage": float(boundary),
        "source_individuality": float(individuality),
    }


def _unit_row(
    *,
    payload: object,
    split: object,
    config: PilotConfig,
    seed: int,
) -> dict[str, object]:
    settings = {
        "latent_dim": config.latent_dim,
        "hidden_dim": config.hidden_dim,
        "context_weight": config.context_weight,
        "modulation_strength": config.modulation_strength,
        "sinkhorn_epsilon": config.sinkhorn_epsilon,
        "ridge": config.ridge,
        "seed": int(seed),
    }
    started = time.perf_counter()
    nonlinear = predict_task_reference_only(payload, split, **settings)
    altered = payload.with_target_evaluation_expr(
        split,
        payload.target_expr[split.evaluation_indices] + 1000.0,
    )
    changed = predict_task_reference_only(altered, split, **settings)
    leakage_check = bool(
        np.allclose(nonlinear.prediction, changed.prediction)
    )
    if not leakage_check:
        raise AssertionError(
            f"Evaluation leakage in {payload.task_id}/{config.name}"
        )
    fixed = predict_existing_method(
        "legacy_v1_fixed_80_20_l3",
        payload,
        split,
    )
    nonlinear_metrics = _metric_bundle(nonlinear.prediction, payload, split)
    fixed_metrics = _metric_bundle(fixed.prediction, payload, split)
    elapsed = time.perf_counter() - started
    row: dict[str, object] = {
        "task_id": payload.task_id,
        "task_type": str(payload.task.get("task_type", "")),
        "benchmark_kind": str(payload.task.get("benchmark_kind", "")),
        "is_cross_slide": bool(payload.is_cross_slide),
        "config_name": config.name,
        "reference_fraction": float(split.reference_fraction),
        "split_seed": int(split.seed),
        "split_type": str(split.split_type),
        "n_reference": int(split.reference_indices.size),
        "n_evaluation": int(split.evaluation_indices.size),
        "n_query": int(payload.query_expr.shape[0]),
        "uses_evaluation_target": False,
        "leakage_mutation_check": leakage_check,
        "calibration": False,
        "elapsed_seconds": float(elapsed),
    }
    row.update(asdict(config))
    row.update(nonlinear_metrics)
    for key, value in fixed_metrics.items():
        row[f"fixed_80_20_{key}"] = value
    row["delta_vs_fixed_80_20"] = (
        nonlinear_metrics["heldout_edistance"]
        - fixed_metrics["heldout_edistance"]
    )
    row["niche_delta_vs_fixed"] = (
        nonlinear_metrics["niche_stratified_edistance"]
        - fixed_metrics["niche_stratified_edistance"]
    )
    row["delta_pcc_diff_vs_fixed"] = (
        nonlinear_metrics["delta_pcc"] - fixed_metrics["delta_pcc"]
    )
    row["delta_mae_diff_vs_fixed"] = (
        nonlinear_metrics["delta_mae"] - fixed_metrics["delta_mae"]
    )
    row["boundary_diff_vs_fixed"] = (
        nonlinear_metrics["boundary_leakage"]
        - fixed_metrics["boundary_leakage"]
    )
    row["boundary_relative_vs_fixed"] = (
        row["boundary_diff_vs_fixed"]
        / max(abs(fixed_metrics["boundary_leakage"]), 1e-8)
    )
    row["individuality_diff_vs_fixed"] = (
        nonlinear_metrics["source_individuality"]
        - fixed_metrics["source_individuality"]
    )
    return row


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for config_name, group in metrics.groupby("config_name", sort=False):
        spatial = group.loc[group["split_type"] == "spatial_block"]
        random = group.loc[group["split_type"] == "random_stratified"]
        delta = group["delta_vs_fixed_80_20"]
        rows.append(
            {
                "config_name": config_name,
                "n_units": int(group.shape[0]),
                "n_tasks": int(group["task_id"].nunique()),
                "mean_delta_vs_fixed": float(delta.mean()),
                "median_delta_vs_fixed": float(delta.median()),
                "wins": int((delta < 0.0).sum()),
                "win_rate": float((delta < 0.0).mean()),
                "random_mean_delta": (
                    float(random["delta_vs_fixed_80_20"].mean())
                    if not random.empty
                    else float("nan")
                ),
                "spatial_block_mean_delta": (
                    float(spatial["delta_vs_fixed_80_20"].mean())
                    if not spatial.empty
                    else float("nan")
                ),
                "mean_niche_delta_vs_fixed": float(
                    group["niche_delta_vs_fixed"].mean()
                ),
                "mean_delta_pcc_diff_vs_fixed": float(
                    group["delta_pcc_diff_vs_fixed"].mean()
                ),
                "mean_delta_mae_diff_vs_fixed": float(
                    group["delta_mae_diff_vs_fixed"].mean()
                ),
                "mean_boundary_diff_vs_fixed": float(
                    group["boundary_diff_vs_fixed"].mean()
                ),
                "mean_boundary_relative_vs_fixed": float(
                    group["boundary_relative_vs_fixed"].mean()
                ),
                "mean_individuality_diff_vs_fixed": float(
                    group["individuality_diff_vs_fixed"].mean()
                ),
                "all_leakage_checks_pass": bool(
                    group["leakage_mutation_check"].all()
                ),
                "mean_elapsed_seconds": float(
                    group["elapsed_seconds"].mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_delta_vs_fixed", "config_name"],
        kind="stable",
    )


def _decision(summary: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, object]:
    selected = summary.iloc[0]
    mean_delta = float(selected["mean_delta_vs_fixed"])
    spatial_delta = float(selected["spatial_block_mean_delta"])
    leakage = bool(selected["all_leakage_checks_pass"])
    delta_gate = float(selected["mean_delta_pcc_diff_vs_fixed"]) >= -0.02
    boundary_gate = (
        float(selected["mean_boundary_relative_vs_fixed"]) <= 0.05
    )
    individuality_gate = (
        float(selected["mean_individuality_diff_vs_fixed"]) >= -0.02
    )
    mechanism_gate = bool(
        delta_gate and boundary_gate and individuality_gate
    )
    distribution_gate = bool(
        mean_delta < 0.0
        and (np.isnan(spatial_delta) or spatial_delta < 0.0)
    )
    if leakage and distribution_gate and mechanism_gate:
        status = "advance_to_multiseed"
    elif leakage and distribution_gate and not mechanism_gate:
        status = "redesign_mechanism_before_multiseed"
    elif leakage and mean_delta < 0.0:
        status = "refine_spatial_block_before_multiseed"
    else:
        status = "do_not_scale_current_monge"
    return {
        "decision": status,
        "selected_config": str(selected["config_name"]),
        "n_units": int(metrics.shape[0]),
        "n_tasks": int(metrics["task_id"].nunique()),
        "mean_delta_vs_fixed": mean_delta,
        "spatial_block_mean_delta": spatial_delta,
        "win_rate": float(selected["win_rate"]),
        "all_leakage_checks_pass": leakage,
        "distribution_gate_pass": distribution_gate,
        "mechanism_gate_pass": mechanism_gate,
        "delta_pcc_gate_pass": bool(delta_gate),
        "boundary_gate_pass": bool(boundary_gate),
        "individuality_gate_pass": bool(individuality_gate),
        "interpretation": (
            "Pilot evidence only; promotion still requires the complete "
            "37-task, five-seed held-out gates."
        ),
    }


def run_pilot(
    *,
    output_dir: Path,
    task_ids: Iterable[str] | None = None,
    configs: tuple[PilotConfig, ...] = DEFAULT_CONFIGS,
    reference_fraction: float = 0.50,
    seed: int = 20260711,
    split_types: tuple[str, ...] = (
        "random_stratified",
        "spatial_block",
    ),
) -> dict[str, Path]:
    manifest = load_manifest()
    selected_ids = (
        tuple(task_ids)
        if task_ids is not None
        else _representative_task_ids(manifest)
    )
    if not selected_ids:
        raise ValueError("No pilot tasks selected")
    if not configs:
        raise ValueError("No pilot configurations supplied")
    indexed = manifest.set_index("task_id", drop=False)
    missing = [task_id for task_id in selected_ids if task_id not in indexed.index]
    if missing:
        raise ValueError(f"Unknown pilot task IDs: {missing}")

    rows: list[dict[str, object]] = []
    total = len(selected_ids) * len(split_types) * len(configs)
    unit = 0
    for task_id in selected_ids:
        payload = load_task_payload(indexed.loc[task_id])
        for split_type in split_types:
            split = split_target_pool(
                payload,
                reference_fraction=float(reference_fraction),
                seed=int(seed),
                split_type=str(split_type),
            )
            if split.status != "ok":
                raise ValueError(
                    f"Pilot split unavailable for {task_id}: {split.reason}"
                )
            for config in configs:
                unit += 1
                print(
                    f"[{unit}/{total}] {task_id} {split_type} {config.name}",
                    flush=True,
                )
                rows.append(
                    _unit_row(
                        payload=payload,
                        split=split,
                        config=config,
                        seed=int(seed),
                    )
                )

    metrics = pd.DataFrame(rows)
    summary = _summarize(metrics)
    decision = _decision(summary, metrics)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.csv"
    decision_path = output_dir / "decision.json"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "metrics": metrics_path,
        "summary": summary_path,
        "decision": decision_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_ROOT / "results" / "pilot",
    )
    parser.add_argument("--reference-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--config-set",
        choices=("base", "intrinsic"),
        default="base",
    )
    args = parser.parse_args()
    outputs = run_pilot(
        output_dir=args.output_dir,
        reference_fraction=args.reference_fraction,
        seed=args.seed,
        configs=(
            INTRINSIC_CONFIGS
            if args.config_set == "intrinsic"
            else DEFAULT_CONFIGS
        ),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
