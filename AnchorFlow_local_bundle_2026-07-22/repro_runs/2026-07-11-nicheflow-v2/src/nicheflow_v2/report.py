"""Statistical comparisons and evidence-gate summaries."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig


def clustered_bootstrap(
    comparison: pd.DataFrame,
    *,
    value_col: str,
    cluster_col: str = "task_id",
    seed: int,
    n_resamples: int,
) -> dict[str, Any]:
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    source = comparison[[cluster_col, value_col]].dropna().copy()
    clusters = source[cluster_col].drop_duplicates().to_numpy()
    if clusters.size == 0:
        raise ValueError("No valid clusters for bootstrap")
    rng = np.random.default_rng(seed)
    samples = np.empty(int(n_resamples), dtype=np.float64)
    grouped = {
        cluster: source.loc[source[cluster_col] == cluster, value_col].to_numpy()
        for cluster in clusters
    }
    for index in range(int(n_resamples)):
        chosen = rng.choice(clusters, size=clusters.size, replace=True)
        values = np.concatenate([grouped[cluster] for cluster in chosen])
        samples[index] = values.mean()
    return {
        "mean": float(source[value_col].mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "n_clusters": int(clusters.size),
        "n_rows": int(source.shape[0]),
        "seed": int(seed),
        "n_resamples": int(n_resamples),
    }


def evaluate_success_gates(
    v2_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    *,
    candidate_method: str,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Evaluate the seven approved promotion gates without a composite score."""
    keys = ["task_id", "reference_fraction", "split_seed", "split_type"]
    candidate = v2_metrics[v2_metrics["method"] == candidate_method].copy()
    if candidate.empty:
        raise ValueError(f"Candidate method not found: {candidate_method}")
    calibration_values = candidate["calibration"].astype(bool).unique()
    if calibration_values.size != 1:
        raise ValueError("Candidate method mixes calibration states")
    candidate_calibration = bool(calibration_values[0])
    baseline = baseline_metrics[
        (baseline_metrics["model_family"] == "a2_reference_only_baseline")
        & (baseline_metrics["calibration"].astype(bool) == candidate_calibration)
    ].copy()
    baseline = (
        baseline.sort_values(keys + ["heldout_edistance", "method"])
        .groupby(keys, as_index=False)
        .first()
    )
    metric_columns = [
        "heldout_edistance", "delta_pcc", "delta_spearman", "delta_mae", "delta_rmse",
        "de_top20_direction_accuracy", "de_top20_opposite_direction_fraction",
        "boundary_leakage_score", "source_individuality_score",
    ]
    paired = candidate[keys + metric_columns].merge(
        baseline[keys + metric_columns],
        on=keys,
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError("No paired candidate/baseline task-split rows")
    paired["delta_edistance"] = (
        paired["heldout_edistance_candidate"] - paired["heldout_edistance_baseline"]
    )
    bootstrap = clustered_bootstrap(
        paired,
        value_col="delta_edistance",
        seed=config.master_seed,
        n_resamples=config.bootstrap_resamples,
    )
    win_rate = float((paired["delta_edistance"] < 0).mean())
    g1 = bool(win_rate > 0.50 and bootstrap["mean"] < 0 and bootstrap["ci_high"] < 0)

    def mean_difference(metric: str) -> float:
        return float(
            (paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]).mean()
        )

    def mean_relative_difference(metric: str) -> float:
        numerator = paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
        denominator = np.maximum(np.abs(paired[f"{metric}_baseline"]), 1e-8)
        return float((numerator / denominator).mean())

    delta_pcc_diff = mean_difference("delta_pcc")
    delta_spearman_diff = mean_difference("delta_spearman")
    delta_mae_relative = mean_relative_difference("delta_mae")
    delta_rmse_relative = mean_relative_difference("delta_rmse")
    g2 = bool(
        delta_pcc_diff >= -0.02
        and delta_spearman_diff >= -0.02
        and delta_mae_relative <= 0.05
        and delta_rmse_relative <= 0.05
    )

    de_direction_diff = mean_difference("de_top20_direction_accuracy")
    opposite_direction_diff = mean_difference("de_top20_opposite_direction_fraction")
    g3 = bool(de_direction_diff >= -0.02 and opposite_direction_diff <= 0.02)

    v1 = baseline_metrics[
        (baseline_metrics["model_family"] == "nicheflow_v1")
        & (baseline_metrics["calibration"].astype(bool) == candidate_calibration)
    ].copy()
    v1 = (
        v1.sort_values(keys + ["heldout_edistance", "method"])
        .groupby(keys, as_index=False)
        .first()
    )
    mechanism_paired = candidate[
        keys + ["boundary_leakage_score", "source_individuality_score"]
    ].merge(
        v1[keys + ["boundary_leakage_score", "source_individuality_score"]],
        on=keys,
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    if mechanism_paired.empty:
        raise ValueError("No paired candidate/NicheFlow-v1 mechanism rows")
    boundary_numerator = (
        mechanism_paired["boundary_leakage_score_candidate"]
        - mechanism_paired["boundary_leakage_score_baseline"]
    )
    boundary_denominator = np.maximum(
        np.abs(mechanism_paired["boundary_leakage_score_baseline"]), 1e-8
    )
    boundary_relative = float((boundary_numerator / boundary_denominator).mean())
    individuality_diff = float(
        (
            mechanism_paired["source_individuality_score_candidate"]
            - mechanism_paired["source_individuality_score_baseline"]
        ).mean()
    )
    g4 = bool(boundary_relative <= 0.05 and individuality_diff >= -0.02)

    reduced = paired.groupby("reference_fraction")["delta_edistance"].agg(["mean", "count"])
    mean_050 = float(reduced.loc[0.50, "mean"]) if 0.50 in reduced.index else float("nan")
    mean_025 = float(reduced.loc[0.25, "mean"]) if 0.25 in reduced.index else float("nan")
    tasks_025 = int(
        paired.loc[paired["reference_fraction"] == 0.25, "task_id"].nunique()
    )
    support_025 = bool(mean_025 < 0) if tasks_025 >= 15 else True
    g5 = bool(np.isfinite(mean_050) and mean_050 < 0 and support_025)

    g6 = bool(bootstrap["mean"] < 0 and win_rate >= 0.50)

    n_tasks = int(paired["task_id"].nunique())
    n_seeds = int(paired["split_seed"].nunique())
    n_split_types = int(paired["split_type"].nunique())
    g7 = bool(n_tasks == 37 and n_seeds >= 5 and n_split_types >= 2)

    records = [
        ("G1", "heldout_edistance_superiority_with_clustered_ci", g1, {
            "win_rate": win_rate, "mean_delta": bootstrap["mean"],
            "ci_low": bootstrap["ci_low"], "ci_high": bootstrap["ci_high"],
        }, "win_rate>0.50, mean<0, ci_high<0"),
        ("G2", "delta_metric_noninferiority", g2, {
            "delta_pcc_diff": delta_pcc_diff,
            "delta_spearman_diff": delta_spearman_diff,
            "delta_mae_relative": delta_mae_relative,
            "delta_rmse_relative": delta_rmse_relative,
        }, "correlations>=-0.02, relative errors<=0.05"),
        ("G3", "de_direction_noninferiority", g3, {
            "direction_accuracy_diff": de_direction_diff,
            "opposite_direction_diff": opposite_direction_diff,
        }, "direction>=-0.02, opposite<=0.02"),
        ("G4", "boundary_and_individuality_noninferiority", g4, {
            "boundary_relative": boundary_relative,
            "individuality_diff": individuality_diff,
        }, "boundary_relative<=0.05, individuality>=-0.02"),
        ("G5", "reduced_reference_support", g5, {
            "mean_delta_fraction_050": mean_050,
            "mean_delta_fraction_025": mean_025,
            "tasks_fraction_025": tasks_025,
        }, "fraction_050 mean<0 and feasible fraction_025 mean<0"),
        ("G6", "calibration_free_gain", g6, {
            "win_rate": win_rate, "mean_delta": bootstrap["mean"],
        }, "win_rate>=0.50 and mean<0"),
        ("G7", "coverage_and_variability_reported", g7, {
            "n_tasks": n_tasks, "n_seeds": n_seeds, "n_split_types": n_split_types,
        }, "37 tasks, >=5 seeds, >=2 split types"),
    ]
    passed_all = all(record[2] for record in records)
    decision = "advance_to_graph_flow" if passed_all else "do_not_advance_to_graph_flow"
    return pd.DataFrame(
        [
            {
                "gate_id": gate_id,
                "criterion": criterion,
                "passed": passed,
                "observed": json.dumps(observed, sort_keys=True),
                "threshold": threshold,
                "candidate_method": candidate_method,
                "decision": decision,
            }
            for gate_id, criterion, passed, observed, threshold in records
        ]
    )
