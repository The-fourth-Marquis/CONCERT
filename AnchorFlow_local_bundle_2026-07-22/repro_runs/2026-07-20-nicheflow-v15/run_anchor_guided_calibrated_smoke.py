#!/usr/bin/env python3
"""Reference-only strength calibration followed by held-out smoke evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
sys.path.insert(0, str(RUN_ROOT))
sys.path.insert(0, str(RUN_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "repro_runs" / "2026-07-11-nicheflow-v2" / "src"))

from nicheflow_v15.anchor_guided_factorized_predict import (
    predict_task_anchor_guided_factorized,
)
from nicheflow_v2.baselines import predict_existing_method
from nicheflow_v2.data import load_manifest, load_task_payload
from nicheflow_v2.metrics import (
    boundary_leakage_score,
    distribution_metrics,
    perturbation_effect_metrics,
    source_individuality_score,
)
from nicheflow_v2.splits import split_target_pool
from run_pilot import _metric_bundle, _target_reference_for_queries


STRENGTHS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.72, 0.74, 0.75, 0.755, 0.76, 0.765, 0.77, 0.775, 0.78, 0.8, 1.0)


def _metrics_against_target(
    prediction: np.ndarray,
    payload: object,
    split: object,
) -> dict[str, float]:
    evaluation = payload.target_expr[split.evaluation_indices]
    evaluation_niche = payload.target_niche[split.evaluation_indices]
    distribution = distribution_metrics(prediction, evaluation, payload.target_expr)
    effects = perturbation_effect_metrics(prediction, payload.query_expr, evaluation)
    target_reference, labels = _target_reference_for_queries(payload, split)
    boundary = boundary_leakage_score(prediction, target_reference, labels)
    individuality, _ = source_individuality_score(prediction, payload.query_expr)
    return {
        "heldout_edistance": float(distribution["heldout_edistance"]),
        "delta_pcc": float(effects["delta_pcc"]),
        "boundary_leakage": float(boundary),
        "source_individuality": float(individuality),
    }


def _select_strength(
    rows: list[dict[str, float]],
    fixed: dict[str, float],
) -> float:
    gated = []
    for row in rows:
        boundary_relative = (
            row["boundary_leakage"] - fixed["boundary_leakage"]
        ) / max(abs(fixed["boundary_leakage"]), 1e-8)
        if (
            row["heldout_edistance"] <= fixed["heldout_edistance"]
            and row["delta_pcc"] - fixed["delta_pcc"] >= -0.02
            and boundary_relative <= 0.05
            and row["source_individuality"] - fixed["source_individuality"] >= -0.02
        ):
            gated.append(row)
    if not gated:
        return 0.0
    pool = gated
    selected = max(
        pool,
        key=lambda row: (
            row["modulation_strength"],
            -row["heldout_edistance"],
        ),
    )
    return float(selected["modulation_strength"])


def _pairwise_geometry_error(prediction: np.ndarray, source: np.ndarray) -> float:
    predicted_dist = np.linalg.norm(
        prediction[:, None, :] - prediction[None, :, :], axis=2
    )
    source_dist = np.linalg.norm(
        source[:, None, :] - source[None, :, :], axis=2
    )
    return float(
        np.mean(np.abs(predicted_dist - source_dist))
        / max(float(source_dist.mean()), 1e-8)
    )


def run_anchor_guided_calibrated_smoke(
    *,
    output_dir: Path,
    task_index: int = 0,
    reference_fraction: float = 0.70,
    seed: int = 20260711,
    split_type: str = "random_stratified",
    strengths: tuple[float, ...] = STRENGTHS,
) -> Path:
    payload = load_task_payload(load_manifest().iloc[int(task_index)])
    split = split_target_pool(
        payload,
        reference_fraction=float(reference_fraction),
        seed=int(seed),
        split_type=str(split_type),
    )
    if split.status != "ok":
        raise ValueError(f"Smoke split unavailable: {split.reason}")

    reference_payload = payload.with_target_subset(split.reference_indices)
    pseudo_split = split_target_pool(
        reference_payload,
        reference_fraction=0.70,
        seed=int(seed) + 1000,
        split_type="random_stratified",
    )
    if pseudo_split.status != "ok":
        raise ValueError(f"Pseudo-validation split unavailable: {pseudo_split.reason}")
    fixed_pseudo_result = predict_existing_method(
        "legacy_v1_fixed_80_20_l3",
        reference_payload,
        pseudo_split,
    )
    fixed_pseudo = _metrics_against_target(
        fixed_pseudo_result.prediction,
        reference_payload,
        pseudo_split,
    )
    calibration_rows: list[dict[str, float]] = []
    for strength in strengths:
        result = predict_task_anchor_guided_factorized(
            reference_payload,
            pseudo_split,
            anchor_temperature=0.25,
            modulation_strength=float(strength),
        )
        row = {
            "modulation_strength": float(strength),
            **_metrics_against_target(result.prediction, reference_payload, pseudo_split),
        }
        calibration_rows.append(row)
    selected_strength = _select_strength(calibration_rows, fixed_pseudo)

    altered = payload.with_target_evaluation_expr(
        split,
        payload.target_expr[split.evaluation_indices] + 1000.0,
    )
    result = predict_task_anchor_guided_factorized(
        payload,
        split,
        anchor_temperature=0.25,
        modulation_strength=selected_strength,
    )
    changed = predict_task_anchor_guided_factorized(
        altered,
        split,
        anchor_temperature=0.25,
        modulation_strength=selected_strength,
    )
    leakage = bool(np.allclose(result.prediction, changed.prediction))
    if not leakage:
        raise AssertionError("Evaluation-expression leakage after calibration")
    fixed = predict_existing_method("legacy_v1_fixed_80_20_l3", payload, split)
    actual_metrics = _metric_bundle(result.prediction, payload, split)
    fixed_metrics = _metric_bundle(fixed.prediction, payload, split)
    actual_row: dict[str, object] = {
        "task_id": payload.task_id,
        "method": "anchor_guided_calibrated",
        "selected_strength": selected_strength,
        "reference_fraction": float(reference_fraction),
        "split_seed": int(seed),
        "split_type": str(split_type),
        "n_reference": int(split.reference_indices.size),
        "n_evaluation": int(split.evaluation_indices.size),
        "n_query": int(payload.query_expr.shape[0]),
        "uses_evaluation_target": False,
        "leakage_mutation_check": leakage,
        "pairwise_geometry_error": _pairwise_geometry_error(
            result.prediction,
            payload.query_expr,
        ),
        **actual_metrics,
    }
    for key, value in fixed_metrics.items():
        actual_row[f"fixed_80_20_{key}"] = value
        actual_row[f"diff_vs_fixed_{key}"] = actual_metrics[key] - value

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "task_id": payload.task_id,
        "reference_fraction": float(reference_fraction),
        "split_seed": int(seed),
        "split_type": str(split_type),
        "pseudo_reference_fraction": 0.70,
        "pseudo_split_seed": int(seed) + 1000,
        "pseudo_split_type": "random_stratified",
        "strengths": [float(value) for value in strengths],
        "selected_strength": selected_strength,
        "pseudo_fixed_80_20": fixed_pseudo,
        "pseudo_calibration_rows": calibration_rows,
        "actual": actual_row,
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        output_dir / "prediction.npz",
        prediction=result.prediction,
        fixed_prediction=fixed.prediction,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_ROOT / "results" / "anchor_guided_calibrated_smoke",
    )
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--reference-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--split-type", default="random_stratified")
    args = parser.parse_args()
    path = run_anchor_guided_calibrated_smoke(
        output_dir=args.output_dir,
        task_index=args.task_index,
        reference_fraction=args.reference_fraction,
        seed=args.seed,
        split_type=args.split_type,
    )
    print(path)


if __name__ == "__main__":
    main()
