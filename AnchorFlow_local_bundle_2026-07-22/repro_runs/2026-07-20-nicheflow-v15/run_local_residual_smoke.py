#!/usr/bin/env python3
"""Smoke-test local anchors plus bounded source-context residual flow."""

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
sys.path.insert(
    0,
    str(REPO_ROOT / "repro_runs" / "2026-07-11-nicheflow-v2" / "src"),
)

from nicheflow_v15.local_residual_flow_predict import (
    predict_task_local_residual_flow,
)
from nicheflow_v2.baselines import predict_existing_method
from nicheflow_v2.data import load_manifest, load_task_payload
from nicheflow_v2.splits import split_target_pool
from run_pilot import _metric_bundle


HORIZONS = (0.0, 0.25, 0.5, 1.0)


def run_local_residual_smoke(
    *,
    output_dir: Path,
    task_index: int = 0,
    reference_fraction: float = 0.70,
    seed: int = 20260711,
    split_type: str = "random_stratified",
    horizons: tuple[float, ...] = HORIZONS,
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

    fixed = predict_existing_method(
        "legacy_v1_fixed_80_20_l3",
        payload,
        split,
    )
    fixed_metrics = _metric_bundle(fixed.prediction, payload, split)
    altered = payload.with_target_evaluation_expr(
        split,
        payload.target_expr[split.evaluation_indices] + 1000.0,
    )

    rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {
        "fixed_80_20": fixed.prediction,
    }
    settings = {
        "latent_dim": 16,
        "hidden_dim": 64,
        "epochs": 80,
        "integration_steps": 16,
        "anchor_temperature": 0.25,
        "modulation_cap": 0.5,
        "distribution_weight": 0.25,
        "identity_weight": 0.05,
        "seed": int(seed),
    }

    for horizon in horizons:
        result = predict_task_local_residual_flow(
            payload,
            split,
            integration_horizon=float(horizon),
            **settings,
        )
        changed = predict_task_local_residual_flow(
            altered,
            split,
            integration_horizon=float(horizon),
            **settings,
        )
        leakage = bool(
            np.allclose(result.base_prediction, changed.base_prediction)
            and np.allclose(result.prediction, changed.prediction)
        )
        if not leakage:
            raise AssertionError(
                f"Evaluation-expression leakage at horizon={horizon}"
            )

        method = "base_prediction" if float(horizon) == 0.0 else (
            f"local_residual_h{float(horizon):g}"
        )
        prediction = result.base_prediction if float(horizon) == 0.0 else result.prediction
        predictions[method] = prediction
        metrics = _metric_bundle(prediction, payload, split)
        row: dict[str, object] = {
            "task_id": payload.task_id,
            "task_type": str(payload.task.get("task_type", "")),
            "benchmark_kind": str(payload.task.get("benchmark_kind", "")),
            "method": method,
            "integration_horizon": float(horizon),
            "reference_fraction": float(reference_fraction),
            "split_seed": int(seed),
            "split_type": str(split_type),
            "n_reference": int(split.reference_indices.size),
            "n_evaluation": int(split.evaluation_indices.size),
            "n_query": int(payload.query_expr.shape[0]),
            "uses_evaluation_target": False,
            "leakage_mutation_check": leakage,
            "training_loss_first": result.metadata["training_loss_first"],
            "training_loss_last": result.metadata["training_loss_last"],
            "mean_modulation_norm": result.metadata["mean_modulation_norm"],
        }
        row.update(settings)
        row.update(metrics)
        for key, value in fixed_metrics.items():
            row[f"fixed_80_20_{key}"] = value
            row[f"diff_vs_fixed_{key}"] = metrics[key] - value
        rows.append(row)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "task_id": payload.task_id,
        "reference_fraction": float(reference_fraction),
        "split_seed": int(seed),
        "split_type": str(split_type),
        "uses_evaluation_target": False,
        "all_leakage_checks_pass": bool(all(r["leakage_mutation_check"] for r in rows)),
        "horizons": [float(h) for h in horizons],
        "rows": rows,
        "fixed_80_20_metrics": fixed_metrics,
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(output_dir / "prediction.npz", **predictions)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_ROOT / "results" / "local_residual_smoke",
    )
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--reference-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--split-type", default="random_stratified")
    args = parser.parse_args()
    path = run_local_residual_smoke(
        output_dir=args.output_dir,
        task_index=args.task_index,
        reference_fraction=args.reference_fraction,
        seed=args.seed,
        split_type=args.split_type,
    )
    print(path)


if __name__ == "__main__":
    main()
