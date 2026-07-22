#!/usr/bin/env python3
"""Smoke-test shared global shift plus bounded context modulation."""

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

from nicheflow_v15.factorized_flow_predict import predict_task_factorized_flow
from nicheflow_v2.baselines import predict_existing_method
from nicheflow_v2.data import load_manifest, load_task_payload
from nicheflow_v2.splits import split_target_pool
from run_pilot import _metric_bundle


CAPS = (0.0, 0.05, 0.1, 0.25, 0.5)
HORIZONS = (0.25, 0.5, 1.0)


def _pairwise_geometry_error(prediction: np.ndarray, source: np.ndarray) -> float:
    predicted_dist = np.linalg.norm(
        prediction[:, None, :] - prediction[None, :, :],
        axis=2,
    )
    source_dist = np.linalg.norm(
        source[:, None, :] - source[None, :, :],
        axis=2,
    )
    scale = max(float(source_dist.mean()), 1e-8)
    return float(np.mean(np.abs(predicted_dist - source_dist)) / scale)


def run_factorized_smoke(
    *,
    output_dir: Path,
    task_index: int = 0,
    reference_fraction: float = 0.70,
    seed: int = 20260711,
    split_type: str = "random_stratified",
    caps: tuple[float, ...] = CAPS,
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
    settings = {
        "latent_dim": 16,
        "hidden_dim": 64,
        "epochs": 80,
        "integration_steps": 16,
        "distribution_weight": 0.25,
        "seed": int(seed),
    }
    rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {"fixed_80_20": fixed.prediction}
    for horizon in horizons:
        for cap in caps:
            result = predict_task_factorized_flow(
                payload,
                split,
                integration_horizon=float(horizon),
                modulation_cap=float(cap),
                **settings,
            )
            changed = predict_task_factorized_flow(
                altered,
                split,
                integration_horizon=float(horizon),
                modulation_cap=float(cap),
                **settings,
            )
            leakage = bool(np.allclose(result.prediction, changed.prediction))
            if not leakage:
                raise AssertionError(
                    f"Evaluation-expression leakage at horizon={horizon}, cap={cap}"
                )
            method = f"factorized_h{float(horizon):g}_cap{float(cap):g}"
            predictions[method] = result.prediction
            metrics = _metric_bundle(result.prediction, payload, split)
            row: dict[str, object] = {
                "task_id": payload.task_id,
                "task_type": str(payload.task.get("task_type", "")),
                "benchmark_kind": str(payload.task.get("benchmark_kind", "")),
                "method": method,
                "integration_horizon": float(horizon),
                "modulation_cap": float(cap),
                "reference_fraction": float(reference_fraction),
                "split_seed": int(seed),
                "split_type": str(split_type),
                "n_reference": int(split.reference_indices.size),
                "n_evaluation": int(split.evaluation_indices.size),
                "n_query": int(payload.query_expr.shape[0]),
                "uses_evaluation_target": False,
                "leakage_mutation_check": leakage,
                "global_shift_norm": result.metadata["global_shift_norm"],
                "max_modulation_norm": result.metadata["max_modulation_norm"],
                "training_loss_first": result.metadata["training_loss_first"],
                "training_loss_last": result.metadata["training_loss_last"],
                "pairwise_geometry_error": _pairwise_geometry_error(
                    result.prediction,
                    payload.query_expr,
                ),
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
        "caps": [float(value) for value in caps],
        "horizons": [float(value) for value in horizons],
        "uses_evaluation_target": False,
        "all_leakage_checks_pass": bool(
            all(row["leakage_mutation_check"] for row in rows)
        ),
        "fixed_80_20_metrics": fixed_metrics,
        "rows": rows,
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
        default=RUN_ROOT / "results" / "factorized_smoke",
    )
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--reference-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--split-type", default="random_stratified")
    args = parser.parse_args()
    path = run_factorized_smoke(
        output_dir=args.output_dir,
        task_index=args.task_index,
        reference_fraction=args.reference_fraction,
        seed=args.seed,
        split_type=args.split_type,
    )
    print(path)


if __name__ == "__main__":
    main()
