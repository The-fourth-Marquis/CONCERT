#!/usr/bin/env python3
"""Compare anchor initialization, residual flow endpoint, and fixed 80/20."""

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

from nicheflow_v15.anchor_flow_predict import (
    predict_task_anchor_residual_flow,
)
from nicheflow_v2.baselines import predict_existing_method
from nicheflow_v2.data import load_manifest, load_task_payload
from nicheflow_v2.splits import split_target_pool
from run_pilot import _metric_bundle


def run_anchor_flow_smoke(
    *,
    output_dir: Path,
    task_index: int = 0,
    reference_fraction: float = 0.70,
    seed: int = 20260711,
    split_type: str = "random_stratified",
    latent_dim: int = 8,
    hidden_dim: int = 32,
    epochs: int = 60,
    integration_steps: int = 16,
    integration_horizon: float = 1.0,
    anchor_temperature: float = 0.25,
    source_skip_weight: float = 0.20,
    distribution_weight: float = 0.25,
    identity_weight: float = 0.05,
    device: str = "cpu",
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
    settings = {
        "latent_dim": int(latent_dim),
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "integration_steps": int(integration_steps),
        "integration_horizon": float(integration_horizon),
        "anchor_temperature": float(anchor_temperature),
        "source_skip_weight": float(source_skip_weight),
        "distribution_weight": float(distribution_weight),
        "identity_weight": float(identity_weight),
        "device": str(device),
        "seed": int(seed),
    }
    result = predict_task_anchor_residual_flow(payload, split, **settings)
    altered = payload.with_target_evaluation_expr(
        split,
        payload.target_expr[split.evaluation_indices] + 1000.0,
    )
    changed = predict_task_anchor_residual_flow(altered, split, **settings)
    leakage = bool(
        np.allclose(result.prediction, changed.prediction)
        and np.allclose(
            result.initial_prediction,
            changed.initial_prediction,
        )
    )
    if not leakage:
        raise AssertionError("Evaluation-expression leakage detected")
    fixed = predict_existing_method(
        "legacy_v1_fixed_80_20_l3",
        payload,
        split,
    )
    final_metrics = _metric_bundle(result.prediction, payload, split)
    initial_metrics = _metric_bundle(
        result.initial_prediction,
        payload,
        split,
    )
    fixed_metrics = _metric_bundle(fixed.prediction, payload, split)
    summary: dict[str, object] = {
        "task_id": payload.task_id,
        "method": result.metadata["output_mode"],
        "reference_fraction": float(reference_fraction),
        "split_seed": int(seed),
        "split_type": str(split_type),
        "n_reference": int(split.reference_indices.size),
        "n_evaluation": int(split.evaluation_indices.size),
        "uses_evaluation_target": False,
        "leakage_mutation_check": leakage,
        "training_loss_first": result.metadata["training_loss_first"],
        "training_loss_last": result.metadata["training_loss_last"],
        "training_loss_min": result.metadata["training_loss_min"],
        "mean_correction_norm": result.metadata["mean_correction_norm"],
    }
    summary.update(settings)
    for key in final_metrics:
        summary[f"final_{key}"] = final_metrics[key]
        summary[f"initial_{key}"] = initial_metrics[key]
        summary[f"fixed_80_20_{key}"] = fixed_metrics[key]
        summary[f"final_diff_vs_fixed_{key}"] = (
            final_metrics[key] - fixed_metrics[key]
        )
        summary[f"final_diff_vs_initial_{key}"] = (
            final_metrics[key] - initial_metrics[key]
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "prediction.npz",
        initial_prediction=result.initial_prediction,
        final_prediction=result.prediction,
        coupling=result.coupling,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_ROOT / "results" / "anchor_flow_smoke",
    )
    parser.add_argument("--anchor-temperature", type=float, default=0.25)
    parser.add_argument("--source-skip-weight", type=float, default=0.20)
    parser.add_argument("--identity-weight", type=float, default=0.05)
    parser.add_argument("--integration-horizon", type=float, default=1.0)
    parser.add_argument("--distribution-weight", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    path = run_anchor_flow_smoke(
        output_dir=args.output_dir,
        anchor_temperature=args.anchor_temperature,
        source_skip_weight=args.source_skip_weight,
        identity_weight=args.identity_weight,
        integration_horizon=args.integration_horizon,
        distribution_weight=args.distribution_weight,
        device=args.device,
    )
    print(path)


if __name__ == "__main__":
    main()
