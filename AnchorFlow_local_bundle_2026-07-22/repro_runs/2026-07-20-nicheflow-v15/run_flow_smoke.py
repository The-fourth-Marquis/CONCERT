#!/usr/bin/env python3
"""Run one held-out conditional residual flow comparison."""

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

from nicheflow_v15.flow_predict import predict_task_conditional_flow
from nicheflow_v2.baselines import predict_existing_method
from nicheflow_v2.data import load_manifest, load_task_payload
from nicheflow_v2.splits import split_target_pool
from run_pilot import _metric_bundle


def run_flow_smoke(
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
    distribution_weight: float = 0.25,
    identity_weight: float = 0.05,
) -> Path:
    manifest = load_manifest()
    payload = load_task_payload(manifest.iloc[int(task_index)])
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
        "distribution_weight": float(distribution_weight),
        "identity_weight": float(identity_weight),
        "seed": int(seed),
    }
    flow = predict_task_conditional_flow(payload, split, **settings)
    altered = payload.with_target_evaluation_expr(
        split,
        payload.target_expr[split.evaluation_indices] + 1000.0,
    )
    changed = predict_task_conditional_flow(altered, split, **settings)
    leakage = bool(np.allclose(flow.prediction, changed.prediction))
    if not leakage:
        raise AssertionError("Evaluation-expression leakage detected")
    fixed = predict_existing_method(
        "legacy_v1_fixed_80_20_l3",
        payload,
        split,
    )
    flow_metrics = _metric_bundle(flow.prediction, payload, split)
    fixed_metrics = _metric_bundle(fixed.prediction, payload, split)
    summary: dict[str, object] = {
        "task_id": payload.task_id,
        "method": flow.metadata["output_mode"],
        "reference_fraction": float(reference_fraction),
        "split_seed": int(seed),
        "split_type": str(split_type),
        "n_reference": int(split.reference_indices.size),
        "n_evaluation": int(split.evaluation_indices.size),
        "uses_evaluation_target": False,
        "leakage_mutation_check": leakage,
        "training_loss_first": flow.metadata["training_loss_first"],
        "training_loss_last": flow.metadata["training_loss_last"],
        "training_loss_min": flow.metadata["training_loss_min"],
    }
    summary.update(settings)
    for key, value in flow_metrics.items():
        summary[key] = value
        summary[f"fixed_80_20_{key}"] = fixed_metrics[key]
        if key in {"heldout_edistance", "niche_stratified_edistance", "delta_mae", "boundary_leakage"}:
            summary[f"{key}_diff_vs_fixed"] = value - fixed_metrics[key]
        else:
            summary[f"{key}_diff_vs_fixed"] = value - fixed_metrics[key]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "prediction.npz",
        prediction=flow.prediction,
        coupling=flow.coupling,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_ROOT / "results" / "flow_smoke",
    )
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--reference-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--split-type",
        choices=("random_stratified", "spatial_block"),
        default="random_stratified",
    )
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--integration-steps", type=int, default=16)
    parser.add_argument("--distribution-weight", type=float, default=0.25)
    parser.add_argument("--identity-weight", type=float, default=0.05)
    args = parser.parse_args()
    path = run_flow_smoke(
        output_dir=args.output_dir,
        task_index=args.task_index,
        reference_fraction=args.reference_fraction,
        seed=args.seed,
        split_type=args.split_type,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        integration_steps=args.integration_steps,
        distribution_weight=args.distribution_weight,
        identity_weight=args.identity_weight,
    )
    print(path)


if __name__ == "__main__":
    main()
