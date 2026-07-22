#!/usr/bin/env python3
"""Compare structured source-shift flow with fixed 80/20."""

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

from nicheflow_v15.structured_flow_predict import predict_task_structured_flow
from nicheflow_v2.baselines import predict_existing_method
from nicheflow_v2.data import load_manifest, load_task_payload
from nicheflow_v2.splits import split_target_pool
from run_pilot import _metric_bundle


def run_structured_smoke(
    *,
    output_dir: Path,
    integration_horizon: float = 0.25,
    anchor_temperature: float = 0.25,
    modulation_cap: float = 0.5,
    identity_weight: float = 0.05,
    reference_fraction: float = 0.70,
    seed: int = 20260711,
) -> Path:
    payload = load_task_payload(load_manifest().iloc[0])
    split = split_target_pool(
        payload,
        reference_fraction=float(reference_fraction),
        seed=int(seed),
        split_type="random_stratified",
    )
    settings = {
        "latent_dim": 8,
        "hidden_dim": 32,
        "epochs": 60,
        "integration_steps": 16,
        "integration_horizon": float(integration_horizon),
        "anchor_temperature": float(anchor_temperature),
        "modulation_cap": float(modulation_cap),
        "identity_weight": float(identity_weight),
        "seed": int(seed),
    }
    result = predict_task_structured_flow(payload, split, **settings)
    altered = payload.with_target_evaluation_expr(
        split,
        payload.target_expr[split.evaluation_indices] + 1000.0,
    )
    changed = predict_task_structured_flow(altered, split, **settings)
    leakage = bool(np.allclose(result.prediction, changed.prediction))
    if not leakage:
        raise AssertionError("Evaluation-expression leakage detected")
    fixed = predict_existing_method(
        "legacy_v1_fixed_80_20_l3",
        payload,
        split,
    )
    metrics = _metric_bundle(result.prediction, payload, split)
    fixed_metrics = _metric_bundle(fixed.prediction, payload, split)
    summary: dict[str, object] = {
        "method": result.metadata["output_mode"],
        "task_id": payload.task_id,
        "reference_fraction": float(reference_fraction),
        "split_seed": int(seed),
        "split_type": "random_stratified",
        "leakage_mutation_check": leakage,
        "uses_evaluation_target": False,
        "training_loss_first": result.metadata["training_loss_first"],
        "training_loss_last": result.metadata["training_loss_last"],
    }
    summary.update(settings)
    for key, value in metrics.items():
        summary[f"structured_{key}"] = value
        summary[f"fixed_80_20_{key}"] = fixed_metrics[key]
        summary[f"diff_vs_fixed_{key}"] = value - fixed_metrics[key]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "prediction.npz",
        prediction=result.prediction,
        coupling=result.coupling,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_ROOT / "results" / "structured_flow_smoke",
    )
    parser.add_argument("--integration-horizon", type=float, default=0.25)
    parser.add_argument("--modulation-cap", type=float, default=0.5)
    parser.add_argument("--anchor-temperature", type=float, default=0.25)
    parser.add_argument("--identity-weight", type=float, default=0.05)
    args = parser.parse_args()
    path = run_structured_smoke(
        output_dir=args.output_dir,
        integration_horizon=args.integration_horizon,
        modulation_cap=args.modulation_cap,
        anchor_temperature=args.anchor_temperature,
        identity_weight=args.identity_weight,
    )
    print(path)


if __name__ == "__main__":
    main()
