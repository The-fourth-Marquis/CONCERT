#!/usr/bin/env python3
"""Run one leakage-controlled NicheFlow-v1.5 comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


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
from nicheflow_v2.metrics import distribution_metrics
from nicheflow_v2.splits import split_target_pool


def run_smoke(
    *,
    output_dir: Path,
    task_index: int = 0,
    reference_fraction: float = 0.70,
    seed: int = 20260711,
    split_type: str = "random_stratified",
    latent_dim: int = 8,
    hidden_dim: int = 16,
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
        "seed": int(seed),
    }
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
        raise AssertionError("Evaluation-expression leakage detected")

    fixed = predict_existing_method(
        "legacy_v1_fixed_80_20_l3",
        payload,
        split,
    )
    evaluation = payload.target_expr[split.evaluation_indices]
    nonlinear_metrics = distribution_metrics(
        nonlinear.prediction,
        evaluation,
        payload.target_expr,
    )
    fixed_metrics = distribution_metrics(
        fixed.prediction,
        evaluation,
        payload.target_expr,
    )
    heldout = float(nonlinear_metrics["heldout_edistance"])
    fixed_heldout = float(fixed_metrics["heldout_edistance"])
    summary = {
        "task_id": payload.task_id,
        "method": nonlinear.metadata["output_mode"],
        "access_level": nonlinear.metadata["access_level"],
        "reference_fraction": float(reference_fraction),
        "split_seed": int(seed),
        "split_type": str(split_type),
        "n_reference": int(split.reference_indices.size),
        "n_evaluation": int(split.evaluation_indices.size),
        "n_query": int(payload.query_expr.shape[0]),
        "heldout_edistance": heldout,
        "fixed_80_20_heldout_edistance": fixed_heldout,
        "delta_vs_fixed_80_20": heldout - fixed_heldout,
        "uses_evaluation_target": False,
        "leakage_mutation_check": leakage_check,
        "calibration": False,
        "latent_dim": int(nonlinear.metadata["latent_dim"]),
        "hidden_dim": int(nonlinear.metadata["hidden_dim"]),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "prediction.npz",
        prediction=nonlinear.prediction,
        coupling=nonlinear.coupling,
        reference_indices=split.reference_indices,
        evaluation_indices=split.evaluation_indices,
    )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_ROOT / "results" / "smoke",
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
    parser.add_argument("--hidden-dim", type=int, default=16)
    args = parser.parse_args()
    output = run_smoke(
        output_dir=args.output_dir,
        task_index=args.task_index,
        reference_fraction=args.reference_fraction,
        seed=args.seed,
        split_type=args.split_type,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    )
    print(output)


if __name__ == "__main__":
    main()
