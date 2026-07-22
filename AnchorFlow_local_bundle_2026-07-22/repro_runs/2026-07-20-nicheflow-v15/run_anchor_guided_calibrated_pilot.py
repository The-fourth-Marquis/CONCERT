#!/usr/bin/env python3
"""Run reference-only anchor-guided calibration on representative tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
sys.path.insert(0, str(RUN_ROOT))
sys.path.insert(0, str(RUN_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "repro_runs" / "2026-07-11-nicheflow-v2" / "src"))

from nicheflow_v15.anchor_guided_factorized_predict import predict_task_anchor_guided_factorized
from nicheflow_v2.baselines import predict_existing_method
from nicheflow_v2.data import load_manifest, load_task_payload
from nicheflow_v2.splits import split_target_pool
from run_anchor_guided_calibrated_smoke import (
    STRENGTHS,
    _metrics_against_target,
    _pairwise_geometry_error,
    _select_strength,
)
from run_pilot import _metric_bundle, _representative_task_ids


def _run_unit(
    payload: object,
    *,
    reference_fraction: float,
    seed: int,
    split_type: str,
    strengths: tuple[float, ...],
) -> dict[str, object]:
    split = split_target_pool(
        payload,
        reference_fraction=float(reference_fraction),
        seed=int(seed),
        split_type=str(split_type),
    )
    if split.status != "ok":
        return {"task_id": payload.task_id, "split_type": split_type, "status": "skipped", "reason": split.reason}
    reference_payload = payload.with_target_subset(split.reference_indices)
    pseudo_split = split_target_pool(
        reference_payload,
        reference_fraction=0.70,
        seed=int(seed) + 1000,
        split_type="random_stratified",
    )
    if pseudo_split.status != "ok":
        return {"task_id": payload.task_id, "split_type": split_type, "status": "skipped", "reason": f"pseudo: {pseudo_split.reason}"}
    fixed_pseudo = _metrics_against_target(
        predict_existing_method("legacy_v1_fixed_80_20_l3", reference_payload, pseudo_split).prediction,
        reference_payload,
        pseudo_split,
    )
    calibration_rows: list[dict[str, float]] = []
    for strength in strengths:
        prediction = predict_task_anchor_guided_factorized(
            reference_payload,
            pseudo_split,
            anchor_temperature=0.25,
            modulation_strength=float(strength),
        ).prediction
        calibration_rows.append({"modulation_strength": float(strength), **_metrics_against_target(prediction, reference_payload, pseudo_split)})
    selected_strength = _select_strength(calibration_rows, fixed_pseudo)
    result = predict_task_anchor_guided_factorized(
        payload,
        split,
        anchor_temperature=0.25,
        modulation_strength=selected_strength,
    )
    fixed = predict_existing_method("legacy_v1_fixed_80_20_l3", payload, split)
    metrics = _metric_bundle(result.prediction, payload, split)
    fixed_metrics = _metric_bundle(fixed.prediction, payload, split)
    row: dict[str, object] = {
        "task_id": payload.task_id,
        "task_type": str(payload.task.get("task_type", "")),
        "benchmark_kind": str(payload.task.get("benchmark_kind", "")),
        "split_type": split_type,
        "split_seed": int(seed),
        "reference_fraction": float(reference_fraction),
        "n_reference": int(split.reference_indices.size),
        "n_evaluation": int(split.evaluation_indices.size),
        "selected_strength": selected_strength,
        "status": "ok",
        "pairwise_geometry_error": _pairwise_geometry_error(result.prediction, payload.query_expr),
        **metrics,
        "pseudo_fixed_heldout_edistance": fixed_pseudo["heldout_edistance"],
        "pseudo_fixed_delta_pcc": fixed_pseudo["delta_pcc"],
        "pseudo_fixed_boundary_leakage": fixed_pseudo["boundary_leakage"],
        "pseudo_fixed_source_individuality": fixed_pseudo["source_individuality"],
    }
    for key, value in fixed_metrics.items():
        row[f"fixed_80_20_{key}"] = value
        row[f"diff_vs_fixed_{key}"] = metrics[key] - value
    row["pseudo_gated_strength_count"] = sum(
        candidate["heldout_edistance"] <= fixed_pseudo["heldout_edistance"]
        and candidate["delta_pcc"] - fixed_pseudo["delta_pcc"] >= -0.02
        and (candidate["boundary_leakage"] - fixed_pseudo["boundary_leakage"])
        / max(abs(fixed_pseudo["boundary_leakage"]), 1e-8) <= 0.05
        and candidate["source_individuality"] - fixed_pseudo["source_individuality"] >= -0.02
        for candidate in calibration_rows
    )
    return row


def run_pilot(
    *,
    output_dir: Path,
    reference_fraction: float = 0.70,
    seed: int = 20260711,
    split_types: tuple[str, ...] = ("random_stratified", "spatial_block"),
    strengths: tuple[float, ...] = STRENGTHS,
) -> dict[str, Path]:
    manifest = load_manifest()
    task_ids = _representative_task_ids(manifest)
    indexed = manifest.set_index("task_id", drop=False)
    rows: list[dict[str, object]] = []
    for task_id in task_ids:
        payload = load_task_payload(indexed.loc[task_id])
        for split_type in split_types:
            print(f"{task_id} {split_type}", flush=True)
            rows.append(
                _run_unit(
                    payload,
                    reference_fraction=reference_fraction,
                    seed=seed,
                    split_type=split_type,
                    strengths=strengths,
                )
            )
    metrics = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    ok = metrics.loc[metrics["status"] == "ok"].copy()
    boundary_relative = (
        ok["diff_vs_fixed_boundary_leakage"]
        / ok["fixed_80_20_boundary_leakage"].abs().clip(lower=1e-8)
    )
    summary = {
        "n_tasks": int(metrics["task_id"].nunique()),
        "n_units": int(metrics.shape[0]),
        "n_ok": int(ok.shape[0]),
        "all_units_ok": bool((metrics["status"] == "ok").all()),
        "mean_delta_vs_fixed": float(ok["diff_vs_fixed_heldout_edistance"].mean()),
        "mean_delta_pcc_diff_vs_fixed": float(ok["diff_vs_fixed_delta_pcc"].mean()),
        "mean_boundary_relative_vs_fixed": float(boundary_relative.mean()),
        "mean_individuality_diff_vs_fixed": float(ok["diff_vs_fixed_source_individuality"].mean()),
        "all_distribution_wins": bool((ok["diff_vs_fixed_heldout_edistance"] < 0.0).all()),
        "all_delta_gates": bool((ok["diff_vs_fixed_delta_pcc"] >= -0.02).all()),
        "all_boundary_gates": bool((boundary_relative <= 0.05).all()),
        "all_individuality_gates": bool((ok["diff_vs_fixed_source_individuality"] >= -0.02).all()),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"metrics": metrics_path, "summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_ROOT / "results" / "anchor_guided_calibrated_pilot",
    )
    parser.add_argument("--reference-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()
    outputs = run_pilot(
        output_dir=args.output_dir,
        reference_fraction=args.reference_fraction,
        seed=args.seed,
    )
    for name, path in outputs.items():
        print(name, path)


if __name__ == "__main__":
    main()
