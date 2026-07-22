#!/usr/bin/env python3
"""Run the frozen four-view AnchorFlow submission confirmation once."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
V15_ROOT = REPO_ROOT / "repro_runs" / "2026-07-20-nicheflow-v15"
V2_SRC = REPO_ROOT / "repro_runs" / "2026-07-11-nicheflow-v2" / "src"
sys.path.insert(0, str(V15_ROOT))
sys.path.insert(0, str(V15_ROOT / "src"))
sys.path.insert(0, str(V2_SRC))

from nicheflow_v2.data import load_manifest
from run_anchor_flow_smoke import run_anchor_flow_smoke


CONFIG_PATH = RUN_ROOT / "config.json"
RESULTS_ROOT = RUN_ROOT / "results"
METRIC_SUFFIXES = (
    "heldout_edistance",
    "delta_pcc",
    "delta_mae",
    "boundary_leakage",
    "source_individuality",
)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric_metrics_are_finite(summary: dict[str, object]) -> bool:
    selected = [
        float(value)
        for key, value in summary.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (key.endswith(METRIC_SUFFIXES) or key.startswith("training_loss_"))
    ]
    return bool(selected) and bool(np.isfinite(np.asarray(selected)).all())


def aggregate_decision(summaries: list[dict[str, object]]) -> dict[str, Any]:
    if len(summaries) != 4:
        raise ValueError(f"Expected four frozen views, found {len(summaries)}")
    view_ids = {(str(row["task_id"]), str(row["split_type"])) for row in summaries}
    if len(view_ids) != 4:
        raise ValueError("Frozen task/split views are not unique")

    final = np.asarray([float(row["final_heldout_edistance"]) for row in summaries])
    initial = np.asarray([float(row["initial_heldout_edistance"]) for row in summaries])
    fixed = np.asarray(
        [float(row["fixed_80_20_heldout_edistance"]) for row in summaries]
    )
    diff_initial = final - initial
    diff_fixed = final - fixed
    all_metrics_finite = bool(
        np.isfinite(np.concatenate([final, initial, fixed])).all()
        and all(_numeric_metrics_are_finite(row) for row in summaries)
        and all(bool(row.get("prediction_arrays_finite", True)) for row in summaries)
    )
    all_mutation_guards = all(
        bool(row.get("leakage_mutation_check", False)) for row in summaries
    )
    joint_wins = (diff_initial < 0.0) & (diff_fixed < 0.0)

    by_split: dict[str, dict[str, float]] = {}
    split_gate = True
    for split_type in ("random_stratified", "spatial_block"):
        indices = [
            index
            for index, row in enumerate(summaries)
            if str(row["split_type"]) == split_type
        ]
        if len(indices) != 2:
            raise ValueError(f"Expected two {split_type} views, found {len(indices)}")
        mean_initial = float(diff_initial[indices].mean())
        mean_fixed = float(diff_fixed[indices].mean())
        by_split[split_type] = {
            "mean_flow_minus_initial": mean_initial,
            "mean_flow_minus_fixed": mean_fixed,
        }
        split_gate = split_gate and mean_initial < 0.0 and mean_fixed < 0.0

    gates = {
        "mean_below_initial": float(diff_initial.mean()) < 0.0,
        "mean_below_fixed": float(diff_fixed.mean()) < 0.0,
        "both_split_strata_improve": bool(split_gate),
        "at_least_three_joint_wins": int(joint_wins.sum()) >= 3,
        "all_mutation_guards": bool(all_mutation_guards),
        "all_metrics_finite": bool(all_metrics_finite),
    }
    decision = (
        "confirmation_positive"
        if all(gates.values())
        else "insufficient_confirmation_for_performance_claim"
    )
    return {
        "decision": decision,
        "n_views": 4,
        "mean_flow_edistance": float(final.mean()),
        "mean_initial_edistance": float(initial.mean()),
        "mean_fixed_80_20_edistance": float(fixed.mean()),
        "mean_flow_minus_initial": float(diff_initial.mean()),
        "mean_flow_minus_fixed": float(diff_fixed.mean()),
        "flow_wins_both_count": int(joint_wins.sum()),
        "all_mutation_guards": bool(all_mutation_guards),
        "all_metrics_finite": bool(all_metrics_finite),
        "by_split": by_split,
        "gates": gates,
    }


def validate_frozen_protocol(config: dict[str, Any]) -> dict[str, Any]:
    expected_tasks = {
        13: "GSM5808055_border_surrogate_KP_to_Tgfbr2",
        16: "GSM5808055_cross_slide_surrogate_KP_to_Jak2_to_GSM5808054",
    }
    configured_tasks = {
        int(item["index"]): str(item["task_id"]) for item in config["tasks"]
    }
    if configured_tasks != expected_tasks:
        raise ValueError("Frozen task mapping changed")
    if config["split_types"] != ["random_stratified", "spatial_block"]:
        raise ValueError("Frozen split types changed")
    settings = config["settings"]
    expected_settings = {
        "reference_fraction": 0.7,
        "seed": 20260711,
        "latent_dim": 8,
        "hidden_dim": 32,
        "epochs": 60,
        "integration_steps": 16,
        "integration_horizon": 0.25,
        "anchor_temperature": 0.25,
        "source_skip_weight": 0.2,
        "distribution_weight": 0.25,
        "identity_weight": 0.05,
        "device": "cuda:0",
        "precision": "float32",
    }
    if settings != expected_settings:
        raise ValueError("Frozen model settings changed")
    manifest = load_manifest()
    for index, task_id in expected_tasks.items():
        observed = str(manifest.iloc[index]["task_id"])
        if observed != task_id:
            raise ValueError(f"Manifest index {index} resolved to {observed}")
    return settings


def validate_cuda_architecture(available: list[str], required: str) -> None:
    if required not in available:
        raise RuntimeError(
            f"PyTorch CUDA kernels do not include required architecture {required}; "
            f"available architectures: {available}"
        )


def configure_shared_gpu(config: dict[str, Any]) -> dict[str, Any]:
    expected_visible = str(config["cuda_visible_devices"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must equal {expected_visible}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one process-visible CUDA device")
    runtime = config["runtime"]
    expected_python = Path(str(runtime["python_executable"])).resolve()
    actual_python = Path(sys.executable).resolve()
    if actual_python != expected_python:
        raise RuntimeError(
            f"Frozen runtime requires {expected_python}, found {actual_python}"
        )
    available_architectures = torch.cuda.get_arch_list()
    required_architecture = str(runtime["required_cuda_arch"])
    validate_cuda_architecture(available_architectures, required_architecture)
    torch.cuda.set_per_process_memory_fraction(
        float(config["cuda_memory_fraction"]),
        device=0,
    )
    return {
        "physical_gpu": int(config["physical_gpu"]),
        "cuda_visible_devices": expected_visible,
        "process_device": "cuda:0",
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python_executable": str(actual_python),
        "cuda_architectures": available_architectures,
        "required_cuda_arch": required_architecture,
        "precision": "float32",
        "mixed_precision": False,
        "memory_fraction": float(config["cuda_memory_fraction"]),
    }


def _metric_rows(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    methods = {
        "fixed_80_20": "fixed_80_20",
        "initial_anchor": "initial",
        "anchor_flow": "final",
    }
    for summary in summaries:
        for method, prefix in methods.items():
            row: dict[str, object] = {
                "task_id": summary["task_id"],
                "split_type": summary["split_type"],
                "method": method,
            }
            for suffix in METRIC_SUFFIXES:
                row[suffix] = summary[f"{prefix}_{suffix}"]
            rows.append(row)
    return rows


def _write_metrics(rows: list[dict[str, object]], path: Path) -> None:
    fields = ["task_id", "split_type", "method", *METRIC_SUFFIXES]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = load_config()
    settings = validate_frozen_protocol(config)
    compute = configure_shared_gpu(config)
    if RESULTS_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen results: {RESULTS_ROOT}")
    RESULTS_ROOT.mkdir(parents=True)

    summaries: list[dict[str, object]] = []
    for task in config["tasks"]:
        for split_type in config["split_types"]:
            view_name = f"{task['task_id']}__{split_type}"
            output_dir = RESULTS_ROOT / view_name
            print(f"RUN {view_name}", flush=True)
            summary_path = run_anchor_flow_smoke(
                output_dir=output_dir,
                task_index=int(task["index"]),
                split_type=str(split_type),
                **{key: value for key, value in settings.items() if key != "precision"},
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            with np.load(output_dir / "prediction.npz") as arrays:
                arrays_finite = all(np.isfinite(arrays[key]).all() for key in arrays.files)
            summary["prediction_arrays_finite"] = bool(arrays_finite)
            summary["compute"] = compute
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            summaries.append(summary)

    rows = _metric_rows(summaries)
    _write_metrics(rows, RESULTS_ROOT / "metrics.csv")
    aggregate = aggregate_decision(summaries)
    aggregate["claim_boundary"] = config["claim_boundary"]
    aggregate["compute"] = compute
    aggregate["task_ids"] = [str(item["task_id"]) for item in config["tasks"]]
    aggregate["split_types"] = list(config["split_types"])
    aggregate_path = RESULTS_ROOT / "aggregate_summary.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(RESULTS_ROOT / "metrics.csv")
    print(aggregate_path)


if __name__ == "__main__":
    main()
