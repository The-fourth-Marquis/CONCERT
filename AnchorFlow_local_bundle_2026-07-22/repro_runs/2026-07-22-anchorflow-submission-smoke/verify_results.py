#!/usr/bin/env python3
"""Independently verify the frozen AnchorFlow submission artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
RESULTS = ROOT / "results"
EXPECTED_TASKS = {
    13: "GSM5808055_border_surrogate_KP_to_Tgfbr2",
    16: "GSM5808055_cross_slide_surrogate_KP_to_Jak2_to_GSM5808054",
}
EXPECTED_SPLITS = ("random_stratified", "spatial_block")
EXPECTED_SETTINGS = {
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
METRIC_SUFFIXES = (
    "heldout_edistance",
    "delta_pcc",
    "delta_mae",
    "boundary_leakage",
    "source_individuality",
)


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def assert_compute(compute: dict[str, object], config: dict[str, object]) -> None:
    runtime = config["runtime"]
    assert isinstance(runtime, dict)
    expected_python = str(Path(str(runtime["python_executable"])).resolve())
    assert compute["physical_gpu"] == 2
    assert compute["cuda_visible_devices"] == "2"
    assert compute["process_device"] == "cuda:0"
    assert close(float(compute["memory_fraction"]), 0.1)
    assert compute["precision"] == "float32"
    assert compute["mixed_precision"] is False
    assert compute["required_cuda_arch"] == "sm_120"
    assert "sm_120" in compute["cuda_architectures"]
    assert compute["python_executable"] == expected_python
    assert compute["torch_version"] == "2.12.0+cu130"
    assert compute["torch_cuda_version"] == "13.0"
    assert "RTX PRO 6000 Blackwell" in str(compute["device_name"])


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert {int(row["index"]): row["task_id"] for row in config["tasks"]} == EXPECTED_TASKS
    assert tuple(config["split_types"]) == EXPECTED_SPLITS
    assert config["settings"] == EXPECTED_SETTINGS
    assert config["runtime"] == {
        "python_executable": "/root/code/nicheformer/.venv/bin/python",
        "required_cuda_arch": "sm_120",
    }
    assert config["physical_gpu"] == 2
    assert config["cuda_visible_devices"] == "2"
    assert close(config["cuda_memory_fraction"], 0.1)

    expected_views = {
        (task_id, split)
        for task_id in EXPECTED_TASKS.values()
        for split in EXPECTED_SPLITS
    }
    summaries: dict[tuple[str, str], dict[str, object]] = {}
    for task_id, split in expected_views:
        view_dir = RESULTS / f"{task_id}__{split}"
        summary = json.loads((view_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["task_id"] == task_id
        assert summary["split_type"] == split
        for key, expected in EXPECTED_SETTINGS.items():
            if key == "precision":
                continue
            assert summary[key] == expected
        assert summary["leakage_mutation_check"] is True
        assert summary["uses_evaluation_target"] is False
        assert summary["prediction_arrays_finite"] is True
        assert_compute(summary["compute"], config)
        with np.load(view_dir / "prediction.npz") as arrays:
            assert arrays.files
            assert all(np.isfinite(arrays[key]).all() for key in arrays.files)
        for prefix in ("fixed_80_20", "initial", "final"):
            for suffix in METRIC_SUFFIXES:
                assert np.isfinite(float(summary[f"{prefix}_{suffix}"]))
        summaries[(task_id, split)] = summary
    assert set(summaries) == expected_views

    with (RESULTS / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        metric_rows = list(csv.DictReader(handle))
    assert len(metric_rows) == 12
    methods = {
        "fixed_80_20": "fixed_80_20",
        "initial_anchor": "initial",
        "anchor_flow": "final",
    }
    seen_rows: set[tuple[str, str, str]] = set()
    for row in metric_rows:
        key = (row["task_id"], row["split_type"], row["method"])
        assert key not in seen_rows
        seen_rows.add(key)
        summary = summaries[(row["task_id"], row["split_type"])]
        prefix = methods[row["method"]]
        for suffix in METRIC_SUFFIXES:
            assert close(row[suffix], summary[f"{prefix}_{suffix}"])
    assert len(seen_rows) == 12

    ordered = [summaries[(task_id, split)] for task_id in EXPECTED_TASKS.values() for split in EXPECTED_SPLITS]
    flow = np.asarray([float(row["final_heldout_edistance"]) for row in ordered])
    initial = np.asarray([float(row["initial_heldout_edistance"]) for row in ordered])
    fixed = np.asarray([float(row["fixed_80_20_heldout_edistance"]) for row in ordered])
    flow_minus_initial = flow - initial
    flow_minus_fixed = flow - fixed
    joint = (flow_minus_initial < 0.0) & (flow_minus_fixed < 0.0)
    by_split: dict[str, dict[str, float]] = {}
    split_gate = True
    for split in EXPECTED_SPLITS:
        indices = [index for index, row in enumerate(ordered) if row["split_type"] == split]
        diff_initial = float(flow_minus_initial[indices].mean())
        diff_fixed = float(flow_minus_fixed[indices].mean())
        by_split[split] = {
            "mean_flow_minus_initial": diff_initial,
            "mean_flow_minus_fixed": diff_fixed,
        }
        split_gate = split_gate and diff_initial < 0.0 and diff_fixed < 0.0

    recomputed_gates = {
        "mean_below_initial": float(flow_minus_initial.mean()) < 0.0,
        "mean_below_fixed": float(flow_minus_fixed.mean()) < 0.0,
        "both_split_strata_improve": bool(split_gate),
        "at_least_three_joint_wins": int(joint.sum()) >= 3,
        "all_mutation_guards": all(bool(row["leakage_mutation_check"]) for row in ordered),
        "all_metrics_finite": bool(np.isfinite(np.concatenate([flow, initial, fixed])).all()),
    }
    expected_decision = (
        "confirmation_positive"
        if all(recomputed_gates.values())
        else "insufficient_confirmation_for_performance_claim"
    )

    aggregate = json.loads((RESULTS / "aggregate_summary.json").read_text(encoding="utf-8"))
    assert aggregate["decision"] == expected_decision
    assert aggregate["gates"] == recomputed_gates
    assert aggregate["flow_wins_both_count"] == int(joint.sum())
    assert close(aggregate["mean_flow_edistance"], flow.mean())
    assert close(aggregate["mean_initial_edistance"], initial.mean())
    assert close(aggregate["mean_fixed_80_20_edistance"], fixed.mean())
    assert close(aggregate["mean_flow_minus_initial"], flow_minus_initial.mean())
    assert close(aggregate["mean_flow_minus_fixed"], flow_minus_fixed.mean())
    for split in EXPECTED_SPLITS:
        for key, value in by_split[split].items():
            assert close(aggregate["by_split"][split][key], value)
    assert_compute(aggregate["compute"], config)

    print("VERIFIED 4 views, 12 metric rows, GPU/runtime metadata, mutation guards, finite arrays")
    print(f"decision={expected_decision}")
    print(f"joint_wins={int(joint.sum())}/4")
    print(f"mean_flow_minus_initial={float(flow_minus_initial.mean()):.12f}")
    print(f"mean_flow_minus_fixed={float(flow_minus_fixed.mean()):.12f}")


if __name__ == "__main__":
    main()
