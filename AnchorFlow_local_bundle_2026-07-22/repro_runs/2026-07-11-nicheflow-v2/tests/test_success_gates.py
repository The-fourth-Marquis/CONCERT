#!/usr/bin/env python3
"""Evidence regression tests for the approved NicheFlow-v2 success gates."""

from pathlib import Path
import json
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nicheflow_v2.config import ExperimentConfig
from nicheflow_v2.report import evaluate_success_gates


def test_full_evidence_produces_all_seven_gates_and_no_advance() -> None:
    baseline = pd.read_csv(ROOT / "results" / "existing_method_reassessment" / "metrics.csv")
    v2 = pd.read_csv(ROOT / "results" / "residual_gating" / "metrics.csv")
    gates = evaluate_success_gates(
        v2,
        baseline,
        candidate_method="fixed_80_20",
        config=ExperimentConfig.default(),
    )
    assert gates["gate_id"].tolist() == [f"G{index}" for index in range(1, 8)]
    assert not bool(gates.set_index("gate_id").loc["G1", "passed"])
    assert not bool(gates.set_index("gate_id").loc["G5", "passed"])
    assert bool(gates.set_index("gate_id").loc["G6", "passed"])
    assert not gates["passed"].all()
    assert set(gates["decision"]) == {"do_not_advance_to_graph_flow"}


def test_uncalibrated_candidate_uses_uncalibrated_access_matched_baseline() -> None:
    rows = []
    candidates = []
    for index in range(37):
        common = {
            "task_id": f"t{index}",
            "reference_fraction": 0.50,
            "split_seed": 20260711,
            "split_type": "random_stratified",
            "delta_pcc": 1.0,
            "delta_spearman": 1.0,
            "delta_mae": 1.0,
            "delta_rmse": 1.0,
            "de_top20_direction_accuracy": 1.0,
            "de_top20_opposite_direction_fraction": 0.0,
            "boundary_leakage_score": 1.0,
            "source_individuality_score": 1.0,
        }
        rows.append({
            **common,
            "model_family": "a2_reference_only_baseline",
            "method": "uncalibrated_l1",
            "calibration": False,
            "heldout_edistance": 2.0,
        })
        rows.append({
            **common,
            "model_family": "a2_reference_only_baseline",
            "method": "calibrated_l2",
            "calibration": True,
            "heldout_edistance": 0.0,
        })
        rows.append({
            **common,
            "model_family": "nicheflow_v1",
            "method": "legacy_v1_uncalibrated",
            "calibration": False,
            "heldout_edistance": 1.5,
        })
        candidates.append({
            **common,
            "model_family": "nicheflow_v2_lightweight",
            "method": "candidate",
            "calibration": False,
            "heldout_edistance": 1.0,
        })
    gates = evaluate_success_gates(
        pd.DataFrame(candidates),
        pd.DataFrame(rows),
        candidate_method="candidate",
        config=ExperimentConfig.default(),
    )
    observed = json.loads(gates.set_index("gate_id").loc["G1", "observed"])
    assert observed["mean_delta"] == -1.0


if __name__ == "__main__":
    test_full_evidence_produces_all_seven_gates_and_no_advance()
    test_uncalibrated_candidate_uses_uncalibrated_access_matched_baseline()
    print("ok")
