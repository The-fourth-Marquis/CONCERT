#!/usr/bin/env python3
"""Decision gate must reject distribution gains that destroy mechanism metrics."""

from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from run_pilot import _decision


def test_decision_rejects_individuality_collapse() -> None:
    summary = pd.DataFrame(
        [
            {
                "config_name": "bad_mechanism",
                "mean_delta_vs_fixed": -2.0,
                "spatial_block_mean_delta": -1.0,
                "win_rate": 0.75,
                "all_leakage_checks_pass": True,
                "mean_delta_pcc_diff_vs_fixed": 0.0,
                "mean_boundary_relative_vs_fixed": 0.0,
                "mean_individuality_diff_vs_fixed": -0.5,
            }
        ]
    )
    metrics = pd.DataFrame(
        [{"task_id": "task", "split_type": "spatial_block"}]
    )

    decision = _decision(summary, metrics)

    assert decision["decision"] == "redesign_mechanism_before_multiseed"
    assert decision["mechanism_gate_pass"] is False


if __name__ == "__main__":
    test_decision_rejects_individuality_collapse()
    print("ok")
