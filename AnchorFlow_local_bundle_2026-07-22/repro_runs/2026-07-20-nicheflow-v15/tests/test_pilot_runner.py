#!/usr/bin/env python3
"""Pilot-run artifact and decision contracts."""

from pathlib import Path
import json
import sys
import tempfile

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from run_pilot import PilotConfig, run_pilot


def test_single_unit_pilot_writes_metrics_summary_and_decision() -> None:
    config = PilotConfig(
        name="test_context",
        latent_dim=4,
        hidden_dim=8,
        context_weight=1.0,
        sinkhorn_epsilon=0.1,
        ridge=0.01,
    )
    with tempfile.TemporaryDirectory() as tmp:
        outputs = run_pilot(
            output_dir=Path(tmp),
            task_ids=("GSM5808054_patchclose_tumor_Jak2",),
            configs=(config,),
            reference_fraction=0.70,
            seed=20260711,
            split_types=("random_stratified",),
        )
        metrics = pd.read_csv(outputs["metrics"])
        summary = pd.read_csv(outputs["summary"])
        decision = json.loads(outputs["decision"].read_text())

    assert metrics.shape[0] == 1
    assert summary.shape[0] == 1
    assert metrics.loc[0, "config_name"] == "test_context"
    assert bool(metrics.loc[0, "leakage_mutation_check"])
    assert metrics.loc[0, "uses_evaluation_target"] == False
    assert "delta_vs_fixed_80_20" in metrics.columns
    assert "delta_pcc_diff_vs_fixed" in metrics.columns
    assert decision["selected_config"] == "test_context"
    assert decision["n_units"] == 1


if __name__ == "__main__":
    test_single_unit_pilot_writes_metrics_summary_and_decision()
    print("ok")
